"""
policy.py — мозг дрона (актор) и централизованный критик для MAPPO.

Мозг видит три вещи:
    * находки глаз — набор переменной длины (внимание/пулинг, порядок не важен);
    * карту — где уже был, где препятствия, где я сейчас (маленькая свёрточная сеть);
    * себя — позиция, скорость, время, размер команды.
Веса общие для всех дронов, поэтому рой работает при любом их числе.
Модули связи CIDA добавляются на следующих шагах между `encode` и `head`.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import DET_FEATURES, MAP_CHANNELS, N_ACTIONS, SELF_FEATURES, EnvConfig, SearchEnv


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


class MapEncoder(nn.Module):
    """Карта G×G с каналами [исследовано, препятствия, я] → вектор признаков."""

    def __init__(self, grid: int, hidden: int):
        super().__init__()
        self.grid = grid
        self.conv = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        side = (grid + 3) // 4
        self.out = nn.Sequential(nn.Linear(32 * side * side, hidden), nn.ReLU())

    def forward(self, cov: torch.Tensor, obstacles: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """cov (..., N, G²) bool, obstacles (..., G²) bool, cell (..., N) → (..., N, hidden)."""
        lead, g = cov.shape[:-1], self.grid
        me = F.one_hot(cell, g * g)
        obst = obstacles.unsqueeze(-2).expand_as(cov)
        x = torch.stack([cov, obst, me.bool()], -2).reshape(-1, MAP_CHANNELS, g, g).float()
        return self.out(self.conv(x).flatten(1)).view(*lead, -1)


class Actor(nn.Module):
    def __init__(self, env_cfg: EnvConfig, hidden: int = 256):
        super().__init__()
        self.det_enc = _mlp(DET_FEATURES, hidden, hidden)
        self.map_enc = MapEncoder(env_cfg.grid, hidden)
        self.self_enc = nn.Sequential(nn.Linear(SELF_FEATURES, hidden), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, N_ACTIONS)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """obs с формой (..., N, ...) → признаки дронов (..., N, hidden)."""
        det = obs["det"]
        valid = det[..., -1] > 0
        h = self.det_enc(det).masked_fill(~valid[..., None], -1e4).max(-2).values
        h = torch.where(valid.any(-1, keepdim=True), h, torch.zeros_like(h))  # порядок находок не важен
        m = self.map_enc(obs["cov"], obs["obstacles"], obs["cell"])
        return self.trunk(torch.cat([h, m, self.self_enc(obs["self"])], -1))

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(obs))


class Critic(nn.Module):
    """Видит весь мир (только при обучении) и оценивает общую награду команды (в нормированной шкале)."""

    def __init__(self, env_cfg: EnvConfig, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SearchEnv.state_dim(env_cfg), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
