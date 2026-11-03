"""
policy.py — мозг дрона (актор) и централизованный критик для MAPPO.

Мозг видит три вещи:
    * находки глаз — набор переменной длины (внимание/пулинг, порядок не важен);
    * карту — окно вокруг себя и грубую карту всего поля (где был, где препятствия);
    * себя — позиция, скорость, время, размер команды.
Веса общие для всех дронов, поэтому рой работает при любом их числе.
Модули связи CIDA добавляются на следующих шагах между `encode` и `head`.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .env import DET_FEATURES, N_ACTIONS, SELF_FEATURES, EnvConfig, SearchEnv


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


class MapEncoder(nn.Module):
    """
    Карта дрона = окно вокруг него (стены рядом, где уже был) + грубая карта всего поля (куда лететь дальше).
    Так устроено зрение у навигационных роботов, и это несколько маленьких матричных умножений вместо
    свёрток по всей карте: свёртки с узкими каналами упирали обучение на A100 в пропускную способность памяти.
    """

    def __init__(self, grid: int, hidden: int, window: int = 7, coarse: int = 6):
        super().__init__()
        if grid % coarse:
            raise ValueError(f"grid={grid} должен делиться на coarse={coarse}")
        self.grid, self.coarse = grid, coarse
        r = window // 2
        offsets = torch.arange(-r, r + 1)
        self.register_buffer("dx", offsets.repeat_interleave(window), persistent=False)
        self.register_buffer("dy", offsets.repeat(window), persistent=False)
        d_in = 2 * window * window + 2 * coarse * coarse
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

    def _coarse(self, m: torch.Tensor) -> torch.Tensor:
        g, c = self.grid, self.coarse
        return m.unflatten(-1, (c, g // c, c, g // c)).float().mean((-3, -1)).flatten(-2)

    def forward(self, cov: torch.Tensor, obstacles: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """cov (..., N, G²) bool, obstacles (..., G²) bool, cell (..., N) → (..., N, hidden)."""
        g = self.grid
        obst = obstacles.unsqueeze(-2).expand_as(cov)
        lx = (cell // g).unsqueeze(-1) + self.dx
        ly = (cell % g).unsqueeze(-1) + self.dy
        inside = (lx >= 0) & (lx < g) & (ly >= 0) & (ly < g)
        idx = lx.clamp(0, g - 1) * g + ly.clamp(0, g - 1)
        near_cov = cov.gather(-1, idx) & inside
        near_obst = obst.gather(-1, idx) | ~inside          # край поля — тоже стена
        x = torch.cat([near_cov.float(), near_obst.float(), self._coarse(cov), self._coarse(obst)], -1)
        return self.net(x)


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
