"""
policy.py — мозг дрона (актор) и централизованный критик для MAPPO.

Шаг 1 — базовая линия без связи: каждый дрон решает только по своим глазам и своей памяти
об исследованных клетках. Веса общие для всех дронов, поэтому рой работает при любом их числе.
Модули связи CIDA добавляются на следующих шагах между `encode` и `head`.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .env import DET_FEATURES, N_ACTIONS, SELF_FEATURES, EnvConfig, SearchEnv


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


class Actor(nn.Module):
    def __init__(self, env_cfg: EnvConfig, hidden: int = 128):
        super().__init__()
        self.det_enc = _mlp(DET_FEATURES, hidden, hidden)
        self.cov_enc = nn.Sequential(nn.Linear(env_cfg.grid ** 2, hidden), nn.ReLU())
        self.self_enc = nn.Sequential(nn.Linear(SELF_FEATURES, hidden), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, N_ACTIONS)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """obs с формой (..., N, ...) → признаки дронов (..., N, hidden)."""
        det = obs["det"]
        valid = det[..., -1] > 0
        h = self.det_enc(det).masked_fill(~valid[..., None], -1e9).max(-2).values
        h = torch.where(valid.any(-1, keepdim=True), h, torch.zeros_like(h))  # порядок находок не важен
        return self.trunk(torch.cat([h, self.cov_enc(obs["cov"]), self.self_enc(obs["self"])], -1))

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(obs))


class Critic(nn.Module):
    """Видит весь мир (только при обучении) и оценивает общую награду команды."""

    def __init__(self, env_cfg: EnvConfig, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SearchEnv.state_dim(env_cfg), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
