"""
policy.py — мозг дрона (актор) и централизованный критик для MAPPO.

Мозг видит три вещи:
    * находки глаз — набор переменной длины (внимание/пулинг, порядок не важен);
    * карту — окно вокруг себя и грубую карту всего поля (где был, где препятствия);
    * себя — позиция, скорость, время, размер команды.
Веса общие для всех дронов, поэтому рой работает при любом их числе.

С comm="cida" между восприятием и решением стоит связь CIDA (см. CidaComm): дроны в радиусе связи
обмениваются сообщениями, и громче звучит тот, чья карта «где люди» сильнее расходится с картой слушателя.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

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


class CidaComm(nn.Module):
    """
    Связь CIDA между дронами одного мира.

    У каждого дрона есть убеждение — для каждой клетки грубой карты вероятность «здесь человек».
    Раунд связи: дрон слушает соседей в радиусе связи; вес соседа = обычное внимание
    + alpha · расхождение их убеждений (средняя разница вероятностей). Так новость от дрона,
    который знает то, чего не знает слушатель, не тонет в хоре согласных — это защита от эхо-камеры.
    Выход связи добавляется к признакам дрона через проекцию с нулевой инициализацией:
    в начале обучения рой ведёт себя как без связи и учится пользоваться ею постепенно.
    """

    def __init__(self, hidden: int, cells: int, d_comm: int = 64, rounds: int = 2, radius: float = 0.3):
        super().__init__()
        self.rounds, self.radius, self.d_comm = rounds, radius, d_comm
        self.belief = nn.Linear(hidden, cells)
        self.q = nn.Linear(hidden, d_comm)
        self.k = nn.Linear(hidden + cells, d_comm)
        self.v = nn.Linear(hidden + cells, d_comm)
        self.alpha = nn.Parameter(torch.tensor(2.0))
        self.o = nn.Linear(d_comm, hidden)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, active: torch.Tensor):
        """
        h (..., N, H), pos (..., N, 2) в координатах поля [0, 1], active (..., N).
        Возвращает новые признаки и словарь: логиты убеждений после каждого раунда, внимание последнего раунда.
        """
        n = h.shape[-2]
        near = torch.cdist(pos.float(), pos.float()) < self.radius
        eye = torch.eye(n, dtype=torch.bool, device=h.device)
        can_hear = near & active[..., None, :] & active[..., :, None] & ~eye        # (..., N, N)
        has_any = can_hear.any(-1, keepdim=True)
        beliefs, attn = [], None
        for _ in range(self.rounds):
            logit = self.belief(h)
            beliefs.append(logit)
            p = torch.sigmoid(logit.float())
            disagreement = (p.unsqueeze(-2) - p.unsqueeze(-3)).abs().mean(-1)        # (..., N, N)
            msg_in = torch.cat([h, p.to(h.dtype)], -1)
            q, k, v = self.q(h), self.k(msg_in), self.v(msg_in)
            scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.d_comm) + self.alpha * disagreement
            attn = torch.softmax(scores.masked_fill(~can_hear, -1e9), -1) * has_any
            h = h + self.o((attn.to(v.dtype) @ v))
        beliefs.append(self.belief(h))
        return h, {"beliefs": beliefs, "attn": attn}


class Actor(nn.Module):
    def __init__(self, env_cfg: EnvConfig, hidden: int = 256, comm: str = "none",
                 comm_radius: float = 0.3, comm_rounds: int = 2):
        super().__init__()
        if comm not in ("none", "cida"):
            raise ValueError(f"comm должен быть none или cida, а не {comm}")
        self.det_enc = _mlp(DET_FEATURES, hidden, hidden)
        self.map_enc = MapEncoder(env_cfg.grid, hidden)
        self.self_enc = nn.Sequential(nn.Linear(SELF_FEATURES, hidden), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.comm = (CidaComm(hidden, self.map_enc.coarse ** 2, rounds=comm_rounds, radius=comm_radius)
                     if comm == "cida" else None)
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

    def forward(self, obs: Dict[str, torch.Tensor], return_aux: bool = False):
        h = self.encode(obs)
        aux: Optional[Dict] = None
        if self.comm is not None:
            pos = (obs["self"][..., :2] + 1) / 2
            h, aux = self.comm(h, pos, obs["active"])
        logits = self.head(h)
        return (logits, aux) if return_aux else logits


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
