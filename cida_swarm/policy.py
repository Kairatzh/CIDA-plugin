"""
policy.py — мозг дрона (актор) и централизованный критик для MAPPO.

Мозг видит три вещи:
    * находки глаз — набор переменной длины (внимание/пулинг, порядок не важен);
    * карту — окно вокруг себя и грубую карту всего поля (где был, где препятствия);
    * себя — позиция, скорость, время, размер команды.
Веса общие для всех дронов, поэтому рой работает при любом их числе.

С comm="cida" между восприятием и решением стоит связь CIDA (см. CidaComm): дроны в радиусе связи
обмениваются сообщениями, и громче звучит тот, чья карта «где люди» сильнее расходится с картой слушателя.

С memory=True (спасательная миссия) у дрона есть память — GRU поверх восприятия; услышанное по радио
на прошлом шаге тоже входит в память. Шаг роя с памятью — policy_step(actor, obs, mem).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .env import DET_FEATURES, N_ACTIONS, SELF_FEATURES, EnvConfig
from .rescue import COARSE_CHANNELS as RESCUE_COARSE_CHANNELS
from .rescue import N_ACTIONS as RESCUE_N_ACTIONS
from .rescue import ROLE_DIM as RESCUE_ROLE_DIM
from .rescue import SELF_FEATURES as RESCUE_SELF_FEATURES
from .rescue import WINDOW_CHANNELS as RESCUE_WINDOW_CHANNELS
from .rescue import RescueConfig

COMM_TYPES = ("none", "attn", "cida")


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
    К услышанному добавляется направление на говорящих (взвешенное вниманием смещение до них):
    без него «держу тяжёлого, нужна помощь» не говорит, куда лететь.
    Услышанное вместе со своими признаками проходит через небольшую сеть, последний слой которой
    инициализирован нулями: в начале обучения рой ведёт себя как без связи и учится пользоваться ею постепенно.
    """

    def __init__(self, hidden: int, cells: int, d_comm: int = 64, rounds: int = 2, radius: float = 0.3,
                 disagreement_bias: bool = True):
        super().__init__()
        self.rounds, self.radius, self.d_comm = rounds, radius, d_comm
        self.belief = nn.Linear(hidden, cells)
        self.q = nn.Linear(hidden, d_comm)
        self.k = nn.Linear(hidden + cells, d_comm)
        self.v = nn.Linear(hidden + cells, d_comm)
        # без смещения разногласием это обычное внимание между агентами (как в TarMAC) — базовая линия
        if disagreement_bias:
            self.alpha = nn.Parameter(torch.tensor(2.0))
        else:
            self.register_buffer("alpha", torch.zeros(()))
        self.o = nn.Sequential(nn.Linear(hidden + d_comm + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn.init.zeros_(self.o[-1].weight)
        nn.init.zeros_(self.o[-1].bias)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, active: torch.Tensor):
        """
        h (..., N, H), pos (..., N, 2) в координатах поля [0, 1], active (..., N).
        Возвращает новые признаки и словарь: логиты убеждений после каждого раунда, внимание последнего раунда.
        """
        n = h.shape[-2]
        pos = pos.float()
        near = torch.cdist(pos, pos) < self.radius
        eye = torch.eye(n, dtype=torch.bool, device=h.device)
        can_hear = near & active[..., None, :] & active[..., :, None] & ~eye        # (..., N, N)
        has_any = can_hear.any(-1, keepdim=True)
        beliefs, attn = [], None
        for _ in range(self.rounds):
            # убеждение читает признаки, но не сдвигает их своей вспомогательной потерей:
            # иначе задача «где люди» перетягивает признаки, нужные для полёта
            logit = self.belief(h.detach())
            beliefs.append(logit)
            p = torch.sigmoid(logit.float())
            disagreement = (p.unsqueeze(-2) - p.unsqueeze(-3)).abs().mean(-1)        # (..., N, N)
            msg_in = torch.cat([h, p.to(h.dtype)], -1)
            q, k, v = self.q(h), self.k(msg_in), self.v(msg_in)
            scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.d_comm) + self.alpha * disagreement
            attn = torch.softmax(scores.masked_fill(~can_hear, -1e9), -1) * has_any
            heard = attn.to(v.dtype) @ v
            toward = (attn @ pos - pos * has_any).to(h.dtype)                        # куда лететь к говорящим
            h = h + self.o(torch.cat([h, heard, toward], -1))
        beliefs.append(self.belief(h.detach()))
        return h, {"beliefs": beliefs, "attn": attn}


def make_comm(comm: str, hidden: int, cells: int, rounds: int, radius: float) -> Optional[CidaComm]:
    """none — без связи; attn — обычное внимание между дронами; cida — внимание со смещением разногласием."""
    if comm == "none":
        return None
    return CidaComm(hidden, cells, rounds=rounds, radius=radius, disagreement_bias=(comm == "cida"))


class Actor(nn.Module):
    def __init__(self, env_cfg: EnvConfig, hidden: int = 256, comm: str = "none",
                 comm_radius: float = 0.3, comm_rounds: int = 2):
        super().__init__()
        if comm not in COMM_TYPES:
            raise ValueError(f"comm должен быть одним из {COMM_TYPES}, а не {comm}")
        self.det_enc = _mlp(DET_FEATURES, hidden, hidden)
        self.map_enc = MapEncoder(env_cfg.grid, hidden)
        self.self_enc = nn.Sequential(nn.Linear(SELF_FEATURES, hidden), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.comm = make_comm(comm, hidden, self.map_enc.coarse ** 2, comm_rounds, comm_radius)
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


class RescueActor(nn.Module):
    """
    Мозг дрона для спасательной миссии: окно вокруг себя (стены, завалы, пострадавшие с весом, пожар, другие дроны,
    исследованное, выход, кто уже держит человека) + грубая карта здания + о себе. Действие — направление × «взять».
    """

    def __init__(self, env_cfg: RescueConfig, hidden: int = 256, comm: str = "none",
                 comm_radius: float = 1.5, comm_rounds: int = 2, memory: bool = False, roles: bool = False,
                 personas: bool = False):
        super().__init__()
        if comm not in COMM_TYPES:
            raise ValueError(f"comm должен быть одним из {COMM_TYPES}, а не {comm}")
        win = (2 * env_cfg.view + 1) ** 2
        self.block_cells = float((env_cfg.grid // env_cfg.coarse) ** 2)  # клеток в блоке грубой карты
        # масштаб каналов окна: вес завала и человека до 3, дронов на клетке до ~3, шагов до выхода до ~100
        self.register_buffer("scale", torch.tensor([1, 3, 3, 1, 3, 1, 1, 3, 100], dtype=torch.float)[:, None],
                             persistent=False)
        self.win_enc = nn.Sequential(nn.Linear(RESCUE_WINDOW_CHANNELS * win, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.coarse_enc = nn.Sequential(nn.Linear(RESCUE_COARSE_CHANNELS * env_cfg.coarse ** 2, hidden), nn.ReLU())
        self.self_enc = nn.Sequential(nn.Linear(RESCUE_SELF_FEATURES, hidden), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.comm = make_comm(comm, hidden, env_cfg.coarse ** 2, comm_rounds, comm_radius)
        self.memory, self.hidden = memory, hidden
        self.role_enc = None
        # постоянная личность каждого дрона (по его номеру): своя на всю жизнь, учится вместе с роем —
        # обучение само решает, кто разведчик, а кто носильщик; с нуля, чтобы дообучение начиналось с прежнего
        self.persona = nn.Parameter(torch.zeros(env_cfg.n_drones, hidden)) if personas else None
        if roles:
            # случайная роль на эпизод разбивает симметрию: одинаковые мозги могут разойтись по разным делам;
            # нулевая инициализация — дообучение стартует с прежнего поведения
            self.role_enc = nn.Linear(RESCUE_ROLE_DIM, hidden)
            nn.init.zeros_(self.role_enc.weight)
            nn.init.zeros_(self.role_enc.bias)
        if memory:
            # память видит восприятие и то, что дрон услышал на прошлом шаге
            self.gru = nn.GRUCell(hidden * (2 if self.comm is not None else 1), hidden)
            # память подмешивается к восприятию через нулевую проекцию: дообучение стартует ровно с роя без памяти
            self.mem_out = nn.Linear(hidden, hidden)
            nn.init.zeros_(self.mem_out.weight)
            nn.init.zeros_(self.mem_out.bias)
        self.head = nn.Linear(hidden, RESCUE_N_ACTIONS)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        w = (obs["window"].float() / self.scale).flatten(-2)
        c = (obs["coarse"].float() / self.block_cells).flatten(-2)
        x = self.trunk(torch.cat([self.win_enc(w), self.coarse_enc(c), self.self_enc(obs["self"])], -1))
        if self.role_enc is not None:
            x = x + self.role_enc(obs["role"].to(x.dtype))
        if self.persona is not None:
            x = x + self.persona.to(x.dtype)
        return x

    @property
    def memory_size(self) -> int:
        """Длина памяти дрона: состояние GRU + услышанное на прошлом шаге (если есть связь)."""
        return self.hidden * (2 if self.comm is not None else 1) if self.memory else 0

    def step(self, obs: Dict[str, torch.Tensor], mem: Optional[torch.Tensor]):
        """Один шаг роя с памятью: (логиты, aux, новая память). mem (..., N, memory_size) или None — пустая."""
        x = self.encode(obs)
        if not self.memory:
            h, aux = x, None
            if self.comm is not None:
                h, aux = self.comm(h, obs["pos"], obs["active"])
            return self.head(h), aux, None
        if mem is None:
            mem = x.new_zeros(*x.shape[:-1], self.memory_size)
        state, heard = mem[..., :self.hidden], mem[..., self.hidden:]
        inp = torch.cat([x, heard.to(x.dtype)], -1) if self.comm is not None else x
        m = self.gru(inp.flatten(0, -2).float(), state.flatten(0, -2).float()).view(*x.shape[:-1], -1)
        m = m * obs["active"][..., None]                      # выбывший дрон ничего не помнит
        base = x + self.mem_out(m.to(x.dtype))
        h, aux = base, None
        if self.comm is not None:
            h, aux = self.comm(base, obs["pos"], obs["active"])
            new_mem = torch.cat([m.float(), (h - base).float()], -1)
        else:
            new_mem = m.float()
        return self.head(h), aux, new_mem

    def forward(self, obs: Dict[str, torch.Tensor], return_aux: bool = False):
        logits, aux, _ = self.step(obs, None)
        return (logits, aux) if return_aux else logits


def init_memory(actor: nn.Module, n_envs: int, n_drones: int, device) -> Optional[torch.Tensor]:
    """Пустая память роя (None, если у мозга памяти нет)."""
    size = getattr(actor, "memory_size", 0)
    return torch.zeros(n_envs, n_drones, size, device=device) if size else None


def policy_step(actor: nn.Module, obs: Dict[str, torch.Tensor], mem: Optional[torch.Tensor]):
    """Шаг любого мозга: (логиты, aux, память). У мозга без памяти память всегда None."""
    if getattr(actor, "memory", False):
        return actor.step(obs, mem)
    logits, aux = actor(obs, return_aux=True)
    return logits, aux, None


def forget_done(mem: Optional[torch.Tensor], done: torch.Tensor) -> Optional[torch.Tensor]:
    """Миры, где эпизод закончился, начинаются заново — их память стирается."""
    return None if mem is None else mem * (~done.bool())[:, None, None]


class Critic(nn.Module):
    """Видит весь мир (только при обучении) и оценивает общую награду команды (в нормированной шкале)."""

    def __init__(self, state_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
