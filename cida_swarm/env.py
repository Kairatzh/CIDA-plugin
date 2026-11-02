"""
env.py — векторизованная среда «поиск в тумане» для роя дронов (шаг 1 CIDA-Swarm).

Тысячи миров считаются одним батчем тензоров на GPU. В каждом мире N дронов ищут M людей
на поле [0, 1]². Видимость маленькая, поэтому без памяти и кооперации поиск идёт долго.

Глаза и ноги — имитация с тем же интерфейсом, что будет у настоящих в 3D:
    глаза: набор находок (относительная позиция, уверенность), с пропусками, шумом и ложными срабатываниями;
    ноги:  команда «желаемое направление», исполняется с инерцией и шумом.

Награда общая для команды: +1 за найденного человека и небольшой бонус за новые исследованные клетки.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch

_S = 1.0 / math.sqrt(2.0)
# 0 — зависнуть, 1..8 — восемь направлений
DIRECTIONS = torch.tensor([
    [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    [_S, _S], [_S, -_S], [-_S, _S], [-_S, -_S],
])
N_ACTIONS = len(DIRECTIONS)
DET_FEATURES = 4   # dx, dy, уверенность, валидность
SELF_FEATURES = 5  # x, y, vx, vy, доля прошедшего времени


@dataclass
class EnvConfig:
    n_envs: int = 1024
    n_drones: int = 6
    n_targets: int = 5
    grid: int = 16              # сетка для учёта исследованных клеток
    max_steps: int = 200
    view_radius: float = 0.15
    find_radius: float = 0.05
    max_speed: float = 0.02
    inertia: float = 0.5        # доля старой скорости — «тяжесть» ног
    move_noise: float = 0.002
    p_detect: float = 0.9       # шанс увидеть человека в радиусе обзора
    false_pos_rate: float = 0.02
    pos_noise: float = 0.01
    k_detect: int = 4           # сколько ближайших находок видит дрон
    reward_find: float = 1.0
    reward_explore: float = 0.01
    share_coverage: bool = False  # стигмергия: дроны видят общую карту исследованного

    def to_dict(self) -> Dict:
        return asdict(self)


class SearchEnv:
    """Батч независимых миров с автоматическим перезапуском завершившихся эпизодов."""

    def __init__(self, cfg: EnvConfig, device: torch.device | str = "cpu", seed: Optional[int] = None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed if seed is not None else torch.seed() % (2 ** 31))
        b, n, m, g = cfg.n_envs, cfg.n_drones, cfg.n_targets, cfg.grid
        self.dirs = DIRECTIONS.to(self.device) * cfg.max_speed
        self.pos = torch.zeros(b, n, 2, device=self.device)
        self.vel = torch.zeros(b, n, 2, device=self.device)
        self.targets = torch.zeros(b, m, 2, device=self.device)
        self.alive = torch.zeros(b, m, dtype=torch.bool, device=self.device)
        self.team_cov = torch.zeros(b, g * g, dtype=torch.bool, device=self.device)
        self.own_cov = torch.zeros(b, n, g * g, dtype=torch.bool, device=self.device)
        self.t = torch.zeros(b, dtype=torch.long, device=self.device)
        self.ep_found = torch.zeros(b, device=self.device)

    # ── служебное ────────────────────────────────────────────────────────────
    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def _cover(self) -> torch.Tensor:
        """Отмечает клетки под дронами; возвращает число новых клеток команды в каждом мире."""
        g = self.cfg.grid
        cell = (self.pos * g).long().clamp_(0, g - 1)
        idx = cell[..., 0] * g + cell[..., 1]                      # (B, N)
        before = self.team_cov.sum(1)
        self.team_cov.scatter_(1, idx, True)
        self.own_cov.scatter_(2, idx.unsqueeze(-1), True)
        return (self.team_cov.sum(1) - before).float()

    def _reset_where(self, mask: torch.Tensor) -> None:
        k = int(mask.sum())
        if k == 0:
            return
        c = self.cfg
        self.pos[mask] = self._rand(k, c.n_drones, 2)
        self.vel[mask] = 0.0
        self.targets[mask] = self._rand(k, c.n_targets, 2)
        self.alive[mask] = True
        self.team_cov[mask] = False
        self.own_cov[mask] = False
        self.t[mask] = 0
        self.ep_found[mask] = 0.0
        self._cover()  # стартовые клетки — без награды

    # ── API ──────────────────────────────────────────────────────────────────
    def reset(self) -> Dict[str, torch.Tensor]:
        self._reset_where(torch.ones(self.cfg.n_envs, dtype=torch.bool, device=self.device))
        return self.observe()

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """actions: (B, N) целые 0..8. Возвращает obs, награду (B,), done (B,), статистику завершённых эпизодов."""
        c = self.cfg
        desired = self.dirs[actions]
        self.vel = c.inertia * self.vel + (1.0 - c.inertia) * desired + c.move_noise * self._randn(*self.vel.shape)
        self.pos = (self.pos + self.vel).clamp_(0.0, 1.0)

        reward = c.reward_explore * self._cover()

        dist = torch.cdist(self.pos, self.targets)                 # (B, N, M)
        found = (dist < c.find_radius).any(1) & self.alive          # (B, M)
        n_found = found.sum(1).float()
        self.alive &= ~found
        self.ep_found += n_found
        reward = reward + c.reward_find * n_found

        self.t += 1
        done = (self.t >= c.max_steps) | ~self.alive.any(1)
        info = {
            "found_frac": self.ep_found[done] / c.n_targets,
            "length": self.t[done].float(),
            "coverage": self.team_cov[done].float().mean(1),
        }
        self._reset_where(done)
        return self.observe(), reward, done, info

    def observe(self) -> Dict[str, torch.Tensor]:
        c = self.cfg
        b, n, m = c.n_envs, c.n_drones, c.n_targets

        # настоящие люди: видны в радиусе обзора с вероятностью p_detect, позиция с шумом
        rel = self.targets[:, None] - self.pos[:, :, None]          # (B, N, M, 2)
        visible = self.alive[:, None] & (rel.norm(dim=-1) < c.view_radius) & (self._rand(b, n, m) < c.p_detect)
        rel = rel + c.pos_noise * self._randn(b, n, m, 2)
        conf = 0.6 + 0.4 * self._rand(b, n, m)

        # ложное срабатывание: случайная точка в радиусе обзора с низкой уверенностью
        fake = self._rand(b, n) < c.false_pos_rate
        ang = 2 * math.pi * self._rand(b, n)
        r = c.view_radius * self._rand(b, n).sqrt()
        fake_rel = torch.stack([r * ang.cos(), r * ang.sin()], -1)
        fake_conf = 0.3 + 0.5 * self._rand(b, n)

        all_rel = torch.cat([rel, fake_rel[:, :, None]], 2)         # (B, N, M+1, 2)
        all_conf = torch.cat([conf, fake_conf[..., None]], 2)
        valid = torch.cat([visible, fake[..., None]], 2)

        # k ближайших валидных находок; невалидные слоты обнуляются
        key = torch.where(valid, all_rel.norm(dim=-1), torch.full_like(all_conf, float("inf")))
        k = min(c.k_detect, m + 1)
        idx = key.topk(k, dim=-1, largest=False).indices
        sel_rel = all_rel.gather(2, idx[..., None].expand(-1, -1, -1, 2))
        sel_conf = all_conf.gather(2, idx)
        sel_valid = valid.gather(2, idx).float()
        det = torch.cat([sel_rel / c.view_radius, sel_conf[..., None], sel_valid[..., None]], -1) * sel_valid[..., None]
        if k < c.k_detect:
            det = torch.cat([det, det.new_zeros(b, n, c.k_detect - k, DET_FEATURES)], 2)

        t_frac = (self.t.float() / c.max_steps)[:, None, None].expand(b, n, 1)
        own = torch.cat([self.pos * 2 - 1, self.vel / c.max_speed, t_frac], -1)
        cov = self.team_cov[:, None].expand(b, n, -1) if c.share_coverage else self.own_cov
        return {"self": own, "det": det, "cov": cov.float()}

    def state(self) -> torch.Tensor:
        """Полное состояние мира для централизованного критика (только при обучении)."""
        c = self.cfg
        return torch.cat([
            self.pos.flatten(1) * 2 - 1,
            self.vel.flatten(1) / c.max_speed,
            (self.targets * 2 - 1).mul(self.alive[..., None]).flatten(1),
            self.alive.float(),
            self.team_cov.float(),
            (self.t.float() / c.max_steps)[:, None],
        ], 1)

    @staticmethod
    def state_dim(cfg: EnvConfig) -> int:
        return cfg.n_drones * 4 + cfg.n_targets * 3 + cfg.grid ** 2 + 1
