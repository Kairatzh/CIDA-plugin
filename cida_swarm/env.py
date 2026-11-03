"""
env.py — векторизованная среда «поиск в тумане» для роя дронов (CIDA-Swarm).

Тысячи миров считаются одним батчем тензоров на GPU. В каждом мире до N дронов ищут M людей
на поле [0, 1]². Чтобы задача была похожа на настоящую, а не на игрушку:
    * здания/лес — случайные препятствия, сквозь которые нельзя пролететь;
    * люди медленно двигаются, поэтому старая информация устаревает;
    * «приманки» — неподвижные объекты, похожие на людей (мешки, манекены), их видно как находки;
    * чем дальше объект, тем реже его замечают; бывают и случайные ложные срабатывания;
    * в каждом мире своё число дронов (от min_drones до n_drones) — рой не привыкает к одному составу.

Глаза и ноги — имитация с тем же интерфейсом, что будет у настоящих в 3D:
    глаза: набор находок (относительная позиция, уверенность);
    ноги:  команда «желаемое направление», исполняется с инерцией и шумом.

Шаг среды не синхронизирует GPU с CPU: перезапуск завершившихся миров идёт через маски,
поэтому GPU не простаивает.
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
SELF_FEATURES = 6  # x, y, vx, vy, доля прошедшего времени, доля активных дронов


@dataclass
class EnvConfig:
    n_envs: int = 8192
    n_drones: int = 8           # максимум дронов в мире
    min_drones: int = 4         # минимум; в каждом мире число выбирается случайно
    n_targets: int = 8
    n_decoys: int = 4
    grid: int = 24              # сетка карты: учёт исследованного и препятствия
    max_steps: int = 200
    view_radius: float = 0.08
    find_radius: float = 0.04
    max_speed: float = 0.015
    inertia: float = 0.5        # доля старой скорости — «тяжесть» ног
    move_noise: float = 0.002
    target_speed: float = 0.002  # люди бродят
    p_detect: float = 0.9       # шанс заметить объект вплотную; на краю обзора — вдвое меньше
    false_pos_rate: float = 0.01
    pos_noise: float = 0.01
    k_detect: int = 6           # сколько ближайших находок видит дрон
    n_obstacles: int = 6
    obstacle_min: int = 2       # размер препятствия в клетках
    obstacle_max: int = 5
    reward_find: float = 1.0
    reward_explore: float = 0.01
    share_coverage: bool = False  # стигмергия: дроны видят общую карту исследованного
    refresh_every: int = 16     # как часто обновлять пул свежих миров для перезапуска

    def __post_init__(self):
        if not 1 <= self.min_drones <= self.n_drones:
            raise ValueError("нужно 1 <= min_drones <= n_drones")

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
        self.cells = g * g
        self.dirs = DIRECTIONS.to(self.device) * cfg.max_speed
        self.pos = torch.zeros(b, n, 2, device=self.device)
        self.vel = torch.zeros(b, n, 2, device=self.device)
        self.active = torch.zeros(b, n, dtype=torch.bool, device=self.device)
        self.targets = torch.zeros(b, m, 2, device=self.device)
        self.alive = torch.zeros(b, m, dtype=torch.bool, device=self.device)
        self.decoys = torch.zeros(b, cfg.n_decoys, 2, device=self.device)
        self.obstacles = torch.zeros(b, self.cells, dtype=torch.bool, device=self.device)
        # последний столбец — «мусорная» клетка, куда пишут неактивные дроны
        self.team_cov = torch.zeros(b, self.cells + 1, dtype=torch.bool, device=self.device)
        self.own_cov = torch.zeros(b, n, self.cells + 1, dtype=torch.bool, device=self.device)
        self.t = torch.zeros(b, dtype=torch.long, device=self.device)
        self.ep_found = torch.zeros(b, device=self.device)

    # ── служебное ────────────────────────────────────────────────────────────
    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def _cell(self, pos: torch.Tensor) -> torch.Tensor:
        g = self.cfg.grid
        c = (pos * g).long().clamp_(0, g - 1)
        return c[..., 0] * g + c[..., 1]

    def _blocked(self, pos: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
        """pos (B, K, 2) → занята ли препятствием клетка каждой точки (B, K)."""
        return obstacles.gather(1, self._cell(pos))

    def _make_obstacles(self) -> torch.Tensor:
        c, b, g = self.cfg, self.cfg.n_envs, self.cfg.grid
        if c.n_obstacles == 0:
            return torch.zeros(b, self.cells, dtype=torch.bool, device=self.device)
        span = c.obstacle_max - c.obstacle_min + 1
        size = c.obstacle_min + (self._rand(b, c.n_obstacles, 2) * span).long()
        corner = (self._rand(b, c.n_obstacles, 2) * (g - size + 1)).long()
        xs = torch.arange(g, device=self.device)
        in_x = (xs >= corner[..., 0:1]) & (xs < corner[..., 0:1] + size[..., 0:1])   # (B, R, G)
        in_y = (xs >= corner[..., 1:2]) & (xs < corner[..., 1:2] + size[..., 1:2])
        return (in_x[..., :, None] & in_y[..., None, :]).any(1).flatten(1)

    def _sample_free(self, k: int, obstacles: torch.Tensor, tries: int = 16) -> torch.Tensor:
        """k случайных точек вне препятствий в каждом мире (из нескольких попыток берётся первая свободная)."""
        b = self.cfg.n_envs
        cand = self._rand(b, k * tries, 2)
        free = ~self._blocked(cand, obstacles).view(b, k, tries)
        first = free.float().argmax(-1)                                               # (B, k)
        cand = cand.view(b, k, tries, 2)
        return cand.gather(2, first[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)

    def _cover(self) -> torch.Tensor:
        """Отмечает клетки под активными дронами; возвращает число новых клеток команды."""
        idx = torch.where(self.active, self._cell(self.pos), torch.full_like(self.t[:, None], self.cells))
        before = self.team_cov[:, :-1].sum(1)
        self.team_cov.scatter_(1, idx, True)
        self.own_cov.scatter_(2, idx.unsqueeze(-1), True)
        return (self.team_cov[:, :-1].sum(1) - before).float()

    def _refresh_pool(self) -> None:
        """
        Заранее генерирует по одному свежему миру на каждый слот. Генерация дорогая, а заканчивается
        за шаг лишь ~1/max_steps миров, поэтому пул обновляется раз в refresh_every шагов, а не каждый шаг.
        """
        c = self.cfg
        obstacles = self._make_obstacles()
        n_active = c.min_drones + (self._rand(c.n_envs) * (c.n_drones - c.min_drones + 1)).long()
        self._fresh = {
            "obstacles": obstacles,
            "active": torch.arange(c.n_drones, device=self.device) < n_active[:, None],
            "pos": self._sample_free(c.n_drones, obstacles),
            "targets": self._sample_free(c.n_targets, obstacles),
            "decoys": self._sample_free(c.n_decoys, obstacles) if c.n_decoys else self.decoys,
        }
        self._since_refresh = 0

    def _reset_where(self, mask: torch.Tensor) -> None:
        """Перезапускает миры по маске без синхронизации с CPU: миры берутся из пула и смешиваются по маске."""
        m1, m2 = mask[:, None], mask[:, None, None]
        f = self._fresh
        self.obstacles = torch.where(m1, f["obstacles"], self.obstacles)
        self.active = torch.where(m1, f["active"], self.active)
        self.pos = torch.where(m2, f["pos"], self.pos)
        self.vel = torch.where(m2, torch.zeros_like(self.vel), self.vel)
        self.targets = torch.where(m2, f["targets"], self.targets)
        self.decoys = torch.where(m2, f["decoys"], self.decoys)
        self.alive = self.alive | mask[:, None]
        self.team_cov &= ~m1
        self.own_cov &= ~m2
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        self.ep_found = torch.where(mask, torch.zeros_like(self.ep_found), self.ep_found)
        self._cover()  # стартовые клетки — без награды

    def _move(self, pos: torch.Tensor, step: torch.Tensor):
        """Сдвиг с учётом препятствий: в занятую клетку не входим. Возвращает (новая позиция, упёрся ли)."""
        new = (pos + step).clamp(0.0, 1.0)
        hit = self._blocked(new, self.obstacles)
        return torch.where(hit[..., None], pos, new), hit

    def free_cells(self) -> torch.Tensor:
        return (~self.obstacles).sum(1).clamp_min(1).float()

    # ── API ──────────────────────────────────────────────────────────────────
    def reset(self) -> Dict[str, torch.Tensor]:
        self.alive.zero_()
        self._refresh_pool()
        self._reset_where(torch.ones(self.cfg.n_envs, dtype=torch.bool, device=self.device))
        return self.observe()

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        actions: (B, N) целые 0..8 (действия неактивных дронов игнорируются).
        Возвращает obs, награду (B,), done (B,) и статистику эпизода для каждого мира
        (имеет смысл там, где done=True).
        """
        c = self.cfg
        act = self.active[..., None]
        desired = self.dirs[actions]
        vel = c.inertia * self.vel + (1.0 - c.inertia) * desired + c.move_noise * self._randn(*self.vel.shape)
        pos, hit = self._move(self.pos, vel)
        self.pos = torch.where(act, pos, self.pos)
        self.vel = torch.where(act & ~hit[..., None], vel, torch.zeros_like(vel))

        tpos, _ = self._move(self.targets, c.target_speed * self._randn(*self.targets.shape))
        self.targets = torch.where(self.alive[..., None], tpos, self.targets)

        reward = c.reward_explore * self._cover()

        dist = torch.cdist(self.pos, self.targets).masked_fill(~act, float("inf"))   # (B, N, M)
        found = (dist < c.find_radius).any(1) & self.alive
        n_found = found.sum(1).float()
        self.alive &= ~found
        self.ep_found += n_found
        reward = reward + c.reward_find * n_found

        self.t += 1
        done = (self.t >= c.max_steps) | ~self.alive.any(1)
        self._since_refresh += 1
        if self._since_refresh >= c.refresh_every:
            self._refresh_pool()
        info = {
            "found_frac": self.ep_found / c.n_targets,
            "length": self.t.float(),
            "coverage": (self.team_cov[:, :-1] & ~self.obstacles).sum(1).float() / self.free_cells(),
        }
        self._reset_where(done)
        return self.observe(), reward, done, info

    def observe(self) -> Dict[str, torch.Tensor]:
        c = self.cfg
        b, n, m, d = c.n_envs, c.n_drones, c.n_targets, c.n_decoys

        # люди и приманки видны в радиусе обзора; чем дальше, тем реже замечаем
        objects = torch.cat([self.targets, self.decoys], 1)                         # (B, M+D, 2)
        present = torch.cat([self.alive, self.alive.new_ones(b, d)], 1)
        rel = objects[:, None] - self.pos[:, :, None]                                # (B, N, M+D, 2)
        dist = rel.norm(dim=-1)
        p_see = c.p_detect * (1.0 - 0.5 * dist / c.view_radius)
        visible = present[:, None] & (dist < c.view_radius) & (self._rand(b, n, m + d) < p_see)
        rel = rel + c.pos_noise * self._randn(b, n, m + d, 2)
        conf = torch.cat([0.5 + 0.5 * self._rand(b, n, m), 0.3 + 0.55 * self._rand(b, n, d)], 2)

        # случайное ложное срабатывание сенсора
        fake = self._rand(b, n) < c.false_pos_rate
        ang = 2 * math.pi * self._rand(b, n)
        r = c.view_radius * self._rand(b, n).sqrt()
        fake_rel = torch.stack([r * ang.cos(), r * ang.sin()], -1)

        all_rel = torch.cat([rel, fake_rel[:, :, None]], 2)
        all_conf = torch.cat([conf, (0.2 + 0.5 * self._rand(b, n))[..., None]], 2)
        valid = torch.cat([visible, fake[..., None]], 2) & self.active[..., None]

        # k ближайших валидных находок; невалидные слоты обнуляются
        key = torch.where(valid, all_rel.norm(dim=-1), torch.full_like(all_conf, float("inf")))
        k = min(c.k_detect, m + d + 1)
        idx = key.topk(k, dim=-1, largest=False).indices
        sel_rel = all_rel.gather(2, idx[..., None].expand(-1, -1, -1, 2))
        sel_conf = all_conf.gather(2, idx)
        sel_valid = valid.gather(2, idx).float()
        det = torch.cat([sel_rel / c.view_radius, sel_conf[..., None], sel_valid[..., None]], -1) * sel_valid[..., None]
        if k < c.k_detect:
            det = torch.cat([det, det.new_zeros(b, n, c.k_detect - k, DET_FEATURES)], 2)

        t_frac = (self.t.float() / c.max_steps)[:, None, None].expand(b, n, 1)
        team = (self.active.sum(1, keepdim=True).float() / c.n_drones)[..., None].expand(b, n, 1)
        own = torch.cat([self.pos * 2 - 1, self.vel / c.max_speed, t_frac, team], -1) * self.active[..., None]
        cov = self.team_cov[:, None, :-1].expand(b, n, -1) if c.share_coverage else self.own_cov[..., :-1]
        return {
            "self": own,
            "det": det,
            "cov": cov.clone(),               # bool (B, N, G²) — карта храним компактно
            "obstacles": self.obstacles.clone(),  # bool (B, G²) — одна на мир
            "cell": self._cell(self.pos),     # (B, N) — где я на карте
            "active": self.active.clone(),    # (B, N)
        }

    def people_map(self, coarse: int = 6) -> torch.Tensor:
        """Где сейчас ненайденные люди на грубой карте coarse×coarse — правда для обучения убеждений (B, coarse²)."""
        c = (self.targets * coarse).long().clamp_(0, coarse - 1)
        idx = torch.where(self.alive, c[..., 0] * coarse + c[..., 1], torch.full_like(c[..., 0], coarse * coarse))
        out = torch.zeros(self.cfg.n_envs, coarse * coarse + 1, dtype=torch.bool, device=self.device)
        return out.scatter_(1, idx, True)[:, :-1]

    def state(self) -> torch.Tensor:
        """Полное состояние мира для централизованного критика (только при обучении)."""
        c = self.cfg
        act = self.active.float()
        return torch.cat([
            (self.pos * 2 - 1).mul(act[..., None]).flatten(1),
            (self.vel / c.max_speed).mul(act[..., None]).flatten(1),
            act,
            (self.targets * 2 - 1).mul(self.alive[..., None]).flatten(1),
            self.alive.float(),
            self.obstacles.float(),
            self.team_cov[:, :-1].float(),
            (self.t.float() / c.max_steps)[:, None],
        ], 1)

    @staticmethod
    def state_dim(cfg: EnvConfig) -> int:
        return cfg.n_drones * 5 + cfg.n_targets * 3 + 2 * cfg.grid ** 2 + 1
