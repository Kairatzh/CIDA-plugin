"""
rescue.py — мир «спасательная миссия» (CIDA-Rescue): рой дронов выносит людей из здания.

Здание — лабиринт из rooms×rooms комнат с дверями (всегда связный, с несколькими обходными путями).
Дроны входят через комнату-вход в углу; она же безопасная зона. Задача — найти пострадавших и вынести их.

Почему здесь без договорённости не обойтись:
    * у пострадавшего есть вес w (1–3): сдвинуть его могут только w дронов, стоящих на его клетке и тянущих
      в ОДНУ сторону; дрон на клетке человека держит его, пока не выберет «отпустить», поэтому один дрон
      у тяжёлого человека может только ждать помощи;
    * завалы в дверях весом 2–3 разбираются, только если столько же дронов толкают завал одновременно;
    * пожар/опасный человек бродит по зданию — дрон рядом с ним выходит из строя;
    * обрушения: посреди миссии дверь может завалить, и маршрут надо перестраивать;
    * каждый дрон видит только окно вокруг себя.

Режим «без подсказок» (для роя с памятью и связью):
    * route_hint=False — нет готовой «дорожки к выходу»: путь назад надо помнить самому или узнать у других;
    * hidden_weight=False → True — вес человека знает только тот, кто его держит: остальным его надо сообщить;
    * hazard_chase > 0 — опасный человек не бродит, а охотится на ближайшего дрона (сложность 4):
      нужны тактики — обходить, отвлекать, разделяться.

Сложность мира d (1–4) выбирается для каждого мира случайно из [difficulty_min, difficulty_max]:
    1 — лёгкие пострадавшие (вес 1); 2 — вес 1–3; 3 — плюс завалы; 4 — плюс пожар и обрушения.
Смесь простых и сложных миров даёт плотную награду с первых итераций и сложные задачи для роста.

Движение по клеткам: действие = направление (стоять, ±x, ±y) × «отпустить» (release). Всё считается батчем на GPU
без синхронизаций с CPU; свежие миры для перезапуска готовятся пачкой (пул), как в env.SearchEnv.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch

MOVES = torch.tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
N_DIRS = len(MOVES)
N_ACTIONS = N_DIRS * 2            # направление × «отпустить»
WINDOW_CHANNELS = 9               # стена, завал, пострадавший, пожар, дроны, исследовано, выход, «держат», путь к выходу
ROUTE_UNKNOWN = 251               # «шагов до выхода» для неисследованных клеток
COARSE_CHANNELS = 2               # исследовано, известные стены
SELF_FEATURES = 9
INF = 1e4


@dataclass
class RescueConfig:
    n_envs: int = 8192
    n_drones: int = 6
    min_drones: int = 3
    n_victims: int = 3
    rooms: int = 5                # здание rooms × rooms комнат 3×3, стены толщиной в клетку → сетка 21×21
    max_steps: int = 300
    view: int = 3                 # радиус обзора: окно (2·view+1)²
    coarse: int = 7               # грубая карта coarse × coarse
    extra_doors: float = 0.25     # доля дополнительных дверей → обходные пути
    difficulty_min: int = 1
    difficulty_max: int = 4
    rubble_prob: float = 0.35     # доля дверей с завалом (сложность ≥ 3)
    collapse_prob: float = 0.01   # шанс обрушения двери за шаг (сложность 4)
    hazard_move_prob: float = 0.5
    reward_deliver: float = 1.0   # + 0.5 за каждую единицу веса сверх 1
    reward_discover: float = 0.2
    reward_contact: float = 0.1   # первый дрон на клетке пострадавшего
    reward_progress: float = 0.05  # за клетку ближе к выходу для переносимого человека
    reward_clear: float = 0.3
    reward_explore: float = 0.002
    penalty_lost: float = 0.5
    share_map: bool = False       # общая карта исследованного (стигмергия)
    route_hint: bool = True       # канал «шагов до выхода» на исследованных клетках
    hidden_weight: bool = False   # вес человека виден только держащему его дрону
    hazard_chase: float = 0.0     # доля ходов, когда опасный человек идёт к ближайшему дрону
    refresh_every: int = 32

    def __post_init__(self):
        if not 1 <= self.min_drones <= self.n_drones:
            raise ValueError("нужно 1 <= min_drones <= n_drones")
        if not 1 <= self.difficulty_min <= self.difficulty_max <= 4:
            raise ValueError("сложность — от 1 до 4")
        if self.grid % self.coarse:
            raise ValueError(f"сетка {self.grid} должна делиться на coarse={self.coarse}")

    @property
    def grid(self) -> int:
        return self.rooms * 4 + 1

    def to_dict(self) -> Dict:
        return asdict(self)


class RescueEnv:
    """Батч зданий; API как у SearchEnv: reset / step / observe / state / belief_target."""

    STATS = ("rescued_frac", "all_rescued", "length", "drones_lost", "found_frac")
    SUCCESS = "rescued_frac"

    def __init__(self, cfg: RescueConfig, device: torch.device | str = "cpu", seed: Optional[int] = None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed if seed is not None else torch.seed() % (2 ** 31))
        b, n, m, g = cfg.n_envs, cfg.n_drones, cfg.n_victims, cfg.grid
        self.G, self.cells = g, g * g
        dev = self.device
        self.moves = MOVES.to(dev)

        # статичная геометрия: внутренности комнат, двери, выход
        xs = torch.arange(g, device=dev)
        interior = (xs[:, None] % 4 != 0) & (xs[None, :] % 4 != 0)
        self.interior = interior.flatten()
        r = cfg.rooms
        east = [((4 * (i + 1)) * g + 4 * j + 2, i, j) for i in range(r - 1) for j in range(r)]
        north = [((4 * i + 2) * g + 4 * (j + 1), i, j) for i in range(r) for j in range(r - 1)]
        self.east_idx = torch.tensor([c for c, _, _ in east], device=dev)
        self.east_room = torch.tensor([i * r + j for _, i, j in east], device=dev)
        self.north_idx = torch.tensor([c for c, _, _ in north], device=dev)
        self.north_room = torch.tensor([i * r + j for _, i, j in north], device=dev)
        self.door_idx = torch.cat([self.east_idx, self.north_idx])
        exit_mask = torch.zeros(g, g, dtype=torch.bool, device=dev)
        exit_mask[1:4, 1:4] = True
        self.exit_mask = exit_mask.flatten()
        self.exit_center = torch.tensor([2, 2], device=dev)

        # окно обзора
        v = cfg.view
        off = torch.arange(-v, v + 1, device=dev)
        self.win_dx = off.repeat_interleave(2 * v + 1)
        self.win_dy = off.repeat(2 * v + 1)

        # состояние
        self.walls = torch.ones(b, self.cells, dtype=torch.bool, device=dev)
        self.rubble = torch.zeros(b, self.cells, dtype=torch.long, device=dev)
        self.dist = torch.full((b, self.cells), INF, device=dev)
        self.difficulty = torch.ones(b, dtype=torch.long, device=dev)
        self.pos = torch.zeros(b, n, 2, dtype=torch.long, device=dev)
        self.active = torch.zeros(b, n, dtype=torch.bool, device=dev)   # дрон участвует в миссии
        self.alive = torch.zeros(b, n, dtype=torch.bool, device=dev)    # и ещё не сгорел
        self.vpos = torch.zeros(b, m, 2, dtype=torch.long, device=dev)
        self.vweight = torch.ones(b, m, dtype=torch.long, device=dev)
        self.present = torch.zeros(b, m, dtype=torch.bool, device=dev)  # ещё не вынесен
        self.seen = torch.zeros(b, m, dtype=torch.bool, device=dev)
        self.touched = torch.zeros(b, m, dtype=torch.bool, device=dev)
        self.hpos = torch.zeros(b, 2, dtype=torch.long, device=dev)
        self.hazard_on = torch.zeros(b, dtype=torch.bool, device=dev)
        self.team_seen = torch.zeros(b, self.cells, dtype=torch.bool, device=dev)
        self.own_seen = torch.zeros(b, n, self.cells, dtype=torch.bool, device=dev)
        self.t = torch.zeros(b, dtype=torch.long, device=dev)
        self.ep_rescued = torch.zeros(b, device=dev)
        self.ep_lost = torch.zeros(b, device=dev)
        self.last = {}  # события последнего шага для демо: кто держит, кто ждёт помощи, кто толкает
        self.level_probs: Optional[torch.Tensor] = None  # вероятности уровней сложности (автокуррикулум)

    # ── служебное ────────────────────────────────────────────────────────────
    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randint(self, high, *shape) -> torch.Tensor:
        return (self._rand(*shape) * high).long()

    def _idx(self, p: torch.Tensor) -> torch.Tensor:
        return p[..., 0] * self.G + p[..., 1]

    def _inside(self, p: torch.Tensor) -> torch.Tensor:
        return ((p >= 0) & (p < self.G)).all(-1)

    def _room_cell(self, room: torch.Tensor) -> torch.Tensor:
        """Случайная клетка внутри комнаты room (индекс i·rooms + j) → (…, 2)."""
        r = self.cfg.rooms
        i, j = room // r, room % r
        return torch.stack([4 * i + 1 + self._randint(3, *room.shape), 4 * j + 1 + self._randint(3, *room.shape)], -1)

    def _relax(self, dist: torch.Tensor, free: torch.Tensor, sweeps: int) -> torch.Tensor:
        """Расстояние до выхода по коридорам: несколько шагов волнового алгоритма (Беллман–Форд)."""
        g = self.G
        exit_mask = self.exit_mask
        for _ in range(sweeps):
            d = dist.view(-1, g, g)
            nb = torch.full_like(d, INF)
            nb[:, 1:] = torch.minimum(nb[:, 1:], d[:, :-1])
            nb[:, :-1] = torch.minimum(nb[:, :-1], d[:, 1:])
            nb[:, :, 1:] = torch.minimum(nb[:, :, 1:], d[:, :, :-1])
            nb[:, :, :-1] = torch.minimum(nb[:, :, :-1], d[:, :, 1:])
            dist = torch.where(free, torch.minimum(dist, nb.flatten(1) + 1), torch.full_like(dist, INF))
            dist = torch.where(exit_mask, torch.zeros_like(dist), dist)
        return dist.clamp_max(INF)

    def _refresh_pool(self) -> None:
        """Пачка свежих зданий для перезапуска завершившихся миров (генерация дорогая — не каждый шаг)."""
        c, b, r, dev = self.cfg, self.cfg.n_envs, self.cfg.rooms, self.device
        if self.level_probs is None:
            diff = c.difficulty_min + self._randint(c.difficulty_max - c.difficulty_min + 1, b)
        else:
            diff = c.difficulty_min + torch.multinomial(self.level_probs, b, replacement=True, generator=self.gen)

        # лабиринт «двоичное дерево»: каждая комната открыта на восток или север → дерево, всё связно
        i = torch.arange(r, device=dev).repeat_interleave(r)
        j = torch.arange(r, device=dev).repeat(r)
        east_ok, north_ok = i < r - 1, j < r - 1
        coin = self._rand(b, r * r) < 0.5
        open_east = east_ok & (coin | ~north_ok)
        open_north = north_ok & (~coin | ~east_ok)
        open_east |= east_ok & (self._rand(b, r * r) < c.extra_doors)
        open_north |= north_ok & (self._rand(b, r * r) < c.extra_doors)
        walls = (~self.interior).expand(b, -1).clone()
        walls[:, self.east_idx] = ~open_east[:, self.east_room]
        walls[:, self.north_idx] = ~open_north[:, self.north_room]

        # завалы в открытых дверях (сложность ≥ 3)
        door_open = ~walls[:, self.door_idx]
        put = door_open & (self._rand(b, len(self.door_idx)) < c.rubble_prob) & (diff[:, None] >= 3)
        rubble = torch.zeros(b, self.cells, dtype=torch.long, device=dev)
        rubble[:, self.door_idx] = put.long() * (2 + self._randint(2, b, len(self.door_idx)))

        # дроны — во входной комнате; пострадавшие и пожар — в других комнатах
        n_active = c.min_drones + self._randint(c.n_drones - c.min_drones + 1, b)
        active = torch.arange(c.n_drones, device=dev) < n_active[:, None]
        pos = self._room_cell(torch.zeros(b, c.n_drones, dtype=torch.long, device=dev))
        vroom = 1 + self._randint(r * r - 1, b, c.n_victims)
        max_w = torch.where(diff >= 2, 3, 1)
        vweight = 1 + (self._rand(b, c.n_victims) * max_w[:, None]).long()
        hroom = 1 + self._randint(r * r - 1, b)
        free = ~walls & (rubble == 0)
        dist = self._relax(torch.full((b, self.cells), INF, device=dev), free, sweeps=4 * r * r + 8)
        self._fresh = {
            "difficulty": diff, "walls": walls, "rubble": rubble, "dist": dist, "active": active, "pos": pos,
            "vpos": self._room_cell(vroom), "vweight": vweight,
            "hpos": self._room_cell(hroom), "hazard_on": diff >= 4,
        }
        self._since_refresh = 0

    def _mark_seen(self) -> torch.Tensor:
        """Отмечает клетки в окне обзора живых дронов; возвращает число новых клеток команды."""
        win = torch.stack([self.pos[..., 0:1] + self.win_dx, self.pos[..., 1:2] + self.win_dy], -1)  # (B,N,W,2)
        ok = self._inside(win) & self.alive[..., None]
        idx = torch.where(ok, self._idx(win.clamp(0, self.G - 1)), torch.full_like(win[..., 0], self.cells))
        before = self.team_seen.sum(1)
        pad = torch.zeros(self.cfg.n_envs, 1, dtype=torch.bool, device=self.device)
        team = torch.cat([self.team_seen, pad], 1).scatter_(1, idx.flatten(1), True)
        self.team_seen = team[:, :-1]
        own = torch.cat([self.own_seen, pad[:, None].expand(-1, self.cfg.n_drones, -1)], 2)
        self.own_seen = own.scatter_(2, idx, True)[..., :-1]
        return (self.team_seen.sum(1) - before).float()

    def _reset_where(self, mask: torch.Tensor) -> None:
        f = self._fresh
        m1, m2 = mask[:, None], mask[:, None, None]
        for name in ("walls", "rubble", "dist", "active", "vweight"):
            setattr(self, name, torch.where(m1, f[name], getattr(self, name)))
        for name in ("pos", "vpos"):
            setattr(self, name, torch.where(m2, f[name], getattr(self, name)))
        self.difficulty = torch.where(mask, f["difficulty"], self.difficulty)
        self.hpos = torch.where(m1, f["hpos"], self.hpos)
        self.hazard_on = torch.where(mask, f["hazard_on"], self.hazard_on)
        self.alive = torch.where(m1, self.active, self.alive)
        self.present = self.present | m1
        self.seen = self.seen & ~m1
        self.touched = self.touched & ~m1
        self.team_seen = self.team_seen & ~m1
        self.own_seen = self.own_seen & ~m2
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        self.ep_rescued = torch.where(mask, torch.zeros_like(self.ep_rescued), self.ep_rescued)
        self.ep_lost = torch.where(mask, torch.zeros_like(self.ep_lost), self.ep_lost)
        self._mark_seen()

    # ── API ──────────────────────────────────────────────────────────────────
    def reset(self) -> Dict[str, torch.Tensor]:
        self.present.zero_()
        self._refresh_pool()
        self._reset_where(torch.ones(self.cfg.n_envs, dtype=torch.bool, device=self.device))
        return self.observe()

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """actions (B, N) в [0, N_ACTIONS): направление = a % 5; a >= 5 — отпустить пострадавшего."""
        c, b, g = self.cfg, self.cfg.n_envs, self.G
        direction, release = actions % N_DIRS, actions >= N_DIRS
        step = self.moves[direction]
        reward = torch.zeros(b, device=self.device)

        # 1) переноска: дрон на клетке человека держит его (если не отпустил); человек сдвигается,
        #    когда ≥ w держащих тянут в одну сторону
        on_victim = (self.pos[:, :, None] == self.vpos[:, None]).all(-1) & self.present[:, None]   # (B,N,M)
        holding = on_victim & (~release & self.alive)[..., None]
        votes = torch.einsum("bnm,bnd->bmd", holding.float(), torch.nn.functional.one_hot(direction, N_DIRS).float())
        votes[..., 0] = 0.0
        best_votes, best_dir = votes.max(-1)
        target = self.vpos + self.moves[best_dir]
        t_idx = self._idx(target.clamp(0, g - 1))
        target_free = self._inside(target) & ~self.walls.gather(1, t_idx) & (self.rubble.gather(1, t_idx) == 0)
        moved = self.present & (best_votes >= self.vweight.float()) & target_free & (best_dir > 0)
        d_before = self.dist.gather(1, self._idx(self.vpos))
        self.vpos = torch.where(moved[..., None], target, self.vpos)
        d_after = self.dist.gather(1, self._idx(self.vpos))
        progress = torch.where(moved & (d_before < INF) & (d_after < INF), d_before - d_after, torch.zeros_like(d_before))
        reward += c.reward_progress * progress.sum(1)
        carriers = (holding & moved[:, None] & (direction[..., None] == best_dir[:, None])).any(-1)
        waiting = holding & (best_votes < self.vweight.float())[:, None]                          # ждут помощи

        delivered = self.present & self.exit_mask[self._idx(self.vpos)]
        reward += (delivered.float() * (c.reward_deliver + 0.5 * (self.vweight - 1).float())).sum(1)
        self.present &= ~delivered
        self.ep_rescued += delivered.sum(1).float()

        # 2) завалы: разбирается, если толкающих не меньше его веса
        dtarget = self.pos + step
        d_idx = self._idx(dtarget.clamp(0, g - 1))
        d_inside = self._inside(dtarget)
        # держащий человека дрон не улетает от него: он ждёт помощи, пока сил не хватит сдвинуть
        free_movers = self.alive & ~holding.any(-1) & (direction > 0) & d_inside
        rubble_before = self.rubble.clone()
        pushing = free_movers & (rubble_before.gather(1, d_idx) > 0)
        push = torch.zeros(b, self.cells + 1, device=self.device)
        push.scatter_add_(1, torch.where(pushing, d_idx, torch.full_like(d_idx, self.cells)), torch.ones_like(d_idx, dtype=torch.float))
        cleared = (rubble_before > 0) & (push[:, :-1] >= rubble_before.float())
        self.rubble = torch.where(cleared, torch.zeros_like(self.rubble), self.rubble)
        reward += c.reward_clear * cleared.sum(1).float()

        # 3) обычное движение (в стены и завалы не входим)
        can_move = free_movers & ~self.walls.gather(1, d_idx) & (rubble_before.gather(1, d_idx) == 0)
        self.pos = torch.where((can_move | carriers)[..., None], dtarget, self.pos)

        # 4) пожар бродит и выводит из строя дронов рядом
        hdir = 1 + self._randint(4, b)
        if c.hazard_chase > 0:
            # охота: шаг по той оси, где до ближайшего живого дрона дальше всего
            gap = self.pos - self.hpos[:, None]
            far = torch.where(self.alive, gap.abs().sum(-1), torch.full_like(gap[..., 0], 10 ** 6))
            prey = gap.gather(1, far.argmin(1)[:, None, None].expand(-1, 1, 2)).squeeze(1)       # (B, 2)
            along_x = prey[:, 0].abs() >= prey[:, 1].abs()
            chase_dir = torch.where(along_x, torch.where(prey[:, 0] > 0, 1, 2), torch.where(prey[:, 1] > 0, 3, 4))
            chase = (self._rand(b) < c.hazard_chase) & self.alive.any(1) & (prey.abs().sum(-1) > 0)
            hdir = torch.where(chase, chase_dir, hdir)
        htarget = self.hpos + self.moves[hdir]
        h_idx = self._idx(htarget.clamp(0, g - 1))
        h_ok = (self._rand(b) < c.hazard_move_prob) & self._inside(htarget) & ~self.walls.gather(1, h_idx[:, None]).squeeze(1) \
            & (self.rubble.gather(1, h_idx[:, None]).squeeze(1) == 0)
        self.hpos = torch.where(h_ok[:, None], htarget, self.hpos)
        burned = self.alive & self.hazard_on[:, None] & ((self.pos - self.hpos[:, None]).abs().max(-1).values <= 1)
        self.alive &= ~burned
        self.ep_lost += burned.sum(1).float()
        reward -= c.penalty_lost * burned.sum(1).float()

        # 5) обрушение: случайная открытая дверь заваливается (сложность 4)
        k = self._randint(len(self.door_idx), b)
        cell = self.door_idx[k]
        occupied = (self._idx(self.pos) == cell[:, None]).any(1) | ((self._idx(self.vpos) == cell[:, None]) & self.present).any(1)
        collapse = (self.difficulty >= 4) & (self._rand(b) < c.collapse_prob) & ~self.walls.gather(1, cell[:, None]).squeeze(1) \
            & (self.rubble.gather(1, cell[:, None]).squeeze(1) == 0) & ~occupied
        self.rubble.scatter_(1, cell[:, None], torch.where(collapse, 2, self.rubble.gather(1, cell[:, None]).squeeze(1))[:, None])
        # после обрушения расстояния могли вырасти — пересчитываем волну для этих миров с нуля
        self.dist = torch.where(collapse[:, None], torch.full_like(self.dist, INF), self.dist)
        self.dist = self._relax(self.dist, ~self.walls & (self.rubble == 0), sweeps=2)

        # 6) находки и исследование
        close = ((self.pos[:, :, None] - self.vpos[:, None]).abs().max(-1).values <= c.view) & self.alive[..., None]
        new_seen = close.any(1) & self.present & ~self.seen
        self.seen |= new_seen
        reward += c.reward_discover * new_seen.sum(1).float()
        on_now = ((self.pos[:, :, None] == self.vpos[:, None]).all(-1) & self.alive[..., None]).any(1) & self.present
        reward += c.reward_contact * (on_now & ~self.touched).sum(1).float()
        self.touched |= on_now
        reward += c.reward_explore * self._mark_seen()

        self.last = {"holding": holding.any(-1), "waiting": waiting.any(-1), "pushing": pushing,
                     "carriers": carriers, "burned": burned, "cleared": cleared.sum(1), "collapse": collapse}

        self.t += 1
        done = (self.t >= c.max_steps) | ~self.present.any(1) | ~self.alive.any(1)
        n_active = self.active.sum(1).clamp_min(1).float()
        info = {
            "rescued_frac": self.ep_rescued / c.n_victims,
            "all_rescued": (self.ep_rescued >= c.n_victims).float(),
            "length": self.t.float(),
            "drones_lost": self.ep_lost / n_active,
            "found_frac": (self.seen | ~self.present).float().mean(1),
            "difficulty": self.difficulty.float(),
        }
        self._since_refresh += 1
        if self._since_refresh >= c.refresh_every:
            self._refresh_pool()
        self._reset_where(done)
        return self.observe(), reward, done, info

    def observe(self) -> Dict[str, torch.Tensor]:
        c, b, n, g, dev = self.cfg, self.cfg.n_envs, self.cfg.n_drones, self.G, self.device
        # мировые каналы (B, C, G²)
        victims = torch.zeros(b, self.cells + 1, dtype=torch.long, device=dev)
        victims.scatter_reduce_(1, torch.where(self.present, self._idx(self.vpos), self.cells),
                                self.vweight, reduce="amax")
        drones = torch.zeros(b, self.cells + 1, dtype=torch.long, device=dev)
        drones.scatter_add_(1, torch.where(self.alive, self._idx(self.pos), self.cells), torch.ones_like(self.pos[..., 0]))
        hazard = torch.zeros(b, self.cells + 1, dtype=torch.long, device=dev)
        hazard.scatter_(1, torch.where(self.hazard_on, self._idx(self.hpos), self.cells)[:, None], 1)
        holders = torch.zeros(b, self.cells + 1, dtype=torch.long, device=dev)
        held = self.last.get("holding")
        if held is not None:
            holders.scatter_add_(1, torch.where(held, self._idx(self.pos), self.cells), torch.ones_like(self.pos[..., 0]))
        world = torch.stack([
            self.walls.long(), self.rubble, victims[:, :-1], hazard[:, :-1], drones[:, :-1],
            torch.zeros_like(self.rubble),  # место для «исследовано» — оно своё у каждого дрона
            self.exit_mask.long().expand(b, -1), holders[:, :-1],
            self.dist.clamp_max(250).long(),  # путь к выходу; видно только на исследованных клетках (ниже)
        ], 1)                                                                   # (B, C, G²)

        win = torch.stack([self.pos[..., 0:1] + self.win_dx, self.pos[..., 1:2] + self.win_dy], -1)  # (B,N,W,2)
        inside = self._inside(win)
        widx = self._idx(win.clamp(0, g - 1))
        w = world[:, None].expand(-1, n, -1, -1).gather(3, widx[:, :, None].expand(-1, -1, WINDOW_CHANNELS, -1))
        seen = (self.team_seen[:, None].expand(-1, n, -1) if c.share_map else self.own_seen)
        seen_win = seen.gather(2, widx)
        w[:, :, 5] = seen_win.long()
        # «хлебные крошки»: рой помнит дорогу назад только там, где уже был
        w[:, :, 8] = torch.where(seen_win, w[:, :, 8], torch.full_like(w[:, :, 8], ROUTE_UNKNOWN))
        if not c.route_hint:
            w[:, :, 8] = ROUTE_UNKNOWN
        if c.hidden_weight:
            w[:, :, 2] = w[:, :, 2].clamp_max(1)          # человек виден, но не сколько он весит
        w = torch.where(inside[:, :, None], w, torch.zeros_like(w))
        w[:, :, 0] = torch.where(inside, w[:, :, 0], torch.ones_like(w[:, :, 0]))   # за краем — стена
        w[:, :, 8] = torch.where(inside, w[:, :, 8], torch.full_like(w[:, :, 8], ROUTE_UNKNOWN))
        window = (w * self.alive[..., None, None]).to(torch.uint8)

        # грубая карта: сколько клеток блока исследовано и сколько из них стены
        q = g // c.coarse
        known_walls = seen & self.walls[:, None]
        blocks = lambda m: m.view(b, n, c.coarse, q, c.coarse, q).sum((3, 5)).flatten(2)
        coarse = torch.stack([blocks(seen), blocks(known_walls)], 2).to(torch.uint8)     # (B,N,2,coarse²)

        on_victim = ((self.pos[:, :, None] == self.vpos[:, None]).all(-1) & self.present[:, None])
        here_w = (on_victim * self.vweight[:, None]).max(-1).values.float()
        to_exit = (self.exit_center - self.pos).float() / g
        holding = self.last.get("holding", torch.zeros_like(self.alive)).float()
        own = torch.cat([
            self.pos.float() / g * 2 - 1, to_exit,
            (self.t.float() / c.max_steps)[:, None, None].expand(b, n, 1),
            (self.alive.sum(1, keepdim=True).float() / c.n_drones)[..., None].expand(b, n, 1),
            here_w[..., None] / 3, holding[..., None],
            (self.difficulty.float() / 4)[:, None, None].expand(b, n, 1),
        ], -1) * self.alive[..., None]
        return {"window": window, "coarse": coarse, "self": own, "active": self.alive.clone(),
                "pos": self.pos.float() / (g - 1)}

    def belief_target(self) -> torch.Tensor:
        """Где сейчас ждут спасения люди на грубой карте (B, coarse²) — правда для обучения убеждений CIDA."""
        c = self.cfg
        q = self.G // c.coarse
        idx = torch.where(self.present, (self.vpos[..., 0] // q) * c.coarse + self.vpos[..., 1] // q, c.coarse ** 2)
        out = torch.zeros(c.n_envs, c.coarse ** 2 + 1, dtype=torch.bool, device=self.device)
        return out.scatter_(1, idx, True)[:, :-1]

    def state(self) -> torch.Tensor:
        c, g = self.cfg, self.G
        v = torch.cat([self.vpos.float() / g, self.vweight[..., None].float() / 3, self.present[..., None].float(),
                       (self.dist.gather(1, self._idx(self.vpos)) / 100).clamp_max(1)[..., None]], -1)
        d = torch.cat([self.pos.float() / g, self.alive[..., None].float()], -1)
        h = torch.cat([self.hpos.float() / g, self.hazard_on[:, None].float()], -1)
        return torch.cat([
            self.walls.float(), self.rubble.float() / 3, self.team_seen.float(),
            (v * self.present[..., None]).flatten(1), d.flatten(1), h,
            (self.t.float() / c.max_steps)[:, None], (self.difficulty.float() / 4)[:, None],
        ], 1)

    @staticmethod
    def state_dim(cfg: RescueConfig) -> int:
        return 3 * cfg.grid ** 2 + 5 * cfg.n_victims + 3 * cfg.n_drones + 3 + 2
