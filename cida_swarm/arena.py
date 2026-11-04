"""
arena.py — «Спасатели против охотников»: в здании работают две обучаемые команды.

Спасатели (рой CIDA) выносят людей, как в rescue.RescueEnv. Вместо запрограммированного пожара по зданию
ходит команда охотников — у них свой мозг, своя память, свои личности и своё радио (радио спасателей они не слышат).

Правила охоты:
    * охотник выводит из строя спасателя, оказавшегося вплотную (по стороне клетки), если охотник не оглушён;
    * если вплотную к охотнику сразу stun_need спасателей — охотник оглушён на stun_steps шагов
      (стоит и никого не ловит): так у спасателей есть ответная тактика — действовать вместе;
    * во входную комнату (безопасную зону) охотники не заходят, там спасателей не ловят;
    * охотники проходят только по коридорам: стены и завалы для них тоже препятствие.

Награда охотников: + за пойманного спасателя, − за каждого вынесенного человека (столько же, сколько получают
спасатели), − за оглушение, и немного — за приближение к ближайшему спасателю. Сумма почти антагонистическая:
что хорошо одной команде, плохо другой — так и рождаются тактики (засады у дверей, приманки, эскорт носильщиков).

Наблюдение охотника — как у спасателя (окно, грубая карта, «о себе», роль), но здание он знает целиком.
Спасатели видят охотников в канале «опасность».
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .rescue import N_DIRS, ROLE_DIM, RescueConfig, RescueEnv


class ArenaEnv(RescueEnv):
    STATS = RescueEnv.STATS + ("stuns",)
    BANK_KEYS = RescueEnv.BANK_KEYS + ("qpos", "q_active")

    def __init__(self, cfg: RescueConfig, device: torch.device | str = "cpu", seed: Optional[int] = None):
        if cfg.n_hunters < 1:
            raise ValueError("в арене нужен хотя бы один охотник (n_hunters >= 1)")
        super().__init__(cfg, device, seed)
        b, h, dev = cfg.n_envs, cfg.n_hunters, self.device
        self.qpos = torch.zeros(b, h, 2, dtype=torch.long, device=dev)
        self.q_active = torch.zeros(b, h, dtype=torch.bool, device=dev)
        self.q_stun = torch.zeros(b, h, dtype=torch.long, device=dev)
        self.q_role = torch.zeros(b, h, ROLE_DIM, device=dev)
        self.ep_stuns = torch.zeros(b, device=dev)
        self.hunter_reward = torch.zeros(b, device=dev)
        self.hunter_actions: Optional[torch.Tensor] = None

    # ── пул и перезапуск ──────────────────────────────────────────────────────
    def _extra_fresh(self, fresh: Dict[str, torch.Tensor]) -> None:
        c, b, r = self.cfg, self.cfg.n_envs, self.cfg.rooms
        n_q = 1 + self._randint(c.n_hunters, b)
        fresh["q_active"] = torch.arange(c.n_hunters, device=self.device) < n_q[:, None]
        fresh["qpos"] = self._room_cell(1 + self._randint(r * r - 1, b, c.n_hunters))
        fresh["q_role"] = self._rand(b, c.n_hunters, ROLE_DIM) * 2 - 1
        fresh["hazard_on"] = torch.zeros_like(fresh["hazard_on"])   # вместо пожара — охотники

    def _reset_where(self, mask: torch.Tensor) -> None:
        super()._reset_where(mask)
        f, m1, m2 = self._fresh, mask[:, None], mask[:, None, None]
        self.qpos = torch.where(m2, f["qpos"], self.qpos)
        self.q_role = torch.where(m2, f["q_role"], self.q_role)
        self.q_active = torch.where(m1, f["q_active"], self.q_active)
        self.q_stun = torch.where(m1, torch.zeros_like(self.q_stun), self.q_stun)
        self.ep_stuns = torch.where(mask, torch.zeros_like(self.ep_stuns), self.ep_stuns)

    # ── шаг ──────────────────────────────────────────────────────────────────
    def step(self, actions: torch.Tensor, hunter_actions: Optional[torch.Tensor] = None):
        """actions (B, N) — спасатели; hunter_actions (B, H) — охотники (направление = a % 5; None — стоят)."""
        self.hunter_actions = hunter_actions
        return super().step(actions)

    def _nearest_gap(self) -> torch.Tensor:
        """Расстояние (по сторонам клеток) от каждого охотника до ближайшего живого спасателя вне входа (B, H)."""
        prey = self.alive & ~self.exit_mask[self._idx(self.pos)]
        d = (self.pos[:, None] - self.qpos[:, :, None]).abs().sum(-1)                   # (B, H, N)
        return torch.where(prey[:, None], d, torch.full_like(d, 4 * self.G)).min(-1).values

    def _hazard(self, reward: torch.Tensor) -> torch.Tensor:
        c, g = self.cfg, self.G
        gap_before = self._nearest_gap()
        # 1) охотники ходят (оглушённые стоят), только по коридорам и не во вход
        a = self.hunter_actions % N_DIRS if self.hunter_actions is not None else torch.zeros_like(self.q_stun)
        target = self.qpos + self.moves[a]
        t_idx = self._idx(target.clamp(0, g - 1))
        ok = self._inside(target) & ~self.walls.gather(1, t_idx) & (self.rubble.gather(1, t_idx) == 0) \
            & ~self.exit_mask[t_idx] & self.q_active & (self.q_stun == 0)
        self.qpos = torch.where(ok[..., None], target, self.qpos)

        # 2) оглушение: вплотную к охотнику сразу stun_need спасателей
        d = (self.pos[:, None] - self.qpos[:, :, None]).abs().sum(-1)                   # (B, H, N)
        touch = (d <= 1) & self.alive[:, None]
        stunned_now = self.q_active & (self.q_stun == 0) & (touch.sum(-1) >= c.stun_need)
        self.q_stun = torch.where(stunned_now, torch.full_like(self.q_stun, c.stun_steps), (self.q_stun - 1).clamp_min(0))
        n_stun = stunned_now.sum(1).float()
        reward += c.reward_stun * n_stun
        self.ep_stuns += n_stun

        # 3) поимка: не оглушённый охотник вплотную, спасатель не во входе
        hunting = self.q_active & (self.q_stun == 0)
        caught = self.alive & ~self.exit_mask[self._idx(self.pos)] & (touch & hunting[..., None]).any(1)
        self.alive &= ~caught
        n_caught = caught.sum(1).float()
        self.ep_lost += n_caught
        reward -= c.penalty_lost * n_caught

        # награда охотников: противоположна спасательной
        delivered = (self._delivered.float() * (c.reward_deliver + 0.5 * (self.vweight - 1).float())).sum(1)
        approach = ((gap_before - self._nearest_gap()).clamp(-1, 1).float() * hunting).sum(1)
        self.hunter_reward = c.reward_catch * n_caught - delivered - c.reward_stun * n_stun + c.hunter_approach * approach
        self.last_hunt = {"caught": caught, "stunned": self.q_stun > 0}
        return caught

    def _extra_info(self) -> Dict[str, torch.Tensor]:
        return {"stuns": self.ep_stuns.clone()}

    def _hazard_map(self) -> torch.Tensor:
        hazard = torch.zeros(self.cfg.n_envs, self.cells + 1, dtype=torch.long, device=self.device)
        idx = torch.where(self.q_active, self._idx(self.qpos), self.cells)
        return hazard.scatter_add_(1, idx, torch.ones_like(idx))

    # ── наблюдения и состояние ─────────────────────────────────────────────────
    def observe_hunters(self) -> Dict[str, torch.Tensor]:
        """Наблюдение охотников — те же ключи, что у спасателей: охотник управляется тем же устройством мозга."""
        c, b, h, g = self.cfg, self.cfg.n_envs, self.cfg.n_hunters, self.G
        everything = torch.ones(b, h, self.cells, dtype=torch.bool, device=self.device)   # здание им знакомо
        window = self._window(self._world(), self.qpos, self.q_active, everything)
        on_victim = (self.qpos[:, :, None] == self.vpos[:, None]).all(-1) & self.present[:, None]
        here_w = (on_victim * self.vweight[:, None]).max(-1).values.float()
        own = torch.cat([
            self.qpos.float() / g * 2 - 1, (self.exit_center - self.qpos).float() / g,
            (self.t.float() / c.max_steps)[:, None, None].expand(b, h, 1),
            (self.alive.sum(1, keepdim=True).float() / c.n_drones)[..., None].expand(b, h, 1),  # сколько спасателей ещё в строю
            here_w[..., None] / 3, (self.q_stun > 0).float()[..., None],
            (self.difficulty.float() / 4)[:, None, None].expand(b, h, 1),
        ], -1) * self.q_active[..., None]
        return {"window": window, "coarse": self._coarse(everything), "self": own, "role": self.q_role.clone(),
                "active": self.q_active.clone(), "pos": self.qpos.float() / (g - 1)}

    def state(self) -> torch.Tensor:
        q = torch.cat([self.qpos.float() / self.G, self.q_active[..., None].float(), (self.q_stun > 0)[..., None].float()], -1)
        return torch.cat([super().state(), q.flatten(1)], 1)

    @staticmethod
    def state_dim(cfg: RescueConfig) -> int:
        return RescueEnv.state_dim(cfg) + 4 * cfg.n_hunters

    def hunter_config(self) -> RescueConfig:
        """Конфиг для мозга охотника: «дронов» столько, сколько охотников (для личностей по номеру)."""
        return RescueConfig(**{**self.cfg.to_dict(), "n_drones": self.cfg.n_hunters, "min_drones": 1})
