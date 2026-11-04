"""
Разбор провалов спасательного роя: почему не вынесены те, кого не вынесли.

Для каждого невынесенного человека — одна причина:
  не найден         — ни один дрон его не увидел;
  найден, не дошли  — видели, но никто не встал на его клетку;
  не хватило рук    — держали, но одновременно меньше дронов, чем его вес, хотя живых хватало;
  мало живых        — к концу живых дронов меньше его веса (сгорели);
  несли, не донесли — держали достаточно, но не дотащили до выхода.

Ещё две доли от всех шагов живых дронов: «толпа» — свободный дрон жмётся к другому свободному вне входа
(ходят стадом вместо того, чтобы разойтись); «стоит» — свободный дрон не сдвинулся с места.

Для моделей со связью дополнительно строка «связь выключена» — те же миры, но дроны не слышат друг друга.
Если она не хуже — рой связью не пользуется.

    python -m cida_swarm.diagnose runs/ft_none/model_best.pt runs/ft2_cida/model_best.pt --labels "без связи" CIDA
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import torch
from torch.distributions import Categorical

from cida_swarm.demo import load_actor
from cida_swarm.policy import forget_done, init_memory, policy_step
from cida_swarm.rescue import RescueConfig, RescueEnv
from cida_swarm.train import autocast

CAUSES = ("delivered", "never_seen", "not_reached", "short_handed", "too_few_alive", "not_carried_out")
LABELS = {"delivered": "вынесен", "never_seen": "не найден", "not_reached": "найден, не дошли",
          "short_handed": "не хватило рук", "too_few_alive": "мало живых", "not_carried_out": "несли, не донесли"}


class _TracedEnv(RescueEnv):
    """Запоминает судьбу каждого человека в первом эпизоде каждого мира — до того, как мир перезапустится."""

    def __init__(self, cfg: RescueConfig, device, seed: Optional[int] = None):
        super().__init__(cfg, device, seed)
        b, m = cfg.n_envs, cfg.n_victims
        self.tracing = False
        self.max_holders = torch.zeros(b, m, dtype=torch.long, device=self.device)
        self.done_once = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.final = {k: None for k in ("present", "seen", "max_holders", "weight", "alive", "difficulty")}
        self.drone_steps = torch.zeros(b, device=self.device)
        self.crowd_steps = torch.zeros(b, device=self.device)
        self.idle_steps = torch.zeros(b, device=self.device)

    def _reset_where(self, mask: torch.Tensor) -> None:
        if self.tracing:
            live = (~self.done_once).float()
            self.drone_steps += self.alive.sum(1).float() * live
            self.crowd_steps += self.last["crowded"].sum(1).float() * live
            self.idle_steps += self.last["idle"].sum(1).float() * live
            on = (self.pos[:, :, None] == self.vpos[:, None]).all(-1) & self.alive[..., None] & self.present[:, None]
            self.max_holders = torch.maximum(self.max_holders, on.sum(1))
            first = mask & ~self.done_once
            snap = {"present": self.present, "seen": self.seen, "max_holders": self.max_holders,
                    "weight": self.vweight, "alive": self.alive.sum(1), "difficulty": self.difficulty}
            for k, v in snap.items():
                prev = self.final[k] if self.final[k] is not None else torch.zeros_like(v)
                sel = first.view(-1, *([1] * (v.dim() - 1)))
                self.final[k] = torch.where(sel, v, prev)
            self.done_once |= mask
            self.max_holders = torch.where(mask[:, None], torch.zeros_like(self.max_holders), self.max_holders)
        super()._reset_where(mask)


@torch.no_grad()
def diagnose(actor, cfg: RescueConfig, n_envs: int, device, seed: int = 12345, mute: bool = False) -> Dict:
    device = torch.device(device)
    cfg = RescueConfig(**{**cfg.to_dict(), "n_envs": n_envs})
    env = _TracedEnv(cfg, device, seed=seed)
    obs = env.reset()
    env.tracing = True
    mem = init_memory(actor, n_envs, cfg.n_drones, device)
    radius = None
    if mute and actor.comm is not None:
        radius, actor.comm.radius = actor.comm.radius, 0.0      # никто никого не слышит
    try:
        for _ in range(cfg.max_steps):
            with autocast(device):
                logits, _, mem = policy_step(actor, obs, mem)
            obs, _, done, _ = env.step(Categorical(logits=logits.float()).sample())
            mem = forget_done(mem, done)
            if bool(env.done_once.all()):
                break
    finally:
        if radius is not None:
            actor.comm.radius = radius
    f = env.final
    w, alive = f["weight"], f["alive"][:, None]
    cause = {
        "delivered": ~f["present"],
        "never_seen": f["present"] & ~f["seen"],
        "not_reached": f["present"] & f["seen"] & (f["max_holders"] == 0),
        "too_few_alive": f["present"] & f["seen"] & (f["max_holders"] > 0) & (alive < w),
        "short_handed": f["present"] & (f["max_holders"] > 0) & (f["max_holders"] < w) & (alive >= w),
        "not_carried_out": f["present"] & (f["max_holders"] >= w) & (alive >= w),
    }
    total = float(w.numel())
    out = {k: float(v.sum()) / total for k, v in cause.items()}
    out["by_weight"] = {str(k): float(cause["delivered"][w == k].float().mean())
                        for k in (1, 2, 3) if bool((w == k).any())}
    steps = env.drone_steps.sum().clamp_min(1.0)
    out["crowded"] = float(env.crowd_steps.sum() / steps)
    out["idle"] = float(env.idle_steps.sum() / steps)
    lvl = f["difficulty"][:, None].expand_as(w)
    out["by_level"] = {str(k): float(cause["delivered"][lvl == k].float().mean())
                       for k in range(1, 5) if bool((lvl == k).any())}
    return out


def table(rows: List[tuple]) -> str:
    head = ["Рой"] + [LABELS[c] for c in CAUSES] + ["вес 1", "вес 2", "вес 3", "толпа", "стоит"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, r in rows:
        cells = [f"{r[c]:.1%}" for c in CAUSES] + [f"{r['by_weight'].get(str(k), float('nan')):.1%}" for k in (1, 2, 3)]
        cells += [f"{r['crowded']:.1%}", f"{r['idle']:.1%}"]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Почему рой не вынес тех, кого не вынес")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--n-envs", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)
    labels = args.labels or args.models
    rows = []
    for path, label in zip(args.models, labels):
        actor, cfg = load_actor(path)
        actor.to(args.device)
        rows.append((label, diagnose(actor, cfg, args.n_envs, args.device, args.seed)))
        if actor.comm is not None:
            rows.append((f"{label}, связь выключена", diagnose(actor, cfg, args.n_envs, args.device, args.seed, mute=True)))
    print(table(rows))
    return rows


if __name__ == "__main__":
    main()
