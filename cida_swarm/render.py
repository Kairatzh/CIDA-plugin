"""
render.py — анимация роя: один или несколько обученных мозгов на одной и той же сцене.

    python -m cida_swarm.render runs/cida_s0/model.pt --out swarm.gif
    python -m cida_swarm.render runs/nocomm_s0/model.pt runs/cida_s0/model.pt \
        --labels "обычный рой" "CIDA-рой" --out compare.gif

На кадре: здания (серые), исследованные клетки (голубые), люди (зелёные; найденные — серые крестики),
приманки (оранжевые ромбы), дроны с радиусом обзора и следом. У CIDA-роя — линии связи (толщина = вес
сообщения) и полупрозрачная тепловая карта общего убеждения роя «где люди».
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.distributions import Categorical

from .env import EnvConfig, SearchEnv
from .policy import Actor


def load_actor(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    env_cfg = EnvConfig(**ckpt["env_config"])
    kwargs = ckpt.get("actor_kwargs", {"hidden": ckpt.get("hidden", 256)})
    actor = Actor(env_cfg, **kwargs).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, env_cfg


@torch.no_grad()
def record(actor: Actor, env_cfg: EnvConfig, seed: int, n_drones: Optional[int] = None,
           greedy: bool = False) -> List[Dict]:
    """Проигрывает один эпизод и возвращает кадры (numpy) до его конца."""
    cfg = EnvConfig(**{**env_cfg.to_dict(), "n_envs": 1})
    env = SearchEnv(cfg, "cpu", seed=seed)
    obs = env.reset()
    if n_drones is not None:
        env.active[:] = torch.arange(cfg.n_drones) < n_drones
        obs = env.observe()
    g, coarse = cfg.grid, actor.map_enc.coarse
    frames = []
    for _ in range(cfg.max_steps):
        logits, aux = actor(obs, return_aux=True)
        frame = {
            "pos": env.pos[0].numpy().copy(),
            "active": env.active[0].numpy().copy(),
            "targets": env.targets[0].numpy().copy(),
            "alive": env.alive[0].numpy().copy(),
            "decoys": env.decoys[0].numpy().copy(),
            "obstacles": env.obstacles[0].view(g, g).numpy().copy(),
            "covered": env.team_cov[0, :-1].view(g, g).numpy().copy(),
            "t": int(env.t[0]),
            "found": int((~env.alive[0]).sum()),
        }
        if aux is not None:
            act = env.active[0]
            belief = torch.sigmoid(aux["beliefs"][-1][0][act]).mean(0)       # общее убеждение активных дронов
            frame["belief"] = belief.view(coarse, coarse).numpy()
            frame["attn"] = aux["attn"][0].numpy()
        frames.append(frame)
        actions = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
        obs, _, done, _ = env.step(actions)
        if bool(done[0]):
            break
    return frames


def _draw(ax, frame: Dict, trail: List[np.ndarray], view_radius: float, title: str, n_targets: int):
    from matplotlib.patches import Circle

    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(frame["covered"].T, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=4, alpha=0.6)
    if "belief" in frame:
        ax.imshow(frame["belief"].T, origin="lower", extent=(0, 1, 0, 1), cmap="Reds", vmin=0, vmax=1, alpha=0.35)
    obst = np.ma.masked_where(~frame["obstacles"].T, frame["obstacles"].T)
    ax.imshow(obst, origin="lower", extent=(0, 1, 0, 1), cmap="Greys", vmin=0, vmax=1.3)

    act = frame["active"]
    pos = frame["pos"]
    if "attn" in frame:
        for i in np.where(act)[0]:
            for j in np.where(act)[0]:
                w = frame["attn"][i, j]
                if w > 0.15:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="purple", lw=0.5 + 2.5 * w, alpha=0.6)
    for path in trail:
        ax.plot(path[:, 0], path[:, 1], color="tab:blue", lw=0.6, alpha=0.4)
    alive, t = frame["alive"], frame["targets"]
    ax.scatter(t[alive, 0], t[alive, 1], s=40, c="tab:green", marker="o", zorder=3)
    ax.scatter(t[~alive, 0], t[~alive, 1], s=40, c="grey", marker="x", zorder=3)
    ax.scatter(frame["decoys"][:, 0], frame["decoys"][:, 1], s=35, c="tab:orange", marker="D", zorder=3)
    for p in pos[act]:
        ax.add_patch(Circle(p, view_radius, color="tab:blue", alpha=0.08))
    ax.scatter(pos[act, 0], pos[act, 1], s=30, c="tab:blue", edgecolors="white", zorder=4)
    ax.set_title(f"{title}\nшаг {frame['t']} · найдено {frame['found']}/{n_targets}", fontsize=10)


def render(paths: List[str], labels: List[str], out: str, seed: int = 0, n_drones: Optional[int] = None,
           fps: int = 15, greedy: bool = False) -> List[Dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    runs = []
    for path in paths:
        actor, env_cfg = load_actor(path)
        runs.append((record(actor, env_cfg, seed, n_drones, greedy), env_cfg))
    n_frames = max(len(f) for f, _ in runs)
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5.4))
    axes = np.atleast_1d(axes)

    def update(k):
        for ax, (frames, cfg), label in zip(axes, runs, labels):
            i = min(k, len(frames) - 1)
            trail = [np.stack([f["pos"][d] for f in frames[max(0, i - 40):i + 1]])
                     for d in np.where(frames[i]["active"])[0]]
            _draw(ax, frames[i], trail, cfg.view_radius, label, cfg.n_targets)
        return []

    FuncAnimation(fig, update, frames=n_frames).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    summary = [{"label": lab, "steps": len(f), "found": f[-1]["found"]} for (f, _), lab in zip(runs, labels)]
    for s in summary:
        print(f"[render] {s['label']}: найдено {s['found']} за {s['steps']} шагов", flush=True)
    print(f"[render] сохранено: {out}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="CIDA-Swarm animation")
    ap.add_argument("models", nargs="+", help="пути к model.pt")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", default="swarm.gif")
    ap.add_argument("--seed", type=int, default=0, help="номер сцены; одинаковый для всех моделей")
    ap.add_argument("--n-drones", type=int, default=None, help="сколько дронов активно (по умолчанию — как выпадет)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--greedy", action="store_true", help="всегда самое вероятное действие")
    args = ap.parse_args()
    labels = args.labels or [p.replace("\\", "/").split("/")[-2] for p in args.models]
    render(args.models, labels, args.out, args.seed, args.n_drones, args.fps, args.greedy)


if __name__ == "__main__":
    main()
