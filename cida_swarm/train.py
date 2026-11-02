"""
train.py — обучение роя через MAPPO (общая политика дронов + централизованный критик).

Шаг 1: базовая линия без связи. Пример (Colab A100):
    python -m cida_swarm.train --out runs/nocomm_s0
    python -m cida_swarm.train --share-coverage --out runs/stigmergy_s0

Качество меряется на свежих мирах и сравнивается со случайной политикой.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import N_ACTIONS, EnvConfig, SearchEnv
from .policy import Actor, Critic


def _stack(obs_list):
    return {k: torch.stack([o[k] for o in obs_list]) for k in obs_list[0]}


# ── оценка ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(actor: Optional[Actor], env_cfg: EnvConfig, n_envs: int, device, seed: int = 12345) -> Dict[str, float]:
    """Один эпизод в каждом из n_envs свежих миров. actor=None — случайная политика."""
    cfg = EnvConfig(**{**env_cfg.to_dict(), "n_envs": n_envs})
    env = SearchEnv(cfg, device, seed=seed)
    obs = env.reset()
    finished = torch.zeros(n_envs, dtype=torch.bool, device=env.device)
    stats = {"found_frac": torch.zeros(n_envs, device=env.device),
             "length": torch.zeros(n_envs, device=env.device),
             "coverage": torch.zeros(n_envs, device=env.device)}
    while not bool(finished.all()):
        if actor is None:
            actions = torch.randint(0, N_ACTIONS, (n_envs, cfg.n_drones), device=env.device)
        else:
            actions = Categorical(logits=actor(obs)).sample()
        obs, _, done, info = env.step(actions)
        first = done & ~finished
        if bool(first.any()):
            sel = first[done]  # info содержит только завершившиеся миры, в порядке индексов
            for k in stats:
                stats[k][first] = info[k][sel]
            finished |= done
    return {
        "found_frac": float(stats["found_frac"].mean()),
        "all_found": float((stats["found_frac"] >= 1.0).float().mean()),
        "length": float(stats["length"].mean()),
        "coverage": float(stats["coverage"].mean()),
    }


# ── обучение ─────────────────────────────────────────────────────────────────
def build_env_config(args) -> EnvConfig:
    return EnvConfig(
        n_envs=args.n_envs, n_drones=args.n_drones, n_targets=args.n_targets, max_steps=args.max_steps,
        view_radius=args.view_radius, max_speed=args.max_speed, share_coverage=args.share_coverage,
    )


def train(args) -> Dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu" and args.device is None:
        print(f"[cida-swarm] WARNING: CUDA недоступна в {sys.executable} — обучение пойдёт на CPU.", flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = build_env_config(args)
    env = SearchEnv(env_cfg, device, seed=args.seed)
    actor = Actor(env_cfg, args.hidden).to(device)
    critic = Critic(env_cfg).to(device)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"[cida-swarm] actor params={n_params:,} device={device} worlds={env_cfg.n_envs} "
          f"drones={env_cfg.n_drones} targets={env_cfg.n_targets} share_coverage={env_cfg.share_coverage}",
          flush=True)

    obs = env.reset()
    state = env.state()
    history, ep_log = [], {"found_frac": [], "length": [], "coverage": []}
    t0 = time.time()
    agent_steps = 0
    T, B = args.rollout, env_cfg.n_envs

    for it in range(1, args.iters + 1):
        # 1) сбор опыта
        buf_obs, buf_state, buf_act, buf_logp, buf_val, buf_rew, buf_done = [], [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(T):
                dist = Categorical(logits=actor(obs))
                actions = dist.sample()
                buf_obs.append(obs)
                buf_state.append(state)
                buf_act.append(actions)
                buf_logp.append(dist.log_prob(actions))
                buf_val.append(critic(state))
                obs, reward, done, info = env.step(actions)
                state = env.state()
                buf_rew.append(reward)
                buf_done.append(done.float())
                for k in ep_log:
                    ep_log[k].append(info[k])
            last_val = critic(state)

            # 2) GAE по общей награде команды
            values = torch.stack(buf_val)
            rewards, dones = torch.stack(buf_rew), torch.stack(buf_done)
            adv = torch.zeros_like(rewards)
            gae = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                next_val = last_val if t == T - 1 else values[t + 1]
                nonterminal = 1.0 - dones[t]
                delta = rewards[t] + args.gamma * next_val * nonterminal - values[t]
                gae = delta + args.gamma * args.lam * nonterminal * gae
                adv[t] = gae
            returns = adv + values
        agent_steps += T * B * env_cfg.n_drones

        # 3) PPO-обновление; пример = (мир, момент времени) со всеми его дронами
        flat_obs = {k: v.flatten(0, 1) for k, v in _stack(buf_obs).items()}
        flat_state = torch.stack(buf_state).flatten(0, 1)
        flat_act = torch.stack(buf_act).flatten(0, 1)
        flat_logp = torch.stack(buf_logp).flatten(0, 1)
        flat_adv, flat_ret = adv.flatten(), returns.flatten()
        n = flat_adv.numel()
        mb = n // args.minibatches
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for i in range(args.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                a = flat_adv[idx]
                a = ((a - a.mean()) / (a.std() + 1e-8))[:, None]
                dist = Categorical(logits=actor({k: v[idx] for k, v in flat_obs.items()}))
                ratio = (dist.log_prob(flat_act[idx]) - flat_logp[idx]).exp()
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                v_loss = F.mse_loss(critic(flat_state[idx]), flat_ret[idx])
                entropy = dist.entropy().mean()
                loss = pg + args.vf_coef * v_loss - args.ent_coef * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()

        if it % args.log_every == 0:
            done_eps = {k: torch.cat(v) for k, v in ep_log.items()}
            n_eps = done_eps["found_frac"].numel()
            msg = " | ".join(f"{k} {float(v.mean()):.3f}" for k, v in done_eps.items()) if n_eps else "no episodes yet"
            print(f"it {it:5d} | {msg} | entropy {entropy.item():.3f} | "
                  f"{agent_steps / (time.time() - t0):,.0f} agent-steps/s", flush=True)
            ep_log = {k: [] for k in ep_log}

        if it % args.eval_every == 0 or it == args.iters:
            metrics = evaluate(actor, env_cfg, args.eval_envs, device)
            metrics.update(iter=it, agent_steps=agent_steps, minutes=(time.time() - t0) / 60)
            history.append(metrics)
            print(f"[eval] it {it} | found {metrics['found_frac']:.3f} | all found {metrics['all_found']:.3f} "
                  f"| length {metrics['length']:.1f} | coverage {metrics['coverage']:.3f}", flush=True)
            torch.save({"env_config": env_cfg.to_dict(), "hidden": args.hidden, "actor": actor.state_dict()},
                       out_dir / "model.pt")

    random_metrics = evaluate(None, env_cfg, args.eval_envs, device)
    print(f"[eval] random policy | found {random_metrics['found_frac']:.3f} "
          f"| all found {random_metrics['all_found']:.3f} | length {random_metrics['length']:.1f}", flush=True)
    result = {
        "name": args.name or out_dir.name,
        "env_config": env_cfg.to_dict(),
        "args": vars(args),
        "params": n_params,
        "train_minutes": (time.time() - t0) / 60,
        "final": history[-1],
        "random": random_metrics,
        "history": history,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description="CIDA-Swarm step 1: MAPPO search swarm without communication")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="runs/swarm")
    # среда
    ap.add_argument("--n-envs", type=int, default=1024)
    ap.add_argument("--n-drones", type=int, default=6)
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--view-radius", type=float, default=0.08)
    ap.add_argument("--max-speed", type=float, default=0.015)
    ap.add_argument("--share-coverage", action="store_true", help="стигмергия: общая карта исследованного")
    # модель и PPO
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    # служебное
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-envs", type=int, default=512)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
