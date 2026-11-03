"""
train.py — обучение роя через MAPPO (общая политика дронов + централизованный критик).

Пример (Colab A100):
    python -m cida_swarm.train --total-steps 2e9 --out runs/nocomm_s0
    python -m cida_swarm.train --total-steps 2e9 --share-coverage --out runs/stigmergy_s0

Чтобы A100 не простаивал: тысячи миров параллельно, bf16, шаг среды без синхронизаций с CPU,
статистика копится на GPU и читается только при логировании.
Подобрать число миров под свою карту: python -m cida_swarm.bench
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import N_ACTIONS, EnvConfig, SearchEnv
from .policy import Actor, Critic

EPISODE_STATS = ("found_frac", "length", "coverage")


def autocast(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


class RunningMeanStd:
    """Нормировка целей критика: награда растёт по ходу обучения, масштаб ценности должен успевать."""

    def __init__(self, device):
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = torch.tensor(1e-4, device=device)

    def update(self, x: torch.Tensor):
        b_mean, b_var, b_count = x.mean(), x.var(unbiased=False), x.numel()
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        self.var = (self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total) / total
        self.count = total

    @property
    def std(self):
        return self.var.sqrt().clamp_min(1e-4)


def _stack(obs_list):
    return {k: torch.stack([o[k] for o in obs_list]) for k in obs_list[0]}


def _sync(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time()


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


# ── оценка ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(actor: Optional[Actor], env_cfg: EnvConfig, n_envs: int, device, seed: int = 12345) -> Dict[str, float]:
    """Один эпизод в каждом из n_envs свежих миров. actor=None — случайная политика."""
    device = torch.device(device)
    cfg = EnvConfig(**{**env_cfg.to_dict(), "n_envs": n_envs})
    env = SearchEnv(cfg, device, seed=seed)
    obs = env.reset()
    finished = torch.zeros(n_envs, dtype=torch.bool, device=device)
    stats = {k: torch.zeros(n_envs, device=device) for k in EPISODE_STATS}
    for _ in range(cfg.max_steps):
        if actor is None:
            actions = torch.randint(0, N_ACTIONS, (n_envs, cfg.n_drones), device=device)
        else:
            with autocast(device):
                logits = actor(obs)
            actions = Categorical(logits=logits.float()).sample()
        obs, _, done, info = env.step(actions)
        first = done & ~finished
        for k in stats:
            stats[k] = torch.where(first, info[k], stats[k])
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
        n_envs=args.n_envs, n_drones=args.n_drones, min_drones=args.min_drones, n_targets=args.n_targets,
        n_decoys=args.n_decoys, n_obstacles=args.n_obstacles, grid=args.grid, max_steps=args.max_steps,
        view_radius=args.view_radius, max_speed=args.max_speed, target_speed=args.target_speed,
        share_coverage=args.share_coverage,
    )


def train(args) -> Dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu" and args.device is None:
        print(f"[cida-swarm] WARNING: CUDA недоступна в {sys.executable} — обучение пойдёт на CPU.", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = build_env_config(args)
    env = SearchEnv(env_cfg, device, seed=args.seed)
    actor = Actor(env_cfg, args.hidden).to(device)
    critic = Critic(env_cfg).to(device)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    run_actor, run_critic = actor, critic
    if args.compile:
        print("[cida-swarm] torch.compile: первые итерации займут пару минут на компиляцию", flush=True)
        run_actor, run_critic = torch.compile(actor), torch.compile(critic)
    ret_norm = RunningMeanStd(device)

    T, B, N = args.rollout, env_cfg.n_envs, env_cfg.n_drones
    steps_per_iter = T * B * N
    iters = math.ceil(args.total_steps / steps_per_iter) if args.total_steps else args.iters
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"[cida-swarm] actor params={n_params:,} device={device} worlds={B} drones={env_cfg.min_drones}-{N} "
          f"targets={env_cfg.n_targets} decoys={env_cfg.n_decoys} obstacles={env_cfg.n_obstacles} "
          f"share_coverage={env_cfg.share_coverage} iters={iters} (~{iters * steps_per_iter / 1e9:.2f}B agent-steps)",
          flush=True)

    obs = env.reset()
    state = env.state()
    history = []
    ep_sum = {k: torch.zeros((), device=device) for k in EPISODE_STATS}
    ep_count = torch.zeros((), device=device)
    t0 = time.time()
    eval_seconds = 0.0
    timing = {"collect": 0.0, "update": 0.0}
    agent_steps = 0

    for it in range(1, iters + 1):
        lr = args.lr * (1.0 - (it - 1) / iters)
        for g in opt.param_groups:
            g["lr"] = lr

        # 1) сбор опыта
        t_start = _sync(device)
        buf_obs, buf_state, buf_act, buf_logp, buf_val, buf_rew, buf_done = [], [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(T):
                with autocast(device):
                    logits, value = run_actor(obs), run_critic(state)
                dist = Categorical(logits=logits.float())
                actions = dist.sample()
                buf_obs.append(obs)
                buf_state.append(state)
                buf_act.append(actions)
                buf_logp.append(dist.log_prob(actions))
                buf_val.append(value.float() * ret_norm.std + ret_norm.mean)
                obs, reward, done, info = env.step(actions)
                state = env.state()
                buf_rew.append(reward)
                buf_done.append(done.float())
                for k in EPISODE_STATS:
                    ep_sum[k] += (info[k] * done).sum()
                ep_count += done.sum()
            with autocast(device):
                last_val = run_critic(state).float() * ret_norm.std + ret_norm.mean

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
            ret_norm.update(returns)
        agent_steps += steps_per_iter
        t_mid = _sync(device)
        timing["collect"] += t_mid - t_start

        # 3) PPO; пример = (мир, момент времени) со всеми его дронами, неактивные дроны маскируются
        flat_obs = {k: v.flatten(0, 1) for k, v in _stack(buf_obs).items()}
        flat_state = torch.stack(buf_state).flatten(0, 1)
        flat_act = torch.stack(buf_act).flatten(0, 1)
        flat_logp = torch.stack(buf_logp).flatten(0, 1)
        flat_adv = adv.flatten()
        flat_ret = ((returns - ret_norm.mean) / ret_norm.std).flatten()
        n = flat_adv.numel()
        mb = min(args.minibatch_size, n)
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n - mb + 1, mb):
                idx = perm[start:start + mb]
                o = {k: v[idx] for k, v in flat_obs.items()}
                mask = o["active"].float()
                a = flat_adv[idx]
                a = ((a - a.mean()) / (a.std() + 1e-8))[:, None]
                with autocast(device):
                    logits, v = run_actor(o), run_critic(flat_state[idx])
                dist = Categorical(logits=logits.float())
                ratio = (dist.log_prob(flat_act[idx]) - flat_logp[idx]).exp()
                pg = -_masked_mean(torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a), mask)
                entropy = _masked_mean(dist.entropy(), mask)
                v_loss = F.mse_loss(v.float(), flat_ret[idx])
                loss = pg + args.vf_coef * v_loss - args.ent_coef * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
        timing["update"] += _sync(device) - t_mid

        if it % args.log_every == 0:
            cnt = ep_count.item()
            msg = (" | ".join(f"{k} {ep_sum[k].item() / cnt:.3f}" for k in EPISODE_STATS)
                   if cnt else "no episodes yet")
            print(f"it {it:5d}/{iters} | {msg} | entropy {entropy.item():.3f} | v_loss {v_loss.item():.3f} | "
                  f"{agent_steps / (time.time() - t0 - eval_seconds):,.0f} agent-steps/s "
                  f"(collect {timing['collect'] / max(sum(timing.values()), 1e-9):.0%})", flush=True)
            ep_sum = {k: torch.zeros((), device=device) for k in EPISODE_STATS}
            ep_count.zero_()

        if it % args.eval_every == 0 or it == iters:
            t_eval = time.time()
            metrics = evaluate(actor, env_cfg, args.eval_envs, device)
            metrics.update(iter=it, agent_steps=agent_steps, minutes=(time.time() - t0) / 60)
            history.append(metrics)
            print(f"[eval] it {it} | found {metrics['found_frac']:.3f} | all found {metrics['all_found']:.3f} "
                  f"| length {metrics['length']:.1f} | coverage {metrics['coverage']:.3f}", flush=True)
            torch.save({"env_config": env_cfg.to_dict(), "hidden": args.hidden, "actor": actor.state_dict()},
                       out_dir / "model.pt")
            eval_seconds += time.time() - t_eval

    train_seconds = time.time() - t0 - eval_seconds
    random_metrics = evaluate(None, env_cfg, args.eval_envs, device)
    print(f"[eval] random policy | found {random_metrics['found_frac']:.3f} "
          f"| all found {random_metrics['all_found']:.3f} | length {random_metrics['length']:.1f}", flush=True)
    result = {
        "name": args.name or out_dir.name,
        "env_config": env_cfg.to_dict(),
        "args": vars(args),
        "params": n_params,
        "train_minutes": (time.time() - t0) / 60,
        "agent_steps": agent_steps,
        "agent_steps_per_s": agent_steps / max(train_seconds, 1e-9),
        "timing_seconds": timing,
        "final": history[-1],
        "random": random_metrics,
        "history": history,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description="CIDA-Swarm: MAPPO search swarm")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="runs/swarm")
    # среда
    ap.add_argument("--n-envs", type=int, default=8192)
    ap.add_argument("--n-drones", type=int, default=8)
    ap.add_argument("--min-drones", type=int, default=4)
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--n-decoys", type=int, default=4)
    ap.add_argument("--n-obstacles", type=int, default=6)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--view-radius", type=float, default=0.08)
    ap.add_argument("--max-speed", type=float, default=0.015)
    ap.add_argument("--target-speed", type=float, default=0.002)
    ap.add_argument("--share-coverage", action="store_true", help="стигмергия: общая карта исследованного")
    # модель и PPO
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--total-steps", type=float, default=None, help="бюджет шагов дронов; задаёт число итераций")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch-size", type=int, default=65536, help="миров-моментов в минибатче")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    # служебное
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-envs", type=int, default=4096)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
