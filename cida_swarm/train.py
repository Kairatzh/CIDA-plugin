"""
train.py — обучение роя через MAPPO (общая политика дронов + централизованный критик).

Пример (Colab A100):
    python -m cida_swarm.train --total-steps 2e9 --out runs/nocomm_s0
    python -m cida_swarm.train --total-steps 2e9 --share-coverage --out runs/stigmergy_s0
    python -m cida_swarm.train --total-steps 2e9 --comm cida --out runs/cida_s0
    python -m cida_swarm.train --comm cida --share-coverage --init-from runs/stigmergy_s0/model.pt \
        --critic-warmup 10 --lr 1e-4 --total-steps 1.5e9 --out runs/cida_ft_s0

Спасательная миссия (мир rescue): --world rescue --n-drones 6 --min-drones 3 --max-steps 300.

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

from . import env as search_world
from . import rescue as rescue_world
from .env import EnvConfig, SearchEnv
from .policy import Actor, Critic, RescueActor
from .rescue import RescueConfig, RescueEnv

# мир → (класс среды, класс мозга, число действий)
WORLDS = {
    "search": (SearchEnv, Actor, search_world.N_ACTIONS),
    "rescue": (RescueEnv, RescueActor, rescue_world.N_ACTIONS),
}


def world_of(env_cfg) -> str:
    return "rescue" if isinstance(env_cfg, RescueConfig) else "search"


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
def evaluate(actor: Optional[torch.nn.Module], env_cfg, n_envs: int, device, seed: int = 12345,
             world: Optional[str] = None) -> Dict:
    """
    Один эпизод в каждом из n_envs свежих миров. actor=None — случайная политика.
    Для спасательной миссии дополнительно — разбивка по сложности мира (by_level).
    """
    device = torch.device(device)
    world = world or world_of(env_cfg)
    env_cls, _, n_actions = WORLDS[world]
    cfg = type(env_cfg)(**{**env_cfg.to_dict(), "n_envs": n_envs})
    env = env_cls(cfg, device, seed=seed)
    obs = env.reset()
    finished = torch.zeros(n_envs, dtype=torch.bool, device=device)
    keys = list(env.STATS) + (["difficulty"] if world == "rescue" else [])
    stats = {k: torch.zeros(n_envs, device=device) for k in keys}
    for _ in range(cfg.max_steps):
        if actor is None:
            actions = torch.randint(0, n_actions, (n_envs, cfg.n_drones), device=device)
        else:
            with autocast(device):
                logits = actor(obs)
            actions = Categorical(logits=logits.float()).sample()
        obs, _, done, info = env.step(actions)
        first = done & ~finished
        for k in stats:
            stats[k] = torch.where(first, info[k], stats[k])
        finished |= done
    out = {k: float(stats[k].mean()) for k in env.STATS}
    if world == "search":
        out["all_found"] = float((stats["found_frac"] >= 1.0).float().mean())
    else:
        out["by_level"] = {}
        for level in range(1, 5):
            sel = stats["difficulty"] == level
            if bool(sel.any()):
                out["by_level"][str(level)] = {k: float(stats[k][sel].mean()) for k in env.STATS}
    return out


def _fmt_metrics(m: Dict) -> str:
    return " | ".join(f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, float))


# ── обучение ─────────────────────────────────────────────────────────────────
def build_env_config(args):
    if args.world == "rescue":
        return RescueConfig(
            n_envs=args.n_envs, n_drones=args.n_drones, min_drones=args.min_drones, n_victims=args.n_victims,
            max_steps=args.max_steps, difficulty_min=args.difficulty_min, difficulty_max=args.difficulty_max,
            share_map=args.share_coverage,
        )
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
    env_cls, actor_cls, _ = WORLDS[args.world]
    env = env_cls(env_cfg, device, seed=args.seed)
    success_key = env.SUCCESS
    # радиус связи: в поиске — соседи; в здании — радиосеть на всё здание (иначе дрон слышит лишь тех, кого и так видит)
    comm_radius = args.comm_radius if args.comm_radius is not None else (1.5 if args.world == "rescue" else 0.3)
    actor_kwargs = {"hidden": args.hidden, "comm": args.comm, "comm_radius": comm_radius,
                    "comm_rounds": args.comm_rounds}
    actor = actor_cls(env_cfg, **actor_kwargs).to(device)
    critic = Critic(env_cls.state_dim(env_cfg)).to(device)
    if args.init_from:
        # дообучение: берём готовый рой; новые модули (связь CIDA) стартуют с нулевым выходом,
        # поэтому в начале рой ведёт себя ровно как исходный и учится только новому
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = actor.load_state_dict(ckpt["actor"], strict=False)
        if unexpected or any(not k.startswith("comm.") for k in missing):
            raise ValueError(f"несовместимый чекпойнт {args.init_from}: missing={missing}, unexpected={unexpected}")
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"])
        print(f"[cida-swarm] init from {args.init_from}: новые модули {sorted({k.split('.')[0] for k in missing}) or '—'}, "
              f"критик {'загружен' if 'critic' in ckpt else 'с нуля (разогрев ' + str(args.critic_warmup) + ' итераций)'}",
              flush=True)
    # новые модули дообучения учатся с нуля, поэтому им шаг крупнее, чем уже обученному рою
    new_names = set(missing) if args.init_from else set()
    new = [p for k, p in actor.named_parameters() if k in new_names]
    old = [p for k, p in actor.named_parameters() if k not in new_names] + list(critic.parameters())
    groups = [{"params": old, "lr_mult": 1.0}] + ([{"params": new, "lr_mult": args.new_lr_mult}] if new else [])
    opt = torch.optim.Adam(groups, lr=args.lr, eps=1e-5)
    params = old + new
    run_actor, run_critic = actor, critic
    if args.compile:
        print("[cida-swarm] torch.compile: первые итерации займут пару минут на компиляцию", flush=True)
        run_actor, run_critic = torch.compile(actor), torch.compile(critic)
    ret_norm = RunningMeanStd(device)

    T, B, N = args.rollout, env_cfg.n_envs, env_cfg.n_drones
    steps_per_iter = T * B * N
    iters = math.ceil(args.total_steps / steps_per_iter) if args.total_steps else args.iters
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"[cida-swarm] world={args.world} actor params={n_params:,} device={device} worlds={B} "
          f"drones={env_cfg.min_drones}-{N} share_map={args.share_coverage} comm={args.comm} "
          f"iters={iters} (~{iters * steps_per_iter / 1e9:.2f}B agent-steps)", flush=True)

    obs = env.reset()
    state = env.state()
    history = []
    ep_sum = {k: torch.zeros((), device=device) for k in env.STATS}
    ep_count = torch.zeros((), device=device)
    best, best_score = None, -1.0
    t0 = time.time()
    eval_seconds = 0.0
    timing = {"collect": 0.0, "update": 0.0}
    agent_steps = 0

    for it in range(1, iters + 1):
        lr = args.lr * (1.0 - (it - 1) / iters)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]

        # 1) сбор опыта
        t_start = _sync(device)
        buf_obs, buf_state, buf_act, buf_logp, buf_val, buf_rew, buf_done = [], [], [], [], [], [], []
        buf_people = []
        with torch.no_grad():
            for _ in range(T):
                with autocast(device):
                    logits, value = run_actor(obs), run_critic(state)
                dist = Categorical(logits=logits.float())
                actions = dist.sample()
                buf_obs.append(obs)
                buf_state.append(state.half())  # состояние мира большое — храним в половинной точности
                buf_people.append(env.belief_target())
                buf_act.append(actions)
                buf_logp.append(dist.log_prob(actions))
                buf_val.append(value.float() * ret_norm.std + ret_norm.mean)
                obs, reward, done, info = env.step(actions)
                state = env.state()
                buf_rew.append(reward)
                buf_done.append(done.float())
                for k in env.STATS:
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
        del buf_obs, buf_state  # исходные списки больше не нужны — освобождаем память до PPO
        flat_people = torch.stack(buf_people).flatten(0, 1)
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
                    (logits, aux), v = run_actor(o, return_aux=True), run_critic(flat_state[idx].float())
                dist = Categorical(logits=logits.float())
                ratio = (dist.log_prob(flat_act[idx]) - flat_logp[idx]).exp()
                pg = -_masked_mean(torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a), mask)
                entropy = _masked_mean(dist.entropy(), mask)
                v_loss = F.mse_loss(v.float(), flat_ret[idx])
                loss = pg + args.vf_coef * v_loss - args.ent_coef * entropy
                if it <= args.critic_warmup:
                    # критик ещё не знает этот рой — пока он учится, политику не трогаем,
                    # иначе шумные оценки испортят уже обученное поведение
                    loss = args.vf_coef * v_loss
                if aux is not None and it > args.critic_warmup:
                    # убеждения учатся предсказывать, где люди: так у сообщений и у расхождения есть смысл
                    truth = flat_people[idx].float()[:, None].expand(-1, mask.shape[1], -1)
                    b_loss = sum(
                        _masked_mean(F.binary_cross_entropy_with_logits(b.float(), truth, reduction="none").mean(-1), mask)
                        for b in aux["beliefs"]
                    ) / len(aux["beliefs"])
                    loss = loss + args.belief_coef * b_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
        timing["update"] += _sync(device) - t_mid

        if it % args.log_every == 0:
            cnt = ep_count.item()
            msg = (" | ".join(f"{k} {ep_sum[k].item() / cnt:.3f}" for k in env.STATS)
                   if cnt else "no episodes yet")
            print(f"it {it:5d}/{iters} | {msg} | entropy {entropy.item():.3f} | v_loss {v_loss.item():.3f} | "
                  f"{agent_steps / (time.time() - t0 - eval_seconds):,.0f} agent-steps/s "
                  f"(collect {timing['collect'] / max(sum(timing.values()), 1e-9):.0%})", flush=True)
            ep_sum = {k: torch.zeros((), device=device) for k in env.STATS}
            ep_count.zero_()

        if it % args.eval_every == 0 or it == iters:
            t_eval = time.time()
            metrics = evaluate(actor, env_cfg, args.eval_envs, device)
            metrics.update(iter=it, agent_steps=agent_steps, minutes=(time.time() - t0) / 60)
            history.append(metrics)
            levels = " ".join(f"L{k}:{v[success_key]:.2f}" for k, v in metrics.get("by_level", {}).items())
            print(f"[eval] it {it} | {_fmt_metrics(metrics)}" + (f" | {levels}" if levels else ""), flush=True)
            ckpt = {"world": args.world, "env_config": env_cfg.to_dict(), "actor_kwargs": actor_kwargs,
                    "actor": actor.state_dict(), "critic": critic.state_dict(), "iter": it}
            torch.save(ckpt, out_dir / "model.pt")
            if metrics[success_key] > best_score:
                # лучшая версия сохраняется отдельно: дообучение может со временем испортить рой
                best, best_score = metrics, metrics[success_key]
                torch.save(ckpt, out_dir / "model_best.pt")
            eval_seconds += time.time() - t_eval

    train_seconds = time.time() - t0 - eval_seconds
    random_metrics = evaluate(None, env_cfg, args.eval_envs, device)
    print(f"[eval] random policy | {_fmt_metrics(random_metrics)}", flush=True)
    print(f"[eval] best it {best['iter']} | {_fmt_metrics(best)}", flush=True)
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
        "best": best,
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
    ap.add_argument("--world", choices=list(WORLDS), default="search", help="search — поиск, rescue — спасательная миссия")
    ap.add_argument("--n-envs", type=int, default=8192)
    ap.add_argument("--n-drones", type=int, default=8)
    ap.add_argument("--min-drones", type=int, default=4)
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--n-victims", type=int, default=3, help="rescue: пострадавших в здании")
    ap.add_argument("--difficulty-min", type=int, default=1, help="rescue: сложность миров от")
    ap.add_argument("--difficulty-max", type=int, default=4, help="rescue: сложность миров до")
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
    ap.add_argument("--comm", choices=["none", "attn", "cida"], default="none",
                    help="связь: none — нет, attn — обычное внимание, cida — внимание со смещением разногласием")
    ap.add_argument("--comm-radius", type=float, default=None,
                    help="радиус связи в долях поля (по умолчанию: поиск 0.3, здание 1.5 — всё здание)")
    ap.add_argument("--comm-rounds", type=int, default=2)
    ap.add_argument("--belief-coef", type=float, default=0.1)
    ap.add_argument("--init-from", default=None, help="model.pt готового роя для дообучения")
    ap.add_argument("--new-lr-mult", type=float, default=10.0,
                    help="во сколько раз больше шаг обучения у новых модулей при дообучении")
    ap.add_argument("--critic-warmup", type=int, default=0, help="итераций, когда учится только критик")
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
