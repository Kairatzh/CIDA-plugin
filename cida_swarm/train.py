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
from .policy import Actor, Critic, RescueActor, forget_done, init_memory, policy_step
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
    mem = init_memory(actor, n_envs, cfg.n_drones, device) if actor is not None else None
    finished = torch.zeros(n_envs, dtype=torch.bool, device=device)
    keys = list(env.STATS) + (["difficulty"] if world == "rescue" else [])
    stats = {k: torch.zeros(n_envs, device=device) for k in keys}
    for _ in range(cfg.max_steps):
        if actor is None:
            actions = torch.randint(0, n_actions, (n_envs, cfg.n_drones), device=device)
        else:
            with autocast(device):
                logits, _, mem = policy_step(actor, obs, mem)
            actions = Categorical(logits=logits.float()).sample()
        obs, _, done, info = env.step(actions)
        mem = forget_done(mem, done)
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


# ── общие части обучения (ими же учатся обе команды арены) ─────────────────────
NEW_MODULES = ("comm.", "gru.", "mem_out.", "role_enc.", "persona")


def load_pretrained(actor, critic, path: str, device, quiet: bool = False):
    """Готовые веса в мозг (и критика, если подходит по размеру). Новые модули стартуют с нуля. → имена новых."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    own = actor.state_dict()
    src = {k: v for k, v in ckpt["actor"].items() if k in own and own[k].shape == v.shape}
    missing, unexpected = actor.load_state_dict(src, strict=False)
    skipped = sorted(set(ckpt["actor"]) - set(src))
    if unexpected or any(not k.startswith(NEW_MODULES) for k in missing):
        raise ValueError(f"несовместимый чекпойнт {path}: missing={missing}, skipped={skipped}")
    critic_ok = "critic" in ckpt and all(
        k in critic.state_dict() and critic.state_dict()[k].shape == v.shape for k, v in ckpt["critic"].items())
    if critic_ok:
        critic.load_state_dict(ckpt["critic"])
    if not quiet:
        print(f"[cida-swarm] init from {path}: новые модули {sorted({k.split('.')[0] for k in missing}) or '—'}, "
              f"критик {'загружен' if critic_ok else 'с нуля'}", flush=True)
    return set(missing)


def build_optimizer(actor, critic, new_names, args, freeze: bool = False, lr_scale: float = 1.0):
    """Adam: новым модулям шаг крупнее; freeze — «глаза и ноги» заморожены, учатся слой CIDA и выход действий."""
    if freeze:
        for k, p in actor.named_parameters():
            if k not in new_names and not k.startswith(NEW_MODULES + ("head.",)):
                p.requires_grad_(False)
    new = [p for k, p in actor.named_parameters() if k in new_names]
    old = [p for k, p in actor.named_parameters() if k not in new_names and p.requires_grad] + list(critic.parameters())
    groups = [{"params": old, "lr_mult": lr_scale}] + ([{"params": new, "lr_mult": lr_scale * args.new_lr_mult}] if new else [])
    return torch.optim.Adam(groups, lr=args.lr, eps=1e-5), old + new


def compute_gae(values, rewards, dones, last_val, gamma: float, lam: float):
    """GAE по общей награде команды. Всё (T, B). → (advantages, returns)."""
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_val)
    T = rewards.shape[0]
    for t in reversed(range(T)):
        next_val = last_val if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


def ppo_update(run_actor, run_critic, opt, params, roll: Dict, args, it: int, device, ent_coef: float):
    """
    PPO; минибатч = отрезки жизни случайных миров (T шагов × все агенты), неактивные агенты маскируются.
    У мозга с памятью отрезок прокручивается заново по шагам с сохранённой памяти — так память учится.
    roll: obs (T,B,...), state, people, act, logp, adv, ret (нормированные), dones, mem0.
    """
    seq_obs, adv, dones, mem0 = roll["obs"], roll["adv"], roll["dones"], roll["mem0"]
    T, B = adv.shape
    mbw = max(1, min(B, args.minibatch_size // T))              # миров в минибатче
    entropy = v_loss = torch.zeros((), device=device)
    for _ in range(args.epochs):
        perm = torch.randperm(B, device=device)
        for start in range(0, B - mbw + 1, mbw):
            w_idx = perm[start:start + mbw]
            o = {k: v[:, w_idx] for k, v in seq_obs.items()}
            mask = o["active"].flatten(0, 1).float()
            a = adv[:, w_idx].flatten()
            a = ((a - a.mean()) / (a.std() + 1e-8))[:, None]
            with autocast(device):
                if mem0 is None:
                    logits, aux, _ = policy_step(run_actor, {k: v.flatten(0, 1) for k, v in o.items()}, None)
                else:
                    m, outs = mem0[w_idx], []
                    for t in range(T):
                        lt, at, m = policy_step(run_actor, {k: v[t] for k, v in o.items()}, m)
                        m = forget_done(m, dones[t, w_idx])
                        outs.append((lt, at))
                    logits = torch.cat([lt for lt, _ in outs])
                    aux = None if outs[0][1] is None else {"beliefs": [
                        torch.cat([at["beliefs"][r] for _, at in outs]) for r in range(len(outs[0][1]["beliefs"]))]}
                v = run_critic(roll["state"][:, w_idx].flatten(0, 1).float())
            dist = Categorical(logits=logits.float())
            ratio = (dist.log_prob(roll["act"][:, w_idx].flatten(0, 1)) - roll["logp"][:, w_idx].flatten(0, 1)).exp()
            pg = -_masked_mean(torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a), mask)
            entropy = _masked_mean(dist.entropy(), mask)
            v_loss = F.mse_loss(v.float(), roll["ret"][:, w_idx].flatten())
            loss = pg + args.vf_coef * v_loss - ent_coef * entropy
            if it <= args.critic_warmup:
                # критик ещё не знает этот рой — пока он учится, политику не трогаем,
                # иначе шумные оценки испортят уже обученное поведение
                loss = args.vf_coef * v_loss
            if aux is not None and it > args.critic_warmup:
                # убеждения учатся предсказывать, где люди: так у сообщений и у расхождения есть смысл
                truth = roll["people"][:, w_idx].flatten(0, 1).float()[:, None].expand(-1, mask.shape[1], -1)
                b_loss = sum(
                    _masked_mean(F.binary_cross_entropy_with_logits(b.float(), truth, reduction="none").mean(-1), mask)
                    for b in aux["beliefs"]
                ) / len(aux["beliefs"])
                loss = loss + args.belief_coef * b_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
    return entropy, v_loss


def actor_kwargs_from(args, world: str) -> Dict:
    # радиус связи: в поиске — соседи; в здании — радиосеть на всё здание (иначе дрон слышит лишь тех, кого и так видит)
    comm_radius = args.comm_radius if args.comm_radius is not None else (0.3 if world == "search" else 1.5)
    kw = {"hidden": args.hidden, "comm": args.comm, "comm_radius": comm_radius, "comm_rounds": args.comm_rounds}
    for flag in ("memory", "roles", "personas"):
        if getattr(args, flag):
            if world == "search":
                raise ValueError(f"--{flag} пока есть только в мире rescue")
            kw[flag] = True
    return kw


# ── обучение ─────────────────────────────────────────────────────────────────
def build_env_config(args):
    if args.world == "rescue":
        return RescueConfig(
            n_envs=args.n_envs, n_drones=args.n_drones, min_drones=args.min_drones, n_victims=args.n_victims,
            max_steps=args.max_steps, difficulty_min=args.difficulty_min, difficulty_max=args.difficulty_max,
            share_map=args.share_coverage, route_hint=not args.no_route_hint, hidden_weight=args.hidden_weight,
            hazard_chase=args.hazard_chase, penalty_crowd=args.crowd_penalty,
            bank_size=args.hard_bank, n_hunters=args.hunters,
        )
    return EnvConfig(
        n_envs=args.n_envs, n_drones=args.n_drones, min_drones=args.min_drones, n_targets=args.n_targets,
        n_decoys=args.n_decoys, n_obstacles=args.n_obstacles, grid=args.grid, max_steps=args.max_steps,
        view_radius=args.view_radius, max_speed=args.max_speed, target_speed=args.target_speed,
        share_coverage=args.share_coverage,
    )


def train(args) -> Dict:
    if args.hunters > 0:
        from .arena_train import train_arena   # «спасатели против охотников»: две обучаемые команды
        return train_arena(args)
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
    actor_kwargs = actor_kwargs_from(args, args.world)
    actor = actor_cls(env_cfg, **actor_kwargs).to(device)
    critic = Critic(env_cls.state_dim(env_cfg)).to(device)
    # дообучение: берём готовый рой; новые модули (связь CIDA) стартуют с нулевым выходом,
    # поэтому в начале рой ведёт себя ровно как исходный и учится только новому
    new_names = load_pretrained(actor, critic, args.init_from, device) if args.init_from else set()
    if args.freeze_base and not args.init_from:
        raise ValueError("--freeze-base нужен вместе с --init-from")
    # «глаза и ноги» готового роя заморожены: учится только слой CIDA (память, личности, роли, радио)
    # и выход действий, через который этот слой управляет телом
    opt, params = build_optimizer(actor, critic, new_names, args, freeze=args.freeze_base)
    if args.freeze_base:
        n_train = sum(p.numel() for p in actor.parameters() if p.requires_grad)
        print(f"[cida-swarm] база заморожена: учатся {n_train:,} из {sum(p.numel() for p in actor.parameters()):,} "
              f"параметров мозга", flush=True)
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

    if args.hint_anneal > 0:
        env.hint_prob = 1.0
    obs = env.reset()
    state = env.state()
    mem = init_memory(actor, B, N, device)
    # автокуррикулум: уровни, где рой то справляется, то нет (успех около 50%), выпадают чаще
    n_levels = env_cfg.difficulty_max - env_cfg.difficulty_min + 1 if args.world == "rescue" else 0
    lvl_sum = torch.zeros(max(n_levels, 1), device=device)
    lvl_cnt = torch.zeros(max(n_levels, 1), device=device)
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
        mem0 = mem.clone() if mem is not None else None   # с этой памяти PPO заново прокрутит отрезок
        with torch.no_grad():
            for _ in range(T):
                with autocast(device):
                    logits, _, mem = policy_step(run_actor, obs, mem)
                    value = run_critic(state)
                dist = Categorical(logits=logits.float())
                actions = dist.sample()
                buf_obs.append(obs)
                buf_state.append(state.half())  # состояние мира большое — храним в половинной точности
                buf_people.append(env.belief_target())
                buf_act.append(actions)
                buf_logp.append(dist.log_prob(actions))
                buf_val.append(value.float() * ret_norm.std + ret_norm.mean)
                obs, reward, done, info = env.step(actions)
                mem = forget_done(mem, done)
                state = env.state()
                if n_levels:
                    lvl = (info["difficulty"].long() - env_cfg.difficulty_min).clamp(0, n_levels - 1)
                    lvl_sum.index_add_(0, lvl, info[success_key] * done)
                    lvl_cnt.index_add_(0, lvl, done.float())
                buf_rew.append(reward)
                buf_done.append(done.float())
                for k in env.STATS:
                    ep_sum[k] += (info[k] * done).sum()
                ep_count += done.sum()
            with autocast(device):
                last_val = run_critic(state).float() * ret_norm.std + ret_norm.mean

            # 2) GAE по общей награде команды
            dones = torch.stack(buf_done)
            adv, returns = compute_gae(torch.stack(buf_val), torch.stack(buf_rew), dones, last_val, args.gamma, args.lam)
            ret_norm.update(returns)
        agent_steps += steps_per_iter
        t_mid = _sync(device)
        timing["collect"] += t_mid - t_start

        # 3) PPO
        roll = {"obs": _stack(buf_obs), "state": torch.stack(buf_state), "people": torch.stack(buf_people),
                "act": torch.stack(buf_act), "logp": torch.stack(buf_logp), "adv": adv,
                "ret": (returns - ret_norm.mean) / ret_norm.std, "dones": dones, "mem0": mem0}
        del buf_obs, buf_state  # исходные списки больше не нужны — освобождаем память до PPO
        entropy, v_loss = ppo_update(run_actor, run_critic, opt, params, roll, args, it, device, args.ent_coef)
        del roll
        timing["update"] += _sync(device) - t_mid

        if args.hint_anneal > 0:
            # леса: доля миров с подсказками плавно падает от 1 до 0 за первую долю hint_anneal обучения
            env.hint_prob = max(0.0, 1.0 - it / (args.hint_anneal * iters))

        if args.curriculum and n_levels and it % 5 == 0:
            succ = lvl_sum / lvl_cnt.clamp_min(1.0)
            score = torch.where(lvl_cnt > 0, succ * (1 - succ), torch.full_like(succ, 0.25)) + 0.1  # каждый уровень не реже ~10%
            env.level_probs = score / score.sum()
            lvl_sum.zero_()
            lvl_cnt.zero_()

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
            curr = (" | curriculum " + " ".join(f"{p:.2f}" for p in env.level_probs.tolist())
                    if getattr(env, "level_probs", None) is not None else "")
            if args.hint_anneal > 0:
                curr += f" | подсказки {env.hint_prob:.0%} миров (оценка — без подсказок)"
            print(f"[eval] it {it} | {_fmt_metrics(metrics)}" + (f" | {levels}" if levels else "") + curr, flush=True)
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
    ap.add_argument("--no-route-hint", action="store_true", help="rescue: без готовой дорожки к выходу")
    ap.add_argument("--hidden-weight", action="store_true", help="rescue: вес человека знает только держащий")
    ap.add_argument("--hazard-chase", type=float, default=0.0, help="rescue: доля ходов охоты опасного человека")
    ap.add_argument("--crowd-penalty", type=float, default=0.0,
                    help="rescue: штраф за шаг свободному дрону, прижавшемуся к другому свободному")
    ap.add_argument("--hard-bank", type=int, default=0,
                    help="rescue: размер банка трудных карт (провалы возвращаются с мутациями); 0 — без банка")
    # арена: команда охотников
    ap.add_argument("--hunters", type=int, default=0, help="охотников в здании; >0 — арена «спасатели против охотников»")
    ap.add_argument("--hunter-init", default=None, help="model.pt для мозга охотников (по умолчанию — как у спасателей)")
    ap.add_argument("--hunter-lr", type=float, default=3e-4)
    ap.add_argument("--hunter-ent-coef", type=float, default=0.01)
    ap.add_argument("--league-every", type=int, default=20, help="раз в столько итераций снимок обеих команд в лигу")
    ap.add_argument("--league-size", type=int, default=8)
    ap.add_argument("--league-prob", type=float, default=0.2,
                    help="доля итераций, где одна команда играет против старой версии соперника")
    ap.add_argument("--hint-anneal", type=float, default=0.0,
                    help="rescue: подсказки (общая карта, дорожка, вес) убираются плавно за эту долю обучения; 0 — сразу")
    ap.add_argument("--curriculum", action="store_true", help="rescue: чаще уровни с успехом около половины")
    # модель и PPO
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--memory", action="store_true", help="rescue: у дрона есть память (GRU)")
    ap.add_argument("--roles", action="store_true", help="rescue: случайная роль дрона на эпизод")
    ap.add_argument("--personas", action="store_true", help="rescue: постоянная обучаемая личность каждого дрона")
    ap.add_argument("--comm", choices=["none", "attn", "cida"], default="none",
                    help="связь: none — нет, attn — обычное внимание, cida — внимание со смещением разногласием")
    ap.add_argument("--comm-radius", type=float, default=None,
                    help="радиус связи в долях поля (по умолчанию: поиск 0.3, здание 1.5 — всё здание)")
    ap.add_argument("--comm-rounds", type=int, default=2)
    ap.add_argument("--belief-coef", type=float, default=0.1)
    ap.add_argument("--init-from", default=None, help="model.pt готового роя для дообучения")
    ap.add_argument("--new-lr-mult", type=float, default=10.0,
                    help="во сколько раз больше шаг обучения у новых модулей при дообучении")
    ap.add_argument("--freeze-base", action="store_true",
                    help="при дообучении заморозить готовый рой и учить только новые модули и выход действий")
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
