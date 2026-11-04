"""
arena_train.py — обучение «спасатели против охотников»: две команды учатся одновременно, игрой друг против друга.

Обе команды — MAPPO со своим мозгом (общие веса внутри команды, у каждого агента своя память, личность и роль)
и своим централизованным критиком. Самоигра с лигой (как у AlphaStar и OpenAI Five): раз в league_every итераций
снимок обеих команд кладётся в лигу; в доле league_prob итераций одна команда играет против случайной старой
версии соперника (и учится только она) — чтобы не забывать, как побеждать прежние тактики, и не ходить по кругу.

    python -m cida_swarm.train --world rescue --hunters 3 --init-from runs/v3_cida/model_best.pt \\
        --memory --personas --roles --comm cida --no-route-hint --hidden-weight --hard-bank 32768 \\
        --total-steps 5e9 --out runs/arena

Вход — тот же train.py (флаг --hunters > 0). Спасатели стартуют с готового роя (--init-from, можно --freeze-base),
охотники — с того же мозга (умеют ходить по зданию), но учатся охотиться сами.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.distributions import Categorical

from .arena import ArenaEnv
from .policy import Critic, RescueActor, forget_done, init_memory, policy_step
from .train import (RunningMeanStd, _fmt_metrics, _stack, _sync, actor_kwargs_from, autocast, build_env_config,
                    build_optimizer, compute_gae, load_pretrained, ppo_update)


class Team:
    """Команда арены: мозг, критик, оптимизатор, нормировка ценности и память агентов."""

    def __init__(self, name: str, actor: RescueActor, critic: Critic, opt, params, ent_coef: float, n_envs: int,
                 n_agents: int, device):
        self.name, self.actor, self.critic, self.opt, self.params = name, actor, critic, opt, params
        self.ent_coef = ent_coef
        self.ret_norm = RunningMeanStd(device)
        self.mem = init_memory(actor, n_envs, n_agents, device)
        self.base_lrs = [g["lr_mult"] for g in opt.param_groups]
        self.league = []                       # снимки мозга (state_dict) для игры против старых версий


@torch.no_grad()
def evaluate_arena(rescuer, hunter, cfg, n_envs: int, device, seed: int = 12345) -> Dict:
    """Один эпизод в каждом из n_envs свежих арен. hunter=None — охотники стоят на месте."""
    device = torch.device(device)
    cfg = type(cfg)(**{**cfg.to_dict(), "n_envs": n_envs, "bank_size": 0})
    env = ArenaEnv(cfg, device, seed=seed)
    obs = env.reset()
    mem = init_memory(rescuer, n_envs, cfg.n_drones, device)
    hmem = init_memory(hunter, n_envs, cfg.n_hunters, device) if hunter is not None else None
    finished = torch.zeros(n_envs, dtype=torch.bool, device=device)
    keys = list(env.STATS) + ["difficulty"]
    stats = {k: torch.zeros(n_envs, device=device) for k in keys}
    for _ in range(cfg.max_steps):
        with autocast(device):
            logits, _, mem = policy_step(rescuer, obs, mem)
            if hunter is not None:
                hlogits, _, hmem = policy_step(hunter, env.observe_hunters(), hmem)
        actions = Categorical(logits=logits.float()).sample()
        h_act = Categorical(logits=hlogits.float()).sample() if hunter is not None else None
        obs, _, done, info = env.step(actions, h_act)
        mem, hmem = forget_done(mem, done), forget_done(hmem, done)
        first = done & ~finished
        for k in stats:
            stats[k] = torch.where(first, info[k], stats[k])
        finished |= done
        if bool(finished.all()):
            break
    out = {k: float(stats[k].mean()) for k in env.STATS}
    out["by_level"] = {str(lv): {k: float(stats[k][stats["difficulty"] == lv].mean()) for k in env.STATS}
                       for lv in range(1, 5) if bool((stats["difficulty"] == lv).any())}
    return out


def _ghost(team: Team) -> Optional[RescueActor]:
    """Случайная старая версия мозга команды из лиги (или None, если лига пуста)."""
    if not team.league:
        return None
    g = copy.deepcopy(team.actor)
    g.load_state_dict(random.choice(team.league))
    return g.eval()


def train_arena(args) -> Dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.world != "rescue":
        raise ValueError("арена работает только в мире rescue (--world rescue)")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = build_env_config(args)
    env = ArenaEnv(env_cfg, device, seed=args.seed)
    h_cfg = env.hunter_config()
    kwargs = actor_kwargs_from(args, "rescue")
    state_dim = ArenaEnv.state_dim(env_cfg)

    # спасатели: готовый рой (+ новые модули с нуля), при желании с замороженными «глазами и ногами»
    r_actor, r_critic = RescueActor(env_cfg, **kwargs).to(device), Critic(state_dim).to(device)
    r_new = load_pretrained(r_actor, r_critic, args.init_from, device) if args.init_from else set()
    r_opt, r_params = build_optimizer(r_actor, r_critic, r_new, args, freeze=args.freeze_base)
    # охотники: тот же мозг-устройство; стартуют с весов спасателей — уже умеют ходить по зданию
    h_actor, h_critic = RescueActor(h_cfg, **kwargs).to(device), Critic(state_dim).to(device)
    h_src = args.hunter_init or args.init_from
    h_new = load_pretrained(h_actor, h_critic, h_src, device, quiet=True) if h_src else set()
    h_opt, h_params = build_optimizer(h_actor, h_critic, h_new, args, lr_scale=args.hunter_lr / args.lr)

    B, N, H, T = env_cfg.n_envs, env_cfg.n_drones, env_cfg.n_hunters, args.rollout
    rescuers = Team("rescuers", r_actor, r_critic, r_opt, r_params, args.ent_coef, B, N, device)
    hunters = Team("hunters", h_actor, h_critic, h_opt, h_params, args.hunter_ent_coef, B, H, device)
    teams = (rescuers, hunters)
    steps_per_iter = T * B * (N + H)
    iters = math.ceil(args.total_steps / steps_per_iter) if args.total_steps else args.iters
    print(f"[arena] спасателей {env_cfg.min_drones}-{N}, охотников 1-{H}, миров {B}, comm={args.comm}, "
          f"iters={iters} (~{iters * steps_per_iter / 1e9:.2f}B шагов агентов), лига: снимок раз в "
          f"{args.league_every}, старый соперник в {args.league_prob:.0%} итераций", flush=True)

    obs, h_obs, state = env.reset(), env.observe_hunters(), env.state()
    history, best = [], None
    ep_sum = {k: torch.zeros((), device=device) for k in env.STATS}
    ep_count = torch.zeros((), device=device)
    first_hunter = copy.deepcopy(h_actor.state_dict())   # для оценки «против самых первых охотников»
    t0, eval_seconds, agent_steps = time.time(), 0.0, 0

    for it in range(1, iters + 1):
        frac = 1.0 - (it - 1) / iters
        for team in teams:
            for g, mult in zip(team.opt.param_groups, team.base_lrs):
                g["lr"] = args.lr * frac * mult
        # кто учится в этой итерации: обе команды против текущих версий — или одна против старой версии другой
        roll_mode = random.random()
        learners = {"rescuers", "hunters"}
        r_play, h_play = r_actor, h_actor
        if roll_mode < args.league_prob / 2 and hunters.league:
            h_play, learners = _ghost(hunters), {"rescuers"}
        elif roll_mode < args.league_prob and rescuers.league:
            r_play, learners = _ghost(rescuers), {"hunters"}

        t_start = _sync(device)
        bufs = {t.name: {"obs": [], "act": [], "logp": [], "val": [], "rew": []} for t in teams}
        buf_state, buf_people, buf_done = [], [], []
        mem0 = {t.name: (t.mem.clone() if t.mem is not None else None) for t in teams}
        with torch.no_grad():
            for _ in range(T):
                with autocast(device):
                    r_logits, _, rescuers.mem = policy_step(r_play, obs, rescuers.mem)
                    h_logits, _, hunters.mem = policy_step(h_play, h_obs, hunters.mem)
                    r_val, h_val = r_critic(state), h_critic(state)
                r_dist, h_dist = Categorical(logits=r_logits.float()), Categorical(logits=h_logits.float())
                r_act, h_act = r_dist.sample(), h_dist.sample()
                for team, o, a, d, v in ((rescuers, obs, r_act, r_dist, r_val), (hunters, h_obs, h_act, h_dist, h_val)):
                    bb = bufs[team.name]
                    bb["obs"].append(o)
                    bb["act"].append(a)
                    bb["logp"].append(d.log_prob(a))
                    bb["val"].append(v.float() * team.ret_norm.std + team.ret_norm.mean)
                buf_state.append(state.half())
                buf_people.append(env.belief_target())
                obs, reward, done, info = env.step(r_act, h_act)
                h_obs, state = env.observe_hunters(), env.state()
                bufs["rescuers"]["rew"].append(reward)
                bufs["hunters"]["rew"].append(env.hunter_reward.clone())
                buf_done.append(done.float())
                rescuers.mem, hunters.mem = forget_done(rescuers.mem, done), forget_done(hunters.mem, done)
                for k in env.STATS:
                    ep_sum[k] += (info[k] * done).sum()
                ep_count += done.sum()
            with autocast(device):
                last = {"rescuers": r_critic(state).float(), "hunters": h_critic(state).float()}
        dones = torch.stack(buf_done)
        seq_state, seq_people = torch.stack(buf_state), torch.stack(buf_people)
        del buf_state, buf_people
        agent_steps += steps_per_iter
        t_mid = _sync(device)

        logs = {}
        for team in teams:
            bb = bufs[team.name]
            if team.name not in learners:
                continue
            with torch.no_grad():
                last_val = last[team.name] * team.ret_norm.std + team.ret_norm.mean
                adv, returns = compute_gae(torch.stack(bb["val"]), torch.stack(bb["rew"]), dones, last_val,
                                           args.gamma, args.lam)
                team.ret_norm.update(returns)
            roll = {"obs": _stack(bb["obs"]), "state": seq_state, "people": seq_people, "act": torch.stack(bb["act"]),
                    "logp": torch.stack(bb["logp"]), "adv": adv, "ret": (returns - team.ret_norm.mean) / team.ret_norm.std,
                    "dones": dones, "mem0": mem0[team.name]}
            bufs[team.name] = None
            entropy, v_loss = ppo_update(team.actor, team.critic, team.opt, team.params, roll, args, it, device,
                                         team.ent_coef)
            logs[team.name] = (entropy, v_loss)
            del roll
        t_end = _sync(device)

        if it % args.league_every == 0:
            for team in teams:
                team.league.append(copy.deepcopy(team.actor.state_dict()))
                team.league = team.league[-args.league_size:]

        if it % args.log_every == 0:
            cnt = ep_count.item()
            msg = (" | ".join(f"{k} {ep_sum[k].item() / cnt:.3f}" for k in env.STATS) if cnt else "no episodes yet")
            ent = " ".join(f"{n[0]}:{e.item():.2f}" for n, (e, _) in logs.items())
            print(f"it {it:5d}/{iters} | {msg} | entropy {ent} | учатся {'+'.join(sorted(learners))} | "
                  f"{agent_steps / (time.time() - t0 - eval_seconds):,.0f} agent-steps/s "
                  f"(collect {(t_mid - t_start) / max(t_end - t_start, 1e-9):.0%})", flush=True)
            ep_sum = {k: torch.zeros((), device=device) for k in env.STATS}
            ep_count.zero_()

        if it % args.eval_every == 0 or it == iters:
            t_eval = time.time()
            r_actor.eval(), h_actor.eval()
            vs_now = evaluate_arena(r_actor, h_actor, env_cfg, args.eval_envs, device)
            old = copy.deepcopy(h_actor)
            old.load_state_dict(first_hunter)
            vs_first = evaluate_arena(r_actor, old, env_cfg, args.eval_envs, device)
            r_actor.train(), h_actor.train()
            metrics = {**vs_now, "vs_first_hunters": vs_first["rescued_frac"], "iter": it,
                       "agent_steps": agent_steps, "minutes": (time.time() - t0) / 60}
            history.append(metrics)
            levels = " ".join(f"L{k}:{v['rescued_frac']:.2f}" for k, v in metrics["by_level"].items())
            print(f"[eval] it {it} | {_fmt_metrics(metrics)} | {levels}", flush=True)
            ckpt = {"world": "arena", "env_config": env_cfg.to_dict(), "actor_kwargs": kwargs,
                    "actor": r_actor.state_dict(), "critic": r_critic.state_dict(),
                    "hunter": h_actor.state_dict(), "hunter_critic": h_critic.state_dict(), "iter": it}
            torch.save(ckpt, out_dir / "model.pt")
            # в самоигре «лучший» по счёту против меняющегося соперника не имеет смысла: свежая версия сильнее
            torch.save(ckpt, out_dir / "model_best.pt")
            best = metrics
            eval_seconds += time.time() - t_eval

    print(f"[eval] final | {_fmt_metrics(best)}", flush=True)
    result = {"name": args.name or out_dir.name, "env_config": env_cfg.to_dict(), "args": vars(args),
              "train_minutes": (time.time() - t0) / 60, "agent_steps": agent_steps,
              "agent_steps_per_s": agent_steps / max(time.time() - t0 - eval_seconds, 1e-9),
              "final": history[-1], "best": best, "history": history}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
