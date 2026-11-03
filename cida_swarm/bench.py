"""
bench.py — подбор числа параллельных миров под конкретную видеокарту.

    python -m cida_swarm.bench --n-envs 4096 8192 16384
    python -m cida_swarm.bench --n-envs 8192 --extra "--compile"

Для каждого размера меряет скорость одной среды, скорость полного цикла обучения
(сбор опыта + PPO), долю времени на каждую часть и пиковую память.
Берите размер, после которого скорость почти перестаёт расти, а память с запасом помещается.
"""

from __future__ import annotations

import argparse

import torch

from .train import WORLDS, _sync, build_env_config, parse_args, train


def env_speed(train_args, steps: int = 50) -> float:
    """Шагов дронов в секунду для одной среды со случайными действиями — потолок скорости обучения."""
    device = torch.device(train_args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = build_env_config(train_args)
    env_cls, _, n_actions = WORLDS[train_args.world]
    env = env_cls(cfg, device, seed=0)
    env.reset()
    actions = torch.randint(0, n_actions, (steps, cfg.n_envs, cfg.n_drones), device=device)
    for i in range(5):
        env.step(actions[i])
    t0 = _sync(device)
    for i in range(steps):
        env.step(actions[i])
        env.state()
    return steps * cfg.n_envs * cfg.n_drones / (_sync(device) - t0)


def main():
    ap = argparse.ArgumentParser(description="CIDA-Swarm throughput benchmark")
    ap.add_argument("--n-envs", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--extra", default="", help="дополнительные флаги train, например '--compile'")
    args = ap.parse_args()

    rows = []
    for n_envs in args.n_envs:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        train_args = parse_args([
            "--n-envs", str(n_envs), "--iters", str(args.iters), "--log-every", "1000",
            "--eval-every", "1000000", "--eval-envs", "256", "--out", f"runs/_bench_{n_envs}",
            *args.extra.split(),
        ])
        try:
            only_env = env_speed(train_args)
            result = train(train_args)
        except torch.cuda.OutOfMemoryError:
            rows.append((n_envs, None, None, None, None))
            print(f"[bench] {n_envs}: не помещается в память", flush=True)
            continue
        timing = result["timing_seconds"]
        collect = timing["collect"] / max(sum(timing.values()), 1e-9)
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else float("nan")
        rows.append((n_envs, result["agent_steps_per_s"], only_env, collect, mem))
        print(f"[bench] {n_envs} worlds: train {result['agent_steps_per_s']:,.0f} agent-steps/s, "
              f"env only {only_env:,.0f}, collect {collect:.0%} of time, peak memory {mem:.1f} GB", flush=True)

    print("\n| Worlds | Train agent-steps/s | Env only | Collect share | Peak memory, GB |\n|---|---|---|---|---|")
    for n_envs, speed, only_env, collect, mem in rows:
        if speed is None:
            print(f"| {n_envs} | OOM | — | — | — |")
        else:
            print(f"| {n_envs} | {speed:,.0f} | {only_env:,.0f} | {collect:.0%} | {mem:.1f} |")


if __name__ == "__main__":
    main()
