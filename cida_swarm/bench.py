"""
bench.py — подбор числа параллельных миров под конкретную видеокарту.

    python -m cida_swarm.bench --n-envs 2048 4096 8192 16384

Для каждого размера меряет скорость полного цикла обучения (сбор опыта + PPO) и пиковую память.
Берите размер, после которого скорость почти перестаёт расти, а память с запасом помещается.
"""

from __future__ import annotations

import argparse

import torch

from .train import parse_args, train


def main():
    ap = argparse.ArgumentParser(description="CIDA-Swarm throughput benchmark")
    ap.add_argument("--n-envs", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--extra", default="", help="дополнительные флаги train, например '--share-coverage'")
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
            speed = train(train_args)["agent_steps_per_s"]
        except torch.cuda.OutOfMemoryError:
            rows.append((n_envs, None, None))
            print(f"[bench] {n_envs}: не помещается в память", flush=True)
            continue
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else float("nan")
        rows.append((n_envs, speed, mem))
        print(f"[bench] {n_envs} worlds: {speed:,.0f} agent-steps/s, peak memory {mem:.1f} GB", flush=True)

    print("\n| Worlds | Agent-steps/s | Peak memory, GB |\n|---|---|---|")
    for n_envs, speed, mem in rows:
        print(f"| {n_envs} | {'OOM' if speed is None else f'{speed:,.0f}'} | {'—' if mem is None else f'{mem:.1f}'} |")


if __name__ == "__main__":
    main()
