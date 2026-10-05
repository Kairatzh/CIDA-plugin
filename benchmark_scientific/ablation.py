"""
benchmark_scientific/ablation.py — Systematic Ablation Study: Agents × Rounds

Investigates:
  1. Agent Count Ablation (1, 2, 4 agents)
  2. Deliberation Rounds Ablation (1, 2, 3, 4 rounds)
  3. Optimal Deliberation Budget Discovery
"""

import time
from typing import List, Dict, Any
import torch

from cida_plugin.transformer_agent import SOTADeliberationCore, UniversalPlanProjector


def run_deliberation_ablation(
    sample_latents: List[torch.Tensor],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates convergence rate and latent dynamics across varying agents and rounds.
    """
    results = {
        "agents_ablation": {},
        "rounds_ablation": {}
    }

    # 1. Rounds Ablation on standard 4-agent core
    core_4a = SOTADeliberationCore(d_input=4096, d_hidden=1024, num_agents=4, num_rounds=4).to(device)
    core_4a.eval()

    round_disagreements = {1: [], 2: [], 3: [], 4: []}
    for h in sample_latents:
        with torch.no_grad():
            out = core_4a(h, min_rounds=4, max_rounds=4)
            traj = out["disagreement_trajectory"]
            for r_idx in range(min(4, len(traj))):
                round_disagreements[r_idx + 1].append(traj[r_idx])

    for r_num, vals in round_disagreements.items():
        avg_d = sum(vals) / max(len(vals), 1)
        results["rounds_ablation"][f"round_{r_num}"] = {
            "round": r_num,
            "mean_disagreement": avg_d,
            "marginal_reduction": (round_disagreements[1][0] - avg_d) if round_disagreements[1] else 0.0
        }

    # 2. Agents Count Ablation (1, 2, 4 agents)
    for n_agents in [1, 2, 4]:
        core_na = SOTADeliberationCore(d_input=4096, d_hidden=1024, num_agents=n_agents, num_rounds=4).to(device)
        core_na.eval()
        latencies = []
        for h in sample_latents:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = core_na(h, min_rounds=2, max_rounds=4)
            latencies.append(time.perf_counter() - t0)

        results["agents_ablation"][f"agents_{n_agents}"] = {
            "num_agents": n_agents,
            "avg_delib_latency_ms": (sum(latencies) / max(len(latencies), 1)) * 1000.0,
            "parameter_overhead_m": sum(p.numel() for p in core_na.parameters()) / 1e6
        }

    return results
