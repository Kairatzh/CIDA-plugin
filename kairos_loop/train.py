"""
train.py — обучение и оценка CIDA-Loop против plain-цикла на судоку.

Пример (Colab A100):
    python -m kairos_loop.train --variant cida  --out runs/cida_s0
    python -m kairos_loop.train --variant plain --out runs/plain_s0
    python -m kairos_loop.train --variant plain --n-latent 24 --out runs/plain_compute_s0
    python -m kairos_loop.report runs/*/result.json

Все метрики считаются на ОТЛОЖЕННЫХ головоломках, которых не было в обучении.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import auroc, expected_calibration_error
from .model import KairosLoop, LoopConfig, count_params
from .sudoku import batches, generate_dataset, load_sudoku_extreme


# ── данные ───────────────────────────────────────────────────────────────────
def load_data(args):
    if args.data == "extreme":
        train = load_sudoku_extreme("train", args.n_train, seed=args.data_seed)
        test = load_sudoku_extreme("test", args.n_test, seed=args.data_seed)
    else:
        train = generate_dataset(args.n_train, seed=args.data_seed, min_clues=args.min_clues)
        test = generate_dataset(args.n_test, seed=args.data_seed + 10_000, min_clues=args.min_clues)
    return train, test


# ── EMA весов (стабилизирует рекуррентные модели, как в TRM) ─────────────────
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1.0 - self.decay)


def _autocast(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


# ── оценка ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: KairosLoop, puzzles: np.ndarray, solutions: np.ndarray,
             device: torch.device, batch_size: int = 256) -> Dict[str, float]:
    model.eval()
    n_sup = model.cfg.n_sup
    correct, halt_p, agree, cell_acc = [], [], [], []
    for i in range(0, len(puzzles), batch_size):
        tokens = torch.as_tensor(puzzles[i:i + batch_size], device=device)
        target = torch.as_tensor(solutions[i:i + batch_size], device=device) - 1
        state = model.initial_state(len(tokens))
        c_steps, h_steps, a_steps = [], [], []
        with _autocast(device):
            for _ in range(n_sup):
                logits, halt_logit, agreement, state = model(tokens, state)
                pred = logits.argmax(-1)
                c_steps.append((pred == target).all(-1))
                h_steps.append(torch.sigmoid(halt_logit.float()))
                if agreement is not None:
                    a_steps.append(agreement.float())
        cell_acc.append((pred == target).float().mean(-1))
        correct.append(torch.stack(c_steps, 1).cpu())
        halt_p.append(torch.stack(h_steps, 1).cpu())
        if a_steps:
            agree.append(torch.stack(a_steps, 1).cpu())

    correct = torch.cat(correct).numpy()   # (N, n_sup)
    halt_p = torch.cat(halt_p).numpy()
    final_ok = correct[:, -1]

    # адаптивная остановка: первый шаг, где модель сама решила «готово»
    halted = halt_p > 0.5
    stop = np.where(halted.any(1), halted.argmax(1), n_sup - 1)
    adaptive_ok = correct[np.arange(len(stop)), stop]

    out = {
        "cell_acc": float(torch.cat(cell_acc).mean()),
        "exact_acc": float(final_ok.mean()),
        "exact_acc_adaptive": float(adaptive_ok.mean()),
        "mean_steps_adaptive": float(stop.mean() + 1),
        "exact_acc_by_step": [float(v) for v in correct.mean(0)],
        "halt_ece": expected_calibration_error(halt_p[:, -1], final_ok),
        "halt_auroc": auroc(halt_p[:, -1], final_ok),
    }
    if agree:
        agree = torch.cat(agree).numpy()
        out["agreement_mean"] = float(agree[:, -1].mean())
        out["agreement_auroc"] = auroc(agree[:, -1], final_ok)
    model.train()
    return out


# ── обучение ─────────────────────────────────────────────────────────────────
def build_config(args) -> LoopConfig:
    return LoopConfig(
        variant=args.variant, d_model=args.d_model, n_layers=args.n_layers, mixer=args.mixer,
        n_heads=args.n_heads, n_latent=args.n_latent, n_cycles=args.n_cycles, n_sup=args.n_sup,
        n_agents=args.n_agents, d_comm=args.d_comm, communicate=not args.no_comm,
        disagreement_bias=not args.no_bias, role_ortho=args.role_ortho,
    )


def train(args) -> Dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu" and args.device is None:
        print(f"[kairos-loop] WARNING: CUDA недоступна в {sys.executable} (torch {torch.__version__}) — "
              "обучение пойдёт на CPU и будет очень медленным. Запускайте тем же Python, что и ядро, "
              "или передайте --device cpu, если это намеренно.", flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    (train_p, train_s), (test_p, test_s) = load_data(args)
    cfg = build_config(args)
    model = KairosLoop(cfg).to(device)
    n_params = count_params(model)
    ema = EMA(model, args.ema_decay)
    if args.compile:
        print("[kairos-loop] torch.compile: первые итерации займут несколько минут на компиляцию", flush=True)
        model.net = torch.compile(model.net)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    print(f"[kairos-loop] variant={cfg.variant} params={n_params:,} device={device} "
          f"train={len(train_p)} test={len(test_p)}", flush=True)

    stream = batches(train_p, train_s, args.batch_size, rng, use_augment=not args.no_augment)
    history, opt_steps, t0 = [], 0, time.time()
    log = {"loss": 0.0, "exact": 0.0, "steps": 0.0, "n": 0}

    for it in range(1, args.iters + 1):
        p, s = next(stream)
        tokens = torch.as_tensor(p, device=device)
        target = torch.as_tensor(s, device=device) - 1
        state = model.initial_state(len(tokens))
        active = torch.ones(len(tokens), dtype=torch.bool, device=device)
        for sup in range(cfg.n_sup):
            lr = args.lr * min(1.0, (opt_steps + 1) / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            with _autocast(device):
                logits, halt_logit, _, state = model(tokens, state)
                ce = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none").mean(-1)
                solved = (logits.argmax(-1) == target).all(-1)
                bce = F.binary_cross_entropy_with_logits(halt_logit.float(), solved.float(), reduction="none")
                w = active.float()
                loss = ((ce + 0.5 * bce) * w).sum() / w.sum().clamp_min(1.0) + model.aux_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            opt_steps += 1
            active = active & ~((halt_logit.detach() > 0) & solved)
            if not bool(active.any()):  # одна синхронизация на шаг супервизии
                break
        log["loss"] += loss.item()
        log["exact"] += float(solved.float().mean())
        log["steps"] += sup + 1
        log["n"] += 1

        if it % args.log_every == 0:
            n = log["n"]
            print(f"it {it:6d} | loss {log['loss'] / n:.4f} | train_exact {log['exact'] / n:.3f} "
                  f"| sup_steps {log['steps'] / n:.1f} | {it * args.batch_size / (time.time() - t0):.0f} puzzles/s", flush=True)
            log = {"loss": 0.0, "exact": 0.0, "steps": 0.0, "n": 0}

        if it % args.eval_every == 0 or it == args.iters:
            metrics = evaluate(ema.model, test_p, test_s, device, args.eval_batch_size)
            metrics.update(iter=it, minutes=(time.time() - t0) / 60)
            history.append(metrics)
            print(f"[eval] it {it} | exact {metrics['exact_acc']:.4f} | adaptive {metrics['exact_acc_adaptive']:.4f} "
                  f"@ {metrics['mean_steps_adaptive']:.1f} steps | cell {metrics['cell_acc']:.4f} "
                  f"| halt AUROC {metrics['halt_auroc']:.3f}", flush=True)
            torch.save({"config": cfg.to_dict(), "ema_state": ema.model.state_dict()}, out_dir / "model.pt")

    result = {
        "name": args.name or out_dir.name,
        "config": cfg.to_dict(),
        "args": vars(args),
        "params": n_params,
        "train_minutes": (time.time() - t0) / 60,
        "final": history[-1],
        "history": history,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def parse_args(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description="Kairos Stage 0: CIDA-Loop vs plain loop on Sudoku")
    ap.add_argument("--variant", choices=["cida", "plain"], default="cida")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="runs/default")
    # данные
    ap.add_argument("--data", choices=["extreme", "generated"], default="extreme")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--min-clues", type=int, default=22)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    # модель
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--mixer", choices=["mlp", "attn"], default="mlp")
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-latent", type=int, default=6)
    ap.add_argument("--n-cycles", type=int, default=3)
    ap.add_argument("--n-sup", type=int, default=16)
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--d-comm", type=int, default=64)
    ap.add_argument("--no-comm", action="store_true")
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--role-ortho", type=float, default=0.1)
    # оптимизация
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2000)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
