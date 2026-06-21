"""
Эксперимент v6: проверка улучшений на синтетическом multi-label датасете
в стиле CheXpert (14 патологий, неоднозначные случаи).

CIDA особенно полезен когда:
  - несколько независимых меток (multi-label)
  - агенты с role priors спорят на пограничных случаях
  - uncertainty коррелирует с ошибкой

Запуск:
    python experiments/exp_v6_improvements.py
    python experiments/exp_v6_improvements.py --quick
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cida_plugin import CIDAPlugin, CIDAPluginConfig, CIDALoss, DebateDiagnostics
from cida_plugin.deliberation import PerspectiveProjector

NUM_CLASSES = 14
D_INPUT = 128
D_HIDDEN = 128
NUM_AGENTS = 4


class SyntheticCheXpertDataset(Dataset):
    """
    Синтетические рентген-подобные признаки с 14 multi-label метками.
    easy: чёткие метки, hard: шум + противоречивые co-occurrence.
    """

    def __init__(self, n: int, hard_fraction: float = 0.35, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.n = n
        self.hard = torch.rand(n, generator=g) < hard_fraction
        self.labels = torch.zeros(n, NUM_CLASSES)
        self.features = torch.randn(n, D_INPUT, generator=g)

        for i in range(n):
            k = int(torch.randint(2, 6, (1,), generator=g).item())
            pos = torch.randperm(NUM_CLASSES, generator=g)[:k]
            self.labels[i, pos] = 1.0
            self.features[i] += 0.8 * self.labels[i].sum() * 0.1
            if self.hard[i]:
                self.features[i] += torch.randn(D_INPUT, generator=g) * 1.5
                flip = torch.randperm(NUM_CLASSES, generator=g)[:2]
                self.labels[i, flip] = 1.0 - self.labels[i, flip]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], bool(self.hard[idx])


class EncoderWithPerspectiveCIDA(nn.Module):
    """Замороженный линейный энкодер + PerspectiveProjector + CIDA multi-label."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D_INPUT, D_INPUT),
            nn.LayerNorm(D_INPUT),
            nn.SiLU(),
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.perspective = PerspectiveProjector(NUM_AGENTS, D_INPUT, D_HIDDEN)
        self.seq_tokens = nn.Parameter(torch.randn(1, 12, D_HIDDEN) * 0.02)

        cfg = CIDAPluginConfig(
            d_input=D_HIDDEN,
            d_hidden=D_HIDDEN,
            num_classes=NUM_CLASSES,
            num_agents=NUM_AGENTS,
            max_rounds=4,
            multi_label=True,
            use_convergence_stop=True,
            convergence_epsilon=5e-4,
            early_stop_threshold=None,
        )
        self.plugin = CIDAPlugin(cfg)

    def forward(self, x):
        with torch.no_grad():
            enc = self.encoder(x)
        B = enc.size(0)
        agent_views = self.perspective(enc)
        pooled = agent_views.mean(dim=1)
        seq = pooled.unsqueeze(1).expand(B, 12, D_HIDDEN) + self.seq_tokens.expand(B, -1, -1) * 0.1
        return self.plugin(pooled, seq_output=seq)


class LinearBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D_INPUT, NUM_CLASSES)

    def forward(self, x):
        return {"p_final": torch.sigmoid(self.fc(x)), "uncertainty": torch.zeros(x.size(0))}


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    rounds = []
    for feat, y, _ in loader:
        feat, y = feat.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(feat)
        b_all = out.get("b_all", [])
        loss, _ = loss_fn(out["p_final"], y, b_all)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        rounds.append(out.get("rounds_used", 0))
    return total_loss / len(loader), sum(rounds) / len(rounds)


@torch.no_grad()
def evaluate(model, loader, device, use_cida_metrics=True):
    model.eval()
    correct = 0
    total = 0
    ece_sum = 0.0
    unc_errors = []
    rounds_easy, rounds_hard = [], []

    for feat, y, hard in loader:
        feat, y = feat.to(device), y.to(device)
        out = model(feat)

        probs = out["p_final"]
        pred = (probs > 0.5).float()
        correct += (pred == y).float().sum().item()
        total += y.numel()

        if use_cida_metrics:
            ece_sum += DebateDiagnostics.expected_calibration_error(probs, y)
            err = ((pred != y).float().mean(dim=-1))
            unc_errors.extend(
                zip(err.cpu().tolist(), out["uncertainty"].cpu().tolist())
            )
            for i, h in enumerate(hard):
                if h:
                    rounds_hard.append(out["rounds_used"])
                else:
                    rounds_easy.append(out["rounds_used"])

    metrics = {
        "f1_elem": correct / max(total, 1),
        "ece": ece_sum / len(loader) if use_cida_metrics else 0.0,
        "avg_rounds": sum(rounds_easy + rounds_hard) / max(len(rounds_easy) + len(rounds_hard), 1),
        "avg_rounds_easy": sum(rounds_easy) / max(len(rounds_easy), 1),
        "avg_rounds_hard": sum(rounds_hard) / max(len(rounds_hard), 1),
    }
    if unc_errors:
        errs, uncs = zip(*unc_errors)
        import numpy as np
        metrics["unc_error_corr"] = float(np.corrcoef(errs, uncs)[0, 1])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train, n_val = (800, 200) if args.quick else (20000, 5000)
    epochs = 2 if args.quick else 6

    train_ds = SyntheticCheXpertDataset(n_train, hard_fraction=0.4, seed=42)
    val_ds = SyntheticCheXpertDataset(n_val, hard_fraction=0.4, seed=99)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    print(f"Device: {device} | train={n_train} val={n_val} | {NUM_CLASSES}-label CheXpert-style")

    cida = EncoderWithPerspectiveCIDA().to(device)
    baseline = LinearBaseline().to(device)
    loss_fn = CIDALoss(multi_label=True, lambda_cal=0.4, lambda_ac=0.15)

    opt_cida = torch.optim.Adam(
        list(cida.perspective.parameters()) + list(cida.plugin.parameters()),
        lr=2e-3,
    )
    opt_base = torch.optim.Adam(baseline.parameters(), lr=2e-3)

    print("\n=== CIDA v6 (multi-label + improvements) ===")
    for ep in range(epochs):
        loss, avg_r = train_epoch(cida, train_loader, opt_cida, loss_fn, device)
        m = evaluate(cida, val_loader, device)
        print(
            f"  ep {ep+1}/{epochs} loss={loss:.4f} F1_elem={m['f1_elem']:.3f} "
            f"ECE={m['ece']:.4f} rounds={m['avg_rounds']:.2f} "
            f"(easy={m['avg_rounds_easy']:.2f} hard={m['avg_rounds_hard']:.2f}) "
            f"unc_err_corr={m.get('unc_error_corr', 0):.3f}"
        )

    print("\n=== Linear baseline ===")
    for ep in range(epochs):
        loss, _ = train_epoch(baseline, train_loader, opt_base, loss_fn, device)
        m = evaluate(baseline, val_loader, device, use_cida_metrics=False)
        print(f"  ep {ep+1}/{epochs} loss={loss:.4f} F1_elem={m['f1_elem']:.3f}")

    cida.eval()
    sample = val_ds[0]
    feat = sample[0].unsqueeze(0).to(device)
    with torch.no_grad():
        out = cida(feat)
    print("\n=== Диагностика одного hard-примера ===")
    print(f"  rounds_used: {out['rounds_used']}")
    print(f"  disagreement: {out['disagreement'].item():.4f}")
    print(f"  uncertainty: {out['uncertainty'].item():.4f}")
    if out["b_all"]:
        print(f"  belief_volatility: {DebateDiagnostics.belief_volatility(out['b_all']):.4f}")
    print("Готово.")


if __name__ == "__main__":
    main()
