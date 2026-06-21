"""
exp_v7_behavioral_audit.py
==========================
CIDA как инфраструктурный слой аудита (XAI).

Что делает:
  1. Строит BehavioralAuditVector из внутренней динамики CIDA:
     - conflict_score   : рост разногласия Прокурор vs Защитник по раундам
     - skeptic_entropy  : траектория сомнения Скептика (энтропия b)
     - consensus_speed  : скорость схождения (через disagreement decay)
     - volatility       : суммарная волатильность убеждений
  2. Прогоняет 4 сценария: Normal / Hard (boundary) / Adversarial / OOD
  3. Логирует все метрики в CSV + stdout
  4. Строит и сохраняет 5 графиков в experiments/results_v7/

Запуск:
    python experiments/exp_v7_behavioral_audit.py
    python experiments/exp_v7_behavioral_audit.py --quick
"""
from __future__ import annotations
import argparse, os, sys, csv, math, time, logging
from dataclasses import dataclass, field
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cida_plugin import (
    CIDAPlugin, CIDAPluginConfig, CIDALoss, DebateDiagnostics
)
from cida_plugin.deliberation import PerspectiveProjector

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("v7_audit")

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES = 14
D_INPUT     = 128
D_HIDDEN    = 128
NUM_AGENTS  = 4
ROLE_NAMES  = ["Prosecutor", "Defender", "Skeptic", "Integrator"]


# ─────────────────────────────── Dataset ──────────────────────────────────────
class AuditDataset(Dataset):
    """
    Синтетический датасет с 4 типами примеров:
      normal      : чёткие метки, малый шум
      hard        : граничный случай, противоречивые co-occurrence
      adversarial : subtle perturbation (~FGSM-like noise pattern)
      ood         : не похожие ни на что (uniform noise)
    """
    TYPES = ["normal", "hard", "adversarial", "ood"]

    def __init__(self, n: int, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        per = n // 4
        feats, labels, types = [], [], []

        for t_idx, t_name in enumerate(self.TYPES):
            for _ in range(per):
                y = torch.zeros(NUM_CLASSES)
                k = int(torch.randint(2, 5, (1,), generator=g).item())
                pos = torch.randperm(NUM_CLASSES, generator=g)[:k]
                y[pos] = 1.0

                if t_name == "normal":
                    x = torch.randn(D_INPUT, generator=g) * 0.5
                    x += y.sum() * 0.3
                elif t_name == "hard":
                    x = torch.randn(D_INPUT, generator=g) * 1.2
                    flip = torch.randperm(NUM_CLASSES, generator=g)[:3]
                    y[flip] = 1.0 - y[flip]
                elif t_name == "adversarial":
                    x = torch.randn(D_INPUT, generator=g) * 0.5
                    x += y.sum() * 0.3
                    adv = torch.sign(torch.randn(D_INPUT, generator=g)) * 0.4
                    x += adv
                elif t_name == "ood":
                    x = torch.rand(D_INPUT, generator=g) * 4.0 - 2.0

                feats.append(x)
                labels.append(y)
                types.append(t_name)

        self.features = torch.stack(feats)
        self.labels   = torch.stack(labels)
        self.types    = types

    def __len__(self): return len(self.types)
    def __getitem__(self, i):
        return self.features[i], self.labels[i], self.types[i]


# ─────────────────────────── BehavioralAuditVector ────────────────────────────
@dataclass
class BehavioralAuditVector:
    """Структурированный вектор поведения CIDA за один forward-pass."""
    conflict_score:   float  # рост разногласия Прокурор-Защитник
    skeptic_entropy:  float  # средняя энтропия убеждений Скептика
    consensus_speed:  float  # скорость убывания disagreement
    volatility:       float  # суммарная волатильность убеждений
    uncertainty:      float  # финальная uncertainty
    disagreement:     float  # финальное disagreement
    rounds_used:      int    # сколько раундов использовано
    p_max:            float  # максимальная уверенность
    danger_flag:      bool   # «красный флаг» — аномалия

    def to_dict(self) -> Dict:
        return {
            "conflict_score":  round(self.conflict_score,  4),
            "skeptic_entropy": round(self.skeptic_entropy, 4),
            "consensus_speed": round(self.consensus_speed, 4),
            "volatility":      round(self.volatility,      4),
            "uncertainty":     round(self.uncertainty,     4),
            "disagreement":    round(self.disagreement,    4),
            "rounds_used":     self.rounds_used,
            "p_max":           round(self.p_max,           4),
            "danger_flag":     int(self.danger_flag),
        }


def extract_behavioral_vectors(out: dict) -> List[BehavioralAuditVector]:
    """Извлекает список BehavioralAuditVector для всех примеров в батче."""
    b_all = out.get("b_all", [])
    u_all = out.get("u_all", [])
    p_final = out["p_final"]
    uncertainty = out["uncertainty"]
    disagreement = out["disagreement"]
    rounds_used = out.get("rounds_used", 0)

    B = p_final.size(0)
    T = len(b_all)
    M = b_all[0].size(1) if T > 0 else 0
    K = b_all[0].size(2) if T > 0 else 0

    conflict_scores = torch.zeros(B, device=p_final.device)
    skeptic_entropies = torch.zeros(B, device=p_final.device)
    consensus_speeds = torch.zeros(B, device=p_final.device)
    volatilities = torch.zeros(B, device=p_final.device)

    if T >= 2:
        b_stacked = torch.stack(b_all, dim=0)
        dists = (b_stacked[:, :, 0, :] - b_stacked[:, :, 1, :]).abs().mean(dim=-1)
        conflict_scores = dists.max(dim=0).values - dists[0]

    if T > 0 and M > 2:
        b_stacked = torch.stack(b_all, dim=0)
        b_sk = b_stacked[:, :, 2, :].clamp(1e-6, 1.0 - 1e-6)
        ent = -(b_sk * b_sk.log() + (1.0 - b_sk) * (1.0 - b_sk).log()).mean(dim=-1)
        skeptic_entropies = ent.mean(dim=0)

    if len(u_all) >= 2:
        u_stacked = torch.stack(u_all, dim=0).squeeze(-1)
        u_mean = u_stacked.mean(dim=-1)
        consensus_speeds = u_mean[0] - u_mean[-1]

    if T >= 2:
        b_stacked = torch.stack(b_all, dim=0)
        diffs = (b_stacked[1:] - b_stacked[:-1]).abs().mean(dim=(2, 3))
        volatilities = diffs.sum(dim=0)

    vectors = []
    for i in range(B):
        cs = conflict_scores[i].item()
        se = skeptic_entropies[i].item()
        csp = consensus_speeds[i].item()
        vol = volatilities[i].item()
        unc = uncertainty[i].item()
        dis = disagreement[i].item()

        if isinstance(rounds_used, torch.Tensor):
            ru = int(rounds_used[i].item())
        else:
            ru = int(rounds_used)

        p_m = p_final[i].max().item()

        danger_flag = (
            cs > 0.001
            or vol > 0.005
            or csp < -0.01
            or unc > 0.33
        )

        vectors.append(BehavioralAuditVector(
            conflict_score=cs,
            skeptic_entropy=se,
            consensus_speed=csp,
            volatility=vol,
            uncertainty=unc,
            disagreement=dis,
            rounds_used=ru,
            p_max=p_m,
            danger_flag=danger_flag
        ))
    return vectors


def extract_behavioral_vector(out: dict) -> BehavioralAuditVector:
    """Извлекает BehavioralAuditVector для первого примера в батче."""
    return extract_behavioral_vectors(out)[0]


# ─────────────────────────────── Models ───────────────────────────────────────
class CIDAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D_INPUT, D_HIDDEN),
            nn.LayerNorm(D_HIDDEN),
            nn.SiLU(),
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

        cfg = CIDAPluginConfig(
            d_input=D_HIDDEN, d_hidden=D_HIDDEN,
            num_classes=NUM_CLASSES, num_agents=NUM_AGENTS,
            max_rounds=4, multi_label=True,
            use_perspective_projector=True,
            early_stop_threshold=None,
        )
        self.plugin = CIDAPlugin(cfg)

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        B = h.size(0)
        seq = h.unsqueeze(1).expand(B, 12, D_HIDDEN) + \
              torch.zeros(B, 12, D_HIDDEN, device=x.device)
        return self.plugin(h, seq_output=seq)


class LinearBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D_INPUT, NUM_CLASSES)

    def forward(self, x):
        p = torch.sigmoid(self.fc(x))
        return {"p_final": p, "uncertainty": torch.zeros(x.size(0), device=x.device)}


# ──────────────────────────── Training utilities ──────────────────────────────
def train_epoch(model, loader, opt, loss_fn, device, is_cida=True):
    model.train()
    total_loss, rounds = 0.0, []
    for feat, y, _ in loader:
        feat, y = feat.to(device), y.to(device)
        opt.zero_grad()
        out = model(feat)
        b_all = out.get("b_all", []) if is_cida else []
        loss, _ = loss_fn(out["p_final"], y, b_all)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        rounds.append(out.get("rounds_used", 0))
    return total_loss / len(loader), (sum(rounds)/len(rounds) if rounds else 0)


@torch.no_grad()
def evaluate(model, loader, device, is_cida=True):
    model.eval()
    correct, total = 0, 0
    ece_sum = 0.0
    audit_by_type: Dict[str, List[BehavioralAuditVector]] = {
        t: [] for t in AuditDataset.TYPES
    }
    for feat, y, types in loader:
        feat, y = feat.to(device), y.to(device)
        out = model(feat)
        probs = out["p_final"]
        pred = (probs > 0.5).float()
        correct += (pred == y).float().sum().item()
        total   += y.numel()
        if is_cida:
            ece_sum += DebateDiagnostics.expected_calibration_error(probs, y)
            bavs = extract_behavioral_vectors(out)
            for i, t in enumerate(types):
                audit_by_type[t].append(bavs[i])
    acc = correct / max(total, 1)
    ece = ece_sum / len(loader) if is_cida else 0.0
    return acc, ece, audit_by_type


# ─────────────────────────────── Plots ────────────────────────────────────────
def build_plots(history: dict, type_vectors: Dict[str, List[BehavioralAuditVector]],
                out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return

    os.makedirs(out_dir, exist_ok=True)
    colors = {"CIDA": "#5E81F4", "Baseline": "#F4A85E"}
    ep_arr = list(range(1, len(history["cida_acc"]) + 1))

    # ── 1. Accuracy & ECE curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Dynamics", fontsize=14)
    ax = axes[0]
    ax.plot(ep_arr, history["cida_acc"],  label="CIDA",     color=colors["CIDA"])
    ax.plot(ep_arr, history["base_acc"],  label="Baseline", color=colors["Baseline"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Elem Accuracy"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Accuracy per Epoch")
    ax = axes[1]
    ax.plot(ep_arr, history["cida_ece"], label="CIDA ECE", color=colors["CIDA"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("ECE ↓"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Calibration (ECE)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_training_dynamics.png"), dpi=120)
    plt.close()
    log.info("Plot saved: 01_training_dynamics.png")

    # ── 2. Behavioral Vector Heatmap по типам примеров ────────────────────────
    metrics = ["conflict_score", "skeptic_entropy", "consensus_speed",
               "volatility", "uncertainty", "disagreement"]
    type_order = AuditDataset.TYPES
    mat = np.zeros((len(metrics), len(type_order)))
    for j, t in enumerate(type_order):
        vecs = type_vectors.get(t, [])
        if not vecs: continue
        for i, m in enumerate(metrics):
            mat[i, j] = np.mean([getattr(v, m) for v in vecs])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(type_order))); ax.set_xticklabels(type_order, fontsize=11)
    ax.set_yticks(range(len(metrics)));  ax.set_yticklabels(metrics, fontsize=10)
    for i in range(len(metrics)):
        for j in range(len(type_order)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title("Behavioral Audit Vector — Mean per Sample Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_behavioral_heatmap.png"), dpi=120)
    plt.close()
    log.info("Plot saved: 02_behavioral_heatmap.png")

    # ── 3. Danger Flag Rate per type ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    rates = []
    for t in type_order:
        vecs = type_vectors.get(t, [])
        r = np.mean([v.danger_flag for v in vecs]) if vecs else 0
        rates.append(r)
    bar_colors = ["#5E81F4", "#F4A85E", "#F45E5E", "#A85EF4"]
    ax.bar(type_order, rates, color=bar_colors)
    ax.set_ylabel("Danger Flag Rate"); ax.set_title("🚨 Danger Flag Rate by Sample Type")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_danger_flag_rate.png"), dpi=120)
    plt.close()
    log.info("Plot saved: 03_danger_flag_rate.png")

    # ── 4. Conflict Score & Volatility radar ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric_name in zip(axes, ["conflict_score", "volatility"]):
        vals = []
        for t in type_order:
            vecs = type_vectors.get(t, [])
            vals.append(np.mean([getattr(v, metric_name) for v in vecs]) if vecs else 0)
        ax.bar(type_order, vals, color=bar_colors)
        ax.set_title(metric_name.replace("_", " ").title())
        ax.set_ylabel("Score"); ax.grid(axis="y", alpha=.3)
    fig.suptitle("Prosecutor-Defender Conflict & Belief Volatility", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_conflict_volatility.png"), dpi=120)
    plt.close()
    log.info("Plot saved: 04_conflict_volatility.png")

    # ── 5. Rounds used per type ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    rounds_per_type = []
    for t in type_order:
        vecs = type_vectors.get(t, [])
        rounds_per_type.append(np.mean([v.rounds_used for v in vecs]) if vecs else 0)
    ax.bar(type_order, rounds_per_type, color=bar_colors)
    ax.set_ylabel("Avg Rounds Used")
    ax.set_title("Deliberation Depth: Avg Rounds by Sample Type")
    ax.grid(axis="y", alpha=.3)
    for i, r in enumerate(rounds_per_type):
        ax.text(i, r + 0.05, f"{r:.2f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_rounds_per_type.png"), dpi=120)
    plt.close()
    log.info("Plot saved: 05_rounds_per_type.png")

    log.info(f"All plots saved to: {out_dir}")


# ─────────────────────────── CSV logging ──────────────────────────────────────
def save_csv(rows: List[dict], path: str):
    if not rows: return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"CSV saved: {path}")


# ──────────────────────────────── Main ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CIDA v7 Behavioral Audit")
    parser.add_argument("--quick", action="store_true",
                        help="Маленький датасет + 2 эпохи (быстрая проверка)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    n_total = 400 if args.quick else 2000
    epochs  = 2   if args.quick else 6
    bs      = 32

    out_dir = os.path.join(ROOT, "experiments", "results_v7")
    os.makedirs(out_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info(f"Building dataset: {n_total} samples ({n_total//4} per type)")
    train_ds = AuditDataset(n_total, seed=42)
    val_ds   = AuditDataset(n_total // 4, seed=99)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs)

    # ── Models ────────────────────────────────────────────────────────────────
    cida     = CIDAModel().to(device)
    baseline = LinearBaseline().to(device)
    loss_fn  = CIDALoss(multi_label=True, lambda_cal=0.4, lambda_ac=0.15)

    opt_cida = torch.optim.AdamW(
        cida.plugin.parameters(),
        lr=2e-3, weight_decay=1e-4
    )
    opt_base = torch.optim.AdamW(baseline.parameters(), lr=2e-3)

    n_cida = sum(p.numel() for p in cida.parameters() if p.requires_grad)
    n_base = sum(p.numel() for p in baseline.parameters())
    log.info(f"CIDA params: {n_cida:,} | Baseline params: {n_base:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {k: [] for k in ["cida_acc","cida_ece","base_acc","cida_rounds"]}
    epoch_rows = []
    last_type_vectors = {}

    log.info("\n" + "="*60)
    log.info("TRAINING CIDA v7 — Behavioral Audit Layer")
    log.info("="*60)

    for ep in range(1, epochs+1):
        t0 = time.time()

        c_loss, c_rounds = train_epoch(cida,     train_loader, opt_cida, loss_fn, device, True)
        b_loss, _        = train_epoch(baseline, train_loader, opt_base, loss_fn, device, False)

        c_acc, c_ece, type_vecs = evaluate(cida,     val_loader, device, True)
        b_acc, _,    _          = evaluate(baseline, val_loader, device, False)

        elapsed = time.time() - t0
        last_type_vectors = type_vecs

        history["cida_acc"].append(c_acc)
        history["cida_ece"].append(c_ece)
        history["base_acc"].append(b_acc)
        history["cida_rounds"].append(c_rounds)

        log.info(
            f"Ep {ep:02d}/{epochs} | "
            f"CIDA loss={c_loss:.4f} acc={c_acc:.3f} ece={c_ece:.4f} "
            f"rounds={c_rounds:.2f} | "
            f"Base loss={b_loss:.4f} acc={b_acc:.3f} | "
            f"t={elapsed:.1f}s"
        )

        row = {"epoch": ep,
               "cida_loss": round(c_loss,4), "cida_acc": round(c_acc,4),
               "cida_ece": round(c_ece,4),   "cida_rounds": round(c_rounds,2),
               "base_loss": round(b_loss,4), "base_acc": round(b_acc,4)}
        epoch_rows.append(row)

    # ── Per-type summary ──────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("BEHAVIORAL AUDIT REPORT — Per Sample Type")
    log.info("="*60)
    audit_rows = []
    for t in AuditDataset.TYPES:
        vecs = last_type_vectors.get(t, [])
        if not vecs:
            continue
        avg = {k: round(sum(getattr(v,k) for v in vecs)/len(vecs), 4)
               for k in ["conflict_score","skeptic_entropy","consensus_speed",
                         "volatility","uncertainty","disagreement","rounds_used","p_max"]}
        danger_rate = sum(v.danger_flag for v in vecs) / len(vecs)
        log.info(
            f"\n  [{t.upper()}]\n"
            f"    conflict_score  : {avg['conflict_score']}\n"
            f"    skeptic_entropy : {avg['skeptic_entropy']}\n"
            f"    consensus_speed : {avg['consensus_speed']}\n"
            f"    volatility      : {avg['volatility']}\n"
            f"    uncertainty     : {avg['uncertainty']}\n"
            f"    disagreement    : {avg['disagreement']}\n"
            f"    rounds_used     : {avg['rounds_used']:.2f}\n"
            f"    p_max           : {avg['p_max']}\n"
            f"    danger_flag_rate: {danger_rate:.2%}"
        )
        row = {"sample_type": t, "n": len(vecs), "danger_rate": round(danger_rate,4), **avg}
        audit_rows.append(row)

    # ── Single-sample inspection ───────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("SINGLE SAMPLE INSPECTION (one per type)")
    log.info("="*60)
    cida.eval()
    for t_name in AuditDataset.TYPES:
        idx = next(i for i, t in enumerate(val_ds.types) if t == t_name)
        feat = val_ds.features[idx].unsqueeze(0).to(device)
        with torch.no_grad():
            out = cida(feat)
        bav = extract_behavioral_vector(out)
        b_all = out.get("b_all", [])
        div   = DebateDiagnostics.belief_diversity_curve(b_all)
        pm    = DebateDiagnostics.persuasion_matrix(b_all)
        log.info(
            f"\n  [{t_name.upper()}]\n"
            f"    BehavioralAuditVector: {bav.to_dict()}\n"
            f"    Belief diversity curve: {[round(d,4) for d in div]}\n"
            f"    Persuasion matrix shape: {pm.shape if pm is not None else None}"
        )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_csv(epoch_rows, os.path.join(out_dir, "epoch_metrics.csv"))
    save_csv(audit_rows, os.path.join(out_dir, "audit_by_type.csv"))

    # ── Build plots ───────────────────────────────────────────────────────────
    build_plots(history, last_type_vectors, out_dir)

    log.info("\n✅ Experiment v7 complete. Results in: " + out_dir)


if __name__ == "__main__":
    main()
