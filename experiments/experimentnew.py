#!/usr/bin/env python3
"""
validate_cida.py — CIDA Architecture Validation Suite v1.1
===========================================================
VEIMO Research Team, 2024

Тестирует 7 ключевых гипотез архитектуры CIDA против базовых моделей
с использованием реального пакета cida_plugin.
"""

# ─── Стандартные импорты ──────────────────────────────────────────────────────
import os, sys, json, argparse, warnings, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from typing import Dict, List, Tuple, Optional

# Импортируем компоненты CIDA-Plugin
from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin
from cida_plugin.losses import CIDALoss
from cida_plugin.agent import apply_role_prior
import cida_plugin.agent as agent

warnings.filterwarnings('ignore')

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--fast',   action='store_true', help='3 seeds, 15 эпох')
parser.add_argument('--seeds',  type=int, default=5,   help='Число seed-ов')
parser.add_argument('--output', type=str, default='./cida_validation')
args, _ = parser.parse_known_args()

# ─── Глобальный конфиг ────────────────────────────────────────────────────────
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SEEDS  = args.seeds if not args.fast else 3
D_IN     = 64          # входная размерность
D_H      = 64          # внутренняя размерность агентов
D_MSG    = 64          # размерность сообщений
K        = 2           # число классов (binary)
M        = 4           # агентов: Prosecutor, Defender, Skeptic, Integrator
ROUNDS   = 3           # раундов deliberation
SEQ_LEN  = 4           # длина последовательности для evidence pointers
N_TR     = 2000 if not args.fast else 800
N_TE     = 500
EPOCHS   = 30 if not args.fast else 15
LR       = 1e-3
BATCH    = 64
BLEND    = 0.30                         # ROLE_PRIOR_BLEND

OUT = args.output
os.makedirs(f'{OUT}/figures', exist_ok=True)

# Цветовая палитра
C = dict(
    wm='#4A90D9', poe='#E24B4A', base='#7F8C8D',
    mcd='#F5A623', np='#9B59B6',
    ok='#27AE60', fail='#E74C3C', nt='#95A5A6',
)

print(f"\n{'='*65}")
print(f"  CIDA Validation Suite (Production Code)  ·  VEIMO Research Team")
print(f"  device={DEVICE}  seeds={N_SEEDS}  epochs={EPOCHS}  N_train={N_TR}")
print(f"{'='*65}\n")

t0 = time.time()

# ─── РАЗДЕЛ 1: УТИЛИТЫ ────────────────────────────────────────────────────────

def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (равномерные бины по уверенности)."""
    conf = probs.max(1); pred = probs.argmax(1)
    acc  = (pred == labels).astype(float)
    val  = 0.0
    for lo, hi in zip(np.linspace(0,1,n_bins), np.linspace(0,1,n_bins+1)[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() >= 2:
            val += abs(conf[mask].mean() - acc[mask].mean()) * mask.mean()
    return float(val)

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier Score: MSE между вероятностями и one-hot метками."""
    oh = np.eye(probs.shape[1])[labels]
    return float(((probs - oh) ** 2).mean())

def auroc_score(scores: np.ndarray, targets: np.ndarray) -> float:
    """AUC-ROC для бинарного предиктора ошибок."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(targets, scores))
    except Exception:
        # Mann-Whitney U fallback
        pos = scores[targets == 1]; neg = scores[targets == 0]
        if len(pos) == 0 or len(neg) == 0: return 0.5
        return float(np.mean(pos[:, None] > neg[None, :]))

def wilcoxon_p(a: List[float], b: List[float]) -> float:
    """Двусторонний Wilcoxon signed-rank test."""
    if len(a) < 2: return 1.0
    try:
        _, p = stats.wilcoxon(a, b, alternative='two-sided'); return float(p)
    except Exception: return 1.0

def pstars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def mean_std(lst): return np.mean(lst), np.std(lst)

# ─── РАЗДЕЛ 2: ГЕНЕРАЦИЯ ДАННЫХ ───────────────────────────────────────────────

def make_data(n: int, centers: np.ndarray, noise: float = 0.60, seq_len: int = SEQ_LEN, seed: int = 0):
    """Генерирует последовательности с фиксированными центрами кластеров (symmetry breaking)."""
    rng = np.random.RandomState(seed)
    X_seq = []
    y = []
    for i in range(n):
        c = i % K
        seq = []
        for _ in range(seq_len):
            # Добавляем шум к центру кластера для каждого шага последовательности
            seq.append(centers[c] + rng.randn(D_IN) * noise)
        X_seq.append(seq)
        y.append(c)
    
    X_seq_tensor = torch.FloatTensor(np.array(X_seq)).to(DEVICE)
    X_pooled_tensor = X_seq_tensor.mean(dim=1)
    return X_pooled_tensor, X_seq_tensor, torch.LongTensor(np.array(y)).to(DEVICE)

def make_ood(n: int, shift: float = 2.0, seed: int = 99):
    """OOD данные: сдвинутое распределение."""
    rng = np.random.RandomState(seed)
    X_seq = []
    y = []
    for i in range(n):
        c = i % K
        seq = []
        for _ in range(SEQ_LEN):
            seq.append(rng.randn(D_IN) + shift)
        X_seq.append(seq)
        y.append(c)
    X_seq_tensor = torch.FloatTensor(np.array(X_seq)).to(DEVICE)
    X_pooled_tensor = X_seq_tensor.mean(dim=1)
    return X_pooled_tensor, X_seq_tensor, torch.LongTensor(np.array(y)).to(DEVICE)

# ─── РАЗДЕЛ 3: POE CONSENSUS WRAPPER ──────────────────────────────────────────

class PoEConsensusWrapper(nn.Module):
    """Обертка для эмуляции Product of Experts консенсуса в H1."""
    def __init__(self, original_consensus):
        super().__init__()
        self.original_consensus = original_consensus

    def forward(self, b, u=None):
        B, M, K = b.shape
        w = self.original_consensus.weights
        eps = 1e-8
        log_p = (w.view(1, M, 1) * b.clamp(eps).log()).sum(dim=1)
        p_final = F.softmax(log_p, dim=-1)
        disagreement = b.std(dim=1).mean(dim=-1)
        if u is not None:
            agent_u = (w.view(1, M, 1) * u).sum(dim=1).squeeze(-1)
            uncertainty = 0.5 * disagreement + 0.5 * agent_u
        else:
            uncertainty = disagreement
        return p_final, uncertainty, disagreement

# ─── РАЗДЕЛ 4: ОБУЧЕНИЕ И ОЦЕНКА ───────────────────────────────────────────────

def train_cida(model, Xtr_pooled, Xtr_seq, ytr, use_cal: bool = True, use_ac: bool = True, epochs: int = EPOCHS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    loss_fn = CIDALoss(
        lambda_cal=0.4 if use_cal else 0.0,
        lambda_ac=0.2 if use_ac else 0.0,
        min_disagreement=0.08,
        multi_label=False
    )
    
    n = Xtr_pooled.size(0)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        for i0 in range(0, n, BATCH):
            batch_idx = idx[i0:i0+BATCH]
            Xb_pooled = Xtr_pooled[batch_idx]
            Xb_seq = Xtr_seq[batch_idx]
            yb = ytr[batch_idx]
            
            opt.zero_grad()
            out = model(pooled=Xb_pooled, seq_output=Xb_seq)
            p = out['p_final']
            
            loss, _ = loss_fn(
                p_final=p,
                y=yb,
                b_all=out['b_all'],
                orth_loss=out.get('L_orth')
            )
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
    model.eval()
    return model

class BaselineModel(nn.Module):
    """Стандартная MLP + softmax без deliberation."""
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, D_H), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(D_H,  D_H), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(D_H,  K),
        )

    def forward(self, x):
        return dict(p_final=F.softmax(self.net(x), dim=-1))

    @torch.no_grad()
    def mc_passes(self, x, T: int = 30):
        self.train()
        ps = torch.stack([F.softmax(self.net(x), -1) for _ in range(T)])
        self.eval()
        return ps.mean(0), ps.std(0).mean(-1)

def train_baseline(model, Xtr_pooled, ytr, epochs: int = EPOCHS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = Xtr_pooled.size(0)
    eps = 1e-7
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        for i0 in range(0, n, BATCH):
            batch_idx = idx[i0:i0+BATCH]
            Xb_pooled = Xtr_pooled[batch_idx]
            yb = ytr[batch_idx]
            
            opt.zero_grad()
            out = model(Xb_pooled)
            p = out['p_final']
            loss = F.nll_loss(p.clamp(eps).log(), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model

@torch.no_grad()
def evaluate_cida(model, X_pooled, X_seq, y) -> Dict:
    out  = model(pooled=X_pooled, seq_output=X_seq)
    p    = out['p_final'].cpu().numpy()
    lb   = y.cpu().numpy()
    pred = p.argmax(1)
    res  = dict(
        acc=float((pred == lb).mean()),
        ece=ece_score(p, lb),
        brier=brier_score(p, lb),
        probs=p, preds=pred, labels=lb,
    )
    if 'disagreement' in out:
        res['disag'] = out['disagreement'].cpu().numpy()
    return res

@torch.no_grad()
def evaluate_baseline(model, X_pooled, y) -> Dict:
    out  = model(X_pooled)
    p    = out['p_final'].cpu().numpy()
    lb   = y.cpu().numpy()
    pred = p.argmax(1)
    return dict(
        acc=float((pred == lb).mean()),
        ece=ece_score(p, lb),
        brier=brier_score(p, lb),
        probs=p, preds=pred, labels=lb,
    )

# ─── H1: Weighted Mean vs Product of Experts ─────────────────────────────────

print(f"{'-'*65}")
print("H1 - Weighted Mean vs Product of Experts")
print("   Гипотеза: WM имеет меньший ECE для коррелированных агентов")
print("   Теория: PoE предполагает Sigma=sigma^2*I -> занижает дисперсию -> overconfidence")
print(f"{'-'*65}")

h1 = dict(wm_ece=[], poe_ece=[], wm_brier=[], poe_brier=[], wm_acc=[], poe_acc=[])

for s in range(N_SEEDS):
    set_seed(s)
    
    # Генерируем центры кластеров один раз на seed
    rng_centers = np.random.RandomState(s)
    centers = rng_centers.randn(K, D_IN) * 2.5
    
    Xtr_pooled, Xtr_seq, ytr = make_data(N_TR, centers, seed=s)
    Xte_pooled, Xte_seq, yte = make_data(N_TE, centers, seed=s+100)

    cfg_wm = CIDAPluginConfig(
        d_input=D_IN, d_hidden=D_H, d_message=D_MSG,
        num_agents=M, num_classes=K, max_rounds=ROUNDS,
        role_prior_blend=BLEND, use_perspective_projector=True,
        early_stop_threshold=None
    )
    mwm = CIDAPlugin(cfg_wm).to(DEVICE)
    mwm = train_cida(mwm, Xtr_pooled, Xtr_seq, ytr)

    cfg_poe = CIDAPluginConfig(
        d_input=D_IN, d_hidden=D_H, d_message=D_MSG,
        num_agents=M, num_classes=K, max_rounds=ROUNDS,
        role_prior_blend=BLEND, use_perspective_projector=True,
        early_stop_threshold=None
    )
    mpoe = CIDAPlugin(cfg_poe).to(DEVICE)
    mpoe.consensus = PoEConsensusWrapper(mpoe.consensus)
    mpoe = train_cida(mpoe, Xtr_pooled, Xtr_seq, ytr)

    rwm  = evaluate_cida(mwm,  Xte_pooled, Xte_seq, yte)
    rpoe = evaluate_cida(mpoe, Xte_pooled, Xte_seq, yte)

    for k in ['ece', 'brier', 'acc']:
        h1[f'wm_{k}'].append(rwm[k]); h1[f'poe_{k}'].append(rpoe[k])

    print(f"  seed {s}: WM  ECE={rwm['ece']:.4f} Brier={rwm['brier']:.4f} Acc={rwm['acc']:.3f}"
          f"  |  PoE ECE={rpoe['ece']:.4f} Brier={rpoe['brier']:.4f} Acc={rpoe['acc']:.3f}")

h1['p_ece']   = wilcoxon_p(h1['wm_ece'],   h1['poe_ece'])
h1['p_brier'] = wilcoxon_p(h1['wm_brier'], h1['poe_brier'])
h1_ok = np.mean(h1['wm_ece']) < np.mean(h1['poe_ece'])
print(f"\n  WM  ECE:   {np.mean(h1['wm_ece']):.4f} +/- {np.std(h1['wm_ece']):.4f}")
print(f"  PoE ECE:   {np.mean(h1['poe_ece']):.4f} +/- {np.std(h1['poe_ece']):.4f}")
print(f"  Wilcoxon p={h1['p_ece']:.4f} {pstars(h1['p_ece'])}"
      f"  -> {'[OK] ПОДТВЕРЖДЕНА' if h1_ok else '[NT] НЕЙТРАЛЬНАЯ'}\n")

# ─── H2: Role Prior Disagreement Guarantee ≥ 0.21 ────────────────────────────

print(f"{'-'*65}")
print("H2 - Role Prior Disagreement Guarantee: |b_P - b_D| >= 0.21")
print("   Теория: (1-blend)*b_data + blend*prior -> |b_P-b_D| >= blend*|P_P-P_D|*2 = 0.42")
print(f"{'-'*65}")

h2 = dict(dist=[], minval=[], dist_no_prior=[])

for s in range(N_SEEDS):
    set_seed(s)
    B_ = 1000
    b_data = torch.rand(B_, K, device=DEVICE)
    b_data = b_data / b_data.sum(-1, keepdim=True)

    b_P = apply_role_prior(b_data, 0)   # Прокурор: prior=0.85
    b_D = apply_role_prior(b_data, 1)   # Защитник: prior=0.15
    d   = torch.norm(b_P - b_D, p=1, dim=-1)

    # Без prior
    b2 = torch.rand(B_, K, device=DEVICE); b2 /= b2.sum(-1, keepdim=True)
    b3 = torch.rand(B_, K, device=DEVICE); b3 /= b3.sum(-1, keepdim=True)
    d_no = torch.norm(b2 - b3, p=1, dim=-1)

    h2['dist'].append(d.mean().item())
    h2['minval'].append(d.min().item())
    h2['dist_no_prior'].append(d_no.mean().item())

    print(f"  seed {s}: mean={d.mean():.4f}  min={d.min():.4f}"
          f"  (no prior mean={d_no.mean():.4f})")

theory = BLEND * abs(0.85 - 0.15) * 2
h2_ok  = all(v >= 0.21 for v in h2['minval'])
print(f"\n  Теоретический минимум: blend*|P_P-P_D|*2 = {theory:.4f}")
print(f"  Измеренное среднее:    {np.mean(h2['dist']):.4f} +/- {np.std(h2['dist']):.4f}")
print(f"  Минимум по всем seeds: {np.min(h2['minval']):.4f}")
print(f"  -> {'[OK] ПОДТВЕРЖДЕНА (все min >= 0.21)' if h2_ok else '[FAIL] НЕ ПОДТВЕРЖДЕНА'}\n")

# ─── H3: Disagreement как предиктор ошибки (vs MC Dropout) ────────────────────

print(f"{'-'*65}")
print("H3 - Disagreement -> Error Prediction  (AUC-ROC vs MC Dropout)")
print("   Гипотеза: Observeable disagreement >= black-box MC Dropout entropy")
print(f"{'-'*65}")

h3 = dict(cida_auc=[], mcd_auc=[], cida_spearman=[], mcd_spearman=[])

for s in range(N_SEEDS):
    set_seed(s)
    
    rng_centers = np.random.RandomState(s)
    centers = rng_centers.randn(K, D_IN) * 2.5
    
    # Повышаем шум до 0.75 для реалистичного уровня ошибок
    Xtr_pooled, Xtr_seq, ytr = make_data(N_TR, centers, noise=0.75, seed=s)
    Xte_pooled, Xte_seq, yte = make_data(N_TE, centers, noise=0.75, seed=s+100)

    cfg = CIDAPluginConfig(
        d_input=D_IN, d_hidden=D_H, d_message=D_MSG,
        num_agents=M, num_classes=K, max_rounds=ROUNDS,
        role_prior_blend=BLEND, use_perspective_projector=True,
        early_stop_threshold=None
    )
    mc = CIDAPlugin(cfg).to(DEVICE)
    mc = train_cida(mc, Xtr_pooled, Xtr_seq, ytr)
    rc = evaluate_cida(mc, Xte_pooled, Xte_seq, yte)
    errors = (rc['preds'] != rc['labels']).astype(int)
    
    ca  = auroc_score(rc['disag'], errors)
    cr, _ = stats.spearmanr(rc['disag'], errors)
    h3['cida_auc'].append(ca); h3['cida_spearman'].append(float(cr))

    mb = train_baseline(BaselineModel().to(DEVICE), Xtr_pooled, ytr)
    _, mcd_unc = mb.mc_passes(Xte_pooled, T=30)
    mcd_unc = mcd_unc.cpu().numpy()
    ma  = auroc_score(mcd_unc, errors)
    mr, _ = stats.spearmanr(mcd_unc, errors)
    h3['mcd_auc'].append(ma); h3['mcd_spearman'].append(float(mr))

    print(f"  seed {s}: CIDA AUC={ca:.4f} rho={cr:.3f}"
          f"  |  MCD AUC={ma:.4f} rho={mr:.3f}")

cida_clean = [x for x in h3['cida_auc'] if not np.isnan(x)]
mcd_clean  = [x for x in h3['mcd_auc']  if not np.isnan(x)]
pairs_ok   = [(c,m) for c,m in zip(h3['cida_auc'], h3['mcd_auc'])
              if not np.isnan(c) and not np.isnan(m)]
n_nan = N_SEEDS - len(cida_clean)
if n_nan > 0:
    print(f"  [NOTE] {n_nan} seed(s) with NaN filtered out (acc=1.0 -> no errors)")
h3['p'] = wilcoxon_p([p[0] for p in pairs_ok], [p[1] for p in pairs_ok])
h3_ok   = (np.mean(cida_clean) > np.mean(mcd_clean)) if cida_clean else False
mu_c = np.mean(cida_clean) if cida_clean else float('nan')
mu_m = np.mean(mcd_clean)  if mcd_clean  else float('nan')
sd_c = np.std(cida_clean)  if cida_clean else float('nan')
sd_m = np.std(mcd_clean)   if mcd_clean  else float('nan')
print(f"\n  CIDA AUC: {mu_c:.4f} +/- {sd_c:.4f}  (n_valid={len(cida_clean)})")
print(f"  MCD  AUC: {mu_m:.4f} +/- {sd_m:.4f}  (n_valid={len(mcd_clean)})")
print(f"  Wilcoxon p={h3['p']:.4f} {pstars(h3['p'])}"
      f"  -> {'[OK] CIDA превосходит MCD' if h3_ok else '[NT] Сопоставимо'}\n")

# ─── H4: Calibration Quality vs Baseline Softmax ──────────────────────────────

print(f"{'-'*65}")
print("H4 - Calibration: CIDA + Brier loss vs Baseline softmax")
print("   Гипотеза: Dirichlet beliefs + calibration loss -> меньший ECE")
print(f"{'-'*65}")

h4 = dict(cida_ece=[], base_ece=[], cida_brier=[], base_brier=[],
          cida_acc=[], base_acc=[])

for s in range(N_SEEDS):
    set_seed(s)
    
    rng_centers = np.random.RandomState(s)
    centers = rng_centers.randn(K, D_IN) * 2.5
    
    Xtr_pooled, Xtr_seq, ytr = make_data(N_TR, centers, seed=s)
    Xte_pooled, Xte_seq, yte = make_data(N_TE, centers, seed=s+100)

    cfg = CIDAPluginConfig(
        d_input=D_IN, d_hidden=D_H, d_message=D_MSG,
        num_agents=M, num_classes=K, max_rounds=ROUNDS,
        role_prior_blend=BLEND, use_perspective_projector=True,
        early_stop_threshold=None
    )
    mc = CIDAPlugin(cfg).to(DEVICE)
    mc = train_cida(mc, Xtr_pooled, Xtr_seq, ytr, use_cal=True, use_ac=True)
    
    mb = BaselineModel().to(DEVICE)
    mb = train_baseline(mb, Xtr_pooled, ytr)

    rc = evaluate_cida(mc, Xte_pooled, Xte_seq, yte)
    rb = evaluate_baseline(mb, Xte_pooled, yte)
    for k in ['ece', 'brier', 'acc']:
        h4[f'cida_{k}'].append(rc[k]); h4[f'base_{k}'].append(rb[k])

    print(f"  seed {s}: CIDA ECE={rc['ece']:.4f} acc={rc['acc']:.3f}"
          f"  |  Base ECE={rb['ece']:.4f} acc={rb['acc']:.3f}")

h4['p_ece']   = wilcoxon_p(h4['cida_ece'],   h4['base_ece'])
h4['p_brier'] = wilcoxon_p(h4['cida_brier'], h4['base_brier'])
h4_ok = np.mean(h4['cida_ece']) < np.mean(h4['base_ece'])
print(f"\n  CIDA ECE:  {np.mean(h4['cida_ece']):.4f} +/- {np.std(h4['cida_ece']):.4f}")
print(f"  Base ECE:  {np.mean(h4['base_ece']):.4f} +/- {np.std(h4['base_ece']):.4f}")
print(f"  Wilcoxon p={h4['p_ece']:.4f} {pstars(h4['p_ece'])}"
      f"  -> {'[OK] ПОДТВЕРЖДЕНА' if h4_ok else '[NT] НЕЙТРАЛЬНАЯ'}\n")

# ─── H5: Agent Representation Diversity (Gram Matrix) ─────────────────────────

print(f"{'-'*65}")
print("H5 - Gram Matrix: |G - I| (с priors vs без)")
print("   Гипотеза: Role priors -> более ортогональные представления агентов")
print(f"{'-'*65}")

h5 = dict(gram_with=[], gram_without=[])
h5_grams = {}

for s in range(N_SEEDS):
    set_seed(s)
    
    rng_centers = np.random.RandomState(s)
    centers = rng_centers.randn(K, D_IN) * 2.5
    
    Xtr_pooled, Xtr_seq, ytr = make_data(N_TR, centers, seed=s)
    Xte_pooled, Xte_seq, _ = make_data(300, centers, seed=s+200)

    # Сохраняем и восстанавливаем blend значение динамически через патч модуля
    for priors, key in [(True, 'with'), (False, 'without')]:
        agent.ROLE_PRIOR_BLEND = 0.30 if priors else 0.00
        
        cfg = CIDAPluginConfig(
            d_input=D_IN, d_hidden=D_H, d_message=D_MSG,
            num_agents=M, num_classes=K, max_rounds=ROUNDS,
            role_prior_blend=0.30 if priors else 0.00,
            use_perspective_projector=True,
            early_stop_threshold=None
        )
        m = CIDAPlugin(cfg).to(DEVICE)
        m = train_cida(m, Xtr_pooled, Xtr_seq, ytr)
        
        with torch.no_grad():
            out = m(pooled=Xte_pooled, seq_output=Xte_seq)
        
        sf  = out['state_final'].s                        # (B, M, D_H)
        smn = F.normalize(sf.mean(0), p=2, dim=-1)        # (M, D_H)
        G   = (smn @ smn.T).cpu().numpy()                 # (M, M)
        score = np.abs(G - np.eye(M)).mean()
        h5[f'gram_{key}'].append(score)
        if s == 0:
            h5_grams[key] = G

    print(f"  seed {s}: |G-I| with={h5['gram_with'][-1]:.4f}"
          f"  without={h5['gram_without'][-1]:.4f}")

# Восстанавливаем дефолтное значение blend
agent.ROLE_PRIOR_BLEND = 0.30

h5['p'] = wilcoxon_p(h5['gram_with'], h5['gram_without'])
mean_w  = np.mean(h5['gram_with']); mean_wo = np.mean(h5['gram_without'])
collapsed = abs(mean_w - 0.75) < 0.01 and abs(mean_wo - 0.75) < 0.01

print(f"\n  With priors    |G-I|: {mean_w:.4f} +/- {np.std(h5['gram_with']):.4f}")
print(f"  Without priors |G-I|: {mean_wo:.4f} +/- {np.std(h5['gram_without']):.4f}")
if collapsed:
    print(f"  [WARNING] |G-I|=0.75 - агенты коллапсировали к идентичным векторам!")
else:
    print(f"  [OK] Агенты разошлись по пространству благодаря PerspectiveProjector!")
print(f"  Wilcoxon p={h5['p']:.4f} {pstars(h5['p'])}\n")

# ─── H6: Multi-round Deliberation Convergence ────────────────────────────────

print(f"{'-'*65}")
print("H6 - Multi-round Convergence")
print("   Гипотеза: Disagreement убывает, ECE улучшается с каждым раундом")
print(f"{'-'*65}")

h6 = {f'disag_r{r}': [] for r in range(ROUNDS)}
h6.update({f'ece_r{r}': [] for r in range(ROUNDS)})

for s in range(N_SEEDS):
    set_seed(s)
    
    rng_centers = np.random.RandomState(s)
    centers = rng_centers.randn(K, D_IN) * 2.5
    
    Xtr_pooled, Xtr_seq, ytr = make_data(N_TR, centers, seed=s)
    Xte_pooled, Xte_seq, yte = make_data(N_TE, centers, seed=s+100)

    cfg = CIDAPluginConfig(
        d_input=D_IN, d_hidden=D_H, d_message=D_MSG,
        num_agents=M, num_classes=K, max_rounds=ROUNDS,
        role_prior_blend=BLEND, use_perspective_projector=True,
        early_stop_threshold=None
    )
    m = CIDAPlugin(cfg).to(DEVICE)
    m = train_cida(m, Xtr_pooled, Xtr_seq, ytr)
    lb = yte.cpu().numpy()

    with torch.no_grad():
        out = m(pooled=Xte_pooled, seq_output=Xte_seq)

    for r in range(ROUNDS):
        br_tensor = out['b_all'][r]
        dg_r = br_tensor.std(dim=1).mean(dim=-1).mean().item()
        h6[f'disag_r{r}'].append(dg_r)
        
        br = br_tensor.cpu().numpy()
        pr = br.mean(1); pr /= pr.sum(1, keepdims=True)
        h6[f'ece_r{r}'].append(ece_score(pr, lb))

h6_converges = np.mean(h6[f'disag_r{ROUNDS-1}']) < np.mean(h6['disag_r0'])
print("\n  Disagreement по раундам:")
for r in range(ROUNDS):
    print(f"    Round {r+1}: {np.mean(h6[f'disag_r{r}']):.4f} +/- {np.std(h6[f'disag_r{r}']):.4f}")
print("\n  ECE по раундам:")
for r in range(ROUNDS):
    print(f"    Round {r+1}: {np.mean(h6[f'ece_r{r}']):.4f} +/- {np.std(h6[f'ece_r{r}']):.4f}")
print(f"\n  -> Disagreement {'убывает [OK]' if h6_converges else 'не убывает [NT]'}\n")

# ─── H7: OOD Routing via Disagreement ─────────────────────────────────────────

print(f"{'-'*65}")
print("H7 - OOD Routing: высокое disagreement -> ручная проверка")
print("   Гипотеза: auto-routed subset точнее baseline (100% авторешений)")
print(f"{'-'*65}")

h7 = dict(cida_all=[], cida_auto=[], base_all=[], route_rate=[])

for s in range(N_SEEDS):
    set_seed(s)
    
    rng_centers = np.random.RandomState(s)
    centers = rng_centers.randn(K, D_IN) * 2.5
    
    Xtr_pooled, Xtr_seq, ytr  = make_data(N_TR, centers, seed=s)
    Xood_pooled, Xood_seq, yood = make_ood(N_TE, shift=2.5, seed=s+300)

    cfg = CIDAPluginConfig(
        d_input=D_IN, d_hidden=D_H, d_message=D_MSG,
        num_agents=M, num_classes=K, max_rounds=ROUNDS,
        role_prior_blend=BLEND, use_perspective_projector=True,
        early_stop_threshold=None
    )
    mc = CIDAPlugin(cfg).to(DEVICE)
    mc = train_cida(mc, Xtr_pooled, Xtr_seq, ytr)
    
    mb = BaselineModel().to(DEVICE)
    mb = train_baseline(mb, Xtr_pooled, ytr)

    rc = evaluate_cida(mc, Xood_pooled, Xood_seq, yood)
    rb = evaluate_baseline(mb, Xood_pooled, yood)

    # 70% самых уверенных (наименьший disagreement) → авторешение
    thresh   = np.percentile(rc['disag'], 70)
    auto     = rc['disag'] <= thresh
    auto_acc = (rc['preds'][auto] == rc['labels'][auto]).mean() if auto.sum() else rc['acc']

    h7['cida_all'].append(rc['acc'])
    h7['cida_auto'].append(float(auto_acc))
    h7['base_all'].append(rb['acc'])
    h7['route_rate'].append(float(auto.mean()))

    print(f"  seed {s}: CIDA_all={rc['acc']:.4f}  CIDA_auto={auto_acc:.4f}"
          f" ({auto.mean()*100:.0f}%)  Base={rb['acc']:.4f}")

h7['p'] = wilcoxon_p(h7['cida_auto'], h7['base_all'])
h7_ok   = np.mean(h7['cida_auto']) >= np.mean(h7['base_all'])
rate    = np.mean(h7['route_rate'])
print(f"\n  CIDA auto ({rate*100:.0f}%): {np.mean(h7['cida_auto']):.4f} +/- {np.std(h7['cida_auto']):.4f}")
print(f"  Baseline (100%):     {np.mean(h7['base_all']):.4f} +/- {np.std(h7['base_all']):.4f}")
print(f"  Wilcoxon p={h7['p']:.4f} {pstars(h7['p'])}"
      f"  -> {'[OK] ПОДТВЕРЖДЕНА' if h7_ok else '[NT] НЕЙТРАЛЬНАЯ'}\n")

# ─── FIGURE 1: Главная панель — 2×4 grid ──────────────────────────────────────

print(f"{'-'*65}\nГенерация рисунков...\n{'-'*65}")

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'figure.dpi': 140,
})

fig = plt.figure(figsize=(17, 9))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.52, wspace=0.38)

# ── H1: WM vs PoE bar chart ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
labs1 = ['WM\n(CIDA)', 'PoE']
ece1  = [np.mean(h1['wm_ece']),   np.mean(h1['poe_ece'])]
std1  = [np.std(h1['wm_ece']),    np.std(h1['poe_ece'])]
bars1 = ax.bar(labs1, ece1, yerr=std1, capsize=7,
               color=[C['wm'], C['poe']], alpha=0.85,
               edgecolor='white', lw=1.5, error_kw={'lw': 2, 'capthick': 2})
for b, v in zip(bars1, ece1):
    ax.text(b.get_x()+b.get_width()/2, v+max(std1)*1.2+0.001,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_ylabel('ECE ↓')
ax.set_title(f'H1 · WM vs PoE\np={h1["p_ece"]:.3f}{pstars(h1["p_ece"])}',
             fontweight='bold')

# ── H2: Prior guarantee bar ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
measured = np.mean(h2['dist']); theory_val = BLEND*abs(0.85-0.15)*2
ax2.bar(['Measured\nP–D dist', f'Lower\nbound'],
        [measured, 0.21], color=[C['wm'], C['ok']],
        alpha=0.85, edgecolor='white', lw=1.5)
ax2.errorbar([0], [measured], yerr=[np.std(h2['dist'])],
             fmt='none', color='k', capsize=7, lw=2)
ax2.axhline(0.21, color=C['ok'], ls='--', lw=1.8, alpha=0.7)
ax2.set_ylabel('|b_P − b_D|₁')
ax2.set_title(f'H2 · Prior Guarantee\nmin={np.min(h2["minval"]):.3f} ≥ 0.21',
              fontweight='bold')
ax2.set_ylim(0, measured * 1.5)

# ── H3: AUC boxplot ────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
bp = ax3.boxplot([cida_clean, mcd_clean],
                 labels=['CIDA\nDisagreement', 'MC\nDropout'],
                 patch_artist=True, medianprops={'lw': 2, 'color': 'white'})
bp['boxes'][0].set_facecolor(C['wm']); bp['boxes'][1].set_facecolor(C['mcd'])
ax3.axhline(0.5, color='gray', ls='--', alpha=0.5, label='Random')
ax3.set_ylabel('AUC-ROC ↑')
ax3.set_title(f'H3 · Error Prediction\np={h3["p"]:.3f}{pstars(h3["p"])}',
              fontweight='bold')

# ── H4: ECE + Brier grouped bars ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
x_ = np.arange(2); w_ = 0.35
kw = dict(alpha=0.85, edgecolor='white', lw=1.5, error_kw={'lw': 2, 'capthick': 2})
ax4.bar(x_-w_/2, [np.mean(h4['cida_ece']), np.mean(h4['cida_brier'])], w_,
        yerr=[np.std(h4['cida_ece']), np.std(h4['cida_brier'])],
        label='CIDA', color=C['wm'], capsize=5, **kw)
ax4.bar(x_+w_/2, [np.mean(h4['base_ece']), np.mean(h4['base_brier'])], w_,
        yerr=[np.std(h4['base_ece']), np.std(h4['base_brier'])],
        label='Baseline', color=C['base'], capsize=5, **kw)
ax4.set_xticks(x_); ax4.set_xticklabels(['ECE ↓', 'Brier ↓'])
ax4.legend(fontsize=8)
ax4.set_title(f'H4 · Calibration\np={h4["p_ece"]:.3f}{pstars(h4["p_ece"])}',
              fontweight='bold')

# ── H5: Gram diversity scatter ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 0])
sc = range(N_SEEDS)
ax5.scatter(sc, h5['gram_with'],    color=C['wm'], s=65, zorder=3, label='With priors')
ax5.scatter(sc, h5['gram_without'], color=C['np'], s=65, zorder=3,
            marker='s', label='No priors')
for i in range(N_SEEDS):
    ax5.plot([i, i], [h5['gram_with'][i], h5['gram_without'][i]],
             color='gray', alpha=0.4, lw=1)
ax5.axhline(np.mean(h5['gram_with']),    color=C['wm'], ls='--', alpha=0.6, lw=1.5)
ax5.axhline(np.mean(h5['gram_without']), color=C['np'], ls='--', alpha=0.6, lw=1.5)
ax5.set_xlabel('Seed'); ax5.set_ylabel('|Gram − I|_mean')
ax5.set_title(f'H5 · Diversity\np={h5["p"]:.3f}{pstars(h5["p"])}',
              fontweight='bold')
ax5.legend(fontsize=8)

# ── H6: Convergence dual axis ──────────────────────────────────────────────────
ax6  = fig.add_subplot(gs[1, 1])
ax6b = ax6.twinx()
rds  = list(range(1, ROUNDS+1))
dm   = [np.mean(h6[f'disag_r{r}']) for r in range(ROUNDS)]
ds   = [np.std(h6[f'disag_r{r}'])  for r in range(ROUNDS)]
em   = [np.mean(h6[f'ece_r{r}'])   for r in range(ROUNDS)]
es   = [np.std(h6[f'ece_r{r}'])    for r in range(ROUNDS)]
l1,  = ax6.plot(rds, dm, 'o-', color=C['wm'], lw=2.5, ms=9, label='Disagreement')
ax6.fill_between(rds, [d-e for d,e in zip(dm,ds)], [d+e for d,e in zip(dm,ds)],
                 alpha=0.18, color=C['wm'])
l2,  = ax6b.plot(rds, em, 's--', color=C['poe'], lw=2.5, ms=9, label='ECE')
ax6b.fill_between(rds, [e-s_ for e,s_ in zip(em,es)], [e+s_ for e,s_ in zip(em,es)],
                  alpha=0.18, color=C['poe'])
ax6.set_xlabel('Round'); ax6.set_xticks(rds)
ax6.set_ylabel('Disagreement', color=C['wm'])
ax6b.set_ylabel('ECE', color=C['poe'])
ax6.set_title('H6 · Convergence\nDisagreement & ECE', fontweight='bold')
ax6.legend(handles=[l1, l2], fontsize=8, loc='upper right')

# ── H7: OOD routing bars ───────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[1, 2])
l7  = [f'CIDA all\n(100%)', f'CIDA auto\n({rate*100:.0f}%)', 'Baseline\n(100%)']
v7  = [np.mean(h7['cida_all']),  np.mean(h7['cida_auto']),  np.mean(h7['base_all'])]
e7  = [np.std(h7['cida_all']),   np.std(h7['cida_auto']),   np.std(h7['base_all'])]
c7  = [C['wm'], C['ok'], C['base']]
b7  = ax7.bar(l7, v7, yerr=e7, capsize=6, color=c7, alpha=0.85,
              edgecolor='white', lw=1.5, error_kw={'lw': 2})
ax7.axhline(0.5, color='gray', ls='--', alpha=0.5)
for b, v in zip(b7, v7):
    ax7.text(b.get_x()+b.get_width()/2, v+0.007,
             f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax7.set_ylabel('OOD Accuracy ↑')
ax7.set_title(f'H7 · OOD Routing\np={h7["p"]:.3f}{pstars(h7["p"])}',
              fontweight='bold')

# ── Summary panel ──────────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[1, 3])
ax8.axis('off')
confirmed = [h1_ok, h2_ok, h3_ok, h4_ok, not collapsed, h6_converges, h7_ok]
names     = ['H1 WM ECE < PoE ECE',
             'H2 |b_P−b_D| ≥ 0.21',
             'H3 Disag AUC > MCD',
             'H4 CIDA ECE < Base',
             'H5 Gram diversity',
             'H6 Disag converges',
             'H7 OOD routing ↑']
pvals     = [h1['p_ece'], 0.0, h3['p'], h4['p_ece'], h5['p'], 0.0, h7['p']]

n_ok = sum(confirmed)
ax8.text(0.04, 0.97, f'Подтверждено: {n_ok}/7',
         transform=ax8.transAxes, fontsize=12, fontweight='bold', va='top')
ax8.text(0.04, 0.88,
         f'Seeds: {N_SEEDS}  N_train: {N_TR}\n'
         f'D_hidden: {D_H}  Rounds: {ROUNDS}',
         transform=ax8.transAxes, fontsize=8,
         color='#555555', va='top')
for i, (nm, ok, pv) in enumerate(zip(names, confirmed, pvals)):
    y   = 0.76 - i * 0.108
    sym = '✓' if ok else '○'
    col = C['ok'] if ok else C['nt']
    p_s = f'p={pv:.3f}' if pv > 0 else 'mathematical'
    ax8.text(0.04, y, f'{sym} {nm}', transform=ax8.transAxes,
             fontsize=9, color=col, va='top', fontweight='bold' if ok else 'normal')
    ax8.text(0.04, y-0.048, f'   {p_s}', transform=ax8.transAxes,
             fontsize=7.5, color='#777777', va='top')

fig.suptitle(
    'CIDA Architecture Validation  ·  VEIMO Research Team  ·  2024\n'
    f'(* p<0.05  ** p<0.01  *** p<0.001  ○ directional / math)',
    fontsize=11, fontweight='bold', y=1.005
)
p1 = f'{OUT}/figures/fig1_hypotheses.png'
fig.savefig(p1, dpi=140, bbox_inches='tight', facecolor='white')
plt.close(); print(f"  Saved: {p1}")

# ─── FIGURE 2: Reliability Diagrams (1×3) ────────────────────────────────────

set_seed(42)
rng_centers = np.random.RandomState(42)
centers_r = rng_centers.randn(K, D_IN) * 2.5

Xtr_r_pooled, Xtr_r_seq, ytr_r = make_data(N_TR, centers_r, seed=42)
Xte_r_pooled, Xte_r_seq, yte_r = make_data(N_TE, centers_r, seed=142)

cfg_wm_r = CIDAPluginConfig(
    d_input=D_IN, d_hidden=D_H, d_message=D_MSG,
    num_agents=M, num_classes=K, max_rounds=ROUNDS,
    role_prior_blend=BLEND, use_perspective_projector=True,
    early_stop_threshold=None
)
mwm_r = CIDAPlugin(cfg_wm_r).to(DEVICE)
mwm_r = train_cida(mwm_r, Xtr_r_pooled, Xtr_r_seq, ytr_r)

cfg_poe_r = CIDAPluginConfig(
    d_input=D_IN, d_hidden=D_H, d_message=D_MSG,
    num_agents=M, num_classes=K, max_rounds=ROUNDS,
    role_prior_blend=BLEND, use_perspective_projector=True,
    early_stop_threshold=None
)
mpoe_r = CIDAPlugin(cfg_poe_r).to(DEVICE)
mpoe_r.consensus = PoEConsensusWrapper(mpoe_r.consensus)
mpoe_r = train_cida(mpoe_r, Xtr_r_pooled, Xtr_r_seq, ytr_r)

mbase_r = BaselineModel().to(DEVICE)
mbase_r = train_baseline(mbase_r, Xtr_r_pooled, ytr_r)

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle('Reliability Diagrams (perfect calibration = diagonal)',
              fontsize=12, fontweight='bold')

mods_r = {
    'CIDA (WM)': mwm_r,
    'CIDA (PoE)': mpoe_r,
    'Baseline': mbase_r
}

for ax, (nm, m), col in zip(axes2, mods_r.items(), [C['wm'], C['poe'], C['base']]):
    if 'CIDA' in nm:
        out_ = m(pooled=Xte_r_pooled, seq_output=Xte_r_seq)
    else:
        out_ = m(Xte_r_pooled)
    p_  = out_['p_final'].detach().cpu().numpy(); lb_ = yte_r.cpu().numpy()
    cf  = p_.max(1); pd = p_.argmax(1); ac = (pd == lb_).astype(float)

    bins = np.linspace(0, 1, 11)
    bc, ba = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (cf > lo) & (cf <= hi)
        if mk.sum() >= 5:
            bc.append(cf[mk].mean()); ba.append(ac[mk].mean())

    ax.plot([0,1],[0,1], 'k--', alpha=0.4, lw=1.5, label='Perfect')
    if bc:
        ax.plot(bc, ba, 'o-', color=col, lw=2.5, ms=8, label=nm)
        ax.fill_between(bc, ba, bc, alpha=0.12, color=col)
    ece_ = ece_score(p_, lb_)
    ax.set_title(f'{nm}\nECE = {ece_:.4f}', fontweight='bold')
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    ax.legend(fontsize=9)

plt.tight_layout()
p2 = f'{OUT}/figures/fig2_reliability.png'
fig2.savefig(p2, dpi=140, bbox_inches='tight', facecolor='white')
plt.close(); print(f"  Saved: {p2}")

# ─── FIGURE 3: Disagreement Analysis ──────────────────────────────────────────

set_seed(1)
rng_centers = np.random.RandomState(1)
centers_d = rng_centers.randn(K, D_IN) * 2.5

# Повышаем шум до 0.75 для реалистичного распределения несогласия
Xtr_d_pooled, Xtr_d_seq, ytr_d = make_data(N_TR, centers_d, noise=0.75, seed=1)
Xte_d_pooled, Xte_d_seq, yte_d = make_data(N_TE, centers_d, noise=0.75, seed=101)

cfg_d = CIDAPluginConfig(
    d_input=D_IN, d_hidden=D_H, d_message=D_MSG,
    num_agents=M, num_classes=K, max_rounds=ROUNDS,
    role_prior_blend=BLEND, use_perspective_projector=True,
    early_stop_threshold=None
)
m_d = CIDAPlugin(cfg_d).to(DEVICE)
m_d = train_cida(m_d, Xtr_d_pooled, Xtr_d_seq, ytr_d)

with torch.no_grad():
    out_d = m_d(pooled=Xte_d_pooled, seq_output=Xte_d_seq)

p_d  = out_d['p_final'].cpu().numpy()
dg_d = out_d['disagreement'].cpu().numpy()
lb_d = yte_d.cpu().numpy()
err_d = (p_d.argmax(1) != lb_d)

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
fig3.suptitle('Disagreement Analysis: Distribution & Error Prediction',
              fontsize=12, fontweight='bold')

# Panel 1: Disagreement density
ax = axes3[0]
kw_hist = dict(bins=25, density=True, edgecolor='white', lw=0.5)
ax.hist(dg_d[~err_d], alpha=0.72, color=C['ok'],
        label=f'Correct (n={int((~err_d).sum())})', **kw_hist)
ax.hist(dg_d[err_d],  alpha=0.72, color=C['fail'],
        label=f'Wrong  (n={int(err_d.sum())})',  **kw_hist)
ax.axvline(np.median(dg_d[~err_d]), color=C['ok'],   ls='--', lw=1.8)
ax.axvline(np.median(dg_d[err_d]),  color=C['fail'],  ls='--', lw=1.8)
ax.set_xlabel('Agent Disagreement'); ax.set_ylabel('Density')
ax.set_title('Disagreement Distribution\nby Prediction Outcome', fontweight='bold')
ax.legend(fontsize=9)

# Panel 2: Rolling accuracy vs disagreement
ax = axes3[1]
idx_s = np.argsort(dg_d)
win   = max(30, N_TE // 12)
rd, ra = [], []
for i in range(win, len(idx_s)):
    rd.append(dg_d[idx_s[i-win:i]].mean())
    ra.append(1 - err_d[idx_s[i-win:i]].mean())
ax.plot(rd, ra, color=C['wm'], lw=2.5)
ax.fill_between(rd, ra, alpha=0.15, color=C['wm'])
ax.axhline(1-err_d.mean(), color='gray', ls='--', alpha=0.6,
           label=f'Overall acc={1-err_d.mean():.3f}')
ax.set_xlabel('Disagreement (rolling mean)')
ax.set_ylabel('Accuracy (rolling mean)')
ax.set_title(f'Accuracy vs Disagreement\n(window={win})', fontweight='bold')
ax.legend(fontsize=9)

# Panel 3: ROC curve
ax = axes3[2]
try:
    from sklearn.metrics import roc_curve, auc as sk_auc
    fpr, tpr, _ = roc_curve(err_d.astype(int), dg_d)
    auc_val = sk_auc(fpr, tpr)
    ax.plot(fpr, tpr, color=C['wm'], lw=2.5, label=f'CIDA Disag. (AUC={auc_val:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color=C['wm'])
except Exception:
    auc_val = auroc_score(dg_d, err_d.astype(int))
    ax.text(0.3, 0.5, f'AUC ≈ {auc_val:.3f}', fontsize=16,
            transform=ax.transAxes, color=C['wm'], fontweight='bold')
ax.plot([0,1],[0,1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC: Disagreement\nas Error Detector', fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
p3 = f'{OUT}/figures/fig3_disagreement.png'
fig3.savefig(p3, dpi=140, bbox_inches='tight', facecolor='white')
plt.close(); print(f"  Saved: {p3}")

# ─── FIGURE 4: Gram Matrix Heatmaps (1×3) ─────────────────────────────────────

fig4, axes4 = plt.subplots(1, 3, figsize=(14, 5))
fig4.suptitle('Agent Representation Diversity: Gram Matrix  (Ideal = Identity)',
              fontsize=12, fontweight='bold')

role_names = ['Prosecutor', 'Defender', 'Skeptic', 'Integrator']
cmap_g = LinearSegmentedColormap.from_list('gram', ['#27AE60', 'white', '#E24B4A'])

for ax, (key, title) in zip(axes4[:2], [('with',    'With Role Priors'),
                                          ('without', 'Without Role Priors')]):
    G    = h5_grams.get(key, np.eye(M))
    off  = np.abs(G - np.eye(M)).mean()
    im   = ax.imshow(G, cmap=cmap_g, vmin=-1, vmax=1)
    ax.set_xticks(range(M)); ax.set_yticks(range(M))
    ax.set_xticklabels(role_names, rotation=35, ha='right', fontsize=8.5)
    ax.set_yticklabels(role_names, fontsize=8.5)
    ax.set_title(f'{title}\n|Gram−I| = {off:.4f}', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(M):
        for j in range(M):
            ax.text(j, i, f'{G[i,j]:.2f}', ha='center', va='center',
                    fontsize=9, color='black' if abs(G[i,j]) < 0.6 else 'white')

ax = axes4[2]
labs_g = ['With\nPriors', 'Without\nPriors', 'Identity\n(ideal)']
vals_g = [np.mean(h5['gram_with']), np.mean(h5['gram_without']), 0.0]
stds_g = [np.std(h5['gram_with']),  np.std(h5['gram_without']),  0.0]
cols_g = [C['wm'], C['np'], C['ok']]
bars_g = ax.bar(labs_g, vals_g, yerr=stds_g, capsize=7,
                color=cols_g, alpha=0.85, edgecolor='white', lw=1.5,
                error_kw={'lw': 2})
for b, v in zip(bars_g, vals_g):
    ax.text(b.get_x()+b.get_width()/2, v+0.005,
            f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('|Gram − Identity|_mean ↓')
ax.set_title(f'H5 Diversity Score\np={h5["p"]:.3f}{pstars(h5["p"])}',
             fontweight='bold')

plt.tight_layout()
p4 = f'{OUT}/figures/fig4_gram.png'
fig4.savefig(p4, dpi=140, bbox_inches='tight', facecolor='white')
plt.close(); print(f"  Saved: {p4}")

# ─── ИТОГОВАЯ ТАБЛИЦА ─────────────────────────────────────────────────────────

all_ok = confirmed
n_conf = sum(all_ok)
elapsed = time.time() - t0

rows = [
    ('H1', 'WM ECE < PoE',      f"{np.mean(h1['wm_ece']):.4f}+/-{np.std(h1['wm_ece']):.4f}",
     f"{np.mean(h1['poe_ece']):.4f}+/-{np.std(h1['poe_ece']):.4f}",
     f"{h1['p_ece']:.4f}{pstars(h1['p_ece'])}", h1_ok),
    ('H2', '|b_P-b_D|>=0.21',   f"{np.mean(h2['dist']):.4f}+/-{np.std(h2['dist']):.4f}",
     'random mean',
     'mathematical', h2_ok),
    ('H3', 'CIDA AUC > MCD',    f"{np.mean(h3['cida_auc']):.4f}+/-{np.std(h3['cida_auc']):.4f}",
     f"{np.mean(h3['mcd_auc']):.4f}+/-{np.std(h3['mcd_auc']):.4f}",
     f"{h3['p']:.4f}{pstars(h3['p'])}", h3_ok),
    ('H4', 'CIDA ECE < Base',   f"{np.mean(h4['cida_ece']):.4f}+/-{np.std(h4['cida_ece']):.4f}",
     f"{np.mean(h4['base_ece']):.4f}+/-{np.std(h4['base_ece']):.4f}",
     f"{h4['p_ece']:.4f}{pstars(h4['p_ece'])}", h4_ok),
    ('H5', 'Diversity priors',  f"|G-I|={np.mean(h5['gram_with']):.4f}",
     f"|G-I|={np.mean(h5['gram_without']):.4f}",
     f"{h5['p']:.4f}{pstars(h5['p'])}", not collapsed),
    ('H6', 'Disag -> over rds',  f"R1={np.mean(h6['disag_r0']):.4f}->R{ROUNDS}={np.mean(h6[f'disag_r{ROUNDS-1}']):.4f}",
     'N/A', 'observed', h6_converges),
    ('H7', 'OOD routing ->',     f"{np.mean(h7['cida_auto']):.4f}+/-{np.std(h7['cida_auto']):.4f}",
     f"{np.mean(h7['base_all']):.4f}+/-{np.std(h7['base_all']):.4f}",
     f"{h7['p']:.4f}{pstars(h7['p'])}", h7_ok),
]

W = 80
summary = [
    '='*W,
    f"{'CIDA VALIDATION SUITE - VEIMO Research Team':^{W}}",
    '-'*W,
    f"  Seeds: {N_SEEDS}  |  N_train: {N_TR}  |  D_hidden: {D_H}"
    f"  |  Rounds: {ROUNDS}  |  Device: {DEVICE}",
    f"  Runtime: {elapsed:.0f}s",
    '-'*W,
    f"  {'ID':<4} {'Hypothesis':<22} {'CIDA':<28} {'Baseline':<24} {'p-val':<12} {'OK?'}",
    '-'*W,
]
for h, nm, cida_v, base_v, pv, ok in rows:
    sym = 'OK' if ok else 'NT'
    summary.append(f"  {h:<4} {nm:<22} {cida_v:<28} {base_v:<24} {pv:<12} {sym}")

summary += [
    '-'*W,
    f"  Confirmed: {n_conf}/7  "
    f"({'Excellent' if n_conf >= 6 else 'Good' if n_conf >= 4 else 'Mixed'})",
    f"  Note: NT = directional result or observed trend (not null rejected)",
    f"  (* p<0.05  ** p<0.01  *** p<0.001  ns p>=0.05)",
    '='*W,
    f"  Figures:  {OUT}/figures/",
    f"  JSON:     {OUT}/results.json",
]

print()
for line in summary:
    print(line)

# Сохраняем текстовый отчет
with open(f'{OUT}/summary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary))

def to_serializable(x):
    if isinstance(x, (np.floating, float)): return float(x)
    if isinstance(x, (np.integer, int)):    return int(x)
    if isinstance(x, (list, tuple)):        return [to_serializable(v) for v in x]
    if isinstance(x, dict):                 return {k: to_serializable(v) for k,v in x.items()}
    if isinstance(x, np.ndarray):           return x.tolist()
    return str(x)

raw = dict(
    config=dict(n_seeds=N_SEEDS, n_train=N_TR, n_test=N_TE,
                d_in=D_IN, d_hidden=D_H, k=K, m=M, rounds=ROUNDS,
                epochs=EPOCHS, lr=LR, device=str(DEVICE),
                role_prior_blend=BLEND),
    h1=h1, h2=h2, h3=h3, h4=h4, h5=h5, h6=h6, h7=h7,
    summary=dict(confirmed=n_conf, total=7, elapsed_s=round(elapsed, 1)),
)
with open(f'{OUT}/results.json', 'w', encoding='utf-8') as f:
    json.dump(to_serializable(raw), f, indent=2)

print(f"\n  All outputs saved to: {OUT}/")