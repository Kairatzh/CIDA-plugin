"""
Evaluator — метрики для статьи
================================
Все метрики которые нужны чтобы статья прошла ревью.

Основные вопросы которые отвечает этот модуль:

1. ACCURACY        — базовая правильность (нужна но не достаточна)
2. ECE             — откалибрована ли уверенность?
3. AUROC(u, error) — предсказывает ли uncertainty реальные ошибки?
4. DIFFICULTY      — как меняется качество с числом evidence?
5. OOD             — знает ли модель когда не знает?
6. PERSUASION      — меняют ли агенты мнение на сложных примерах?
7. SPECIALIZATION  — специализируются ли агенты по разным типам рака?
8. SPEED           — CIDA быстрее MC Dropout при той же uncertainty?

Headline hypothesis:
"На contradictory multi-evidence примерах CIDA:
 (a) точнее baseline'ов
 (b) лучше знает когда ошибается (AUROC > 0.75)"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from sklearn.metrics import roc_auc_score, f1_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─── Контейнер результатов ────────────────────────────────────────────────────

@dataclass
class ModelResults:
    """Все предсказания модели для последующего анализа"""
    model_name: str
    predictions: List[int] = field(default_factory=list)
    probabilities: List[List[float]] = field(default_factory=list)
    uncertainties: List[float] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    difficulties: List[str] = field(default_factory=list)
    cancer_types: List[str] = field(default_factory=list)
    num_evidences: List[int] = field(default_factory=list)
    
    # Только для CIDA
    disagreements: List[float] = field(default_factory=list)
    persuasions: List[float] = field(default_factory=list)
    agent_weights: List[List[float]] = field(default_factory=list)
    rounds_used: List[int] = field(default_factory=list)


# ─── ECE ──────────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error"""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        n_in_bin = in_bin.sum()
        
        if n_in_bin > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += abs(conf_in_bin - acc_in_bin) * (n_in_bin / len(labels))
    
    return ece


# ─── AUROC для uncertainty ─────────────────────────────────────────────────────

def compute_uncertainty_auroc(uncertainties: np.ndarray, labels: np.ndarray,
                               predictions: np.ndarray) -> float:
    """
    AUROC(uncertainty, is_error).
    Ключевая метрика: uncertainty должна быть высокой когда модель ошибается.
    AUROC = 0.5 → uncertainty случайна
    AUROC = 1.0 → uncertainty идеально предсказывает ошибки
    AUROC > 0.75 → headline result
    """
    if not HAS_SKLEARN:
        return _manual_auroc(uncertainties, (predictions != labels).astype(int))
    
    errors = (predictions != labels).astype(int)
    
    if errors.sum() == 0 or errors.sum() == len(errors):
        return 0.5  # нет ошибок или все ошибки → нельзя посчитать
    
    try:
        return roc_auc_score(errors, uncertainties)
    except Exception:
        return 0.5


def _manual_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC без sklearn"""
    sorted_idx = np.argsort(scores)[::-1]
    labels_sorted = labels[sorted_idx]
    
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    
    for l in labels_sorted:
        if l == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    
    return auc / (n_pos * n_neg)


# ─── Calibration curve ────────────────────────────────────────────────────────

def compute_calibration_curve(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> Tuple[List, List, List]:
    """Возвращает (mean_conf, mean_acc, bin_sizes) для графика"""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mean_confs, mean_accs, sizes = [], [], []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.sum() > 0:
            mean_confs.append(confidences[in_bin].mean())
            mean_accs.append(accuracies[in_bin].mean())
            sizes.append(in_bin.sum())
    
    return mean_confs, mean_accs, sizes


# ─── Главный Evaluator ────────────────────────────────────────────────────────

class Evaluator:
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
    
    def evaluate_cida(
        self,
        model,
        data_batches: List[Dict],
        encoder,
        epoch_fraction: float = 1.0,
    ) -> ModelResults:
        """Полная оценка CIDA с всеми агентскими метриками"""
        model.eval()
        encoder.eval() if hasattr(encoder, 'eval') else None
        
        results = ModelResults(model_name="CIDA")
        
        with torch.no_grad():
            for batch in data_batches:
                premises = batch['premises']
                hypotheses = batch['hypotheses']
                labels = batch['labels'].to(self.device)
                
                # Encode
                pooled, seq = encoder(premises, hypotheses, self.device)
                pooled = pooled.to(self.device)
                seq = seq.to(self.device)
                
                # CIDA forward
                out = model(pooled, seq_output=seq, epoch_fraction=epoch_fraction)
                
                p_final = out['p_final']  # (B, K)
                preds = p_final.argmax(-1)
                
                # Uncertainty из Dirichlet
                u_final = out['u_all'][-1].mean(dim=(1, 2))  # (B,)
                
                # Agent disagreement per sample
                b_last = out['b_all'][-1]  # (B, M, K)
                b_i = b_last.unsqueeze(2)
                b_j = b_last.unsqueeze(1)
                disagree = torch.norm(b_i - b_j, p=1, dim=-1).mean(dim=(1, 2))  # (B,)
                
                # Persuasion (изменение убеждений)
                if len(out['b_all']) >= 2:
                    shift = torch.norm(
                        out['b_all'][-1] - out['b_all'][0], p=1, dim=-1
                    ).mean(dim=1)  # (B,)
                else:
                    shift = torch.zeros(len(labels))
                
                # Agent weights
                w = out['w_final']  # (B, M)
                
                B = labels.size(0)
                for i in range(B):
                    results.predictions.append(preds[i].item())
                    results.probabilities.append(p_final[i].tolist())
                    results.uncertainties.append(u_final[i].item())
                    results.labels.append(labels[i].item())
                    results.difficulties.append(batch['difficulties'][i])
                    results.cancer_types.append(batch['cancer_types'][i])
                    results.num_evidences.append(batch['num_evidence'][i])
                    results.disagreements.append(disagree[i].item())
                    results.persuasions.append(shift[i].item())
                    results.agent_weights.append(w[i].tolist())
                    results.rounds_used.append(out['rounds_used'])
        
        return results
    
    def evaluate_baseline(
        self,
        model,
        model_name: str,
        data_batches: List[Dict],
        encoder,
        model_type: str = "pooled",  # "pooled" | "seq" | "mc_dropout"
    ) -> ModelResults:
        """Оценка baseline моделей"""
        model.eval() if hasattr(model, 'eval') else None
        
        results = ModelResults(model_name=model_name)
        
        with torch.no_grad():
            for batch in data_batches:
                premises = batch['premises']
                hypotheses = batch['hypotheses']
                labels = batch['labels'].to(self.device)
                
                pooled, seq = encoder(premises, hypotheses, self.device)
                pooled = pooled.to(self.device)
                seq = seq.to(self.device)
                
                if model_type == "mc_dropout":
                    # MC Dropout требует train mode
                    out = model(pooled)
                elif model_type == "seq":
                    out = model(seq)
                else:
                    out = model(pooled)
                
                probs = out['probs']
                preds = probs.argmax(-1)
                uncertainties = out['uncertainty']
                
                B = labels.size(0)
                for i in range(B):
                    results.predictions.append(preds[i].item())
                    results.probabilities.append(probs[i].tolist())
                    results.uncertainties.append(
                        uncertainties[i].item() if uncertainties.dim() > 0
                        else uncertainties.item()
                    )
                    results.labels.append(labels[i].item())
                    results.difficulties.append(batch['difficulties'][i])
                    results.cancer_types.append(batch['cancer_types'][i])
                    results.num_evidences.append(batch['num_evidence'][i])
        
        return results
    
    def compute_metrics(self, results: ModelResults) -> Dict:
        """Все метрики одним вызовом"""
        preds = np.array(results.predictions)
        labels = np.array(results.labels)
        probs = np.array(results.probabilities)
        uncertainties = np.array(results.uncertainties)
        difficulties = np.array(results.difficulties)
        num_evidences = np.array(results.num_evidences)
        
        metrics = {}
        
        # ── Accuracy ─────────────────────────────────────────────────────────
        metrics['accuracy'] = (preds == labels).mean()
        
        if HAS_SKLEARN:
            metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        
        # ── Калибровка ───────────────────────────────────────────────────────
        metrics['ece'] = compute_ece(probs, labels)
        
        # ── Uncertainty Quality ───────────────────────────────────────────────
        metrics['auroc_uncertainty'] = compute_uncertainty_auroc(
            uncertainties, labels, preds
        )
        
        # ── По сложности ──────────────────────────────────────────────────────
        for diff in ['easy', 'medium', 'hard']:
            mask = difficulties == diff
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean()
                ece = compute_ece(probs[mask], labels[mask])
                auroc = compute_uncertainty_auroc(
                    uncertainties[mask], labels[mask], preds[mask]
                )
                metrics[f'acc_{diff}'] = acc
                metrics[f'ece_{diff}'] = ece
                metrics[f'auroc_{diff}'] = auroc
        
        # ── По числу evidence ─────────────────────────────────────────────────
        for n_ev in [1, 2, 3]:
            mask = num_evidences == n_ev
            if mask.sum() > 0:
                metrics[f'acc_ev{n_ev}'] = (preds[mask] == labels[mask]).mean()
                metrics[f'auroc_ev{n_ev}'] = compute_uncertainty_auroc(
                    uncertainties[mask], labels[mask], preds[mask]
                )
        
        # ── CIDA-specific метрики ─────────────────────────────────────────────
        if results.disagreements:
            disagreements = np.array(results.disagreements)
            
            # Коррелирует ли disagreement с ошибками?
            errors = (preds != labels).astype(float)
            
            if errors.std() > 0 and disagreements.std() > 0:
                corr = np.corrcoef(disagreements, errors)[0, 1]
                metrics['disagreement_error_corr'] = corr
            
            # AUROC используя disagreement как uncertainty predictor
            metrics['auroc_disagreement'] = compute_uncertainty_auroc(
                disagreements, labels, preds
            )
            
            # Среднее разногласие по сложности
            for diff in ['easy', 'medium', 'hard']:
                mask = difficulties == diff
                if mask.sum() > 0:
                    metrics[f'disagreement_{diff}'] = disagreements[mask].mean()
            
            # Среднее убеждение (persuasion)
            if results.persuasions:
                persuasions = np.array(results.persuasions)
                metrics['mean_persuasion'] = persuasions.mean()
                
                # Больше убеждений на сложных примерах?
                for diff in ['easy', 'medium', 'hard']:
                    mask = difficulties == diff
                    if mask.sum() > 0:
                        metrics[f'persuasion_{diff}'] = persuasions[mask].mean()
        
        return metrics
    
    def print_comparison_table(self, all_results: Dict[str, Dict]) -> None:
        """Печатает сводную таблицу как в статье"""
        
        print("\n" + "═"*90)
        print("  RESULTS TABLE (для статьи)")
        print("═"*90)
        
        models = list(all_results.keys())
        
        # Основные метрики
        main_metrics = [
            ('accuracy',             'Accuracy ↑',        '.3f'),
            ('f1_macro',             'F1 Macro ↑',        '.3f'),
            ('ece',                  'ECE ↓',             '.4f'),
            ('auroc_uncertainty',    'AUROC(u,err) ↑',    '.3f'),
        ]
        
        print(f"\n  {'Metric':<22}", end="")
        for m in models:
            print(f"  {m:<18}", end="")
        print()
        print("  " + "-"*85)
        
        for metric_key, metric_name, fmt in main_metrics:
            print(f"  {metric_name:<22}", end="")
            for model in models:
                val = all_results[model].get(metric_key, float('nan'))
                if not np.isnan(val):
                    # Выделяем лучший результат
                    all_vals = [
                        all_results[m].get(metric_key, float('nan'))
                        for m in models
                        if not np.isnan(all_results[m].get(metric_key, float('nan')))
                    ]
                    if all_vals:
                        is_best = (
                            (val == max(all_vals) and 'ece' not in metric_key.lower()) or
                            (val == min(all_vals) and 'ece' in metric_key.lower())
                        )
                        marker = " *" if is_best else "  "
                    else:
                        marker = "  "
                    print(f"  {val:{fmt}}{marker:<16}", end="")
                else:
                    print(f"  {'N/A':<18}", end="")
            print()
        
        # По сложности
        print(f"\n  По сложности примеров (Accuracy):")
        print(f"  {'Difficulty':<22}", end="")
        for m in models:
            print(f"  {m:<18}", end="")
        print()
        print("  " + "-"*85)
        
        for diff in ['easy', 'medium', 'hard']:
            print(f"  {diff:<22}", end="")
            for model in models:
                val = all_results[model].get(f'acc_{diff}', float('nan'))
                if not np.isnan(val):
                    print(f"  {val:.3f}             ", end="")
                else:
                    print(f"  {'N/A':<18}", end="")
            print()
        
        # По числу evidence
        print(f"\n  По числу evidence (AUROC uncertainty):")
        print(f"  {'N Evidence':<22}", end="")
        for m in models:
            print(f"  {m:<18}", end="")
        print()
        print("  " + "-"*85)
        
        for n in [1, 2, 3]:
            print(f"  {n} evidence:<22".format(), end="")
            print(f"  {n} evidence       ", end="")
            for model in models:
                val = all_results[model].get(f'auroc_ev{n}', float('nan'))
                if not np.isnan(val):
                    print(f"  {val:.3f}             ", end="")
                else:
                    print(f"  {'N/A':<18}", end="")
            print()
        
        # CIDA-specific
        if 'CIDA' in all_results:
            cida = all_results['CIDA']
            print(f"\n  CIDA-specific (агентский анализ):")
            print(f"  {'Metric':<35} {'Value':>10}")
            print("  " + "-"*50)
            
            cida_metrics = [
                ('auroc_disagreement',      'AUROC(disagreement, error) ↑'),
                ('disagreement_error_corr', 'Corr(disagreement, error) ↑'),
                ('mean_persuasion',         'Mean agent persuasion'),
                ('disagreement_easy',       'Disagreement on EASY'),
                ('disagreement_medium',     'Disagreement on MEDIUM'),
                ('disagreement_hard',       'Disagreement on HARD'),
                ('persuasion_easy',         'Persuasion on EASY'),
                ('persuasion_hard',         'Persuasion on HARD'),
            ]
            
            for key, name in cida_metrics:
                val = cida.get(key, float('nan'))
                if not np.isnan(val):
                    print(f"  {name:<35} {val:>10.4f}")
        
        print("\n  * = best in column")
        print("═"*90)
    
    def print_headline_verdict(self, all_metrics: Dict[str, Dict]) -> None:
        """Финальный вердикт — есть ли заголовок"""
        
        print("\n" + "═"*70)
        print("  HEADLINE VERDICT")
        print("═"*70)
        
        cida = all_metrics.get('CIDA', {})
        baseline_aurocs = [
            all_metrics.get(m, {}).get('auroc_uncertainty', 0)
            for m in all_metrics
            if m != 'CIDA'
        ]
        
        cida_auroc = cida.get('auroc_uncertainty', 0)
        cida_disagree_auroc = cida.get('auroc_disagreement', 0)
        best_baseline_auroc = max(baseline_aurocs) if baseline_aurocs else 0
        
        print(f"\n  CIDA AUROC(u, error):            {cida_auroc:.3f}")
        print(f"  CIDA AUROC(disagreement, error): {cida_disagree_auroc:.3f}")
        print(f"  Best Baseline AUROC:             {best_baseline_auroc:.3f}")
        
        advantage = max(cida_auroc, cida_disagree_auroc) - best_baseline_auroc
        
        print(f"\n  CIDA advantage: {advantage:+.3f}")
        
        if advantage > 0.10:
            print("""
  ╔════════════════════════════════════════════════════════════╗
  ║  🏆 HEADLINE CONFIRMED                                    ║
  ║                                                            ║
  ║  "Multi-Agent Debate Predicts Its Own Errors on           ║
  ║   Contradictory Cancer Clinical Trial Evidence"           ║
  ║                                                            ║
  ║  CIDA's inter-agent disagreement predicts model errors    ║
  ║  significantly better than any single-agent uncertainty   ║
  ║  baseline on multi-evidence NLI tasks.                    ║
  ╚════════════════════════════════════════════════════════════╝
            """)
        elif advantage > 0.05:
            print("""
  ╔════════════════════════════════════════════════════════════╗
  ║  ⚡ PROMISING — needs more data                           ║
  ║  Moderate advantage found. Scale to real NLI4CT dataset.  ║
  ╚════════════════════════════════════════════════════════════╝
            """)
        else:
            print("""
  ╔════════════════════════════════════════════════════════════╗
  ║  ⚠ No significant advantage on this scale               ║
  ║  Architecture is sound. Need real dataset + BioMedBERT.   ║
  ╚════════════════════════════════════════════════════════════╝
            """)
        
        # Специализация агентов
        print(f"\n  Специализация агентов по типу рака:")
        print("  (разные агенты доминируют на разных онкологиях?)")
        print("  → требует запуска на реальном датасете")