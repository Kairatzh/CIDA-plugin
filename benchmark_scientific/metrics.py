"""
benchmark_scientific/metrics.py — Statistical, Efficiency & Calibration Analytics

Computes:
  1. Paired Bootstrap 95% Confidence Intervals & Contingency Analysis
  2. Reasoning Efficiency & Compute Economics (FLOPs & tokens per accuracy point)
  3. Calibration: Expected Calibration Error (ECE) and Brier Score
  4. Deliberation Uncertainty Predictability (Disagreement vs Correctness)
"""

import math
import random
from typing import List, Dict, Any, Tuple


# ==============================================================================
# 1. Statistical Significance & Paired Bootstrap Analysis
# ==============================================================================

def bootstrap_confidence_interval(
    values: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Computes mean and non-parametric bootstrap confidence interval [lower, upper].
    """
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    mean_val = sum(values) / n
    if n < 2:
        return mean_val, mean_val, mean_val

    boot_means = []
    for _ in range(n_resamples):
        sample = [random.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()

    alpha = 1.0 - confidence_level
    low_idx = int(alpha / 2.0 * n_resamples)
    high_idx = int((1.0 - alpha / 2.0) * n_resamples)
    low_idx = max(0, min(low_idx, n_resamples - 1))
    high_idx = max(0, min(high_idx, n_resamples - 1))

    return mean_val, boot_means[low_idx], boot_means[high_idx]


def paired_bootstrap_delta(
    hits_a: List[int],
    hits_b: List[int],
    n_resamples: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Calculates paired difference Delta = Acc(B) - Acc(A) with 95% Bootstrap CI.
    """
    assert len(hits_a) == len(hits_b), "Datasets must be paired and identical length"
    n = len(hits_a)
    if n == 0:
        return {"delta_mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    deltas = [b - a for a, b in zip(hits_a, hits_b)]
    delta_mean = sum(deltas) / n

    boot_deltas = []
    for _ in range(n_resamples):
        indices = [random.randint(0, n - 1) for _ in range(n)]
        b_mean = sum(deltas[i] for i in indices) / n
        boot_deltas.append(b_mean)
    boot_deltas.sort()

    alpha = 1.0 - confidence_level
    low_idx = max(0, int(alpha / 2.0 * n_resamples))
    high_idx = min(n_resamples - 1, int((1.0 - alpha / 2.0) * n_resamples))

    # Approximate 2-tailed p-value: proportion of bootstrap samples crossing 0
    if delta_mean > 0:
        p_val = sum(1 for d in boot_deltas if d <= 0) / n_resamples * 2.0
    else:
        p_val = sum(1 for d in boot_deltas if d >= 0) / n_resamples * 2.0
    p_val = min(1.0, max(0.0, p_val))

    return {
        "delta_mean": delta_mean * 100.0,
        "ci_lower": boot_deltas[low_idx] * 100.0,
        "ci_upper": boot_deltas[high_idx] * 100.0,
        "p_value": p_val
    }


def compute_contingency_table(hits_base: List[int], hits_cida: List[int]) -> Dict[str, Any]:
    """
    Builds 2x2 contingency table:
      - both_correct
      - base_only (Base Win)
      - cida_only (CIDA Win)
      - both_wrong
    """
    assert len(hits_base) == len(hits_cida)
    both_correct, base_wins, cida_wins, both_wrong = 0, 0, 0, 0
    for b, c in zip(hits_base, hits_cida):
        if b == 1 and c == 1:
            both_correct += 1
        elif b == 1 and c == 0:
            base_wins += 1
        elif b == 0 and c == 1:
            cida_wins += 1
        else:
            both_wrong += 1
    total = len(hits_base)
    win_rate = (cida_wins / (cida_wins + base_wins)) if (cida_wins + base_wins) > 0 else 0.5
    return {
        "both_correct": both_correct,
        "base_wins": base_wins,
        "cida_wins": cida_wins,
        "both_wrong": both_wrong,
        "total": total,
        "cida_win_rate_vs_base": win_rate * 100.0
    }


# ==============================================================================
# 2. Reasoning Efficiency & Resource Economics
# ==============================================================================

def calculate_reasoning_efficiency(accuracy_pct: float, avg_tokens: float) -> float:
    """Computes accuracy score per generated token (higher is more efficient)."""
    return (accuracy_pct / max(avg_tokens, 1.0)) * 100.0


def calculate_accuracy_gain_per_1k_tokens(
    acc_base: float,
    acc_target: float,
    tokens_target: float
) -> float:
    """Measures gain in accuracy per 1,000 tokens generated."""
    delta_acc = acc_target - acc_base
    return (delta_acc / max(tokens_target, 1.0)) * 1000.0


# ==============================================================================
# 3. Model Calibration: ECE & Brier Score
# ==============================================================================

def compute_calibration_metrics(
    confidences: List[float],
    accuracies: List[int],
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and Brier Score.
    """
    assert len(confidences) == len(accuracies)
    n = len(confidences)
    if n == 0:
        return {"ece": 0.0, "brier_score": 0.0, "bins": []}

    brier_score = sum((c - y) ** 2 for c, y in zip(confidences, accuracies)) / n

    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    bins = []
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        indices = [
            j for j in range(n)
            if (bin_lower <= confidences[j] < bin_upper) or (i == num_bins - 1 and confidences[j] == bin_upper)
        ]
        count = len(indices)
        if count > 0:
            bin_acc = sum(accuracies[j] for j in indices) / count
            bin_conf = sum(confidences[j] for j in indices) / count
            bin_error = abs(bin_acc - bin_conf)
            ece += (count / n) * bin_error
            bins.append({
                "bin_id": i,
                "range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
            })
        else:
            bins.append({
                "bin_id": i,
                "range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": (bin_lower + bin_upper) / 2.0,
            })

    return {
        "ece": ece * 100.0,
        "brier_score": brier_score,
        "bins": bins,
    }


# ==============================================================================
# 4. Deliberation Uncertainty Prediction (Disagreement vs Accuracy)
# ==============================================================================

def analyze_disagreement_predictability(
    final_disagreements: List[float],
    accuracies: List[int]
) -> Dict[str, Any]:
    """
    Empirically investigates whether inter-agent disagreement D_t is predictive
    of response correctness.
    """
    assert len(final_disagreements) == len(accuracies)
    low_d_hits, low_d_total = 0, 0       # D < 0.15
    med_d_hits, med_d_total = 0, 0       # 0.15 <= D < 0.35
    high_d_hits, high_d_total = 0, 0     # D >= 0.35

    for d, acc in zip(final_disagreements, accuracies):
        if d < 0.15:
            low_d_total += 1
            if acc == 1:
                low_d_hits += 1
        elif d < 0.35:
            med_d_total += 1
            if acc == 1:
                med_d_hits += 1
        else:
            high_d_total += 1
            if acc == 1:
                high_d_hits += 1

    acc_low = (low_d_hits / low_d_total * 100.0) if low_d_total > 0 else 0.0
    acc_med = (med_d_hits / med_d_total * 100.0) if med_d_total > 0 else 0.0
    acc_high = (high_d_hits / high_d_total * 100.0) if high_d_total > 0 else 0.0

    return {
        "low_disagreement": {"range": "< 0.15", "count": low_d_total, "accuracy_pct": acc_low},
        "medium_disagreement": {"range": "0.15 - 0.35", "count": med_d_total, "accuracy_pct": acc_med},
        "high_disagreement": {"range": ">= 0.35", "count": high_d_total, "accuracy_pct": acc_high},
        "is_predictive": (acc_low >= acc_med >= acc_high) if (low_d_total > 0 and high_d_total > 0) else True
    }
