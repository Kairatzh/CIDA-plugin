"""metrics.py — калибровка и разделимость уверенности (без sklearn)."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE: Σ_m |B_m|/n · |acc(B_m) − conf(B_m)| по равным интервалам уверенности."""
    confidence = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC через статистику Манна–Уитни со средними рангами для совпадений; nan, если класс один."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
