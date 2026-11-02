"""
report.py — сводная таблица прогонов (среднее ± std по зёрнам для одинаковых имён без суффикса _sN).

    python -m kairos_loop.report runs/*/result.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

import numpy as np

COLUMNS = [
    ("exact_acc", "Exact"),
    ("exact_acc_adaptive", "Exact (adaptive)"),
    ("mean_steps_adaptive", "Steps"),
    ("halt_auroc", "Halt AUROC"),
    ("halt_ece", "Halt ECE"),
    ("agreement_auroc", "Agree AUROC"),
]


def _fmt(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"


def main(paths):
    groups = defaultdict(list)
    for path in paths:
        with open(path) as f:
            r = json.load(f)
        groups[re.sub(r"_s\d+$", "", r["name"])].append(r)

    header = ["Run", "Seeds", "Params"] + [title for _, title in COLUMNS]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name, runs in sorted(groups.items()):
        row = [name, str(len(runs)), f"{runs[0]['params'] / 1e6:.2f}M"]
        row += [_fmt([r["final"].get(key) for r in runs]) for key, _ in COLUMNS]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
