"""Визуализация «магии» CIDA: голоса агентов и индикатор неуверенности."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from models import CIDAInferenceResult

AGENT_COLORS = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]


def plot_agent_votes(result: CIDAInferenceResult) -> plt.Figure:
    """Столбчатая диаграмма: уверенность / P(положительный) каждого агента."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(result.agent_names))

    # Для binary: P(положительный); для multi-class — уверенность в top-1
    if len(result.class_names) == 2:
        values = result.agent_prob_positive
        ylabel = "P(положительный класс)"
    else:
        values = result.agent_votes
        ylabel = "Уверенность агента"

    bars = ax.bar(
        x,
        values,
        color=AGENT_COLORS[: len(result.agent_names)],
        edgecolor="white",
        linewidth=1.2,
        width=0.6,
    )

    for bar, label, val in zip(bars, result.agent_labels, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{label}\n{val:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(result.agent_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Распределение мнений 4 агентов", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="Порог 50%")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def uncertainty_gauge_html(result: CIDAInferenceResult) -> str:
    """HTML-шкала неуверенности: красная зона = нужна проверка человеком."""
    score = min(max(result.uncertainty, 0.0), 1.0)
    pct = int(score * 100)

    if result.needs_review:
        color = "#e74c3c"
        status = "⚠ Требуется проверка человеком"
        bg = "#fdecea"
        border = "#e74c3c"
    elif score > 0.08:
        color = "#f39c12"
        status = "Умеренная неуверенность"
        bg = "#fef9e7"
        border = "#f39c12"
    else:
        color = "#27ae60"
        status = "Агенты согласны — высокая уверенность"
        bg = "#eafaf1"
        border = "#27ae60"

    return f"""
    <div style="
        font-family: system-ui, sans-serif;
        padding: 20px;
        border-radius: 12px;
        background: {bg};
        border: 2px solid {border};
    ">
        <div style="font-size: 18px; font-weight: 700; margin-bottom: 8px; color: {color};">
            {status}
        </div>
        <div style="font-size: 13px; color: #555; margin-bottom: 12px;">
            Несогласие агентов: <b>{result.disagreement:.3f}</b> &nbsp;|&nbsp;
            Неуверенность: <b>{result.uncertainty:.3f}</b> &nbsp;|&nbsp;
            Раундов дебатов: <b>{result.rounds_used}</b>
        </div>
        <div style="
            height: 28px;
            border-radius: 14px;
            background: linear-gradient(to right, #27ae60 0%, #f39c12 50%, #e74c3c 100%);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                left: {pct}%;
                top: -4px;
                width: 4px;
                height: 36px;
                background: #2c3e50;
                border-radius: 2px;
                transform: translateX(-50%);
                box-shadow: 0 0 6px rgba(0,0,0,0.4);
            "></div>
        </div>
        <div style="
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #777;
            margin-top: 4px;
        ">
            <span>Низкая (согласны)</span>
            <span>{pct}%</span>
            <span>Высокая (спорят)</span>
        </div>
    </div>
    """
