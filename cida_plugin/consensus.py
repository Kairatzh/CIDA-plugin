"""
consensus.py — Упрощённый ConsensusAggregator

Ключевые исправления:

1. УБРАН ReliabilityTracker
   Причина: круговая зависимость rho ← accuracy ← predictions ← weights(rho).
   На малых датасетах вырождается. Без теоретической гарантии сходимости.
   Замена: статические веса ролей (обоснованные a priori).

2. ИСПРАВЛЕН ConsensusAggregator
   БЫЛО: PoE = log p ∝ Σ w_i · log(b_i)
   ПРОБЛЕМА: PoE предполагает НЕЗАВИСИМЫХ экспертов.
   Агенты CIDA КОРРЕЛИРОВАНЫ — общий энкодер + communication между агентами.
   PoE на коррелированных агентах → усиление shared bias → overconfidence.

   СТАЛО: Weighted Mean + Disagreement-as-Uncertainty
   p_final = Σ w_i · b_i  (взвешенное среднее)
   uncertainty = std(b) по агентам

   Математическое обоснование:
   Для коррелированных экспертов с ковариационной матрицей Σ:
       Var(Σ w_i · b_i) = w^T · Σ · w  (правильно учитывает корреляции)
       Var(PoE(b_i)) ≠ w^T · Σ · w     (PoE неверно предполагает Σ = σ²I)
   При положительной корреляции: PoE под-оценивает variance → overconfidence.

3. УПРОЩЁН интерфейс
   Убраны: epoch_fraction, temperature scheduling, normalize_components флаг.
   Всё это компенсировало нестабильность PoE на коррелированных агентах.
   При правильной агрегации это не нужно.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Статические веса ролей (a priori обоснованы ролями)
# Интегратор = наибольший вес (его роль — синтезировать)
# Скептик = наименьший вес (его роль — поднимать вопросы, не решать)
ROLE_WEIGHTS = torch.tensor([0.28, 0.28, 0.12, 0.32])  # P, D, S, I (сумма = 1.0)


class ConsensusAggregator(nn.Module):
    """
    Weighted Mean консенсус для КОРРЕЛИРОВАННЫХ агентов.

    v4: Dynamic Trust Weights (CortexDebate MDM, ACL 2025).
    Статические веса заменяются на EMA trust — вес агента растёт
    когда его предсказания совпадают с финальным консенсусом.

    trust_i += γ·(1[b_i_correct] - trust_i) per batch, detached from main graph
    """

    def __init__(
        self,
        num_agents: int = 4,
        multi_label: bool = False,
        use_dynamic_trust: bool = False,
        trust_ema_gamma: float = 0.9,
    ):
        super().__init__()
        self.multi_label = multi_label
        self.use_dynamic_trust = use_dynamic_trust
        self.trust_ema_gamma = trust_ema_gamma

        # Статические веса ролей — обучаемые для адаптации к задаче
        init_weights = ROLE_WEIGHTS[:num_agents].clone()
        init_weights = init_weights / init_weights.sum()
        self.role_weights = nn.Parameter(init_weights)

        # v4: Dynamic trust (EMA buffer, не участвует в градиентах)
        if use_dynamic_trust:
            self.register_buffer(
                'trust_scores',
                torch.ones(num_agents) / num_agents
            )

    @property
    def weights(self) -> torch.Tensor:
        """Нормализованные веса ролей (в simplex)."""
        if self.use_dynamic_trust and hasattr(self, 'trust_scores'):
            return self.trust_scores
        return F.softmax(self.role_weights, dim=0)  # (M,)

    def update_trust(self, b: torch.Tensor, p_final: torch.Tensor):
        """
        Обновляет trust scores на основе согласованности агента с консенсусом.
        Вызывается ПОСЛЕ forward(), detached от основного графа.

        b: (B, M, K) — убеждения агентов
        p_final: (B, K) — финальный консенсус
        """
        if not self.use_dynamic_trust:
            return
        with torch.no_grad():
            # Per-agent correctness = 1 - L1(b_i, p_final)
            correctness = 1.0 - (b - p_final.unsqueeze(1)).abs().mean(dim=(0, 2))  # (M,)
            gamma = self.trust_ema_gamma
            self.trust_scores = gamma * self.trust_scores + (1 - gamma) * correctness
            self.trust_scores = self.trust_scores / self.trust_scores.sum()

    def forward(
        self,
        b: torch.Tensor,          # (B, M, K)
        u: torch.Tensor = None,   # (B, M, 1) — опционально
        u_epi: torch.Tensor = None,   # (B, M, 1) — v4: epistemic
        u_alea: torch.Tensor = None,  # (B, M, 1) — v4: aleatoric
    ) -> tuple:
        """
        Returns:
            p_final:     (B, K)  — консенсус
            uncertainty: (B,)   — мера неопределённости
            disagreement:(B,)   — мера несогласия агентов
        """
        B, M, K = b.shape
        w = self.weights  # (M,) — нормализованные role weights

        # ── Взвешенное среднее (корректно для коррелированных агентов) ────────
        p_final = (w.view(1, M, 1) * b).sum(dim=1)  # (B, K)

        if not self.multi_label:
            # FIX v4: Нормализация через log-softmax — численно стабильна
            p_final = F.softmax(p_final.log().clamp(-10, 10), dim=-1)

        # ── Несогласие = std убеждений по агентам ─────────────────────────────
        disagreement = b.std(dim=1).mean(dim=-1)  # (B,)

        # ── Неопределённость = несогласие + собственная uncertainty агентов ──
        if u is not None:
            agent_u = (w.view(1, M, 1) * u).sum(dim=1).squeeze(-1)  # (B,)
            uncertainty = 0.5 * disagreement + 0.5 * agent_u
        else:
            uncertainty = disagreement

        # v4: Обновляем dynamic trust (если включён)
        if self.training and self.use_dynamic_trust:
            self.update_trust(b, p_final)

        return p_final, uncertainty, disagreement


class HaltingPredictor(nn.Module):
    """
    Предсказатель остановки (упрощённый).
    Используется для ACT: когда агенты достаточно согласны — останавливаемся.

    В отличие от оригинала, решение основано на НЕСОГЛАСИИ (наблюдаемо),
    а не на выученном MLP поверх всех состояний (трудно интерпретировать).
    """

    def __init__(self, agreement_threshold: float = 0.15):
        """
        agreement_threshold: при несогласии ниже этого порога — останавливаемся.
        """
        super().__init__()
        self.threshold = agreement_threshold
        # Небольшой выученный bias для подстройки порога под задачу
        self.bias = nn.Parameter(torch.tensor(0.0))

    def should_halt(self, disagreement: torch.Tensor) -> torch.Tensor:
        """
        disagreement: (B,)
        Returns: (B,) bool — True если стоит остановиться
        """
        effective_threshold = self.threshold + self.bias.sigmoid() * 0.1
        return disagreement < effective_threshold

    def halting_probability(self, disagreement: torch.Tensor) -> torch.Tensor:
        """
        Мягкая версия: вероятность остановки ∈ [0, 1].
        disagreement: (B,)
        Returns: (B,)
        """
        effective_threshold = self.threshold + self.bias.sigmoid() * 0.1
        # Sigmoid(-scale * (d - threshold)): высокая вероятность при d < threshold
        return torch.sigmoid(-10.0 * (disagreement - effective_threshold))