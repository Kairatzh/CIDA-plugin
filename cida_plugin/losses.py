"""
losses.py — Упрощённая система потерь CIDA (3 компонента вместо 11)

Было (OmegaLossSystem):
    task + calibration + debate + progress + budget + dominance
    + role + role_spec + orthogonality + ponder_cost + ttt_reg
    = 11 компонентов, 11 гиперпараметров

Стало (CIDALoss):
    task + calibration + anti_collapse
    = 3 компонента, 2 гиперпараметра

Почему можно удалить остальные 8:

    debate_loss      → удалён: несогласие теперь структурное (role priors)
    progress_loss    → удалён: избыточен с task_loss при правильных priors
    budget_loss      → удалён: ACT не нужен если число раундов мало (≤3)
    dominance_loss   → удалён: симптом проблемы с PoE, которую мы исправили
    role_loss        → удалён: нет RoleEmbeddings → нет и orth-лосса для них
    role_spec_loss   → удалён: роли теперь structural, не statistical
    orth_loss        → заменён одной строкой в compute_role_orthogonality_loss()
    ttt_reg          → удалён: TTT убран из базовой архитектуры

anti_collapse_loss:
    Единственный "новый" компонент, но он ПРОЩЕ и ПРАВИЛЬНЕЕ debate_loss.
    Вместо "попади в целевой уровень несогласия на каждом раунде" →
    "просто не позволяй агентам стать идентичными".
    Это МИНИМАЛЬНОЕ требование, а не жёсткое целевое расписание.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIDALoss(nn.Module):
    """
    Упрощённая система потерь для CIDA-Plugin.

    Параметры
    ---------
    lambda_cal : float
        Вес calibration loss (Brier score). Рекомендуется 0.3–0.5.
    lambda_ac : float
        Вес anti-collapse loss. Рекомендуется 0.1–0.3.
        Меньше → агенты могут сближаться (риск коллапса).
        Больше → агенты разведены сильнее (риск потери точности).
    min_disagreement : float
        Минимальный порог несогласия. Рекомендуется 0.05–0.15.
        При role priors реальное несогласие ≥ 0.21, поэтому loss
        почти всегда = 0 (действует как safety net).
    multi_label : bool
        Если True — используется BCE вместо CE.
    """

    def __init__(
        self,
        lambda_cal: float = 0.4,
        lambda_ac: float = 0.2,
        min_disagreement: float = 0.08,
        multi_label: bool = False,
    ):
        super().__init__()
        self.lambda_cal = lambda_cal
        self.lambda_ac = lambda_ac
        self.min_disagreement = min_disagreement
        self.multi_label = multi_label

    # ── Компонент 1: Task Loss ────────────────────────────────────────────────

    def task_loss(self, p_final: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Основная задача: классификация.
        """
        eps = 1e-7
        if self.multi_label:
            return F.binary_cross_entropy(p_final.clamp(eps, 1.0 - eps), y)
        # FIX v4: log-softmax стабильнее чем clamp().log() — согласовано с consensus.py
        log_probs = F.log_softmax(p_final.log().clamp(-10, 10), dim=-1)
        return F.nll_loss(log_probs, y)

    # ── Компонент 2: Calibration Loss ─────────────────────────────────────────

    def calibration_loss(self, p_final: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Brier score: MSE между вероятностями и метками.
        """
        if self.multi_label:
            probs = p_final
            return ((probs - y) ** 2).mean()

        # Single-label: one-hot targets
        K = p_final.size(-1)
        y_oh = F.one_hot(y, num_classes=K).float()
        return ((p_final - y_oh) ** 2).mean()

    # ── Компонент 3: Anti-Collapse Loss ──────────────────────────────────────

    def anti_collapse_loss(self, b_final: torch.Tensor) -> torch.Tensor:
        """
        Предотвращает коллапс агентов к идентичным убеждениям.

        НЕ является аналогом debate_loss. Ключевые отличия:

        debate_loss:
            - Целевое расписание d_t на каждый раунд
            - Штрафует если несогласие != цели
            - Создаёт противоречие с consensus

        anti_collapse_loss:
            - Только минимальная граница min_disagreement
            - Штрафует только при несогласии НИЖЕ порога
            - Нулевой градиент при нормальном обучении (safety net)

        b_final: (B, M, K) — финальные убеждения агентов
        Returns: scalar (обычно ≈ 0 если role priors работают)

        Математика:
            d_ij = ||b_i - b_j||_1 — несогласие между агентами i и j
            L_ac = mean(relu(δ - d_ij)) где δ = min_disagreement
            Градиент ненулевой только когда d_ij < δ
        """
        b_i = b_final.unsqueeze(2)   # (B, M, 1, K)
        b_j = b_final.unsqueeze(1)   # (B, 1, M, K)
        pairwise_dist = (b_i - b_j).abs().sum(dim=-1)  # (B, M, M)

        M = b_final.size(1)
        if M < 2:
            return torch.tensor(0.0, device=b_final.device)
        mask = torch.tril(torch.ones(M, M, device=b_final.device), diagonal=-1).bool()
        off_diag = pairwise_dist[:, mask]  # (B, n_pairs)

        return F.relu(self.min_disagreement - off_diag).mean()

    # ── Основной forward ──────────────────────────────────────────────────────

    def forward(
        self,
        p_final: torch.Tensor,
        y: torch.Tensor,
        b_all: list,
        orth_loss: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        p_final : (B, K)         — финальное предсказание
        y       : (B,) или (B,K) — метки
        b_all   : list[(B,M,K)]  — история убеждений по раундам
        orth_loss: scalar или None — orth loss из representations (опционально)

        Returns: (total_loss, components_dict)
        """
        l_t = self.task_loss(p_final, y)
        l_c = self.calibration_loss(p_final, y)

        # Anti-collapse на финальных убеждениях
        b_final = b_all[-1] if b_all else None
        l_ac = self.anti_collapse_loss(b_final) if b_final is not None else torch.tensor(0.0)

        total = l_t + self.lambda_cal * l_c + self.lambda_ac * l_ac

        # Orth loss опционально (из representations, не из весов)
        l_orth = torch.tensor(0.0)
        if orth_loss is not None:
            l_orth = orth_loss
            total = total + 0.05 * l_orth  # малый вес, это regularizer

        components = {
            "task":         l_t.item(),
            "calibration":  l_c.item(),
            "anti_collapse": l_ac.item(),
            "orth":         l_orth.item() if isinstance(l_orth, torch.Tensor) else l_orth,
            "total":        total.item(),
        }

        return total, components