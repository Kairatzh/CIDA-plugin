"""
agent.py — CIDA-Plugin Agent (упрощённая версия)

Ключевое изменение:
    БЫЛО:  RoleEmbeddings (аддитивный bias) + debate_loss + role_spec_loss + orth_loss
    СТАЛО: ROLE_PRIORS — структурные априорные убеждения агентов

Почему это лучше:
    RoleEmbeddings создаёт СТАТИСТИЧЕСКОЕ различие (агенты начинают чуть
    по-разному и потом схожи). Role priors создают СТРУКТУРНОЕ различие:

    Прокурор  → P(y=1) = 0.85 до просмотра данных
    Защитник  → P(y=1) = 0.15 до просмотра данных
    Скептик   → P(y=1) = 0.50 (максимальная неопределённость)
    Интегратор → P(y=1) = данные (без prior, объективен)

    Это байесовски обоснованная альтернатива: агенты представляют разные
    prior beliefs, а консенсус — это posterior после объединения.

    Математически гарантировано:
        |b_0 - b_1| ≥ BLEND * |P_0 - P_1| = 0.3 * 0.7 = 0.21 > 0
    То есть несогласие является СВОЙСТВОМ архитектуры, а не результатом loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ── Роли агентов: структурные prior beliefs ───────────────────────────────────
#
# Для binary/multi-label: prior = вероятность positive класса
# Для multi-class: prior = вектор вероятностей классов
#
# BLEND = насколько сильно prior смешивается с данными
#   0.0 → prior игнорируется (все агенты одинаковы)
#   0.3 → 30% prior, 70% данные (рекомендуется)
#   1.0 → агент игнорирует данные (только prior)

ROLE_PRIOR_POSITIVE = [
    0.85,  # Прокурор: склонен считать что патология есть
    0.15,  # Защитник: склонен считать что патологии нет
    0.50,  # Скептик: максимальная неопределённость
    None,  # Интегратор: без prior, смотрит объективно
]

ROLE_PRIOR_BLEND = 0.30  # Сила prior относительно данных


@dataclass
class AgentState:
    """Явное представление состояния агента."""
    s: torch.Tensor       # (B, d)   - Hidden state
    b: torch.Tensor       # (B, K)   - Belief (posterior после prior blending)
    u: torch.Tensor       # (B, 1)   - Total uncertainty (backward compat)
    p: torch.Tensor       # (B, n)   - Evidence pointer
    e: torch.Tensor       # (B, d)   - Evidence vector
    alpha: torch.Tensor   # (B, K)   - Dirichlet parameters
    # v4: Разделение неопределённости (Josang 2002 + Meinert 2024)
    u_epi: torch.Tensor = None    # (B, 1)   - Epistemic: K²/alpha_sum² (vacuity)
    u_alea: torch.Tensor = None   # (B, 1)   - Aleatoric: 1 - max(b) (data noise)


def apply_role_prior(
    b_data: torch.Tensor,
    role_idx: int,
    multi_label: bool = False,
    positive_class_idx: int = 1,
) -> torch.Tensor:
    """
    Смешивает data-driven belief с role-specific prior.

    b_data: (B, K) — убеждение из данных
    role_idx: индекс роли агента
    positive_class_idx: индекс "положительного" класса для multi-class задач
                       (по умолчанию 1, но может быть 0, 2 и т.д.)

    Returns: (B, K) — posterior после применения prior

    Математика (binary case):
        b_posterior = (1 - λ) · b_data + λ · b_prior
        где λ = ROLE_PRIOR_BLEND

    Это эквивалентно Bayesian belief blending с фиксированным prior.
    Чем сильнее BLEND, тем больше роль определяет начальную позицию агента.
    """
    prior_val = ROLE_PRIOR_POSITIVE[role_idx % 4]

    if prior_val is None:
        # Интегратор: без prior
        return b_data

    K = b_data.size(-1)

    if multi_label or K == 1:
        # Каждый выход — независимая binary классификация
        prior = torch.full_like(b_data, prior_val)
    elif K == 2:
        # Binary single-label
        prior = torch.tensor(
            [1.0 - prior_val, prior_val],
            device=b_data.device,
            dtype=b_data.dtype,
        ).expand_as(b_data)
    else:
        # Multi-class: prior_val для "positive" класса, равномерно для остальных
        uniform_val = (1.0 - prior_val) / max(K - 1, 1)
        prior = torch.full_like(b_data, uniform_val)
        prior = prior.clone()
        prior[:, positive_class_idx] = prior_val  # указанный класс = "positive"
        # FIX v4: Явная нормализация — без неё при prior_val=0.50, K=14
        # сумма = ~7.0 вместо 1.0, что ломает blending и обучение на NIH-14
        prior = prior / prior.sum(dim=-1, keepdim=True)

    return (1.0 - ROLE_PRIOR_BLEND) * b_data + ROLE_PRIOR_BLEND * prior


def compute_role_orthogonality_loss(agent_states: torch.Tensor) -> torch.Tensor:
    """
    Мягкая ортогональность через ПРЕДСТАВЛЕНИЯ, а не через веса.

    БЫЛО: orth_loss через Gram(weights) — веса могут быть ортогональны,
          но representations при этом коллапсируют (если вход низкоранговый).

    СТАЛО: orth_loss через Gram(representations) — напрямую измеряем
           то, что нас реально интересует.

    agent_states: (B, M, d) — hidden states агентов
    Returns: scalar — потери (0 при полной ортогональности)

    Математически:
        G_ij = <s_i, s_j> / (||s_i|| · ||s_j||) — косинусное сходство
        Loss = mean((G - I)²) → 0 при G = I
    """
    B, M, d = agent_states.shape
    # Усредняем по батчу, нормализуем
    s_mean = agent_states.mean(0)           # (M, d)
    s_norm = F.normalize(s_mean, p=2, dim=-1)  # (M, d)
    gram = s_norm @ s_norm.T                # (M, M)
    eye = torch.eye(M, device=gram.device)
    return ((gram - eye) ** 2).mean()


class RoleNames:
    """Имена агентов для диагностики."""
    NAMES = ["Prosecutor", "Defender", "Skeptic", "Integrator"]

    @classmethod
    def get(cls, idx: int) -> str:
        return cls.NAMES[idx % 4]