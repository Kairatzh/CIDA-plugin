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
    Смешивает data-driven belief с role-specific prior (обобщённая версия для любого K).
    """
    role = role_idx % 4
    if role == 3: # Интегратор: без prior
        return b_data

    K = b_data.size(-1)
    device = b_data.device
    dtype = b_data.dtype

    if multi_label or K == 1:
        prior_val = ROLE_PRIOR_POSITIVE[role]
        prior = torch.full_like(b_data, prior_val)
    elif K == 2:
        prior_val = ROLE_PRIOR_POSITIVE[role]
        prior = torch.tensor(
            [1.0 - prior_val, prior_val],
            device=device,
            dtype=dtype,
        ).expand_as(b_data)
    else:
        # Multi-class
        if role == 0:   # Prosecutor: biased to positive class
            prior_val = 0.85
            uniform_val = (1.0 - prior_val) / max(K - 1, 1)
            prior = torch.full_like(b_data, uniform_val)
            prior = prior.clone()
            prior[:, positive_class_idx] = prior_val
        elif role == 1: # Defender: biased against positive class (pro-alternative classes)
            prior_val = 0.15
            uniform_val = (1.0 - prior_val) / max(K - 1, 1)
            prior = torch.full_like(b_data, uniform_val)
            prior = prior.clone()
            prior[:, positive_class_idx] = prior_val
        elif role == 2: # Skeptic: neutral / high uncertainty (uniform)
            prior = torch.full_like(b_data, 1.0 / K)
            
        prior = prior / prior.sum(dim=-1, keepdim=True)

    return (1.0 - ROLE_PRIOR_BLEND) * b_data + ROLE_PRIOR_BLEND * prior


def apply_role_priors_batched(
    b_data: torch.Tensor,
    M: int,
    multi_label: bool = False,
    positive_class_idx: int = 1,
) -> torch.Tensor:
    """
    Векторизованная версия apply_role_prior для батча агентов сразу (B, M, K).
    """
    B, _, K = b_data.shape
    device = b_data.device
    dtype = b_data.dtype
    
    priors = torch.zeros((M, K), device=device, dtype=dtype)
    blend_mask = torch.full((M, 1), ROLE_PRIOR_BLEND, device=device, dtype=dtype)
    
    for i in range(M):
        role = i % 4
        if role == 3: # Интегратор: без prior
            blend_mask[i, 0] = 0.0
            continue
            
        if multi_label or K == 1:
            priors[i, :] = ROLE_PRIOR_POSITIVE[role]
        elif K == 2:
            priors[i, 0] = 1.0 - ROLE_PRIOR_POSITIVE[role]
            priors[i, 1] = ROLE_PRIOR_POSITIVE[role]
        else:
            # Multi-class
            if role == 0:   # Prosecutor: biased to positive class
                prior_val = 0.85
                uniform_val = (1.0 - prior_val) / max(K - 1, 1)
                priors[i, :] = uniform_val
                priors[i, positive_class_idx] = prior_val
            elif role == 1: # Defender: biased against positive class
                prior_val = 0.15
                uniform_val = (1.0 - prior_val) / max(K - 1, 1)
                priors[i, :] = uniform_val
                priors[i, positive_class_idx] = prior_val
            elif role == 2: # Skeptic: neutral / uniform
                priors[i, :] = 1.0 / K
                
            priors[i] = priors[i] / priors[i].sum()
            
    priors = priors.unsqueeze(0)        # (1, M, K)
    blend_mask = blend_mask.unsqueeze(0) # (1, M, 1)
    
    return (1.0 - blend_mask) * b_data + blend_mask * priors



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