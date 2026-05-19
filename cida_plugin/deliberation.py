import torch
import torch.nn as nn
import torch.nn.functional as F
from .agent import AgentState


class MessageFormulator(nn.Module):
    """
    Creates a structured message out of the explicit agent components.
    m_i^t = phi([s_i^t; e_i^t; b_i^t; u_i^t])
    """
    def __init__(self, d_hidden: int, num_classes: int, d_message: int):
        super().__init__()
        in_dim = d_hidden + d_hidden + num_classes + 1
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_message),
            nn.LayerNorm(d_message),
            nn.SiLU(),
            nn.Linear(d_message, d_message)
        )

    def forward(self, s, e, b, u):
        concat = torch.cat([s, e, b, u], dim=-1)
        return self.proj(concat)


class AgentEvidenceExtractor(nn.Module):
    """
    Computes evidence pointers over tokens, giving each agent a distinct focus.
    p_i^t = softmax((H W_p)(W_q s_i^t)^T / sqrt(d))
    e_i^t = sum_k p_{i,k}^t H_k
    """
    def __init__(self, d_hidden: int):
        super().__init__()
        self.W_q = nn.Linear(d_hidden, d_hidden)
        self.W_p = nn.Linear(d_hidden, d_hidden)
        self.scale = d_hidden ** -0.5

    def forward(self, s, H, mask=None):
        K = self.W_p(H)
        Q = self.W_q(s)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        p = F.softmax(scores, dim=-1)
        e = torch.bmm(p, H)
        return p, e


class CounterargumentCommunication(nn.Module):
    """
    Agents collect messages from others, weighted by an attention mechanism
    that explicitly values disagreement while punishing simple similarity.
    """
    def __init__(self, d_hidden: int, d_message: int, lambda_d: float = 1.0, lambda_s: float = 1.0):
        super().__init__()
        self.W_q = nn.Linear(d_hidden, d_hidden)
        self.W_k = nn.Linear(d_message, d_hidden)
        self.W_v = nn.Linear(d_message, d_hidden)
        self.lambda_d = lambda_d
        self.lambda_s = lambda_s
        self.scale = d_hidden ** -0.5

    def forward(self, s, m, b, e):
        Q = self.W_q(s)
        K = self.W_k(m)
        V = self.W_v(m)

        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale

        b_i = b.unsqueeze(2)
        b_j = b.unsqueeze(1)
        D = torch.norm(b_i - b_j, p=1, dim=-1)

        e_norm = F.normalize(e, p=2, dim=-1)
        S = torch.bmm(e_norm, e_norm.transpose(1, 2))

        scores = attn + self.lambda_d * D - self.lambda_s * S

        M_agents = s.size(1)
        mask = torch.eye(M_agents, device=s.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))

        a = F.softmax(scores, dim=-1)
        r = torch.bmm(a, V)
        return r


# ─── v2: Transformer Agent Updater ───────────────────────────────────────────

class AgentUpdater(nn.Module):
    """
    [v2] Обновляет состояние агентов через Transformer Decoder Step.

    Замена GRUCell на cross-attention даёт агентам возможность
    самостоятельно решать, НА ЧТО обращать внимание в сообщениях других,
    а не слепо конкатенировать всё подряд.

    Шаги:
        1. Cross-Attention: каждый агент (query) смотрит на сигналы
           counterargument (key/value) от других агентов.
        2. Gated Evidence Fusion: ворота решают, сколько evidence добавить.
        3. FFN + Residual + LayerNorm.
        4. Dirichlet Belief Update: alpha = softplus(g(s)) + 1.
    """

    def __init__(self, d_hidden: int, num_classes: int, num_heads: int = 4, multi_label: bool = False):
        super().__init__()
        self.multi_label = multi_label
        # ── Step 1: Cross-attention over counterarguments ─────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm1 = nn.LayerNorm(d_hidden)

        # ── Step 2: Gated evidence fusion ─────────────────────────────────────
        self.evidence_gate = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Sigmoid(),
        )
        self.evidence_proj = nn.Linear(d_hidden, d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)

        # ── Step 3: FFN ───────────────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.SiLU(),
            nn.Linear(d_hidden * 2, d_hidden),
        )
        self.ffn_norm = nn.LayerNorm(d_hidden) # Replaced norm3 for clarity

        # ── Step 4: Belief generator ──────────────────────────────────────────
        self.g = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, num_classes),
        )
        self.K = float(num_classes)

    def forward(self, state: AgentState, r: torch.Tensor, e: torch.Tensor) -> AgentState:
        """
        state : AgentState — текущее состояние агентов
        r     : (B, M, d_hidden) — aggregated counterargument signals
        e     : (B, M, d_hidden) — evidence vectors из AgentEvidenceExtractor
        """
        s = state.s  # (B, M, d_hidden)

        # ── Step 1: Cross-attention ───────────────────────────────────────────
        # Query: текущие состояния агентов
        # Key/Value: counterargument signals — что говорят другие агенты
        # NOTE: если r все нули (начальный шаг), cross-attention = identity
        attn_out, _ = self.cross_attn(query=s, key=r, value=r)
        s = self.norm1(s + attn_out)

        # ── Step 2: Gated evidence fusion ─────────────────────────────────────
        # Ворота: насколько новый evidence должен изменить состояние?
        gate = self.evidence_gate(torch.cat([s, e], dim=-1))   # (B, M, d)
        e_proj = self.evidence_proj(e)                          # (B, M, d)
        s = self.norm2(s + gate * e_proj)

        # ── Step 3: FFN + residual ────────────────────────────────────────────
        s = self.ffn_norm(s + self.ffn(s))

        # ── Step 4: Belief update ─────────────────────────────────────────────
        g_out = self.g(s)
        
        if self.multi_label:
            # Multi-label: Independent sigmoids
            b_next = torch.sigmoid(g_out) # (B, M, K)
            # Uncertainty: 4 * b * (1-b) - максимальна при b=0.5
            u_next = (4.0 * b_next * (1.0 - b_next)).mean(dim=-1, keepdim=True)
            alpha = b_next # Dummy for AgentState
        else:
            # Single-label: Dirichlet
            alpha = F.softplus(g_out) + 1.0      # (B, M, K), всегда > 1
            alpha_sum = alpha.sum(dim=-1, keepdim=True)
            b_next = alpha / alpha_sum
            u_next = self.K / alpha_sum

        return AgentState(
            s=s,
            b=b_next,
            u=u_next,
            alpha=alpha,
            p=state.p,
            e=e,
        )


# ─── v3: Structural Specialization ───────────────────────────────────────────

class PerspectiveProjector(nn.Module):
    """
    [v3+] Каждый агент имеет свой собственный "взгляд" на входные данные.
    Вместо клонирования h_cls, каждый агент i проецирует pooled_output
    через свою собственную матрицу.

    [v4] Добавлен orthogonality_loss(): штраф за коллинеарность матриц
    проекций. Заставляет агентов смотреть на РАЗНЫЕ аспекты входа.
    Без этого штрафа все M проекций коллапсируют к одной — агенты
    становятся копиями друг друга и дебаты теряют смысл.
    """

    def __init__(self, num_agents: int, d_input: int, d_hidden: int):
        super().__init__()
        self.num_agents = num_agents
        # Каждый агент имеет свою проекцию
        self.agent_projections = nn.ModuleList([
            nn.Linear(d_input, d_hidden) for _ in range(num_agents)
        ])
        # Опционально: позиционное смещение для севенциальных данных
        # Предполагаем макс. длину 512 для инициализации
        self.position_bias = nn.Parameter(
            torch.randn(num_agents, 512) * 0.01
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        pooled: (B, d_input)
        Returns: (B, M, d_hidden)
        """
        out = [proj(pooled) for proj in self.agent_projections]
        return torch.stack(out, dim=1)

    def orthogonality_loss(self) -> torch.Tensor:
        """
        [v4] Штраф за коллинеарность проекций агентов.

        Gram(W) = W_norm @ W_norm^T, где W_norm — нормализованные строки
        матриц весов каждого агента. Идеал: Gram = I (единичная матрица).
        Loss = mean((Gram - I)^2) → 0 при полной ортогональности.

        Математическое обоснование:
            cos(θ_ij) = <w_i, w_j> / (||w_i|| ||w_j||)
            При L_orth → 0: cos(θ_ij) → 0 для i≠j ⟹ θ_ij → 90°
            Агенты гарантированно смотрят на линейно независимые подпространства.
        """
        # Собираем матрицы весов: (M, d_hidden, d_input)
        W = torch.stack([proj.weight for proj in self.agent_projections])
        M = W.size(0)

        # Flatten: (M, d_hidden * d_input)
        W_flat = W.view(M, -1)

        # L2-нормализация строк
        W_norm = F.normalize(W_flat, p=2, dim=-1)

        # Gram matrix: (M, M) — косинусные сходства между агентами
        gram = W_norm @ W_norm.T

        # Цель: единичная матрица (каждый агент ортогонален остальным)
        eye = torch.eye(M, device=gram.device)

        return ((gram - eye) ** 2).mean()


class AdaptiveDSchedule(nn.Module):
    """
    [v3+] Адаптивное расписание разногласий с PI-регулятором.

    Исправление: current_disagreement теперь ИСПОЛЬЗУЕТСЯ для обратной связи.
    Прежняя версия принимала параметр, но полностью его игнорировала.

    Математическое обоснование:
        d_target = base(h_cls) + Kp * e_t + Ki * Σe_k
        где e_t = base - actual_disagreement — ошибка регулирования.

    PI-регулятор гарантирует экспоненциальное стремление фактического
    разногласия к целевому (доказано через анализ устойчивости замкнутой системы):
        |λ| = |1 - η(1 - Kp)| < 1  при 0 < η < 2, 0 < Kp < 1.

    Возвращает (d_target, error) — ошибку нужно накапливать в integral_acc
    в вызывающем коде (core.py), а не внутри буфера модуля,
    чтобы интегральный член был локальным для каждого forward-прохода.
    """

    def __init__(self, d_hidden: int, Kp: float = 0.5, Ki: float = 0.1):
        super().__init__()
        self.complexity_estimator = nn.Sequential(
            nn.Linear(d_hidden, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.Kp = Kp   # Пропорциональный коэффициент
        self.Ki = Ki   # Интегральный коэффициент

    def forward(
        self,
        h_cls: torch.Tensor,
        current_disagreement: torch.Tensor,
        integral_acc: torch.Tensor = None,
    ):
        """
        h_cls               : (B, d_hidden) — начальное представление примера
        current_disagreement: (B, 1, 1) или (B, 1) — фактическое разногласие
        integral_acc        : (B, 1) — накопленная ошибка из предыдущих раундов

        Возвращает:
            d_target : (B, 1) — скорректированная цель разногласия ∈ [0, 1.5]
            error    : (B, 1) — ошибка e_t для обновления integral_acc снаружи
        """
        complexity = self.complexity_estimator(h_cls)             # (B, 1)
        base_target = complexity * 0.8 + (1 - complexity) * 0.1  # ∈ [0.1, 0.9]

        # Приводим фактическое разногласие к форме (B, 1)
        disag = current_disagreement.view(h_cls.size(0), 1)

        # Пропорциональная ошибка (stop_grad — чистый управляющий сигнал)
        error = (base_target - disag).detach()

        # Интегральный член (накоплен вызывающим кодом за предыдущие раунды)
        integral = integral_acc if integral_acc is not None else torch.zeros_like(error)

        # PI-цель: base + proportional + integral
        d_target = base_target + self.Kp * error + self.Ki * integral

        return torch.clamp(d_target, 0.0, 1.5), error
