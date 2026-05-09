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

    def __init__(self, d_hidden: int, num_classes: int, num_heads: int = 4):
        super().__init__()
        # ── Step 1: Cross-attention over counterarguments ─────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm1 = nn.LayerNorm(d_hidden)

        # ── Step 2: Gated evidence fusion ─────────────────────────────────────
        # Gate decides how much of the evidence to absorb into agent state
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
        self.norm3 = nn.LayerNorm(d_hidden)

        # ── Step 4: Dirichlet belief generator ───────────────────────────────
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
        s = self.norm3(s + self.ffn(s))

        # ── Step 4: Dirichlet belief update ───────────────────────────────────
        g_out = self.g(s)
        alpha = F.softplus(g_out) + 1.0      # (B, M, K), всегда > 1
        alpha_sum = alpha.sum(dim=-1, keepdim=True)
        b_next = alpha / alpha_sum            # ожидаемое распределение Дирихле
        u_next = self.K / alpha_sum           # неопределённость = K / sum(alpha)

        return AgentState(
            s=s,
            b=b_next,
            u=u_next,
            alpha=alpha,
            p=state.p,
            e=e,
        )
