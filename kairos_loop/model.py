"""
model.py — рекуррентное ядро Kairos: plain-цикл (как TRM) и CIDA-Loop.

Общая схема (TRM, arXiv:2510.04871):
    x — вход, y — текущий ответ, z — латентное «рассуждение».
    Цикл: n_latent раз z ← net(x + y + z), затем y ← net(y + consensus(z)).
    n_cycles циклов на шаг супервизии (градиент только через последний),
    до n_sup шагов глубокой супервизии с отсоединением состояния между шагами.

CIDA-Loop отличается только z:
    z имеет форму (B, K, L, D) — K агентов-полос над КАЖДЫМ токеном (а не один пулинговый вектор).
    После каждого шага агенты обмениваются сообщениями: внимание K×K по токену,
    смещённое к разногласию (L1 между предсказаниями агентов), без внимания на себя.
    Ответ обновляется консенсусом, взвешенным уверенностью агентов.
    Согласие агентов возвращается как встроенная мера уверенности.

Сеть net общая для всех агентов: CIDA добавляет только ролевые векторы и модуль связи.
Внутри forward нет .item() и синхронизаций — модуль совместим с torch.compile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoopConfig:
    variant: str = "cida"          # "cida" | "plain"
    seq_len: int = 81
    vocab: int = 10                # 0 = пусто, 1..9
    n_classes: int = 9
    d_model: int = 256
    n_layers: int = 2
    mixer: str = "mlp"             # "mlp" (смешивание по токенам, как TRM-MLP) | "attn"
    n_heads: int = 8
    expansion: float = 4.0
    n_latent: int = 6              # обновлений z на цикл
    n_cycles: int = 3              # циклов на шаг супервизии
    n_sup: int = 16                # шагов глубокой супервизии
    n_agents: int = 4
    d_comm: int = 64
    communicate: bool = True       # False → независимые агенты (абляция «без связи»)
    disagreement_bias: bool = True # False → обычное внимание без смещения к разногласию
    role_ortho: float = 0.1        # вес регуляризатора ортогональности ролей

    def __post_init__(self):
        if self.variant not in ("cida", "plain"):
            raise ValueError(f"unknown variant: {self.variant}")
        if self.mixer not in ("mlp", "attn"):
            raise ValueError(f"unknown mixer: {self.mixer}")
        if self.variant == "cida" and self.communicate and self.n_agents < 2:
            raise ValueError("communication needs n_agents >= 2")

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d: int, expansion: float):
        super().__init__()
        hidden = max(32, int(round(expansion * d * 2 / 3 / 32)) * 32)
        self.w_in = nn.Linear(d, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


class Block(nn.Module):
    """Post-norm блок: смешивание токенов (MLP по последовательности или self-attention) + SwiGLU."""

    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.mixer = cfg.mixer
        d = cfg.d_model
        if cfg.mixer == "mlp":
            self.token_mix = SwiGLU(cfg.seq_len, cfg.expansion)
        else:
            self.n_heads = cfg.n_heads
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.proj = nn.Linear(d, d, bias=False)
        self.ffn = SwiGLU(d, cfg.expansion)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mixer == "mlp":
            mixed = self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        else:
            b, l, d = x.shape
            q, k, v = self.qkv(x).view(b, l, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
            mixed = self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, l, d))
        x = self.norm1(x + mixed)
        return self.norm2(x + self.ffn(x))


class AgentCommunication(nn.Module):
    """Обмен сообщениями между K агентами по каждому токену (внимание K×K, смещённое к разногласию)."""

    def __init__(self, cfg: LoopConfig):
        super().__init__()
        d, dc = cfg.d_model, cfg.d_comm
        self.scale = dc ** -0.5
        self.norm = RMSNorm(d)
        self.q = nn.Linear(d, dc, bias=False)
        self.k = nn.Linear(d, dc, bias=False)
        self.v = nn.Linear(d, dc, bias=False)
        self.o = nn.Linear(dc, d, bias=False)
        nn.init.zeros_(self.o.weight)  # на старте связь — тождественное отображение
        self.alpha = nn.Parameter(torch.tensor(1.0)) if cfg.disagreement_bias else None
        self.register_buffer("self_mask", torch.eye(cfg.n_agents, dtype=torch.bool), persistent=False)

    def forward(self, z: torch.Tensor, beliefs: Optional[torch.Tensor]) -> torch.Tensor:
        # z: (B, K, L, D); beliefs: (B, K, L, C) — вероятности агентов
        h = self.norm(z)
        q, k, v = self.q(h), self.k(h), self.v(h)
        scores = torch.einsum("bild,bjld->blij", q, k) * self.scale
        if self.alpha is not None and beliefs is not None:
            dis = 0.5 * (beliefs.unsqueeze(2) - beliefs.unsqueeze(1)).abs().sum(-1)  # (B, K, K, L)
            scores = scores + self.alpha * dis.permute(0, 3, 1, 2).to(scores.dtype)
        scores = scores.masked_fill(self.self_mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        msg = torch.einsum("blij,bjld->bild", attn, v)
        return z + self.o(msg)


def pairwise_disagreement(beliefs: torch.Tensor) -> torch.Tensor:
    """Средняя total-variation дистанция между агентами по токенам: (B, K, L, C) → (B,)."""
    k = beliefs.shape[1]
    dis = 0.5 * (beliefs.unsqueeze(2) - beliefs.unsqueeze(1)).abs().sum(-1)  # (B, K, K, L)
    return dis.sum(dim=(1, 2)).mean(-1) / max(k * (k - 1), 1)


class KairosLoop(nn.Module):
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab, d)
        self.pos = nn.Parameter(torch.randn(cfg.seq_len, d) * 0.02)
        self.net = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(d, cfg.n_classes)
        self.halt_head = nn.Linear(d, 1)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -5.0)
        # Начальные состояния фиксированы (как в TRM): градиент до них не доходит через no_grad-циклы.
        self.register_buffer("y_init", torch.randn(d) * 0.02)
        self.register_buffer("z_init", torch.randn(d) * 0.02)
        self.is_cida = cfg.variant == "cida"
        if self.is_cida:
            self.roles = nn.Parameter(torch.randn(cfg.n_agents, d) * 0.02)
            self.comm = AgentCommunication(cfg) if cfg.communicate else None

    # ── состояние ────────────────────────────────────────────────────────────
    def initial_state(self, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        L, d = self.cfg.seq_len, self.cfg.d_model
        y = self.y_init.expand(batch, L, d)
        if self.is_cida:
            z = self.z_init.expand(batch, self.cfg.n_agents, L, d)
        else:
            z = self.z_init.expand(batch, L, d)
        return y, z

    # ── один цикл рассуждения ────────────────────────────────────────────────
    def _beliefs(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z).float().softmax(-1)

    def _latent(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.is_cida:
            return self.net(x + y + z)
        b, k, L, d = z.shape
        # роль агента подаётся на каждом шаге — это его постоянная «точка зрения»
        inp = (x + y).unsqueeze(1) + z + self.roles[None, :, None, :]
        z = self.net(inp.reshape(b * k, L, d)).reshape(b, k, L, d)
        if self.comm is not None:
            beliefs = self._beliefs(z) if self.comm.alpha is not None else None
            z = self.comm(z, beliefs)
        return z

    def _consensus(self, z: torch.Tensor) -> torch.Tensor:
        if not self.is_cida:
            return z
        p = self._beliefs(z)
        entropy = -(p * p.clamp_min(1e-9).log()).sum(-1)       # (B, K, L)
        w = (-entropy).softmax(dim=1).unsqueeze(-1).to(z.dtype)  # уверенные агенты весят больше
        return (w * z).sum(1)

    def _cycle(self, x, y, z):
        for _ in range(self.cfg.n_latent):
            z = self._latent(x, y, z)
        y = self.net(y + self._consensus(z))
        return y, z

    # ── шаг глубокой супервизии ──────────────────────────────────────────────
    def forward(self, tokens: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        """
        tokens: (B, L) int; state: (y, z) из initial_state или прошлого шага.
        Returns: logits (B, L, C), halt_logit (B,), agreement (B,) или None, новое состояние (detached).
        """
        x = self.embed(tokens) + self.pos
        y, z = state
        with torch.no_grad():
            for _ in range(self.cfg.n_cycles - 1):
                y, z = self._cycle(x, y, z)
        y, z = self._cycle(x, y, z)
        logits = self.head(y)
        halt_logit = self.halt_head(y.mean(1)).squeeze(-1)
        agreement = 1.0 - pairwise_disagreement(self._beliefs(z)) if self.is_cida else None
        return logits, halt_logit, agreement, (y.detach(), z.detach())

    def aux_loss(self) -> torch.Tensor:
        """Регуляризатор ортогональности ролевых векторов: ||R̂R̂ᵀ − I||²_F."""
        if not self.is_cida or self.cfg.role_ortho <= 0:
            return self.head.weight.new_zeros(())
        r = F.normalize(self.roles, dim=-1)
        eye = torch.eye(r.shape[0], device=r.device, dtype=r.dtype)
        return self.cfg.role_ortho * (r @ r.T - eye).pow(2).sum()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
