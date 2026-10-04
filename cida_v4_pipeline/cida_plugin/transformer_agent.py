"""
transformer_agent.py — SOTA Modern Transformer Deliberation Core for CIDA

Replaces shallow linear/MLP agents with modern, high-throughput transformer blocks:
1. RMSNorm (Root Mean Square Normalization without mean centering)
2. SwiGLU FFN (Gated linear unit with SiLU activation)
3. Deliberation RoPE (Rotary Position Embeddings indexed by round t in [0, T])
4. Flash-SDPA (Scaled Dot-Product Attention with additive disagreement mask)
5. UniversalPlanProjector (Projects M=4 agents into P=8 virtual tokens in d_llm space)
"""

import math
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019; Llama-3 style).
    Computationally faster than LayerNorm by ~15-20% by omitting mean subtraction.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLUFFN(nn.Module):
    """
    Swish-Gated Linear Unit Feed-Forward Network (Shazeer, 2020; Llama-3 style).
    SwiGLU(x) = (x * W_gate * silu(x * W_up)) * W_down
    Provides ~25-30% higher parameter efficiency than standard GELU MLPs.
    """
    def __init__(self, d_model: int, d_ffn: Optional[int] = None, dropout: float = 0.05):
        super().__init__()
        if d_ffn is None:
            # Standard ratio: 8/3 of d_model rounded to multiple of 64
            d_ffn = int(2 * d_model * 4 / 3)
            d_ffn = (d_ffn + 63) // 64 * 64
            
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class DeliberationRoPE(nn.Module):
    """
    Rotary Position Embeddings adapted for Temporal Deliberation Rounds (t = 0 ... T).
    Encodes the progression of debate phase (exploration vs convergence) into query/key phase rotations.
    """
    def __init__(self, d_head: int, max_rounds: int = 16, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        self.max_rounds = max_rounds
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, round_idx: int) -> torch.Tensor:
        # x: (B, num_heads, M, d_head)
        t = torch.tensor([round_idx], device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (1, d_head // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (1, d_head)
        cos = emb.cos()
        sin = emb.sin()
        return (x * cos) + (self.rotate_half(x) * sin)


class SOTATransformerAgentBlock(nn.Module):
    """
    High-speed SOTA Transformer Agent block:
    Pre-RMSNorm -> Multihead Attention with SDPA & Disagreement Mask -> Pre-RMSNorm -> SwiGLU FFN.
    """
    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 8,
        d_ffn: Optional[int] = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = DeliberationRoPE(self.d_head)
        
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ffn=d_ffn, dropout=dropout)

    def forward(
        self,
        s: torch.Tensor,
        round_idx: int,
        disagreement_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        s: (B, M, d_model) - agent states
        round_idx: current deliberation round index
        disagreement_mask: (B, M, M) - custom attention bias from contrastive divergence
        """
        B, M, _ = s.shape
        h = self.norm1(s)
        
        # Q, K, V projections
        Q = self.q_proj(h).view(B, M, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, M, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, M, self.num_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE for round phase
        Q = self.rope(Q, round_idx)
        K = self.rope(K, round_idx)
        
        # Reshape custom disagreement mask to match multi-head attention: (B, 1, M, M)
        attn_mask = None
        if disagreement_mask is not None:
            attn_mask = disagreement_mask.unsqueeze(1)
            
        # Hardware-accelerated Flash SDPA
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, self.d_model)
        h = s + self.out_proj(attn_out)
        
        # Second residual block with SwiGLU FFN
        out = h + self.ffn(self.norm2(h))
        return out


class SOTADeliberationCore(nn.Module):
    """
    Complete CIDA-SOTA Deliberation Core (M=4 agents, recurrent weight-tied transformer block).
    Features:
    - M=4 role-specialized perspective projections (Prosecutor, Defender, Skeptic, Integrator)
    - Orthogonality regularization
    - Disagreement-biased contrastive attention matrix: M_attn = lambda_D * D_ij - lambda_S * S_ij
    - Subjective Logic / Dirichlet belief updates
    """
    def __init__(
        self,
        d_input: int = 4096,
        d_hidden: int = 1024,
        num_agents: int = 4,
        num_rounds: int = 4,
        lambda_d: float = 1.0,
        lambda_s: float = 0.5,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.lambda_d = lambda_d
        self.lambda_s = lambda_s
        
        # 1. Perspective Projections: separate linear layer + role embedding per agent
        self.agent_projections = nn.ModuleList([
            nn.Linear(d_input, d_hidden, bias=False) for _ in range(num_agents)
        ])
        self.role_embeddings = nn.Parameter(torch.randn(num_agents, d_hidden) * 0.02)
        
        # 2. Universal SOTA Deliberation Transformer Block (recurrent across rounds)
        self.transformer_block = SOTATransformerAgentBlock(
            d_model=d_hidden,
            num_heads=8,
            dropout=0.05
        )
        
        # 3. Evidential Belief Predictors (Maps agent state to Dirichlet evidence parameters)
        self.belief_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden // 2),
                nn.SiLU(),
                nn.Linear(d_hidden // 2, 2)
            ) for _ in range(num_agents)
        ])
        
        # Historical reliability tracking (frozen at test-time)
        self.register_buffer("reliability_scores", torch.ones(num_agents))

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """Penalizes correlation between the 4 agent role embeddings."""
        r_norm = F.normalize(self.role_embeddings, p=2, dim=-1)
        gram = torch.matmul(r_norm, r_norm.T)
        identity = torch.eye(self.num_agents, device=gram.device)
        return torch.norm(gram - identity, p="fro")

    def forward(
        self,
        h_last: torch.Tensor,
        min_rounds: int = 2,
        max_rounds: Optional[int] = None,
        target_disagreement: float = 0.05
    ) -> Dict[str, Any]:
        """
        h_last: (B, d_input) - context vector from frozen LLM prompt pass.
        Returns:
            s_final: (B, M, d_hidden)
            disagreement_history: list of float values
            p_consensus: (B, 2)
        """
        B = h_last.size(0)
        max_r = max_rounds or self.num_rounds
        
        # Step 1: Initial perspective projections
        states = []
        for i in range(self.num_agents):
            init_s = self.agent_projections[i](h_last) + self.role_embeddings[i].unsqueeze(0)
            states.append(init_s)
        s = torch.stack(states, dim=1)  # (B, M, d_hidden)
        
        disagreement_trajectory = []
        final_beliefs = None
        
        # Step 2: Multi-Round Latent Deliberation Loop
        for r in range(max_r):
            # Compute current beliefs b_i in (0, 1) and evidence vectors e_i
            logits_list = [self.belief_heads[i](s[:, i, :]) for i in range(self.num_agents)]
            logits = torch.stack(logits_list, dim=1)  # (B, M, 2)
            beliefs = F.softmax(logits, dim=-1)       # (B, M, 2)
            final_beliefs = beliefs
            
            # Pairwise disagreement matrix D_ij = ||b_i - b_j||_1
            b_i = beliefs.unsqueeze(2)  # (B, M, 1, 2)
            b_j = beliefs.unsqueeze(1)  # (B, 1, M, 2)
            D = torch.abs(b_i - b_j).sum(dim=-1)  # (B, M, M)
            
            # Pairwise redundancy matrix S_ij = cos(s_i, s_j)
            s_norm = F.normalize(s, p=2, dim=-1)
            S = torch.bmm(s_norm, s_norm.transpose(1, 2))  # (B, M, M)
            
            # Construct contrastive attention mask
            disagreement_mask = self.lambda_d * D - self.lambda_s * S
            # Mask self-attention (agent debates others, not self)
            eye_mask = torch.eye(self.num_agents, device=s.device, dtype=torch.bool).unsqueeze(0)
            disagreement_mask = disagreement_mask.masked_fill(eye_mask, -1e4)
            
            mean_disagreement = D.mean().item()
            disagreement_trajectory.append(mean_disagreement)
            
            # Early exit if consensus achieved after minimum rounds
            if r >= min_rounds and mean_disagreement < target_disagreement:
                break
                
            # Update agent states via SOTA Transformer Block
            s = self.transformer_block(s, round_idx=r, disagreement_mask=disagreement_mask)

        # Step 3: Product-of-Experts Consensus
        # Epistemic uncertainty u_i = 4 * b_i * (1 - b_i)
        p_pos = final_beliefs[:, :, 1]  # (B, M)
        u = 4.0 * p_pos * (1.0 - p_pos)  # (B, M)
        
        weights = F.softmax(-1.0 * u + 0.5 * self.reliability_scores.unsqueeze(0), dim=-1)  # (B, M)
        log_b = torch.log(torch.clamp(final_beliefs, min=1e-6))
        poe_logits = (weights.unsqueeze(-1) * log_b).sum(dim=1)
        p_consensus = F.softmax(poe_logits, dim=-1)

        return {
            "s_final": s,  # (B, M, d_hidden)
            "p_consensus": p_consensus,  # (B, 2)
            "disagreement_trajectory": disagreement_trajectory,
            "final_disagreement": disagreement_trajectory[-1] if disagreement_trajectory else 0.0,
            "orth_loss": self.compute_orthogonality_loss(),
        }


class UniversalPlanProjector(nn.Module):
    """
    Projects converged multi-agent state (B, M, d_hidden) into P=8 virtual plan tokens
    in the LLM decoder embedding space (B, 8, d_llm) with convex gating.
    """
    def __init__(
        self,
        d_hidden: int = 1024,
        d_llm: int = 4096,
        num_agents: int = 4,
        num_plan_tokens: int = 8,
    ):
        super().__init__()
        self.num_plan_tokens = num_plan_tokens
        self.d_llm = d_llm
        d_flat = num_agents * d_hidden
        
        self.projector = nn.Sequential(
            nn.Linear(d_flat, d_flat * 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(d_flat * 2, num_plan_tokens * d_llm),
            RMSNorm(num_plan_tokens * d_llm)
        )
        
        # Dynamic Gating projection: decides balance between base LLM embedding and plan tokens
        self.gate_proj = nn.Linear(d_llm * 2, 1)

    def forward(self, s_final: torch.Tensor, h_base: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        """
        s_final: (B, M, d_hidden)
        h_base: (B, d_llm)
        Returns:
            mixed_plan: (B, num_plan_tokens, d_llm)
        """
        B = s_final.size(0)
        s_flat = s_final.reshape(B, -1)
        raw_plan = self.projector(s_flat).view(B, self.num_plan_tokens, self.d_llm)
        
        # Compute dynamic gate from h_base and pooled plan
        plan_pooled = raw_plan.mean(dim=1)
        gate_in = torch.cat([h_base, plan_pooled], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_in)).unsqueeze(1)  # (B, 1, 1)
        
        h_base_expanded = h_base.unsqueeze(1).expand(-1, self.num_plan_tokens, -1)
        mixed_plan = gate * (raw_plan * alpha) + (1.0 - gate) * h_base_expanded
        return mixed_plan
