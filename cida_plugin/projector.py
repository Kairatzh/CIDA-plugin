import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentAEProjector(nn.Module):
    """
    SOTA Projector: compresses agent states into plan-tokens using a
    Cross-Attention Bottleneck (inspired by Perceiver IO, Jaegle et al. 2021).
    """
    def __init__(self, d_cida: int, d_llm: int, num_agents: int, num_plan: int = 8, num_heads: int = 8):
        super().__init__()
        self.num_plan = num_plan
        self.d_cida = d_cida
        self.d_llm = d_llm
        
        # Learnable query slots
        self.plan_queries = nn.Parameter(torch.randn(1, num_plan, d_cida) * 0.02)
        
        # Cross-Attention: queries <- agent states
        self.cross_attn = nn.MultiheadAttention(d_cida, num_heads, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(d_cida)
        
        # Self-Attention: plan tokens <-> plan tokens
        self.self_attn = nn.MultiheadAttention(d_cida, num_heads, batch_first=True, dropout=0.1)
        self.self_norm = nn.LayerNorm(d_cida)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_cida, d_cida * 2),
            nn.SiLU(),
            nn.Linear(d_cida * 2, d_cida)
        )
        self.ffn_norm = nn.LayerNorm(d_cida)
        
        # Projection to LLM space
        self.out_proj = nn.Sequential(
            nn.Linear(d_cida, d_llm * 2),
            nn.SiLU(),
            nn.Linear(d_llm * 2, d_llm),
            nn.LayerNorm(d_llm)
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, agent_states: torch.Tensor) -> torch.Tensor:
        # agent_states: (B, num_agents, d_cida)
        B = agent_states.size(0)
        q = self.plan_queries.expand(B, -1, -1)
        
        # Cross-attn: each query slot attends to agent states
        ca, _ = self.cross_attn(q, agent_states, agent_states)
        q = self.cross_norm(q + ca)
        
        # Self-attn: plan coherence
        sa, _ = self.self_attn(q, q, q)
        q = self.self_norm(q + sa)
        
        # FFN
        q = self.ffn_norm(q + self.ffn(q))
        
        # Project and scale via gated mechanism
        return self.out_proj(q) * self.gate.sigmoid()


class TargetEncoder(nn.Module):
    """
    Compresses target hidden states (e.g. from Chain of Thought) into reference latent vectors.
    """
    def __init__(self, d_llm: int, num_plan: int = 8, num_heads: int = 8):
        super().__init__()
        self.num_plan = num_plan
        self.queries = nn.Parameter(torch.randn(1, num_plan, d_llm) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_llm, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_llm)

    def forward(self, cot_hidden: torch.Tensor) -> torch.Tensor:
        # cot_hidden: (B, seq_len, d_llm)
        B = cot_hidden.size(0)
        q = self.queries.expand(B, -1, -1)
        out, _ = self.cross_attn(q, cot_hidden, cot_hidden)
        return self.norm(q + out)
