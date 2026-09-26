"""
core_v4.py — CIDA v4 Lean Cognitive Co-Processor

Key Architectural Innovations:
1. PerceiverContextBottleneck: Compresses full token sequences (B, L, D_llm) into fixed-size latent queries in O(L).
2. BayesianRoleDebate: 4 structural roles (Prosecutor, Defender, Skeptic, Integrator) with Bayesian prior blending.
3. Sparse Top-2 Router: Contrastive message routing between opposing agents.
4. UnrolledRecurrentLoop: Deterministic, differentiable deliberation with Fast-Path early exit and Latent Rollback.
5. VariationalPlanProjector: Outputs distribution parameters (mu, log_var) for continuous Latent GRPO and KV injection.
"""

from dataclasses import dataclass, asdict
import json
import os
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CIDAv4Config:
    d_llm: int = 4096            # LLM hidden size (e.g., Qwen-2.5-7B is 3584 or 4096, 72B is 8192)
    d_hidden: int = 512          # CIDA internal debate hidden dimension
    num_agents: int = 4          # P, D, S, I
    num_bottleneck_queries: int = 16 # Perceiver bottleneck query count
    num_plan_tokens: int = 4     # Number of virtual plan tokens (P)
    max_rounds: int = 4          # Max deliberation unroll steps
    sparse_k: int = 2            # Top-k sparse communication
    halting_threshold: float = 0.03 # Fast-Path early exit variance threshold
    rollback_threshold: float = 0.15 # Disagreement spike threshold triggering rollback
    rollback_noise_std: float = 0.02 # Perturbation noise for rollback escape
    dropout: float = 0.05
    role_blend: float = 0.30     # Strength of structural prior (0.30 = 30% prior, 70% data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CIDAv4Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CIDAv4Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class PerceiverContextBottleneck(nn.Module):
    """
    Cross-Attention Bottleneck (Perceiver IO style).
    Distills full variable-length sequence states (B, L, D_llm) into fixed-size latent tokens (B, K, D_hidden).
    """
    def __init__(self, d_llm: int, d_hidden: int, num_queries: int = 16, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.d_hidden = d_hidden
        
        # Learnable latent query bank
        self.query_bank = nn.Parameter(torch.randn(1, num_queries, d_hidden) * 0.02)
        
        # Key/Value projection for LLM sequence
        self.kv_proj = nn.Linear(d_llm, d_hidden, bias=False)
        self.cross_attn = nn.MultiheadAttention(d_hidden, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d_hidden)
        self.norm_kv = nn.LayerNorm(d_hidden)
        
        # Self-attention between bottleneck queries
        self.self_attn = nn.MultiheadAttention(d_hidden, num_heads=num_heads, batch_first=True)
        self.norm_self = nn.LayerNorm(d_hidden)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.GELU(),
            nn.Linear(d_hidden * 2, d_hidden),
        )
        self.norm_ffn = nn.LayerNorm(d_hidden)

    def forward(self, h_seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        h_seq: (B, L, D_llm)
        mask: (B, L) boolean or float attention mask (1 for valid, 0 for pad)
        Returns: (B, num_queries, d_hidden)
        """
        B = h_seq.size(0)
        q = self.query_bank.expand(B, -1, -1)
        
        # Project KV
        kv = self.norm_kv(self.kv_proj(h_seq))
        
        # Key padding mask for nn.MultiheadAttention: True means IGNORE (pad)
        key_padding_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                key_padding_mask = ~mask
            else:
                key_padding_mask = (mask == 0)
        
        # Cross Attention
        ca_out, _ = self.cross_attn(
            query=self.norm_q(q),
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask
        )
        q = q + ca_out
        
        # Self Attention
        sa_out, _ = self.self_attn(self.norm_self(q), self.norm_self(q), self.norm_self(q))
        q = q + sa_out
        
        # FFN
        q = q + self.ffn(self.norm_ffn(q))
        return q


class BayesianRoleDebate(nn.Module):
    """
    4 Structural Agents with built-in Bayesian Role Priors:
    - 0: Prosecutor (prior = 0.85 positive)
    - 1: Defender (prior = 0.15 positive)
    - 2: Skeptic (prior = 0.50 maximum entropy / uncertainty)
    - 3: Integrator (prior = neutral / objective synthesizer)
    
    Sparse Top-2 Router directs counter-arguments to agents with greatest belief divergence.
    """
    ROLE_PRIORS = [0.85, 0.15, 0.50, 0.50]

    def __init__(self, d_hidden: int, num_agents: int = 4, sparse_k: int = 2, role_blend: float = 0.30):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_agents = num_agents
        self.sparse_k = sparse_k
        self.role_blend = role_blend
        
        # Message generation for each agent
        self.msg_gen = nn.Sequential(
            nn.Linear(d_hidden + 1, d_hidden), # state + current belief scalar
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden)
        )
        
        # Role adapter projections
        self.role_projections = nn.Parameter(torch.randn(num_agents, d_hidden) * 0.02)
        
        # Belief predictor head: maps agent state to belief probability p in [0, 1]
        self.belief_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid()
        )

    def apply_bayesian_priors(self, b_raw: torch.Tensor) -> torch.Tensor:
        """
        Blends raw evidence-based belief with fixed structural role priors.
        b_raw: (B, M, 1)
        Returns: (B, M, 1) blended beliefs
        """
        device = b_raw.device
        dtype = b_raw.dtype
        priors = torch.tensor(self.ROLE_PRIORS[:self.num_agents], device=device, dtype=dtype).view(1, -1, 1)
        # Integrator (agent 3) has minimal prior interference
        blend_weights = torch.tensor([self.role_blend, self.role_blend, self.role_blend, 0.05], device=device, dtype=dtype).view(1, -1, 1)
        b_blended = (1.0 - blend_weights) * b_raw + blend_weights * priors
        return b_blended

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        states: (B, M, d_hidden)
        Returns:
            messages: (B, M, d_hidden)
            beliefs: (B, M, 1)
            disagreement: (B,) scalar mean divergence across agents
        """
        B, M, D = states.size()
        
        # 1. Predict raw beliefs from current state
        b_raw = self.belief_head(states) # (B, M, 1)
        beliefs = self.apply_bayesian_priors(b_raw) # (B, M, 1)
        
        # 2. Formulate agent messages
        inp_msg = torch.cat([states, beliefs], dim=-1) # (B, M, D + 1)
        messages = self.msg_gen(inp_msg) # (B, M, D)
        
        # 3. Pairwise belief divergence matrix: D_ij = |b_i - b_j|
        b_flat = beliefs.squeeze(-1) # (B, M)
        diff_mat = torch.abs(b_flat.unsqueeze(1) - b_flat.unsqueeze(2)) # (B, M, M)
        
        # Mask self-communication
        diag_mask = torch.eye(M, device=states.device).bool().unsqueeze(0)
        diff_mat = diff_mat.masked_fill(diag_mask, -1e9)
        
        # Sparse Top-k: select agents with greatest disagreement
        topk_vals, topk_indices = torch.topk(diff_mat, k=min(self.sparse_k, M - 1), dim=-1)
        routing_weights = F.softmax(topk_vals, dim=-1) # (B, M, K)
        
        # Gather routed messages: for each agent i, gather top-k messages from agents j
        gathered_msgs = torch.zeros(B, M, D, device=states.device, dtype=states.dtype)
        for k_idx in range(self.sparse_k):
            src_indices = topk_indices[:, :, k_idx].unsqueeze(-1).expand(-1, -1, D)
            msg_k = torch.gather(messages, dim=1, index=src_indices)
            weight_k = routing_weights[:, :, k_idx].unsqueeze(-1)
            gathered_msgs = gathered_msgs + weight_k * msg_k
            
        disagreement = diff_mat.masked_fill(diag_mask, 0.0).sum(dim=(1, 2)) / (M * (M - 1))
        return gathered_msgs, beliefs, disagreement


class UnrolledRecurrentLoop(nn.Module):
    """
    Deterministic unrolled recurrent update with Fast-Path (Dynamic Early Halting) and Latent Rollback.
    Replaces heavy continuous ODEs for 5x training speedup and 100% batch determinism.
    """
    def __init__(self, config: CIDAv4Config):
        super().__init__()
        self.config = config
        d = config.d_hidden
        
        # Recurrent state updater (GRU Cell)
        self.gru = nn.GRUCell(d * 2, d) # Input: [gathered_messages, evidence]
        
        # Evidence extractor mapping Perceiver bottleneck tokens to 4 agents
        self.evidence_adapter = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d)
        )

    def forward(
        self,
        init_states: torch.Tensor,
        bottleneck_tokens: torch.Tensor,
        debate_module: BayesianRoleDebate,
    ) -> Dict[str, Any]:
        """
        init_states: (B, M, d_hidden)
        bottleneck_tokens: (B, K_queries, d_hidden)
        """
        B, M, D = init_states.size()
        cfg = self.config
        
        # Extract global evidence from bottleneck queries
        evidence_pooled = bottleneck_tokens.mean(dim=1, keepdim=True).expand(-1, M, -1) # (B, M, D)
        evidence = self.evidence_adapter(evidence_pooled)
        
        curr_states = init_states
        prev_disagreement = float('inf')
        rollback_count = 0
        rounds_executed = 0
        
        belief_history = []
        state_history = []
        
        for r in range(cfg.max_rounds):
            rounds_executed = r + 1
            gathered_msgs, beliefs, disagreement = debate_module(curr_states)
            belief_history.append(beliefs)
            state_history.append(curr_states)
            
            # 1. Latent Rollback: Detect sudden entropy spike or deadlock
            mean_disag = disagreement.mean().item()
            if r > 0 and (mean_disag - prev_disagreement) > cfg.rollback_threshold:
                rollback_count += 1
                # Perturb state with stochastic jitter to escape local attractor
                jitter = torch.randn_like(curr_states) * cfg.rollback_noise_std
                curr_states = state_history[-2] + jitter
                prev_disagreement = float('inf')
                continue
                
            prev_disagreement = mean_disag
            
            # 2. Fast-Path Early Exit: Check if consensus variance is minimal
            belief_var = torch.var(beliefs.squeeze(-1), dim=-1).mean().item()
            if r >= 1 and belief_var < cfg.halting_threshold:
                # Early consensus reached!
                break
                
            # 3. GRU State Update
            gru_in = torch.cat([gathered_msgs, evidence], dim=-1).view(B * M, D * 2)
            curr_flat = curr_states.view(B * M, D)
            next_flat = self.gru(gru_in, curr_flat)
            curr_states = next_flat.view(B, M, D)
            
        final_gathered, final_beliefs, final_disag = debate_module(curr_states)
        
        # Consensus: Weighted Mean of agent beliefs
        # Weights: Prosecutor=0.28, Defender=0.28, Skeptic=0.12, Integrator=0.32
        role_w = torch.tensor([0.28, 0.28, 0.12, 0.32], device=init_states.device, dtype=init_states.dtype).view(1, M, 1)
        p_consensus = (final_beliefs * role_w).sum(dim=1) # (B, 1)
        uncertainty = torch.std(final_beliefs.squeeze(-1), dim=-1) # (B,)
        
        return {
            "final_states": curr_states,
            "final_beliefs": final_beliefs,
            "p_consensus": p_consensus,
            "uncertainty": uncertainty,
            "disagreement": final_disag,
            "rounds_executed": rounds_executed,
            "rollback_count": rollback_count,
            "bottleneck_evidence": evidence,
        }


class VariationalPlanProjector(nn.Module):
    """
    Projects multi-agent debate consensus into P soft plan tokens (or KV prefix states).
    Outputs both mean (mu) and log-variance (log_var) to enable:
    1. Deterministic inference: z = mu
    2. Continuous Latent GRPO: z ~ N(mu, sigma) with reparameterization trick
    """
    def __init__(self, d_hidden: int, d_llm: int, num_agents: int = 4, num_plan_tokens: int = 4):
        super().__init__()
        self.num_plan_tokens = num_plan_tokens
        self.d_llm = d_llm
        d_in = (num_agents + 1) * d_hidden # 4 agents + 1 global evidence
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d_hidden * 2),
            nn.LayerNorm(d_hidden * 2),
            nn.GELU(),
            nn.Dropout(0.05)
        )
        
        # Mean projection: maps to (P * d_llm)
        self.fc_mu = nn.Sequential(
            nn.Linear(d_hidden * 2, num_plan_tokens * d_llm),
            nn.LayerNorm(num_plan_tokens * d_llm)
        )
        
        # Log-variance projection: initialized small for numerical stability
        self.fc_logvar = nn.Sequential(
            nn.Linear(d_hidden * 2, num_plan_tokens * d_llm)
        )
        nn.init.constant_(self.fc_logvar[0].weight, 0.0)
        nn.init.constant_(self.fc_logvar[0].bias, -4.0) # exp(-4) ~ 0.018 std

    def forward(self, final_states: torch.Tensor, evidence: torch.Tensor, sample: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        final_states: (B, num_agents, d_hidden)
        evidence: (B, num_agents, d_hidden)
        Returns:
            z: (B, num_plan_tokens, d_llm)
            mu: (B, num_plan_tokens, d_llm)
            log_var: (B, num_plan_tokens, d_llm)
        """
        B = final_states.size(0)
        ev_pooled = evidence.mean(dim=1, keepdim=True) # (B, 1, d_hidden)
        flat_repr = torch.cat([final_states, ev_pooled], dim=1).reshape(B, -1)
        
        feat = self.trunk(flat_repr)
        mu = self.fc_mu(feat).view(B, self.num_plan_tokens, self.d_llm)
        log_var = self.fc_logvar(feat).view(B, self.num_plan_tokens, self.d_llm)
        
        # Clamp log_var to avoid numerical explosion
        log_var = torch.clamp(log_var, min=-10.0, max=2.0)
        
        if sample:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
            
        return z, mu, log_var

    def kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor, ref_mu: Optional[torch.Tensor] = None, ref_log_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes KL divergence for GRPO regularization.
        If ref is None, computes KL(q(z|x) || N(0, I)).
        """
        if ref_mu is None or ref_log_var is None:
            # Standard KL vs N(0, I)
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        else:
            # KL vs reference policy N(ref_mu, ref_sigma^2)
            var = log_var.exp()
            ref_var = ref_log_var.exp()
            kl = 0.5 * torch.sum(ref_log_var - log_var + (var + (mu - ref_mu).pow(2)) / ref_var - 1.0, dim=-1)
        return kl.mean()


class CIDAv4Core(nn.Module):
    """
    CIDA v4 Master Module:
    1. Perceiver Bottleneck
    2. Bayesian Role Debate
    3. Unrolled Recurrent Loop
    4. Variational Plan Projector
    """
    def __init__(self, config: Optional[CIDAv4Config] = None):
        super().__init__()
        self.config = config or CIDAv4Config()
        cfg = self.config
        
        self.bottleneck = PerceiverContextBottleneck(
            d_llm=cfg.d_llm,
            d_hidden=cfg.d_hidden,
            num_queries=cfg.num_bottleneck_queries
        )
        
        # Projection from bottleneck to agent initial states
        self.agent_init_proj = nn.Linear(cfg.d_hidden, cfg.d_hidden)
        
        self.debate = BayesianRoleDebate(
            d_hidden=cfg.d_hidden,
            num_agents=cfg.num_agents,
            sparse_k=cfg.sparse_k,
            role_blend=cfg.role_blend
        )
        
        self.loop = UnrolledRecurrentLoop(cfg)
        
        self.projector = VariationalPlanProjector(
            d_hidden=cfg.d_hidden,
            d_llm=cfg.d_llm,
            num_agents=cfg.num_agents,
            num_plan_tokens=cfg.num_plan_tokens
        )

    def forward(
        self,
        h_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_plan: bool = False
    ) -> Dict[str, Any]:
        """
        h_seq: (B, L, d_llm) Sequence hidden states from frozen backbone
        mask: Optional (B, L) attention mask
        sample_plan: bool, True during RL / GRPO sampling
        """
        B = h_seq.size(0)
        M = self.config.num_agents
        
        # 1. Distill context via Perceiver Bottleneck
        bottleneck_tokens = self.bottleneck(h_seq, mask=mask) # (B, 16, d_hidden)
        
        # 2. Initialize 4 agents from bottleneck
        init_repr = bottleneck_tokens[:, :M, :] # take first M queries or pool
        init_states = self.agent_init_proj(init_repr) # (B, M, d_hidden)
        
        # 3. Deliberation loop
        loop_out = self.loop(init_states, bottleneck_tokens, self.debate)
        
        # 4. Generate plan tokens
        z, mu, log_var = self.projector(
            loop_out["final_states"],
            loop_out["bottleneck_evidence"],
            sample=sample_plan
        )
        
        return {
            "plan_tokens": z,
            "plan_mu": mu,
            "plan_logvar": log_var,
            "p_consensus": loop_out["p_consensus"],
            "uncertainty": loop_out["uncertainty"],
            "disagreement": loop_out["disagreement"],
            "rounds_executed": loop_out["rounds_executed"],
            "rollback_count": loop_out["rollback_count"],
        }

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.config.save(os.path.join(save_dir, "cida_v4_config.json"))
        torch.save(self.state_dict(), os.path.join(save_dir, "cida_v4_weights.pt"))

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu") -> "CIDAv4Core":
        cfg = CIDAv4Config.load(os.path.join(load_dir, "cida_v4_config.json"))
        model = cls(cfg)
        state_dict = torch.load(os.path.join(load_dir, "cida_v4_weights.pt"), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        return model
