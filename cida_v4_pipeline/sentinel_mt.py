"""
cida_plugin/sentinel_mt.py — Multi-Turn Agentic Risk Deliberation Sentinel (Kairos-Sentinel-MT)

Specialized for multi-turn tool-using agent safety benchmarks (e.g., MT-AgentRisk, ICML 2026).
Defends against multi-turn attack vectors (Decomposition, Addition, Context Obfuscation, Privilege Escalation).

Architecture:
1. Trajectory Graph Aggregator (TGA):
   - Cross-Attention over multi-turn interaction history (Q = proposed action, K, V = past turns)
   - Bidirectional Recurrent GRU State Tracker
   - Turn-level attention weights for causal exploit attribution
2. Quad-Agent Dialectical Deliberation:
   - Agent 0 (Prosecutor): Specialized in detecting latent multi-turn exploit chains
   - Agent 1 (Defender): Specialized in preserving benign tool utility and minimizing false blocks
   - Agent 2 (Skeptic): Cross-examines tool arguments against sandbox security boundaries
   - Agent 3 (Integrator): Subjective Logic consensus (belief b, epistemic vacuity u_epi, disagreement d)
3. O-LoRA Multi-Task Action Decoders:
   - head_risk: Binary Risk (0: Safe, 1: Harmful / Attack)
   - head_taxonomy: 8 MT-AgentRisk Attack Subcategories
   - head_mcp_tool: 5 MCP Tool Target Classes (Filesystem, PostgreSQL, Terminal, Playwright, Notion)
   - head_intervention: 3 Policy Actions (0: ALLOW, 1: BLOCK, 2: REQUIRE_CONFIRMATION)
4. Adaptive Temperature Scaling (ATS) & Split-Conformal Decision Engine.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin
from cida_plugin.consensus import ConsensusAggregator
from cida_plugin.multitask_pro import OLoRATaskAdapter, AdaptiveTemperatureScaling


# ── 1. Taxonomy & Tool Constants for MT-AgentRisk ─────────────────────────────

MT_AGENT_RISK_TAXONOMY = [
    "Addition-MethodA-DataFiles",
    "Addition-MethodA-EnvironmentStates",
    "Addition-MethodB-DataFiles",
    "Addition-MethodB-EnvironmentStates",
    "Decomposition-MethodA-DataFiles",
    "Decomposition-MethodA-EnvironmentStates",
    "Decomposition-MethodB-DataFiles",
    "Decomposition-MethodB-EnvironmentStates",
]

MT_MCP_TOOLS = [
    "Filesystem-MCP",
    "PostgreSQL-MCP",
    "Terminal",
    "Playwright-MCP",
    "Notion-MCP",
]

INTERVENTION_POLICIES = [
    "ALLOW",
    "BLOCK",
    "REQUIRE_CONFIRMATION",
]


# ── 2. Trajectory Graph Aggregator (TGA) ──────────────────────────────────────

class TrajectoryAggregator(nn.Module):
    """
    Aggregates multi-turn conversation and tool-calling trajectories.
    Uses cross-attention between the current proposed action and all previous turns,
    fused with bidirectional recurrent state tracking.
    """
    def __init__(self, d_input: int = 1024, d_hidden: int = 512, num_heads: int = 8, max_turns: int = 16):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.num_heads = num_heads

        # Positional encoding for turn order
        self.turn_pos_emb = nn.Embedding(max_turns, d_input)

        # Bidirectional Recurrent History Tracker
        self.history_gru = nn.GRU(
            input_size=d_input,
            hidden_size=d_hidden // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_proj = nn.Linear(d_hidden, d_input)

        # Multi-Head Cross-Attention: Query = Current Action, Key/Value = Trajectory History
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_input,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

        self.norm1 = nn.LayerNorm(d_input)
        self.norm2 = nn.LayerNorm(d_input)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_input, d_input * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_input * 2, d_input),
        )

    def forward(
        self,
        turn_embeddings: torch.Tensor,
        turn_mask: Optional[torch.Tensor] = None,
        proposed_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        turn_embeddings: (B, T, d_input) — latent vectors of conversation turns 1..T
        turn_mask: (B, T) — boolean tensor, True for valid turns, False for padding
        proposed_action: (B, d_input) — latent vector of current proposed tool invocation
                         (if None, the last valid turn is used as proposed_action)

        Returns:
            fused_trajectory: (B, d_input) — contextualized multi-turn state vector
            attn_weights: (B, 1, T) — attribution weights across previous turns
        """
        B, T, D = turn_embeddings.shape
        device = turn_embeddings.device

        # 1. Add turn positional encoding
        turn_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        turn_indices = turn_indices.clamp(max=self.turn_pos_emb.num_embeddings - 1)
        h_pos = turn_embeddings + self.turn_pos_emb(turn_indices)

        # 2. Track sequential history via Bidirectional GRU
        gru_out, _ = self.history_gru(h_pos)  # (B, T, d_hidden)
        h_seq = self.gru_proj(gru_out)        # (B, T, d_input)

        # 3. Identify query (current proposed action)
        if proposed_action is None:
            if turn_mask is not None:
                # Select last valid turn per batch element
                last_indices = turn_mask.sum(dim=1).long() - 1
                last_indices = last_indices.clamp(min=0, max=T - 1)
                batch_idx = torch.arange(B, device=device)
                q_act = h_seq[batch_idx, last_indices].unsqueeze(1)  # (B, 1, D)
            else:
                q_act = h_seq[:, -1:, :]  # (B, 1, D)
        else:
            q_act = proposed_action.unsqueeze(1)  # (B, 1, D)

        # 4. Multi-Head Cross Attention
        # Key padding mask expects True for positions that SHOULD BE IGNORED
        key_padding_mask = (~turn_mask) if turn_mask is not None else None

        attn_out, attn_weights = self.cross_attn(
            query=q_act,
            key=h_seq,
            value=h_seq,
            key_padding_mask=key_padding_mask,
        )  # attn_out: (B, 1, D), attn_weights: (B, 1, T)

        # Strictly renormalize attention weights over valid turns
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-7)

        # 5. Residual + Feed-Forward
        fused = self.norm1(q_act + attn_out)
        fused = self.norm2(fused + self.feed_forward(fused))
        fused_vec = fused.squeeze(1)  # (B, d_input)

        return fused_vec, attn_weights


# ── 3. Kairos-Sentinel-MT Model ───────────────────────────────────────────────

class KairosSentinelMT(nn.Module):
    """
    Kairos-Sentinel-MT: Multi-Turn Deliberative Safety Coprocessor.
    Intercepts and arbitrates tool-using agent actions across multi-turn trajectories.
    """
    def __init__(
        self,
        d_input: int = 1024,
        d_hidden: int = 512,
        d_message: int = 512,
        d_proj: int = 256,
        num_agents: int = 4,
        max_delib_rounds: int = 3,
        num_taxonomy_classes: int = 8,
        num_mcp_tools: int = 5,
        num_policy_actions: int = 3,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_message = d_message
        self.d_proj = d_proj
        self.num_agents = num_agents

        self.num_taxonomy_classes = num_taxonomy_classes
        self.num_mcp_tools = num_mcp_tools
        self.num_policy_actions = num_policy_actions

        # ── Step 1. Trajectory Graph Aggregator ───────────────────────────────
        self.trajectory_aggregator = TrajectoryAggregator(
            d_input=d_input,
            d_hidden=d_hidden,
            num_heads=8,
            max_turns=20,
        )

        # ── Step 2. Dialectical 4-Agent Deliberation Core ─────────────────────
        self.cida_config = CIDAPluginConfig(
            d_input=d_input,
            d_hidden=d_hidden,
            d_message=d_message,
            num_agents=num_agents,
            num_classes=d_hidden,
            max_rounds=max_delib_rounds,
            role_prior_blend=0.15,  # active dialectical role specialization
            use_vae_perspective=False,
            multi_label=False,
        )
        self.deliberation_core = CIDAPlugin(self.cida_config)

        # ── Step 3. O-LoRA Task Orthogonal Projections ────────────────────────
        self.adapter_risk = OLoRATaskAdapter(d_hidden, d_proj)
        self.adapter_taxonomy = OLoRATaskAdapter(d_hidden, d_proj)
        self.adapter_mcp = OLoRATaskAdapter(d_hidden, d_proj)
        self.adapter_policy = OLoRATaskAdapter(d_hidden, d_proj)

        # ── Step 4. Decoders & Consensus Modules ──────────────────────────────
        # Head 1: Binary Risk (0: Safe, 1: Harmful)
        self.head_risk = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_proj // 2, 2),
        )
        self.consensus_risk = ConsensusAggregator(num_agents=num_agents)
        self.ats_risk = AdaptiveTemperatureScaling(d_input=d_proj, base_temp=1.0)

        # Head 2: 8-Class MT-AgentRisk Taxonomy
        self.head_taxonomy = nn.Sequential(
            nn.Linear(d_proj, d_proj),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_proj, num_taxonomy_classes),
        )
        self.consensus_taxonomy = ConsensusAggregator(num_agents=num_agents)
        self.ats_taxonomy = AdaptiveTemperatureScaling(d_input=d_proj, base_temp=1.0)

        # Head 3: 5 MCP Target Tool Categories
        self.head_mcp = nn.Sequential(
            nn.Linear(d_proj, d_proj),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_proj, num_mcp_tools),
        )
        self.consensus_mcp = ConsensusAggregator(num_agents=num_agents)
        self.ats_mcp = AdaptiveTemperatureScaling(d_input=d_proj, base_temp=1.0)

        # Head 4: Intervention Policy (ALLOW, BLOCK, REQUIRE_CONFIRMATION)
        self.head_policy = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_proj // 2, num_policy_actions),
        )
        self.consensus_policy = ConsensusAggregator(num_agents=num_agents)
        self.ats_policy = AdaptiveTemperatureScaling(d_input=d_proj, base_temp=1.0)

    def compute_task_orthogonality_loss(self) -> torch.Tensor:
        """
        Computes Frobenius orthogonality across the 4 O-LoRA adapters:
        Risk _|_ Taxonomy _|_ MCP _|_ Policy.
        """
        weights = [
            self.adapter_risk.weight,
            self.adapter_taxonomy.weight,
            self.adapter_mcp.weight,
            self.adapter_policy.weight,
        ]
        loss_orth = torch.tensor(0.0, device=weights[0].device)
        eps = 1e-7
        count = 0
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                w_i = weights[i]
                w_j = weights[j]
                gram = torch.matmul(w_i, w_j.t())
                pair_loss = (gram ** 2).sum() / (torch.norm(w_i) * torch.norm(w_j) + eps)
                loss_orth = loss_orth + pair_loss
                count += 1
        return loss_orth / max(count, 1)

    def _decode_branch(
        self,
        s_final: torch.Tensor,
        adapter: OLoRATaskAdapter,
        head_module: nn.Module,
        consensus_module: ConsensusAggregator,
        ats_module: AdaptiveTemperatureScaling,
        num_classes: int,
    ) -> Dict[str, torch.Tensor]:
        """Maps deliberated states through O-LoRA adapter, Dirichlet consensus, and ATS."""
        B, M, _ = s_final.shape
        h_task = adapter(s_final)       # (B, M, d_proj)
        logits = head_module(h_task)    # (B, M, K)

        alpha_prior = 1.0 / float(num_classes)
        alpha = F.softplus(logits) + alpha_prior
        b = alpha / alpha.sum(dim=-1, keepdim=True)  # (B, M, K)

        # Subjective Logic Epistemic Vacuity (Josang 2002)
        s_0 = num_classes * alpha_prior  # 1.0
        u_epi = (s_0 / alpha.sum(dim=-1, keepdim=True)) ** 2
        u_alea = 1.0 - b.max(dim=-1, keepdim=True).values
        u = u_epi + u_alea

        p_consensus, uncertainty, disagreement = consensus_module(b=b, u=u)

        # Adaptive Temperature Scaling
        h_mean = h_task.mean(dim=1)
        u_mean = uncertainty.unsqueeze(-1) if uncertainty.dim() == 1 else uncertainty
        T_sample = ats_module(h_mean, u_mean)

        mean_logits = logits.mean(dim=1)
        calibrated_logits = mean_logits / T_sample
        p_calibrated = F.softmax(calibrated_logits, dim=-1)

        return {
            "p_final": p_consensus,
            "p_calibrated": p_calibrated,
            "calibrated_logits": calibrated_logits,
            "uncertainty": uncertainty,
            "disagreement": disagreement,
            "temperature": T_sample.squeeze(-1),
            "b_all": [b],
        }

    def forward(
        self,
        turn_embeddings: torch.Tensor,
        turn_mask: Optional[torch.Tensor] = None,
        proposed_action: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for multi-turn safety evaluation and intervention.

        turn_embeddings: (B, T, 1024)
        turn_mask: (B, T) boolean mask
        proposed_action: (B, 1024) optional current tool call vector
        task: optional ('risk', 'taxonomy', 'mcp', 'policy') or None for all.
        """
        # 1. Aggregate multi-turn trajectory with cross-attention
        fused_state, attn_weights = self.trajectory_aggregator(
            turn_embeddings=turn_embeddings,
            turn_mask=turn_mask,
            proposed_action=proposed_action,
        )

        # 2. Quad-Agent Deliberation
        core_out = self.deliberation_core(pooled=fused_state)
        state_final = core_out.get("state_final")
        if state_final is not None and hasattr(state_final, "s"):
            s_final = state_final.s  # (B, M, d_hidden)
        else:
            s_final = self.deliberation_core.input_proj(fused_state).unsqueeze(1).expand(
                fused_state.size(0), self.num_agents, self.d_hidden
            ).clone()

        L_agent_orth = core_out.get("L_orth", core_out.get("orth_loss", torch.tensor(0.0, device=turn_embeddings.device)))
        L_task_orth = self.compute_task_orthogonality_loss()
        L_total_orth = L_agent_orth + 0.5 * L_task_orth

        results = {
            "fused_state": fused_state,
            "attn_weights": attn_weights,  # (B, 1, T) causal attribution across turns
            "orth_loss": L_total_orth,
            "task_orth_loss": L_task_orth,
        }

        # 3. Decode requested or all heads
        if task == "risk" or task is None:
            risk_out = self._decode_branch(
                s_final, self.adapter_risk, self.head_risk, self.consensus_risk, self.ats_risk, 2
            )
            results["risk"] = risk_out
            if task == "risk":
                results["p_final"] = risk_out["p_final"]
                results["p_calibrated"] = risk_out["p_calibrated"]
                results["b_all"] = risk_out["b_all"]

        if task == "taxonomy" or task is None:
            tax_out = self._decode_branch(
                s_final, self.adapter_taxonomy, self.head_taxonomy, self.consensus_taxonomy, self.ats_taxonomy, self.num_taxonomy_classes
            )
            results["taxonomy"] = tax_out
            if task == "taxonomy":
                results["p_final"] = tax_out["p_final"]
                results["p_calibrated"] = tax_out["p_calibrated"]
                results["b_all"] = tax_out["b_all"]

        if task == "mcp" or task is None:
            mcp_out = self._decode_branch(
                s_final, self.adapter_mcp, self.head_mcp, self.consensus_mcp, self.ats_mcp, self.num_mcp_tools
            )
            results["mcp"] = mcp_out
            if task == "mcp":
                results["p_final"] = mcp_out["p_final"]
                results["p_calibrated"] = mcp_out["p_calibrated"]
                results["b_all"] = mcp_out["b_all"]

        if task == "policy" or task is None:
            pol_out = self._decode_branch(
                s_final, self.adapter_policy, self.head_policy, self.consensus_policy, self.ats_policy, self.num_policy_actions
            )
            results["policy"] = pol_out
            if task == "policy":
                results["p_final"] = pol_out["p_final"]
                results["p_calibrated"] = pol_out["p_calibrated"]
                results["b_all"] = pol_out["b_all"]

        # 4. Synthesize Autonomous Intervention Policy
        if task is None:
            p_harm = results["risk"]["p_calibrated"][:, 1]
            u_epi = results["risk"]["uncertainty"]
            # Intervention rule: Block if harmful or high epistemic vacuity on multi-turn trajectory
            should_block = (p_harm >= 0.5) | (u_epi >= 0.45)
            policy_action = torch.where(should_block, 1, 0)  # 1: BLOCK, 0: ALLOW

            results["intervention"] = {
                "p_harm": p_harm,
                "uncertainty": u_epi,
                "should_block": should_block,
                "policy_action": policy_action,
                "action_name": [INTERVENTION_POLICIES[idx] for idx in policy_action.tolist()],
            }

        return results
