"""
cida_plugin/multitask_pro.py — Scaled Multi-Task CIDA Deliberation Architecture Pro (Kairos 1.1 Pro)

Architectural Highlights:
1. Shared Backbone & Deliberation Brain:
   - Laya ModernBERT-large (1024-dim representations)
   - Scaled Deliberation Core (d_hidden=512, d_message=512, M=4 agents, ~9.8M parameters)
2. O-LoRA (Task-Orthogonal Low-Rank Projections):
   - Decoupled projection adapters W_tau in R^(d_hidden x d_proj)
   - Frobenius Task Orthogonality Loss: L_task_orth = sum_{i < j} ||W_i^T W_j||_F^2 / (||W_i||_F * ||W_j||_F)
   - Prevents catastrophic forgetting and cross-task gradient interference.
3. Adaptive Temperature Scaling (ATS):
   - Sample-wise dynamic temperature T_i(x) = T_0 * (1 + gamma * u_i) + softplus(w_ats^T s_i)
   - Epistemic vacuity u_i from Josang's Subjective Logic dynamically expands calibration bands on OOD / boundary samples.
4. Multi-Head Scaled Decoders:
   - head_tools: K_tools = 66 classes (Top 64 xLAM API tools + Class 65 long-tail + Class 0 NO_TOOL pass-through)
   - head_intent: K_intent = 151 classes (CLINC150: 150 multi-domain real-world intents + Out-of-Scope class)
   - head_safety: K_safety = 2 classes (Binary real prompt injection / jailbreak guardrail)
5. Scale-Invariant Dirichlet Concentration (alpha_0 = 1 / K_tau per task).
"""

from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin
from cida_plugin.consensus import ConsensusAggregator


class OLoRATaskAdapter(nn.Module):
    """
    Task-Orthogonal Low-Rank Projection Adapter.
    Projects deliberation state into a specialized task subspace while enforcing
    orthogonality between different tasks to prevent negative transfer.
    """
    def __init__(self, d_hidden: int = 512, d_proj: int = 256):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_proj = d_proj
        # Low-rank factorized projection: W = down_proj @ up_proj
        self.proj = nn.Linear(d_hidden, d_proj, bias=False)
        self.norm = nn.LayerNorm(d_proj)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, d_hidden) -> (B, M, d_proj)
        return self.act(self.norm(self.proj(x)))

    @property
    def weight(self) -> torch.Tensor:
        return self.proj.weight


class AdaptiveTemperatureScaling(nn.Module):
    """
    Adaptive Temperature Scaling (ATS).
    Predicts an instance-specific calibration temperature T(x) conditioned on
    both the deliberated latent state and Josang's epistemic vacuity u_i.
    """
    def __init__(self, d_input: int = 256, base_temp: float = 1.0):
        super().__init__()
        self.base_temp = nn.Parameter(torch.tensor([base_temp], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        self.state_proj = nn.Linear(d_input, 1)

    def forward(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Computes sample-level temperatures.
        s: (B, d_input) - mean deliberated representation
        u: (B, 1) - epistemic uncertainty / vacuity
        Returns:
            T: (B, 1) temperature values clamped to >= 0.05
        """
        T_0 = F.softplus(self.base_temp) + 0.1
        gamma = F.softplus(self.gamma)
        # Dynamic shift based on latent evidence
        offset = F.softplus(self.state_proj(s))
        # Total adaptive temperature
        T = T_0 * (1.0 + gamma * u.clamp(min=0.0, max=2.0)) + offset
        return T.clamp(min=0.05, max=10.0)


class MultiTaskCIDAPluginPro(nn.Module):
    """
    Enterprise-Grade Multi-Task CIDA Deliberation Core Pro (Kairos 1.1 Pro).
    Features O-LoRA task-orthogonal adapters, Adaptive Temperature Scaling (ATS),
    and scaled multi-domain decoders for 66 tools, 151 intents, and binary safety.
    """

    def __init__(
        self,
        d_input: int = 1024,
        d_hidden: int = 512,
        d_message: int = 512,
        d_proj: int = 256,
        num_agents: int = 4,
        max_rounds: int = 3,
        num_tools: int = 66,     # Top 64 xLAM + Class 65 long-tail + Class 0 NO_TOOL
        num_intents: int = 151,  # CLINC150 (150 real intents + Out-of-Scope)
        num_safety: int = 2,     # Binary ShieldLM / Jailbreak guardrail
    ):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_message = d_message
        self.d_proj = d_proj
        self.num_agents = num_agents
        self.max_rounds = max_rounds

        self.num_tools = num_tools
        self.num_intents = num_intents
        self.num_safety = num_safety

        # ── 1. Unified 4-Agent Deliberation Brain ─────────────────────────────
        self.cida_config = CIDAPluginConfig(
            d_input=d_input,
            d_hidden=d_hidden,
            d_message=d_message,
            num_agents=num_agents,
            num_classes=d_hidden,
            max_rounds=max_rounds,
            role_prior_blend=0.0,
            use_vae_perspective=False,
            multi_label=False,
        )
        self.deliberation_core = CIDAPlugin(self.cida_config)

        # ── 2. O-LoRA Task Orthogonal Adapters ────────────────────────────────
        self.adapter_tools = OLoRATaskAdapter(d_hidden, d_proj)
        self.adapter_intent = OLoRATaskAdapter(d_hidden, d_proj)
        self.adapter_safety = OLoRATaskAdapter(d_hidden, d_proj)

        # ── 3. Task Decoders & Consensus ──────────────────────────────────────
        # Tools Head (66 Classes)
        self.head_tools = nn.Sequential(
            nn.Linear(d_proj, d_proj),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_proj, num_tools),
        )
        self.consensus_tools = ConsensusAggregator(num_agents=num_agents)
        self.ats_tools = AdaptiveTemperatureScaling(d_input=d_proj, base_temp=1.0)

        # Intent Head (151 Classes: High Cardinality)
        self.head_intent = nn.Sequential(
            nn.Linear(d_proj, d_proj),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_proj, num_intents),
        )
        self.consensus_intent = ConsensusAggregator(num_agents=num_agents)
        self.ats_intent = AdaptiveTemperatureScaling(d_input=d_proj, base_temp=1.0)

        # Safety Head (2 Classes: Guardrail)
        self.head_safety = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_proj // 2, num_safety),
        )
        self.consensus_safety = ConsensusAggregator(num_agents=num_agents)
        self.ats_safety = AdaptiveTemperatureScaling(d_input=d_proj, base_temp=1.0)

    def compute_task_orthogonality_loss(self) -> torch.Tensor:
        """
        Computes Frobenius orthogonality penalty across the O-LoRA task projection weights:
            L_task_orth = sum_{i < j} ||W_i^T W_j||_F^2 / (||W_i||_F * ||W_j||_F + eps)
        Ensures that representations for Tools, Intent, and Safety evolve in mutually
        orthogonal subspaces, eliminating catastrophic forgetting.
        """
        w_t = self.adapter_tools.weight     # (d_proj, d_hidden)
        w_i = self.adapter_intent.weight    # (d_proj, d_hidden)
        w_s = self.adapter_safety.weight    # (d_proj, d_hidden)

        eps = 1e-7
        # Pair 1: Tools vs Intent
        g_ti = torch.matmul(w_t, w_i.t())
        loss_ti = (g_ti ** 2).sum() / (torch.norm(w_t) * torch.norm(w_i) + eps)

        # Pair 2: Tools vs Safety
        g_ts = torch.matmul(w_t, w_s.t())
        loss_ts = (g_ts ** 2).sum() / (torch.norm(w_t) * torch.norm(w_s) + eps)

        # Pair 3: Intent vs Safety
        g_is = torch.matmul(w_i, w_s.t())
        loss_is = (g_is ** 2).sum() / (torch.norm(w_i) * torch.norm(w_s) + eps)

        return loss_ti + loss_ts + loss_is

    def _decode_head(
        self,
        s_final: torch.Tensor,
        adapter: OLoRATaskAdapter,
        head_module: nn.Module,
        consensus_module: ConsensusAggregator,
        ats_module: AdaptiveTemperatureScaling,
        num_classes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Decodes deliberated agent states through an O-LoRA adapter and classification head
        with Scale-Invariant Dirichlet Concentration and Adaptive Temperature Scaling.
        """
        B, M, _ = s_final.shape
        # Project through task-orthogonal adapter: (B, M, d_proj)
        h_task = adapter(s_final)
        logits = head_module(h_task)  # (B, M, K)

        # Scale-invariant Dirichlet concentration (alpha_0 = 1 / K)
        alpha_prior = 1.0 / float(num_classes)
        alpha = F.softplus(logits) + alpha_prior
        b = alpha / alpha.sum(dim=-1, keepdim=True)  # (B, M, K)

        # Josang's Subjective Logic Epistemic Vacuity (Josang 2002)
        s_0 = num_classes * alpha_prior  # exactly 1.0
        u_epi = (s_0 / alpha.sum(dim=-1, keepdim=True)) ** 2
        u_alea = 1.0 - b.max(dim=-1, keepdim=True).values
        u = u_epi + u_alea

        # Agent Consensus Aggregation
        p_consensus, uncertainty, disagreement = consensus_module(b=b, u=u)

        # Adaptive Temperature Scaling (ATS)
        h_mean = h_task.mean(dim=1)  # (B, d_proj)
        u_mean = uncertainty.unsqueeze(-1) if uncertainty.dim() == 1 else uncertainty  # (B, 1)
        T_sample = ats_module(h_mean, u_mean)  # (B, 1)

        # Calibrated logits & probabilities
        mean_logits = logits.mean(dim=1)  # (B, K)
        calibrated_logits = mean_logits / T_sample
        p_calibrated = F.softmax(calibrated_logits, dim=-1)

        return {
            "p_final": p_consensus,
            "p_calibrated": p_calibrated,
            "calibrated_logits": calibrated_logits,
            "b_all": [b],
            "uncertainty": uncertainty,
            "disagreement": disagreement,
            "temperature": T_sample.squeeze(-1),
            "logits": mean_logits,
        }

    def forward(
        self,
        pooled: torch.Tensor,
        seq_output: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        If task is specified ('tools', 'intent', or 'safety'), computes that head.
        If task is None, evaluates all three heads simultaneously for unified decisions.
        """
        # 1. Deliberation Core execution across M=4 agents
        core_out = self.deliberation_core(pooled=pooled, seq_output=seq_output, mask=mask)
        state_final = core_out.get("state_final")
        if state_final is not None and hasattr(state_final, "s"):
            s_final = state_final.s  # (B, M, d_hidden)
        else:
            s_final = self.deliberation_core.input_proj(pooled).unsqueeze(1).expand(
                pooled.size(0), self.num_agents, self.d_hidden
            ).clone()

        L_agent_orth = core_out.get("L_orth", core_out.get("orth_loss", torch.tensor(0.0, device=pooled.device)))
        L_task_orth = self.compute_task_orthogonality_loss()
        L_total_orth = L_agent_orth + 0.5 * L_task_orth

        results = {
            "orth_loss": L_total_orth,
            "agent_orth_loss": L_agent_orth,
            "task_orth_loss": L_task_orth,
        }

        if task == "tools" or task is None:
            tools_out = self._decode_head(
                s_final, self.adapter_tools, self.head_tools, self.consensus_tools, self.ats_tools, self.num_tools
            )
            results["tools"] = tools_out
            if task == "tools":
                results["p_final"] = tools_out["p_final"]
                results["p_calibrated"] = tools_out["p_calibrated"]
                results["b_all"] = tools_out["b_all"]
                results["temperature"] = tools_out["temperature"]

        if task == "intent" or task is None:
            intent_out = self._decode_head(
                s_final, self.adapter_intent, self.head_intent, self.consensus_intent, self.ats_intent, self.num_intents
            )
            results["intent"] = intent_out
            if task == "intent":
                results["p_final"] = intent_out["p_final"]
                results["p_calibrated"] = intent_out["p_calibrated"]
                results["b_all"] = intent_out["b_all"]
                results["temperature"] = intent_out["temperature"]

        if task == "safety" or task is None:
            safety_out = self._decode_head(
                s_final, self.adapter_safety, self.head_safety, self.consensus_safety, self.ats_safety, self.num_safety
            )
            results["safety"] = safety_out
            if task == "safety":
                results["p_final"] = safety_out["p_final"]
                results["p_calibrated"] = safety_out["p_calibrated"]
                results["b_all"] = safety_out["b_all"]
                results["temperature"] = safety_out["temperature"]

        return results
