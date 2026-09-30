"""
cida_plugin/multitask.py — Scaled Multi-Task CIDA Deliberation Architecture (MT-Kairos)

Architecture:
- Shared Feature Extractor: Laya ModernBERT-large (1024-dim representations)
- Scaled Deliberation Core (d_hidden=512, d_message=512, M=4 agents, ~8.2M parameters)
- Multi-Head Action Decoders:
    1. head_tools: K_tools classes (Top API tools + Class 0 = NO_TOOL / PASS_THROUGH)
    2. head_intent: K_intent classes (77 banking & customer intents)
    3. head_safety: 2 classes (0: Benign / Safe, 1: Prompt Injection / Jailbreak Attack)
- Scale-Invariant Dirichlet Concentration (alpha_0 = 1/K_tau per task)
- Latent Rollback & Contrastive Gram Orthogonality
"""

from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin
from cida_plugin.consensus import ConsensusAggregator


class MultiTaskCIDAPlugin(nn.Module):
    """
    Scaled Multi-Task CIDA Deliberation Core (MT-Kairos).
    
    Contains a unified 4-agent deliberation brain operating in d_hidden=512,
    routing decisions to specialized action heads for Tools, Intent, and Safety.
    """

    def __init__(
        self,
        d_input: int = 1024,
        d_hidden: int = 512,
        d_message: int = 512,
        num_agents: int = 4,
        max_rounds: int = 3,
        num_tools: int = 65,     # 64 tools + 1 NO_TOOL
        num_intents: int = 77,   # Banking77
        num_safety: int = 2,     # Binary guardrail
    ):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_message = d_message
        self.num_agents = num_agents
        self.max_rounds = max_rounds

        self.num_tools = num_tools
        self.num_intents = num_intents
        self.num_safety = num_safety

        # ── 1. Scaled Deliberation Engine ─────────────────────────────────────
        # Deliberation operates in d_hidden=512 across M=4 agents with role_prior_blend=0.0
        self.cida_config = CIDAPluginConfig(
            d_input=d_input,
            d_hidden=d_hidden,
            d_message=d_message,
            num_agents=num_agents,
            num_classes=d_hidden,  # internal representation space
            max_rounds=max_rounds,
            role_prior_blend=0.0,
            use_vae_perspective=False,
            multi_label=False,
        )
        self.deliberation_core = CIDAPlugin(self.cida_config)

        # ── 2. Task-Specific Consensus & Output Heads ─────────────────────────
        # Tools Head: 64 Tools + NO_TOOL rejection class
        self.head_tools = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, num_tools),
        )
        self.consensus_tools = ConsensusAggregator(num_agents=num_agents)

        # Intent Head: 77 Customer Intent Classes
        self.head_intent = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, num_intents),
        )
        self.consensus_intent = ConsensusAggregator(num_agents=num_agents)

        # Safety Head: Binary Prompt Injection / Jailbreak Guardrail
        self.head_safety = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden // 2, num_safety),
        )
        self.consensus_safety = ConsensusAggregator(num_agents=num_agents)

    def _decode_head(
        self,
        s_final: torch.Tensor,
        head_module: nn.Module,
        consensus_module: ConsensusAggregator,
        num_classes: int,
    ) -> Dict[str, torch.Tensor]:
        """Maps deliberated states through a task head with Scale-Invariant Dirichlet."""
        B, M, _ = s_final.shape
        logits = head_module(s_final)  # (B, M, K)

        # Scale-invariant Dirichlet concentration (alpha_0 = 1 / K)
        alpha_prior = 1.0 / float(num_classes)
        alpha = F.softplus(logits) + alpha_prior
        b = alpha / alpha.sum(dim=-1, keepdim=True)  # (B, M, K)

        # Epistemic vacuity (Josang 2002)
        s_0 = num_classes * alpha_prior  # exactly 1.0
        u_epi = (s_0 / alpha.sum(dim=-1, keepdim=True)) ** 2
        u_alea = 1.0 - b.max(dim=-1, keepdim=True).values
        u = u_epi + u_alea

        p_final, uncertainty, disagreement = consensus_module(b=b, u=u)

        return {
            "p_final": p_final,
            "b_all": [b],
            "uncertainty": uncertainty,
            "disagreement": disagreement,
            "logits": logits.mean(dim=1),  # mean logits across agents for calibration
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
        # Run deliberation core
        core_out = self.deliberation_core(pooled=pooled, seq_output=seq_output, mask=mask)
        # Extract deliberated states across agents after cross-agent communication
        state_final = core_out.get("state_final")
        if state_final is not None and hasattr(state_final, "s"):
            s_final = state_final.s
        else:
            s_final = self.deliberation_core.input_proj(pooled).unsqueeze(1).expand(
                pooled.size(0), self.num_agents, self.d_hidden
            ).clone()

        L_orth = core_out.get("L_orth", core_out.get("orth_loss", torch.tensor(0.0, device=pooled.device)))
        results = {"orth_loss": L_orth}

        if task == "tools" or task is None:
            tools_out = self._decode_head(
                s_final, self.head_tools, self.consensus_tools, self.num_tools
            )
            results["tools"] = tools_out
            if task == "tools":
                results["p_final"] = tools_out["p_final"]
                results["b_all"] = tools_out["b_all"]

        if task == "intent" or task is None:
            intent_out = self._decode_head(
                s_final, self.head_intent, self.consensus_intent, self.num_intents
            )
            results["intent"] = intent_out
            if task == "intent":
                results["p_final"] = intent_out["p_final"]
                results["b_all"] = intent_out["b_all"]

        if task == "safety" or task is None:
            safety_out = self._decode_head(
                s_final, self.head_safety, self.consensus_safety, self.num_safety
            )
            results["safety"] = safety_out
            if task == "safety":
                results["p_final"] = safety_out["p_final"]
                results["b_all"] = safety_out["b_all"]

        return results
