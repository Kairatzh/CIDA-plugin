"""
CIDA-Plugin: Universal Evidence-Grounded Multi-Agent Deliberation Layer (v3).

Использование (минимальное):
    from cida_plugin import CIDAPlugin, CIDAPluginConfig

    cfg = CIDAPluginConfig(d_input=768, num_classes=2)
    plugin = CIDAPlugin(cfg)
    out = plugin(pooled_output)   # pooled_output: (B, 768)
    logits = out["p_final"]       # (B, 2)

Использование (с seq_output для evidence pointers):
    out = plugin(pooled_output, seq_output=hidden_states, mask=attention_mask)
"""

from .config import CIDAPluginConfig, CIDAPluginConfig as LatentConfig, CIDAPluginConfig as CidaConfig
from .core import CIDAPlugin, CIDAPlugin as LatentCIDAPlugin, CIDAPlugin as CidaModel, PlanProjector
from .agent import (
    AgentState,
    apply_role_prior,
    compute_role_orthogonality_loss,
    RoleNames,
)
from .deliberation import (
    AgentEvidenceExtractor,
    MessageFormulator,
    CounterargumentCommunication,
    AgentUpdater,
)
from .consensus import ConsensusAggregator, HaltingPredictor
from .losses import CIDALoss
from .diagnostics import DebateDiagnostics
from .ttt import TestTimeTrainer
from .liquid_dynamics import LiquidDeliberationSolver
from .hf import wrap_hf_model, HFModelWithCIDA, wrap_hf_causal_lm, HFModelForCausalLMWithCIDA
from .projector import LatentAEProjector, TargetEncoder
from .inference import CIDAWrapper, LatentMemory, LatentReasoner, LatentAgent, CidaSwarm

__all__ = [
    # Главный интерфейс
    "CIDAPlugin",
    "CIDAPluginConfig",
    "LatentCIDAPlugin",
    "LatentConfig",
    "CidaModel",
    "CidaConfig",
    "PlanProjector",
    "LatentAEProjector",
    "TargetEncoder",
    "wrap_hf_model",
    "HFModelWithCIDA",
    "wrap_hf_causal_lm",
    "HFModelForCausalLMWithCIDA",
    # Потери и диагностика
    "CIDALoss",
    "DebateDiagnostics",
    # [v5] Test-Time Training
    "TestTimeTrainer",
    # [v5] Liquid Neural ODE Dynamics
    "LiquidDeliberationSolver",
    # Компоненты (для кастомных архитектур)
    "AgentState",
    "apply_role_prior",
    "compute_role_orthogonality_loss",
    "RoleNames",
    "AgentEvidenceExtractor",
    "MessageFormulator",
    "CounterargumentCommunication",
    "AgentUpdater",
    "ConsensusAggregator",
    "HaltingPredictor",
    # [v1.5] Inference / Agentic classes
    "CIDAWrapper",
    "LatentMemory",
    "LatentReasoner",
    "LatentAgent",
    # [v1.2] Multi-agent Swarm
    "CidaSwarm",
]

