"""
CIDA-Plugin: Universal Evidence-Grounded Multi-Agent Deliberation Layer.

Использование (минимальное):
    from cida_plugin import CIDAPlugin, CIDAPluginConfig

    cfg = CIDAPluginConfig(d_input=768, num_classes=2)
    plugin = CIDAPlugin(cfg)
    out = plugin(pooled_output)   # pooled_output: (B, 768)
    logits = out["p_final"]       # (B, 2)

Использование (с seq_output для evidence pointers):
    out = plugin(pooled_output, seq_output=hidden_states, mask=attention_mask)
"""

from .config import CIDAPluginConfig
from .core import CIDAPlugin
from .agent import AgentState, RoleEmbeddings
from .deliberation import (
    AgentEvidenceExtractor,
    MessageFormulator,
    CounterargumentCommunication,
    AgentUpdater,
)
from .consensus import ConsensusAggregator, HaltingPredictor
from .losses import OmegaLossSystem
from .diagnostics import DebateDiagnostics

__all__ = [
    # Главный интерфейс
    "CIDAPlugin",
    "CIDAPluginConfig",
    # Потери и диагностика
    "OmegaLossSystem",
    "DebateDiagnostics",
    # Компоненты (для кастомных архитектур)
    "AgentState",
    "RoleEmbeddings",
    "AgentEvidenceExtractor",
    "MessageFormulator",
    "CounterargumentCommunication",
    "AgentUpdater",
    "ConsensusAggregator",
    "HaltingPredictor",
]
