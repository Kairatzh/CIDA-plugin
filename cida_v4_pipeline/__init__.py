"""
CIDA v4 Production Pipeline
Universal Cognitive Co-Processor for Large Language Models.
"""

from .core_v4 import (
    CIDAv4Core,
    CIDAv4Config,
    PerceiverContextBottleneck,
    BayesianRoleDebate,
    UnrolledRecurrentLoop,
    VariationalPlanProjector,
)
from .kv_steerer import PrefixKVSteerer
from .dataset_curator import DatasetCurator
from .cache_latents import OfflineLatentCacher

__version__ = "4.0.0"
__all__ = [
    "CIDAv4Core",
    "CIDAv4Config",
    "PerceiverContextBottleneck",
    "BayesianRoleDebate",
    "UnrolledRecurrentLoop",
    "VariationalPlanProjector",
    "PrefixKVSteerer",
    "DatasetCurator",
    "OfflineLatentCacher",
    "CognitiveStatePack",
    "LatentVAEBottleneck",
    "JEPATrajectoryEnergyPredictor",
    "LangevinDynamicsRefiner",
    "ContrastiveActivationSteerer",
    "WhiteBoxSentinel",
    "BlackBoxSentinel",
    "OmniSentinel",
    "FullCycleSentinelTrainer",
]

from .sentinel_dual import (
    CognitiveStatePack,
    LatentVAEBottleneck,
    JEPATrajectoryEnergyPredictor,
    LangevinDynamicsRefiner,
    ContrastiveActivationSteerer,
    WhiteBoxSentinel,
    BlackBoxSentinel,
    OmniSentinel,
)
from .train_full_cycle import FullCycleSentinelTrainer
