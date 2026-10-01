"""
cida_v4_pipeline/multitask_pro.py — Standalone Mirror of Kairos 1.1 Pro Architecture
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from cida_plugin.multitask_pro import (
    MultiTaskCIDAPluginPro,
    OLoRATaskAdapter,
    AdaptiveTemperatureScaling,
)

__all__ = [
    "MultiTaskCIDAPluginPro",
    "OLoRATaskAdapter",
    "AdaptiveTemperatureScaling",
]
