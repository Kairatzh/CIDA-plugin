"""
cida_inference — Backward-Compatible Alias
==========================================

This package is a compatibility shim. All classes now live in
``cida_plugin.inference`` (the canonical location).

Both import styles work identically:

    # New canonical form (recommended)
    from cida_plugin.inference import CIDAWrapper, LatentAgent

    # Old form — still works, no code changes needed
    from cida_inference import CIDAWrapper, LatentAgent

Deprecated since: cida-plugin v1.2.0
"""
import warnings as _warnings

# Re-export everything from the canonical package location
from cida_plugin.inference import (
    CIDAWrapper,
    LatentMemory,
    LatentReasoner,
    LatentAgent,
)

# Also expose sub-modules so that deep imports like
# `from cida_inference.parser_utils import ...` keep working
from cida_plugin.inference import parser_utils   # noqa: F401
from cida_plugin.inference import callbacks      # noqa: F401
from cida_plugin.inference import backbone       # noqa: F401

__all__ = [
    "CIDAWrapper",
    "LatentMemory",
    "LatentReasoner",
    "LatentAgent",
]

__version__ = "1.2.0"
