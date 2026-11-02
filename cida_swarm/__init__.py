"""CIDA-Swarm: рой дронов с коллективным мышлением на основе CIDA (см. docs/CIDA_SWARM_ARCHITECTURE.md)."""

from .env import EnvConfig, SearchEnv
from .policy import Actor, Critic

__all__ = ["EnvConfig", "SearchEnv", "Actor", "Critic"]
