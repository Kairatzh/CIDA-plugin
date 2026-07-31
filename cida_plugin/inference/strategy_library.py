"""
Strategy Library — Long-Term Latent Memory for Meta-Learning & Tactical Reuse.

Stores successful latent states, FiLM modulation weights, and problem-solving tactics.
Prevents CIDA from re-deliberating from scratch when encountering similar problem classes.
Supports strategy evolution (mutation: S_{t+1} = S_t + ε).

Usage:
    lib = StrategyLibrary()
    lib.save_strategy("code_review", latent_state, film_params, score=0.95)

    best_strat = lib.find_best("code_review", query_embed)
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class Strategy:
    """
    Data container for a saved problem-solving strategy.
    """
    task_type: str
    latent_state: torch.Tensor          # (1, d_cida)
    film_params: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None  # [(gamma, beta), ...]
    score: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mutate(self, noise_std: float = 0.05) -> 'Strategy':
        """
        Evolve strategy via perturbation: S_{t+1} = S_t + ε.
        """
        noise = torch.randn_like(self.latent_state) * noise_std
        mutated_state = self.latent_state + noise

        mutated_film = None
        if self.film_params:
            mutated_film = []
            for gamma, beta in self.film_params:
                g_noise = torch.randn_like(gamma) * noise_std
                b_noise = torch.randn_like(beta) * noise_std
                mutated_film.append((gamma + g_noise, beta + b_noise))

        return Strategy(
            task_type=self.task_type,
            latent_state=mutated_state,
            film_params=mutated_film,
            score=self.score,
            metadata={**self.metadata, "parent_score": self.score, "mutated": True}
        )


class BaseStrategyStore:
    """Abstract storage engine for strategy library."""
    def add(self, strategy: Strategy):
        raise NotImplementedError
    def get_all(self, task_type: Optional[str] = None) -> List[Strategy]:
        raise NotImplementedError
    def clear(self):
        raise NotImplementedError


class InMemoryStrategyStore(BaseStrategyStore):
    """In-memory storage backend."""
    def __init__(self):
        self.strategies: List[Strategy] = []

    def add(self, strategy: Strategy):
        self.strategies.append(strategy)

    def get_all(self, task_type: Optional[str] = None) -> List[Strategy]:
        if task_type is None:
            return list(self.strategies)
        return [s for s in self.strategies if s.task_type == task_type]

    def clear(self):
        self.strategies = []


class StrategyLibrary:
    """
    Manager for storing, retrieving, and evolving latent strategies.

    Parameters
    ----------
    store : BaseStrategyStore, optional
        Backend store instance (default: InMemoryStrategyStore).
    similarity_threshold : float
        Min cosine similarity to match query context (default: 0.5).
    """

    def __init__(self, store: Optional[BaseStrategyStore] = None, similarity_threshold: float = 0.5):
        self.store = store if store is not None else InMemoryStrategyStore()
        self.similarity_threshold = similarity_threshold

    def save_strategy(
        self,
        task_type: str,
        latent_state: torch.Tensor,
        film_params: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Strategy:
        """Save a validated, high-performing strategy."""
        strat = Strategy(
            task_type=task_type,
            latent_state=latent_state.detach().cpu(),
            film_params=[(g.detach().cpu(), b.detach().cpu()) for g, b in film_params] if film_params else None,
            score=score,
            metadata=metadata or {}
        )
        self.store.add(strat)
        return strat

    def find_best(
        self,
        task_type: str,
        query_embedding: torch.Tensor,
        top_k: int = 1
    ) -> List[Tuple[Strategy, float]]:
        """
        Find top-k strategies matching task_type and highest cosine similarity to query_embedding.
        """
        candidates = self.store.get_all(task_type)
        if not candidates:
            return []

        q_norm = F.normalize(query_embedding.detach().cpu().float().reshape(1, -1), dim=-1)

        matches = []
        for strat in candidates:
            s_vec = strat.latent_state.float().reshape(1, -1)
            # Ensure dimensions match for similarity computation
            min_dim = min(q_norm.size(1), s_vec.size(1))
            s_norm = F.normalize(s_vec[:, :min_dim], dim=-1)
            sim = F.cosine_similarity(q_norm[:, :min_dim], s_norm).item()

            if sim >= self.similarity_threshold or len(candidates) <= 2:
                matches.append((strat, sim * strat.score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def evolve_strategy(self, strategy: Strategy, noise_std: float = 0.05) -> Strategy:
        """Generate a mutated strategy variant for exploration."""
        mutated = strategy.mutate(noise_std=noise_std)
        self.store.add(mutated)
        return mutated

    def save_to_disk(self, filepath: str):
        """Save entire strategy library to disk using torch.save."""
        all_strats = self.store.get_all()
        data = []
        for s in all_strats:
            data.append({
                "task_type": s.task_type,
                "latent_state": s.latent_state,
                "film_params": s.film_params,
                "score": s.score,
                "timestamp": s.timestamp,
                "metadata": s.metadata
            })
        torch.save(data, filepath)

    def load_from_disk(self, filepath: str, map_location="cpu"):
        """Load strategy library from disk."""
        data = torch.load(filepath, map_location=map_location)
        self.store.clear()
        for d in data:
            strat = Strategy(
                task_type=d["task_type"],
                latent_state=d["latent_state"].to(map_location),
                film_params=[(g.to(map_location), b.to(map_location)) for g, b in d["film_params"]] if d["film_params"] else None,
                score=d["score"],
                timestamp=d.get("timestamp", time.time()),
                metadata=d.get("metadata", {})
            )
            self.store.add(strat)
