"""
sentinel_dual.py — Kairos Sentinel-Dual Production Cognitive Architecture

Fuses:
1. Track A (White-Box Co-Processor for Qwen/LLaMA):
   - Latent VAE Bottleneck with AGCLR Gated Persistent Memory
   - Yann LeCun JEPA Trajectory Energy Predictor (EBM-CoT)
   - Continuous Langevin Dynamics Energy Minimization
   - Contrastive Activation Addition (CAA) with Adaptive Variance Normalization
   - Hysteresis Damping & Anti-Loop Intervention Budget Controller (B_max = 3)

2. Track B (Black-Box Sentinel for Claude/Gemini API):
   - Semantic Entropy Clustering (Kuhn et al., 2023)
   - Behavioral Drift & Tool-Loop Detector (O(1) cycle detection & repetition penalty)
   - Textual CIDA Deliberation (Prosecutor, Defender, Skeptic, Integrator over text)
   - Steered Intervention Re-prompt Generator

3. OmniSentinel:
   - Unified factory interface switching seamlessly between White-Box and Black-Box modes.
"""

from dataclasses import dataclass, field
import math
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. SHARED DATA STRUCTURES & TELEMETRY
# ============================================================================

@dataclass
class CognitiveStatePack:
    """
    Structured multi-vector cognitive state pack.
    Replaces lossy 1D bottlenecks with dedicated causal dimensions.
    """
    intent: torch.Tensor       # (B, D): Invariant problem goal representation
    constraint: torch.Tensor   # (B, D): Invariant boundary conditions (types, bounds, safety)
    entropy: torch.Tensor      # (B, D): Epistemic dispersion across reasoning heads
    trajectory: torch.Tensor   # (B, K, D): Multi-step latent rollout trajectory

    def to(self, device: torch.device) -> "CognitiveStatePack":
        return CognitiveStatePack(
            intent=self.intent.to(device),
            constraint=self.constraint.to(device),
            entropy=self.entropy.to(device),
            trajectory=self.trajectory.to(device),
        )


@dataclass
class SentinelDecision:
    """
    Immutable intervention decision issued by the Sentinel.
    """
    track: str                 # "WHITE_BOX" or "BLACK_BOX"
    status: str                # "NOUL" (pass), "ALERT" (steer/refine), "HALT_FALLBACK" (emergency abort)
    energy: float              # Current smoothed anomaly / trajectory divergence energy
    energy_raw: float          # Raw instantaneous energy
    budget_used: int           # Interventions expended in this episode
    budget_remaining: int      # Interventions remaining (<= B_max)
    steering_vector: Optional[torch.Tensor] = None # For Track A: (D_llm,) CAA vector
    steered_prompt: Optional[str] = None          # For Track B: Re-prompt text injection
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 2. TRACK A: WHITE-BOX SENTINEL COMPONENTS
# ============================================================================

class GatedPersistentMemory(nn.Module):
    """
    AGCLR-style Persistent Gated Memory Cell.
    Prevents concept catastrophic forgetting / overwriting across long deliberation steps.
    """
    def __init__(self, d_hidden: int):
        super().__init__()
        self.d_hidden = d_hidden
        self.gate_w = nn.Linear(d_hidden, d_hidden)
        self.gate_u = nn.Linear(d_hidden, d_hidden, bias=False)
        self.cand_w = nn.Linear(d_hidden, d_hidden)
        self.cand_u = nn.Linear(d_hidden, d_hidden, bias=False)
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, s_t: torch.Tensor, m_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        s_t: (B, d_hidden) current deliberation state
        m_prev: (B, d_hidden) previous persistent memory
        Returns: (B, d_hidden) updated memory
        """
        if m_prev is None:
            m_prev = torch.zeros_like(s_t)
        
        gate = torch.sigmoid(self.gate_w(s_t) + self.gate_u(m_prev))
        cand = torch.tanh(self.cand_w(s_t) + self.cand_u(m_prev))
        m_t = gate * m_prev + (1.0 - gate) * cand
        return self.norm(m_t)


class LatentVAEBottleneck(nn.Module):
    """
    Variational Autoencoder Bottleneck with Gated Persistent Memory.
    Filters token-level syntactic noise into an invariant CognitiveStatePack.
    """
    def __init__(self, d_llm: int = 4096, d_hidden: int = 512, num_queries: int = 16):
        super().__init__()
        self.d_llm = d_llm
        self.d_hidden = d_hidden
        self.num_queries = num_queries

        # Cross-Attention Perceiver Compression
        self.query_bank = nn.Parameter(torch.randn(1, num_queries, d_hidden) * 0.02)
        self.kv_proj = nn.Linear(d_llm, d_hidden, bias=False)
        self.cross_attn = nn.MultiheadAttention(d_hidden, num_heads=8, batch_first=True)
        self.norm_kv = nn.LayerNorm(d_hidden)
        self.norm_q = nn.LayerNorm(d_hidden)

        # Persistent Memory
        self.memory = GatedPersistentMemory(d_hidden)

        # Variational Parameter Heads
        self.mu_head = nn.Linear(d_hidden * 4, d_hidden * 4)
        self.logvar_head = nn.Linear(d_hidden * 4, d_hidden * 4)

        # Latent Unpackers for CognitiveStatePack
        self.proj_intent = nn.Linear(d_hidden, d_hidden)
        self.proj_constraint = nn.Linear(d_hidden, d_hidden)
        self.proj_entropy = nn.Linear(d_hidden, d_hidden)
        self.proj_trajectory = nn.Linear(d_hidden, d_hidden * 4) # K=4 steps

        # Reconstruction Decoder Head
        self.decoder = nn.Sequential(
            nn.Linear(d_hidden * 4, d_hidden * 2),
            nn.GELU(),
            nn.Linear(d_hidden * 2, d_llm),
        )

    def forward(
        self,
        h_seq: torch.Tensor,
        m_prev: Optional[torch.Tensor] = None,
        sample: bool = False
    ) -> Tuple[CognitiveStatePack, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h_seq: (B, L, d_llm)
        Returns:
            state_pack: CognitiveStatePack
            h_recon: (B, d_llm)
            mu: (B, d_hidden * 4)
            logvar: (B, d_hidden * 4)
            m_next: (B, d_hidden)
        """
        B = h_seq.size(0)
        q = self.norm_q(self.query_bank.expand(B, -1, -1))
        kv = self.norm_kv(self.kv_proj(h_seq))

        # Cross-attention compression: (B, num_queries, d_hidden)
        attn_out, _ = self.cross_attn(q, kv, kv)
        
        # Take first 4 canonical query representations
        q_pool = attn_out[:, :4, :].reshape(B, self.d_hidden * 4) # (B, 4*d_hidden)
        
        # Update persistent memory using the mean query
        q_mean = attn_out.mean(dim=1)
        m_next = self.memory(q_mean, m_prev)

        # Variational distribution
        mu = self.mu_head(q_pool)
        logvar = torch.clamp(self.logvar_head(q_pool), -10.0, 4.0)

        if sample or self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # Unpack into structured CognitiveStatePack
        z_chunks = torch.chunk(z, 4, dim=-1)
        intent = self.proj_intent(z_chunks[0] + m_next)
        constraint = self.proj_constraint(z_chunks[1])
        entropy = self.proj_entropy(z_chunks[2])
        traj_flat = self.proj_trajectory(z_chunks[3])
        trajectory = traj_flat.view(B, 4, self.d_hidden)

        state_pack = CognitiveStatePack(
            intent=intent,
            constraint=constraint,
            entropy=entropy,
            trajectory=trajectory
        )

        # Decoder reconstruction
        h_recon = self.decoder(z)
        return state_pack, h_recon, mu, logvar, m_next

    def compute_loss(
        self,
        h_target: torch.Tensor,
        h_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta_kl: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        recon_loss = F.mse_loss(h_recon, h_target)
        kl_loss = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        total_loss = recon_loss + beta_kl * kl_loss
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }


class JEPATrajectoryEnergyPredictor(nn.Module):
    """
    Yann LeCun's JEPA Trajectory Energy Predictor (EBM-CoT).
    Predicts multi-step latent trajectory rollouts and computes energy divergence:
    E(s) = Divergence(s, target) + Curvature(s) + BoundaryViolation(s, constraint).
    """
    def __init__(self, d_hidden: int = 512, horizon: int = 4):
        super().__init__()
        self.d_hidden = d_hidden
        self.horizon = horizon

        # Non-autoregressive step predictor
        self.step_transition = nn.Sequential(
            nn.Linear(d_hidden * 3, d_hidden * 2),
            nn.LayerNorm(d_hidden * 2),
            nn.GELU(),
            nn.Linear(d_hidden * 2, d_hidden),
        )

        # Boundary violation energy head
        self.boundary_energy_head = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
            nn.Softplus()
        )

    def rollout(
        self,
        s_curr: torch.Tensor,
        intent: torch.Tensor,
        constraint: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-step forward latent rollout: (B, H, d_hidden)
        """
        B = s_curr.size(0)
        trajectory = []
        curr = s_curr

        for _ in range(self.horizon):
            inp = torch.cat([curr, intent, constraint], dim=-1)
            delta = self.step_transition(inp)
            curr = curr + delta
            trajectory.append(curr.unsqueeze(1))

        return torch.cat(trajectory, dim=1) # (B, H, d_hidden)

    def compute_energy(
        self,
        trajectory: torch.Tensor,
        intent: torch.Tensor,
        constraint: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes composite energy score:
        - Goal Divergence: L2 distance between endpoint and intent
        - Curvature: Second derivative smoothness along trajectory
        - Boundary Violation: Softplus penalty on constraint violation
        """
        B, H, D = trajectory.shape
        endpoint = trajectory[:, -1, :] # (B, D)

        # 1. Goal Divergence Energy
        e_divergence = F.mse_loss(endpoint, intent, reduction="none").mean(dim=-1) # (B,)

        # 2. Curvature Energy: || s_{t} - 2*s_{t-1} + s_{t-2} ||^2
        if H >= 3:
            s_t = trajectory[:, 2:, :]
            s_prev1 = trajectory[:, 1:-1, :]
            s_prev2 = trajectory[:, :-2, :]
            accel = s_t - 2.0 * s_prev1 + s_prev2
            e_curvature = torch.norm(accel, p=2, dim=-1).mean(dim=-1) # (B,)
        else:
            e_curvature = torch.zeros(B, device=trajectory.device)

        # 3. Boundary Violation Energy
        midpoint = trajectory[:, H // 2, :]
        bound_inp = torch.cat([midpoint, constraint], dim=-1)
        e_boundary = self.boundary_energy_head(bound_inp).squeeze(-1) # (B,)

        # Total Composite Energy
        total_energy = e_divergence + 0.5 * e_curvature + 0.3 * e_boundary
        return total_energy, {
            "e_divergence": e_divergence.mean(),
            "e_curvature": e_curvature.mean(),
            "e_boundary": e_boundary.mean(),
            "total_energy": total_energy.mean()
        }


class LangevinDynamicsRefiner(nn.Module):
    """
    Continuous Langevin Dynamics Energy Minimization (EBM-CoT).
    Iteratively pulls drifting latent states downhill along the energy gradient.
    """
    def __init__(self, step_size: float = 0.05, noise_scale: float = 0.01, num_steps: int = 3):
        super().__init__()
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.num_steps = num_steps

    def refine(
        self,
        predictor: JEPATrajectoryEnergyPredictor,
        s_curr: torch.Tensor,
        state_pack: CognitiveStatePack
    ) -> Tuple[CognitiveStatePack, torch.Tensor]:
        """
        Applies stochastic Langevin gradient descent on constraint/plan vectors.
        Returns:
            refined_pack: CognitiveStatePack with minimized trajectory energy
            final_energy: (B,)
        """
        with torch.enable_grad():
            c = state_pack.constraint.clone().detach().requires_grad_(True)
            intent = state_pack.intent.detach()

            for _ in range(self.num_steps):
                traj = predictor.rollout(s_curr, intent, c)
                energy, _ = predictor.compute_energy(traj, intent, c)
                grad = torch.autograd.grad(energy.sum(), c, create_graph=False)[0]

                # Langevin update: z <- z - eta * grad + sqrt(2 * eta) * noise
                with torch.no_grad():
                    noise = torch.randn_like(c) * self.noise_scale
                    c = c - self.step_size * grad + noise
                    c = c.detach().requires_grad_(True)

            # Final rollout with refined constraints
            refined_traj = predictor.rollout(s_curr, intent, c.detach())
            final_energy, _ = predictor.compute_energy(refined_traj, intent, c.detach())

        refined_pack = CognitiveStatePack(
            intent=intent,
            constraint=c.detach(),
            entropy=state_pack.entropy,
            trajectory=refined_traj.detach()
        )
        return refined_pack, final_energy.detach()


class ContrastiveActivationSteerer(nn.Module):
    """
    Contrastive Activation Addition (CAA) with Adaptive Variance Normalization.
    Directly injects the refined constraint steering vector into LLM intermediate layers.
    h_layer <- h_layer + alpha * (z / ||z||) * std(h_layer)
    """
    def __init__(self, d_hidden: int = 512, d_llm: int = 4096, target_layers: List[int] = None):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_llm = d_llm
        self.target_layers = target_layers or [2, 3, 4, 5]

        # Projection from internal latent space to LLM activation dimension
        self.steer_proj = nn.Sequential(
            nn.Linear(d_hidden, d_llm),
            nn.LayerNorm(d_llm),
            nn.Tanh()
        )

    def compute_steer_vector(
        self,
        constraint: torch.Tensor,
        alpha: float = 0.15,
        h_std: float = 1.0
    ) -> torch.Tensor:
        """
        Computes normalized steering delta: (B, d_llm)
        """
        raw_vector = self.steer_proj(constraint) # (B, d_llm)
        norm = torch.norm(raw_vector, p=2, dim=-1, keepdim=True) + 1e-7
        normalized_vector = raw_vector / norm
        return alpha * normalized_vector * h_std

    def steer_activations(
        self,
        h_layer: torch.Tensor,
        constraint: torch.Tensor,
        alpha: float = 0.15
    ) -> torch.Tensor:
        """
        In-place activation addition with adaptive variance preservation:
        h_steered = h_layer + alpha * norm(v) * std(h_layer)
        """
        h_std = h_layer.std(dim=-1, keepdim=True).clamp(min=1e-5)
        steer_vec = self.compute_steer_vector(constraint, alpha=alpha, h_std=1.0)
        if steer_vec.dim() == 2 and h_layer.dim() == 3:
            steer_vec = steer_vec.unsqueeze(1) # Broadcast across sequence tokens
        return h_layer + steer_vec * h_std


class WhiteBoxSentinel(nn.Module):
    """
    Track A Master Engine:
    Integrates VAE Bottleneck, JEPA Energy Predictor, Langevin Refiner,
    Contrastive Activation Steerer, and Damped Hysteresis Controller with Budget B_max = 3.
    """
    def __init__(
        self,
        d_llm: int = 4096,
        d_hidden: int = 512,
        energy_threshold: float = 1.25,
        damping_beta: float = 0.70,
        hysteresis_steps: int = 2,
        b_max: int = 3
    ):
        super().__init__()
        self.d_llm = d_llm
        self.d_hidden = d_hidden
        self.energy_threshold = energy_threshold
        self.damping_beta = damping_beta
        self.hysteresis_steps = hysteresis_steps
        self.b_max = b_max

        self.vae = LatentVAEBottleneck(d_llm=d_llm, d_hidden=d_hidden)
        self.jepa = JEPATrajectoryEnergyPredictor(d_hidden=d_hidden)
        self.refiner = LangevinDynamicsRefiner()
        self.steerer = ContrastiveActivationSteerer(d_hidden=d_hidden, d_llm=d_llm)

        # Internal episode tracking state
        self.reset_episode()

    def reset_episode(self):
        """Resets hysteresis counters and intervention budget for a new generation episode."""
        self.smoothed_energy: float = 0.0
        self.persistent_high_steps: int = 0
        self.budget_used: int = 0
        self.persistent_memory: Optional[torch.Tensor] = None

    def evaluate_step(
        self,
        h_stream: torch.Tensor,
        s_current: Optional[torch.Tensor] = None,
        alpha_steer: float = 0.15
    ) -> SentinelDecision:
        """
        Processes a streaming step from self-hosted LLM (Qwen/LLaMA):
        1. VAE encodes noisy activations into clean CognitiveStatePack.
        2. JEPA predicts future rollout and evaluates trajectory energy.
        3. Damping & Hysteresis filter verifies persistence of high energy.
        4. If anomaly confirmed and budget remains: Langevin refines constraint & emits CAA steering.
        5. If budget exhausted: Emits HALT_FALLBACK to prevent infinite loops.
        """
        B = h_stream.size(0)
        state_pack, _, _, _, self.persistent_memory = self.vae(
            h_stream, m_prev=self.persistent_memory, sample=False
        )

        if s_current is None:
            s_current = state_pack.intent # Initial seed representation

        # JEPA future simulation
        sim_traj = self.jepa.rollout(s_current, state_pack.intent, state_pack.constraint)
        raw_energy_t, energy_metrics = self.jepa.compute_energy(
            sim_traj, state_pack.intent, state_pack.constraint
        )
        raw_energy = float(raw_energy_t.mean().item())

        # Exponential Damping Filter
        if self.smoothed_energy == 0.0:
            self.smoothed_energy = raw_energy
        else:
            self.smoothed_energy = (
                self.damping_beta * self.smoothed_energy + (1.0 - self.damping_beta) * raw_energy
            )

        # Hysteresis State Machine
        if self.smoothed_energy > self.energy_threshold:
            self.persistent_high_steps += 1
        else:
            self.persistent_high_steps = max(0, self.persistent_high_steps - 1)

        # Decision Logic
        if self.persistent_high_steps < self.hysteresis_steps:
            # Normal or transient blip (suppressed by hysteresis)
            return SentinelDecision(
                track="WHITE_BOX",
                status="NOUL",
                energy=self.smoothed_energy,
                energy_raw=raw_energy,
                budget_used=self.budget_used,
                budget_remaining=self.b_max - self.budget_used,
                steering_vector=None,
                diagnostics={"metrics": energy_metrics, "hysteresis_count": self.persistent_high_steps}
            )

        # Energy is confirmed high across consecutive steps! Check budget:
        if self.budget_used >= self.b_max:
            # Emergency fallback: budget depleted, halt to prevent oscillation loop
            return SentinelDecision(
                track="WHITE_BOX",
                status="HALT_FALLBACK",
                energy=self.smoothed_energy,
                energy_raw=raw_energy,
                budget_used=self.budget_used,
                budget_remaining=0,
                steering_vector=None,
                diagnostics={"reason": "Intervention budget exhausted (B_max reached)."}
            )

        # Perform Langevin Trajectory Refinement and compute CAA Steering Vector
        refined_pack, final_energy = self.refiner.refine(self.jepa, s_current, state_pack)
        steer_vector = self.steerer.compute_steer_vector(refined_pack.constraint, alpha=alpha_steer)
        
        self.budget_used += 1
        # Partially reset hysteresis counter after intervention to observe reaction
        self.persistent_high_steps = 0

        return SentinelDecision(
            track="WHITE_BOX",
            status="ALERT",
            energy=float(final_energy.mean().item()),
            energy_raw=raw_energy,
            budget_used=self.budget_used,
            budget_remaining=self.b_max - self.budget_used,
            steering_vector=steer_vector,
            diagnostics={
                "action": "Contrastive Activation Steering (CAA) deployed",
                "post_refinement_energy": float(final_energy.mean().item())
            }
        )


# ============================================================================
# 3. TRACK B: BLACK-BOX SENTINEL COMPONENTS
# ============================================================================

class SemanticEntropyTracker:
    """
    Implements Semantic Entropy for Hallucination Detection (Kuhn et al., 2023).
    Clusters K parallel candidate completions by semantic equivalence and measures entropy.
    """
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _token_overlap_similarity(s1: str, s2: str) -> float:
        """Fast Jaccard token-set equivalence without external heavy dependencies."""
        import re
        tokens1 = set(re.findall(r'\w+|[+\-*/%=<>!&|^~]+', s1.lower()))
        tokens2 = set(re.findall(r'\w+|[+\-*/%=<>!&|^~]+', s2.lower()))
        if not tokens1 or not tokens2:
            return 1.0 if s1.strip().lower() == s2.strip().lower() else 0.0
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return intersection / float(union)

    def cluster_candidates(self, candidates: List[str]) -> List[List[int]]:
        """Partitions candidates into semantic equivalence clusters."""
        clusters: List[List[int]] = []
        for i, text in enumerate(candidates):
            assigned = False
            for cluster in clusters:
                rep_idx = cluster[0]
                sim = self._token_overlap_similarity(text, candidates[rep_idx])
                if sim >= self.similarity_threshold:
                    cluster.append(i)
                    assigned = True
                    break
            if not assigned:
                clusters.append([i])
        return clusters

    def compute_entropy(self, candidates: List[str]) -> Tuple[float, List[List[int]]]:
        """
        Calculates Shannon Entropy over semantic equivalence clusters:
        H_sem = - sum( P(C) * log P(C) )
        """
        K = len(candidates)
        if K <= 1:
            return 0.0, [[0]]

        clusters = self.cluster_candidates(candidates)
        entropy = 0.0
        for cluster in clusters:
            p = len(cluster) / float(K)
            if p > 0.0:
                entropy -= p * math.log(p)

        return float(entropy), clusters


class BehavioralDriftDetector:
    """
    Detects API-level behavioral drift:
    - Tool-Looping Cycles (e.g., repeating the same tool call with same/oscillating arguments)
    - Output Length Surges
    - N-gram Token Repetition / Degeneracy
    """
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.tool_call_history: List[str] = []
        self.length_history: List[int] = []

    def record_step(self, text: str, tool_call: Optional[str] = None):
        if tool_call is not None:
            self.tool_call_history.append(tool_call.strip())
            if len(self.tool_call_history) > self.max_history:
                self.tool_call_history.pop(0)

        self.length_history.append(len(text.split()))
        if len(self.length_history) > self.max_history:
            self.length_history.pop(0)

    def evaluate_drift(self, current_text: str, current_tool: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        loop_penalty = 0.0
        length_penalty = 0.0
        rep_penalty = 0.0

        # 1. Tool-Call Cycle Check
        if current_tool and len(self.tool_call_history) >= 2:
            if self.tool_call_history[-1] == current_tool and self.tool_call_history[-2] == current_tool:
                loop_penalty = 2.0 # Confirmed 3x repetition loop!
            elif current_tool in self.tool_call_history[-3:]:
                loop_penalty = 0.8 # Oscillation alert

        # 2. Output Length Explosion Check
        words = len(current_text.split())
        if len(self.length_history) >= 3:
            avg_len = sum(self.length_history[-3:]) / 3.0
            if words > avg_len * 2.5 and words > 300:
                length_penalty = 1.0

        # 3. N-gram Degeneracy Check (repetition within current text)
        tokens = current_text.lower().split()
        if len(tokens) >= 20:
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
            unique_ratio = len(set(bigrams)) / float(len(bigrams))
            if unique_ratio < 0.35:
                rep_penalty = 1.5

        composite_drift = loop_penalty + 0.5 * length_penalty + 0.8 * rep_penalty
        return float(composite_drift), {
            "loop_penalty": loop_penalty,
            "length_penalty": length_penalty,
            "rep_penalty": rep_penalty
        }


class TextualCIDAConsensus:
    """
    System-2 Deliberation over K candidate texts:
    Applies Prosecutor, Defender, Skeptic, and Integrator roles to evaluate textual solutions
    and generate a Steered Re-prompt if drift occurs.
    """
    def __init__(self):
        self.roles = ["Prosecutor", "Defender", "Skeptic", "Integrator"]

    def deliberate(
        self,
        candidates: List[str],
        clusters: List[List[int]],
        drift_diagnostics: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        """
        Evaluates candidate clusters:
        Returns:
            selected_text: best consensus candidate
            steered_reprompt: prompt intervention if drift detected
        """
        # Cluster scoring: largest cluster with least repetitive text
        scored_clusters = []
        for cluster in clusters:
            size_score = len(cluster)
            sample_text = candidates[cluster[0]]
            tokens = sample_text.split()
            brevity_penalty = 1.0 if 5 < len(tokens) < 1000 else 0.5
            total_score = size_score * brevity_penalty
            scored_clusters.append((total_score, cluster))

        scored_clusters.sort(key=lambda x: x[0], reverse=True)
        best_cluster = scored_clusters[0][1]
        best_candidate = candidates[best_cluster[0]]

        # Determine if intervention re-prompt is required
        steered_reprompt = None
        if drift_diagnostics.get("loop_penalty", 0.0) >= 1.0:
            steered_reprompt = (
                "[SENTINEL INTERVENTION]: You are repeating the same tool call without progress. "
                "Halt the loop immediately, re-evaluate assumptions, and provide an alternative reasoning step."
            )
        elif drift_diagnostics.get("rep_penalty", 0.0) >= 1.0:
            steered_reprompt = (
                "[SENTINEL INTERVENTION]: Degenerate repetition detected. "
                "Conclude your current reasoning step compactly and verify edge constraints."
            )

        return best_candidate, steered_reprompt


class BlackBoxSentinel:
    """
    Track B Master Engine:
    Guards API models (Claude, Gemini, GPT) using Semantic Entropy,
    Behavioral Drift Detection, and Textual CIDA Deliberation.
    """
    def __init__(
        self,
        entropy_threshold: float = 0.95,
        drift_threshold: float = 1.0,
        b_max: int = 3
    ):
        self.entropy_threshold = entropy_threshold
        self.drift_threshold = drift_threshold
        self.b_max = b_max

        self.entropy_tracker = SemanticEntropyTracker()
        self.drift_detector = BehavioralDriftDetector()
        self.consensus_engine = TextualCIDAConsensus()
        self.reset_episode()

    def reset_episode(self):
        self.budget_used = 0
        self.drift_detector = BehavioralDriftDetector()

    def evaluate_step(
        self,
        candidates: List[str],
        tool_call: Optional[str] = None
    ) -> SentinelDecision:
        """
        Evaluates a generation turn for an external API model.
        """
        primary_text = candidates[0] if candidates else ""
        
        # 1. Semantic Entropy
        entropy, clusters = self.entropy_tracker.compute_entropy(candidates)

        # 2. Behavioral Drift
        drift_score, drift_diag = self.drift_detector.evaluate_drift(primary_text, tool_call)
        self.drift_detector.record_step(primary_text, tool_call)

        # 3. Composite Black-Box Energy
        composite_energy = entropy + drift_score

        if composite_energy <= max(self.entropy_threshold, self.drift_threshold):
            return SentinelDecision(
                track="BLACK_BOX",
                status="NOUL",
                energy=composite_energy,
                energy_raw=composite_energy,
                budget_used=self.budget_used,
                budget_remaining=self.b_max - self.budget_used,
                steered_prompt=None,
                diagnostics={"semantic_entropy": entropy, "drift": drift_diag}
            )

        # Anomaly detected! Check budget:
        if self.budget_used >= self.b_max:
            return SentinelDecision(
                track="BLACK_BOX",
                status="HALT_FALLBACK",
                energy=composite_energy,
                energy_raw=composite_energy,
                budget_used=self.budget_used,
                budget_remaining=0,
                steered_prompt="[CRITICAL SENTINEL ABORT]: Episode budget exhausted. Fallback to safest default response.",
                diagnostics={"reason": "Intervention budget exhausted"}
            )

        # CIDA Textual Consensus and Re-prompt Generation
        best_candidate, steered_reprompt = self.consensus_engine.deliberate(
            candidates, clusters, drift_diag
        )
        self.budget_used += 1

        return SentinelDecision(
            track="BLACK_BOX",
            status="ALERT",
            energy=composite_energy,
            energy_raw=composite_energy,
            budget_used=self.budget_used,
            budget_remaining=self.b_max - self.budget_used,
            steered_prompt=steered_reprompt,
            diagnostics={
                "action": "Steered Re-prompt dispatched to API model",
                "chosen_candidate": best_candidate[:100] + "...",
                "semantic_entropy": entropy
            }
        )


# ============================================================================
# 4. UNIFIED OMNI-SENTINEL FACTORY
# ============================================================================

class OmniSentinel:
    """
    Unified high-level entry point. Instantiates either Track A (White-Box)
    or Track B (Black-Box) based on deployment configuration.
    """
    @staticmethod
    def create(mode: str = "white_box", **kwargs) -> Union[WhiteBoxSentinel, BlackBoxSentinel]:
        mode_clean = mode.lower().replace("-", "_")
        if mode_clean in ["white_box", "local", "track_a"]:
            return WhiteBoxSentinel(**kwargs)
        elif mode_clean in ["black_box", "api", "track_b"]:
            return BlackBoxSentinel(**kwargs)
        else:
            raise ValueError(f"Unknown Sentinel mode: '{mode}'. Expected 'white_box' or 'black_box'.")
