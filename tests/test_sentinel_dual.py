"""
test_sentinel_dual.py — Comprehensive Unit & Integration Test Suite for Sentinel-Dual

Validates:
1. LatentVAEBottleneck & AGCLR Gated Persistent Memory
2. JEPATrajectoryEnergyPredictor & Curvature/Divergence Scoring
3. LangevinDynamicsRefiner Energy Minimization
4. ContrastiveActivationSteerer (CAA) with Adaptive Variance Normalization
5. WhiteBoxSentinel Hysteresis Damping & Anti-Loop Budget (B_max = 3)
6. BlackBoxSentinel Kuhn Semantic Entropy & Tool-Looping Detection
7. Full 4-Stage Training Cycle Integration
"""

import os
import tempfile
import torch
import pytest

from cida_v4_pipeline.sentinel_dual import (
    CognitiveStatePack,
    GatedPersistentMemory,
    LatentVAEBottleneck,
    JEPATrajectoryEnergyPredictor,
    LangevinDynamicsRefiner,
    ContrastiveActivationSteerer,
    WhiteBoxSentinel,
    SemanticEntropyTracker,
    BehavioralDriftDetector,
    TextualCIDAConsensus,
    BlackBoxSentinel,
    OmniSentinel
)
from cida_v4_pipeline.train_full_cycle import (
    ContrastiveLatentDataset,
    FullCycleSentinelTrainer
)


def test_gated_persistent_memory():
    B, D = 2, 128
    mem = GatedPersistentMemory(d_hidden=D)
    s_t = torch.randn(B, D)
    m0 = mem(s_t, m_prev=None)
    assert m0.shape == (B, D)

    # Next step updates memory smoothly
    s_t1 = torch.randn(B, D)
    m1 = mem(s_t1, m_prev=m0)
    assert m1.shape == (B, D)
    # Norm should be finite
    assert torch.isfinite(m1).all()


def test_latent_vae_bottleneck_and_state_pack():
    B, L, D_llm, D_hidden = 2, 16, 256, 64
    vae = LatentVAEBottleneck(d_llm=D_llm, d_hidden=D_hidden, num_queries=8)
    h_seq = torch.randn(B, L, D_llm)

    state_pack, h_recon, mu, logvar, m_next = vae(h_seq, sample=True)

    assert isinstance(state_pack, CognitiveStatePack)
    assert state_pack.intent.shape == (B, D_hidden)
    assert state_pack.constraint.shape == (B, D_hidden)
    assert state_pack.entropy.shape == (B, D_hidden)
    assert state_pack.trajectory.shape == (B, 4, D_hidden)
    assert h_recon.shape == (B, D_llm)

    # Test loss computation
    loss_dict = vae.compute_loss(h_seq.mean(dim=1), h_recon, mu, logvar)
    assert "loss" in loss_dict and "recon_loss" in loss_dict and "kl_loss" in loss_dict
    assert loss_dict["loss"].item() > 0


def test_jepa_trajectory_energy_predictor():
    B, D_hidden, H = 2, 64, 4
    jepa = JEPATrajectoryEnergyPredictor(d_hidden=D_hidden, horizon=H)

    s_curr = torch.randn(B, D_hidden)
    intent = torch.randn(B, D_hidden)
    constraint = torch.randn(B, D_hidden)

    traj = jepa.rollout(s_curr, intent, constraint)
    assert traj.shape == (B, H, D_hidden)

    energy, metrics = jepa.compute_energy(traj, intent, constraint)
    assert energy.shape == (B,)
    assert "e_divergence" in metrics
    assert "e_curvature" in metrics
    assert "e_boundary" in metrics
    assert (energy >= 0).all()


def test_langevin_dynamics_refiner():
    B, D_hidden = 2, 64
    jepa = JEPATrajectoryEnergyPredictor(d_hidden=D_hidden, horizon=4)
    refiner = LangevinDynamicsRefiner(step_size=0.1, num_steps=2)

    s_curr = torch.randn(B, D_hidden)
    intent = torch.randn(B, D_hidden)
    constraint = torch.randn(B, D_hidden)
    pack = CognitiveStatePack(
        intent=intent,
        constraint=constraint,
        entropy=torch.zeros(B, D_hidden),
        trajectory=jepa.rollout(s_curr, intent, constraint)
    )

    refined_pack, final_energy = refiner.refine(jepa, s_curr, pack)
    assert refined_pack.constraint.shape == (B, D_hidden)
    assert final_energy.shape == (B,)


def test_contrastive_activation_steerer():
    B, L, D_llm, D_hidden = 2, 8, 256, 64
    steerer = ContrastiveActivationSteerer(d_hidden=D_hidden, d_llm=D_llm)

    constraint = torch.randn(B, D_hidden)
    steer_vec = steerer.compute_steer_vector(constraint, alpha=0.15)
    assert steer_vec.shape == (B, D_llm)

    # In-place activation addition test
    h_layer = torch.randn(B, L, D_llm)
    h_steered = steerer.steer_activations(h_layer, constraint, alpha=0.15)
    assert h_steered.shape == h_layer.shape
    # Layer variance preserved
    assert torch.isfinite(h_steered).all()


def test_white_box_sentinel_hysteresis_and_budget():
    B, L, D_llm, D_hidden = 2, 8, 256, 64
    sentinel = WhiteBoxSentinel(
        d_llm=D_llm,
        d_hidden=D_hidden,
        energy_threshold=0.5, # low threshold to trigger alerts reliably
        damping_beta=0.5,
        hysteresis_steps=2,
        b_max=3
    )

    h_stream = torch.randn(B, L, D_llm)

    # Step 1: High energy detected once -> hysteresis suppresses it -> NOUL
    dec1 = sentinel.evaluate_step(h_stream)
    assert dec1.status == "NOUL"
    assert dec1.budget_used == 0

    # Step 2: High energy persists -> triggers ALERT with CAA vector
    dec2 = sentinel.evaluate_step(h_stream)
    assert dec2.status == "ALERT"
    assert dec2.budget_used == 1
    assert dec2.steering_vector is not None

    # Exhaust remaining budget
    sentinel.evaluate_step(h_stream) # step 3: count 1
    sentinel.evaluate_step(h_stream) # step 4: count 2 -> ALERT, budget 2
    sentinel.evaluate_step(h_stream) # step 5: count 1
    sentinel.evaluate_step(h_stream) # step 6: count 2 -> ALERT, budget 3

    # Now budget is exhausted -> HALT_FALLBACK
    sentinel.evaluate_step(h_stream) # count 1
    dec_halt = sentinel.evaluate_step(h_stream) # count 2 -> HALT_FALLBACK
    assert dec_halt.status == "HALT_FALLBACK"
    assert dec_halt.budget_remaining == 0


def test_black_box_semantic_entropy_and_tool_loops():
    tracker = SemanticEntropyTracker()
    drift = BehavioralDriftDetector()

    # 1. Consensus samples -> Zero / Low entropy
    candidates_agree = [
        "return a + b",
        "return a + b",
        "return (a + b)"
    ]
    entropy_low, _ = tracker.compute_entropy(candidates_agree)
    assert entropy_low == 0.0

    # 2. Conflicting samples -> High entropy
    candidates_disagree = [
        "return a + b",
        "return a * b",
        "raise ValueError('invalid')"
    ]
    entropy_high, _ = tracker.compute_entropy(candidates_disagree)
    assert entropy_high > 0.5

    # 3. Tool Loop Detector
    drift.record_step("trying grep", tool_call="grep_tool")
    drift.record_step("trying grep again", tool_call="grep_tool")
    score, diag = drift.evaluate_drift("trying grep third time", current_tool="grep_tool")
    assert diag["loop_penalty"] >= 2.0
    assert score >= 2.0


def test_omni_sentinel_factory():
    wb = OmniSentinel.create("white_box", d_llm=128, d_hidden=32)
    assert isinstance(wb, WhiteBoxSentinel)

    bb = OmniSentinel.create("black_box")
    assert isinstance(bb, BlackBoxSentinel)


def test_full_cycle_smoke_training():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = ContrastiveLatentDataset.generate_contrastive_pairs(
            num_samples=8,
            d_llm=128,
            d_hidden=32,
            seq_len=6,
            horizon=3
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        trainer = FullCycleSentinelTrainer(
            d_llm=128,
            d_hidden=32,
            horizon=3,
            device="cpu",
            save_dir=tmpdir
        )

        res_s1 = trainer.train_stage1_vae(loader, epochs=1)
        assert "final_loss" in res_s1

        res_s2 = trainer.train_stage2_jepa(loader, epochs=1)
        assert "final_loss" in res_s2

        res_s3 = trainer.train_stage3_grpo(loader, epochs=1)
        assert "mean_reward" in res_s3

        res_s4 = trainer.run_stage4_evaluation(loader)
        assert "ablation" in res_s4
        assert os.path.exists(os.path.join(tmpdir, "sentinel_dual_master.pt"))
