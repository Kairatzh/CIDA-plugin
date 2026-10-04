"""
tests/test_sentinel_mt.py — Comprehensive Unit Tests for Kairos-Sentinel-MT

Validates:
1. Trajectory Graph Aggregator (TGA) multi-turn encoding and cross-attention attribution.
2. KairosSentinelMT forward execution across Risk (2), Taxonomy (8), MCP (5), and Policy (3).
3. O-LoRA task-orthogonality loss between the 4 multi-turn safety heads.
4. Autonomous intervention policy with Epistemic Vacuity Gating.
5. End-to-end multi-turn gradient backpropagation.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cida_plugin import KairosSentinelMT, TrajectoryAggregator
from cida_plugin.losses import CIDALoss


def test_trajectory_aggregator_forward_and_attention():
    """Verify that TrajectoryAggregator fuses multi-turn history and produces valid attention weights."""
    B, T, D = 3, 5, 1024
    aggregator = TrajectoryAggregator(d_input=D, d_hidden=512, num_heads=8, max_turns=16)

    turn_embs = torch.randn(B, T, D)
    # Batch element 0 has 5 turns, element 1 has 3 turns, element 2 has 4 turns
    mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, True, False, False],
        [True, True, True, True, False],
    ])

    fused_state, attn_weights = aggregator(turn_embs, turn_mask=mask)

    assert fused_state.shape == (B, D)
    assert attn_weights.shape == (B, 1, T)
    # Verify attention is non-negative and sums to 1
    assert (attn_weights >= 0.0).all()
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(B, 1), atol=1e-4)


def test_kairos_sentinel_mt_initialization_and_forward():
    """Verify full forward execution of KairosSentinelMT across all 4 multi-turn decoders."""
    B, T, D = 2, 4, 1024
    model = KairosSentinelMT(
        d_input=D,
        d_hidden=512,
        d_message=512,
        d_proj=256,
        num_agents=4,
        max_delib_rounds=3,
        num_taxonomy_classes=8,
        num_mcp_tools=5,
        num_policy_actions=3,
    )

    turn_embs = torch.randn(B, T, D)
    mask = torch.tensor([
        [True, True, True, True],
        [True, True, False, False],
    ])

    out = model(turn_embs, turn_mask=mask, task=None)

    # 1. Output structures exist
    assert "risk" in out
    assert "taxonomy" in out
    assert "mcp" in out
    assert "policy" in out
    assert "intervention" in out
    assert "attn_weights" in out
    assert "orth_loss" in out

    # 2. Shape verifications
    assert out["risk"]["p_calibrated"].shape == (B, 2)
    assert out["taxonomy"]["p_calibrated"].shape == (B, 8)
    assert out["mcp"]["p_calibrated"].shape == (B, 5)
    assert out["policy"]["p_calibrated"].shape == (B, 3)

    # 3. Probability properties
    assert torch.allclose(out["risk"]["p_calibrated"].sum(dim=-1), torch.ones(B), atol=1e-4)
    assert torch.allclose(out["taxonomy"]["p_calibrated"].sum(dim=-1), torch.ones(B), atol=1e-4)
    assert torch.allclose(out["mcp"]["p_calibrated"].sum(dim=-1), torch.ones(B), atol=1e-4)
    assert torch.allclose(out["policy"]["p_calibrated"].sum(dim=-1), torch.ones(B), atol=1e-4)

    # 4. Intervention decision synthesis
    interv = out["intervention"]
    assert interv["p_harm"].shape == (B,)
    assert interv["uncertainty"].shape == (B,)
    assert interv["should_block"].shape == (B,)
    assert len(interv["action_name"]) == B
    assert interv["action_name"][0] in ["ALLOW", "BLOCK", "REQUIRE_CONFIRMATION"]


def test_sentinel_mt_orthogonality_and_backprop():
    """Verify O-LoRA task-orthogonality and end-to-end backpropagation across all 4 heads."""
    B, T, D = 2, 3, 128
    model = KairosSentinelMT(
        d_input=D,
        d_hidden=64,
        d_message=64,
        d_proj=32,
        num_agents=4,
        max_delib_rounds=2,
        num_taxonomy_classes=8,
        num_mcp_tools=5,
        num_policy_actions=3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = CIDALoss(lambda_cal=0.4, label_smoothing=0.1)

    turn_embs = torch.randn(B, T, D)
    mask = torch.tensor([[True, True, True], [True, True, False]])

    y_risk = torch.randint(0, 2, (B,))
    y_tax = torch.randint(0, 8, (B,))
    y_mcp = torch.randint(0, 5, (B,))

    optimizer.zero_grad()
    out = model(turn_embs, turn_mask=mask, task=None)

    # Risk loss
    l_risk, _ = loss_fn(out["risk"]["p_final"], y_risk, out["risk"]["b_all"])
    l_risk_cal = F.nll_loss(torch.log(out["risk"]["p_calibrated"] + 1e-12), y_risk)

    # Taxonomy loss
    l_tax, _ = loss_fn(out["taxonomy"]["p_final"], y_tax, out["taxonomy"]["b_all"])
    l_tax_cal = F.nll_loss(torch.log(out["taxonomy"]["p_calibrated"] + 1e-12), y_tax)

    # MCP tool target loss
    l_mcp, _ = loss_fn(out["mcp"]["p_final"], y_mcp, out["mcp"]["b_all"])
    l_mcp_cal = F.nll_loss(torch.log(out["mcp"]["p_calibrated"] + 1e-12), y_mcp)

    # O-LoRA Orthogonality loss
    l_orth = model.compute_task_orthogonality_loss()
    assert l_orth.dim() == 0
    assert l_orth.item() >= 0.0

    total_loss = (l_risk + l_risk_cal) + 0.5 * (l_tax + l_tax_cal) + 0.5 * (l_mcp + l_mcp_cal) + 0.1 * l_orth
    total_loss.backward()

    # Verify gradients flow into all 4 adapters, GRU aggregator, and attention
    assert model.trajectory_aggregator.history_gru.weight_ih_l0.grad is not None
    assert model.trajectory_aggregator.cross_attn.in_proj_weight.grad is not None
    assert model.adapter_risk.proj.weight.grad is not None
    assert model.adapter_taxonomy.proj.weight.grad is not None
    assert model.adapter_mcp.proj.weight.grad is not None
    assert model.adapter_policy.proj.weight.grad is not None
    assert model.head_risk[0].weight.grad is not None

    optimizer.step()
