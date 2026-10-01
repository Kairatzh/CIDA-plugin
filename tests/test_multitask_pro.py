"""
tests/test_multitask_pro.py — Comprehensive Unit Tests for Kairos 1.1 Pro Architecture

Tests:
1. Model capacity scaling (~9.8M parameters, PTRM sweet spot).
2. O-LoRA task-orthogonal adapter loss and decoupled subspaces.
3. Adaptive Temperature Scaling (ATS) monotonic calibration with epistemic vacuity.
4. Scale-invariant Dirichlet forward pass across 66 tools, 151 intents, and 2 safety classes.
5. End-to-end multi-task backpropagation with combined loss.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from cida_plugin import MultiTaskCIDAPluginPro, OLoRATaskAdapter, AdaptiveTemperatureScaling
from cida_plugin.losses import CIDALoss


def test_multitask_pro_initialization_and_parameters():
    """Verify Kairos 1.1 Pro capacity scaling to ~9.8M parameters with 151 intents and 66 tools."""
    model = MultiTaskCIDAPluginPro(
        d_input=1024,
        d_hidden=512,
        d_message=512,
        d_proj=256,
        num_agents=4,
        max_rounds=3,
        num_tools=66,
        num_intents=151,
        num_safety=2,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Test] MultiTaskCIDAPluginPro Trainable Parameters: {n_params:,}")
    # Verify parameter count is in the 8.5M - 11.5M range
    assert 8_500_000 <= n_params <= 11_500_000


def test_olo_ra_orthogonality_loss():
    """Verify that O-LoRA computes non-negative Frobenius orthogonality loss across task adapters."""
    model = MultiTaskCIDAPluginPro(
        d_input=128,
        d_hidden=64,
        d_message=64,
        d_proj=32,
        num_agents=4,
        num_tools=10,
        num_intents=20,
        num_safety=2,
    )

    loss_orth = model.compute_task_orthogonality_loss()
    assert loss_orth.dim() == 0
    assert loss_orth.item() >= 0.0
    assert loss_orth.requires_grad


def test_adaptive_temperature_scaling():
    """Verify that ATS increases temperature monotonically with higher epistemic uncertainty."""
    ats = AdaptiveTemperatureScaling(d_input=32, base_temp=1.0)
    B = 4
    s = torch.randn(B, 32)
    # Low uncertainty vs high uncertainty
    u_low = torch.tensor([[0.05], [0.10], [0.02], [0.08]])
    u_high = torch.tensor([[1.20], [1.50], [1.80], [1.95]])

    T_low = ats(s, u_low)
    T_high = ats(s, u_high)

    assert T_low.shape == (B, 1)
    assert T_high.shape == (B, 1)
    # With all else equal, higher uncertainty must yield higher temperature
    assert (T_high > T_low).all()
    # Temperatures must be bounded and positive
    assert (T_low >= 0.05).all()


def test_multitask_pro_forward_all_tasks():
    """Verify forward passes for tools (66), intent (151), and safety (2), plus unified inference."""
    B = 3
    d_in = 1024
    model = MultiTaskCIDAPluginPro(
        d_input=d_in,
        d_hidden=512,
        d_message=512,
        d_proj=256,
        num_agents=4,
        num_tools=66,
        num_intents=151,
        num_safety=2,
    )
    x = torch.randn(B, d_in)

    # 1. Tools Head (66 classes)
    out_t = model(x, task="tools")
    assert out_t["p_final"].shape == (B, 66)
    assert out_t["p_calibrated"].shape == (B, 66)
    assert out_t["temperature"].shape == (B,)
    assert torch.allclose(out_t["p_final"].sum(dim=-1), torch.ones(B), atol=1e-4)

    # 2. Intent Head (151 classes)
    out_i = model(x, task="intent")
    assert out_i["p_final"].shape == (B, 151)
    assert out_i["p_calibrated"].shape == (B, 151)
    assert out_i["temperature"].shape == (B,)
    assert torch.allclose(out_i["p_final"].sum(dim=-1), torch.ones(B), atol=1e-4)

    # 3. Safety Head (2 classes)
    out_s = model(x, task="safety")
    assert out_s["p_final"].shape == (B, 2)
    assert out_s["p_calibrated"].shape == (B, 2)
    assert out_s["temperature"].shape == (B,)
    assert torch.allclose(out_s["p_final"].sum(dim=-1), torch.ones(B), atol=1e-4)

    # 4. Unified Multi-Task Forward (all heads simultaneously)
    out_all = model(x, task=None)
    assert "tools" in out_all and out_all["tools"]["p_final"].shape == (B, 66)
    assert "intent" in out_all and out_all["intent"]["p_final"].shape == (B, 151)
    assert "safety" in out_all and out_all["safety"]["p_final"].shape == (B, 2)
    assert "orth_loss" in out_all and out_all["orth_loss"].item() >= 0.0
    assert "task_orth_loss" in out_all and out_all["task_orth_loss"].item() >= 0.0


def test_multitask_pro_end_to_end_backprop():
    """Verify complete gradient backpropagation with CIDA loss and O-LoRA task orthogonality penalty."""
    B = 2
    model = MultiTaskCIDAPluginPro(
        d_input=128,
        d_hidden=64,
        d_message=64,
        d_proj=32,
        num_agents=4,
        num_tools=10,
        num_intents=15,
        num_safety=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = CIDALoss(lambda_cal=0.4, label_smoothing=0.1)

    x = torch.randn(B, 128)
    y_tools = torch.randint(0, 10, (B,))
    y_intent = torch.randint(0, 15, (B,))
    y_safety = torch.randint(0, 2, (B,))

    optimizer.zero_grad()

    # Tools step with both consensus and ATS calibration loss
    out_t = model(x, task="tools")
    l_t, _ = loss_fn(out_t["p_final"], y_tools, out_t["b_all"])
    l_t_cal = F.nll_loss(torch.log(out_t["p_calibrated"] + 1e-12), y_tools)

    # Intent step with both consensus and ATS calibration loss
    out_i = model(x, task="intent")
    l_i, _ = loss_fn(out_i["p_final"], y_intent, out_i["b_all"])
    l_i_cal = F.nll_loss(torch.log(out_i["p_calibrated"] + 1e-12), y_intent)

    # Safety step
    out_s = model(x, task="safety")
    l_s, _ = loss_fn(out_s["p_final"], y_safety, out_s["b_all"])
    l_s_cal = F.nll_loss(torch.log(out_s["p_calibrated"] + 1e-12), y_safety)

    # Orthogonality loss
    l_orth = model.compute_task_orthogonality_loss()

    # Total combined multi-task loss
    total_loss = (l_t + l_t_cal) + (l_i + l_i_cal) + 0.5 * (l_s + l_s_cal) + 0.1 * l_orth
    total_loss.backward()

    # Verify gradients exist in all adapters and heads
    assert model.adapter_tools.proj.weight.grad is not None
    assert model.adapter_intent.proj.weight.grad is not None
    assert model.adapter_safety.proj.weight.grad is not None
    assert model.head_tools[0].weight.grad is not None
    assert model.head_intent[0].weight.grad is not None
    assert model.head_safety[0].weight.grad is not None
    assert model.ats_tools.base_temp.grad is not None
    assert model.ats_intent.base_temp.grad is not None

    optimizer.step()
