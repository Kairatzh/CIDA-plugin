"""
tests/test_multitask_cida.py — Unit Tests for MultiTaskCIDAPlugin (MT-Kairos)
"""

import pytest
import torch
from cida_plugin.multitask import MultiTaskCIDAPlugin
from cida_plugin.losses import CIDALoss


def test_multitask_cida_parameter_count_and_shapes():
    """Verify that MT-Kairos scales to ~8.2M parameters and has correct output shapes."""
    model = MultiTaskCIDAPlugin(
        d_input=1024,
        d_hidden=512,
        d_message=512,
        num_agents=4,
        max_rounds=3,
        num_tools=65,
        num_intents=77,
        num_safety=2,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Test] MultiTaskCIDAPlugin total trainable parameters: {n_params:,}")
    # Verify parameter count is in the 7.5M - 9.0M range (PTRM sweet spot)
    assert 7_000_000 <= n_params <= 9_500_000

    B = 4
    x = torch.randn(B, 1024)

    # 1. Test Tools Task
    out_tools = model(pooled=x, task="tools")
    assert out_tools["p_final"].shape == (B, 65)
    assert torch.allclose(out_tools["p_final"].sum(dim=-1), torch.ones(B), atol=1e-4)

    # 2. Test Intent Task
    out_intent = model(pooled=x, task="intent")
    assert out_intent["p_final"].shape == (B, 77)
    assert torch.allclose(out_intent["p_final"].sum(dim=-1), torch.ones(B), atol=1e-4)

    # 3. Test Safety Task
    out_safety = model(pooled=x, task="safety")
    assert out_safety["p_final"].shape == (B, 2)
    assert torch.allclose(out_safety["p_final"].sum(dim=-1), torch.ones(B), atol=1e-4)

    # 4. Test Unified Joint Execution
    out_all = model(pooled=x, task=None)
    assert "tools" in out_all and out_all["tools"]["p_final"].shape == (B, 65)
    assert "intent" in out_all and out_all["intent"]["p_final"].shape == (B, 77)
    assert "safety" in out_all and out_all["safety"]["p_final"].shape == (B, 2)
    assert out_all["orth_loss"].item() >= 0.0


def test_multitask_cida_gradients():
    """Verify end-to-end backpropagation and gradient flow through all heads."""
    model = MultiTaskCIDAPlugin(
        d_input=128,
        d_hidden=64,
        d_message=64,
        num_agents=4,
        num_tools=10,
        num_intents=12,
        num_safety=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = CIDALoss(lambda_cal=0.4, label_smoothing=0.1)

    B = 2
    x = torch.randn(B, 128)

    # Gradient step on tools
    optimizer.zero_grad()
    y_tools = torch.randint(0, 10, (B,))
    out_t = model(pooled=x, task="tools")
    loss_t, _ = loss_fn(out_t["p_final"], y_tools, out_t["b_all"])
    loss_t.backward()
    assert model.head_tools[0].weight.grad is not None
    if model.deliberation_core.perspective_projector is not None:
        assert model.deliberation_core.perspective_projector.agent_projections[0].weight.grad is not None
    else:
        assert model.deliberation_core.input_proj[0].weight.grad is not None
    optimizer.step()

    # Gradient step on safety
    optimizer.zero_grad()
    y_safety = torch.randint(0, 2, (B,))
    out_s = model(pooled=x, task="safety")
    loss_s, _ = loss_fn(out_s["p_final"], y_safety, out_s["b_all"])
    loss_s.backward()
    assert model.head_safety[0].weight.grad is not None
    optimizer.step()
