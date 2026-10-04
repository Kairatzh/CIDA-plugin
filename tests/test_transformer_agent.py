"""
test_transformer_agent.py — Rigorous Verification of SOTA Transformer Deliberation Core
"""

import pytest
import torch
import torch.nn.functional as F

from cida_plugin.transformer_agent import (
    RMSNorm,
    SwiGLUFFN,
    DeliberationRoPE,
    SOTATransformerAgentBlock,
    SOTADeliberationCore,
    UniversalPlanProjector,
)


def test_rmsnorm_invariance_and_shape():
    norm = RMSNorm(d_model=512)
    x = torch.randn(2, 4, 512)
    out = norm(x)
    assert out.shape == (2, 4, 512)
    # Check that output RMS is approximately 1.0 (since weights are 1.0)
    rms = torch.sqrt(out.pow(2).mean(-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_swiglu_ffn_dimensions():
    ffn = SwiGLUFFN(d_model=512, d_ffn=1024)
    x = torch.randn(4, 8, 512)
    out = ffn(x)
    assert out.shape == (4, 8, 512)


def test_deliberation_rope_rotation():
    rope = DeliberationRoPE(d_head=64)
    x = torch.randn(2, 4, 4, 64)
    out_round0 = rope(x, round_idx=0)
    out_round3 = rope(x, round_idx=3)
    
    # Round 0 cos=1, sin=0, so out_round0 should equal x
    assert torch.allclose(out_round0, x, atol=1e-4)
    # Round 3 should be rotated and different from Round 0
    assert not torch.allclose(out_round0, out_round3, atol=1e-2)
    # Norm should be preserved under rotation
    assert torch.allclose(out_round0.norm(dim=-1), out_round3.norm(dim=-1), atol=1e-3)


def test_sota_transformer_agent_block():
    block = SOTATransformerAgentBlock(d_model=256, num_heads=4)
    s = torch.randn(2, 4, 256)
    # Custom disagreement mask
    d_mask = torch.randn(2, 4, 4)
    out = block(s, round_idx=1, disagreement_mask=d_mask)
    assert out.shape == (2, 4, 256)


def test_sota_deliberation_core_full_forward():
    d_input = 4096
    d_hidden = 256
    core = SOTADeliberationCore(d_input=d_input, d_hidden=d_hidden, num_agents=4, num_rounds=3)
    
    h_last = torch.randn(2, d_input)
    res = core(h_last, min_rounds=2, max_rounds=3)
    
    assert "s_final" in res
    assert "p_consensus" in res
    assert "disagreement_trajectory" in res
    assert "orth_loss" in res
    
    assert res["s_final"].shape == (2, 4, d_hidden)
    assert res["p_consensus"].shape == (2, 2)
    assert len(res["disagreement_trajectory"]) >= 2
    assert res["orth_loss"].item() >= 0.0


def test_universal_plan_projector():
    d_hidden = 256
    d_llm = 4096
    projector = UniversalPlanProjector(d_hidden=d_hidden, d_llm=d_llm, num_agents=4, num_plan_tokens=8)
    
    s_final = torch.randn(2, 4, d_hidden)
    h_base = torch.randn(2, d_llm)
    
    plan = projector(s_final, h_base)
    assert plan.shape == (2, 8, d_llm)
