"""
test_cida_v4_pipeline.py — Comprehensive Unit & Integration Tests for CIDA v4
"""

import os
import shutil
import tempfile
import torch
import pytest

from cida_v4_pipeline.core_v4 import (
    CIDAv4Config,
    CIDAv4Core,
    PerceiverContextBottleneck,
    BayesianRoleDebate,
    UnrolledRecurrentLoop,
    VariationalPlanProjector
)
from cida_v4_pipeline.kv_steerer import PrefixKVSteerer
from cida_v4_pipeline.dataset_curator import DatasetCurator
from cida_v4_pipeline.train_stage2_grpo import MathRuleVerifier, CodeUnitTester, verify_answer


def test_cida_v4_config_serde():
    cfg = CIDAv4Config(d_llm=2048, d_hidden=256, num_plan_tokens=6)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        cfg.save(path)
        loaded = CIDAv4Config.load(path)
        assert loaded.d_llm == 2048
        assert loaded.d_hidden == 256
        assert loaded.num_plan_tokens == 6


def test_perceiver_bottleneck_shapes():
    B, L, D_llm = 2, 32, 256
    d_hidden = 64
    bottleneck = PerceiverContextBottleneck(d_llm=D_llm, d_hidden=d_hidden, num_queries=8, num_heads=4)
    h_seq = torch.randn(B, L, D_llm)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, 20:] = False # simulate padding
    
    out = bottleneck(h_seq, mask=mask)
    assert out.shape == (B, 8, d_hidden)


def test_bayesian_role_debate():
    B, M, D = 2, 4, 64
    debate = BayesianRoleDebate(d_hidden=D, num_agents=M, sparse_k=2)
    states = torch.randn(B, M, D)
    
    msgs, beliefs, disag = debate(states)
    assert msgs.shape == (B, M, D)
    assert beliefs.shape == (B, M, 1)
    assert disag.shape == (B,)
    # Verify beliefs are bounded in (0, 1)
    assert (beliefs >= 0.0).all() and (beliefs <= 1.0).all()


def test_unrolled_recurrent_loop():
    cfg = CIDAv4Config(d_llm=128, d_hidden=64, max_rounds=3, halting_threshold=0.01)
    loop = UnrolledRecurrentLoop(cfg)
    debate = BayesianRoleDebate(d_hidden=64, num_agents=4)
    
    B = 2
    init_states = torch.randn(B, 4, 64)
    bottleneck_tokens = torch.randn(B, 8, 64)
    
    out = loop(init_states, bottleneck_tokens, debate)
    assert out["final_states"].shape == (B, 4, 64)
    assert out["p_consensus"].shape == (B, 1)
    assert out["uncertainty"].shape == (B,)
    assert out["rounds_executed"] >= 1


def test_variational_plan_projector():
    d_hidden, d_llm, P = 64, 128, 4
    projector = VariationalPlanProjector(d_hidden=d_hidden, d_llm=d_llm, num_agents=4, num_plan_tokens=P)
    
    B = 3
    final_states = torch.randn(B, 4, d_hidden)
    evidence = torch.randn(B, 4, d_hidden)
    
    # 1. Deterministic inference
    z_det, mu, logvar = projector(final_states, evidence, sample=False)
    assert z_det.shape == (B, P, d_llm)
    assert torch.allclose(z_det, mu)
    
    # 2. Stochastic sampling
    z_sample, mu, logvar = projector(final_states, evidence, sample=True)
    assert z_sample.shape == (B, P, d_llm)
    
    # 3. KL divergence
    kl = projector.kl_divergence(mu, logvar)
    assert kl.dim() == 0 and not torch.isnan(kl)


def test_cida_v4_core_end_to_end():
    cfg = CIDAv4Config(d_llm=128, d_hidden=64, num_bottleneck_queries=8, num_plan_tokens=4, max_rounds=2)
    core = CIDAv4Core(cfg)
    
    B, L = 2, 16
    h_seq = torch.randn(B, L, 128)
    
    core.eval()
    out = core(h_seq)
    assert out["plan_tokens"].shape == (B, 4, 128)
    assert out["p_consensus"].shape == (B, 1)
    assert out["uncertainty"].shape == (B,)
    assert out["disagreement"].shape == (B,)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        core.save_pretrained(tmpdir)
        loaded = CIDAv4Core.from_pretrained(tmpdir)
        loaded.eval()
        out_loaded = loaded(h_seq)
        assert torch.allclose(out["plan_tokens"], out_loaded["plan_tokens"], atol=1e-5)


def test_prefix_kv_steerer():
    d_llm = 128
    num_layers = 4
    num_heads = 4
    head_dim = 32
    
    steerer = PrefixKVSteerer(
        d_llm=d_llm,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        target_layer_indices=[1, 2],
        num_plan_tokens=4
    )
    
    B = 2
    plan_tokens = torch.randn(B, 4, d_llm)
    kv_dict = steerer(plan_tokens)
    
    assert 1 in kv_dict and 2 in kv_dict
    k, v = kv_dict[1]
    assert k.shape == (B, num_heads, 4, head_dim)
    assert v.shape == (B, num_heads, 4, head_dim)
    
    # Test fallback embedding prepending
    prompt_embeds = torch.randn(B, 10, d_llm)
    mask = torch.ones(B, 10, dtype=torch.long)
    comb_emb, comb_mask = steerer.inject_inputs_embeds_fallback(prompt_embeds, plan_tokens, mask)
    assert comb_emb.shape == (B, 14, d_llm)
    assert comb_mask.shape == (B, 14)


def test_dataset_curator_seed_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        curator = DatasetCurator(output_dir=tmpdir)
        path = curator.generate_local_seed_data(num_samples=10)
        assert os.path.exists(path)
        assert curator.extract_boxed_math_answer("Therefore, x = \\boxed{42}.") == "42"
        assert curator.extract_boxed_math_answer("The answer is 100") == "100"


def test_rule_based_verifiers():
    # Math verification
    assert MathRuleVerifier.verify("The result is \\boxed{28}.", "28") == 1.0
    assert MathRuleVerifier.verify("So the answer is 42.", "42") == 1.0
    assert MathRuleVerifier.verify("I don't know the answer.", "42") == 0.0
    
    # Code verification
    code_pass = "def add(a, b):\n    return a + b"
    test_cases = [{"in": "add(2, 3)", "out": "5"}, {"in": "add(-1, 1)", "out": "0"}]
    assert CodeUnitTester.verify(code_pass, test_cases) == 1.0
    
    code_fail = "def add(a, b):\n    return a - b"
    assert CodeUnitTester.verify(code_fail, test_cases) == 0.0
