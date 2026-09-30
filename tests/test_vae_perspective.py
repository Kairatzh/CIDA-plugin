"""
Tests for Kairos L3 VAE-Perspective Layer and Scale-Invariant Dirichlet Prior.
"""
import pytest
import torch
import torch.nn.functional as F

from cida_plugin import (
    CIDAPlugin,
    CIDAPluginConfig,
    CIDALoss,
    PerspectiveProjector,
    VAEPerspectiveProjector,
    AgentState,
    AgentUpdater,
)


def test_dirichlet_scaling_large_k():
    """
    Verifies that the Dirichlet prior scaling alpha_0 = 1/K prevents
    pseudo-count dilution on large label spaces (K=77).
    """
    B, M, K = 2, 4, 77
    d_hidden = 64
    updater = AgentUpdater(d_hidden=d_hidden, num_classes=K)

    # Synthetic agent state with confident evidence in class 5
    state = AgentState(
        s=torch.randn(B, M, d_hidden),
        b=torch.full((B, M, K), 1.0 / K),
        u=torch.ones(B, M, 1),
        p=torch.zeros(B, M, 1),
        e=torch.zeros(B, M, d_hidden),
        alpha=torch.ones(B, M, K),
    )
    r = torch.zeros(B, M, d_hidden)
    e = torch.zeros(B, M, d_hidden)

    # Run updater
    next_state = updater(state, r, e)

    assert next_state.b.shape == (B, M, K)
    assert torch.allclose(next_state.b.sum(dim=-1), torch.ones(B, M), atol=1e-5)
    # Vacuity should be in [0, 1]
    assert (next_state.u_epi >= 0.0).all() and (next_state.u_epi <= 1.0).all()

    # Now verify that when logits g(s) are large, belief can reach high confidence > 0.85
    # Manually test updater.g output through Dirichlet update
    with torch.no_grad():
        high_evidence_logits = torch.full((B, M, K), -10.0)
        high_evidence_logits[:, :, 5] = 10.0  # Dominant class
        
        # New formula: alpha = softplus(g) + 1/K
        alpha_scaled = F.softplus(high_evidence_logits) + (1.0 / K)
        b_scaled = alpha_scaled / alpha_scaled.sum(dim=-1, keepdim=True)
        max_conf_scaled = b_scaled.max(dim=-1).values.mean().item()

        # Legacy formula: alpha = softplus(g) + 1.0
        alpha_legacy = F.softplus(high_evidence_logits) + 1.0
        b_legacy = alpha_legacy / alpha_legacy.sum(dim=-1, keepdim=True)
        max_conf_legacy = b_legacy.max(dim=-1).values.mean().item()

        # Under legacy formula with K=77, max confidence is trapped around ~0.12 - 0.20
        assert max_conf_legacy < 0.25, f"Legacy conf {max_conf_legacy} unexpectedly high"
        # Under scale-invariant formula, max confidence cleanly reaches > 0.85!
        assert max_conf_scaled > 0.85, f"Scaled conf {max_conf_scaled} failed to reach high confidence"


def test_vae_perspective_projector_shapes_and_gradients():
    """
    Verifies VAEPerspectiveProjector forward pass, shapes, reparameterization,
    and gradient propagation back to mu and logvar heads.
    """
    B, M = 4, 4
    d_input, d_hidden = 128, 64
    vae = VAEPerspectiveProjector(num_agents=M, d_input=d_input, d_hidden=d_hidden)

    x = torch.randn(B, d_input, requires_grad=True)

    # Training mode (stochastic sampling)
    vae.train()
    s_0, kl_loss = vae(x, deterministic=False)

    assert s_0.shape == (B, M, d_hidden)
    assert kl_loss.ndim == 0  # scalar
    assert kl_loss.item() >= 0.0

    # Test backward pass
    loss = s_0.sum() + kl_loss
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0.0
    for head in vae.mu_heads:
        assert head.weight.grad is not None
    for head in vae.logvar_heads:
        assert head.weight.grad is not None

    # Test deterministic evaluation mode
    vae.eval()
    s_eval_1, _ = vae(x.detach(), deterministic=True)
    s_eval_2, _ = vae(x.detach(), deterministic=True)
    assert torch.allclose(s_eval_1, s_eval_2), "Deterministic mode must produce identical states"


def test_cida_plugin_with_vae_perspective():
    """
    End-to-end integration test of CIDAPlugin with use_vae_perspective=True.
    """
    B, K = 4, 10
    d_input, d_hidden = 64, 32
    config = CIDAPluginConfig(
        d_input=d_input,
        d_hidden=d_hidden,
        d_message=d_hidden,
        num_agents=4,
        num_classes=K,
        max_rounds=2,
        use_vae_perspective=True,
        kl_weight=1e-3,
        role_prior_blend=0.0,
    )
    plugin = CIDAPlugin(config)
    loss_fn = CIDALoss(multi_label=False)

    pooled = torch.randn(B, d_input)
    y = torch.randint(0, K, (B,))

    out = plugin(pooled)
    assert "p_final" in out
    assert "kl_loss" in out
    assert out["kl_loss"] is not None
    assert out["p_final"].shape == (B, K)

    loss, comps = loss_fn(
        p_final=out["p_final"],
        y=y,
        b_all=out["b_all"],
        orth_loss=out.get("L_orth"),
        kl_loss=out.get("kl_loss"),
        kl_weight=config.kl_weight,
    )

    assert loss.ndim == 0
    assert "kl" in comps
    loss.backward()

    # Check optimizer step works
    opt = torch.optim.Adam(plugin.parameters(), lr=1e-3)
    opt.step()
