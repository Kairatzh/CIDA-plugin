"""
Тесты для CIDA v3 — Role Priors и CIDALoss.

Покрывает:
    1. Role Priors: гарантированное разногласие
    2. CIDALoss: вычисление лоссов (task, calibration, anti-collapse)
    3. Обучение: градиентный шаг на батче
"""

import pytest
import torch
import torch.nn.functional as F

from cida_plugin import (
    CIDAPlugin,
    CIDAPluginConfig,
    CIDALoss,
    apply_role_prior,
    RoleNames,
)


@pytest.fixture
def device():
    return torch.device("cpu")


def test_role_priors_guaranteed_disagreement(device):
    """
    Проверяет, что применение Role Priors гарантирует ненулевое
    несогласие (disagreement) между агентами даже при одинаковом входе.
    """
    B, M, K = 4, 4, 2
    
    # 1. Без priors (все агенты идентичны)
    b_no_priors = torch.full((B, M, K), 0.5, device=device)
    disagreement_no_priors = b_no_priors.std(dim=1).mean().item()
    assert disagreement_no_priors == 0.0

    # 2. С priors
    b_priors = b_no_priors.clone()
    beliefs_with_priors = []
    for idx in range(M):
        b_i = apply_role_prior(b_priors[:, idx, :], role_idx=idx, multi_label=False)
        beliefs_with_priors.append(b_i)
    b_priors = torch.stack(beliefs_with_priors, dim=1)
    
    disagreement_with_priors = b_priors.std(dim=1).mean().item()
    # Теоретическое расхождение для blend=0.30 и priors=[0.85, 0.15, 0.50, None]
    assert disagreement_with_priors > 0.05
    assert RoleNames.get(0) == "Prosecutor"
    assert RoleNames.get(1) == "Defender"


def test_cida_loss_computation(device):
    """
    Проверяет корректность вычисления лоссов в CIDALoss.
    """
    loss_fn = CIDALoss(
        lambda_cal=0.4,
        lambda_ac=0.2,
        min_disagreement=0.08,
        multi_label=False,
    )
    
    B, M, K = 2, 4, 3
    p_final = F.softmax(torch.randn(B, K, device=device), dim=-1)
    y = torch.tensor([0, 2], device=device)
    
    b_all = [
        torch.softmax(torch.randn(B, M, K, device=device), dim=-1)
        for _ in range(3)
    ]
    
    loss, components = loss_fn(p_final, y, b_all)
    
    assert loss.shape == ()
    assert "task" in components
    assert "calibration" in components
    assert "anti_collapse" in components
    assert "total" in components
    assert components["total"] == loss.item()


def test_cida_loss_multi_label(device):
    """
    Проверяет корректность вычисления лоссов в multi-label режиме.
    """
    loss_fn = CIDALoss(
        lambda_cal=0.5,
        lambda_ac=0.1,
        min_disagreement=0.05,
        multi_label=True,
    )
    
    B, M, K = 2, 4, 5
    p_final = torch.randn(B, K, device=device)  # logits
    y = torch.randint(0, 2, (B, K), dtype=torch.float32, device=device)
    
    b_all = [
        torch.sigmoid(torch.randn(B, M, K, device=device))
        for _ in range(2)
    ]
    
    loss, components = loss_fn(p_final, y, b_all)
    
    assert loss.shape == ()
    assert components["total"] == loss.item()


def test_plug_and_loss_gradient_step(device):
    """
    Проверяет полный тренировочный шаг: forward + loss + backward.
    """
    config = CIDAPluginConfig(d_input=32, d_hidden=32, num_classes=2, max_rounds=2)
    plugin = CIDAPlugin(config).to(device)
    loss_fn = CIDALoss(lambda_cal=0.4, lambda_ac=0.2).to(device)
    
    optimizer = torch.optim.Adam(plugin.parameters(), lr=1e-3)
    
    pooled = torch.randn(4, 32, device=device)
    seq = torch.randn(4, 5, 32, device=device)
    mask = torch.ones(4, 5, device=device)
    y = torch.tensor([1, 0, 1, 0], device=device)
    
    # Forward
    out = plugin(pooled, seq_output=seq, mask=mask)
    
    # Loss
    loss, components = loss_fn(
        out["p_final"],
        y,
        out["b_all"],
        orth_loss=out["L_orth"],
    )
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Проверка градиентов
    # v4: при anonymous_messages=True, input_proj может не получать gradient
    # потому что s не включается в messages → path через message_formulator разрывается.
    # Это корректное поведение v4.
    skip_params = {"halter", "input_proj", "position_bias"}
    for name, param in plugin.named_parameters():
        if param.requires_grad:
            if any(skip in name for skip in skip_params):
                continue
            assert param.grad is not None, f"No gradient for {name}"
            # Проекции входных данных и updater должны обучаться
            if "updater" in name:
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    
    optimizer.step()
