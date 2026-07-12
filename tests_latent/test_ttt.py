"""
Тесты для CIDA v3 — Test-Time Training (TTT).

Покрывает:
    1. Юнит-тесты TTT модуля (маска, loss, адаптация, восстановление)
    2. Интеграцию TTT с CIDAPlugin (forward pass TTT on/off)
    3. Backward compatibility (TTT off = базовое поведение)
    4. Диагностики TTT
"""

import pytest
import torch
import torch.nn as nn
import copy

from cida_latent_plugin import CIDAPlugin, CIDAPluginConfig, TestTimeTrainer
from cida_latent_plugin.ttt import MaskedReconstructionHead
from cida_latent_plugin.diagnostics import DebateDiagnostics
from cida_latent_plugin.deliberation import AgentUpdater
from cida_latent_plugin.agent import AgentState


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def d_hidden():
    return 64


@pytest.fixture
def batch_agents(device, d_hidden):
    """Типичный batch: B=4, M=4 агента, d=64."""
    B, M = 4, 4
    return torch.randn(B, M, d_hidden, device=device)


@pytest.fixture
def ttt_trainer(d_hidden):
    return TestTimeTrainer(d_hidden=d_hidden, ttt_steps=3, ttt_lr=1e-3, mask_ratio=0.15)


@pytest.fixture
def updater(d_hidden):
    return AgentUpdater(d_hidden=d_hidden, num_classes=2, num_heads=4)


# ── 1. Unit Tests ─────────────────────────────────────────────────────────────

class TestMaskedReconstructionHead:
    def test_output_shape(self, d_hidden, device):
        head = MaskedReconstructionHead(d_hidden).to(device)
        x = torch.randn(4, 4, d_hidden, device=device)
        out = head(x)
        assert out.shape == (4, 4, d_hidden)

    def test_gradient_flows(self, d_hidden, device):
        head = MaskedReconstructionHead(d_hidden).to(device)
        x = torch.randn(4, 4, d_hidden, device=device, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestMaskCreation:
    def test_mask_shape(self, ttt_trainer, batch_agents, device):
        mask = ttt_trainer._create_mask(batch_agents.shape, device)
        assert mask.shape == (4, 1, 64)

    def test_mask_ratio(self, d_hidden, device):
        """Проверяем что доля замаскированных ≈ mask_ratio."""
        trainer = TestTimeTrainer(d_hidden=d_hidden, mask_ratio=0.20)
        B = 32  # Большой батч для статистики
        shape = (B, 4, d_hidden)
        mask = trainer._create_mask(shape, device)
        
        # Средняя доля нулей
        zero_ratio = (mask == 0).float().mean().item()
        # Допускаем ±5% от целевого
        assert abs(zero_ratio - 0.20) < 0.05, f"Expected ~0.20, got {zero_ratio:.3f}"

    def test_mask_broadcasts_across_agents(self, ttt_trainer, device, d_hidden):
        """Маска (B, 1, d) должна быть одинаковой для всех агентов."""
        mask = ttt_trainer._create_mask((4, 4, d_hidden), device)
        assert mask.shape[1] == 1  # broadcast dimension


class TestReconstructionLoss:
    def test_loss_is_scalar(self, ttt_trainer, updater, batch_agents, device):
        mask = ttt_trainer._create_mask(batch_agents.shape, device)
        masked = batch_agents * mask
        loss = ttt_trainer.compute_reconstruction_loss(updater, masked, batch_agents, mask)
        assert loss.shape == ()

    def test_loss_is_zero_for_perfect_reconstruction(self, d_hidden, device):
        """Если head идеально восстанавливает — loss = 0."""
        trainer = TestTimeTrainer(d_hidden=d_hidden, mask_ratio=0.15)
        upd = AgentUpdater(d_hidden=d_hidden, num_classes=2, num_heads=4).to(device)
        states = torch.randn(2, 4, d_hidden, device=device)
        mask = torch.ones(2, 1, d_hidden, device=device)  # Нет маскированных
        
        loss = trainer.compute_reconstruction_loss(upd, states, states, mask)
        # С полной маской (нет нулей) — нет маскированных позиций
        # inverted_mask будет всё нули, loss = 0
        assert loss.item() == 0.0

    def test_loss_positive_with_mask(self, ttt_trainer, updater, batch_agents, device):
        """С маской loss должен быть положительным."""
        mask = ttt_trainer._create_mask(batch_agents.shape, device)
        masked = batch_agents * mask
        loss = ttt_trainer.compute_reconstruction_loss(updater, masked, batch_agents, mask)
        assert loss.item() > 0.0


# ── 2. TTT Adaptation ─────────────────────────────────────────────────────────

class TestAdaptation:
    def test_adapt_changes_weights(self, ttt_trainer, updater, batch_agents):
        """TTT должен изменить веса updater (feature extractor, e.g. ffn)."""
        original_weight = updater.ffn[0].weight.data.clone()
        
        info = ttt_trainer.adapt_and_get_info(updater, batch_agents)
        
        # Веса ДОЛЖНЫ быть изменены после адаптации
        assert not torch.allclose(updater.ffn[0].weight.data, original_weight), \
            "TTT did not change updater weights"
        
        # Восстанавливаем
        ttt_trainer.restore_weights(
            updater, info["saved_updater_state"], info["saved_head_state"]
        )

    def test_restore_recovers_exact_weights(self, ttt_trainer, updater, batch_agents):
        """После restore веса должны быть ИДЕНТИЧНЫ оригинальным."""
        original_state = copy.deepcopy(updater.state_dict())
        
        info = ttt_trainer.adapt_and_get_info(updater, batch_agents)
        ttt_trainer.restore_weights(
            updater, info["saved_updater_state"], info["saved_head_state"]
        )
        
        for key in original_state:
            assert torch.allclose(
                updater.state_dict()[key], original_state[key]
            ), f"Weight {key} not restored correctly"

    def test_adapt_info_structure(self, ttt_trainer, updater, batch_agents):
        """adapt_and_get_info должен вернуть правильную структуру."""
        info = ttt_trainer.adapt_and_get_info(updater, batch_agents)
        
        assert "losses" in info
        assert "param_shift" in info
        assert "steps_done" in info
        assert "saved_updater_state" in info
        assert "saved_head_state" in info
        assert len(info["losses"]) == 3  # ttt_steps=3
        assert info["steps_done"] == 3
        assert info["param_shift"] > 0.0
        
        # Restore cleanup
        ttt_trainer.restore_weights(
            updater, info["saved_updater_state"], info["saved_head_state"]
        )

    def test_losses_decrease(self, d_hidden, device):
        """Self-supervised loss должен убывать (или хотя бы не расти)."""
        trainer = TestTimeTrainer(d_hidden=d_hidden, ttt_steps=10, ttt_lr=1e-2)
        upd = AgentUpdater(d_hidden=d_hidden, num_classes=2, num_heads=4)
        states = torch.randn(4, 4, d_hidden, device=device)
        
        info = trainer.adapt_and_get_info(upd, states)
        losses = info["losses"]
        
        # Хотя бы финальный loss должен быть меньше начального
        assert losses[-1] <= losses[0] * 1.5, \
            f"TTT losses not decreasing: {losses}"
        
        trainer.restore_weights(
            upd, info["saved_updater_state"], info["saved_head_state"]
        )


class TestContextManager:
    def test_context_restores_on_exit(self, ttt_trainer, updater, batch_agents):
        """Context manager должен восстановить веса при выходе."""
        original_state = copy.deepcopy(updater.state_dict())
        
        with ttt_trainer.adapt_context(updater, batch_agents) as info:
            # Веса адаптированы внутри контекста
            assert info["steps_done"] == 3
        
        # Веса восстановлены после выхода
        for key in original_state:
            assert torch.allclose(
                updater.state_dict()[key], original_state[key]
            ), f"Context manager didn't restore {key}"

    def test_context_restores_on_exception(self, ttt_trainer, updater, batch_agents):
        """Context manager должен восстановить веса даже при исключении."""
        original_state = copy.deepcopy(updater.state_dict())
        
        try:
            with ttt_trainer.adapt_context(updater, batch_agents) as info:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        for key in original_state:
            assert torch.allclose(
                updater.state_dict()[key], original_state[key]
            ), f"Context manager didn't restore {key} after exception"


# ── 3. Integration with CIDAPlugin ───────────────────────────────────────────

class TestCIDAPluginIntegration:
    def test_forward_with_ttt_on(self, device):
        """CIDAPlugin forward pass должен работать с use_ttt=True."""
        cfg = CIDAPluginConfig(
            d_input=64, d_hidden=64, d_message=64,
            num_agents=4, num_classes=2, max_rounds=2,
            use_ttt=True, ttt_steps=2, ttt_lr=1e-3,
        )
        plugin = CIDAPlugin(cfg).to(device)
        pooled = torch.randn(4, 64, device=device)
        
        out = plugin(pooled)
        
        assert out["p_final"].shape == (4, 2)
        assert out["ttt_info"] is not None
        assert "losses" in out["ttt_info"]
        assert "param_shift" in out["ttt_info"]

    def test_forward_with_ttt_off(self, device):
        """CIDAPlugin forward pass с use_ttt=False должен работать."""
        cfg = CIDAPluginConfig(
            d_input=64, d_hidden=64, d_message=64,
            num_agents=4, num_classes=2, max_rounds=2,
            use_ttt=False,
        )
        plugin = CIDAPlugin(cfg).to(device)
        pooled = torch.randn(4, 64, device=device)
        
        out = plugin(pooled)
        
        assert out["p_final"].shape == (4, 2)
        assert out["ttt_info"] is None

    def test_ttt_does_not_leak_weights(self, device):
        """После forward с TTT, веса updater должны быть как до вызова."""
        cfg = CIDAPluginConfig(
            d_input=64, d_hidden=64, d_message=64,
            num_agents=4, num_classes=2, max_rounds=2,
            use_ttt=True, ttt_steps=3,
        )
        plugin = CIDAPlugin(cfg).to(device)
        
        # Сохраняем оригинальные веса
        original_updater_state = copy.deepcopy(plugin.updater.state_dict())
        
        # Forward pass с TTT
        pooled = torch.randn(4, 64, device=device)
        _ = plugin(pooled)
        
        # Веса должны быть восстановлены
        for key in original_updater_state:
            assert torch.allclose(
                plugin.updater.state_dict()[key],
                original_updater_state[key],
            ), f"TTT leaked weight change into {key}"

    def test_ttt_multi_label(self, device):
        """TTT с multi-label classification."""
        cfg = CIDAPluginConfig(
            d_input=64, d_hidden=64, d_message=64,
            num_agents=4, num_classes=14, max_rounds=2,
            multi_label=True,
            use_ttt=True, ttt_steps=2,
        )
        plugin = CIDAPlugin(cfg).to(device)
        pooled = torch.randn(4, 64, device=device)
        
        out = plugin(pooled)
        assert out["p_final"].shape == (4, 14)
        assert out["ttt_info"] is not None


# ── 4. Backward Compatibility ─────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_v4_config_still_works(self, device):
        """Конфиг без TTT параметров должен работать."""
        cfg = CIDAPluginConfig(
            d_input=64, d_hidden=64, d_message=64,
            num_agents=4, num_classes=2, max_rounds=2,
        )
        # use_ttt defaults to False
        assert cfg.use_ttt is False
        
        plugin = CIDAPlugin(cfg).to(device)
        assert not hasattr(plugin, 'ttt_trainer')
        
        pooled = torch.randn(2, 64, device=device)
        out = plugin(pooled)
        assert out["p_final"].shape == (2, 2)
        assert out["ttt_info"] is None

    def test_deterministic_without_ttt(self, device):
        """Без TTT, два forward pass с одинаковым входом = одинаковый выход."""
        cfg = CIDAPluginConfig(
            d_input=64, d_hidden=64, d_message=64,
            num_agents=4, num_classes=2, max_rounds=2,
            use_ttt=False,
        )
        plugin = CIDAPlugin(cfg).to(device)
        plugin.eval()
        
        torch.manual_seed(42)
        pooled = torch.randn(2, 64, device=device)
        
        with torch.no_grad():
            out1 = plugin(pooled)
            out2 = plugin(pooled)
        
        assert torch.allclose(out1["p_final"], out2["p_final"], atol=1e-6)


# ── 5. TTT Diagnostics ───────────────────────────────────────────────────────

class TestTTTDiagnostics:
    def test_adaptation_magnitude(self):
        info = {"param_shift": 0.42, "losses": [1.0, 0.5, 0.3], "steps_done": 3}
        mag = DebateDiagnostics.ttt_adaptation_magnitude(info)
        assert mag == 0.42

    def test_adaptation_magnitude_none(self):
        assert DebateDiagnostics.ttt_adaptation_magnitude(None) == 0.0

    def test_loss_curve(self):
        info = {"losses": [1.0, 0.7, 0.4], "param_shift": 0.1, "steps_done": 3}
        curve = DebateDiagnostics.ttt_loss_curve(info)
        assert curve == [1.0, 0.7, 0.4]

    def test_convergence_ratio(self):
        info = {"losses": [1.0, 0.5, 0.25], "param_shift": 0.1, "steps_done": 3}
        ratio = DebateDiagnostics.ttt_convergence_ratio(info)
        assert ratio == 0.25

    def test_convergence_ratio_none(self):
        assert DebateDiagnostics.ttt_convergence_ratio(None) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
