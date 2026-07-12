"""
Тесты для CIDA v3 — Hugging Face Wrapper.

Покрывает:
    1. Инициализацию HFModelWithCIDA с кастомными параметрами.
    2. Обертку mock Hugging Face модели.
    3. Формат вывода с return_dict=True и return_dict=False.
    4. Вычисление лоссов при наличии labels.
    5. Прохождение градиентов обратно в энкодер.
"""

import pytest
import torch
import torch.nn as nn
from cida_latent_plugin import wrap_hf_model, HFModelWithCIDA, CIDAPluginConfig

# ── Mock Classes ──────────────────────────────────────────────────────────────

class MockConfig:
    def __init__(self):
        self.hidden_size = 64
        self.num_labels = 2
        self.use_return_dict = True


class MockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 64)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # input_ids будет иметь форму (B, seq_len)
        B, seq_len = input_ids.shape
        x = torch.randn(B, seq_len, 32, device=input_ids.device)
        last_hidden_state = self.linear(x)
        
        # Возвращаем объект, похожий на BaseModelOutput
        class MockOutput:
            def __init__(self, lhs):
                self.last_hidden_state = lhs
        return MockOutput(last_hidden_state)


class MockHFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        self.base_model = MockEncoder()
        self.classifier = nn.Linear(64, 2)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids, attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return logits


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")


def test_wrap_initialization():
    hf_model = MockHFModel()
    wrapped = wrap_hf_model(hf_model, max_rounds=3)
    
    assert isinstance(wrapped, HFModelWithCIDA)
    assert wrapped.num_labels == 2
    assert wrapped.plugin.config.max_rounds == 3
    assert wrapped.plugin.config.d_input == 64


def test_forward_return_dict(device):
    hf_model = MockHFModel().to(device)
    wrapped = wrap_hf_model(hf_model).to(device)
    
    input_ids = torch.randint(0, 100, (4, 10), device=device)
    attention_mask = torch.ones(4, 10, device=device)
    
    # 1. С return_dict=True
    out = wrapped(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    
    assert out.loss is None
    assert out.logits.shape == (4, 2)
    assert hasattr(out, "b_all")
    assert hasattr(out, "uncertainty")
    assert len(out.b_all) == 3  # initial belief + 2 deliberation rounds = 3
    assert out.uncertainty.shape == (4,)

    # 2. С return_dict=False
    out_tuple = wrapped(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    
    assert isinstance(out_tuple, tuple)
    assert out_tuple[0].shape == (4, 2)


def test_forward_with_labels_and_loss(device):
    hf_model = MockHFModel().to(device)
    wrapped = wrap_hf_model(hf_model).to(device)
    
    input_ids = torch.randint(0, 100, (4, 10), device=device)
    attention_mask = torch.ones(4, 10, device=device)
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long, device=device)
    
    out = wrapped(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    assert out.loss is not None
    assert out.loss.shape == ()
    assert out.loss.item() > 0.0


def test_backward_gradient_propagation(device):
    hf_model = MockHFModel().to(device)
    wrapped = wrap_hf_model(hf_model).to(device)
    
    # Убеждаемся, что градиенты текут в энкодер
    for p in wrapped.encoder.parameters():
        p.requires_grad = True

    input_ids = torch.randint(0, 100, (4, 10), device=device)
    attention_mask = torch.ones(4, 10, device=device)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    
    out = wrapped(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss
    loss.backward()
    
    # Проверяем градиент на весах линейного слоя в энкодере
    encoder_weight_grad = wrapped.encoder.linear.weight.grad
    assert encoder_weight_grad is not None
    assert encoder_weight_grad.abs().sum() > 0.0
