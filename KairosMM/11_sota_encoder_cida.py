import torch
import torch.nn as nn
import timm

from cida_plugin.core import CIDAPlugin
from cida_plugin.config import CIDAPluginConfig

def make_cida_config_sota(
    num_classes: int = 14,
    multi_label: bool = True,
    d_input: int = 1280, # Зависит от энкодера (у EfficientNet-B0 это 1280)
    d_hidden: int = 128,
    max_rounds: int = 3,
    num_agents: int = 4,
    **kwargs,
) -> CIDAPluginConfig:
    return CIDAPluginConfig(
        d_input=d_input, 
        d_hidden=d_hidden,
        num_classes=num_classes,
        num_agents=num_agents,
        max_rounds=max_rounds,
        multi_label=multi_label,
        # KDD (Kairos Dynamic Deliberation)
        enable_rollback=True,
        superposition_threshold=0.4,
        # Architectural
        comm_heads=4,
        dropout=0.1,
        **kwargs
    )

class SOTAEncoderCIDA(nn.Module):
    """
    Эпичный медицинский модуль: SOTA Encoder (через timm) + CIDA Cognitive Core.
    """
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 14,
        d_hidden: int = 128,
        max_rounds: int = 3,
        num_agents: int = 4,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        # 1. Загружаем SOTA энкодер через timm
        # num_classes=0 отключает финальный классификатор, возвращая только признаки (pooled features)
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Определяем размерность выхода энкодера (на лету)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            d_input = self.encoder(dummy_input).shape[1]
            
        print(f"✅ SOTA Энкодер [{model_name}] загружен. Выходной вектор: {d_input}D")

        # 2. Создаем конфигурацию CIDA с правильным d_input
        self.cida_config = make_cida_config_sota(
            num_classes=num_classes,
            d_input=d_input,
            d_hidden=d_hidden,
            max_rounds=max_rounds,
            num_agents=num_agents
        )
        
        # 3. Инициализируем плагин CIDA
        self.cida = CIDAPlugin(self.cida_config)
        
        # 4. Заморозка энкодера (для Progressive Unfreezing)
        if freeze_encoder:
            self.freeze_backbone()
            
    def freeze_backbone(self):
        """Замораживает веса энкодера."""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Размораживает веса энкодера."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x_224: torch.Tensor, warmup: bool = False, temperature: float = 1.0):
        # 1. Извлекаем признаки (Global Average Pooling внутри timm)
        # Формат выхода: (batch_size, d_input)
        features = self.encoder(x_224)
        
        # 2. Передаем в CIDA
        out = self.cida(features, warmup=warmup, temperature=temperature)
        return out
