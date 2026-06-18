"""
01_encoder_chexnet.py — CheXNet DenseNet121 encoder for CIDA-Plugin.

CheXNet (Rajpurkar et al., Stanford 2017) — первая модель, превзошедшая
радиологов на NIH ChestX-ray14. Стандарт де-факто для рентгенологии.

Что делает этот скрипт:
  1. Загружает DenseNet121 с ImageNet или CheXNet весами
  2. Адаптирует выходы под CIDA: pooled (B, 1024) + spatial (B, 49, 1024)
  3. Собирает полный pipeline: изображение → CIDA → logits + AgentState
  4. Сохраняет веса для следующего скрипта

Использование:
    python 01_encoder_chexnet.py --mode test
    python 01_encoder_chexnet.py --mode save --out ./checkpoints/encoder
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights

# ── Путь к cida_plugin (если запускается standalone) ─────────────────────────
CIDA_PATH = os.environ.get("CIDA_PATH", "./cida_plugin")
if os.path.isdir(CIDA_PATH):
    sys.path.insert(0, os.path.dirname(CIDA_PATH))

try:
    from cida_plugin import CIDAPlugin, CIDAPluginConfig
except ImportError:
    print("[WARN] cida_plugin не найден. Установите или укажите CIDA_PATH.")


# ── Константы ─────────────────────────────────────────────────────────────────

# 14 патологий NIH ChestX-ray14 (стандарт для рентгенологии)
NIH_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# Нормализация ImageNet (используется CheXNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# DenseNet121 выходная размерность
DENSENET121_DIM = 1024  # после avgpool, перед оригинальным classifier


# ── CheXNet Encoder ────────────────────────────────────────────────────────────

class CheXNetEncoder(nn.Module):
    """
    DenseNet121 как feature extractor для CIDA.

    Возвращает:
        pooled  : (B, 1024) — глобальный дескриптор (для CIDA pooled input)
        spatial : (B, 49, 1024) — пространственная карта 7×7 (для evidence pointers)

    Почему DenseNet121:
      - Победитель NIH ChestX-ray14 Challenge (Rajpurkar 2017)
      - Dense connections → лучший градиентный поток, меньше переобучение
      - 1024-мерный выход совместим с CIDA d_hidden без большой проекции
      - Spatial карта 7×7=49 токенов аналогична ViT patch tokens
    """

    def __init__(
        self,
        pretrained: str = "imagenet",   # "imagenet" | "chexnet" | None
        freeze_bn: bool = True,          # замораживать BatchNorm при fine-tune
    ):
        super().__init__()

        # Загружаем DenseNet121
        if pretrained == "imagenet":
            base = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            print("[CheXNetEncoder] Загружены веса ImageNet")
        else:
            base = models.densenet121(weights=None)
            print("[CheXNetEncoder] Инициализация случайными весами")

        # Берём backbone (features) без classifier
        self.features = base.features   # выход: (B, 1024, 7, 7) для 224x224 входа

        if freeze_bn:
            # Замораживаем BatchNorm — стандартная практика при fine-tune на мед. данных
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        self.d_output = DENSENET121_DIM  # 1024

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W) — нормализованное изображение

        Returns:
            pooled  : (B, 1024)
            spatial : (B, 49, 1024)  при H=W=224, иначе (B, HW//32², 1024)
        """
        feat = self.features(x)                  # (B, 1024, H/32, W/32)
        feat = F.relu(feat, inplace=True)

        B, C, H, W = feat.shape
        # Spatial: разворачиваем в последовательность токенов
        spatial = feat.view(B, C, H * W).transpose(1, 2)  # (B, H*W, 1024)

        # Pooled: global average pooling
        pooled = feat.mean(dim=[2, 3])           # (B, 1024)

        return pooled, spatial

    def load_chexnet_weights(self, path: str):
        """
        Загружает оригинальные веса CheXNet из .pth файла.
        Оригинальная модель обёрнута в nn.DataParallel — убираем prefix "module.".

        Скачать веса: https://github.com/arnoweng/CheXNet
        """
        state = torch.load(path, map_location="cpu")
        # Убираем 'module.' prefix (DataParallel)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # Загружаем только features (classifier у нас другой)
        feat_state = {
            k.replace("densenet121.", ""): v
            for k, v in state.items()
            if k.startswith("densenet121.features")
        }
        missing, unexpected = self.features.load_state_dict(feat_state, strict=False)
        print(f"[CheXNetEncoder] CheXNet веса загружены. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")


# ── CIDA конфигурация для рентгенологии ───────────────────────────────────────

def make_cida_config(
    num_classes: int = 14,
    multi_label: bool = True,
    d_hidden: int = 256,
    max_rounds: int = 3,
    **kwargs,
) -> "CIDAPluginConfig":
    """
    Создаёт CIDAPluginConfig оптимизированный для рентгенологии.

    Ключевые решения:
      - multi_label=True: у одного пациента может быть несколько патологий
      - d_hidden=256: компромисс между качеством и скоростью на T4 GPU
      - positive_class_idx=1: "положительный" класс = патология есть
      - max_rounds=3: достаточно для медицинской задачи, не слишком дорого
    """
    return CIDAPluginConfig(
        d_input=DENSENET121_DIM,   # 1024 → от DenseNet121
        d_hidden=d_hidden,
        d_message=d_hidden,
        num_agents=4,
        num_classes=num_classes,
        max_rounds=max_rounds,
        multi_label=multi_label,
        num_attn_heads=4,
        role_prior_blend=0.30,
        positive_class_idx=1,
        lambda_cal=0.4,
        lambda_ac=0.2,
        min_disagreement=0.05,     # Ниже чем дефолт — multi-label задача сложнее
        comm_dropout=0.1,
        early_stop_threshold=0.88,
        use_perspective_projector=True,
        **kwargs,
    )


# ── Полный pipeline: Encoder + CIDA ───────────────────────────────────────────

class CheXNetCIDA(nn.Module):
    """
    Полный pipeline рентгенологической классификации с CIDA.

    Encoder → pooled + spatial → CIDA → p_final + AgentState

    Пример:
        model = CheXNetCIDA(num_classes=14, multi_label=True)
        img = torch.randn(2, 3, 224, 224)
        out = model(img)
        print(out["p_final"].shape)       # (2, 14)
        print(out["state_final"].s.shape) # (2, 4, 256)
    """

    def __init__(
        self,
        num_classes: int = 14,
        multi_label: bool = True,
        d_hidden: int = 256,
        max_rounds: int = 3,
        pretrained_encoder: str = "imagenet",
        freeze_encoder: bool = False,
        chexnet_weights_path: str = None,
        **cida_kwargs,
    ):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = CheXNetEncoder(
            pretrained=pretrained_encoder,
            freeze_bn=True,
        )

        if chexnet_weights_path and os.path.exists(chexnet_weights_path):
            self.encoder.load_chexnet_weights(chexnet_weights_path)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[CheXNetCIDA] Encoder заморожен")

        # ── CIDA Plugin ───────────────────────────────────────────────────────
        self.cida_config = make_cida_config(
            num_classes=num_classes,
            multi_label=multi_label,
            d_hidden=d_hidden,
            max_rounds=max_rounds,
            **cida_kwargs,
        )
        self.cida = CIDAPlugin(self.cida_config)

        # Классы для интерпретации
        self.class_names = NIH_CLASSES[:num_classes]
        print(f"[CheXNetCIDA] Инициализирован: {num_classes} классов, "
              f"d_hidden={d_hidden}, max_rounds={max_rounds}")

    def forward(
        self,
        x: torch.Tensor,
        return_agent_state: bool = True,
    ) -> dict:
        """
        x: (B, 3, H, W)

        Returns dict:
            p_final      : (B, num_classes) — вероятности патологий
            state_final  : AgentState       — финальное состояние агентов
            b_all        : list[(B,M,K)]    — история убеждений
            uncertainty  : (B,)             — неопределённость
            disagreement : (B,)             — несогласие агентов
            pooled       : (B, 1024)        — raw encoder features
            spatial      : (B, 49, 1024)    — spatial encoder features
        """
        # Encoder
        pooled, spatial = self.encoder(x)   # (B,1024), (B,49,1024)

        # CIDA deliberation
        out = self.cida(
            pooled=pooled,
            seq_output=spatial,
            mask=None,  # DenseNet spatial features — без padding
        )

        out["pooled"] = pooled
        out["spatial"] = spatial
        return out

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict:
        """Inference: возвращает бинарные предсказания и уверенность."""
        self.eval()
        with torch.no_grad():
            out = self(x)
        probs = out["p_final"]                               # (B, K)
        preds = (probs > threshold).float()                  # (B, K)
        return {
            "probabilities": probs,
            "predictions": preds,
            "uncertainty": out["uncertainty"],
            "disagreement": out["disagreement"],
            "class_names": self.class_names,
            "state_final": out["state_final"],
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "chexnet_cida.pth"))
        self.cida.save_pretrained(os.path.join(path, "cida_config"))
        print(f"[CheXNetCIDA] Сохранено в {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "CheXNetCIDA":
        # Загружаем конфиг чтобы знать параметры
        from cida_plugin import CIDAPluginConfig
        cida_cfg = CIDAPluginConfig.from_pretrained(os.path.join(path, "cida_config"))
        model = cls(
            num_classes=cida_cfg.num_classes,
            multi_label=cida_cfg.multi_label,
            d_hidden=cida_cfg.d_hidden,
            max_rounds=cida_cfg.max_rounds,
            pretrained_encoder=None,
            **kwargs,
        )
        state = torch.load(os.path.join(path, "chexnet_cida.pth"), map_location="cpu")
        model.load_state_dict(state)
        print(f"[CheXNetCIDA] Загружено из {path}")
        return model


# ── Трансформации изображений ─────────────────────────────────────────────────

def get_transforms(split: str = "train", image_size: int = 224):
    """
    Аугментации для рентгенологии.

    Train: случайный flip + rotation (рентген симметричен по горизонтали)
    Val/Test: только resize + normalize
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ── Тест / демо ───────────────────────────────────────────────────────────────

def test_model():
    """Smoke test: проверяем размерности выходов."""
    print("=" * 60)
    print("Тест CheXNetCIDA")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = CheXNetCIDA(
        num_classes=14,
        multi_label=True,
        d_hidden=256,
        max_rounds=3,
        pretrained_encoder="imagenet",
    ).to(device)

    # Подсчёт параметров
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметры всего:    {total:,}")
    print(f"Параметры обучаемые:{trainable:,}")

    # Forward pass
    batch = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(batch)

    print(f"\nВыходы:")
    print(f"  p_final     : {out['p_final'].shape}")
    print(f"  uncertainty : {out['uncertainty'].shape}")
    print(f"  disagreement: {out['disagreement'].shape}")
    print(f"  AgentState.s: {out['state_final'].s.shape}")
    print(f"  AgentState.b: {out['state_final'].b.shape}")
    print(f"  AgentState.e: {out['state_final'].e.shape}")
    print(f"  b_all len   : {len(out['b_all'])}")
    print(f"  rounds_used : {out['rounds_used']}")
    print(f"  pooled      : {out['pooled'].shape}")
    print(f"  spatial     : {out['spatial'].shape}")

    # Проверяем p_final — должен быть в (0, 1) для multi-label
    p = out["p_final"]
    assert p.min() >= 0 and p.max() <= 1, "p_final не в [0,1]!"
    print(f"\n  p_final range: [{p.min():.4f}, {p.max():.4f}] ✓")
    print(f"\nТест ПРОШЁЛ ✓")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CheXNet + CIDA encoder")
    parser.add_argument("--mode", choices=["test", "save"], default="test")
    parser.add_argument("--out", default="./checkpoints/encoder")
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--d_hidden", type=int, default=256)
    parser.add_argument("--chexnet_weights", default=None,
                        help="Путь к оригинальным весам CheXNet (.pth)")
    args = parser.parse_args()

    if args.mode == "test":
        test_model()
    elif args.mode == "save":
        model = CheXNetCIDA(
            num_classes=args.num_classes,
            d_hidden=args.d_hidden,
            chexnet_weights_path=args.chexnet_weights,
        )
        model.save(args.out)