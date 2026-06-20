import argparse
import os
import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ── GridDistortion Fallback or Import ─────────────────────────────────────────
try:
    from torchvision.transforms.v2 import GridDistortion
    _HAS_V2_GRID = True
except ImportError:
    _HAS_V2_GRID = False


class SimpleGridDistortion(nn.Module):
    """
    Эквивалент GridDistortion на чистом PyTorch для переносимости на H100.
    """
    def __init__(self, num_cells=5, distort_limit=0.1):
        super().__init__()
        self.num_cells = num_cells
        self.distort_limit = distort_limit

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            return img
        B, C, H, W = img.shape if img.dim() == 4 else (1, *img.shape)
        
        # Создаем искаженную сетку
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device),
            torch.linspace(-1, 1, W, device=img.device),
            indexing='ij'
        )
        
        # Добавляем случайные смещения в узлах сетки
        noise = (torch.rand(2, self.num_cells, self.num_cells, device=img.device) * 2 - 1) * self.distort_limit
        noise_expanded = F.interpolate(
            noise.unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)
        
        grid_x_distorted = grid_x + noise_expanded[0]
        grid_y_distorted = grid_y + noise_expanded[1]
        
        grid = torch.stack([grid_y_distorted, grid_x_distorted], dim=-1) # (H, W, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        input_img = img if img.dim() == 4 else img.unsqueeze(0)
        out = F.grid_sample(input_img, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        return out if img.dim() == 4 else out.squeeze(0)


# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from cida_plugin import CIDAPlugin, CIDAPluginConfig
    _CIDA_OK = True
except ImportError:
    _CIDA_OK = False
    print("[WARN] cida_plugin не найден.")

# ── Constants ─────────────────────────────────────────────────────────────────
NIH_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
VIT_HIDDEN_DIM = 768


class BiomedCLIPEncoder(nn.Module):
    """
    BiomedCLIP ViT-B/16 с поддержкой разных масштабов разрешения.
    """
    def __init__(self, freeze: bool = True):
        super().__init__()
        self.d_output = VIT_HIDDEN_DIM

        try:
            if os.environ.get("FORCE_FALLBACK") == "1":
                raise ImportError("Принудительный fallback режим активирован через FORCE_FALLBACK=1")
            import open_clip
            model, _, self.preprocess = open_clip.create_model_and_transforms(
                BIOMEDCLIP_MODEL_ID
            )
            self.visual = model.visual
            self._has_biomedclip = True
            print("[BiomedCLIPEncoder] ✅ BiomedCLIP loaded (ViT-B/16)")
        except Exception as e:
            print(f"[BiomedCLIPEncoder] ⚠️ BiomedCLIP не найден, используем ViT-ImageNet: {e}")
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.visual = vit
            self._has_biomedclip = False

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad = False
            print("[BiomedCLIPEncoder] Encoder weights frozen.")

    def forward_scale(self, x: torch.Tensor, spatial_size: int = 196):
        """
        Кодирует один масштаб изображения.
        spatial_size: ожидаемое количество патчей (196 для 224x224, 784 для 448x448).
        """
        if self._has_biomedclip:
            trunk = self.visual.trunk if hasattr(self.visual, 'trunk') else self.visual
            has_patch_embed = hasattr(trunk, 'patch_embed')
            has_pos_embed = hasattr(trunk, 'pos_embed') and trunk.pos_embed is not None
            
            orig_img_size = trunk.patch_embed.img_size if has_patch_embed else (224, 224)
            orig_pos_embed = trunk.pos_embed if has_pos_embed else None

            try:
                if has_patch_embed:
                    # Динамически выставляем текущий размер изображения, чтобы пройти проверку в timm
                    trunk.patch_embed.img_size = (x.shape[-2], x.shape[-1])
                
                if has_pos_embed and x.shape[-2:] != orig_img_size:
                    # Интерполяция позиционных эмбеддингов (pos_embed) для поддержки нестандартных сеток (например 448x448)
                    grid_h_orig, grid_w_orig = orig_img_size[0] // 16, orig_img_size[1] // 16
                    grid_h_new, grid_w_new = x.shape[-2] // 16, x.shape[-1] // 16
                    
                    cls_pos = orig_pos_embed[:, :1]
                    spatial_pos = orig_pos_embed[:, 1:].reshape(1, grid_h_orig, grid_w_orig, -1).permute(0, 3, 1, 2)
                    
                    spatial_pos_resized = F.interpolate(spatial_pos, size=(grid_h_new, grid_w_new), mode='bilinear', align_corners=False)
                    spatial_pos_resized = spatial_pos_resized.permute(0, 2, 3, 1).reshape(1, grid_h_new * grid_w_new, -1)
                    
                    trunk.pos_embed = torch.nn.Parameter(torch.cat([cls_pos, spatial_pos_resized], dim=1).to(x.device))

                if hasattr(trunk, 'forward_features'):
                    # Передаем напрямую в trunk.forward_features.
                    spatial_out = trunk.forward_features(x)
                    if isinstance(spatial_out, dict):
                        spatial = spatial_out.get('x', spatial_out.get('last_hidden_state', None))
                    else:
                        spatial = spatial_out
                else:
                    spatial = None
            finally:
                if has_patch_embed:
                    trunk.patch_embed.img_size = orig_img_size
                if has_pos_embed:
                    trunk.pos_embed = orig_pos_embed

            if spatial is None or spatial.dim() != 3:
                B = x.size(0)
                pooled = torch.zeros(B, self.d_output, device=x.device)
                spatial = pooled.unsqueeze(1).expand(B, spatial_size, -1)
                return pooled, spatial

            # Отрезаем CLS токен если он есть
            if spatial.size(1) == spatial_size + 1:
                pooled = spatial[:, 0, :]
                spatial = spatial[:, 1:, :]
            else:
                pooled = spatial.mean(dim=1)
                spatial = spatial[:, :spatial_size, :]

            return pooled, spatial
        else:
            # Torchvision ViT fallback (требует строго 224x224)
            B = x.size(0)
            if x.shape[-2:] != (224, 224):
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            out = self.visual(x)
            pooled = F.adaptive_avg_pool1d(out.unsqueeze(1), VIT_HIDDEN_DIM).squeeze(1)
            spatial = pooled.unsqueeze(1).expand(B, spatial_size, -1)
            return pooled, spatial

    def forward(self, x_224: torch.Tensor, x_448: torch.Tensor):
        """
        Двухмасштабное кодирование:
        x_224: (B, 3, 224, 224)
        x_448: (B, 3, 448, 448)
        """
        # Семантический масштаб (224x224) -> 196 токенов
        pooled_224, spatial_224 = self.forward_scale(x_224, spatial_size=196)
        
        # Локальный детальный масштаб (448x448) -> 784 токена
        pooled_448, spatial_448 = self.forward_scale(x_448, spatial_size=784)

        # Слияние:
        # Глобальный pooled: среднее семантики и локальных деталей
        pooled = 0.5 * pooled_224 + 0.5 * pooled_448
        
        # Пространственные токены: конкатенация для детального CIDA рассуждения
        # 196 + 784 = 980 пространственных токенов
        spatial = torch.cat([spatial_224, spatial_448], dim=1) # (B, 980, 768)

        return pooled, spatial


# ── CIDA Config for BiomedCLIP ────────────────────────────────────────────────
def make_cida_config_v2(
    num_classes: int = 14,
    multi_label: bool = True,
    d_hidden: int = 768,
    max_rounds: int = 3,
    num_agents: int = 4,
    **kwargs,
) -> "CIDAPluginConfig":
    return CIDAPluginConfig(
        d_input=VIT_HIDDEN_DIM,
        d_hidden=d_hidden,
        d_message=d_hidden,
        num_agents=num_agents,
        num_classes=num_classes,
        max_rounds=max_rounds,
        multi_label=multi_label,
        num_attn_heads=8,
        role_prior_blend=0.30,
        positive_class_idx=1,
        lambda_cal=0.4,
        lambda_ac=0.2,
        min_disagreement=0.05,
        comm_dropout=0.1,
        early_stop_threshold=0.88,
        use_perspective_projector=True,
        **kwargs,
    )


class BiomedCLIPCIDA(nn.Module):
    """
    Multi-Scale BiomedCLIP + CIDA Reasoning Engine (KairosMM H100 edition).
    """
    def __init__(
        self,
        num_classes: int = 14,
        multi_label: bool = True,
        d_hidden: int = 768,
        max_rounds: int = 3,
        num_agents: int = 4,
        freeze_encoder: bool = True,
        use_compile: bool = False,
    ):
        super().__init__()
        self.encoder = BiomedCLIPEncoder(freeze=freeze_encoder)

        if _CIDA_OK:
            self.cida_config = make_cida_config_v2(
                num_classes=num_classes,
                multi_label=multi_label,
                d_hidden=d_hidden,
                max_rounds=max_rounds,
                num_agents=num_agents,
            )
            self.cida = CIDAPlugin(self.cida_config)
        else:
            raise ImportError("cida_plugin необходим для BiomedCLIPCIDA")

        self.class_names = NIH_CLASSES[:num_classes]

        if use_compile and hasattr(torch, 'compile'):
            print("[BiomedCLIPCIDA] 🚀 torch.compile(mode='max-autotune') активирован")
            self.cida = torch.compile(self.cida, mode='max-autotune')

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[BiomedCLIPCIDA] Params: {total:,} total, {trainable:,} trainable")

    def forward(self, x_224: torch.Tensor, x_448: torch.Tensor) -> dict:
        """
        x_224: (B, 3, 224, 224)
        x_448: (B, 3, 448, 448)
        """
        pooled, spatial = self.encoder(x_224, x_448)

        # CIDA принимает 980 пространственных токенов
        out = self.cida(
            pooled=pooled,
            seq_output=spatial,
            mask=None,
        )

        out["pooled"] = pooled
        out["spatial"] = spatial
        return out

    def predict(self, x_224: torch.Tensor, x_448: torch.Tensor, threshold: float = 0.5) -> dict:
        self.eval()
        with torch.no_grad():
            out = self(x_224, x_448)
        probs = out["p_final"]
        preds = (probs > threshold).float()
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
        torch.save(self.state_dict(), os.path.join(path, "biomedclip_cida.pth"))
        print(f"[BiomedCLIPCIDA] Сохранено → {path}")


# ── Transforms & Augmentations with GridDistortion ────────────────────────────
def get_transforms_v2(split: str = "train", image_size: int = 224):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Подготовка аугментации GridDistortion
    distortion = []
    if split == "train":
        if _HAS_V2_GRID:
            distortion = [GridDistortion(num_cells=5, distort_limit=0.15)]
        else:
            distortion = [SimpleGridDistortion(num_cells=5, distort_limit=0.15)]

    transform_list = [
        transforms.Resize((image_size + 32, image_size + 32) if split == "train" else (image_size, image_size)),
    ]
    if split == "train":
        transform_list.extend([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ])
        transform_list.extend(distortion)
        transform_list.extend([
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        normalize,
    ])

    return transforms.Compose(transform_list)


def test_model():
    print("=" * 60)
    print("🔬 Тест Multi-Scale BiomedCLIPCIDA (KairosMM H100)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    model = BiomedCLIPCIDA(
        num_classes=14,
        multi_label=True,
        d_hidden=768,
        max_rounds=3,
        freeze_encoder=True,
    ).to(device)

    # Два входа разных масштабов
    batch_224 = torch.randn(2, 3, 224, 224).to(device)
    batch_448 = torch.randn(2, 3, 448, 448).to(device)
    
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
        out = model(batch_224, batch_448)

    print(f"\n📊 Выходы:")
    print(f"  p_final     : {out['p_final'].shape}")
    print(f"  uncertainty : {out['uncertainty'].shape}")
    print(f"  disagreement: {out['disagreement'].shape}")
    print(f"  AgentState.s: {out['state_final'].s.shape}")
    print(f"  spatial     : {out['spatial'].shape} (Ожидалось 980 токенов)")
    print(f"  b_all len   : {len(out['b_all'])}")
    print(f"  rounds_used : {out['rounds_used']}")

    assert out['spatial'].shape[1] == 980, f"Ошибка размерности spatial: {out['spatial'].shape}"
    print(f"\n✅ Тест Multi-Scale BiomedCLIPCIDA ПРОШЁЛ ✓")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiomedCLIP + CIDA encoder v2")
    parser.add_argument("--mode", choices=["test", "save"], default="test")
    parser.add_argument("--out", default="./checkpoints/encoder_v2")
    parser.add_argument("--d_hidden", type=int, default=768)
    args = parser.parse_args()

    if args.mode == "test":
        test_model()
    elif args.mode == "save":
        model = BiomedCLIPCIDA(d_hidden=args.d_hidden)
        model.save(args.out)
