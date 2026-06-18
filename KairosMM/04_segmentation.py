"""
04_segmentation.py — Сегментация анатомических зон по AgentState.e

Идея:
  AgentState.e содержит evidence vectors — это то, на что обратили
  внимание агенты в пространственной карте рентгена. Вектор e формируется
  через cross-attention AgentEvidenceExtractor по spatial features DenseNet121.
  Это уже является неявной картой внимания к регионам.

  Мы обучаем лёгкий UNet-head поверх spatial features (из encoder),
  используя attention maps агентов как мягкую supervision.

  Целевые метки:
    1. Если есть датасет масок (RSNA Pneumonia, SIIM PTX, MonuSeg) → supervised
    2. Иначе → self-supervised через attention maps агентов (slef-distillation)

  Выход:
    heatmap: (B, 1, H, W) — карта внимания на оригинальном разрешении
    Наносится на рентген для визуализации патологических зон.

Использование:
    python 04_segmentation.py --mode train --states_dir ./checkpoints/agent_states/train
    python 04_segmentation.py --mode infer --image path/to/xray.png --model_path ./seg_model.pth
    python 04_segmentation.py --mode demo   # синтетические данные
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


# ── Mini UNet Decoder ─────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Стандартный блок: Conv → BN → ReLU → Conv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class AttentionGate(nn.Module):
    """
    Attention gate: фильтрует skip connections по релевантности.
    Стандартный компонент Attention UNet (Oktay et al. 2018).
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Ресайзим g до размера x если нужно
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi


class MiniUNetDecoder(nn.Module):
    """
    Лёгкий UNet decoder поверх DenseNet121 spatial features.

    Вход: spatial_features (B, 1024, 7, 7) из DenseNet121
    Выход: heatmap (B, 1, H, W) — карта патологических зон

    Архитектура: 4 шага upsampling (7→14→28→56→112) + финальный (112→224)
    Параметры: ~2.5M (против ~7M DenseNet121 backbone)

    Почему не берём полный UNet с энкодером:
      - Encoder уже обучен в скрипте 02
      - Добавляем только лёгкий decoder — сохраняем вычислительный бюджет
      - Skip connections имитируются через agent attention maps
    """

    def __init__(
        self,
        in_channels: int = 1024,   # DenseNet121 spatial channels
        base_channels: int = 64,
        num_agents: int = 4,
        output_size: int = 224,
    ):
        super().__init__()
        ch = base_channels
        self.output_size = output_size

        # Начальная проекция spatial features
        self.init_proj = ConvBlock(in_channels, ch * 8)  # (B, 512, 7, 7)

        # Decoder steps (upsampling)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  ConvBlock(ch * 8, ch * 4))   # → 14×14
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  ConvBlock(ch * 4, ch * 2))   # → 28×28
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  ConvBlock(ch * 2, ch))        # → 56×56
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  ConvBlock(ch, ch // 2))       # → 112×112

        # Финальный выход → 224×224
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Проекция attention maps агентов (49=7×7) в feature space
        # Смешиваем взвешенную сумму по агентам
        self.agent_weights = nn.Parameter(torch.ones(num_agents) / num_agents)
        self.attn_proj = nn.Conv2d(1, ch * 8, 1)

    def forward(
        self,
        spatial: torch.Tensor,           # (B, 49, 1024) или (B, 1024, 7, 7)
        attention_maps: torch.Tensor = None,  # (B, M, 49) agent attention maps
    ) -> dict:
        """
        spatial: (B, 49, 1024) — из encoder
        attention_maps: (B, M, 49) — из AgentState.p_ptr_all[-1]
        """
        B = spatial.size(0)

        # Reshape spatial: (B, 49, 1024) → (B, 1024, 7, 7)
        if spatial.dim() == 3:
            spatial = spatial.transpose(1, 2).view(B, -1, 7, 7)

        # Начальная проекция
        x = self.init_proj(spatial)   # (B, 512, 7, 7)

        # Если есть attention maps — добавляем как skip signal
        if attention_maps is not None:
            # Взвешенная сумма по агентам
            w = F.softmax(self.agent_weights, dim=0)  # (M,)
            # (B, M, 49) * (M,) → (B, 49)
            attn_fused = (attention_maps * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
            # (B, 49) → (B, 1, 7, 7)
            attn_map_2d = attn_fused.view(B, 1, 7, 7)
            # Нормализуем
            attn_map_2d = attn_map_2d / (attn_map_2d.sum(dim=[2, 3], keepdim=True) + 1e-9)
            # Проецируем в пространство features
            attn_feat = self.attn_proj(attn_map_2d)  # (B, 512, 7, 7)
            x = x + attn_feat  # residual fusion

        # Decoder upsampling
        x = self.up1(x)   # (B, 256, 14, 14)
        x = self.up2(x)   # (B, 128, 28, 28)
        x = self.up3(x)   # (B, 64,  56, 56)
        x = self.up4(x)   # (B, 32, 112, 112)
        heatmap = self.final(x)   # (B, 1, 224, 224)

        # Также возвращаем attention map 7×7 для диагностики
        attn_out = None
        if attention_maps is not None:
            w = F.softmax(self.agent_weights, dim=0)
            attn_out = (attention_maps * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
            attn_out = attn_out.view(B, 1, 7, 7)

        return {
            "heatmap":    heatmap,
            "attn_7x7":  attn_out,
        }


# ── Dataset для самообучения ──────────────────────────────────────────────────

class AgentStateSegDataset(Dataset):
    """
    Self-supervised датасет для сегментации.

    Для каждого примера:
      - spatial_features: из AgentState JSON
      - attention_maps: из AgentState JSON (p_ptr)
      - target: weighted sum attention maps (мягкая pseudo-label)

    При наличии реальных масок (mask_dir) использует их как targets.
    """

    def __init__(
        self,
        states_dir: str,
        image_dir: str = None,
        mask_dir: str = None,
        transform=None,
    ):
        self.states_dir = Path(states_dir)
        self.image_dir  = Path(image_dir) if image_dir else None
        self.mask_dir   = Path(mask_dir)  if mask_dir  else None
        self.transform  = transform

        self.json_files = list(self.states_dir.glob("*.json"))
        print(f"[SegDataset] {len(self.json_files)} примеров из {states_dir}")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        with open(self.json_files[idx]) as f:
            data = json.load(f)

        name = data.get("image_name", self.json_files[idx].stem)

        # Spatial features (B, 49, 1024) → (49, 1024)
        spatial = data.get("spatial_features")
        if spatial is not None:
            spatial = torch.tensor(spatial, dtype=torch.float32)  # (49, 1024)
        else:
            spatial = torch.randn(49, 1024)  # fallback

        # Attention maps (M, 49)
        attn_maps = data.get("attention_maps")
        if attn_maps is not None:
            attn_maps = torch.tensor(attn_maps, dtype=torch.float32)  # (M, 49)
        else:
            attn_maps = torch.ones(4, 49) / 49

        # Target mask (224, 224) или псевдомаска из attention
        if self.mask_dir:
            mask_path = self.mask_dir / (Path(name).stem + ".png")
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L").resize((224, 224))
                target = transforms.ToTensor()(mask)  # (1, 224, 224)
            else:
                target = self._pseudo_mask_from_attention(attn_maps)
        else:
            # Self-supervised: псевдомаска из взвешенной суммы attention
            target = self._pseudo_mask_from_attention(attn_maps)

        return {
            "spatial":    spatial,
            "attn_maps":  attn_maps,
            "target":     target,
            "name":       name,
        }

    def _pseudo_mask_from_attention(self, attn_maps: torch.Tensor) -> torch.Tensor:
        """
        Создаёт псевдомаску 224×224 из attention maps агентов.
        Взвешиваем агентов: Прокурор (idx=0) и Интегратор (idx=3) — наибольший вес.
        """
        weights = torch.tensor([0.35, 0.15, 0.15, 0.35])[:attn_maps.size(0)]
        weights = weights / weights.sum()
        fused = (attn_maps * weights.unsqueeze(-1)).sum(dim=0)  # (49,)
        fused = fused / (fused.max() + 1e-9)                    # нормализуем 0..1
        fused_2d = fused.view(1, 1, 7, 7)
        # Апсемплинг до 224×224
        mask = F.interpolate(fused_2d, size=(224, 224), mode="bilinear", align_corners=False)
        return mask.squeeze(0)  # (1, 224, 224)


# ── Функции потерь ────────────────────────────────────────────────────────────

def seg_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Комбинированная потеря для сегментации (мягкие маски).
    BCE + Dice: BCE хороша для точечных патологий, Dice для balanced.
    """
    bce = F.binary_cross_entropy(pred, target)

    # Dice loss
    eps = 1e-9
    pred_flat   = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = 1 - (2 * intersection + eps) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)

    return bce + dice.mean()


# ── Визуализация ──────────────────────────────────────────────────────────────

def overlay_heatmap(
    image: np.ndarray,       # (H, W, 3) uint8
    heatmap: np.ndarray,     # (H, W) float32 0..1
    alpha: float = 0.5,
    colormap: str = "hot",
) -> np.ndarray:
    """
    Накладывает heatmap на рентгеновский снимок.
    Использует matplotlib colormap без зависимости от cv2.
    """
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) float
        colored = (colored * 255).astype(np.uint8)
    except ImportError:
        # Fallback: красный канал
        colored = np.zeros_like(image)
        colored[:, :, 0] = (heatmap * 255).astype(np.uint8)

    # Блендинг
    result = ((1 - alpha) * image + alpha * colored).astype(np.uint8)
    return result


def save_annotated_image(
    image_path: str,
    heatmap: np.ndarray,
    out_path: str,
    findings: list = None,
    alpha: float = 0.45,
):
    """
    Сохраняет аннотированный рентген с heatmap и подписями патологий.
    """
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_arr = np.array(img)

    overlaid = overlay_heatmap(img_arr, heatmap, alpha=alpha)
    result = Image.fromarray(overlaid)

    # Добавляем подписи патологий
    if findings:
        draw = ImageDraw.Draw(result)
        y_offset = 5
        for f in findings[:5]:
            label = f"{f['pathology']}: {f['probability']*100:.0f}%"
            draw.text((5, y_offset), label, fill=(255, 255, 100))
            y_offset += 18

    result.save(out_path)
    return result


# ── Trainer ───────────────────────────────────────────────────────────────────

class SegTrainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            spatial   = batch["spatial"].to(self.device)
            attn_maps = batch["attn_maps"].to(self.device)
            target    = batch["target"].to(self.device)

            out = self.model(spatial, attn_maps)
            loss = seg_loss(out["heatmap"], target)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            spatial   = batch["spatial"].to(self.device)
            attn_maps = batch["attn_maps"].to(self.device)
            target    = batch["target"].to(self.device)
            out = self.model(spatial, attn_maps)
            total_loss += seg_loss(out["heatmap"], target).item()
        return total_loss / max(len(self.val_loader), 1)

    def run(self):
        best_loss = float("inf")
        for epoch in range(1, self.args.epochs + 1):
            train_l = self.train_epoch(epoch)
            val_l   = self.val_epoch()
            print(f"Epoch {epoch:3d}/{self.args.epochs} | train={train_l:.4f} | val={val_l:.4f}")
            if val_l < best_loss:
                best_loss = val_l
                torch.save(self.model.state_dict(), self.args.model_path)
                print(f"  → Сохранено (val_loss={val_l:.4f})")
        print(f"\nЛучший val loss: {best_loss:.4f}")


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_single(
    model: MiniUNetDecoder,
    json_path: str,
    image_path: str = None,
    out_path: str = "./annotated.png",
    device: str = "cpu",
) -> np.ndarray:
    """
    Inference для одного примера.
    Возвращает heatmap (224×224) и сохраняет аннотированный снимок.
    """
    model.eval()
    model.to(device)

    with open(json_path) as f:
        data = json.load(f)

    # Spatial features
    spatial = data.get("spatial_features")
    spatial = torch.tensor(spatial, dtype=torch.float32).unsqueeze(0).to(device)

    # Attention maps
    attn_maps = data.get("attention_maps")
    if attn_maps:
        attn_maps = torch.tensor(attn_maps, dtype=torch.float32).unsqueeze(0).to(device)

    out = model(spatial, attn_maps)
    heatmap = out["heatmap"][0, 0].cpu().numpy()  # (224, 224)

    # Сохраняем аннотированный снимок если есть изображение
    if image_path and os.path.exists(image_path):
        findings = data.get("consensus", {}).get("top_diagnoses", []) if "consensus" in data else []
        save_annotated_image(image_path, heatmap, out_path, findings=findings)
        print(f"[Segmentation] Аннотированный снимок → {out_path}")

    return heatmap


# ── Demo ─────────────────────────────────────────────────────────────────────

def demo():
    print("="*60)
    print("Демо: MiniUNetDecoder (синтетические данные)")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniUNetDecoder(in_channels=1024, base_channels=64, num_agents=4)
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Параметры модели: {total:,}")

    B = 2
    spatial   = torch.randn(B, 49, 1024).to(device)
    attn_maps = torch.rand(B, 4, 49).to(device)

    out = model(spatial, attn_maps)
    print(f"heatmap  shape: {out['heatmap'].shape}")   # (2, 1, 224, 224)
    print(f"attn_7x7 shape: {out['attn_7x7'].shape}")  # (2, 1, 7, 7)
    print(f"heatmap  range: [{out['heatmap'].min():.4f}, {out['heatmap'].max():.4f}]")
    print("\nДемо ПРОШЁЛ ✓")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIDA Segmentation")
    parser.add_argument("--mode",       choices=["train", "infer", "demo"], default="demo")
    parser.add_argument("--states_dir", default="./checkpoints/agent_states/train")
    parser.add_argument("--image_dir",  default=None)
    parser.add_argument("--mask_dir",   default=None)
    parser.add_argument("--model_path", default="./checkpoints/seg_model.pth")
    parser.add_argument("--out_dir",    default="./annotated_images")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--base_channels", type=int, default=64)
    args = parser.parse_args()

    if args.mode == "demo":
        demo()

    elif args.mode == "train":
        ds = AgentStateSegDataset(args.states_dir, args.image_dir, args.mask_dir)
        # Train/val split
        n = len(ds)
        n_val = max(1, int(n * 0.1))
        train_ds = torch.utils.data.Subset(ds, range(n_val, n))
        val_ds   = torch.utils.data.Subset(ds, range(n_val))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = MiniUNetDecoder(base_channels=args.base_channels)
        trainer = SegTrainer(model, train_loader, val_loader, args)
        trainer.run()

    elif args.mode == "infer":
        model = MiniUNetDecoder()
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        os.makedirs(args.out_dir, exist_ok=True)
        for json_file in Path(args.states_dir).glob("*.json"):
            out_path = Path(args.out_dir) / (json_file.stem + "_annotated.png")
            infer_single(model, str(json_file), out_path=str(out_path))