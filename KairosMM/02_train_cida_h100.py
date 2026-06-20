import argparse
import os
import sys
import time
from pathlib import Path

# Переопределяем кодировку stdout для Windows/Linux консолей
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

import importlib
try:
    encoder_mod = importlib.import_module("KairosMM.01_encoder_biomedclip")
    BiomedCLIPCIDA = encoder_mod.BiomedCLIPCIDA
    get_transforms_v2 = encoder_mod.get_transforms_v2
    NIH_CLASSES = encoder_mod.NIH_CLASSES
    _IMPORTS_OK = True
except (ImportError, SyntaxError) as err:
    _IMPORTS_OK = False
    print(f"[WARN] Импорты BiomedCLIPCIDA не удались: {err}. Запуск в режиме Dry-Run/Синтетика.")


def compute_auc(y_true: np.ndarray, y_score: np.ndarray, class_names: list) -> dict:
    results = {}
    aucs = []
    for i, cls in enumerate(class_names):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            try:
                auc = roc_auc_score(y_true[:, i], y_score[:, i])
                results[cls] = auc
                aucs.append(auc)
            except:
                results[cls] = float("nan")
        else:
            results[cls] = float("nan")
    results["macro_auc"] = np.mean(aucs) if aucs else 0.0
    return results


def find_nih_dataset(base_dir="./data"):
    """
    Интеллектуальный поиск директории оригинального датасета NIH ChestX-ray14
    """
    base_path = Path(base_dir)
    if base_path.exists():
        csv_candidates = list(base_path.rglob("Data_Entry_2017.csv"))
        if csv_candidates:
            print(f"[Dataset] Найден оригинальный файл Data_Entry_2017.csv в {csv_candidates[0].parent}")
            return csv_candidates[0].parent
    return None


class NIHChestXray14Dataset(torch.utils.data.Dataset):
    """
    NIH ChestXray14 Dataset, возвращающий изображение в двух разрешениях (224 и 448)
    с ограничением выборки для ускорения обучения на H100.
    """
    def __init__(self, data_dir: str, split: str = "train", val_fraction: float = 0.1, num_classes: int = 14, max_samples: int = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_classes = num_classes
        self.class_names = NIH_CLASSES[:num_classes] if _IMPORTS_OK else [f"Class_{i}" for i in range(num_classes)]

        self.transform_224 = get_transforms_v2(split, image_size=224) if _IMPORTS_OK else None
        self.transform_448 = get_transforms_v2(split, image_size=448) if _IMPORTS_OK else None

        csv_path = self.data_dir / "Data_Entry_2017.csv"
        if not csv_path.exists():
            csv_path = self.data_dir / "metadata.csv"

        if not csv_path.exists():
            print(f"[Dataset] Данные не найдены в {self.data_dir}. Синтетический режим.")
            self._synthetic_mode = True
            self.images = [f"fake_{i}.png" for i in range(100)]
            self.labels = np.random.randint(0, 2, size=(100, num_classes)).astype(np.float32)
            return

        df = pd.read_csv(csv_path)
        
        # Случайное сэмплирование для ограничения времени обучения
        if max_samples and len(df) > max_samples:
            print(f"[Dataset] Ограничение размера выборки: {len(df)} -> {max_samples}")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        label_matrix = np.zeros((len(df), self.num_classes), dtype=np.float32)
        has_class_cols = "Class_0" in df.columns

        for idx, row in df.iterrows():
            if has_class_cols:
                for i in range(self.num_classes):
                    label_matrix[idx, i] = float(row.get(f"Class_{i}", 0.0))
            else:
                for finding in str(row.get("Finding Labels", "")).split("|"):
                    finding = finding.strip()
                    if finding in self.class_names:
                        label_matrix[idx, self.class_names.index(finding)] = 1.0

        mask = pd.Series([True] * len(df))
        indices = np.where(mask)[0]
        np.random.seed(42)
        np.random.shuffle(indices)
        n_val = int(len(indices) * val_fraction)
        
        if self.split == "val":
            mask = pd.Series(False, index=range(len(df)))
            mask.iloc[indices[:n_val]] = True
        elif self.split == "train":
            mask = pd.Series(False, index=range(len(df)))
            mask.iloc[indices[n_val:]] = True

        df_split = df[mask].reset_index(drop=True)
        self.labels = label_matrix[mask]
        self.images = df_split["Image Index"].tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if getattr(self, "_synthetic_mode", False):
            return torch.randn(3, 224, 224), torch.randn(3, 448, 448), torch.tensor(self.labels[idx]), self.images[idx]
            
        img_name = self.images[idx]
        img_path = self.data_dir / "images" / img_name
        if not img_path.exists():
            img_path = self.data_dir / "images" / "images" / img_name

        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                img_224 = self.transform_224(img) if self.transform_224 else torch.randn(3, 224, 224)
                img_448 = self.transform_448(img) if self.transform_448 else torch.randn(3, 448, 448)
            except Exception:
                img_224 = torch.randn(3, 224, 224)
                img_448 = torch.randn(3, 448, 448)
        else:
            img_224 = torch.randn(3, 224, 224)
            img_448 = torch.randn(3, 448, 448)
            
        return img_224, img_448, torch.tensor(self.labels[idx]), img_name


class H100Trainer:
    """
    H100-optimized trainer.
    - BF16 mixed precision
    - TF32 matmul and cudnn enabled
    - Fused optimizer
    - pin_memory=True + non_blocking=True
    - torch.compile support
    - Автоматический обход CUDA Out of Memory (OOM) во время обучения
    """
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda" and hasattr(torch, "compile") and args.compile:
            print("[H100] Активация torch.compile(mode='max-autotune')...")
            self.model.cida = torch.compile(self.model.cida, mode='max-autotune')

        self.model.to(self.device)
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.lr, 
            weight_decay=1e-4,
            fused=True if self.device.type == "cuda" else False
        )
        
        self.dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        t0 = time.time()
        
        # Индекс батча для логирования
        batch_idx = 0
        loader_iter = iter(self.train_loader)
        
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device.type == "cuda":
                    print(f"\n  [WARN] CUDA OOM при загрузке батча! Уменьшаем batch_size...")
                    torch.cuda.empty_cache()
                    old_bs = self.train_loader.batch_size
                    new_bs = max(1, old_bs // 2)
                    if new_bs == old_bs:
                        raise e
                    self.train_loader = DataLoader(
                        self.train_loader.dataset,
                        batch_size=new_bs,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=self.train_loader.num_workers
                    )
                    loader_iter = iter(self.train_loader)
                    continue
                else:
                    raise e
                    
            imgs_224, imgs_448, labels, _ = batch
            imgs_224 = imgs_224.to(self.device, non_blocking=True)
            imgs_448 = imgs_448.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            try:
                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.device.type == 'cuda')):
                    out = self.model(imgs_224, imgs_448)
                    loss = F.binary_cross_entropy(out["p_final"], labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batch_idx += 1

                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] loss={loss.item():.4f}")
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device.type == "cuda":
                    print(f"\n  [WARN] CUDA OOM во время forward/backward! Сброс кэша и уменьшение batch_size...")
                    torch.cuda.empty_cache()
                    old_bs = self.train_loader.batch_size
                    new_bs = max(1, old_bs // 2)
                    self.train_loader = DataLoader(
                        self.train_loader.dataset,
                        batch_size=new_bs,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=self.train_loader.num_workers
                    )
                    loader_iter = iter(self.train_loader)
                    continue
                else:
                    raise e

        t1 = time.time()
        print(f"  Epoch {epoch} time: {t1-t0:.2f}s | Throughput: {len(self.train_loader.dataset)/(t1-t0):.1f} imgs/s")
        return total_loss / max(1, len(self.train_loader))

    def run(self):
        print(f"\n🚀 Запуск обучения на H100 (Precision: {self.args.precision})")
        for epoch in range(1, self.args.epochs + 1):
            loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} Avg Loss: {loss:.4f}\n")
            
        print("Обучение завершено. Сохранение...")
        self.model.save(self.args.out_dir)


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self): return 128
    def __getitem__(self, i): 
        return torch.randn(3, 224, 224), torch.randn(3, 448, 448), torch.rand(14).round(), f"fake_{i}.png"


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(224*224*3, 14)
        self.cida = nn.Linear(14, 14)
    def forward(self, x_224, x_448):
        x = x_224.view(x_224.size(0), -1)
        p = torch.sigmoid(self.cida(self.net(x)))
        return {"p_final": p}
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "biomedclip_cida.pth"))
        print("Saved dummy model weight.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--out_dir", default="./checkpoints/h100_model")
    parser.add_argument("--synthetic", action="store_true", default=False, help="Use fake data for dry run")
    parser.add_argument("--max_samples", type=int, default=5000, help="Limit maximum samples from dataset")
    args = parser.parse_args()

    # Интеллектуальный поиск датасета
    data_dir = find_nih_dataset("./data")

    if not _IMPORTS_OK or args.synthetic or data_dir is None:
        print("[Demo] Использование синтетических данных (оригинальный датасет не найден).")
        train_ds = DummyDataset()
        val_ds = DummyDataset()
        model = DummyModel()
    else:
        print(f"[Dataset] Загрузка датасета из {data_dir}")
        train_ds = NIHChestXray14Dataset(data_dir, split="train", max_samples=args.max_samples)
        val_ds = NIHChestXray14Dataset(data_dir, split="val", max_samples=args.max_samples // 10 if args.max_samples else None)
        model = BiomedCLIPCIDA(num_classes=14, d_hidden=768)

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True if torch.cuda.is_available() else False, 
        num_workers=4 if torch.cuda.is_available() else 0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    trainer = H100Trainer(model, train_loader, val_loader, args)
    trainer.run()


if __name__ == "__main__":
    main()
