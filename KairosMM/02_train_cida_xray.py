"""
02_train_cida_xray.py — Обучение CheXNetCIDA на NIH ChestX-ray14.

Датасет NIH ChestX-ray14:
  - 112,120 фронтальных рентгеновских снимков от 30,805 пациентов
  - 14 патологий (multi-label)
  - Скачать: https://nihcc.app.box.com/v/ChestXray-NIHCC
  - Или через kaggle: nih-chest-xrays

Что делает этот скрипт:
  1. Загружает датасет и разбивает на train/val/test
  2. Обучает CheXNetCIDA (encoder заморожен первые N эпох, затем fine-tune всё)
  3. Валидация по AUC (стандартная метрика для NIH ChestX-ray14)
  4. Сохраняет лучшую модель + AgentState для каждого примера из test set

Использование:
    python 02_train_cida_xray.py --data_dir ./data/nih_cxr14
    python 02_train_cida_xray.py --data_dir ./data/nih_cxr14 --epochs 20 --batch_size 32
    python 02_train_cida_xray.py --data_dir ./data/nih_cxr14 --quick_test  # 100 примеров
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

# ── Пути ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from cida_plugin import CIDALoss
    from encoder_chexnet import CheXNetCIDA, get_transforms, NIH_CLASSES  # type: ignore
    _CIDA_OK = True
except ImportError:
    # Fallback для standalone запуска из другой директории
    from importlib import import_module
    _CIDA_OK = False
    print("[WARN] Убедитесь что 01_encoder_chexnet.py и cida_plugin доступны")

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("[WARN] sklearn не установлен: pip install scikit-learn")


# ── Dataset ───────────────────────────────────────────────────────────────────

class NIHChestXray14Dataset(Dataset):
    """
    NIH ChestX-ray14 Dataset.

    Структура директории:
        data_dir/
            images/          ← все PNG файлы
            Data_Entry_2017.csv
            train_val_list.txt
            test_list.txt

    Если файлы разметки отсутствуют — создаём синтетический датасет для теста.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",           # "train" | "val" | "test"
        transform=None,
        val_fraction: float = 0.1,      # Доля от train_val_list для валидации
        num_classes: int = 14,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        self.class_names = NIH_CLASSES[:num_classes]

        self.images, self.labels = self._load_data(val_fraction)
        print(f"[NIHDataset] {split}: {len(self.images)} примеров")

    def _load_data(self, val_fraction):
        csv_path = self.data_dir / "Data_Entry_2017.csv"
        train_list_path = self.data_dir / "train_val_list.txt"
        test_list_path  = self.data_dir / "test_list.txt"

        # ── Синтетический режим (для отладки без реальных данных) ────────────
        if not csv_path.exists():
            print(f"[NIHDataset] Данные не найдены в {self.data_dir}. "
                  "Используется синтетический датасет для отладки.")
            return self._synthetic_data()

        # ── Реальные данные ────────────────────────────────────────────────────
        df = pd.read_csv(csv_path)

        # Парсинг меток (строка "Atelectasis|Cardiomegaly|..." → вектор)
        label_matrix = np.zeros((len(df), self.num_classes), dtype=np.float32)
        for idx, row in df.iterrows():
            findings = str(row["Finding Labels"]).split("|")
            for finding in findings:
                finding = finding.strip()
                if finding in self.class_names:
                    label_matrix[idx, self.class_names.index(finding)] = 1.0

        # Разбивка по файлам списков
        if self.split == "test" and test_list_path.exists():
            with open(test_list_path) as f:
                file_set = set(f.read().splitlines())
            mask = df["Image Index"].isin(file_set)
        else:
            if train_list_path.exists():
                with open(train_list_path) as f:
                    file_set = set(f.read().splitlines())
                mask = df["Image Index"].isin(file_set)
            else:
                mask = pd.Series([True] * len(df))

            # Дополнительное разбиение train/val
            indices = np.where(mask)[0]
            np.random.seed(42)
            np.random.shuffle(indices)
            n_val = int(len(indices) * val_fraction)
            if self.split == "val":
                mask = pd.Series(False, index=range(len(df)))
                mask.iloc[indices[:n_val]] = True
            else:
                mask = pd.Series(False, index=range(len(df)))
                mask.iloc[indices[n_val:]] = True

        df_split = df[mask].reset_index(drop=True)
        labels = label_matrix[mask]

        images = df_split["Image Index"].tolist()
        return images, labels

    def _synthetic_data(self):
        """Синтетический датасет для отладки (без реальных изображений)."""
        n = {"train": 500, "val": 100, "test": 100}[self.split]
        images = [f"synthetic_{i:06d}.png" for i in range(n)]
        labels = np.random.randint(0, 2, size=(n, self.num_classes)).astype(np.float32)
        self._synthetic_mode = True
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Синтетический режим — случайный тензор вместо изображения
        if getattr(self, "_synthetic_mode", False):
            img = torch.randn(3, 224, 224)
            return img, label, img_name

        # Реальный режим — загрузка из файла
        img_path = self.data_dir / "images" / img_name
        # NIH CXR14 иногда имеет вложенные директории images/images/
        if not img_path.exists():
            img_path = self.data_dir / "images" / "images" / img_name
        if not img_path.exists():
            # Ищем рекурсивно
            found = list(self.data_dir.rglob(img_name))
            img_path = found[0] if found else None

        if img_path is None or not img_path.exists():
            # Fallback: случайный тензор
            img = torch.randn(3, 224, 224)
        else:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)

        return img, label, img_name


# ── Метрики ───────────────────────────────────────────────────────────────────

def compute_auc(y_true: np.ndarray, y_score: np.ndarray, class_names: list) -> dict:
    """
    Вычисляет AUC для каждого класса и macro-AUC.
    Стандартная метрика для NIH ChestX-ray14.
    """
    results = {}
    valid_classes = []
    aucs = []

    for i, cls in enumerate(class_names):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            try:
                auc = roc_auc_score(y_true[:, i], y_score[:, i])
                results[cls] = auc
                aucs.append(auc)
                valid_classes.append(cls)
            except Exception:
                results[cls] = float("nan")
        else:
            results[cls] = float("nan")

    results["macro_auc"] = np.mean(aucs) if aucs else 0.0
    return results


# ── Trainer ───────────────────────────────────────────────────────────────────

class CIDATrainer:
    """
    Двухфазный trainer для CheXNetCIDA:
      Фаза A (warm-up): обучаем только CIDA, encoder заморожен
      Фаза B (fine-tune): обучаем всё с меньшим LR

    Это стандартная практика для медицинских моделей — избегаем
    разрушения ImageNet features на маленьких датасетах.
    """

    def __init__(
        self,
        model: "CheXNetCIDA",
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: argparse.Namespace,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = CIDALoss(
            multi_label=True,
            lambda_cal=0.4,
            lambda_ac=0.2,
            min_disagreement=0.05,
        )

        self.best_auc = 0.0
        self.history = {"train_loss": [], "val_auc": []}
        os.makedirs(args.out_dir, exist_ok=True)

    def _make_optimizer(self, phase: str) -> optim.Optimizer:
        """Разные LR для encoder и CIDA."""
        if phase == "warmup":
            # Только CIDA параметры
            params = [p for p in self.model.cida.parameters() if p.requires_grad]
            return optim.AdamW(params, lr=self.args.lr_cida, weight_decay=1e-4)
        else:
            # Всё, но с разными LR
            encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
            cida_params    = [p for p in self.model.cida.parameters() if p.requires_grad]
            return optim.AdamW([
                {"params": encoder_params, "lr": self.args.lr_encoder},
                {"params": cida_params,    "lr": self.args.lr_cida},
            ], weight_decay=1e-4)

    def train_epoch(self, optimizer: optim.Optimizer, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (imgs, labels, _) in enumerate(self.train_loader):
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            out = self.model(imgs)

            loss, components = self.loss_fn(
                p_final=out["p_final"],
                y=labels,
                b_all=out["b_all"],
                orth_loss=out.get("L_orth"),
            )
            loss.backward()

            # Gradient clipping — важно для стабильности CIDA
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"loss={loss.item():.4f} "
                      f"task={components['task']:.4f} "
                      f"cal={components['calibration']:.4f} "
                      f"ac={components['anti_collapse']:.4f}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        all_probs  = []
        all_labels = []

        for imgs, labels, _ in self.val_loader:
            imgs = imgs.to(self.device)
            out  = self.model(imgs)
            all_probs.append(out["p_final"].cpu().numpy())
            all_labels.append(labels.numpy())

        all_probs  = np.concatenate(all_probs,  axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if _SKLEARN_OK:
            metrics = compute_auc(all_labels, all_probs, self.model.class_names)
        else:
            # Fallback: accuracy
            preds = (all_probs > 0.5).astype(float)
            acc = (preds == all_labels).mean()
            metrics = {"macro_auc": acc, "note": "sklearn недоступен — используется accuracy"}

        return metrics

    @torch.no_grad()
    def save_agent_states(self, loader: DataLoader, split_name: str):
        """
        Прогоняет весь split через модель и сохраняет AgentState.
        Эти данные используются в следующих скриптах (03, 04, 05).
        """
        self.model.eval()
        states_dir = Path(self.args.out_dir) / "agent_states" / split_name
        states_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Сохранение AgentState] {split_name}...")
        for batch_idx, (imgs, labels, img_names) in enumerate(loader):
            imgs = imgs.to(self.device)
            out  = self.model(imgs)

            state = out["state_final"]
            B = imgs.size(0)

            for i in range(B):
                sample_data = {
                    # AgentState компоненты
                    "s":     state.s[i].cpu().numpy().tolist(),     # (M, d_hidden)
                    "b":     state.b[i].cpu().numpy().tolist(),     # (M, K)
                    "u":     state.u[i].cpu().numpy().tolist(),     # (M, 1)
                    "e":     state.e[i].cpu().numpy().tolist(),     # (M, d_hidden)
                    "alpha": state.alpha[i].cpu().numpy().tolist(), # (M, K)
                    # Финальные предсказания
                    "p_final":     out["p_final"][i].cpu().numpy().tolist(),
                    "uncertainty": out["uncertainty"][i].item(),
                    "disagreement": out["disagreement"][i].item(),
                    "rounds_used": out["rounds_used"],
                    # Метаданные
                    "image_name": img_names[i],
                    "true_labels": labels[i].numpy().tolist(),
                    "class_names": self.model.class_names,
                    # Пространственные features для сегментации (04_segmentation.py)
                    "spatial_features": out["spatial"][i].cpu().numpy().tolist(),  # (49, 1024)
                }
                # Attention maps (p_ptr) — куда смотрел каждый агент
                if out["p_ptr_all"]:
                    p_ptr_last = out["p_ptr_all"][-1]  # (B, M, 49)
                    sample_data["attention_maps"] = p_ptr_last[i].cpu().numpy().tolist()

                name = Path(img_names[i]).stem
                out_path = states_dir / f"{name}.json"
                with open(out_path, "w") as f:
                    json.dump(sample_data, f)

            if batch_idx % 20 == 0:
                print(f"  Сохранено: {(batch_idx+1)*loader.batch_size} / {len(loader.dataset)}")

        print(f"[Сохранение AgentState] Готово → {states_dir}")

    def run(self):
        """Полный цикл обучения."""
        args = self.args
        print(f"\n{'='*60}")
        print(f"Обучение CheXNetCIDA")
        print(f"Device: {self.device}")
        print(f"Epochs: {args.epochs} (warmup: {args.warmup_epochs})")
        print(f"{'='*60}\n")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            # Фаза warmup → fine-tune
            if epoch <= args.warmup_epochs:
                if epoch == 1:
                    print("[Фаза A] Обучение только CIDA (encoder заморожен)")
                    self.model.encoder.requires_grad_(False)
                    optimizer = self._make_optimizer("warmup")
            else:
                if epoch == args.warmup_epochs + 1:
                    print("[Фаза B] Fine-tune всей модели")
                    self.model.encoder.requires_grad_(True)
                    optimizer = self._make_optimizer("finetune")

            train_loss = self.train_epoch(optimizer, epoch)
            val_metrics = self.validate()
            macro_auc = val_metrics.get("macro_auc", 0.0)

            elapsed = time.time() - t0
            print(f"\nEpoch {epoch:3d}/{args.epochs} | "
                  f"loss={train_loss:.4f} | "
                  f"val_AUC={macro_auc:.4f} | "
                  f"time={elapsed:.1f}s")

            # Top-5 по AUC
            if _SKLEARN_OK:
                top5 = sorted(
                    [(k, v) for k, v in val_metrics.items()
                     if k not in ("macro_auc", "note") and not np.isnan(v)],
                    key=lambda x: x[1], reverse=True
                )[:5]
                print("  Top-5 AUC:", ", ".join(f"{k}={v:.3f}" for k, v in top5))

            self.history["train_loss"].append(train_loss)
            self.history["val_auc"].append(macro_auc)

            # Сохраняем лучшую модель
            if macro_auc > self.best_auc:
                self.best_auc = macro_auc
                self.model.save(os.path.join(args.out_dir, "best_model"))
                print(f"  → Новый лучший AUC: {macro_auc:.4f} (сохранено)")

        # Сохраняем историю
        history_path = os.path.join(args.out_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nИстория обучения → {history_path}")

        return self.best_auc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CheXNetCIDA Training")
    parser.add_argument("--data_dir",       default="./data/nih_cxr14")
    parser.add_argument("--out_dir",        default="./checkpoints")
    parser.add_argument("--epochs",         type=int, default=15)
    parser.add_argument("--warmup_epochs",  type=int, default=5,
                        help="Эпохи обучения только CIDA (encoder заморожен)")
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--lr_cida",        type=float, default=3e-4)
    parser.add_argument("--lr_encoder",     type=float, default=1e-5)
    parser.add_argument("--d_hidden",       type=int, default=256)
    parser.add_argument("--max_rounds",     type=int, default=3)
    parser.add_argument("--num_classes",    type=int, default=14)
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--quick_test",     action="store_true",
                        help="Использовать 100 примеров для быстрой проверки")
    parser.add_argument("--save_states",    action="store_true", default=True,
                        help="Сохранить AgentState для test set после обучения")
    parser.add_argument("--chexnet_weights", default=None,
                        help="Путь к предобученным весам CheXNet")
    args = parser.parse_args()

    # Датасеты
    train_ds = NIHChestXray14Dataset(
        args.data_dir, split="train",
        transform=get_transforms("train"),
        num_classes=args.num_classes,
    )
    val_ds = NIHChestXray14Dataset(
        args.data_dir, split="val",
        transform=get_transforms("val"),
        num_classes=args.num_classes,
    )
    test_ds = NIHChestXray14Dataset(
        args.data_dir, split="test",
        transform=get_transforms("test"),
        num_classes=args.num_classes,
    )

    if args.quick_test:
        print("[QUICK TEST] Используем 100 примеров")
        train_ds = Subset(train_ds, range(min(100, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(50,  len(val_ds))))
        test_ds  = Subset(test_ds,  range(min(50,  len(test_ds))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Модель
    model = CheXNetCIDA(
        num_classes=args.num_classes,
        multi_label=True,
        d_hidden=args.d_hidden,
        max_rounds=args.max_rounds,
        pretrained_encoder="imagenet",
        chexnet_weights_path=args.chexnet_weights,
    )

    # Обучение
    trainer = CIDATrainer(model, train_loader, val_loader, args)
    best_auc = trainer.run()
    print(f"\nЛучший val AUC: {best_auc:.4f}")

    # Сохранение AgentState для всех сплитов
    if args.save_states:
        print("\nСохранение AgentState для всех сплитов...")
        trainer.save_agent_states(test_loader,  "test")
        trainer.save_agent_states(val_loader,   "val")
        print("AgentState сохранены → используйте 03_agent_to_json.py")


if __name__ == "__main__":
    main()