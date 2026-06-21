import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import importlib

# Импорт модели CIDA через importlib из-за цифр в названии файла
try:
    mod = importlib.import_module("KairosMM.10_densenet_cida")
    DenseNetCIDA = mod.DenseNetCIDA
    NIH_CLASSES = mod.NIH_CLASSES
except Exception as e:
    print(f"[ERROR] Не удалось импортировать DenseNetCIDA: {e}")
    sys.exit(1)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_prob = np.clip(y_prob, 0, 1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

class BenchmarkDataset(Dataset):
    def __init__(self, csv_path, images_dir, max_samples=2000):
        self.df = pd.read_csv(csv_path)
        if len(self.df) > max_samples:
            # Берем случайную выборку, зафиксировав сид, чтобы результаты были воспроизводимы
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            
        self.images_dir = Path(images_dir)
        self.transform_224 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_448 = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.has_class_cols = "Class_0" in self.df.columns
        self.nih_classes = NIH_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = self.images_dir / img_name
        
        try:
            img = Image.open(img_path).convert("RGB")
            img_224 = self.transform_224(img)
            img_448 = self.transform_448(img)
        except Exception:
            img_224 = torch.zeros(3, 224, 224)
            img_448 = torch.zeros(3, 448, 448)
            
        if self.has_class_cols:
            labels = np.array([row[f"Class_{i}"] for i in range(14)], dtype=np.float32)
        else:
            labels = np.zeros(14, dtype=np.float32)
            findings = str(row.get("Finding Labels", "")).split("|")
            for f in findings:
                f = f.strip()
                if f in self.nih_classes:
                    labels[self.nih_classes.index(f)] = 1.0
            
        return img_224, img_448, torch.tensor(labels)

def main():
    parser = argparse.ArgumentParser(description="CIDA Benchmark Tool")
    parser.add_argument("--weights", type=str, default="checkpoints/cida_best.pth", help="Path to trained CIDA weights")
    parser.add_argument("--samples", type=int, default=2000, help="Number of samples to evaluate on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--data_dir", type=str, default="data/nih_chest_xray_hf", help="Dataset directory")
    args = parser.parse_args()

    print("="*70)
    print(" 🏥 KAIROS MM CIDA - ИНВЕСТОРСКИЙ БЕНЧМАРК")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Используемое устройство: {device}")

    # 1. Загрузка данных
    csv_path = Path(args.data_dir) / "Data_Entry_2017.csv"
    images_dir = Path(args.data_dir) / "images"
    
    if not csv_path.exists() or not images_dir.exists():
        print(f"[ERROR] Датасет не найден по пути {args.data_dir}. Запустите автообучение сначала.")
        return

    print(f"[*] Подготовка тестового набора ({args.samples} снимков)...")
    dataset = BenchmarkDataset(csv_path, images_dir, max_samples=args.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Инициализация модели
    print(f"[*] Инициализация архитектуры DenseNetCIDA...")
    model = DenseNetCIDA(num_classes=14, d_hidden=512, freeze_encoder=False)
    
    if Path(args.weights).exists():
        try:
            state_dict = torch.load(args.weights, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"[*] ✅ Веса успешно загружены из {args.weights}")
        except Exception as e:
            print(f"[ERROR] Не удалось загрузить веса: {e}")
            return
    else:
        print(f"[WARN] Файл весов {args.weights} не найден. Тест будет запущен с необученными весами.")
        
    model.to(device)
    model.eval()

    # 3. Инференс
    all_preds = []
    all_labels = []
    all_uncertainty = []

    print(f"[*] Старт оценки качества (Инференс)...")
    t0 = time.time()
    
    with torch.no_grad():
        for imgs_224, imgs_448, labels in tqdm(loader, desc="Benchmarking", unit="batch"):
            imgs_224 = imgs_224.to(device, non_blocking=True)
            imgs_448 = imgs_448.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(imgs_224, imgs_448)
                
            all_preds.append(out["p_final"].float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Если неопределенность возвращается как тензор
            if "u_alea" in out and "u_epi" in out:
                unc = (out["u_alea"].mean() + out["u_epi"].mean()).item()
                all_uncertainty.append(unc)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    elapsed = time.time() - t0
    print(f"[*] Инференс завершен за {elapsed:.2f} сек. ({(elapsed/args.samples)*1000:.2f} мс/снимок)")

    # 4. Расчет метрик
    results = []
    print("\n" + "="*80)
    print(f"{'Патология (Класс)':<22} | {'ROC-AUC':<8} | {'ECE':<8} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 80)

    for i, class_name in enumerate(NIH_CLASSES):
        y_true = all_labels[:, i]
        y_prob = all_preds[:, i]
        y_pred = (y_prob > 0.5).astype(int)
        
        # Если в тестовой выборке нет ни одного позитивного примера для класса
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            auc = 0.0
            p, r, f, ece = 0.0, 0.0, 0.0, 0.0
        else:
            auc = roc_auc_score(y_true, y_prob)
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            ece = expected_calibration_error(y_true, y_prob)
            
        results.append({
            "Class": class_name,
            "AUC": auc,
            "ECE": ece,
            "Precision": p,
            "Recall": r,
            "F1": f
        })
        
        print(f"{class_name:<22} | {auc:.4f}   | {ece:.4f}   | {p:.4f}      | {r:.4f}   | {f:.4f}")

    print("="*70)
    df_res = pd.DataFrame(results)
    macro_auc = df_res[df_res['AUC'] > 0]['AUC'].mean()
    print(f"🔥 MACRO AVERAGE AUC: {macro_auc:.4f}")
    if all_uncertainty:
        print(f"🧠 Средняя Неопределенность (Uncertainty) CIDA: {np.mean(all_uncertainty):.4f}")
    print("="*70)

    # Сохраняем отчет
    report_path = ROOT_DIR / "benchmark_report.csv"
    df_res.to_csv(report_path, index=False)
    print(f"[*] Детальный отчет сохранен: {report_path}")

if __name__ == "__main__":
    main()
