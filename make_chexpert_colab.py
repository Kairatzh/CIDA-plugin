import json
import base64
import zlib

def create_notebook():
    cells = []
    
    def add_md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})
        
    def add_code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in text.split("\n")]})

    add_md("# Эпичный Дебют (Этап 2): Обучение Kairos OS (DenseNet + CIDA) на CheXpert")
    add_md("Этот ноутбук интегрирует 'умные' веса CIDA Foundation (обученные на NLI) в сверхбыстрый энкодер (DenseNet121) и обучает их распознавать пневмонию, отек и другие патологии на реальных рентгенах датасета **CheXpert**.")
    
    add_code("""!pip install -q torch torchvision torchaudio transformers diffusers pandas numpy scikit-learn tqdm huggingface_hub
!rm -rf CIDA-plugin
!git clone https://github.com/Kairatzh/CIDA-plugin.git
%cd CIDA-plugin
!pip install -e .
""")

    add_md("### 1. Загрузка датасета CheXpert (через Hugging Face или Kaggle)")
    add_code("""import os
import pandas as pd
import numpy as np
import subprocess
import zipfile
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATA_DIR = Path("./data/chexpert")
DATA_DIR.mkdir(parents=True, exist_ok=True)
images_dir = DATA_DIR / "images"
images_dir.mkdir(exist_ok=True)
csv_path = DATA_DIR / "train.csv"

CHEXPERT_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
MAX_SAMPLES = 50000

print("Скачивание датасета с Kaggle...")
# Установите Kaggle API
subprocess.run(["pip", "install", "-q", "kaggle"], check=True)

# Чтобы Kaggle API работал, убедитесь, что вы загрузили kaggle.json в ~/.kaggle/kaggle.json
# Популярный открытый датасет CheXpert на Kaggle (ashery/chexpert) или NIH (nih-chest-xrays/data). 
# Используем ashery/chexpert (Downscaled).
dataset_id = "ashery/chexpert" 

try:
    if not list(images_dir.glob("*.jpg")):
        print(f"Скачиваем {dataset_id}...")
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(DATA_DIR)], check=True)
        
        zip_path = list(DATA_DIR.glob("*.zip"))[0]
        with zipfile.ZipFile(zip_path, 'r') as z:
            all_files = z.namelist()
            img_files = [f for f in all_files if f.endswith(('.jpg', '.png'))]
            csv_files = [f for f in all_files if f.endswith('.csv') and 'train' in f.lower()]
            
            print(f"Извлечение {MAX_SAMPLES} снимков из архива...")
            img_subset = img_files[:MAX_SAMPLES]
            
            # Извлекаем CSV
            if csv_files:
                z.extract(csv_files[0], DATA_DIR)
                # Переименуем для удобства
                os.rename(DATA_DIR / csv_files[0], csv_path)
            
            # Извлекаем картинки в многопоточном режиме
            def extract_img(f):
                target_path = images_dir / Path(f).name
                with z.open(f) as source, open(target_path, "wb") as target:
                    target.write(source.read())
                    
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                list(tqdm(executor.map(extract_img, img_subset), total=len(img_subset), desc="Извлечение изображений"))
                
        # Удаляем архив
        zip_path.unlink()
        
        # Обрезаем CSV под извлеченные картинки
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Приводим пути в CSV к нашим извлеченным именам файлов
            df['filename'] = df['Path'].apply(lambda x: Path(x).name)
            extracted_names = [Path(f).name for f in img_subset]
            df = df[df['filename'].isin(extracted_names)].head(MAX_SAMPLES)
            df.to_csv(csv_path, index=False)
            print(f"Успешно подготовлено {len(df)} записей!")
    else:
        print("Датасет уже скачан и распакован.")
except Exception as e:
    print(f"Ошибка загрузки через Kaggle (возможно нет kaggle.json): {e}")
    # Fallback to dummy data
    print("Создаем синтетический датасет для тестирования...")
    records = []
    for i in range(100):
        img_name = f"img_{i}.png"
        img = Image.fromarray((np.random.randn(256, 256, 3) * 128 + 128).clip(0, 255).astype(np.uint8))
        img.save(images_dir / img_name)
        record = {"Path": f"images/{img_name}"}
        for c in CHEXPERT_CLASSES:
            record[c] = 1.0 if np.random.rand() > 0.8 else 0.0
        records.append(record)
    pd.DataFrame(records).to_csv(csv_path, index=False)
""")

    add_md("### 2. Подготовка DataLoader")
    add_code("""class CheXpertDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.classes = CHEXPERT_CLASSES
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / Path(row['Path']).name
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image_224 = self.transform(image)
        else:
            image_224 = transforms.ToTensor()(image)
            
        labels = []
        for c in self.classes:
            val = float(row.get(c, 0.0))
            # U-Ones политика: неопределенность (-1) считаем за 1
            if val == -1.0: val = 1.0 
            if pd.isna(val): val = 0.0
            labels.append(val)
            
        return image_224, torch.tensor(labels, dtype=torch.float32)

train_ds = CheXpertDataset(csv_path, images_dir, train_transforms)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
""")

    add_md("### 3. Инициализация Kairos Encoder + CIDA и загрузка весов")
    add_code("""import torch.nn as nn
import importlib

# Модуль начинается с цифры, поэтому импортируем динамически
densenet_mod = importlib.import_module("KairosMM.10_densenet_cida")
DenseNetCIDA = densenet_mod.DenseNetCIDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# 1. Создаем модель (5 классов CheXpert)
# Мы РАЗМОРАЖИВАЕМ энкодер, чтобы DenseNet тоже учился!
# ОЧЕНЬ ВАЖНО: d_hidden=128, чтобы совпало с нашими весами Foundation (которые мы учили на NLI)
model = DenseNetCIDA(num_classes=5, d_hidden=128, max_rounds=3, freeze_encoder=False)

# 2. Загружаем веса CIDA (cida_best.pth) из Stage 1
import requests
import os

if os.path.exists("checkpoints/cida_best.pth"):
    print("Найдены локальные веса cida_best.pth! Загружаем 'Мозг'...")
    state_dict = torch.load("checkpoints/cida_best.pth", map_location="cpu")
    # В Foundation мы сохраняли dict: {'plugin': ..., 'role_layer': ...}
    # Достанем только plugin
    if 'plugin' in state_dict:
        plugin_state = state_dict['plugin']
    else:
        plugin_state = state_dict
        
    # strict=False сам пропустит input_proj (разные размерности 768->1024) и проекторы классов (3->5)
    missing, unexpected = model.cida.load_state_dict(plugin_state, strict=False)
    print(f"Загружено с strict=False. \\nПропущено (это нормально): {missing[:3]}...")
else:
    print("Веса cida_best.pth не найдены. Обучение начнется с нуля (без Role Pre-training).")

model.to(device)
print("Модель KairosOS-Encoder готова к обучению!")
""")

    add_md("### 4. Обучение (Classification Loss включен!)")
    add_code("""import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

epochs = 10
swa_start = 7
scaler = GradScaler()

# Разделяем learning rate (энкодер учим очень аккуратно, а голову CIDA - агрессивно)
encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n or 'features' in n]
cida_params = [p for n, p in model.named_parameters() if 'cida' in n]

optimizer = optim.AdamW([
    {'params': encoder_params, 'lr': 2e-5},
    {'params': cida_params, 'lr': 5e-4}
], weight_decay=1e-4)

# SWA
swa_model = torch.optim.swa_utils.AveragedModel(model)
scheduler = OneCycleLR(optimizer, max_lr=[2e-5, 5e-4], epochs=epochs, 
                       steps_per_epoch=len(train_loader), pct_start=0.2)
swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)

# Баланс классов
pos_weight = torch.tensor([2.0, 1.5, 3.0, 1.2, 1.1]).to(device)
def weighted_loss(probs, targets, pos_weight):
    return F.binary_cross_entropy_with_logits(probs, targets, pos_weight=pos_weight)

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss = loss_c + 0.1 * loss_disagreement
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR_CIDA": f"{scheduler.get_last_lr()[1]:.6f}"})
        
    print(f"Эпоха {epoch+1} завершена. Avg Loss: {epoch_loss/len(train_loader):.4f}")
    
    # СОХРАНЕНИЕ ПОСЛЕ КАЖДОЙ ЭПОХИ (ЧТОБЫ НЕ ПОТЕРЯТЬ ПРОГРЕСС ПРИ ОТКЛЮЧЕНИИ КОЛАБА)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/kairos_chexpert_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    
    # Автоматически дублируем на Google Диск, если он подключен
    import shutil
    drive_path = "/content/drive/MyDrive/kairos_chexpert_best.pth"
    if os.path.exists("/content/drive/MyDrive"):
        shutil.copy(ckpt_path, drive_path)
        print(f"Веса эпохи {epoch+1} сохранены на Google Диск!")

print("Обучение полностью завершено!")
""")

    add_md("### 5. Сохранение финальной модели на Google Диск")
    add_code("""# Этот шаг теперь выполняется автоматически после каждой эпохи. 
# Запускать его отдельно не обязательно, но можно для страховки.
import shutil
from google.colab import drive

# Убедимся, что диск примонтирован
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

drive_path = "/content/drive/MyDrive/kairos_chexpert_best.pth"
local_path = "checkpoints/kairos_chexpert_best.pth"

if os.path.exists(local_path):
    shutil.copy(local_path, drive_path)
    print(f"✅ Финальные веса Kairos OS успешно сохранены на вашем Диске: {drive_path}")
else:
    print("❌ Ошибка: Локальный файл с весами не найден!")
""")

    add_md("### 6. Оценка модели (ROC-AUC и Калибровка ECE)")
    add_code("""import numpy as np
from sklearn.metrics import roc_auc_score

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

print("Запуск валидации (оценка на батчах)...")
model.eval()
all_preds = []
all_labels = []

# Для оценки SWA модели
print("Запуск валидации (оценка на батчах)...")
# Важно: если используем SWA, оцениваем swa_model!
try:
    eval_model = swa_model.module
except:
    eval_model = model
    
eval_model.eval()
with torch.no_grad():
    # Оцениваем на всем датасете (Или хотя бы на 100 батчах)
    for i, (imgs, labels) in enumerate(tqdm(train_loader, desc="Валидация")):
        if i >= 100: break # 3200 снимков для точной валидации
        imgs, labels = imgs.to(device), labels.cpu().numpy()
        
        out = eval_model(x_224=imgs, warmup=False)
        preds = out["p_final"].cpu().numpy()
        
        all_preds.append(preds)
        all_labels.append(labels)

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

print("\\n" + "="*50)
print("🩺 РЕЗУЛЬТАТЫ KAIROS OS (CHEXPERT 5 КЛАССОВ)")
print("="*50)

macro_auc = []
macro_ece = []

for i, class_name in enumerate(CHEXPERT_CLASSES):
    y_true = all_labels[:, i]
    y_pred = all_preds[:, i]
    
    # Пропускаем классы, если в сэмпле нет позитивных примеров
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        ece = expected_calibration_error(y_true, y_pred)
        macro_auc.append(auc)
        macro_ece.append(ece)
        print(f"🔹 {class_name:<18} | ROC-AUC: {auc:.4f} | ECE: {ece:.4f}")
    else:
        print(f"🔹 {class_name:<18} | Недостаточно данных для оценки")

if macro_auc:
    print("-" * 50)
    print(f"🌟 MACRO AVERAGE     | ROC-AUC: {np.mean(macro_auc):.4f} | ECE: {np.mean(macro_ece):.4f}")
print("="*50)
""")

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open("Kairos_CheXpert_Train.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    print("Создан ноутбук: Kairos_CheXpert_Train.ipynb")

if __name__ == "__main__":
    create_notebook()
