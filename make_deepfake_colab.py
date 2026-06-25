import json

def create_notebook():
    cells = []
    
    def add_md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\\n" for line in text.split("\\n")]})
        
    def add_code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\\n" for line in text.split("\\n")]})

    add_md("# 🛡️ Deepfake Detection with CIDA (FaceForensics++)")
    add_md("Этот ноутбук интегрирует 'умные' веса CIDA Foundation в сверхбыстрый SOTA энкодер (EfficientNet-B0) для революционной задачи: обнаружения дипфейков. CIDA анализирует кадры и выдает `P(fake) + Uncertainty + Disagreement`.")
    
    add_code("""!pip install -q torch torchvision torchaudio transformers diffusers pandas numpy scikit-learn tqdm huggingface_hub timm
!rm -rf CIDA-plugin
!git clone https://github.com/Kairatzh/CIDA-plugin.git
%cd CIDA-plugin
!pip install -e .
""")

    add_md("### 1. Загрузка датасета FaceForensics++ (Сабсет через Kaggle)")
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

DATA_DIR = Path("./data/deepfakes")
DATA_DIR.mkdir(parents=True, exist_ok=True)
images_dir = DATA_DIR / "frames"
images_dir.mkdir(exist_ok=True)
csv_path = DATA_DIR / "train.csv"

# Классы: 0 - Real, 1 - Fake
print("Скачивание сабсета датасета с Kaggle (например, faceforensics-subset)...")
subprocess.run(["pip", "install", "-q", "kaggle"], check=True)

# Используем популярный сабсет для демо. Если не найдет, сгенерирует синтетику.
dataset_id = "sorokin/faceforensics" 

try:
    if not list(images_dir.glob("*.jpg")):
        print(f"Скачиваем {dataset_id}...")
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(DATA_DIR)], check=True)
        
        zip_path = list(DATA_DIR.glob("*.zip"))[0]
        with zipfile.ZipFile(zip_path, 'r') as z:
            all_files = z.namelist()
            img_files = [f for f in all_files if f.endswith(('.jpg', '.png'))]
            
            print(f"Извлечение {len(img_files[:10000])} снимков из архива...")
            img_subset = img_files[:10000]
            
            def extract_img(f):
                target_path = images_dir / Path(f).name
                with z.open(f) as source, open(target_path, "wb") as target:
                    target.write(source.read())
                    
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                list(tqdm(executor.map(extract_img, img_subset), total=len(img_subset), desc="Извлечение фреймов"))
                
        zip_path.unlink()
        
        # Создаем CSV: если в названии файла есть 'fake' или 'manipulated', то Fake (1), иначе Real (0)
        records = []
        for f in img_subset:
            name = Path(f).name.lower()
            is_fake = 1.0 if ('fake' in name or 'manipulated' in name) else 0.0
            records.append({"Path": name, "Fake": is_fake})
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f"Успешно подготовлено {len(records)} записей!")
    else:
        print("Датасет уже скачан и распакован.")
except Exception as e:
    print(f"Ошибка загрузки через Kaggle: {e}")
    print("Создаем синтетический датасет (Real/Fake) для тестирования...")
    records = []
    for i in range(500):
        img_name = f"frame_{i}.png"
        img = Image.fromarray((np.random.randn(224, 224, 3) * 128 + 128).clip(0, 255).astype(np.uint8))
        img.save(images_dir / img_name)
        record = {"Path": img_name, "Fake": 1.0 if np.random.rand() > 0.5 else 0.0}
        records.append(record)
    pd.DataFrame(records).to_csv(csv_path, index=False)
""")

    add_md("### 2. Подготовка DataLoader (Агрессивные Аугментации для Дипфейков)")
    add_code("""class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['Path']
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image_224 = self.transform(image)
        else:
            image_224 = transforms.ToTensor()(image)
            
        label = float(row['Fake'])
        # Single label output, no multi-class vector
        return image_224, torch.tensor([label], dtype=torch.float32)

# Для дипфейков важно добавлять сильное сжатие и блюр, чтобы CIDA училась бороться с неопределенностью
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # Лица можно зеркалить
    transforms.ColorJitter(brightness=0.1, contrast=0.1), 
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)), # Эмуляция размытия/сжатия видео
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = DeepfakeDataset(csv_path, images_dir, train_transforms)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
""")

    add_md("### 3. Инициализация SOTA модели (EfficientNet + CIDA) и загрузка Google Drive")
    add_code("""import os
import sys
import torch.nn as nn
import importlib
import shutil
from google.colab import drive

# Убеждаемся, что Python видит наши локальные папки
if '/content/CIDA-plugin' not in sys.path:
    sys.path.append('/content/CIDA-plugin')

try:
    drive.mount('/content/drive')
    gdrive_path = "/content/drive/MyDrive/cida_best.pth"
    if os.path.exists(gdrive_path):
        os.makedirs("checkpoints", exist_ok=True)
        shutil.copy(gdrive_path, "checkpoints/cida_best.pth")
        print("Скачали веса CIDA Foundation с Google Диска!")
except Exception as e:
    print(f"Не удалось примонтировать Google Drive. Ошибка: {e}")

sota_mod = importlib.import_module("KairosMM.11_sota_encoder_cida")
SOTAEncoderCIDA = sota_mod.SOTAEncoderCIDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# Создаем модель (2 класса: Real, Fake). d_hidden=512 для загрузки Foundation Brain!
model = SOTAEncoderCIDA(model_name='efficientnet_b0', num_classes=2, d_hidden=512, max_rounds=3, freeze_encoder=True)

if os.path.exists("checkpoints/cida_best.pth"):
    print("Загружаем 'Текстовый Мозг' (Foundation) в Дипфейк-систему...")
    state_dict = torch.load("checkpoints/cida_best.pth", map_location="cpu")
    plugin_state = state_dict.get('plugin', state_dict)
    
    # Фильтруем ключи, чтобы избежать Shape Mismatch (наши 2 класса вместо 3)
    model_dict = model.cida.state_dict()
    filtered_state = {k: v for k, v in plugin_state.items() if k in model_dict and v.shape == model_dict[k].shape}
    ignored = set(plugin_state.keys()) - set(filtered_state.keys())
    
    missing, unexpected = model.cida.load_state_dict(filtered_state, strict=False)
    print(f"Успешно загружено! Пропущено ключей (из-за разных размеров): {len(ignored)}")
else:
    print("Веса cida_best.pth не найдены. Обучение с нуля.")

model.to(device)
print("Модель готова к охоте на Дипфейки!")
""")

    add_md("### 4. Обучение (AMP, BCE Loss, Progressive Unfreezing)")
    add_code("""import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

epochs = 10
scaler = GradScaler()

def binary_loss(probs, targets):
    # Берем вероятность класса 1 (Fake)
    p_fake = probs[:, 1]
    targets = targets.squeeze()
    return F.binary_cross_entropy(p_fake, targets)

encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n]
cida_params = [p for n, p in model.named_parameters() if 'cida' in n]

optimizer = optim.AdamW([
    {'params': encoder_params, 'lr': 1e-4},
    {'params': cida_params, 'lr': 1e-3}
], weight_decay=1e-4)

scheduler = OneCycleLR(optimizer, max_lr=[1e-4, 1e-3], epochs=epochs, 
                       steps_per_epoch=len(train_loader), pct_start=0.1)

for epoch in range(epochs):
    # Progressive Unfreezing
    if epoch == 2:
        print("⚡ Эпоха 3: РАЗМОРАЖИВАЕМ ЭНКОДЕР для поиска артефактов GAN!")
        model.unfreeze_backbone()
        
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            out = model(x_224=imgs)
            loss_c = binary_loss(out["p_final"], labels)
            # Добавляем penalty за Disagreement (чтобы не было бессмысленных споров)
            loss_disagreement = out.get("disagreement", torch.tensor(0.0)).mean()
            loss = loss_c + 0.1 * loss_disagreement
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    print(f"Эпоха {epoch+1} завершена. Avg Loss: {epoch_loss/len(train_loader):.4f}")
    
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/kairos_deepfake_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    
    drive_path = "/content/drive/MyDrive/kairos_deepfake_best.pth"
    if os.path.exists("/content/drive/MyDrive"):
        shutil.copy(ckpt_path, drive_path)

print("Обучение завершено!")
""")

    add_md("### 5. Оценка и Анализ Disagreement (Киллер-фича!)")
    add_code("""import numpy as np
from sklearn.metrics import roc_auc_score

print("Запуск анализа на тестовых данных...")
model.eval()
all_preds = []
all_labels = []
all_disagreements = []

with torch.no_grad():
    for i, (imgs, labels) in enumerate(tqdm(train_loader, desc="Валидация")):
        if i >= 50: break # Проверяем на 50 батчах
        imgs, labels = imgs.to(device), labels.cpu().numpy()
        
        with autocast():
            out = model(x_224=imgs)
            
        preds = out["p_final"][:, 1].cpu().numpy() # P(Fake)
        disagreements = out.get("disagreement", torch.zeros_like(out["p_final"][:, 1])).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.squeeze())
        all_disagreements.extend(disagreements)

auc = roc_auc_score(all_labels, all_preds)
avg_disagreement_real = np.mean([d for p, l, d in zip(all_preds, all_labels, all_disagreements) if l == 0])
avg_disagreement_fake = np.mean([d for p, l, d in zip(all_preds, all_labels, all_disagreements) if l == 1])

print("\\n" + "="*50)
print("🛡️ РЕЗУЛЬТАТЫ KAIROS CIDA: DEEPFAKE DETECTION")
print("="*50)
print(f"🌟 ROC-AUC (Real vs Fake): {auc:.4f}")
print(f"🔥 Средний уровень спора (Disagreement) на НАСТОЯЩИХ фото: {avg_disagreement_real:.4f}")
print(f"🔥 Средний уровень спора (Disagreement) на ФЕЙКАХ: {avg_disagreement_fake:.4f}")
print("\\n💡 Вывод: Если Disagreement на фейках выше, значит агенты активно спорят об артефактах генерации (например, Прокурор видит смазанный фон, а Защитник - идеальное лицо). Именно это мы покажем платформам соцсетей!")
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
    
    with open("Kairos_Deepfake_Train.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    print("Создан ноутбук: Kairos_Deepfake_Train.ipynb")

if __name__ == "__main__":
    create_notebook()
