import json

def create_arc_notebook():
    cells = []
    
    def add_md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})
        
    def add_code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in text.split("\n")]})

    add_md("# 🧩 Эпичное Обучение CIDA на ARC-AGI (Универсальный Интеллект)")
    add_md("Этот ноутбук обучает **CIDA-Plugin** решать задачи из **ARC-AGI** (Abstraction and Reasoning Corpus) — самого сложного бенчмарка общего ИИ в мире.")
    
    add_code("""!pip install -q torch torchvision pandas numpy scikit-learn tqdm timm matplotlib
!rm -rf CIDA-plugin
!git clone https://github.com/Kairatzh/CIDA-plugin.git
%cd CIDA-plugin
!pip install -e .
""")

    add_md("### 1. Загрузка официального датасета ARC-AGI")
    add_code("""import subprocess
import os
from pathlib import Path

# Скачиваем оригинальный репозиторий ARC от Франсуа Шолле
if not os.path.exists("ARC-AGI"):
    print("Скачивание репозитория ARC-AGI...")
    subprocess.run(["git", "clone", "https://github.com/fchollet/ARC-AGI.git"], check=True)
    print("Датасет загружен!")
else:
    print("Датасет уже загружен.")
""")

    add_md("### 2. Подготовка DataLoader (Парсинг и Паддинг Grid-ов)")
    add_code("""import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader

class ARCDataset(Dataset):
    def __init__(self, json_dir, max_size=30):
        self.samples = []
        self.max_size = max_size
        
        # Загружаем все задачи из папки
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        for f in json_files:
            with open(f, 'r') as file:
                data = json.load(file)
                # Извлекаем все пары input-output из блока train
                for pair in data.get("train", []):
                    self.samples.append((pair["input"], pair["output"]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        inp_grid, out_grid = self.samples[idx]
        
        # ARC сетки имеют разные размеры (от 3x3 до 30x30).
        # Мы дополняем их нулями (черный цвет) до максимального размера 30x30,
        # чтобы сохранить пространственные координаты и пропорции.
        inp_tensor = torch.zeros(self.max_size, self.max_size, dtype=torch.long)
        out_tensor = torch.zeros(self.max_size, self.max_size, dtype=torch.long)
        
        h_in, w_in = len(inp_grid), len(inp_grid[0])
        h_out, w_out = len(out_grid), len(out_grid[0])
        
        for r in range(min(h_in, self.max_size)):
            for c in range(min(w_in, self.max_size)):
                inp_tensor[r, c] = int(inp_grid[r][c])
                
        for r in range(min(h_out, self.max_size)):
            for c in range(min(w_out, self.max_size)):
                out_tensor[r, c] = int(out_grid[r][c])
                
        # One-hot кодирование цветов (10 возможных цветов в ARC)
        inp_one_hot = torch.zeros(10, self.max_size, self.max_size)
        inp_one_hot.scatter_(0, inp_tensor.unsqueeze(0), 1.0)
        
        return inp_one_hot, out_tensor

train_ds = ARCDataset("ARC-AGI/data/training")
val_ds = ARCDataset("ARC-AGI/data/evaluation")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

print(f"Загружено задач обучения: {len(train_ds)}")
print(f"Загружено задач валидации: {len(val_ds)}")
""")

    add_md("### 3. Архитектура CIDAforARC (Сверточный Кодер + Дебатное Ядро + Сверточный Декодер)")
    add_code("""import torch.nn as nn
from cida_plugin.core import CIDAPlugin
from cida_plugin.config import CIDAPluginConfig

class ARCGridEncoder(nn.Module):
    \"\"\"
    Кодирует 2D-сетку ARC в компактный вектор признаков.
    \"\"\"
    def __init__(self, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (15x15)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (8x8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.fc = nn.Linear(128 * 4 * 4, d_model)
        
    def forward(self, x):
        return self.fc(self.conv(x))

class ARCGridDecoder(nn.Module):
    \"\"\"
    Декодирует консенсусное верование CIDA обратно в 2D-сетку ARC.
    \"\"\"
    def __init__(self, d_model=256):
        super().__init__()
        self.fc = nn.Linear(d_model, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (8x8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (16x16)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((30, 30)),
            nn.Conv2d(32, 10, kernel_size=3, padding=1) # Логиты по 10 классам цветов
        )
        
    def forward(self, h):
        return self.deconv(self.fc(h))

class CIDAforARC(nn.Module):
    def __init__(self, d_model=256, max_rounds=3):
        super().__init__()
        self.encoder = ARCGridEncoder(d_model)
        
        # Настройка когнитивного ядра CIDA
        self.cida_config = CIDAPluginConfig(
            d_input=d_model,
            d_hidden=d_model,
            num_classes=d_model,
            num_agents=4,
            max_rounds=max_rounds,
            multi_label=False
        )
        self.cida = CIDAPlugin(self.cida_config)
        self.decoder = ARCGridDecoder(d_model)
        
    def forward(self, x_in):
        h = self.encoder(x_in)
        out = self.cida(h)
        grid_logits = self.decoder(out["p_final"])
        return grid_logits, out
""")

    add_md("### 4. Обучение модели")
    add_code("""import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIDAforARC(d_model=256, max_rounds=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 20
scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader))

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_cells = 0
    total_cells = 0
    
    pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}")
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, out = model(imgs)
        
        # Считаем CrossEntropy по пикселям сетки
        loss_c = criterion(logits, targets)
        loss_disagreement = out.get("disagreement", torch.tensor(0.0)).mean()
        
        loss = loss_c + 0.1 * loss_disagreement
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        
        preds = logits.argmax(dim=1)
        correct_cells += (preds == targets).sum().item()
        total_cells += targets.numel()
        
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "Acc": f"{correct_cells/total_cells:.4f}"
        })
        
    avg_loss = running_loss / len(train_loader)
    train_acc = correct_cells / total_cells
    print(f"Эпоха {epoch+1} завершена. Avg Loss: {avg_loss:.4f} | Точность ячеек: {train_acc:.4f}")
""")

    add_md("### 5. Валидация модели")
    add_code("""model.eval()
val_correct = 0
val_total = 0

with torch.no_grad():
    for imgs, targets in tqdm(val_loader, desc="Валидация"):
        imgs, targets = imgs.to(device), targets.to(device)
        logits, _ = model(imgs)
        preds = logits.argmax(dim=1)
        val_correct += (preds == targets).sum().item()
        val_total += targets.numel()
        
val_acc = val_correct / val_total
print(f"\\n" + "="*50)
print(f"🏆 Результат на Валидации ARC-AGI | Точность ячеек: {val_acc:.4f}")
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
    
    with open("Kairos_ARC_Train.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    print("Создан ноутбук: Kairos_ARC_Train.ipynb")

if __name__ == "__main__":
    create_arc_notebook()
