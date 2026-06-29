import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

os.environ["FORCE_FALLBACK"] = "1"

import importlib
encoder_mod = importlib.import_module("KairosMM.01_encoder_biomedclip")
BiomedCLIPCIDA = encoder_mod.BiomedCLIPCIDA
get_transforms_v2 = encoder_mod.get_transforms_v2
NIH_CLASSES = encoder_mod.NIH_CLASSES
from KairosMM.train_and_benchmark_subset import SubsetDataset, pre_cache_features

device = torch.device("cpu")
print("Loading data...")
data_dir = "data/chest_xray_subset"
csv_path = os.path.join(data_dir, "metadata.csv")
images_dir = os.path.join(data_dir, "images")

df = pd.read_csv(csv_path)
df_pos = df[df["Class_6"] == 1.0]
df_neg = df[df["Class_6"] == 0.0]
train_df = pd.concat([df_pos.iloc[:40], df_neg.iloc[:40]]).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = pd.concat([df_pos.iloc[40:], df_neg.iloc[40:]]).sample(frac=1, random_state=42).reset_index(drop=True)

train_dataset = SubsetDataset(train_df, images_dir, split="train")
val_dataset = SubsetDataset(val_df, images_dir, split="val")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print("Initializing encoder...")
encoder_model = BiomedCLIPCIDA(num_classes=14, d_hidden=768, freeze_encoder=True)
encoder_model.to(device)

print("Caching training features...")
t0 = time.time()
train_cached = pre_cache_features(encoder_model, train_loader, device)
print(f"Train cached in {time.time() - t0:.2f}s")

print("Caching validation features...")
t0 = time.time()
val_cached = pre_cache_features(encoder_model, val_loader, device)
print(f"Val cached in {time.time() - t0:.2f}s")

del encoder_model

# Let's run a profiled version of train_and_eval
train_pooled, train_spatial, train_labels = train_cached
val_pooled, val_spatial, val_labels = val_cached

print("Initializing training model...")
model = BiomedCLIPCIDA(num_classes=14, d_hidden=768, freeze_encoder=True)
model.cida.config.max_rounds = 3

optimizer = torch.optim.AdamW(model.cida.parameters(), lr=2e-3, weight_decay=1e-4)
criterion = nn.BCELoss()

train_ds = torch.utils.data.TensorDataset(train_pooled, train_spatial, train_labels)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

print("Starting epoch timing...")
for epoch in range(5):
    t0 = time.time()
    model.train()
    for batch_idx, (p, s, l) in enumerate(train_loader):
        tb = time.time()
        optimizer.zero_grad()
        out = model.cida(p, seq_output=s)
        t_fwd = time.time() - tb
        
        loss = criterion(out["p_final"], l)
        loss.backward()
        optimizer.step()
        t_step = time.time() - tb
        print(f"  Epoch {epoch}, Batch {batch_idx}: forward={t_fwd:.4f}s, total_step={t_step:.4f}s")
    
    t_epoch = time.time() - t0
    print(f"Epoch {epoch} completed in {t_epoch:.4f} seconds")
