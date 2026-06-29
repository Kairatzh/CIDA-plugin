import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

os.environ["FORCE_FALLBACK"] = "1"

import importlib
encoder_mod = importlib.import_module("KairosMM.01_encoder_biomedclip")
BiomedCLIPCIDA = encoder_mod.BiomedCLIPCIDA
NIH_CLASSES = encoder_mod.NIH_CLASSES

from KairosMM.train_and_benchmark_subset import SubsetDataset, pre_cache_features, expected_calibration_error

device = torch.device("cpu")
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

print("Pre-caching features...")
encoder_model = BiomedCLIPCIDA(num_classes=14, d_hidden=768, freeze_encoder=True)
encoder_model.to(device)
train_cached = pre_cache_features(encoder_model, train_loader, device)
val_cached = pre_cache_features(encoder_model, val_loader, device)
del encoder_model

train_pooled, train_spatial, train_labels = train_cached
val_pooled, val_spatial, val_labels = val_cached

p_idx = NIH_CLASSES.index("Pneumonia")
y_true = val_df[f"Class_{p_idx}"].values

# Grid Search
results = []
lrs = [5e-4, 1e-3, 2e-3, 5e-3]
wds = [1e-4, 1e-3, 1e-2, 1e-1]
epochs_list = [10, 15, 20]

for lr in lrs:
    for wd in wds:
        for epochs in epochs_list:
            for max_rounds in [0, 3]:
                # Train
                model = BiomedCLIPCIDA(num_classes=14, d_hidden=768, freeze_encoder=True)
                model.cida.config.max_rounds = max_rounds
                
                optimizer = torch.optim.AdamW(model.cida.parameters(), lr=lr, weight_decay=wd)
                criterion = nn.BCELoss()
                
                train_ds = torch.utils.data.TensorDataset(train_pooled, train_spatial, train_labels)
                tr_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
                
                best_auc = 0.0
                best_preds = None
                
                for epoch in range(epochs):
                    model.train()
                    for p, s, l in tr_loader:
                        optimizer.zero_grad()
                        out = model.cida(p, seq_output=s)
                        loss = criterion(out["p_final"], l)
                        loss.backward()
                        optimizer.step()
                        
                    model.eval()
                    with torch.no_grad():
                        out = model.cida(val_pooled, seq_output=val_spatial)
                        val_preds = out["p_final"].cpu().numpy()
                        
                    y_prob_p = val_preds[:, p_idx]
                    try:
                        auc = roc_auc_score(y_true, y_prob_p)
                    except Exception:
                        auc = 0.5
                    if auc > best_auc:
                        best_auc = auc
                        best_preds = val_preds
                
                ece = expected_calibration_error(y_true, best_preds[:, p_idx])
                results.append({
                    "max_rounds": max_rounds,
                    "lr": lr,
                    "wd": wd,
                    "epochs": epochs,
                    "auc": best_auc,
                    "ece": ece
                })
                print(f"Rounds: {max_rounds}, LR: {lr}, WD: {wd}, Epochs: {epochs} -> AUC: {best_auc:.4f}, ECE: {ece:.4f}")

# Find best for Baseline (max_rounds=0) and CIDA (max_rounds=3)
baseline_res = [r for r in results if r["max_rounds"] == 0]
cida_res = [r for r in results if r["max_rounds"] == 3]

best_baseline = max(baseline_res, key=lambda x: x["auc"])
best_cida = max(cida_res, key=lambda x: x["auc"])

print("\n=== BEST HYPERPARAMETERS ===")
print("Baseline:", best_baseline)
print("CIDA:", best_cida)
