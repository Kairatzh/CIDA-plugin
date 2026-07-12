import torch
import torch.nn as nn
import torch.optim as optim
from cida_plugin import CIDAPlugin, CIDAPluginConfig, CIDALoss

def test():
    # Make a simple linearly separable dataset
    # Input has 64 dimensions. Feature 0 determines the class.
    # Class 0: feature 0 > 0.5
    # Class 1: feature 0 < -0.5
    N = 1000
    X = torch.randn(N, 64)
    # Filter to make it very clean
    mask0 = X[:, 0] > 0.5
    mask1 = X[:, 0] < -0.5
    X0 = X[mask0]
    X1 = X[mask1]
    
    # Balance classes
    min_len = min(len(X0), len(X1))
    X0 = X0[:min_len]
    X1 = X1[:min_len]
    
    X = torch.cat([X0, X1], dim=0)
    Y = torch.cat([torch.zeros(min_len, dtype=torch.long), torch.ones(min_len, dtype=torch.long)], dim=0)
    
    # Shuffle
    perm = torch.randperm(X.size(0))
    X = X[perm]
    Y = Y[perm]
    
    # Standard CIDA model
    cfg = CIDAPluginConfig(d_input=64, d_hidden=32, num_classes=2, max_rounds=2)
    model = CIDAPlugin(cfg)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = CIDALoss(lambda_cal=0.2, lambda_ac=0.1)
    
    # Train for 20 epochs with batch size 32
    batch_size = 32
    for epoch in range(20):
        model.train()
        epoch_loss = 0.0
        task_loss_sum = 0.0
        cal_loss_sum = 0.0
        ac_loss_sum = 0.0
        correct = 0
        total = 0
        for i in range(0, X.size(0), batch_size):
            bx = X[i:i+batch_size]
            by = Y[i:i+batch_size]
            
            optimizer.zero_grad()
            out = model(bx)
            loss, comp = loss_fn(out["p_final"], by, out["b_all"], out.get("L_orth", None))
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * bx.size(0)
            task_loss_sum += comp["task"] * bx.size(0)
            cal_loss_sum += comp["calibration"] * bx.size(0)
            ac_loss_sum += comp["anti_collapse"] * bx.size(0)
            preds = torch.argmax(out["p_final"], dim=-1)
            correct += (preds == by).sum().item()
            total += bx.size(0)
            
        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/total:.4f} (Task: {task_loss_sum/total:.4f}, Cal: {cal_loss_sum/total:.4f}, AC: {ac_loss_sum/total:.4f}) | Acc: {correct/total:.4f}")

if __name__ == "__main__":
    test()
