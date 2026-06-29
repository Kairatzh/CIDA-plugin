import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np

import importlib
sota_mod = importlib.import_module("KairosMM.11_sota_encoder_cida")
SOTAEncoderCIDA = sota_mod.SOTAEncoderCIDA

def test_overfit():
    print("Starting overfit proof of concept...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Модель
    model = SOTAEncoderCIDA(model_name='efficientnet_b0', num_classes=5, d_hidden=128, max_rounds=2, freeze_encoder=False)
    model.to(device)

    # 2. Фейковые данные (20 картинок)
    batch_size = 20
    torch.manual_seed(42)
    imgs = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Случайные метки (0 или 1) для 5 классов
    labels = torch.randint(0, 2, (batch_size, 5), dtype=torch.float32).to(device)
    pos_weight = torch.tensor([1.0] * 5).to(device)

    # 3. Новая (ПРАВИЛЬНАЯ) функция потерь
    def weighted_loss(probs, targets):
        probs = probs.clamp(1e-6, 1 - 1e-6)
        loss = -pos_weight * targets * torch.log(probs) - (1 - targets) * torch.log(1 - probs)
        return loss.mean()

    # 4. Оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=2e-3)

    print("\\nStarting overfit (25 epochs). Goal: Loss -> 0, AUC -> 1.0")
    print("-" * 50)
    
    for epoch in range(25):
        model.train()
        optimizer.zero_grad()
        
        out = model(x_224=imgs)
        preds = out["p_final"]
        
        loss_c = weighted_loss(preds, labels)
        loss_disagreement = out.get("disagreement", torch.tensor(0.0)).mean()
        loss = loss_c + 0.1 * loss_disagreement
        
        loss.backward()
        optimizer.step()
        
        # Считаем AUC на этом же батче
        model.eval()
        with torch.no_grad():
            out_eval = model(x_224=imgs)
            p_numpy = out_eval["p_final"].cpu().numpy()
            l_numpy = labels.cpu().numpy()
            
            aucs = []
            for i in range(5):
                y_t = l_numpy[:, i]
                y_p = p_numpy[:, i]
                if y_t.sum() > 0 and y_t.sum() < len(y_t):
                    try:
                        aucs.append(roc_auc_score(y_t, y_p))
                    except ValueError:
                        pass
            
            mean_auc = np.mean(aucs) if aucs else 0.5
            
        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Avg ROC-AUC: {mean_auc:.4f}")
        
        if mean_auc > 0.99 and loss.item() < 0.1:
            print("\\nSUCCESS! Model perfectly memorized data. Gradients flow correctly!")
            break
            
if __name__ == "__main__":
    test_overfit()
