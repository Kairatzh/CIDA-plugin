import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def test_bce():
    # Симулируем 100% уверенную модель в FP16, которая вызывает баг с округлением до 1.0
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16, requires_grad=True)
    targets = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    pos_weight = torch.tensor([2.0, 1.5], dtype=torch.float32)
    
    print("Test 1: Manual BCE with clamp (The Buggy Version)")
    try:
        with autocast():
            # Это то, что было в Колабе
            p_clamp = probs.clamp(1e-6, 1 - 1e-6)
            loss1 = -pos_weight * targets * torch.log(p_clamp) - (1 - targets) * torch.log(1 - p_clamp)
            loss1 = loss1.mean()
        print(f"Manual Loss: {loss1.item()}")
    except Exception as e:
        print(f"Manual BCE Failed: {e}")

    print("\\nTest 2: PyTorch BCE with weight mask (The Fix)")
    try:
        with autocast():
            weight_mask = targets * pos_weight + (1 - targets)
            loss2 = F.binary_cross_entropy(probs.float(), targets, weight=weight_mask)
        print(f"PyTorch BCE Loss: {loss2.item()}")
    except Exception as e:
        print(f"PyTorch BCE Failed: {e}")

if __name__ == "__main__":
    test_bce()
