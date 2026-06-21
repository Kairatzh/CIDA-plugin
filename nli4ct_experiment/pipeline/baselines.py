"""
Baselines
=========
Все конкуренты с которыми сравниваем CIDA.

1. SingleMLP           — самый простой, нет uncertainty
2. MCDropout           — Gal & Ghahramani 2016, стандарт uncertainty
3. TemperatureScaling  — post-hoc calibration поверх MLP
4. EvidenceAveraging   — наивный multi-evidence: просто усредняем
5. VotingEnsemble      — 4 независимых MLP, majority vote

Главный вопрос: кто лучше знает когда не знает?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


# ─── 1. Single MLP (BERT-like fine-tune имитация) ─────────────────────────────

class SingleMLP(nn.Module):
    """
    Имитирует fine-tuned BERT: pooled_output → classifier.
    Нет uncertainty. Всегда уверен. Классический baseline.
    """
    def __init__(self, d_input: int = 128, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, pooled: torch.Tensor) -> Dict:
        logits = self.net(pooled)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        
        return {
            'probs': probs,
            'logits': logits,
            'confidence': confidence,
            'uncertainty': 1.0 - confidence,  # наивная uncertainty
        }
    
    def predict(self, pooled):
        out = self.forward(pooled)
        return out['probs'].argmax(-1)


# ─── 2. MC Dropout (Bayesian Approximation) ───────────────────────────────────

class MCDropout(nn.Module):
    """
    Gal & Ghahramani 2016: dropout во время инференса = байесовская аппроксимация.
    Запускаем N forward pass, смотрим на дисперсию предсказаний.
    Это текущий стандарт для uncertainty в NLP.
    
    CIDA должен бить это по:
    - Качеству uncertainty (AUROC)
    - Скорости (MC Dropout нужно N forward pass)
    - Интерпретируемости (какой агент не согласен и почему)
    """
    
    def __init__(self, d_input: int = 128, num_classes: int = 2,
                 dropout: float = 0.3, n_samples: int = 20):
        super().__init__()
        self.n_samples = n_samples
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, pooled: torch.Tensor, n_samples: int = None) -> Dict:
        n = n_samples or self.n_samples
        
        # Dropout активен во время инференса!
        self.train()
        
        all_probs = []
        for _ in range(n):
            logits = self.net(pooled)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
        
        all_probs = torch.stack(all_probs)  # (n_samples, B, K)
        
        mean_probs = all_probs.mean(0)       # (B, K)
        var_probs = all_probs.var(0)         # (B, K) — epistemic uncertainty
        
        # Predictive entropy как uncertainty
        entropy = -(mean_probs * torch.log(mean_probs + 1e-9)).sum(-1)  # (B,)
        
        return {
            'probs': mean_probs,
            'uncertainty': entropy,
            'confidence': mean_probs.max(-1).values,
            'variance': var_probs.mean(-1),
            'all_samples': all_probs,
        }
    
    def predict(self, pooled):
        out = self.forward(pooled)
        return out['probs'].argmax(-1)


# ─── 3. Temperature Scaling ───────────────────────────────────────────────────

class TemperatureScaling(nn.Module):
    """
    Guo et al. 2017: один параметр температуры улучшает калибровку.
    Post-hoc метод: обучаем SingleMLP, потом на val находим T.
    
    Это самый честный baseline для калибровки — 
    простейший способ улучшить ECE без изменения архитектуры.
    """
    
    def __init__(self, base_model: SingleMLP):
        super().__init__()
        self.base = base_model
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, pooled: torch.Tensor) -> Dict:
        with torch.no_grad():
            logits = self.base.net(pooled)
        
        # Масштабируем логиты температурой
        scaled_logits = logits / self.temperature.clamp(min=0.1)
        probs = F.softmax(scaled_logits, dim=-1)
        confidence = probs.max(-1).values
        
        return {
            'probs': probs,
            'confidence': confidence,
            'uncertainty': 1.0 - confidence,
            'temperature': self.temperature.item(),
        }
    
    def calibrate(self, val_pooled: torch.Tensor, val_labels: torch.Tensor,
                  n_steps: int = 100):
        """Находит оптимальную температуру на val set"""
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=n_steps)
        
        def closure():
            optimizer.zero_grad()
            out = self.forward(val_pooled)
            loss = F.cross_entropy(torch.log(out['probs'] + 1e-9), val_labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        print(f"  Temperature calibrated: T = {self.temperature.item():.3f}")


# ─── 4. Evidence Averaging (наивный multi-evidence) ───────────────────────────

class EvidenceAveraging(nn.Module):
    """
    Наивный подход: кодируем каждый evidence отдельно,
    усредняем предсказания. Нет механизма взвешивания.
    
    Это важный baseline: показывает что просто усреднение
    не работает на contradictory evidence.
    """
    
    def __init__(self, d_input: int = 128, num_classes: int = 2):
        super().__init__()
        self.classifier = SingleMLP(d_input, num_classes)
    
    def forward(self, seq_output: torch.Tensor) -> Dict:
        """
        seq_output: (B, n_evidence, d_input)
        Классифицируем каждый evidence отдельно, потом усредняем.
        """
        B, n_ev, d = seq_output.shape
        
        flat = seq_output.view(B * n_ev, d)
        out = self.classifier(flat)
        probs_flat = out['probs']  # (B*n_ev, K)
        
        probs = probs_flat.view(B, n_ev, -1).mean(dim=1)  # (B, K)
        confidence = probs.max(-1).values
        
        return {
            'probs': probs,
            'confidence': confidence,
            'uncertainty': 1.0 - confidence,
        }
    
    def predict(self, seq_output):
        return self.forward(seq_output)['probs'].argmax(-1)


# ─── 5. Voting Ensemble ───────────────────────────────────────────────────────

class VotingEnsemble(nn.Module):
    """
    4 независимых MLP с разной инициализацией.
    Majority vote + disagreement как uncertainty.
    
    Этот baseline важен: если CIDA не лучше ансамбля,
    то мультиагентность не добавляет ничего нового.
    """
    
    def __init__(self, d_input: int = 128, num_classes: int = 2, n_members: int = 4):
        super().__init__()
        self.members = nn.ModuleList([
            SingleMLP(d_input, num_classes, dropout=0.1 + 0.05 * i)
            for i in range(n_members)
        ])
        self.n_members = n_members
    
    def forward(self, pooled: torch.Tensor) -> Dict:
        all_probs = []
        for member in self.members:
            out = member(pooled)
            all_probs.append(out['probs'])
        
        all_probs = torch.stack(all_probs)   # (M, B, K)
        mean_probs = all_probs.mean(0)        # (B, K)
        
        # Disagreement между членами ансамбля = uncertainty
        predictions = all_probs.argmax(-1)   # (M, B)
        # Доля "не согласных" голосов
        majority = mean_probs.argmax(-1)      # (B,)
        agreement = (predictions == majority.unsqueeze(0)).float().mean(0)  # (B,)
        disagreement = 1.0 - agreement
        
        # Entropy uncertainty
        entropy = -(mean_probs * torch.log(mean_probs + 1e-9)).sum(-1)
        
        return {
            'probs': mean_probs,
            'confidence': mean_probs.max(-1).values,
            'uncertainty': entropy,
            'disagreement': disagreement,
            'all_probs': all_probs,
        }
    
    def predict(self, pooled):
        return self.forward(pooled)['probs'].argmax(-1)


# ─── Training helper ──────────────────────────────────────────────────────────

def train_baseline(model: nn.Module, train_data: List[Tuple],
                   epochs: int = 50, lr: float = 3e-4,
                   label_smoothing: float = 0.05) -> List[float]:
    """
    train_data: list of (pooled_tensor, seq_tensor, label_tensor)
    Возвращает историю loss
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for pooled, seq, labels in train_data:
            # Выбираем правильный входной тензор
            if isinstance(model, EvidenceAveraging):
                out = model(seq)
            else:
                out = model(pooled)
            
            probs = out['probs']
            loss = F.cross_entropy(
                torch.log(probs + 1e-9),
                labels,
                label_smoothing=label_smoothing
            )
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        losses.append(epoch_loss / len(train_data))
    
    return losses