"""
Encoder Pipeline
================
BioMedBERT → pooled_output для CIDA Plugin.

Ключевая особенность NLI4CT: multi-evidence.
Нужно энкодировать [premise_1; premise_2; ... ; hypothesis]
и получить один вектор который несёт всю информацию.

Три стратегии pooling:
1. CLS  — стандартный BERT pooling
2. Mean — среднее по токенам (лучше для длинных текстов)
3. Multi-Evidence — каждый premise отдельно, потом агрегация
   (это то что CIDA использует через seq_output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import sys

# Проверяем есть ли transformers
try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ─── Лёгкий энкодер (без transformers, для быстрого теста) ───────────────────

class LightweightEncoder(nn.Module):
    """
    TF-IDF вдохновлённый энкодер без BERT.
    Используется когда нет GPU или для быстрого прототипирования.
    
    Даёт правдоподобный d=128 вектор из текста через:
    character n-gram hashing → linear projection → LayerNorm
    """
    
    def __init__(self, d_output: int = 128, vocab_size: int = 8192):
        super().__init__()
        self.d_output = d_output
        self.vocab_size = vocab_size
        
        self.proj = nn.Sequential(
            nn.Linear(vocab_size, 512),
            nn.SiLU(),
            nn.Linear(512, d_output),
            nn.LayerNorm(d_output),
        )
    
    def _text_to_vector(self, text: str) -> torch.Tensor:
        """Character 3-gram hashing trick"""
        vec = torch.zeros(self.vocab_size)
        text = text.lower()
        
        # Trigrams
        for i in range(len(text) - 2):
            ngram = text[i:i+3]
            idx = hash(ngram) % self.vocab_size
            vec[idx] += 1.0
        
        # Унормализуем
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """texts → (B, vocab_size)"""
        vecs = [self._text_to_vector(t) for t in texts]
        return torch.stack(vecs)
    
    def forward(
        self,
        premises: List[List[str]],
        hypotheses: List[str],
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает:
            pooled:     (B, d_output)   — для CIDAPlugin.pooled
            seq_output: (B, n_ev, d_output) — для CIDAPlugin.seq_output
        """
        if device is None:
            device = next(self.parameters()).device
        
        B = len(hypotheses)
        max_evidence = max(len(p) for p in premises)
        
        pooled_list = []
        seq_list = []
        
        for i in range(B):
            hyp = hypotheses[i]
            prem_list = premises[i]
            
            # Кодируем каждый premise отдельно
            ev_vecs = self.encode_texts(prem_list).to(device)     # (n_ev, vocab)
            hyp_vec = self.encode_texts([hyp]).to(device)         # (1, vocab)
            
            # Проецируем
            ev_proj = self.proj(ev_vecs)    # (n_ev, d_output)
            hyp_proj = self.proj(hyp_vec)   # (1, d_output)
            
            # Pooled = среднее evidence + hypothesis
            pooled = torch.cat([ev_proj, hyp_proj], dim=0).mean(0)  # (d_output,)
            pooled_list.append(pooled)
            
            # Seq: паддим до max_evidence + 1 (hypothesis)
            if ev_proj.size(0) < max_evidence:
                pad = torch.zeros(
                    max_evidence - ev_proj.size(0), self.d_output, device=device
                )
                ev_proj = torch.cat([ev_proj, pad], dim=0)
            
            full_seq = torch.cat([ev_proj, hyp_proj], dim=0)  # (max_ev+1, d)
            seq_list.append(full_seq)
        
        pooled_out = torch.stack(pooled_list)   # (B, d_output)
        seq_out = torch.stack(seq_list)         # (B, max_ev+1, d_output)
        
        return pooled_out, seq_out


# ─── BioMedBERT энкодер (реальный) ───────────────────────────────────────────

class BioMedBERTEncoder(nn.Module):
    """
    microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
    
    Специально обученный на PubMed + PMC full texts.
    Понимает клинические термины, названия препаратов, статистику.
    
    Стратегия для multi-evidence NLI:
    Каждый premise кодируется вместе с hypothesis:
        [CLS] hypothesis [SEP] premise_i [SEP]
    
    Это даёт модели контекст гипотезы при чтении каждого evidence.
    Потом K векторов объединяются через learned aggregation.
    """
    
    MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    
    def __init__(
        self,
        model_name: str = None,
        d_output: int = 128,
        max_length: int = 128,
        pooling: str = "cls",  # "cls", "mean", "max"
        freeze_layers: int = 8,  # замораживаем первые N слоёв
    ):
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("pip install transformers")
        
        self.model_name = model_name or self.MODEL_NAME
        self.d_output = d_output
        self.max_length = max_length
        self.pooling = pooling
        
        print(f"  Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)
        
        bert_dim = self.bert.config.hidden_size  # 768
        
        # Замораживаем первые N слоёв для стабильности
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        
        # Проекция 768 → d_output
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, d_output * 2),
            nn.SiLU(),
            nn.Linear(d_output * 2, d_output),
            nn.LayerNorm(d_output),
        )
        
        # Агрегация нескольких evidence векторов
        self.evidence_aggregator = nn.MultiheadAttention(
            embed_dim=d_output,
            num_heads=4,
            batch_first=True,
        )
        
        print(f"  Encoder ready: {bert_dim} → {d_output}")
    
    def _encode_pair(self, hypothesis: str, premise: str, device) -> torch.Tensor:
        """Кодирует одну пару (hypothesis, premise) → (d_output,)"""
        tokens = self.tokenizer(
            hypothesis,
            premise,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            out = self.bert(**tokens)
        
        if self.pooling == "cls":
            hidden = out.last_hidden_state[:, 0, :]  # (1, 768)
        elif self.pooling == "mean":
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            hidden = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        else:  # max
            hidden = out.last_hidden_state.max(dim=1).values
        
        return self.proj(hidden).squeeze(0)  # (d_output,)
    
    def forward(
        self,
        premises: List[List[str]],
        hypotheses: List[str],
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Оптимизированный forward: батчит все пары (hyp, prem) вместе.
        """
        if device is None:
            device = next(self.parameters()).device
        
        B = len(hypotheses)
        max_ev = max(len(p) for p in premises)
        
        # 1. Формируем плоский список всех пар для кодирования
        flat_hyps = []
        flat_prems = []
        offsets = [0]
        
        for i in range(B):
            hyp = hypotheses[i]
            for prem in premises[i]:
                flat_hyps.append(hyp)
                flat_prems.append(prem)
            offsets.append(len(flat_hyps))
        
        # 2. Кодируем всё одним батчем (или частями если слишком много)
        if not flat_hyps:
            # Пустой батч? Возвращаем нули
            return torch.zeros(B, self.d_output, device=device), torch.zeros(B, max_ev, self.d_output, device=device)
            
        # Используем tokenizer с padding
        tokens = self.tokenizer(
            flat_hyps,
            flat_prems,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            # Если батч слишком большой для CPU/RAM, можно разбить, 
            # но для 32 * ~3 = ~100 пар должно быть ок.
            out = self.bert(**tokens)
        
        if self.pooling == "cls":
            hidden = out.last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            hidden = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        else:
            hidden = out.last_hidden_state.max(dim=1).values
            
        all_vecs = self.proj(hidden) # (Total_Pairs, d_output)
        
        # 3. Собираем обратно
        all_pooled = []
        all_seq = []
        
        for i in range(B):
            start, end = offsets[i], offsets[i+1]
            ev_vecs = all_vecs[start:end] # (n_ev, d_output)
            
            # Seq output с паддингом
            n_ev = ev_vecs.size(0)
            if n_ev < max_ev:
                pad = torch.zeros(max_ev - n_ev, self.d_output, device=device)
                seq = torch.cat([ev_vecs, pad], dim=0)
            else:
                seq = ev_vecs
            all_seq.append(seq)
            
            # Pooled через attention
            ev_unsq = ev_vecs.unsqueeze(0) # (1, n_ev, d)
            agg, _ = self.evidence_aggregator(ev_unsq, ev_unsq, ev_unsq)
            pooled = agg.squeeze(0).mean(0)
            all_pooled.append(pooled)
            
        return torch.stack(all_pooled), torch.stack(all_seq)


# ─── Фабрика энкодеров ────────────────────────────────────────────────────────

def make_encoder(
    encoder_type: str = "lightweight",
    d_output: int = 128,
    **kwargs,
) -> nn.Module:
    """
    encoder_type: "lightweight" | "biomedbert"
    
    Для первых экспериментов используй lightweight.
    Для финальных результатов — biomedbert.
    """
    if encoder_type == "lightweight":
        return LightweightEncoder(d_output=d_output)
    
    elif encoder_type == "biomedbert":
        if not HAS_TRANSFORMERS:
            print("  WARNING: transformers not installed, falling back to lightweight")
            return LightweightEncoder(d_output=d_output)
        return BioMedBERTEncoder(d_output=d_output, **kwargs)
    
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")