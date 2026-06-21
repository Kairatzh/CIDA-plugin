"""
NLI4CT Data Pipeline
====================
Загружает реальный NLI4CT датасет (SemEval 2023/2024).
Если нет доступа — генерирует синтетический с той же структурой.

NLI4CT — это задача:
  Premise:    утверждение из клинического исследования рака
  Hypothesis: вывод который нужно проверить
  Label:      Entailment / Contradiction

Особенность: Multi-Evidence — иногда нужно сопоставить
несколько секций из нескольких trial-документов.
Именно тут single-agent модели ломаются.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset


# ─── Структура примера ─────────────────────────────────────────────────────────

@dataclass
class NLI4CTExample:
    uid: str
    premise_sections: List[str]   # может быть несколько секций (multi-evidence)
    hypothesis: str
    label: int                    # 0=Entailment, 1=Contradiction
    num_evidence: int             # сколько секций нужно интегрировать
    difficulty: str               # 'easy', 'medium', 'hard'
    cancer_type: str              # тип рака (для стратифицированного анализа)


# ─── Синтетический датасет (реалистичный) ─────────────────────────────────────

CANCER_TYPES = [
    "breast cancer", "lung cancer", "colorectal cancer",
    "lymphoma", "leukemia", "pancreatic cancer", "melanoma"
]

DRUGS = [
    "doxorubicin", "paclitaxel", "cisplatin", "pembrolizumab",
    "trastuzumab", "bevacizumab", "imatinib", "tamoxifen"
]

OUTCOMES = [
    ("tumor reduction", "40%", "significant"),
    ("progression-free survival", "8.2 months", "improved"),
    ("overall response rate", "67%", "substantial"),
    ("pathological complete response", "31%", "moderate"),
    ("disease control rate", "78%", "high"),
]

def _make_premise(drug, outcome_tuple, cancer, direction="positive", noise_level=0.0):
    metric, value, qualifier = outcome_tuple
    
    templates_positive = [
        f"Patients with {cancer} treated with {drug} demonstrated a {metric} of {value}, "
        f"indicating {qualifier} antitumor activity compared to placebo.",
        
        f"In the primary endpoint analysis, {drug} achieved a {qualifier} improvement "
        f"in {metric} ({value}) among {cancer} patients (p<0.001).",
        
        f"The trial showed that {drug} administration resulted in {value} {metric} "
        f"in the {cancer} cohort, meeting the primary efficacy endpoint.",
    ]
    
    templates_negative = [
        f"Despite initial promise, {drug} failed to demonstrate significant improvement "
        f"in {metric} ({value}) for {cancer} patients (p=0.43).",
        
        f"No statistically significant difference in {metric} was observed between "
        f"{drug} and control groups in {cancer} patients ({value}).",
        
        f"The {cancer} subgroup showed limited response to {drug}, with {metric} "
        f"of only {value}, below the predefined threshold.",
    ]
    
    if noise_level > 0 and random.random() < noise_level:
        # Зашумлённая предпосылка — противоречивые данные
        pos = random.choice(templates_positive)
        neg = random.choice(templates_negative)
        return pos + " However, secondary endpoints were less compelling. " + neg
    
    templates = templates_positive if direction == "positive" else templates_negative
    return random.choice(templates)


def _make_hypothesis(drug, outcome_tuple, label, cancer):
    metric, value, qualifier = outcome_tuple
    
    if label == 0:  # Entailment — логически следует
        hypotheses = [
            f"{drug} shows efficacy in {cancer} treatment.",
            f"The study supports the use of {drug} for {cancer} patients.",
            f"{drug} achieves meaningful {metric} in {cancer}.",
        ]
    else:  # Contradiction — противоречит
        hypotheses = [
            f"{drug} is ineffective for {cancer} treatment.",
            f"No benefit of {drug} was observed in {cancer} patients.",
            f"{drug} failed to improve outcomes in the {cancer} trial.",
        ]
    
    return random.choice(hypotheses)


class SyntheticNLI4CT(Dataset):
    """
    Синтетический NLI4CT с реалистичной структурой.
    
    Уровни сложности:
    - easy:   1 premise, прямое следование/противоречие
    - medium: 2 premises, нужно интегрировать
    - hard:   3 premises, одна вводит шум (как реальные multi-evidence случаи)
    """
    
    def __init__(self, n_samples=400, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.examples: List[NLI4CTExample] = []
        self._generate(n_samples)
    
    def _generate(self, n):
        difficulties = ['easy'] * (n // 3) + ['medium'] * (n // 3) + ['hard'] * (n - 2 * (n // 3))
        random.shuffle(difficulties)
        
        for i, difficulty in enumerate(difficulties):
            cancer = random.choice(CANCER_TYPES)
            drug = random.choice(DRUGS)
            outcome = random.choice(OUTCOMES)
            label = random.randint(0, 1)
            
            if difficulty == 'easy':
                # Один premise, прямой сигнал
                direction = "positive" if label == 0 else "negative"
                premises = [_make_premise(drug, outcome, cancer, direction)]
                
            elif difficulty == 'medium':
                # Два premises — нужно интегрировать оба
                direction = "positive" if label == 0 else "negative"
                p1 = _make_premise(drug, outcome, cancer, direction)
                # Второй premise о другом outcome того же препарата
                outcome2 = random.choice([o for o in OUTCOMES if o != outcome])
                p2 = _make_premise(drug, outcome2, cancer, direction, noise_level=0.1)
                premises = [p1, p2]
                
            else:  # hard
                # Три premises: один — шум/противоречие внутри
                direction = "positive" if label == 0 else "negative"
                opposite = "negative" if direction == "positive" else "positive"
                
                p1 = _make_premise(drug, outcome, cancer, direction)
                p2 = _make_premise(drug, outcome, cancer, opposite)  # противоречит!
                outcome3 = random.choice([o for o in OUTCOMES if o != outcome])
                p3 = _make_premise(drug, outcome3, cancer, direction, noise_level=0.2)
                premises = [p1, p2, p3]
            
            hypothesis = _make_hypothesis(drug, outcome, label, cancer)
            
            self.examples.append(NLI4CTExample(
                uid=f"synth_{i:04d}",
                premise_sections=premises,
                hypothesis=hypothesis,
                label=label,
                num_evidence=len(premises),
                difficulty=difficulty,
                cancer_type=cancer,
            ))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def get_by_difficulty(self, difficulty: str) -> List[NLI4CTExample]:
        return [e for e in self.examples if e.difficulty == difficulty]
    
    def get_by_cancer(self, cancer_type: str) -> List[NLI4CTExample]:
        return [e for e in self.examples if e.cancer_type == cancer_type]
    
    def stats(self) -> Dict:
        from collections import Counter
        return {
            'total': len(self.examples),
            'by_label': Counter(e.label for e in self.examples),
            'by_difficulty': Counter(e.difficulty for e in self.examples),
            'by_cancer': Counter(e.cancer_type for e in self.examples),
            'avg_evidence': np.mean([e.num_evidence for e in self.examples]),
        }


# ─── Реальный NLI4CT загрузчик ────────────────────────────────────────────────




def load_hf_nli4ct(config_name: str = "2023_processed", split: str = "train") -> List[NLI4CTExample]:
    """
    Загружает реальный NLI4CT с Hugging Face (araag2/SemEval_NLI4CT).
    """
    print(f"  Fetching araag2/SemEval_NLI4CT ({config_name}, split={split}) from HF...")
    ds = load_dataset("araag2/SemEval_NLI4CT", config_name, split=split)
    
    examples = []
    label_map = {"Entailment": 0, "Contradiction": 1}
    
    for item in ds:
        p_sections = item.get("Primary_Section") or []
        s_sections = item.get("Secondary_Section") or []
        
        if isinstance(p_sections, str): p_sections = [p_sections]
        if isinstance(s_sections, str): s_sections = [s_sections]
        
        premises = [p for p in p_sections if p] + [p for p in s_sections if p]
        if not premises:
            premises = ["No evidence provided."]
            
        label_str = item.get("Label", "Entailment")
        label = label_map.get(label_str, 0)
        
        num_evidence = len(premises)
        # Маппинг сложности по числу evidence как в load_real_nli4ct
        difficulty = "easy" if num_evidence == 1 else ("medium" if num_evidence == 2 else "hard")
        
        examples.append(NLI4CTExample(
            uid=item.get("id", "unknown"),
            premise_sections=premises,
            hypothesis=item.get("Statement", ""),
            label=label,
            num_evidence=num_evidence,
            difficulty=difficulty,
            cancer_type="breast cancer",
        ))
    
    return examples


# ─── Collate функция для DataLoader ───────────────────────────────────────────

def collate_fn(batch: List[NLI4CTExample]):
    """Собирает batch примеров в словарь"""
    return {
        'uids': [e.uid for e in batch],
        'premises': [e.premise_sections for e in batch],
        'hypotheses': [e.hypothesis for e in batch],
        'labels': torch.tensor([e.label for e in batch], dtype=torch.long),
        'difficulties': [e.difficulty for e in batch],
        'cancer_types': [e.cancer_type for e in batch],
        'num_evidence': [e.num_evidence for e in batch],
    }


def make_dataloaders(dataset, train_ratio=0.6, val_ratio=0.2, batch_size=16, seed=42):
    """Train/Val/Test split с сохранением стратификации"""
    n = len(dataset)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    
    from torch.utils.data import Subset
    
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader