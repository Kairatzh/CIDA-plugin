import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

# Импортируем CIDA
from cida_plugin.core import CIDAPlugin
from cida_plugin.config import CIDAPluginConfig

# 1. Словарь и шаблоны
vocab = ["[PAD]", "[MASK]", "A", "lives", "in", "the", "and", "eats", 
         "fish", "bird", "lion", "camel", 
         "water", "forest", "savanna", "desert", 
         "algae", "seeds", "meat", "grass"]

vocab_map = {word: i for i, word in enumerate(vocab)}
id_map = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

# Корпус валидных предложений
valid_triplets = [
    ("fish", "water", "algae"),
    ("bird", "forest", "seeds"),
    ("lion", "savanna", "meat"),
    ("camel", "desert", "grass")
]

def make_sentence(animal, habitat, food):
    return ["A", animal, "lives", "in", "the", habitat, "and", "eats", food]

def get_sentence_ids(sentence):
    return torch.tensor([vocab_map[w] for w in sentence], dtype=torch.long)

# 2. Модели
class VanillaDiffusionModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.embedding(x) + self.pos_embedding(positions)
        h = self.transformer(h)
        logits = self.proj(h)
        return logits

class CognitiveDiffusionModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, d_hidden=64, max_len=9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        
        # Настраиваем CIDA на уровне всей последовательности
        # Классифицируем все токены сразу: num_classes = max_len * vocab_size
        self.config = CIDAPluginConfig(
            d_input=d_model,
            d_hidden=d_hidden,
            num_classes=max_len * vocab_size,
            multi_label=False,
            max_rounds=2
        )
        self.cida = CIDAPlugin(self.config)
        self.max_len = max_len
        self.vocab_size = vocab_size
        
    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.embedding(x) + self.pos_embedding(positions)
        h = self.transformer(h)
        
        # CIDA работает на пулированном векторе всей последовательности, 
        # но видит все токены через cross-attention (seq_output=h)
        pooled = h.mean(dim=1)  # (B, d_model)
        out = self.cida(pooled=pooled, seq_output=h)
        p_final = out["p_final"]  # (B, L * vocab_size)
        
        # Решейпим вероятности в (B, L, vocab_size)
        p_final = p_final.view(B, L, self.vocab_size)
        
        # Переводим в логиты
        logits = torch.log(p_final.clamp(1e-10, 1.0))
        return logits

# 3. Генерация обучающих данных с маскированием
def generate_batch(batch_size=32):
    inputs = []
    targets = []
    for _ in range(batch_size):
        # Выбираем случайное валидное животное
        triplet = random.choice(valid_triplets)
        clean_sentence = make_sentence(*triplet)
        clean_ids = get_sentence_ids(clean_sentence)
        
        # Случайно маскируем от 1 до 3 ключевых позиций (1, 5, 8)
        masked_ids = clean_ids.clone()
        mask_indices = [1, 5, 8]
        num_masks = random.randint(1, 3)
        to_mask = random.sample(mask_indices, num_masks)
        for idx in to_mask:
            masked_ids[idx] = vocab_map["[MASK]"]
            
        inputs.append(masked_ids)
        targets.append(clean_ids)
        
    return torch.stack(inputs), torch.stack(targets)

# 4. Обучение
def train_model(model, epochs=20, steps_per_epoch=50):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            
            x, y = generate_batch(64)
            logits = model(x)
            
            # Считаем лосс только по маскированным позициям для честности
            mask = (x == vocab_map["[MASK]"])
            if mask.sum() > 0:
                loss = criterion(logits[mask], y[mask])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        print(f"  Epoch {epoch+1:02d} | Loss: {epoch_loss/steps_per_epoch:.4f}", flush=True)
            
# 5. Тестирование когерентности генерации
def evaluate_coherence(model, name, num_trials=100):
    model.eval()
    valid_count = 0
    incoherent_count = 0
    
    with torch.no_grad():
        for _ in range(num_trials):
            # Начинаем с предложения, где все ключевые слова замаскированы
            # "A [MASK] lives in the [MASK] and eats [MASK]"
            current_sentence = make_sentence("[MASK]", "[MASK]", "[MASK]")
            x = get_sentence_ids(current_sentence).unsqueeze(0)
            
            # 3 шага диффузии (параллельное дешумирование)
            for _ in range(3):
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                
                # Обновляем маскированные токены наиболее вероятными значениями
                for idx in [1, 5, 8]:
                    pred_id = probs[0, idx].argmax().item()
                    x[0, idx] = pred_id
            
            # Проверяем результат
            gen_words = [id_map[idx.item()] for idx in x[0]]
            animal, habitat, food = gen_words[1], gen_words[5], gen_words[8]
            
            if _ < 5:
                print(f"  Сгенерировано: {' '.join(gen_words)}")
            
            # Проверяем, совпадает ли сгенерированная тройка с валидными
            is_valid = False
            for v_a, v_h, v_f in valid_triplets:
                if animal == v_a and habitat == v_h and food == v_f:
                    is_valid = True
                    break
            
            if is_valid:
                valid_count += 1
            else:
                incoherent_count += 1
                
    accuracy = (valid_count / num_trials) * 100
    print(f"[{name}] Generation Coherence Accuracy: {accuracy:.2f}% | Semantic Errors: {incoherent_count}")
    return accuracy

if __name__ == "__main__":
    print("Starting Experiment: Cognitive Diffusion vs Vanilla Diffusion")
    print("Task: generate semantically coherent text in parallel (3 steps)")
    print("-" * 60)
    
    # Инициализация моделей
    vanilla_model = VanillaDiffusionModel(vocab_size)
    cognitive_model = CognitiveDiffusionModel(vocab_size)
    
    # Обучение
    print("Training Vanilla Diffusion model...")
    train_model(vanilla_model, epochs=25)
    
    print("Training Cognitive Diffusion (CIDA) model...")
    train_model(cognitive_model, epochs=25)
    
    print("-" * 60)
    # Оценка
    evaluate_coherence(vanilla_model, "Vanilla Diffusion")
    evaluate_coherence(cognitive_model, "Cognitive Diffusion (CIDA)")
