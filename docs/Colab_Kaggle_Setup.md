# 🚀 CIDA Kairos — Запуск в облаке (Colab / Kaggle)

## Что это?
**CIDA Kairos** — интеграция архитектуры CIDA (Liquid Neural ODE) с DiffusionGemma.
Агенты CIDA "обсуждают" генерируемый код в непрерывном времени, направляя шаги диффузии.

## Требования к GPU
| Платформа | GPU | VRAM | Подходит? |
|-----------|-----|------|-----------|
| **Kaggle** | 2× T4 | 2×16 GB | ✅ Да |
| **Colab Free** | T4 | 16 GB | ⚠️ Впритык (нужен `batch_size=1`) |
| **Colab Pro** | A100 | 40-80 GB | ✅ Идеально |
| Локальный RTX 3050 | — | 4-8 GB | ❌ Нет (для DiffusionGemma) |

---

## Ячейки для Jupyter Notebook

### Ячейка 1: Установка зависимостей
```python
# Установка свежих версий библиотек прямо из GitHub (так как DiffusionGemma — сверхновая модель)
!pip install git+https://github.com/huggingface/transformers.git -q
!pip install git+https://github.com/unslothai/unsloth.git -q
!pip install torchdiffeq bitsandbytes accelerate peft datasets -q

# Проверка GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### Ячейка 2: Клонирование репозитория
```python
!git clone https://github.com/YOUR_USERNAME/CIDA-plugin.git
%cd CIDA-plugin
!pip install -e . -q
```

### Ячейка 3: Авторизация HuggingFace (DiffusionGemma — gated model)
```python
from huggingface_hub import login
login()  # Введите ваш токен
```

### Ячейка 4: Запуск обучения
```python
!python experiments/exp_cida_diffusion.py \
    --max_steps 200 \
    --lora_r 16 \
    --cida_agents 4
```

### Ячейка 5 (опционально): Мониторинг VRAM
```python
!nvidia-smi
```

---

## Ключевые оптимизации

| Техника | Что делает | Экономия VRAM |
|---------|-----------|---------------|
| **Unsloth** | Оптимизированные Triton-ядра | 2-5x быстрее, -70% VRAM |
| **QLoRA (NF4)** | 4-bit квантование весов | ~52 GB → ~18 GB |
| **LoRA (r=16)** | Обучаем только 0.1% параметров | Огромная экономия |
| **Gradient Checkpointing** | Пересчёт активаций вместо хранения | -40% VRAM |
| **odeint_adjoint** | Adjoint method для ODE | Градиенты без графа |
| **AdamW 8-bit** | Квантованный оптимизатор | -50% памяти оптимизатора |
| **CIDA Gate Init** | gate ≈ 0 в начале | Стабильность обучения |
