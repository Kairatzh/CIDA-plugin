# 💻 CIDA Developer Quickstart — Полное Руководство Разработчика

> **Назначение:** Инструкция для внешних разработчиков, которые хотят интегрировать CIDA в свои проекты. Написано так, чтобы человек, впервые видящий библиотеку, мог запуститься за 10 минут.

---

## ⚡ Часть 1: Установка (30 секунд)

### Стандартная установка (локальная машина с GPU)
```bash
pip install cida-plugin
```

### Установка для Google Colab (T4 GPU, 15GB VRAM)
```python
# Обязательно bitsandbytes для 4-битного сжатия 7B модели
!pip install cida-plugin bitsandbytes
# После установки: Runtime → Restart Session
```

### Проверка установки
```python
import cida_plugin
print(cida_plugin.__version__)  # Должно быть: 1.4.2

from cida_plugin.inference.latent_engineering import LatentPrompt
from cida_plugin.inference import LatentAgent, LatentMemory, CidaSwarm
from cida_plugin.orchestrator import KairosOrchestrator
print("Все импорты OK!")
```

---

## 🚀 Часть 2: За 5 Минут — Полный Рабочий Пример

```python
import torch
from cida_plugin.orchestrator import KairosOrchestrator
from cida_plugin.inference import LatentAgent
from cida_plugin.inference.latent_engineering import LatentPrompt

# ─── ШАГ 1: Инициализация Оркестратора ─────────────────────────────────────
# KairosOrchestrator загружает и склеивает два компонента:
# 1) Frozen LLM backbone (Qwen2.5-7B, думает и говорит)
# 2) Обученный CIDA adapter (ваш "молчаливый совет экспертов")
orchestrator = KairosOrchestrator(
    backbone_model_name="Qwen/Qwen2.5-7B-Instruct",   # Любая HF-модель
    cida_model_path_or_repo="kirtzh/Kairos-7B-CIDA-v1.5",  # Наши обученные веса
    device="cuda",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True  # Обязательно для Colab T4 (15 GB VRAM)
)

# ─── ШАГ 2: Создание Латентного Промпта ─────────────────────────────────────
# Это делается ОДИН РАЗ. Потом можно сохранить и загружать из файла.
system_instructions = """
Ты — Senior DevOps-инженер с 10-летним опытом.
Ты никогда не удаляешь данные из production без явного подтверждения.
При диагностике ты всегда сначала собираешь факты, потом действуешь.
Ты предпочитаешь минимально инвазивные решения.
"""
brain = LatentPrompt.compile_from_text(orchestrator, system_instructions)
# Выход: "Компиляция Латентного Промпта. Всего концепций (чанков): 4"
# Выход: "Успешно скомпилировано! Токенов: 183 -> Векторов: 4"

# Сохраняем для будущего использования
brain.save_local("my_devops_brain.pt")

# ─── ШАГ 3: Определение Инструментов (Функций) ──────────────────────────────
# Агент может вызывать любые Python-функции
import requests

def check_server_status(server_id: str) -> str:
    """Проверяет статус сервера по его ID."""
    # В реальном коде здесь был бы API-запрос
    return f"Server {server_id}: STATUS=DOWN, CPU=98%, RAM=99%, DB_CONNECTIONS=500/500"

def restart_service(service_name: str) -> str:
    """Перезапускает указанный сервис."""
    return f"Service '{service_name}' successfully restarted."

tools = {
    "check_server_status": check_server_status,
    "restart_service": restart_service,
}

# ─── ШАГ 4: Создание и Запуск Агента ────────────────────────────────────────
agent = LatentAgent(
    wrapper=orchestrator,
    llm=orchestrator.llm,
    tokenizer=orchestrator.tokenizer,
    tools=tools,
    latent_prompt=brain,  # Мозг инъецируется в память с бессмертием (decay=999)
    silent_mode=False,    # True = агент не печатает внутренние мысли
    default_reasoning_rounds=3  # Количество раундов дебатов (3-7 рекомендуется)
)

result = agent.run(
    "Production сервер prod-01 не отвечает. Диагностируй и устрани проблему."
)
print(result)
```

---

## 📝 Часть 3: Продвинутые Сценарии

### Сценарий А: Загрузить существующий Латентный Промпт с диска

```python
from cida_plugin.inference.latent_engineering import LatentPrompt

# Загружаем сохранённый мозг
brain = LatentPrompt.load_local("my_devops_brain.pt", device="cuda")
print(f"Загружено концептов: {len(brain.tensors)}")
print(f"Метаданные: {brain.metadata}")
# > Загружено концептов: 4
# > Метаданные: {'original_length': 183, 'num_concepts': 4, 'chunking_strategy': 'double_newline'}
```

### Сценарий Б: Загрузить Латентный Промпт из Hugging Face Hub

```python
from cida_plugin.inference.latent_engineering import LatentPrompt

# В будущем, когда Latent Hub будет заполнен:
coding_brain = LatentPrompt.from_huggingface(
    repo_id="kirtzh/CIDA-Latent-Hub",
    filename="expert_coder_python.pt",
    device="cuda"
)
```

### Сценарий В: Мульти-Агентный Рой (CidaSwarm) — Pipeline Mode

```python
from cida_plugin.inference import CidaSwarm, LatentMemory

# Создаём общую память для всех агентов роя
shared_memory = LatentMemory(max_size=200, device="cuda")

# Создаём специализированных агентов
analyst_agent = LatentAgent(
    wrapper=orchestrator, llm=orchestrator.llm, tokenizer=orchestrator.tokenizer,
    tools={"search_logs": search_logs_func},
    latent_prompt=LatentPrompt.load_local("analyst_brain.pt")
)

reporter_agent = LatentAgent(
    wrapper=orchestrator, llm=orchestrator.llm, tokenizer=orchestrator.tokenizer,
    tools={},
    latent_prompt=LatentPrompt.load_local("reporter_brain.pt")
)

# Создаём Рой в режиме Pipeline (Agent A → Agent B)
swarm = CidaSwarm(
    agents=[analyst_agent, reporter_agent],
    shared_memory=shared_memory,
    mode="pipeline"
)

# Запускаем весь пайплайн одной командой
result = swarm.run("Проанализируй последние 100 строк access.log и подготовь отчёт об угрозах.")
```

### Сценарий Г: Запуск без обученных весов (для тестирования кода)

```python
orchestrator = KairosOrchestrator(
    backbone_model_name="Qwen/Qwen2.5-7B-Instruct",
    cida_model_path_or_repo=None,  # <-- Создаёт случайные веса CIDA
    device="cuda",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)
# Выход: "[CIDA-OS] No CIDA adapter path provided. Initializing fresh CIDA weights for testing."
```

---

## 🔧 Часть 4: Конфигурация и Параметры

### KairosOrchestrator — Полный Список Параметров

```python
KairosOrchestrator(
    backbone_model_name: str,           # HF repo ID или локальный путь к LLM
    cida_model_path_or_repo: str|None,  # HF repo ID, локальный путь или None
    device: str = "cuda",               # "cuda" или "cpu"
    torch_dtype = torch.float16,        # torch.float16 или torch.bfloat16
    # Любые дополнительные kwargs передаются в AutoModelForCausalLM:
    load_in_4bit=True,                  # 4-бит квантизация (нужен bitsandbytes)
    load_in_8bit=True,                  # 8-бит квантизация (нужен bitsandbytes)
    attn_implementation="flash_attention_2",  # Быстрее на A100/H100
)
```

### LatentAgent — Полный Список Параметров

```python
LatentAgent(
    wrapper,                          # Экземпляр KairosOrchestrator
    llm,                              # orchestrator.llm
    tokenizer,                        # orchestrator.tokenizer
    tools: dict = None,               # {"func_name": callable}
    latent_prompt: LatentPrompt = None,  # Инъецируется в память с бессмертием
    silent_mode: bool = False,        # True = только векторное мышление, без текста
    default_reasoning_rounds: int = 3,   # Базовое число раундов дебатов
    max_turns: int = 8,               # Максимальное число ReAct-итераций
)
```

---

## ❓ Часть 5: FAQ — Частые Вопросы

**Q: Нужно ли мне скачивать полную 7B модель Qwen при каждом запуске Colab?**  
A: Только при первом запуске (~15 ГБ). Потом Colab кэширует её в `/root/.cache/huggingface/`. Если не сбрасывать дисковый кэш, при повторном запуске загрузка занимает секунды.

**Q: Могу ли я использовать другую модель вместо Qwen2.5-7B?**  
A: Да. CIDA агностична к бэкбону. Главное — убедиться, что `d_llm` (hidden_size) вашей новой модели совпадает с тем, на котором обучен CIDA-адаптер. Для `kirtzh/Kairos-7B-CIDA-v1.5` это **4096** (Qwen1.5-7B пространство). Если возникает несоответствие размерности — нужен projector-слой.

**Q: Что такое файл `.pt` с Латентным Промптом и можно ли его передавать другим людям?**  
A: Файл `.pt` — это бинарный PyTorch-тензор. Он не содержит читаемый текст ваших инструкций — только матрицу чисел. Передавать безопасно, кража интеллектуальной собственности через него крайне затруднена. Именно это называется "Latent IP Protection".

**Q: Как CIDA знает, сколько раундов дебатов достаточно?**  
A: Есть три механизма:
1. Если дисперсия агентов < 0.05 — Fast Path, 0 раундов (тривиальная задача)
2. Если `Halting Predictor` предсказывает консенсус — остановка
3. Максимум: `default_reasoning_rounds` (ваш параметр, 3-15)

**Q: Что делать если `OutOfMemoryError`?**  
A: Убедитесь, что используете `load_in_4bit=True` + `torch_dtype=torch.bfloat16`. Также попробуйте добавить в начало: `import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`.

---

## 📬 Часть 6: Как Сообщить о Проблеме

1. Откройте Issue на [GitHub](https://github.com/Kairatzh/CIDA-plugin/issues)
2. Укажите: версию библиотеки (`pip show cida-plugin`), версию Python, GPU (если применимо), полный stack trace
3. Если возможно — минимальный воспроизводящий пример (20-30 строк кода)
