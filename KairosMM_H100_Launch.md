# 🚀 CIDA KairosMM: H100 Launchpad

Это руководство содержит пошаговый процесс развертывания KairosMM на мощном облачном инстансе с GPU (например, NVIDIA H100, A100 на платформах вроде RunPod, AWS, Lambda Labs). От клонирования репозитория до запуска интерактивного демо.

---

## Шаг 1: Клонирование репозитория и настройка окружения
Зайдите по SSH в ваш арендованный сервер или откройте его терминал. Сначала загрузим код и базовые инструменты.

```bash
git clone https://github.com/ВАШ_ПОЛЬЗОВАТЕЛЬ/CIDA-plugin.git
cd CIDA-plugin
pip install -q kaggle
```

---

## Шаг 2: Загрузка датасета NIH ChestX-ray14
> [!TIP]
> **Уже скачали датасет?** Если у вас в корне проекта уже есть папка `data/nih_cxr14/` с картинками и файлом `Data_Entry_2017.csv`, просто **пропустите этот шаг полностью**! Мой код автоматически найдет ваши данные.

Если данных еще нет, для скачивания (размер около 40 ГБ) вам понадобится ваш токен Kaggle (`kaggle.json`). Загрузите его на сервер.

```bash
# Настройка Kaggle API (предварительно закиньте kaggle.json в корень сервера)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Создаем структуру папок для данных внутри проекта
mkdir -p data/nih_cxr14/images

# Скачиваем датасет
kaggle datasets download -d nih-chest-xrays/data -p data/nih_cxr14 --unzip

# Перемещаем распакованные файлы в правильную структуру
mv data/nih_cxr14/images*/* data/nih_cxr14/images/ 2>/dev/null || true
```

---

## Шаг 3: Установка специфичных зависимостей для H100
Устанавливаем `flash-attn` для максимальной скорости вычислений и все библиотеки из нашего конфигурационного файла.

```bash
# Оптимизация под архитектуру Hopper/Ampere
pip install -q flash-attn --no-build-isolation

# Запускаем наш скрипт-установщик (torchdiffeq, diffusers, transformers и т.д.)
python KairosMM/00_setup.py --install
```

---

## Шаг 4: Кэширование весов моделей (Место хранения)
> [!IMPORTANT]  
> **Где хранить тяжелые веса?** Лучшая практика на облачных инстансах — хранить веса на **примонтированном диске (Volume)** (обычно это папка `/workspace`), чтобы они не удалились при перезапуске сервера.

Мы используем системную переменную `HF_HOME`, чтобы HuggingFace скачивал все модели (Qolda 4B, BiomedCLIP) в надежное место.

```bash
# Экспортируем переменную окружения (укажите ваш постоянный диск, например /workspace/hf_cache)
export HF_HOME="./hf_models_cache"

# Скачиваем все веса заранее, чтобы обучение не прервалось из-за обрывов сети
python KairosMM/00_setup.py --download_models --cache_dir ./hf_models_cache
```

---

## Шаг 5: Турбо-обучение на H100
Запускаем процесс обучения архитектуры CIDA. 
*   **Оптимизации:** Используется нативный для H100 формат `bfloat16` (в 2-3 раза быстрее) и компиляция `torch.compile`.
*   **Сохранение:** Веса будут сохранены в папку `./checkpoints/h100_best`.

```bash
# Размер батча (64) можно увеличить до 128, если у вас 80GB VRAM
python KairosMM/02_train_cida_h100.py --batch_size 64 --epochs 5 --precision bf16 --out_dir ./checkpoints/h100_best
```

---

## Шаг 6: Взрывной Дебют (Gradio UI)
Запускаем визуальный интерфейс "Visual Chain-of-Thought".

> [!TIP]
> Если вы хотите показать систему инвесторам или коллегам, измените в файле `08_visual_cot_demo.py` параметр `share=False` на `share=True`.

```bash
# Быстрая замена через sed (чтобы сгенерировать публичную ссылку для инвесторов)
sed -i 's/share=False/share=True/g' KairosMM/08_visual_cot_demo.py

# Запуск интерактивного демо
python KairosMM/08_visual_cot_demo.py
```

После выполнения последней команды в терминале появится ссылка вида `https://1234abc.gradio.live`. Перейдите по ней с любого устройства, чтобы протестировать KairosMM в реальном времени!
