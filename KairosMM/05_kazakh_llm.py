"""
05_kazakh_llm.py — Казахский мини-LLM для профессиональных рентгенологических отчётов.

Модель: Qwen2.5-1.5B-Instruct
  - Нативная поддержка казахского (кириллица)
  - Влезает в Colab T4 через 4-bit QLoRA
  - Instruct-тюнинг: понимает задачи "опиши", "составь отчёт"

Два режима:
  A) fine-tune  — дообучаем на синтетических медицинских отчётах (QLoRA)
  B) inference  — генерируем отчёт по JSON из 03_agent_to_json.py

Структура отчёта:
  1. Клиническая информация
  2. Описание рентгенограммы (что видят агенты, на что смотрят)
  3. Выявленные патологии с вероятностями
  4. Заключение
  5. Рекомендации

Использование:
    # Только inference (без fine-tune, демо):
    python 05_kazakh_llm.py --mode inference --json_path ./llm_inputs/00000001_000.json

    # Fine-tune + inference:
    python 05_kazakh_llm.py --mode finetune --data_dir ./llm_inputs --out_dir ./llm_model
    python 05_kazakh_llm.py --mode inference --json_path ./llm_inputs/00000001_000.json \
                            --model_path ./llm_model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── Проверка зависимостей ─────────────────────────────────────────────────────
try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    print("[WARN] PyTorch не установлен")

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        GenerationConfig,
    )
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False
    print("[WARN] transformers не установлен: pip install transformers")

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
    )
    _PEFT_OK = True
except ImportError:
    _PEFT_OK = False
    print("[WARN] peft не установлен: pip install peft")

try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    _TRL_OK = True
except ImportError:
    _TRL_OK = False
    print("[WARN] trl не установлен: pip install trl")

try:
    from datasets import Dataset as HFDataset
    _DATASETS_OK = True
except ImportError:
    _DATASETS_OK = False


# ── Константы ─────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Системный промпт на казахском
SYSTEM_PROMPT = """Сіз — жасанды интеллект негізіндегі рентгенолог көмекшісісіз. 
Сізге пациенттің кеуде қуысының рентгенограммасын талдаған мультиагенттік CIDA жүйесінің 
нәтижелері беріледі. Осы нәтижелер негізінде қазақ тілінде кәсіби рентгенологиялық 
қорытынды дайындаңыз.

Қорытынды мынадай бөлімдерден тұруы тиіс:
1. КЛИНИКАЛЫҚ АҚПАРАТ
2. РЕНТГЕНОГРАММАНЫ СИПАТТАУ  
3. АНЫҚТАЛҒАН ПАТОЛОГИЯЛАР
4. ҚОРЫТЫНДЫ
5. ҰСЫНЫСТАР

Тек медициналық терминологияны қолданыңыз. Нәтижелерді асыра бағаламаңыз."""

# Шаблон для structured output
REPORT_TEMPLATE = """<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
Мына жасанды интеллект талдауы негізінде рентгенологиялық қорытынды жазыңыз:

{ai_analysis}
<|im_end|>
<|im_start|>assistant
{report}<|im_end|>"""


# ── Синтетические обучающие данные ───────────────────────────────────────────

SYNTHETIC_REPORTS = [
    {
        "pathologies": ["Effusion", "Atelectasis"],
        "probs": {"Effusion": 0.82, "Atelectasis": 0.71},
        "uncertainty": 0.18,
        "disagreement": 0.12,
        "report": """КЛИНИКАЛЫҚ АҚПАРАТ
Кеуде қуысының тікелей рентгенограммасы.

РЕНТГЕНОГРАММАНЫ СИПАТТАУ
Кеуде қуысының рентгенограммасы талданды. Жасанды интеллект жүйесінің 4 агенті 3 раунд 
пікірталасынан өтті. Агенттер назарын негізінен төменгі өкпе өрістеріне аударды.

АНЫҚТАЛҒАН ПАТОЛОГИЯЛАР
• Плевра экссудаты (Effusion): 82% ықтималдығымен анықталды — ЖОҒАРЫ
  Прокурор мен Интегратор агенттері: 0.85 және 0.79 сенімділікпен растады.
• Ателектаз (Atelectasis): 71% ықтималдығымен анықталды — ОРТАША
  Агенттер арасындағы сәйкессіздік: 0.12 (келісімге жақын).

ЖАЛПЫ БЕЛГІСІЗДІК: төмен (0.18) — диагнозға сенімділік жоғары.

ҚОРЫТЫНДЫ
Кеуде қуысының рентгенограммасында оң жақ плевра экссудаты мен базалды ателектаз 
белгілері анықталды. Клиникалық жағдаймен салыстыру ұсынылады.

ҰСЫНЫСТАР
1. Плевра пункциясы немесе торакоцентез мүмкіндігін бағалау
2. Компьютерлік томография (КТ) — нақтырақ бағалау үшін
3. Клиникалық симптомдармен корреляция жасау""",
    },
    {
        "pathologies": ["Pneumonia"],
        "probs": {"Pneumonia": 0.76},
        "uncertainty": 0.24,
        "disagreement": 0.22,
        "report": """КЛИНИКАЛЫҚ АҚПАРАТ
Кеуде қуысының тікелей рентгенограммасы.

РЕНТГЕНОГРАММАНЫ СИПАТТАУ
Жасанды интеллект жүйесі рентгенограмманы 3 раунд ішінде талдады. 
Прокурор агент патология белгілерін белсенді іздеді (prior=0.85),
Қорғаушы агент нормаға ұмтылды (prior=0.15).
Агенттер арасындағы сәйкессіздік орташа (0.22) — диагноз даулы.

АНЫҚТАЛҒАН ПАТОЛОГИЯЛАР
• Пневмония (Pneumonia): 76% ықтималдығымен — ОРТАША ДЕҢГЕЙ
  Прокурор: 0.81 | Қорғаушы: 0.48 | Скептик: 0.52 | Интегратор: 0.74
  Назар: оң жақ орта өкпе өрісі, перибронхиалды инфильтрация.

ЖАЛПЫ БЕЛГІСІЗДІК: орташа (0.24) — қосымша тексеру ұсынылады.

ҚОРЫТЫНДЫ
Оң жақ өкпенің орта өрісінде пневмония белгілері анықталды. Алайда агенттер 
арасындағы сәйкессіздік диагностикалық күмән тудырады.

ҰСЫНЫСТАР
1. Клиникалық белгілермен (температура, жөтел, лейкоцитоз) салыстыру
2. 7-10 күннен кейін бақылау рентгенограммасы
3. Антибиотикотерапия мүмкіндігін бағалау""",
    },
    {
        "pathologies": [],
        "probs": {},
        "uncertainty": 0.09,
        "disagreement": 0.06,
        "report": """КЛИНИКАЛЫҚ АҚПАРАТ
Кеуде қуысының тікелей рентгенограммасы, профилактикалық тексеру.

РЕНТГЕНОГРАММАНЫ СИПАТТАУ
Жасанды интеллект жүйесінің 4 агенті 3 раунд пікірталасынан өтті. 
Барлық агенттер консенсусқа жылдам жетті (сәйкессіздік: 0.06 — өте төмен).
Өкпе өрістерінің барлық аймақтарына бірдей назар аударылды.

АНЫҚТАЛҒАН ПАТОЛОГИЯЛАР
Патологиялар анықталмады. Барлық 14 сыныптың ықтималдығы 0.30 мәнінен төмен.

ЖАЛПЫ БЕЛГІСІЗДІК: өте төмен (0.09) — диагнозға жоғары сенімділік.

ҚОРЫТЫНДЫ
Кеуде қуысының рентгенограммасы патологиялық өзгерістерсіз. 
Өкпе өрістері ашық, синустар еркін, орта көлеңке қалыпты.

ҰСЫНЫСТАР
Кезекті профилактикалық тексеру жоспарланған мерзімде өткізіледі.""",
    },
    {
        "pathologies": ["Cardiomegaly", "Edema"],
        "probs": {"Cardiomegaly": 0.89, "Edema": 0.77},
        "uncertainty": 0.14,
        "disagreement": 0.09,
        "report": """КЛИНИКАЛЫҚ АҚПАРАТ
Кеуде қуысының тікелей рентгенограммасы.

РЕНТГЕНОГРАММАНЫ СИПАТТАУ
Жасанды интеллект жүйесі 3 раунд пікірталасынан кейін консенсусқа жетті.
Агенттер негізінен орталық аймаққа (жүрек, орта көлеңке) назар аударды — 79%.

АНЫҚТАЛҒАН ПАТОЛОГИЯЛАР
• Кардиомегалия (Cardiomegaly): 89% ықтималдығымен — ЖОҒАРЫ
  Барлық агенттер растады: 0.91 / 0.82 / 0.85 / 0.90
• Өкпе ісінуі (Edema): 77% ықтималдығымен — ЖОҒАРЫ
  Прокурор: 0.83 | Интегратор: 0.79 сенімділікпен анықтады.

ЖАЛПЫ БЕЛГІСІЗДІК: төмен (0.14) — диагнозға жоғары сенімділік.
АГЕНТТЕР КОНСЕНСУСЫ: өте жоғары (сәйкессіздік: 0.09).

ҚОРЫТЫНДЫ
Кардиомегалия және екіжақты өкпе ісінуі белгілері анықталды.
Жүрек жеткіліксіздігін жоққа шығару үшін кардиологтың кеңесі қажет.

ҰСЫНЫСТАР
1. Кардиолог кеңесі — ШҰҒЫЛ
2. ЭхоКГ (эхокардиография)
3. Диуретикалық терапияны бағалау
4. BNP/NT-proBNP зерттеуі""",
    },
]


# ── Загрузка модели ───────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    use_4bit: bool = True,
    model_path: Optional[str] = None,
):
    """
    Загружает Qwen2.5-1.5B-Instruct с опциональным 4-bit квантованием.

    use_4bit=True → требует bitsandbytes: pip install bitsandbytes
    use_4bit=False → больший VRAM, но без зависимостей
    """
    if not _TRANSFORMERS_OK:
        raise ImportError("pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit квантование для Colab T4
    if use_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            quant_kwargs = {"quantization_config": bnb_config}
        except Exception:
            print("[WARN] bitsandbytes недоступен, загружаем в float16")
            quant_kwargs = {"torch_dtype": torch.float16}
    else:
        quant_kwargs = {"torch_dtype": torch.float16}

    # Загружаем модель
    base_path = model_path if model_path and os.path.isdir(model_path) else model_id
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        trust_remote_code=True,
        **quant_kwargs,
    )

    # Если есть LoRA адаптер — загружаем поверх
    if model_path and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        if _PEFT_OK:
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            print(f"[LLM] LoRA адаптер загружен из {model_path}")

    model.eval()
    print(f"[LLM] Модель загружена: {base_path}")
    return model, tokenizer


# ── Fine-tune с QLoRA ─────────────────────────────────────────────────────────

def prepare_training_data(json_dir: Optional[str] = None) -> list:
    """
    Подготавливает обучающие данные для fine-tune.

    Если json_dir указан — генерируем данные из реальных JSON файлов.
    Иначе — используем синтетические примеры выше.
    """
    samples = []

    # Синтетические примеры (всегда добавляем)
    for ex in SYNTHETIC_REPORTS:
        # Конструируем ai_analysis из примера
        ai_lines = ["=== ЖИ ТАЛДАУЫ ==="]
        if ex["pathologies"]:
            ai_lines.append("АНЫҚТАЛҒАН ПАТОЛОГИЯЛАР:")
            for p in ex["pathologies"]:
                ai_lines.append(f"  • {p}: {ex['probs'].get(p, 0)*100:.0f}%")
        else:
            ai_lines.append("Патологиялар анықталмады.")
        ai_lines.append(f"Белгісіздік: {ex['uncertainty']:.2f}")
        ai_lines.append(f"Агент сәйкессіздігі: {ex['disagreement']:.2f}")
        ai_analysis = "\n".join(ai_lines)

        text = REPORT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            ai_analysis=ai_analysis,
            report=ex["report"],
        )
        samples.append({"text": text})

    # Реальные данные из JSON
    if json_dir and os.path.isdir(json_dir):
        for json_path in Path(json_dir).glob("*.json"):
            if json_path.name.startswith("_"):
                continue
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)

                # Используем for_llm_prompt если есть
                if "for_llm_prompt" in data:
                    ai_analysis = data["for_llm_prompt"]
                else:
                    ai_analysis = f"Бейне: {data.get('image_name', 'unknown')}"

                # Генерируем шаблонный отчёт из данных
                report = _generate_template_report(data)

                text = REPORT_TEMPLATE.format(
                    system=SYSTEM_PROMPT,
                    ai_analysis=ai_analysis,
                    report=report,
                )
                samples.append({"text": text})
            except Exception as e:
                continue

    print(f"[LLM] Обучающих примеров: {len(samples)}")
    return samples


def _generate_template_report(data: dict) -> str:
    """Генерирует шаблонный отчёт на казахском из структурированного JSON."""
    cons = data.get("consensus", {})
    findings = cons.get("positive_findings", [])
    unc = cons.get("uncertainty_info", {})
    dis = cons.get("disagreement_info", {})
    image_name = data.get("image_name", "unknown")

    lines = []
    lines.append("КЛИНИКАЛЫҚ АҚПАРАТ")
    lines.append(f"Бейне: {image_name}")
    lines.append("Кеуде қуысының тікелей рентгенограммасы.\n")

    lines.append("РЕНТГЕНОГРАММАНЫ СИПАТТАУ")
    lines.append(f"Мультиагенттік CIDA жүйесі рентгенограмманы {data.get('metadata', {}).get('rounds_used', 3)} раунд ішінде талдады.")
    lines.append(f"Агенттер арасындағы сәйкессіздік: {dis.get('level', '')} — {dis.get('description', '')}\n")

    lines.append("АНЫҚТАЛҒАН ПАТОЛОГИЯЛАР")
    if findings:
        for f in findings:
            prob_pct = f['probability'] * 100
            lines.append(f"• {f['pathology']}: {prob_pct:.0f}%")
    else:
        lines.append("Патологиялар анықталмады (норма).")
    lines.append("")

    lines.append(f"ЖАЛПЫ БЕЛГІСІЗДІК: {unc.get('level', '')} — {unc.get('recommendation', '')}\n")

    lines.append("ҚОРЫТЫНДЫ")
    if findings:
        pathology_list = ", ".join(f['pathology'] for f in findings[:3])
        lines.append(f"Рентгенограммада {pathology_list} белгілері анықталды.")
    else:
        lines.append("Рентгенограммада патологиялық өзгерістер анықталмады.")
    lines.append("")

    lines.append("ҰСЫНЫСТАР")
    lines.append("1. Клиникалық симптомдармен корреляция жасау")
    if findings:
        lines.append("2. Мамандандырылған дәрігер кеңесі")
        lines.append("3. КТ немесе МРТ — нақтырақ бағалау үшін")

    return "\n".join(lines)


def finetune(args):
    """QLoRA fine-tune Qwen2.5-1.5B на казахских медицинских отчётах."""
    if not (_PEFT_OK and _TRL_OK and _DATASETS_OK):
        print("[ERROR] Установите: pip install peft trl datasets bitsandbytes")
        return

    print("\n[Fine-tune] Загрузка модели...")
    model, tokenizer = load_model_and_tokenizer(use_4bit=True)

    # LoRA конфигурация
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Данные
    samples = prepare_training_data(args.data_dir)
    dataset = HFDataset.from_list(samples)

    # Тренировочные аргументы
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )

    print("\n[Fine-tune] Обучение...")
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"[Fine-tune] Модель сохранена → {args.out_dir}")


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_report(
    json_path: str,
    model=None,
    tokenizer=None,
    model_path: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """
    Генерирует профессиональный отчёт на казахском.

    json_path: путь к структурированному JSON из 03_agent_to_json.py
    """
    if not _TRANSFORMERS_OK:
        raise ImportError("pip install transformers")

    # Загружаем модель если не передана
    if model is None:
        model, tokenizer = load_model_and_tokenizer(
            model_path=model_path, use_4bit=True
        )

    # Читаем JSON
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Получаем промпт (из 03_agent_to_json.py)
    ai_analysis = data.get("for_llm_prompt", "")
    if not ai_analysis:
        # Fallback: строим из данных
        ai_analysis = _generate_template_report(data)

    # Формируем входной промпт
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Мына жасанды интеллект талдауы негізінде рентгенологиялық қорытынды жазыңыз:\n\n{ai_analysis}"},
    ]

    # Применяем chat template Qwen
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Декодируем только новые токены
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    report = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return report


def batch_generate(json_dir: str, out_dir: str, model_path: Optional[str] = None):
    """Генерирует отчёты для всех JSON файлов в директории."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = [f for f in Path(json_dir).glob("*.json")
                  if not f.name.startswith("_")]

    print(f"\n[Batch inference] {len(json_files)} файлов...")

    # Загружаем модель один раз
    model, tokenizer = load_model_and_tokenizer(model_path=model_path, use_4bit=True)

    for json_path in json_files:
        try:
            report = generate_report(
                str(json_path), model=model, tokenizer=tokenizer
            )
            out_path = out_dir / (json_path.stem + "_report.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Бейне: {json_path.stem}\n")
                f.write("="*60 + "\n")
                f.write(report)
                f.write("\n" + "="*60 + "\n")
            print(f"  ✓ {json_path.name}")
        except Exception as e:
            print(f"  [WARN] Ошибка {json_path.name}: {e}")

    print(f"\n[Batch inference] Готово → {out_dir}")


# ── Demo (без настоящей модели) ───────────────────────────────────────────────

def demo_without_llm(json_path: Optional[str] = None):
    """
    Демо без загрузки LLM — показывает промпт и шаблонный отчёт.
    Полезно для проверки pipeline без GPU.
    """
    print("="*60)
    print("ДЕМО: LLM Report Generator (без загрузки модели)")
    print("="*60)

    # Синтетический JSON если файл не указан
    if json_path and os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Создаём минимальный синтетический пример
        data = {
            "image_name": "demo_patient_001.png",
            "metadata": {"rounds_used": 3},
            "consensus": {
                "positive_findings": [
                    {"pathology": "Effusion", "probability": 0.82},
                    {"pathology": "Atelectasis", "probability": 0.71},
                ],
                "uncertainty_info": {"level": "төмен", "recommendation": "Диагнозға сенімділік жоғары"},
                "disagreement_info": {"level": "консенсус", "description": "Барлық агенттер келіседі"},
            },
            "for_llm_prompt": SYNTHETIC_REPORTS[0]["report"],
        }

    ai_analysis = data.get("for_llm_prompt", "")
    template_report = _generate_template_report(data)

    print("\n[ПРОМПТ ДЛЯ LLM]")
    print("-"*40)
    print(f"SYSTEM: {SYSTEM_PROMPT[:200]}...")
    print(f"\nUSER:\n{ai_analysis[:500]}...")

    print("\n\n[ШАБЛОННЫЙ ОТЧЁТ (без LLM)]")
    print("-"*40)
    print(template_report)

    print("\n[СИНТЕТИЧЕСКИЙ ОТЧЁТ (пример вывода LLM)]")
    print("-"*40)
    print(SYNTHETIC_REPORTS[0]["report"])

    return template_report


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Казахский LLM для рентгенологии")
    parser.add_argument("--mode",       choices=["finetune", "inference", "batch", "demo"],
                        default="demo")
    parser.add_argument("--json_path",  default=None, help="Путь к JSON из 03_agent_to_json.py")
    parser.add_argument("--data_dir",   default="./llm_inputs",
                        help="Директория с JSON для fine-tune/batch")
    parser.add_argument("--out_dir",    default="./llm_model",
                        help="Куда сохранять fine-tuned модель / отчёты")
    parser.add_argument("--model_path", default=None,
                        help="Путь к дообученной модели (если есть)")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--no_4bit",    action="store_true",
                        help="Отключить 4-bit квантование")
    args = parser.parse_args()

    if args.mode == "demo":
        demo_without_llm(args.json_path)

    elif args.mode == "finetune":
        finetune(args)

    elif args.mode == "inference":
        if not args.json_path:
            print("[ERROR] Укажите --json_path")
            sys.exit(1)
        report = generate_report(args.json_path, model_path=args.model_path)
        print("\n" + "="*60)
        print("МЕДИЦИНАЛЫҚ ҚОРЫТЫНДЫ:")
        print("="*60)
        print(report)
        # Сохраняем отчёт
        out_path = Path(args.out_dir) / (Path(args.json_path).stem + "_report.txt")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nСохранено → {out_path}")

    elif args.mode == "batch":
        batch_generate(args.data_dir, args.out_dir, model_path=args.model_path)