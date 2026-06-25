import json

def create_notebook():
    cells = []
    
    def add_md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\\n" for line in text.split("\\n")]})
        
    def add_code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\\n" for line in text.split("\\n")]})

    add_md("# 🧬 Kairos OS: Этап 3 (End-to-End Пайплайн и Медицинская Диссертация)")
    add_md("Этот ноутбук собирает всю систему воедино. Мы объединяем **DenseNet (Глаза) + CIDA (Мозг) + Переводчик (Мост) + LLM (Генератор отчетов)**.")
    add_md("LLM настроена на написание отчетов в стиле **научной диссертации**, подробно аргументируя выводы на основе сомнений агентов.")

    add_code("""!pip install -q torch torchvision transformers diffusers bitsandbytes accelerate peft datasets
!rm -rf CIDA-plugin
!git clone https://github.com/Kairatzh/CIDA-plugin.git
%cd CIDA-plugin
!pip install -e .
""")

    add_md("### 1. Подключение Энкодера и CIDA (Замороженные веса)")
    add_code("""import torch
import importlib
import os
from google.colab import drive

# Подключаем диск, чтобы загрузить обученный на Этапе 2 мозг
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

densenet_mod = importlib.import_module("KairosMM.10_densenet_cida")
DenseNetCIDA = densenet_mod.DenseNetCIDA

# Инициализируем (5 классов CheXpert)
perception_core = DenseNetCIDA(num_classes=5, d_hidden=128, max_rounds=3, freeze_encoder=True)

ckpt_path = "/content/drive/MyDrive/kairos_chexpert_best.pth"
if os.path.exists(ckpt_path):
    print("Найдены веса Этапа 2! Загружаем...")
    perception_core.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
else:
    print("ВНИМАНИЕ: Веса не найдены! Используем случайную инициализацию для теста.")

# Полностью замораживаем глаза и мозг (Они уже умные)
for param in perception_core.parameters():
    param.requires_grad = False

perception_core.to(device)
perception_core.eval()
print("Perception & Reasoning Layer готов.")
""")

    add_md("### 2. Инициализация Переводчика и Малой LLM (Qwen 1.5B 4-bit + LoRA)")
    add_code("""from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from cida_plugin.translator import LatentSemanticTranslator

# SOTA 2026: Qwen2.5-1.5B-Instruct (Очень умная, но влезет в 5 ГБ памяти)
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

# 4-bit квантование для запуска на бесплатном T4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# Добавляем LoRA, чтобы научить LLM медицинскому сленгу без разрушения базовых знаний
peft_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
llm = get_peft_model(llm, peft_config)

# Инициализируем наш Переводчик (Мост между CIDA и LLM)
# LLM Qwen1.5B имеет скрытую размерность 1536
d_llm = llm.config.hidden_size
translator = LatentSemanticTranslator(d_cida=128, d_semantic=d_llm).to(device)

print("Переводчик и LLM готовы к поэтапному обучению!")
""")

    add_md("### 3. Системный Промпт (Режим: Медицинская Диссертация)")
    add_code("""SYSTEM_PROMPT = \"\"\"Ты — профессор радиологии мирового уровня и доктор медицинских наук. 
Твоя задача — анализировать латентные выводы ИИ-агентов (Прокурора и Защитника) и писать финальный отчет.

Пиши отчет в стиле медицинского исследования (диссертации). 
СТРУКТУРА ОТЧЕТА:
1. АБСТРАКТ (Краткая выжимка)
2. АРГУМЕНТЫ 'ПРОКУРОРА' (Патологии)
3. АРГУМЕНТЫ 'ЗАЩИТНИКА' (Здоровые признаки / Артефакты)
4. ДИФФЕРЕНЦИАЛЬНЫЙ ДИАГНОЗ
5. ФИНАЛЬНОЕ КЛИНИЧЕСКОЕ ЗАКЛЮЧЕНИЕ

Обязательно используй сложную медицинскую терминологию и аргументируй логику.\"\"\"

def generate_dissertation_report(image_tensor):
    # 1. Глаза смотрят
    with torch.no_grad():
        out = perception_core(image_tensor)
        
    # Суперпозиция: берем векторы Прокурора (0) и Защитника (1)
    b_all = out["b_all"][-1] # (1, 4, 5)
    prosecutor_b = b_all[:, 0, :].unsqueeze(1)
    defender_b = b_all[:, 1, :].unsqueeze(1)
    
    # Сливаем их вместе (1, 2, 5) -> Для переводчика
    latent_conflict = torch.cat([prosecutor_b, defender_b], dim=1)
    
    # 2. Переводчик декодирует
    semantic_embeds = translator(latent_conflict) # (1, seq_len, 1536)
    
    # 3. LLM пишет диссертацию
    prompt = f"Системный промпт: {SYSTEM_PROMPT}\\n\\nПроанализируй эти визуальные векторы и напиши отчет:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Имитация проброса эмбеддингов (В реальном обучении мы вшиваем semantic_embeds прямо в inputs_embeds)
    # Для ДЕМО: мы заставим LLM отвечать на основе вероятностей
    p_final = out["p_final"].squeeze().tolist()
    classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    findings = ", ".join([f"{c}: {p*100:.1f}%" for c, p in zip(classes, p_final)])
    
    demo_prompt = f"{prompt}\\nВероятности, найденные консилиумом ИИ:\\n{findings}\\n\\nОТЧЕТ:"
    
    input_ids = tokenizer(demo_prompt, return_tensors="pt").input_ids.to(device)
    
    outputs = llm.generate(input_ids, max_new_tokens=512, temperature=0.7, do_sample=True)
    report = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return report

print("Функция генерации отчетов готова!")
""")

    add_md("### 4. Тестирование всего Пайплайна (Инференс)")
    add_code("""import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Создадим фиктивный "рентген" для теста (или загрузите свой)
dummy_img = (np.random.randn(224, 224, 3) * 128 + 128).clip(0, 255).astype(np.uint8)
img = Image.fromarray(dummy_img)

# Показываем снимок
plt.imshow(img)
plt.axis('off')
plt.title("Загруженный снимок")
plt.show()

# Превращаем в тензор
img_tensor = torch.randn(1, 3, 224, 224).to(device)

print("\\n" + "="*50)
print("🧠 ЗАПУСК KAIROS OS (ГЕНЕРАЦИЯ ДИССЕРТАЦИИ)")
print("="*50)

report = generate_dissertation_report(img_tensor)

# Красивый вывод отчета
report_body = report.split("ОТЧЕТ:")[-1].strip()
print(report_body)
""")

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open("Kairos_End_to_End_Pipeline.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    print("Создан ноутбук: Kairos_End_to_End_Pipeline.ipynb")

if __name__ == "__main__":
    create_notebook()
