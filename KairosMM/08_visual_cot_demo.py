"""
08_visual_cot_demo.py — Visual Chain of Thought (Gradio UI).

Это "Взрывной дебют" Кайроса. Красивый UI, который показывает:
1. Входной рентген.
2. Процесс мышления агентов (Disagreement & Uncertainty).
3. "Машину времени" (Агент-Целитель) — контрфактический снимок.
4. Отчёт на казахском языке (Qolda).

Использование:
    pip install gradio
    python KairosMM/08_visual_cot_demo.py
"""

import argparse
import random
import sys
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from PIL import Image

try:
    import gradio as gr
    _HAS_GRADIO = True
except ImportError:
    _HAS_GRADIO = False

# Заглушки для логики (в реальности здесь будут вызовы модулей)
def mock_cida_process(image):
    time.sleep(1) # Имитация вычислений
    diagnoses = {"Пневмония (Pneumonia)": 0.88, "Ателектаз (Atelectasis)": 0.45, "Қалыпты (Normal)": 0.02}
    disagreement = 0.12
    uncertainty = 0.25
    return diagnoses, disagreement, uncertainty

def mock_healer(image):
    time.sleep(2) # Имитация диффузии
    # В демо просто инвертируем или осветляем
    img_array = np.array(image)
    healed = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
    return Image.fromarray(healed)

def mock_qolda_report(diagnoses):
    time.sleep(1.5) # Имитация LLM
    return """**CIDA KairosMM Қорытындысы:**

Рентгенограммада оң жақ өкпенің төменгі бөлігінде қарайю аймақтары байқалады, бұл **Пневмония** (88%) белгілеріне сәйкес келеді. 

Агенттер арасындағы келіспеушілік төмен (0.12), жүйе өз диагнозына сенімді. 
Целитель-агенті патологиясыз (сау) өкпе бейнесін генерациялады, салыстыру үшін оны көре аласыз.

**Ұсыныс:** Пульмонолог дәрігерінің шұғыл консультациясы және антибактериалды терапияны бастау."""


def process_xray(image):
    if image is None:
        return None, None, None, "Суретті жүктеңіз (Загрузите снимок)"
    
    # 1. CIDA Agents
    diagnoses, disag, unc = mock_cida_process(image)
    
    # 2. Healer Agent (Counterfactual)
    healed_img = mock_healer(image)
    
    # 3. LLM Report (Qolda)
    report = mock_qolda_report(diagnoses)
    
    # 4. Format outputs
    dx_text = "\n".join([f"🔹 {k}: {v*100:.1f}%" for k, v in diagnoses.items()])
    metrics = f"⚠️ Келіспеушілік (Disagreement): {disag:.2f}\n❓ Сенімсіздік (Uncertainty): {unc:.2f}"
    
    return healed_img, dx_text, metrics, report


def build_ui():
    if not _HAS_GRADIO:
        print("Gradio не установлен. Запустите: pip install gradio")
        return
        
    theme = gr.themes.Soft(
        primary_hue="blue", 
        secondary_hue="indigo",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
    )

    with gr.Blocks(theme=theme, title="CIDA KairosMM 🚀") as demo:
        gr.Markdown(
            """
            # 🧬 CIDA KairosMM: H100 Edition (Взрывной Дебют)
            **Первый в мире мультиагентный медицинский ИИ с Liquid Flow динамикой и Генеративными Контрфактами.**
            Загрузите рентгеновский снимок, чтобы увидеть процесс мышления агентов.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Түпнұсқа рентген (Оригинал)")
                submit_btn = gr.Button("🔍 Анализ жасау (Анализировать)", variant="primary")
                
                gr.Markdown("### Агенттер консенсусы (CIDA)")
                dx_out = gr.Textbox(label="Диагноздар", lines=4)
                metrics_out = gr.Textbox(label="Метрикалар", lines=2)
                
            with gr.Column(scale=1):
                healed_out = gr.Image(type="pil", label="Агент-Целитель: Сау контрфакт (Здоровый снимок)")
                report_out = gr.Textbox(label="Qolda 4B: Қазақша қорытынды (Отчет)", lines=8)

        submit_btn.click(
            fn=process_xray,
            inputs=[input_img],
            outputs=[healed_out, dx_out, metrics_out, report_out]
        )
        
        gr.Markdown("""
        *Разработано: CIDA Team. На базе Qolda (ISSAI NU), BiomedCLIP и Flow Matching.*
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    if demo:
        print("Запуск сервера Gradio...")
        demo.launch(share=False, inbrowser=True)
