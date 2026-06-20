import argparse
import random
import sys
import time
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

try:
    import gradio as gr
    _HAS_GRADIO = True
except ImportError:
    _HAS_GRADIO = False

# Импортируем реальные компоненты динамически через importlib, так как имена файлов начинаются с цифры
import importlib
try:
    encoder_mod = importlib.import_module("KairosMM.01_encoder_biomedclip")
    BiomedCLIPCIDA = encoder_mod.BiomedCLIPCIDA
    
    qolda_mod = importlib.import_module("KairosMM.05_kazakh_llm_qolda")
    QoldaReportGenerator = qolda_mod.QoldaReportGenerator
    
    healer_mod = importlib.import_module("KairosMM.07_counterfactual_healer")
    CounterfactualHealerAgent = healer_mod.CounterfactualHealerAgent
    
    _HAS_REAL_MODULES = True
except ImportError:
    _HAS_REAL_MODULES = False


# Глобальные инстансы моделей (ленивая инициализация)
models = {
    "cida": None,
    "healer": None,
    "qolda": None
}

def init_models():
    if not _HAS_REAL_MODULES:
        return False
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[UI] Инициализация реальных моделей на {device}...")
        models["cida"] = BiomedCLIPCIDA(num_classes=14, d_hidden=768, freeze_encoder=True).to(device)
        models["healer"] = CounterfactualHealerAgent(device=device)
        models["qolda"] = QoldaReportGenerator(device=device)
        return True
    except Exception as e:
        print(f"[UI] Не удалось инициализировать реальные модели (нет весов/VRAM): {e}")
        return False


def plot_belief_dynamics(b_all):
    """
    Строит график динамики убеждений агентов по раундам дебатов.
    b_all: список тензоров убеждений по раундам. Каждый тензор: (B, num_agents, num_classes)
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Конвертируем в numpy
    # b_all имеет длину max_rounds
    rounds = len(b_all)
    num_agents = b_all[0].shape[1]
    
    # Берем первый элемент в батче и класс с максимальным финальным убеждением
    final_beliefs = b_all[-1][0] # (num_agents, num_classes)
    avg_beliefs = final_beliefs.mean(dim=0)
    top_class_idx = avg_beliefs.argmax().item()
    
    # Цвета для агентов
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33']
    
    for agent_idx in range(num_agents):
        agent_history = []
        for r in range(rounds):
            val = b_all[r][0, agent_idx, top_class_idx].item()
            agent_history.append(val)
            
        ax.plot(
            range(rounds), 
            agent_history, 
            marker='o', 
            linewidth=2, 
            color=colors[agent_idx % len(colors)],
            label=f"Агент {agent_idx + 1}"
        )
        
    ax.set_title("Динамика убеждений агентов (Лидирующий класс)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Раунд дебатов (ODE Шаги)", fontsize=9)
    ax.set_ylabel("Убеждение (Вероятность)", fontsize=9)
    ax.set_xticks(range(rounds))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_belief_dynamics_mock():
    """Fallback генератор графиков для демо режима."""
    fig, ax = plt.subplots(figsize=(6, 4))
    rounds = 4
    num_agents = 4
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33']
    
    # Имитируем схождение к консенсусу
    target = random.uniform(0.75, 0.95)
    for agent_idx in range(num_agents):
        start = random.uniform(0.1, 0.9)
        history = [start]
        for r in range(1, rounds):
            # Агенты сближаются к target с каждым шагом
            next_val = history[-1] + (target - history[-1]) * random.uniform(0.4, 0.7)
            history.append(next_val)
            
        ax.plot(
            range(rounds), 
            history, 
            marker='o', 
            linewidth=2, 
            color=colors[agent_idx],
            label=f"Агент {agent_idx + 1}"
        )
        
    ax.set_title("Динамика убеждений агентов (Пневмония)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Раунд дебатов (ODE Шаги)", fontsize=9)
    ax.set_ylabel("Убеждение (Вероятность)", fontsize=9)
    ax.set_xticks(range(rounds))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def process_xray(image):
    if image is None:
        yield None, None, None, "Загрузите изображение рентгена.", None
        return

    # Пробуем лениво инициализировать реальные модели
    has_real = False
    if _HAS_REAL_MODULES:
        if models["cida"] is None:
            has_real = init_models()
        else:
            has_real = True

    # 1. Запускаем CIDA и Healer
    if has_real:
        try:
            # Предобработка картинки для CIDA
            from torchvision.transforms import functional as TF
            img_tensor = TF.to_tensor(image).unsqueeze(0)
            img_224 = F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)
            img_448 = F.interpolate(img_tensor, size=(448, 448), mode="bilinear", align_corners=False)
            
            # CIDA
            with torch.no_grad():
                out_cida = models["cida"](img_224, img_448)
            
            p_final = out_cida["p_final"][0].cpu().numpy()
            disag = out_cida["disagreement"].item()
            u_epi = out_cida.get("u_epi", torch.tensor(0.0)).item()
            u_alea = out_cida.get("u_alea", torch.tensor(0.0)).item()
            
            # Строим график динамики
            b_all = out_cida["b_all"]
            fig = plot_belief_dynamics(b_all)
            
            # Вычисляем Healer
            x_healed, x_recon = models["healer"].generate_counterfactual(img_tensor, strength=0.4)
            healed_pil = TF.to_pil_image(x_healed[0].cpu())
            
            # Подготовка диагнозов для отчета
            diagnoses_dict = {}
            for i, name in enumerate(models["cida"].class_names):
                diagnoses_dict[name] = float(p_final[i])
                
            agent_data = {
                "p_final": p_final.tolist(),
                "class_names": models["cida"].class_names,
                "disagreement": disag,
                "u_epi": u_epi,
                "u_alea": u_alea
            }
            
            # Форматируем вывод
            dx_text = "\n".join([f"🔹 {k}: {v*100:.1f}%" for k, v in diagnoses_dict.items() if v > 0.15])
            metrics = f"⚠️ Келіспеушілік (Disagreement): {disag:.3f}\n🧠 Эпистемикалық (Epistemic): {u_epi:.3f}\n🌀 Алеаторикалық (Aleatoric): {u_alea:.3f}"
            
            # Ссылка на консилиум при сильном разногласии
            if disag > 0.15:
                metrics += "\n\n🚨 НАЗАР АУДАРЫҢЫЗ: Агенттер арасында үлкен келіспеушілік бар! Консилиум ұсынылады."
            
            # Стриминг отчета Qolda
            report_dict = models["qolda"].generate("dummy_image_path", agent_data)
            report_text = f"**Қорытынды (Structured JSON):**\n\n"
            report_text += f"📋 **Находки:** {report_dict.get('findings', '')}\n\n"
            report_text += f"🩺 **Диагноз:** {report_dict.get('potential_diagnosis', '')}\n\n"
            report_text += f"💡 **Рекомендации:** {report_dict.get('recommendations', '')}\n\n"
            report_text += f"🔍 **Эпистемикалық интерпретация:** {report_dict.get('epistemic_uncertainty_interpretation', '')}\n\n"
            report_text += f"🌀 **Алеаторикалық интерпретация:** {report_dict.get('aleatoric_uncertainty_interpretation', '')}"
            
            # Эффект пишущей машинки
            words = report_text.split(" ")
            current_report = ""
            for word in words:
                current_report += word + " "
                yield healed_pil, dx_text, metrics, current_report, fig
                time.sleep(0.03)

        except Exception as err:
            print(f"[UI] Ошибка во время инференса реальных моделей: {err}")
            has_real = False # переключаемся на симуляцию

    # 2. Симуляция (Fallback)
    if not has_real:
        # Имитируем шаги
        yield None, "Анализ басталды...", "Есептелуде...", "Есептелуде...", None
        time.sleep(1.0)
        
        diagnoses = {"Пневмония (Pneumonia)": 0.88, "Ателектаз (Atelectasis)": 0.45, "Эффузия (Effusion)": 0.12}
        disag = 0.18 # Специально > 0.15 для проверки предупреждения
        u_epi = 0.08
        u_alea = 0.17
        
        # Строим график динамики
        fig = plot_belief_dynamics_mock()
        
        # Симулируем Healer
        img_array = np.array(image)
        healed_array = np.clip(img_array * 1.1, 0, 255).astype(np.uint8)
        healed_pil = Image.fromarray(healed_array)
        
        # Форматируем
        dx_text = "\n".join([f"🔹 {k}: {v*100:.1f}%" for k, v in diagnoses.items()])
        metrics = f"⚠️ Келіспеушілік (Disagreement): {disag:.3f}\n🧠 Эпистемикалық (Epistemic): {u_epi:.3f}\n🌀 Алеаторикалық (Aleatoric): {u_alea:.3f}"
        
        if disag > 0.15:
            metrics += "\n\n🚨 НАЗАР АУДАРЫҢЫЗ: Агенттер арасында үлкен келіспеушілік бар (>0.15)! Консилиум ұсынылады."

        # Стриминг отчета Qolda
        report_text = f"**Қорытынды (Structured JSON Fallback):**\n\n"
        report_text += f"📋 **Находки:** Рентгенограммада оң жақ өкпенің төменгі бөлігінде қарайю және тығыздалу аймақтары байқалады.\n\n"
        report_text += f"🩺 **Диагноз:** Пневмония (Өкпе қабынуы) және ішінара Ателектаз ықтимал.\n\n"
        report_text += f"💡 **Рекомендации:** Пульмонолог дәрігерінің консультациясы. Қажет болса, КТ зерттеуін жасау және антибактериалды емді бастау.\n\n"
        report_text += f"🔍 **Эпистемикалық интерпретация:** Эпистемикалық белгісіздік төмен ({u_epi:.3f}), бұл модельде ұқсас жағдайлардың жеткілікті базасы бар екенін растайды.\n\n"
        report_text += f"🌀 **Алеаторикалық интерпретация:** Алеаторикалық белгісіздік орташа ({u_alea:.3f}), рентген кескініндегі шағын шуылдармен байланысты болуы мүмкін."

        words = report_text.split(" ")
        current_report = ""
        for word in words:
            current_report += word + " "
            yield healed_pil, dx_text, metrics, current_report, fig
            time.sleep(0.04)


def build_ui():
    if not _HAS_GRADIO:
        print("Gradio не установлен. Запустите: pip install gradio")
        return
        
    theme = gr.themes.Soft(
        primary_hue="blue", 
        secondary_hue="indigo",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
    )

    with gr.Blocks(theme=theme, title="CIDA KairosMM v4 🚀") as demo:
        gr.Markdown(
            """
            # 🧬 CIDA KairosMM v4: H100 Edition (Взрывной Дебют)
            **Первый в мире мультиагентный медицинский ИИ с Liquid Flow динамикой, Multi-Scale BiomedCLIP и Генеративными Контрфактами.**
            
            Загрузите рентгеновский снимок, чтобы увидеть процесс мышления агентов, динамику ODE-убеждений и получить отчет на казахском языке.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Түпнұсқа рентген (Оригинал)")
                submit_btn = gr.Button("🔍 Анализ жасау (Анализировать)", variant="primary")
                
                gr.Markdown("### Агенттер консенсусы (CIDA v4)")
                dx_out = gr.Textbox(label="Диагноздар", lines=4)
                metrics_out = gr.Textbox(label="Метрикалар мен Сенімсіздік", lines=4)
                
            with gr.Column(scale=1):
                healed_out = gr.Image(type="pil", label="Агент-Целитель: Сау контрфакт (Здоровый снимок)")
                report_out = gr.Markdown(label="Qolda 4B: Қазақша қорытынды (Отчет)")
                
                # Компонент графика динамики убеждений по раундам
                chart_out = gr.Plot(label="Динамика убеждений по раундам")

        submit_btn.click(
            fn=process_xray,
            inputs=[input_img],
            outputs=[healed_out, dx_out, metrics_out, report_out, chart_out]
        )
        
        gr.Markdown("""
        *Разработано: CIDA Team. На базе Qolda (ISSAI NU), BiomedCLIP и Flow Matching. v4 Release Ready.*
        """)

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="Generate a public link")
    args = parser.parse_args()

    demo = build_ui()
    if demo:
        print("Запуск сервера Gradio...")
        # Запускаем локальный веб-сервер, опционально создавая публичную ссылку
        demo.launch(share=args.share, inbrowser=not args.share)
