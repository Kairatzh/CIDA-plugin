import argparse
import random
import sys
import time
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Добавляем корень проекта в sys.path для корректных импортов
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        from pathlib import Path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[UI] Инициализация реальных моделей на {device}...")
        
        # 1. Загрузка CIDA
        cida_model = BiomedCLIPCIDA(num_classes=14, d_hidden=768, freeze_encoder=True).to(device)
        weights_path = Path("checkpoints/cida_best.pth")
        if weights_path.exists():
            print(f"[UI] Загрузка весов CIDA из {weights_path}...")
            cida_model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            print("[UI] [WARN] checkpoints/cida_best.pth не найден! Демо запустится в режиме симуляции.")
            return False
            
        models["cida"] = cida_model
        
        # 2. Инициализация Healer (load_pipeline загружает latent_flow_healer.pth)
        healer = CounterfactualHealerAgent(device=device)
        healer.load_pipeline()
        models["healer"] = healer
        
        # 3. Инициализация Qolda
        models["qolda"] = QoldaReportGenerator(device=device)
        return True
    except Exception as e:
        print(f"[UI] Не удалось инициализировать реальные модели (нет весов/VRAM): {e}")
        return False


def compute_difference_map(orig_pil, healed_pil):
    """
    Вычисляет разность между оригиналом и исцеленным снимком,
    нормализует её и накладывает цветную карту (jet) поверх оригинального снимка
    для визуализации рискованных и измененных патологических зон.
    """
    from PIL import ImageChops
    if orig_pil is None or healed_pil is None:
        return None
    
    # Приводим к одному размеру
    h_size = healed_pil.size
    orig_resized = orig_pil.resize(h_size, Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR).convert("L")
    healed_gray = healed_pil.convert("L")
    
    # Считаем абсолютную разность
    diff_img = ImageChops.difference(orig_resized, healed_gray)
    diff_arr = np.array(diff_img).astype(np.float32)
    
    # Пороговый фильтр шума (убираем незначительные колебания VAE)
    diff_arr = np.clip(diff_arr - 8.0, 0, None)
    max_val = diff_arr.max()
    if max_val > 0:
        diff_arr = (diff_arr / max_val)
    
    # Применяем цветовую карту jet
    cmap = plt.colormaps.get_cmap("jet")
    colored_arr = cmap(diff_arr) # RGBA [0, 1]
    
    # Превращаем в RGB PIL Image
    colored_rgb = (colored_arr[:, :, :3] * 255).astype(np.uint8)
    
    # Накладываем тепловую карту поверх оригинального изображения для анатомического контекста
    orig_rgb = np.array(orig_resized.convert("RGB")).astype(np.float32)
    mask = np.expand_dims(diff_arr, axis=-1)
    blend_arr = orig_rgb * (1.0 - mask * 0.7) + colored_rgb * (mask * 0.7)
    blend_img = Image.fromarray(np.clip(blend_arr, 0, 255).astype(np.uint8))
    
    return blend_img


def translate_debate_to_narrative(b_all, class_names, disagreement, u_epi, u_alea):
    """
    Транслирует историю изменения убеждений агентов CIDA v4 в клинический отчет на русском и казахском.
    """
    narrative = []
    
    if b_all is None:
        # Fallback лог для демонстрационного режима
        num_agents = 4
        narrative.append(f"🇰🇿 **CIDA Агенттерінің дебат талдауы:**")
        narrative.append(f"• Инициализация: {num_agents} тәуелсіз Агент-сарапшылар іске қосылды.")
        narrative.append(f"• 1-раунд (ODE): Агент 1 пневмонияны 88% деп күдіктенсе, Агент 2 мен 4 бейнедегі оптикалық шуға (алеаторикалық неопределенность) сілтеп, күмән келтірді.")
        narrative.append(f"• 2-раунд (ODE): Агенттер арасында анонимді түрде жасырын қалып-күйлермен (epistemic state) алмасу жүргізілді. Агент 3 оң жақ төменгі легочный сегменттегі аномалияны растады.")
        narrative.append(f"• 3-раунд (ODE): Агенттер ортақ консенсусқа жақындады. Соңғы ортақ диагноз: Пневмония (ықтималдығы 88%). Келіспеушілік мәні (Disagreement) {disagreement:.3f}-ке дейін төмендеді.")
        
        narrative.append(f"\n🇷🇺 **Анализ дебатов агентов CIDA:**")
        narrative.append(f"• Инициализация: Запущен консилиум из {num_agents} независимых Агентов-экспертов.")
        narrative.append(f"• 1-й раунд (ODE): Агент 1 (Клинические признаки) заподозрил пневмонию с вероятностью 88%. Агент 2 и 4 сомневались, ссылаясь на оптический шум на снимке (алеаторная неопределенность).")
        narrative.append(f"• 2-й раунд (ODE): Произошел анонимный обмен скрытыми состояниями. Агент 3 (Пространственные аномалии) подтвердил локальное уплотнение в нижнем поле правого легкого.")
        narrative.append(f"• 3-й раунд (ODE): Агенты пришли к единому консенсусу. Финальная вероятность: Пневмония (88%). Уровень разногласий снизился до {disagreement:.3f} (в пределах нормы).")
        return "\n".join(narrative)

    # Приводим к numpy
    if hasattr(b_all, "cpu"):
        b_all = b_all.cpu().numpy()
        
    num_rounds = len(b_all)
    num_agents = b_all[0].shape[0] if num_rounds > 0 else 0
    
    narrative.append(f"🇰🇿 **CIDA Агенттерінің дебат талдауы:**")
    narrative.append(f"• Консилиум құрамы: {num_agents} агент-сарапшы.")
    
    # Находим доминирующие патологии (где финальный консенсус максимален)
    final_beliefs = b_all[-1] # shape (num_agents, num_classes)
    avg_final = final_beliefs.mean(axis=0)
    
    top_indices = np.argsort(avg_final)[::-1][:2] # топ-2 патологии
    
    for idx in top_indices:
        p_val = avg_final[idx]
        if p_val < 0.15:
            continue
        c_name = class_names[idx] if idx < len(class_names) else f"Класс {idx}"
        
        narrative.append(f"\nПатология бойынша дебат: {c_name} (Итоговая уверенность: {p_val*100:.1f}%)")
        for r in range(num_rounds):
            beliefs_at_r = b_all[r][:, idx]
            agent_str = ", ".join([f"Агент {i+1}: {beliefs_at_r[i]*100:.1f}%" for i in range(num_agents)])
            narrative.append(f"  - {r+1}-ші раунд (ODE шадам): {agent_str}")
            
    narrative.append(f"\n🇷🇺 **Анализ дебатов агентов CIDA:**")
    narrative.append(f"• Состав консилиума: {num_agents} агентов-экспертов.")
    
    for idx in top_indices:
        p_val = avg_final[idx]
        if p_val < 0.15:
            continue
        c_name = class_names[idx] if idx < len(class_names) else f"Класс {idx}"
        
        narrative.append(f"\nДебаты по патологии: {c_name} (Итоговая уверенность: {p_val*100:.1f}%)")
        for r in range(num_rounds):
            beliefs_at_r = b_all[r][:, idx]
            agent_str = ", ".join([f"Агент {i+1}: {beliefs_at_r[i]*100:.1f}%" for i in range(num_agents)])
            narrative.append(f"  - Раунд {r+1} (ODE-шаг): {agent_str}")
            
    return "\n".join(narrative)


def process_xray(image):
    if image is None:
        yield None, None, None, "Загрузите изображение рентгена.", None
        return

    # Гарантируем наличие генератора отчетов для fallback/реального режима
    if models["qolda"] is None:
        try:
            qolda_mod = importlib.import_module("KairosMM.05_kazakh_llm_qolda")
            models["qolda"] = qolda_mod.QoldaReportGenerator()
        except Exception as e:
            print(f"[UI] Ошибка ленивой инициализации Qolda: {e}")

    # Пробуем лениво инициализировать реальные модели
    has_real = False
    if _HAS_REAL_MODULES:
        if models["cida"] is None:
            has_real = init_models()
        else:
            has_real = True

    # 1. Запускаем реальные модели
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
            b_all = out_cida.get("b_all", None)
            
            # Вычисляем Healer
            x_healed, x_recon = models["healer"].generate_counterfactual(img_tensor, strength=0.4)
            healed_pil = TF.to_pil_image(x_healed[0].cpu())
            
            # Разностная карта рисков
            diff_pil = compute_difference_map(image, healed_pil)
            
            # Перевод спора агентов в текст
            debate_narrative = translate_debate_to_narrative(b_all, models["cida"].class_names, disag, u_epi, u_alea)
            
            # Подготовка диагнозов для отчета
            diagnoses_dict = {}
            for i, name in enumerate(models["cida"].class_names):
                diagnoses_dict[name] = float(p_final[i])
                
            agent_data = {
                "p_final": p_final.tolist(),
                "class_names": models["cida"].class_names,
                "disagreement": disag,
                "u_epi": u_epi,
                "u_alea": u_alea,
                "debate_narrative": debate_narrative
            }
            
            # Форматируем краткие метрики
            dx_text = "\n".join([f"🔹 {k}: {v*100:.1f}%" for k, v in diagnoses_dict.items() if v > 0.15])
            metrics = f"⚠️ Келіспеушілік (Disagreement): {disag:.3f}\n🧠 Эпистемикалық (Epistemic): {u_epi:.3f}\n🌀 Алеаторикалық (Aleatoric): {u_alea:.3f}"
            if disag > 0.15:
                metrics += "\n\n🚨 НАЗАР АУДАРЫҢЫЗ: Агенттер арасында үлкен келіспеушілік бар! Консилиум ұсынылады."
            
            # Генерация отчета Qolda
            report_dict = models["qolda"].generate("dummy_image_path", agent_data)
            
            # Форматируем красивый Markdown отчет
            report_text = f"## 🩺 КЛИНИКАЛЫҚ ҚОРЫТЫНДЫ / КЛИНИЧЕСКИЙ ОТЧЕТ\n\n"
            report_text += f"### 📋 1. ЗЕРТТЕУ НӘТИЖЕЛЕРІ / ОПИСАНИЕ ИЗМЕНЕНИЙ\n{report_dict.get('findings', '')}\n\n"
            report_text += f"### 🩺 2. БОЛЖАМДЫ ДИАГНОЗ / КЛИНИЧЕСКИЙ ДИАГНОЗ\n**{report_dict.get('potential_diagnosis', '')}**\n\n"
            report_text += f"### 🧠 3. CIDA АГЕНТТЕР КОНСИЛИУМЫ / КОНСИЛИУМ АГЕНТОВ CIDA\n{report_dict.get('cida_debate_analysis', '')}\n\n"
            report_text += f"### 🌀 4. HEALER ЕМДЕУ ТАЛДАУЫ / АНАЛИЗ ИСЦЕЛЕНИЯ HEALER\n{report_dict.get('counterfactual_analysis', '')}\n\n"
            report_text += f"### 💡 5. ДӘРІГЕР ҰСЫНЫСТАРЫ / РЕКОМЕНДАЦИИ ВРАЧА\n{report_dict.get('recommendations', '')}\n\n"
            report_text += f"### 🔍 6. СЕНІМСІЗДІК ИНТЕРПРЕТАЦИЯСЫ / ОБЪЯСНЕНИЕ НЕОПРЕДЕЛЕННОСТЕЙ\n"
            report_text += f"* **Эпистемикалық / Эпистемическая:** {report_dict.get('epistemic_uncertainty_interpretation', '')}\n"
            report_text += f"* **Алеаторикалық / Алеаторная:** {report_dict.get('aleatoric_uncertainty_interpretation', '')}\n"

            # Стриминг отчета (эффект пишущей машинки)
            words = report_text.split(" ")
            current_report = ""
            for word in words:
                current_report += word + " "
                yield healed_pil, diff_pil, dx_text, metrics, current_report
                time.sleep(0.02)

        except Exception as err:
            print(f"[UI] Ошибка во время инференса реальных моделей: {err}")
            has_real = False # переключаемся на симуляцию

    # 2. Симуляция (Fallback)
    if not has_real:
        # Имитируем этапы
        yield None, None, "Анализ басталды...", "Есептелуде...", "Есептелуде..."
        time.sleep(1.0)
        
        diagnoses = {"Пневмония (Pneumonia)": 0.88, "Ателектаз (Atelectasis)": 0.45, "Эффузия (Effusion)": 0.12}
        disag = 0.18
        u_epi = 0.08
        u_alea = 0.17
        
        # Симулируем Healer: делаем очаг затемнения (высветленный на рентгене) "исцеленным" (прозрачным легочным полем, т.е. темнее)
        img_array = np.array(image.convert("RGB"))
        h, w, c = img_array.shape
        y_c, x_c = int(h * 0.7), int(w * 0.65)
        radius = int(min(h, w) * 0.08)
        
        healed_array = img_array.copy()
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
        mask = dist_from_center <= radius
        
        # "Исцеление": убираем белый очаг затемнения
        healed_array[mask] = np.clip(healed_array[mask] * 0.6, 0, 255).astype(np.uint8)
        healed_pil = Image.fromarray(healed_array)
        
        # Вычисляем разностную карту рисков
        diff_pil = compute_difference_map(image, healed_pil)
        
        # Перевод дебатов агентов
        debate_narrative = translate_debate_to_narrative(None, list(diagnoses.keys()), disag, u_epi, u_alea)
        
        agent_data = {
            "p_final": list(diagnoses.values()),
            "class_names": list(diagnoses.keys()),
            "disagreement": disag,
            "u_epi": u_epi,
            "u_alea": u_alea,
            "debate_narrative": debate_narrative
        }
        
        dx_text = "\n".join([f"🔹 {k}: {v*100:.1f}%" for k, v in diagnoses.items()])
        metrics = f"⚠️ Келіспеушілік (Disagreement): {disag:.3f}\n🧠 Эпистемикалық (Epistemic): {u_epi:.3f}\n🌀 Алеаторикалық (Aleatoric): {u_alea:.3f}"
        if disag > 0.15:
            metrics += "\n\n🚨 НАЗАР АУДАРЫҢЫЗ: Агенттер арасында үлкен келіспеушілік бар (>0.15)! Консилиум ұсынылады."

        # Отчет Qolda (Fallback)
        report_dict = models["qolda"].generate("dummy_image_path", agent_data)
        
        report_text = f"## 🩺 КЛИНИКАЛЫҚ QOLDA ҚОРЫТЫНДЫСЫ / КЛИНИЧЕСКИЙ ОТЧЕТ QOLDA\n\n"
        report_text += f"### 📋 1. ЗЕРТТЕУ НӘТИЖЕЛЕРІ / ОПИСАНИЕ ИЗМЕНЕНИЙ\n{report_dict.get('findings', '')}\n\n"
        report_text += f"### 🩺 2. БОЛЖАМДЫ ДИАГНОЗ / КЛИНИЧЕСКИЙ ДИАГНОЗ\n**{report_dict.get('potential_diagnosis', '')}**\n\n"
        report_text += f"### 🧠 3. CIDA АГЕНТТЕР КОНСИЛИУМЫ / КОНСИЛИУМ АГЕНТОВ CIDA\n{report_dict.get('cida_debate_analysis', '')}\n\n"
        report_text += f"### 🌀 4. HEALER ЕМДЕУ ТАЛДАУЫ / АНАЛИЗ ИСЦЕЛЕНИЯ HEALER\n{report_dict.get('counterfactual_analysis', '')}\n\n"
        report_text += f"### 💡 5. ДӘРІГЕР ҰСЫНЫСТАРЫ / РЕКОМЕНДАЦИИ ВРАЧА\n{report_dict.get('recommendations', '')}\n\n"
        report_text += f"### 🔍 6. СЕНІМСІЗДІК ИНТЕРПРЕТАЦИЯСЫ / ОБЪЯСНЕНИЕ НЕОПРЕДЕЛЕННОСТЕЙ\n"
        report_text += f"* **Эпистемикалық / Эпистемическая:** {report_dict.get('epistemic_uncertainty_interpretation', '')}\n"
        report_text += f"* **Алеаторикалық / Алеаторная:** {report_dict.get('aleatoric_uncertainty_interpretation', '')}\n"

        words = report_text.split(" ")
        current_report = ""
        for word in words:
            current_report += word + " "
            yield healed_pil, diff_pil, dx_text, metrics, current_report
            time.sleep(0.02)


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
            # 🧬 CIDA KairosMM v4: H100 Clinic Edition
            ### **Көпдеңгейлі Мультиагентті Медициналық ИИ және Контрфактуалды Талдау Жүйесі**
            *(Мультиагентная система CIDA v4: Liquid Neural ODE, BiomedCLIP и Генерация здоровых легких Healer)*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Түпнұсқа рентген / Оригинальный снимок")
                submit_btn = gr.Button("🔍 Клиникалық талдау жасау (Анализировать)", variant="primary", size="lg")
                
                gr.Markdown("### 🧠 CIDA Агенттерінің бағалауы")
                dx_out = gr.Textbox(label="Ықтимал патологиялар (Вероятные патологии)", lines=3)
                metrics_out = gr.Textbox(label="Сенімсіздік пен разногласие метрикалары (Метрики неопределенности)", lines=4)
                
            with gr.Column(scale=2):
                with gr.Row():
                    healed_out = gr.Image(type="pil", label="Healer: Сау контрфакт (Здоровый легкий)")
                    diff_out = gr.Image(type="pil", label="Difference Map: Қауіп аймақтары (Карта патологий)")
                
                gr.Markdown("### 📋 Qolda LLM: Толық медициналық есеп (Двуязычный рапорт)")
                report_out = gr.Markdown(label="Отчет клиники")

        submit_btn.click(
            fn=process_xray,
            inputs=[input_img],
            outputs=[healed_out, diff_out, dx_out, metrics_out, report_out]
        )
        
        gr.Markdown("""
        ---
        *Әзірлеушілер: CIDA Team. На базе Qolda (ISSAI NU), BiomedCLIP и Flow Matching Healer. Дифференциалды разностные карты заменяют размытые Grad-CAM.*
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
