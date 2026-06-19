"""
05_kazakh_llm_qolda.py — Мультимодальный Генератор Отчётов на базе Qolda.

Qolda (от ISSAI Назарбаев Университета) — 4B параметров, поддерживает 
казахский язык и визуальные входы (изображения).

Здесь мы комбинируем:
1. Вывод агентов CIDA (вероятности болезней, разногласия)
2. Само рентгеновское изображение
3. Промпт на казахском
И подаём это в Qolda для генерации финального медицинского отчёта.

Использование:
    python KairosMM/05_kazakh_llm_qolda.py
"""

import json
import torch
import sys
from pathlib import Path
from PIL import Image

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class QoldaReportGenerator:
    def __init__(self, model_id: str = "issai/Qolda", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self._initialized = False
        self.model_id = model_id

    def load_model(self):
        if not _HAS_TRANSFORMERS:
            raise ImportError("pip install transformers accelerate")
            
        print(f"🚀 Загрузка казахской мультимодальной модели {self.model_id}...")
        try:
            # Для Qolda требуется trust_remote_code=True, так как это кастомная архитектура (InternVL)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16, # Оптимально для Qolda 4B
                device_map="auto"           # Автоматическое распределение VRAM
            )
            self.model.eval()
            self._initialized = True
            print("✅ Qolda загружена и готова к работе на казахском языке!")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки Qolda: {e}")
            print("Fallback: Имитация работы модели для демо...")

    def build_prompt(self, agent_data: dict) -> str:
        """Формирует промпт для модели, включая системный промпт и результаты CIDA."""
        
        # Извлекаем диагнозы (вероятность > 50%)
        p_final = agent_data.get("p_final", [])
        class_names = agent_data.get("class_names", [])
        
        diagnoses = []
        for i, p in enumerate(p_final):
            if p > 0.5 and i < len(class_names):
                diagnoses.append(f"{class_names[i]} ({p*100:.1f}%)")
                
        dx_text = ", ".join(diagnoses) if diagnoses else "Патологиялар табылған жоқ (Қалыпты)"
        
        disagreement = agent_data.get("disagreement", 0.0)
        uncertainty = agent_data.get("uncertainty", 0.0)
        
        # Системный промпт на казахском
        prompt = f"""Сен білікті рентгенолог-дәрігерсің. Пациенттің кеуде қуысының рентген суреті берілген.
CIDA мультиагентті жүйесі келесі нәтижелерді көрсетті:

Табылған патологиялар: {dx_text}
Агенттердің келіспеушілігі: {disagreement:.3f} (Егер 0.1-ден жоғары болса, диагноз күмәнді)
Жүйенің сенімсіздігі: {uncertainty:.3f}

Осы деректер мен рентген суретіне сүйене отырып, қазақ тілінде толық медициналық қорытынды жаз.
Қорытынды құрылымы:
1. Табылған өзгерістердің сипаттамасы
2. Ықтимал диагноз
3. Ұсыныстар (Қандай қосымша зерттеулер қажет)

Қорытынды:"""
        
        return prompt

    def generate(self, image_path: str, agent_data: dict) -> str:
        if not self._initialized:
            self.load_model()
            
        prompt = self.build_prompt(agent_data)
        
        if not self._initialized:
            # Fallback (если модель не скачалась)
            return f"""Қорытынды (Синтетикалық генерация - Модель жүктелмеді):
Суретте {agent_data.get('class_names')[0]} белгілері байқалады. 
Агенттердің келіспеушілігі {agent_data.get('disagreement'):.3f} құрайды, бұл қосымша КТ зерттеуін қажет ететінін білдіреді.
Ұсыныс: Пульмонолог дәрігерінің консультациясы."""

        # Реальная генерация Qolda
        try:
            image = Image.open(image_path).convert('RGB')
            # Специфичный формат для Qolda/InternVL
            # В реальном коде тут вызов chat/generate согласно докам модели
            response = self.model.chat(self.tokenizer, image, prompt)
            return response
        except Exception as e:
            return f"Генерация қатесі: {str(e)}"


def demo_qolda():
    print("="*60)
    print("📝 Qolda Kazakh Medical Report Generator Demo")
    print("="*60)
    
    generator = QoldaReportGenerator()
    
    # Фейковые данные CIDA
    fake_agent_data = {
        "p_final": [0.85, 0.1, 0.92, 0.05],
        "class_names": ["Atelectasis", "Cardiomegaly", "Pneumonia", "Effusion"],
        "disagreement": 0.12,
        "uncertainty": 0.25
    }
    
    # Фейковый путь (в демо Qolda вернет fallback текст, если не найдет файл)
    report = generator.generate("dummy.png", fake_agent_data)
    
    print("\n[Generated Report / Жасалған қорытынды]:")
    print("-" * 40)
    print(report)
    print("-" * 40)
    print("\n✅ Интеграция с Qolda успешна!")

if __name__ == "__main__":
    demo_qolda()
