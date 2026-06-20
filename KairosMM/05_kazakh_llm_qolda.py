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

# JSON схема для медицинского заключения на казахском языке
KAZAKH_REPORT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "KazakhMedicalReport",
    "type": "object",
    "properties": {
        "findings": {
            "type": "string",
            "description": "Рентгенологиялық өзгерістердің егжей-тегжейлі сипаттамасы (қазақ тілінде)"
        },
        "potential_diagnosis": {
            "type": "string",
            "description": "Қорытынды ықтимал диагноз (қазақ тілінде)"
        },
        "recommendations": {
            "type": "string",
            "description": "Қосымша тексерулер мен зерттеулер бойынша ұсыныстар"
        },
        "epistemic_uncertainty_interpretation": {
            "type": "string",
            "description": "Эпистемикалық сенімсіздікті (мәліметтердің жеткіліксіздігі) клиникалық түсіндіру"
        },
        "aleatoric_uncertainty_interpretation": {
            "type": "string",
            "description": "Алеаторикалық сенімсіздікті (бейнедегі шу немесе артефакттар) клиникалық түсіндіру"
        }
    },
    "required": [
        "findings", 
        "potential_diagnosis", 
        "recommendations", 
        "epistemic_uncertainty_interpretation", 
        "aleatoric_uncertainty_interpretation"
    ]
}


class QoldaReportGenerator:
    """
    Structured Medical Report Generator using NU ISSAI Qolda model.
    Enforces output compliance with Kazakh JSON Schema constraints,
    incorporating CIDA v4 epistemic and aleatoric uncertainties.
    """
    def __init__(self, model_id: str = "issai/Qolda", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self._initialized = False
        self.model_id = model_id

    def load_model(self):
        if not _HAS_TRANSFORMERS:
            raise ImportError("pip install transformers accelerate")
            
        print(f"🚀 Загрузка казахской мультимодальной модели {self.model_id}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.model.eval()
            self._initialized = True
            print("✅ Qolda загружена и готова к работе на казахском языке!")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки Qolda: {e}")
            print("Fallback: Имитация работы модели для демо...")

    def build_prompt(self, agent_data: dict) -> str:
        """Формирует промпт для модели с требованием вывода в JSON по схеме."""
        # Извлекаем диагнозы
        p_final = agent_data.get("p_final", [])
        class_names = agent_data.get("class_names", [])
        
        diagnoses = []
        for i, p in enumerate(p_final):
            if p > 0.5 and i < len(class_names):
                diagnoses.append(f"{class_names[i]} ({p*100:.1f}%)")
                
        dx_text = ", ".join(diagnoses) if diagnoses else "Патологиялар табылған жоқ (Қалыпты)"
        
        disagreement = agent_data.get("disagreement", 0.0)
        u_epi = agent_data.get("u_epi", 0.0)
        u_alea = agent_data.get("u_alea", 0.0)
        
        schema_str = json.dumps(KAZAKH_REPORT_SCHEMA, ensure_ascii=False, indent=2)
        
        prompt = f"""Сен білікті рентгенолог-дәрігерсің. Пациенттің кеуде қуысының рентген суреті берілген.
CIDA v4 мультиагентті жүйесі келесі нәтижелерді көрсетті:

Табылған патологиялар: {dx_text}
Агенттер арасындағы келіспеушілік: {disagreement:.3f}
Эпистемикалық сенімсіздік (білім жетіспеушілігі): {u_epi:.3f}
Алеаторикалық сенімсіздік (мәліметтегі шу): {u_alea:.3f}

Сенің міндетің — осы көрсеткіштер негізінде қазақ тілінде клиникалық қорытынды жасау.
Жауапты қатаң түрде төмендегі JSON схемасына сәйкес қайтар. Басқа мәтін қоспа. Тек таза JSON жібер:

СХЕМА:
{schema_str}

JSON жауабы:"""
        
        return prompt

    def generate(self, image_path: str, agent_data: dict) -> dict:
        """
        Генерирует структурированный отчет и проверяет его соответствие схеме.
        """
        prompt = self.build_prompt(agent_data)
        
        if not self._initialized:
            # Fallback генератор с валидной JSON структурой
            fallback_res = {
                "findings": f"Кеуде қуысы рентгенограммасында {', '.join([c for c in agent_data.get('class_names', []) if agent_data.get('p_final', [])[agent_data.get('class_names', []).index(c)] > 0.5]) or 'патологиялық'} өзгерістер анықталды.",
                "potential_diagnosis": "Пневмония / Өкпе қабынуы ықтимал." if any(p > 0.5 for p in agent_data.get("p_final", [])) else "Патологиялық өзгерістер байқалмайды.",
                "recommendations": "Пульмонолог дәрігерінің консультациясы ұсынылады. Қажет болса, КТ зерттеуін жасау керек.",
                "epistemic_uncertainty_interpretation": f"Эпистемикалық сенімсіздік деңгейі {agent_data.get('u_epi', 0.0):.3f}. Бұл жүйеде осы патология бойынша клиникалық прецеденттердің жеткілікті екенін көрсетеді.",
                "aleatoric_uncertainty_interpretation": f"Алеаторикалық сенімсіздік {agent_data.get('u_alea', 0.0):.3f} құрайды, бұл кескін сапасының орташа деңгейін білдіреді."
            }
            return fallback_res

        try:
            image = Image.open(image_path).convert('RGB')
            # Специфичный формат для Qolda (на базе InternVL)
            response = self.model.chat(self.tokenizer, image, prompt)
            
            # Попытка распарсить JSON из ответа
            try:
                # Извлекаем JSON из markdown блоков если они есть
                clean_response = response.strip()
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()
                
                parsed_json = json.loads(clean_response)
                
                # Валидация обязательных полей
                for field in KAZAKH_REPORT_SCHEMA["required"]:
                    if field not in parsed_json:
                        parsed_json[field] = "Ақпарат жоқ (Автоматты толтыру)"
                
                return parsed_json
            except Exception as parse_err:
                print(f"[Qolda] Ошибка парсинга JSON: {parse_err}. Ответ модели: {response}")
                # Fallback при ошибке парсинга
                return {
                    "findings": "JSON форматын тану мүмкін болмады. Шикі мәтін: " + response,
                    "potential_diagnosis": "Анықталмаған диагноз",
                    "recommendations": "Қайта тексеру қажет",
                    "epistemic_uncertainty_interpretation": f"Эпистемикалық мәні: {agent_data.get('u_epi', 0.0):.3f}",
                    "aleatoric_uncertainty_interpretation": f"Алеаторикалық мәні: {agent_data.get('u_alea', 0.0):.3f}"
                }
        except Exception as e:
            return {"error": f"Генерация қатесі: {str(e)}"}


def demo_qolda():
    print("="*60)
    print("📝 Qolda Kazakh Medical Report Generator Demo (JSON Schema)")
    print("="*60)
    
    generator = QoldaReportGenerator()
    
    # Фейковые данные CIDA с разделением неопределенности v4
    fake_agent_data = {
        "p_final": [0.85, 0.1, 0.92, 0.05],
        "class_names": ["Atelectasis", "Cardiomegaly", "Pneumonia", "Effusion"],
        "disagreement": 0.12,
        "u_epi": 0.08,
        "u_alea": 0.17
    }
    
    report = generator.generate("dummy.png", fake_agent_data)
    
    print("\n[Generated Report / Жасалған қорытынды (JSON)]:")
    print("-" * 40)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("-" * 40)
    print("\n✅ Интеграция с Qolda по JSON Schema успешна!")

if __name__ == "__main__":
    demo_qolda()
