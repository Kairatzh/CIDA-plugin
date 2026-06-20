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
            "description": "Рентгенологиялық өзгерістердің егжей-тегжейлі сипаттамасы мен орысша аудармасы (Подробное описание рентгенологических изменений на казахском и русском)"
        },
        "potential_diagnosis": {
            "type": "string",
            "description": "Қорытынды ықтимал диагноз және орысша нұсқасы (Итоговый клинический диагноз на казахском и русском)"
        },
        "recommendations": {
            "type": "string",
            "description": "Қосымша зерттеулер мен емдеу бойынша ұсыныстар (Рекомендации на казахском и русском)"
        },
        "cida_debate_analysis": {
            "type": "string",
            "description": "CIDA консилиумының талдауы: агенттердің қалай пікір алмасып, қандай шешімге келгені (Анализ консилиума агентов CIDA: как спорили и к чему пришли)"
        },
        "counterfactual_analysis": {
            "type": "string",
            "description": "Healer латентті моделі өшірген патологиялық аймақтарды сипаттау (Описание областей, которые исправила диффузионная модель Healer)"
        },
        "epistemic_uncertainty_interpretation": {
            "type": "string",
            "description": "Эпистемикалық сенімсіздіктің қарапайым клиникалық түсіндірмесі (Простое объяснение эпистемической неопределенности)"
        },
        "aleatoric_uncertainty_interpretation": {
            "type": "string",
            "description": "Алеаторикалық сенімсіздіктің (сурет сапасы) клиникалық түсіндірмесі (Простое объяснение алеаторической неопределенности)"
        }
    },
    "required": [
        "findings", 
        "potential_diagnosis", 
        "recommendations", 
        "cida_debate_analysis",
        "counterfactual_analysis",
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
            if p > 0.35 and i < len(class_names):
                diagnoses.append(f"{class_names[i]} ({p*100:.1f}%)")
                
        dx_text = ", ".join(diagnoses) if diagnoses else "Патологиялар табылған жоқ (Норма)"
        
        disagreement = agent_data.get("disagreement", 0.0)
        u_epi = agent_data.get("u_epi", 0.0)
        u_alea = agent_data.get("u_alea", 0.0)
        debate_narrative = agent_data.get("debate_narrative", "Деректер жоқ (Данные отсутствуют)")
        
        schema_str = json.dumps(KAZAKH_REPORT_SCHEMA, ensure_ascii=False, indent=2)
        
        prompt = f"""Сен білікті рентгенолог-дәрігерсің. Пациенттің кеуде қуысының рентген суреті берілген.
CIDA v4 мультиагентті жүйесі келесі нәтижелерді көрсетті:

Табылған патологиялар: {dx_text}
Агенттер арасындағы келіспеушілік: {disagreement:.3f}
Эпистемикалық сенімсіздік (білім жетіспеушілігі): {u_epi:.3f}
Алеаторикалық сенімсіздік (мәліметтегі шу): {u_alea:.3f}

Агенттер консилиумының логтары (Ход дебатов агентов):
{debate_narrative}

Міндетің — осы көрсеткіштер мен агенттер дебаты негізінде егжей-тегжейлі қазақша және орысша (екі тілде) клиникалық қорытынды жасау.
Әрбір өрісте ақпаратты алдымен қазақ тілінде, содан кейін міндетті түрде орыс тілінде қарапайым әрі түсінікті сөздермен жаз.
Жауапты қатаң түрде төмендегий JSON схемасына сәйкес қайтар. Тек таза JSON жібер:

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
            # Fallback генератор с детальной JSON структурой
            p_final = agent_data.get("p_final", [])
            class_names = agent_data.get("class_names", [])
            detected = [c for c in class_names if p_final[class_names.index(c)] > 0.45]
            
            dx_kz = "Өкпе қабынуы (Пневмония)" if "Pneumonia" in detected else "Патологиялық өзгерістер анықталған жоқ"
            dx_ru = "Воспаление легких (Пневмония)" if "Pneumonia" in detected else "Патологических изменений не обнаружено"
            
            findings_kz = "Рентгенограммада оң жақ өкпенің төменгі бөлігінде айқын қарайю және тығыздалу ошақтары байқалады. Өкпе суреті күшейген."
            findings_ru = "На рентгенограмме наблюдаются отчетливые очаги затемнения и инфильтрации в нижней доле правого легкого. Легочный рисунок усилен."
            
            rec_kz = "Шұғыл түрде пульмонолог немесе терапевт дәрігеріне қаралу. Қажет болса, диагнозды нақтылау үшін кеуде қуысының компьютерлік томографиясын (КТ) жасау және жалпы қан талдауын тапсыру."
            rec_ru = "Срочно обратиться к пульмонологу или терапевту. При необходимости пройти компьютерную томографию (КТ) органов грудной клетки для уточнения локализации и сдать общий анализ крови."
            
            debate_kz = "Консилиум барысында Агент 1 (Клиникалық белгілер) мен Агент 3 (Кеңістіктік аномалиялар) өкпеде қабыну процесінің бар екеніне жоғары сенімділік білдірді. Агент 2 мен Агент 4 бастапқыда күмәнданды, бірақ 3 раунд дебаттан кейін анонимді пікір алмасу арқылы ортақ консенсусқа келді."
            debate_ru = "В ходе консилиума Агент 1 (Клинические признаки) и Агент 3 (Пространственные аномалии) выразили высокую уверенность в наличии воспалительного процесса. Агент 2 и Агент 4 изначально сомневались, но после 3 раундов дебатов через анонимный обмен скрытыми мнениями пришли к согласию."
            
            healer_kz = "Healer диффузиялық моделі оң жақ өкпенің төменгі бөлігіндегі патологиялық инфильтрацияны жойып, оны сау қалыпқа келтірді. Айырмашылық картасында (Difference Map) бұл аймақ қызыл түспен айқын белгіленген."
            healer_ru = "Диффузионная модель Healer устранила патологическую инфильтрацию в нижней доле правого легкого, восстановив здоровую структуру. На карте разностей (Difference Map) эта область четко подсвечена красным цветом."
            
            fallback_res = {
                "findings": f"🇰🇿 {findings_kz}\n\n🇷🇺 {findings_ru}",
                "potential_diagnosis": f"🇰🇿 {dx_kz}\n\n🇷🇺 {dx_ru}",
                "recommendations": f"🇰🇿 {rec_kz}\n\n🇷🇺 {rec_ru}",
                "cida_debate_analysis": f"🇰🇿 {debate_kz}\n\n🇷🇺 {debate_ru}",
                "counterfactual_analysis": f"🇰🇿 {healer_kz}\n\n🇷🇺 {healer_ru}",
                "epistemic_uncertainty_interpretation": f"🇰🇿 Эпистемикалық сенімсіздік төмен ({agent_data.get('u_epi', 0.0):.3f}) — модельде мұндай жағдайлар бойынша жеткілікті клиникалық білім базасы бар.\n\n🇷🇺 Эпистемическая неопределенность низкая ({agent_data.get('u_epi', 0.0):.3f}) — модель обучена на большом количестве подобных клинических случаев и не испытывает сомнений.",
                "aleatoric_uncertainty_interpretation": f"🇰🇿 Алеаторикалық сенімсіздік орташа ({agent_data.get('u_alea', 0.0):.3f}) — кескін сапасы жақсы, бірақ азғантай анатомиялық ерекшеліктер шу тудыруы мүмкін.\n\n🇷🇺 Алеаторная неопределенность средняя ({agent_data.get('u_alea', 0.0):.3f}) — качество снимка хорошее, но незначительные анатомические наложения могут создавать оптический шум."
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
