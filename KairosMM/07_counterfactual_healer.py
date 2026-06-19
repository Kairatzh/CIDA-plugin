"""
07_counterfactual_healer.py — Агент-Целитель (Generative Counterfactuals).

Этот скрипт реализует 5-го агента в системе CIDA: "Healer".
В отличие от обычных классифицирующих агентов, Healer использует 
Stable Diffusion (или Flow Matching), чтобы сгенерировать контрфактический
рентгеновский снимок: "Как бы выглядели эти лёгкие, если бы они были здоровы?".

Разница между (Реальный Снимок - Здоровый Контрфакт) является 
высокоинтерпретируемой картой патологии (Anomaly Map) уровня 3 по Judea Pearl.

Использование:
    python KairosMM/07_counterfactual_healer.py
"""

import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL, UNet2DConditionModel
    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False


class CounterfactualHealerAgent(nn.Module):
    """
    Level 3 Causal Agent: Generates counterfactual "healthy" images
    from diseased images to provide visual evidence.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self._initialized = False

    def load_pipeline(self):
        """Ленивая загрузка SD pipeline для экономии VRAM."""
        if not _HAS_DIFFUSERS:
            raise ImportError("pip install diffusers transformers accelerate")
            
        print("[HealerAgent] Загрузка Stable Diffusion Pipeline...")
        # Используем базовый SD 1.5 для демо. В реальном H100 можно использовать SDXL или Med-SD
        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
            safety_checker=None
        ).to(self.device)
        
        # На H100 обязательно используем xformers / FlashAttention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[HealerAgent] xformers memory efficient attention enabled.")
        except Exception:
            pass
            
        self._initialized = True

    def generate_counterfactual(
        self, 
        image: torch.Tensor, 
        strength: float = 0.4,
        guidance_scale: float = 7.5
    ):
        """
        Генерирует здоровый контрфакт из входного рентгена.
        image: (B, 3, 224, 224) или PIL Image
        strength: 0.0 = нет изменений, 1.0 = полная перерисовка. 
                  0.3-0.5 идеально для сохранения анатомии, но "исцеления" болезни.
        """
        if not self._initialized:
            self.load_pipeline()
            
        # Конвертация tensor -> PIL (требование diffusers img2img)
        # В продакшене лучше работать прямо в латентном пространстве VAE
        if isinstance(image, torch.Tensor):
            import torchvision.transforms as T
            to_pil = T.ToPILImage()
            pil_image = to_pil(image[0].cpu() if image.dim() == 4 else image.cpu())
        else:
            pil_image = image

        # Prompt engineering для "исцеления"
        prompt = "Healthy clear lungs, normal chest x-ray, no pathology, sharp bone structure, high resolution medical scan"
        negative_prompt = "pneumonia, opacity, effusion, cardiomegaly, mass, nodule, blurred, artifacts, disease"

        print(f"[HealerAgent] Генерация контрфакта (strength={strength})...")
        # Для H100 можно использовать 50 шагов за секунду
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=30
        ).images[0]

        return result

    def compute_anomaly_map(self, original_pil, counterfactual_pil):
        """
        Вычисляет карту аномалий как разницу пикселей (простая версия).
        В реальности вычисляется L2 distance в латентном пространстве VAE.
        """
        import numpy as np
        orig = np.array(original_pil).astype(float)
        cf = np.array(counterfactual_pil).astype(float)
        
        # Абсолютная разница
        diff = np.abs(orig - cf)
        
        # Перевод в heatmap (один канал)
        heatmap = diff.mean(axis=2)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap


def demo_healer():
    print("="*60)
    print("🧬 Counterfactual Healer Agent Demo")
    print("="*60)
    
    if not _HAS_DIFFUSERS:
        print("Diffusers не установлен. Запустите: pip install diffusers")
        return
        
    try:
        from PIL import Image
        import numpy as np
        
        # Создаем dummy image (шум) для демо, если нет реального рентгена
        dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        
        healer = CounterfactualHealerAgent()
        
        # Генерируем "исцеленный" снимок
        cf_image = healer.generate_counterfactual(dummy_img, strength=0.4)
        
        # Считаем разницу
        heatmap = healer.compute_anomaly_map(dummy_img, cf_image)
        
        print(f"\n✅ Успех! Контрфакт сгенерирован: {cf_image.size}")
        print(f"✅ Anomaly Map вычислена: {heatmap.shape}, Max diff: {heatmap.max():.2f}")
        
    except Exception as e:
        print(f"Ошибка во время демо: {e}")
        print("Убедитесь, что у вас есть доступ к интернету для загрузки весов SD 1.5.")

if __name__ == "__main__":
    demo_healer()
