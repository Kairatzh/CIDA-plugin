import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers import AutoencoderKL
    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False


class LatentFlowVelocityField(nn.Module):
    """
    v_theta(z, t) — векторное поле для Flow Matching в латентном пространстве VAE.
    Принимает z: (B, 4, 28, 28) и t: скаляр/тензор.
    Предсказывает направление изменения латентного представления для "исцеления".
    """
    def __init__(self, channels=4, height=28, width=28):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.d_latent = channels * height * width

        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.time_proj = nn.Linear(64, 1)
        # Сверточный предиктор направления скорости
        self.conv1 = nn.Conv2d(channels + 1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, channels, kernel_size=3, padding=1)
        
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape
        # Подготовка временного слоя
        t_val = t.view(1, 1).expand(B, 1) if t.dim() == 0 else t.view(B, 1)
        t_emb = self.time_proj(self.time_embed(t_val)).view(B, 1, 1, 1).expand(B, 1, H, W)
        
        # Объединяем латентное состояние и время
        x = torch.cat([z, t_emb], dim=1)
        h = F.silu(self.conv1(x))
        h = F.silu(self.conv2(h))
        v = self.conv3(h)
        return v


class CounterfactualHealerAgent(nn.Module):
    """
    Level 3 Causal Agent (v4): Generates healthy counterfactuals directly in VAE latent space.
    Использует AutoencoderKL для кодирования/декодирования и Flow Matching для "исцеления".
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self._initialized = False

    def load_pipeline(self):
        if not _HAS_DIFFUSERS:
            raise ImportError("pip install diffusers transformers accelerate")
            
        print("[HealerAgent] Загрузка VAE и инициализация Flow Matching...")
        # Загружаем VAE (stabilityai/sd-vae-ft-mse)
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.vae.eval()

        # Инициализируем латентное векторное поле
        self.flow_v = LatentFlowVelocityField(channels=4, height=28, width=28).to(self.device)
        
        # Загружаем предобученные веса
        from pathlib import Path
        weights_path = Path("./checkpoints/latent_flow_healer.pth")
        if weights_path.exists():
            print(f"[HealerAgent] Загрузка обученных весов Flow Matching Healer из {weights_path}...")
            # Приводим веса к нужному типу перед загрузкой
            state_dict = torch.load(weights_path, map_location=self.device)
            self.flow_v.load_state_dict(state_dict)
        else:
            print("[HealerAgent] [WARN] Файл весов latent_flow_healer.pth не найден. Используем случайную инициализацию.")

        if self.device == "cuda":
            self.flow_v = self.flow_v.half()

        self._initialized = True

    def flow_ode(self, t, z):
        """Интеграционная функция для ODE решателя."""
        # z: (B, 4, 28, 28)
        # Возвращает dz/dt
        return self.flow_v(z, t)

    def generate_counterfactual(
        self, 
        image: torch.Tensor, 
        strength: float = 0.4,
        num_steps: int = 15
    ):
        """
        Генерирует здоровый контрфакт из входного рентгена в латентном пространстве VAE.
        image: (B, 3, 224, 224)
        strength: сила исцеления (доля шагов интеграции)
        """
        if not self._initialized:
            self.load_pipeline()

        # Превращаем в правильный тензор
        if isinstance(image, torch.Tensor):
            x = image.to(self.device)
            if x.dim() == 3:
                x = x.unsqueeze(0)
        else:
            # Если это PIL Image
            from torchvision.transforms import functional as TF
            x = TF.to_tensor(image).unsqueeze(0).to(self.device)

        # Приводим к размеру 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        x = x.to(dtype)

        with torch.no_grad():
            # 1. Кодируем в латентное пространство
            # z ∈ R^(B, 4, 28, 28)
            posterior = self.vae.encode(x)
            z = posterior.latent_dist.sample() * 0.18215

            # 2. Выполняем Flow Matching (интеграция ODE)
            # Мы интегрируем от t=0 (исходное больное состояние) до t=strength (здоровое состояние)
            t_span = torch.linspace(0.0, strength, num_steps, device=self.device, dtype=dtype)
            
            # Простая Euler-интеграция для скорости на H100
            dt = t_span[1] - t_span[0]
            z_healed = z.clone()
            for step in range(len(t_span) - 1):
                t_curr = t_span[step]
                v = self.flow_v(z_healed, t_curr)
                z_healed = z_healed + v * dt

            # 3. Декодируем обратно в пиксельное пространство
            x_healed = self.vae.decode(z_healed / 0.18215).sample
            # Клипаем в диапазон [0, 1]
            x_healed = torch.clamp((x_healed + 1.0) / 2.0, 0.0, 1.0)

            # Декодируем и оригинальное латентное представление для честного сравнения
            x_recon = self.vae.decode(z / 0.18215).sample
            x_recon = torch.clamp((x_recon + 1.0) / 2.0, 0.0, 1.0)

        return x_healed, x_recon

    def compute_anomaly_map(self, original_tensor: torch.Tensor, healed_tensor: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет карту аномалий как попиксельную разницу между оригиналом и контрфактом.
        """
        # Считаем L2 расстояние
        diff = torch.abs(original_tensor - healed_tensor).mean(dim=1, keepdim=True)
        # Нормализуем карту аномалий
        B, C, H, W = diff.shape
        diff_flat = diff.view(B, -1)
        d_min = diff_flat.min(dim=-1, keepdim=True).values
        d_max = diff_flat.max(dim=-1, keepdim=True).values
        diff_norm = (diff_flat - d_min) / (d_max - d_min + 1e-8)
        return diff_norm.view(B, 1, H, W)


def demo_healer():
    print("="*60)
    print("🧬 Latent Flow Matching Counterfactual Healer Demo")
    print("="*60)
    
    if not _HAS_DIFFUSERS:
        print("Diffusers не установлен. Запустите: pip install diffusers")
        return
        
    try:
        # Создаем dummy image (шум)
        dummy_img = torch.randn(1, 3, 224, 224)
        healer = CounterfactualHealerAgent()
        
        # Генерируем "исцеленный" снимок
        x_healed, x_recon = healer.generate_counterfactual(dummy_img, strength=0.4)
        
        # Считаем разницу
        heatmap = healer.compute_anomaly_map(x_recon, x_healed)
        
        print(f"\n✅ Успех! Latent Counterfactual сгенерирован: {x_healed.shape}")
        print(f"✅ Anomaly Map вычислена: {heatmap.shape}, Max diff: {heatmap.max().item():.2f}")
        
    except Exception as e:
        print(f"Ошибка во время демо: {e}")

if __name__ == "__main__":
    demo_healer()
