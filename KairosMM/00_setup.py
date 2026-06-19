"""
00_setup.py — Подготовка среды для KairosMM H100.

Запускать ДО основного обучения (можно на дешёвом инстансе).
Скачивает модели, устанавливает зависимости, проверяет GPU.

Использование:
    python KairosMM/00_setup.py                  # Полная установка
    python KairosMM/00_setup.py --check_only     # Только проверка
    python KairosMM/00_setup.py --download_models # Только скачивание моделей
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Fix for Windows UnicodeEncodeError with emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))


def check_gpu():
    """Проверяет наличие и тип GPU."""
    print("\n" + "=" * 60)
    print("🔍 Проверка GPU")
    print("=" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            cap = torch.cuda.get_device_capability(0)
            print(f"  ✅ GPU: {name}")
            print(f"  ✅ VRAM: {vram:.1f} GB")
            print(f"  ✅ Compute Capability: {cap[0]}.{cap[1]}")

            if "H100" in name or cap[0] >= 9:
                print("  🚀 H100 / Hopper обнаружен! Включаем все оптимизации.")
                return "h100"
            elif cap[0] >= 8:
                print("  ⚡ Ampere (A100/RTX 30xx/40xx) обнаружен.")
                return "ampere"
            else:
                print("  ⚠️  Старая архитектура GPU. Некоторые оптимизации недоступны.")
                return "legacy"
        else:
            print("  ⚠️  CUDA недоступен. Работаем на CPU.")
            return "cpu"
    except ImportError:
        print("  ❌ PyTorch не установлен!")
        return "none"


def check_dependencies():
    """Проверяет наличие ключевых библиотек."""
    print("\n" + "=" * 60)
    print("📦 Проверка зависимостей")
    print("=" * 60)
    deps = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "open_clip": "OpenCLIP (BiomedCLIP)",
        "diffusers": "Diffusers (Flow Matching)",
        "torchdiffeq": "TorchDiffEq (Liquid ODE)",
        "peft": "PEFT (LoRA/QLoRA)",
        "wandb": "Weights & Biases",
        "gradio": "Gradio (Demo UI)",
        "sklearn": "scikit-learn",
        "imageio": "imageio (Video export)",
    }
    missing = []
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} — НЕ УСТАНОВЛЕН")
            missing.append(module)

    # Проверяем FlashAttention (опционально)
    try:
        import flash_attn  # noqa: F401
        print(f"  🚀 FlashAttention — УСТАНОВЛЕН")
    except ImportError:
        print(f"  ⚠️  FlashAttention — не установлен (опционально для H100)")

    # Проверяем cida_plugin
    try:
        from cida_plugin import CIDAPlugin  # noqa: F401
        print(f"  ✅ CIDA Plugin")
    except ImportError:
        print(f"  ❌ CIDA Plugin — НЕ НАЙДЕН. Убедитесь что cida_plugin/ в корне репозитория.")
        missing.append("cida_plugin")

    return missing


def download_models(cache_dir: str = None):
    """Скачивает веса моделей заранее для H100 сессии."""
    print("\n" + "=" * 60)
    print("⬇️  Скачивание моделей (pre-cache)")
    print("=" * 60)

    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        print(f"  Кэш: {cache_dir}")

    # 1. BiomedCLIP
    print("\n  [1/3] BiomedCLIP (Microsoft)...")
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        del model
        print("  ✅ BiomedCLIP скачан")
    except Exception as e:
        print(f"  ⚠️  BiomedCLIP: {e}")

    # 2. Qolda (Казахский VLM)
    print("\n  [2/3] Qolda (ISSAI, Назарбаев Университет)...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "issai/Qolda", trust_remote_code=True
        )
        del tokenizer
        print("  ✅ Qolda tokenizer скачан")
    except Exception as e:
        print(f"  ⚠️  Qolda: {e} — будет использован fallback Qwen2.5")

    # 3. Stable Diffusion VAE (для Flow Matching)
    print("\n  [3/3] Stable Diffusion VAE (для латентного пространства)...")
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
        )
        del vae
        print("  ✅ SD VAE скачан")
    except Exception as e:
        print(f"  ⚠️  SD VAE: {e}")

    print("\n  ✅ Pre-cache завершён!")


def install_deps():
    """Устанавливает зависимости из requirements_h100.txt."""
    req_path = SCRIPT_DIR / "requirements_h100.txt"
    if req_path.exists():
        print(f"\n  Установка из {req_path}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(req_path), "-q"
        ])
        print("  ✅ Зависимости установлены")
    else:
        print(f"  ❌ {req_path} не найден!")


def main():
    parser = argparse.ArgumentParser(description="KairosMM Setup")
    parser.add_argument("--check_only", action="store_true",
                        help="Только проверка, без установки")
    parser.add_argument("--download_models", action="store_true",
                        help="Только скачивание моделей")
    parser.add_argument("--cache_dir", default=None,
                        help="Директория для кэша моделей HuggingFace")
    parser.add_argument("--install", action="store_true",
                        help="Установить зависимости")
    args = parser.parse_args()

    print("🔬 KairosMM H100 Setup")
    print("=" * 60)

    gpu_type = check_gpu()
    missing = check_dependencies()

    if args.check_only:
        if missing:
            print(f"\n❌ Отсутствуют: {', '.join(missing)}")
            print("   Запустите: python KairosMM/00_setup.py --install")
        else:
            print("\n✅ Все зависимости в порядке!")
        return

    if args.install:
        install_deps()

    if args.download_models:
        download_models(args.cache_dir)

    if not args.install and not args.download_models:
        # Полная установка
        if missing:
            install_deps()
        download_models(args.cache_dir)

    print("\n" + "=" * 60)
    print("✅ KairosMM Setup завершён!")
    print(f"   GPU: {gpu_type}")
    print("   Следующий шаг: python KairosMM/01_encoder_biomedclip.py --mode test")
    print("=" * 60)


if __name__ == "__main__":
    main()
