"""
CIDA-Plugin — интерактивное демо для Hugging Face Spaces.

Показывает три ключевые вещи:
  1. Конечный вердикт (цифра MNIST или тональность отзыва)
  2. Голоса 4 агентов (Прокурор, Защитник, Скептик, Интегратор)
  3. Индикатор неуверенности (красный = агенты спорят → нужен человек)
"""

from __future__ import annotations

import os
import sys

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import load_mnist_model, load_sentiment_model
from visualize import plot_agent_votes, uncertainty_gauge_html

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MNIST_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

EXAMPLE_REVIEWS = [
    "This movie was absolutely fantastic! Best film I've seen all year.",
    "Terrible acting, boring plot, complete waste of time.",
    "It was okay. Some good moments but nothing special.",
    "I loved the cinematography but the story felt rushed and confusing.",
]

_mnist_model = None
_sentiment_model = None
_tokenizer = None


def _ensure_models() -> None:
    global _mnist_model, _sentiment_model, _tokenizer

    mnist_ckpt = os.path.join(CHECKPOINT_DIR, "mnist_cida.pt")
    sentiment_ckpt = os.path.join(CHECKPOINT_DIR, "sentiment_cida.pt")

    if not os.path.isfile(mnist_ckpt) or not os.path.isfile(sentiment_ckpt):
        print("Чекпоинты не найдены — запускаю быстрое обучение...")
        from train_demo import train_mnist, train_sentiment

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        train_mnist(CHECKPOINT_DIR, DEVICE, epochs=2, subset_size=8000, batch_size=128)
        train_sentiment(CHECKPOINT_DIR, DEVICE, epochs=2, subset_size=2000, batch_size=32)

    if _mnist_model is None:
        _mnist_model = load_mnist_model(CHECKPOINT_DIR, DEVICE)
    if _sentiment_model is None:
        _sentiment_model, _tokenizer = load_sentiment_model(CHECKPOINT_DIR, DEVICE)


def _preprocess_mnist(image: Image.Image | np.ndarray) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:
            image = Image.fromarray(image).convert("L")
        else:
            image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        image = Image.open(image).convert("L")
    tensor = MNIST_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    return tensor


def predict_mnist(image) -> tuple[str, object, str]:
    _ensure_models()
    if image is None:
        return "Загрузите или нарисуйте цифру", None, ""

    tensor = _preprocess_mnist(image)
    result = _mnist_model.predict(tensor)
    fig = plot_agent_votes(result)
    gauge = uncertainty_gauge_html(result)
    return result.verdict, fig, gauge


def predict_sentiment(text: str) -> tuple[str, object, str]:
    _ensure_models()
    if not text or not text.strip():
        return "Введите текст отзыва", None, ""

    enc = _tokenizer(
        text.strip(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    result = _sentiment_model.predict(input_ids, attention_mask)
    fig = plot_agent_votes(result)
    gauge = uncertainty_gauge_html(result)
    return result.verdict, fig, gauge


DESCRIPTION = """
# CIDA-Plugin: Multi-Agent Deliberation Demo

**CIDA** — это не одна «чёрная коробка», а **4 агента с разными ролями**, которые обсуждают вход и приходят к консенсусу.

| Агент | Роль | Prior |
|-------|------|-------|
| **Прокурор** | Склонен видеть «положительный» класс | 85% |
| **Защитник** | Склонен видеть «отрицательный» класс | 15% |
| **Скептик** | Максимальная неопределённость | 50% |
| **Интегратор** | Объективный синтезатор (без prior) | — |

**Чем CIDA отличается от обычной модели?** Обычный классификатор выдаёт одно число.
CIDA показывает *распределение мнений* и *несогласие* — когда агенты спорят, система честно говорит «нужна проверка человеком».
"""

DEMO_CSS = """
    .verdict-box { font-size: 1.4rem; font-weight: 700; padding: 16px; border-radius: 8px;
                  background: linear-gradient(135deg, #667eea22, #764ba222); }
"""

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.Tab("🔢 MNIST — Распознавание цифр"):
            gr.Markdown(
                "Загрузите изображение рукописной цифры или нарисуйте её. "
                "CIDA покажет, как 4 агента «спорят» о том, какая это цифра."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    mnist_input = gr.Image(
                        label="Изображение цифры",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=280,
                    )
                    mnist_btn = gr.Button("Анализировать", variant="primary")
                with gr.Column(scale=2):
                    mnist_verdict = gr.Textbox(
                        label="Конечный вердикт",
                        elem_classes=["verdict-box"],
                        interactive=False,
                    )
                    mnist_chart = gr.Plot(label="Голоса агентов")
                    mnist_gauge = gr.HTML(label="Неуверенность")

            mnist_btn.click(
                predict_mnist,
                inputs=[mnist_input],
                outputs=[mnist_verdict, mnist_chart, mnist_gauge],
            )
            mnist_input.change(
                predict_mnist,
                inputs=[mnist_input],
                outputs=[mnist_verdict, mnist_chart, mnist_gauge],
            )

        with gr.Tab("🎬 Sentiment — Тональность отзыва"):
            gr.Markdown(
                "Введите отзыв о фильме на английском (SST-2). "
                "Попробуйте неоднозначные отзывы — агенты будут спорить!"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Текст отзыва",
                        placeholder="This movie was...",
                        lines=4,
                    )
                    gr.Examples(
                        examples=[[e] for e in EXAMPLE_REVIEWS],
                        inputs=[text_input],
                        label="Примеры",
                    )
                    sentiment_btn = gr.Button("Анализировать", variant="primary")
                with gr.Column(scale=2):
                    sentiment_verdict = gr.Textbox(
                        label="Конечный вердикт",
                        elem_classes=["verdict-box"],
                        interactive=False,
                    )
                    sentiment_chart = gr.Plot(label="Голоса агентов")
                    sentiment_gauge = gr.HTML(label="Неуверенность")

            sentiment_btn.click(
                predict_sentiment,
                inputs=[text_input],
                outputs=[sentiment_verdict, sentiment_chart, sentiment_gauge],
            )

    gr.Markdown(
        """
        ---
        **Как читать результаты:**
        - **Конечный вердикт** — взвешенный консенсус всех агентов (Weighted Mean).
        - **Диаграмма** — каждый агент голосует независимо; разброс = «магия» CIDA.
        - **Красный индикатор** — высокое несогласие (`disagreement = std(beliefs)`) → случай требует проверки.

        [GitHub](https://github.com/Kairatzh/CIDA-plugin) · CIDA-Plugin v3
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=DEMO_CSS,
        show_error=True,
    )
