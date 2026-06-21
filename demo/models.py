"""Модели для интерактивного демо CIDA-Plugin."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from cida_plugin import CIDAPlugin, CIDAPluginConfig, RoleNames


AGENT_LABELS_RU = {
    "Prosecutor": "Прокурор",
    "Defender": "Защитник",
    "Skeptic": "Скептик",
    "Integrator": "Интегратор",
}


@dataclass
class CIDAInferenceResult:
    verdict: str
    predicted_idx: int
    confidence: float
    agent_names: list[str]
    agent_votes: list[float]
    agent_prob_positive: list[float]  # P(class=1) для binary / top-1 conf иначе
    agent_labels: list[str]
    class_names: list[str]
    uncertainty: float
    disagreement: float
    needs_review: bool
    rounds_used: int
    p_final: list[float]


class MNISTEncoder(nn.Module):
    """Лёгкий CNN-энкодер для MNIST с spatial-представлением для evidence pointers."""

    def __init__(self, d_output: int = 128):
        super().__init__()
        self.d_output = d_output
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.spatial_dim = 64
        self.proj = nn.Linear(self.spatial_dim, d_output)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = self.features(x)
        batch, channels, height, width = spatial.shape
        seq = spatial.view(batch, channels, -1).transpose(1, 2)  # (B, 49, 64)
        seq = self.proj(seq)  # (B, 49, d_output)
        pooled = seq.mean(dim=1)
        return pooled, seq


class MNISTCIDAModel(nn.Module):
    """MNIST: замороженный CNN + обучаемый CIDA-Plugin."""

    def __init__(self, num_classes: int = 10, d_hidden: int = 128):
        super().__init__()
        self.encoder = MNISTEncoder(d_output=d_hidden)
        self.plugin = CIDAPlugin(
            CIDAPluginConfig(
                d_input=d_hidden,
                d_hidden=d_hidden,
                num_classes=num_classes,
                max_rounds=3,
                early_stop_threshold=0.92,
            )
        )
        self.class_names = [str(i) for i in range(num_classes)]

    def forward(self, x: torch.Tensor) -> dict:
        pooled, seq = self.encoder(x)
        return self.plugin(pooled, seq_output=seq)

    def predict(self, image: torch.Tensor) -> CIDAInferenceResult:
        self.eval()
        with torch.no_grad():
            out = self.forward(image)
        return _build_result(
            out,
            class_names=self.class_names,
            positive_label="цифра",
            verdict_template="Изображение похоже на цифру {label} ({conf:.0%} уверенности)",
        )


class SentimentCIDAModel(nn.Module):
    """SST-2: замороженный DistilBERT + CIDA-Plugin."""

    def __init__(self, encoder: nn.Module, d_model: int):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.plugin = CIDAPlugin(
            CIDAPluginConfig(
                d_input=d_model,
                d_hidden=128,
                num_classes=2,
                max_rounds=3,
                early_stop_threshold=0.90,
            )
        )
        self.class_names = ["Негативный", "Положительный"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            encoder_out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        pooled = encoder_out.last_hidden_state[:, 0, :]
        seq = encoder_out.last_hidden_state
        return self.plugin(pooled, seq_output=seq, mask=attention_mask)

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CIDAInferenceResult:
        self.eval()
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask)
        return _build_result(
            out,
            class_names=self.class_names,
            positive_label="положительный",
            verdict_template='Отзыв {label} ({conf:.0%} уверенности)',
        )


def _build_result(
    out: dict,
    class_names: list[str],
    positive_label: str,
    verdict_template: str,
) -> CIDAInferenceResult:
    p_final = out["p_final"][0].cpu()
    beliefs = out["b_all"][-1][0].cpu()
    uncertainty = float(out["uncertainty"][0].cpu())
    disagreement = float(out["disagreement"][0].cpu())
    rounds_used = int(out["rounds_used"])

    predicted_idx = int(p_final.argmax())
    confidence = float(p_final[predicted_idx])
    label = class_names[predicted_idx]

    agent_names = [AGENT_LABELS_RU[RoleNames.get(i)] for i in range(beliefs.size(0))]
    agent_labels = [class_names[int(b.argmax())] for b in beliefs]
    agent_votes = [float(b.max()) for b in beliefs]
    if len(class_names) == 2:
        agent_prob_positive = [float(b[1]) for b in beliefs]
    else:
        agent_prob_positive = agent_votes

    needs_review = disagreement > 0.12 or uncertainty > 0.15

    return CIDAInferenceResult(
        verdict=verdict_template.format(label=label, conf=confidence),
        predicted_idx=predicted_idx,
        confidence=confidence,
        agent_names=agent_names,
        agent_votes=agent_votes,
        agent_prob_positive=agent_prob_positive,
        agent_labels=agent_labels,
        class_names=class_names,
        uncertainty=uncertainty,
        disagreement=disagreement,
        needs_review=needs_review,
        rounds_used=rounds_used,
        p_final=[float(v) for v in p_final],
    )


def load_mnist_model(checkpoint_dir: str, device: torch.device) -> MNISTCIDAModel:
    model = MNISTCIDAModel()
    weights_path = os.path.join(checkpoint_dir, "mnist_cida.pt")
    if os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_sentiment_model(
    checkpoint_dir: str,
    device: torch.device,
) -> tuple[SentimentCIDAModel, object]:
    from transformers import AutoModel, AutoTokenizer

    encoder_name = "distilbert-base-uncased"
    encoder = AutoModel.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    model = SentimentCIDAModel(encoder, d_model=encoder.config.hidden_size)
    weights_path = os.path.join(checkpoint_dir, "sentiment_cida.pt")
    if os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.plugin.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, tokenizer
