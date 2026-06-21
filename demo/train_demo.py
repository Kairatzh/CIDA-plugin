"""
Быстрое обучение демо-моделей (MNIST + SST-2 sentiment).

Запуск из корня репозитория:
    python demo/train_demo.py
    python demo/train_demo.py --quick   # ещё быстрее (для CI)
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cida_plugin import CIDALoss
from models import MNISTCIDAModel, SentimentCIDAModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_mnist(
    checkpoint_dir: str,
    device: torch.device,
    epochs: int = 3,
    subset_size: int = 12000,
    batch_size: int = 128,
) -> None:
    print("=== Обучение MNIST + CIDA ===")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(
        root=os.path.join(checkpoint_dir, "data"),
        train=True,
        download=True,
        transform=transform,
    )
    indices = torch.randperm(len(train_set))[:subset_size].tolist()
    loader = DataLoader(
        Subset(train_set, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = MNISTCIDAModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = CIDALoss(lambda_cal=0.4, lambda_ac=0.2)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss, _ = loss_fn(out["p_final"], labels, out["b_all"], orth_loss=out["L_orth"])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out["p_final"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / max(total, 1)
        print(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}  acc={acc:.3f}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "mnist_cida.pt")
    torch.save(model.state_dict(), path)
    print(f"  Сохранено: {path}")


def train_sentiment(
    checkpoint_dir: str,
    device: torch.device,
    epochs: int = 2,
    subset_size: int = 3000,
    batch_size: int = 32,
) -> None:
    print("=== Обучение Sentiment (SST-2) + CIDA ===")
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    ds = load_dataset("glue", "sst2", split="train")
    indices = list(range(min(subset_size, len(ds))))
    subset = ds.select(indices)

    encoder_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name)

    model = SentimentCIDAModel(encoder, d_model=encoder.config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.plugin.parameters(), lr=2e-3)
    loss_fn = CIDALoss(lambda_cal=0.4, lambda_ac=0.2)

    def collate(batch):
        texts = [item["sentence"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], labels

    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask)
            loss, _ = loss_fn(
                out["p_final"],
                labels,
                out["b_all"],
                orth_loss=out["L_orth"],
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out["p_final"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / max(total, 1)
        print(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}  acc={acc:.3f}")

    path = os.path.join(checkpoint_dir, "sentiment_cida.pt")
    torch.save(model.plugin.state_dict(), path)
    print(f"  Сохранено: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Быстрый режим для smoke-test")
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.join(os.path.dirname(__file__), "checkpoints"),
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.quick:
        train_mnist(args.checkpoint_dir, device, epochs=1, subset_size=2000, batch_size=64)
        train_sentiment(args.checkpoint_dir, device, epochs=1, subset_size=500, batch_size=16)
    else:
        train_mnist(args.checkpoint_dir, device)
        train_sentiment(args.checkpoint_dir, device)

    print("Готово!")


if __name__ == "__main__":
    main()
