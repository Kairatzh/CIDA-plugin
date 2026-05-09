"""
Эксперимент: Confidence-based Early Stopping в CIDA-Plugin.

Исследует вычислительную эффективность: при каком пороге уверенности
агенты могут досрочно завершить deliberation без потери качества?

Метрики:
    - % примеров, остановившихся на раунде 1, 2, ... max_rounds
    - ECE и Accuracy при каждом пороге
    - Среднее число использованных раундов

Запуск:
    python experiments/exp_plugin_early_stop.py
    python experiments/exp_plugin_early_stop.py --dry-run
"""
import sys
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cida_plugin import CIDAPlugin, CIDAPluginConfig, OmegaLossSystem, DebateDiagnostics


# ─── Конфигурация ────────────────────────────────────────────────────────────

ENCODER_HF = "distilbert-base-uncased"  # Подтверждённый рабочий энкодер
D_MODEL = 768
NUM_CLASSES = 2
BATCH_SIZE = 32
TRAIN_EPOCHS = 3
MAX_ROUNDS = 4              # Больше раундов — интереснее наблюдать остановку
MAX_SEQ_LEN = 128
DATASET_SIZE = 2000
SEED = 42

# Пороги early stopping для сравнения
THRESHOLDS = [None, 0.80, 0.85, 0.90, 0.95]
# None = без ранней остановки (baseline)


# ─── Утилиты (аналогично exp_plugin_encoders.py) ─────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sst2_subset(tokenizer, size: int, split: str = "train"):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split=split)
    indices = list(range(min(size, len(ds))))
    random.shuffle(indices)
    indices = indices[:size]

    texts = [ds[i]["sentence"] for i in indices]
    labels = [ds[i]["label"] for i in indices]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )

    class SST2Subset(Dataset):
        def __len__(self): return len(labels)
        def __getitem__(self, idx):
            return {
                "input_ids": encodings["input_ids"][idx],
                "attention_mask": encodings["attention_mask"][idx],
                "label": torch.tensor(labels[idx], dtype=torch.long),
            }

    return SST2Subset()


class EncoderWithPlugin(nn.Module):
    def __init__(self, encoder, plugin: CIDAPlugin):
        super().__init__()
        self.encoder = encoder
        self.plugin = plugin
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        seq_out = out.last_hidden_state
        return self.plugin(pooled, seq_output=seq_out, mask=attention_mask)


def train_plugin(model, loader, optimizer, loss_system, device, epochs):
    """Обучает плагин и возвращает историю потерь."""
    model.train()
    losses = []
    for ep in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, mask)

            # [v2] d_schedule=None → learnable internal schedule
            loss, _ = loss_system(
                p_final=out["p_final"],
                y=labels,
                b_all=out["b_all"],
                u_all=out["u_all"],
                h_all=out["h_all"],
                l_dom_val=out["L_dom"],
                l_role_val=out["L_role"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        losses.append(epoch_loss)
        print(f"      Epoch {ep+1}/{epochs} | Loss: {epoch_loss:.4f}")
    return losses


@torch.no_grad()
def evaluate_with_stopping(model, loader, device, threshold):
    """
    Оценка модели с заданным порогом early stopping.
    Дополнительно трекает, сколько раундов было использовано.
    """
    model.eval()
    # Устанавливаем порог
    model.plugin.config.early_stop_threshold = threshold

    all_probs, all_labels = [], []
    rounds_distribution = defaultdict(int)  # rounds_used → count

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        out = model(input_ids, mask)
        probs = out["p_final"]
        rounds = out["rounds_used"]

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        rounds_distribution[rounds] += input_ids.size(0)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    preds = all_probs.argmax(dim=-1)
    accuracy = (preds == all_labels).float().mean().item()
    ece = DebateDiagnostics.expected_calibration_error(all_probs, all_labels)

    total = sum(rounds_distribution.values())
    rounds_pct = {r: cnt / total * 100 for r, cnt in sorted(rounds_distribution.items())}
    avg_rounds = sum(r * cnt for r, cnt in rounds_distribution.items()) / total

    return {
        "accuracy": accuracy,
        "ece": ece,
        "rounds_distribution_pct": rounds_pct,
        "avg_rounds": avg_rounds,
    }


# ─── Основной запуск ─────────────────────────────────────────────────────────

def run_experiment(dry_run: bool = False):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Эксперимент: Confidence-based Early Stopping")
    print(f"  Устройство: {device} | max_rounds: {MAX_ROUNDS}")
    print(f"  Пороги: {THRESHOLDS}")
    print(f"  Режим: {'DRY RUN' if dry_run else 'FULL'}")
    print(f"{'='*60}\n")

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("ОШИБКА: pip install transformers datasets")
        return

    epochs = 1 if dry_run else TRAIN_EPOCHS
    dataset_size = 64 if dry_run else DATASET_SIZE

    print(f"Загрузка энкодера '{ENCODER_HF}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ENCODER_HF)
        encoder = AutoModel.from_pretrained(ENCODER_HF).to(device)
    except Exception as e:
        print(f"ОШИБКА загрузки энкодера: {e}")
        return

    print("Загрузка SST-2...")
    try:
        train_ds = load_sst2_subset(tokenizer, size=dataset_size, split="train")
        val_ds = load_sst2_subset(tokenizer, size=dataset_size // 4, split="validation")
    except Exception as e:
        print(f"ОШИБКА загрузки данных: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # ─── Одно обучение, несколько порогов при инференсе ──────────────────────
    # Обучаем с early_stop_threshold=None (полные раунды в train всегда)
    print("\nОбучение CIDA-Plugin (without early stopping during training)...")
    plugin_cfg = CIDAPluginConfig(
        d_input=D_MODEL,
        d_hidden=128,
        d_message=128,
        num_agents=4,
        num_classes=NUM_CLASSES,
        max_rounds=MAX_ROUNDS,
        early_stop_threshold=None,   # During training — no early stop
        num_attn_heads=4,            # [v2] Transformer heads
        rho_decay=0.9,               # [v2] EMA decay
        learnable_d_schedule=True,   # [v2] learnable schedule
    )
    plugin = CIDAPlugin(plugin_cfg).to(device)
    model = EncoderWithPlugin(encoder, plugin).to(device)
    loss_system = OmegaLossSystem(
        lambda_task=1.0, lambda_cal=0.5, lambda_deb=0.5,
        lambda_prog=0.3, lambda_budget=0.1, lambda_role=0.1, lambda_dom=0.1,
        max_rounds=plugin_cfg.max_rounds,
        learnable_d_schedule=plugin_cfg.learnable_d_schedule,
    )
    optimizer = torch.optim.AdamW(plugin.parameters(), lr=1e-3, weight_decay=0.01)
    train_plugin(model, train_loader, optimizer, loss_system, device, epochs)

    # ─── Инференс с разными порогами ─────────────────────────────────────────
    print("\nОценка при разных порогах early stopping...\n")
    results = {}

    for threshold in THRESHOLDS:
        label = f"threshold={threshold}" if threshold is not None else "no_early_stop"
        print(f"  Порог: {label}")
        metrics = evaluate_with_stopping(model, val_loader, device, threshold)
        results[label] = metrics

        rdist = metrics["rounds_distribution_pct"]
        rdist_str = " | ".join([f"r{r}: {pct:.1f}%" for r, pct in sorted(rdist.items())])

        print(f"    ACC: {metrics['accuracy']:.4f} | ECE: {metrics['ece']:.4f} "
              f"| Avg rounds: {metrics['avg_rounds']:.2f}")
        print(f"    Раунды: [{rdist_str}]")

    # ─── Итоговая таблица ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ИТОГОВАЯ ТАБЛИЦА (Early Stopping Analysis)")
    print(f"{'='*60}")
    print(f"{'Threshold':<22} {'ACC':>8} {'ECE':>8} {'AvgRounds':>10} {'Compute%':>10}")
    print("-" * 62)

    baseline_rounds = results.get("no_early_stop", {}).get("avg_rounds", MAX_ROUNDS)

    for label, m in results.items():
        compute_pct = m["avg_rounds"] / MAX_ROUNDS * 100
        print(
            f"{label:<22}"
            f"{m['accuracy']:>8.4f}"
            f"{m['ece']:>8.4f}"
            f"{m['avg_rounds']:>10.2f}"
            f"{compute_pct:>9.1f}%"
        )

    # Сохранение
    out_path = os.path.join(ROOT, "results", "exp_plugin_early_stop.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # rounds_distribution_pct ключи — int, нужна сериализация
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            **v,
            "rounds_distribution_pct": {str(r): pct for r, pct in v["rounds_distribution_pct"].items()}
        }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Результаты сохранены → {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIDA-Plugin: Early Stopping Experiment")
    parser.add_argument("--dry-run", action="store_true", help="Быстрый smoke-test")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run)
