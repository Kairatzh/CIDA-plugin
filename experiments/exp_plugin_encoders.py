"""
Эксперимент: Один CIDA-Plugin — три энкодера.

Проверяет, что CIDA-Plugin улучшает калибровку (ECE) независимо от
базового энкодера. Энкодер замораживается, обучается только плагин.

Архитектура теста:
    [Frozen Encoder] → pooled_output → [CIDAPlugin] → logits → CE loss
                                     vs.
    [Frozen Encoder] → pooled_output → [Linear Head] → logits → CE loss

Результат: таблица ECE / Accuracy для каждого энкодера.

Запуск:
    python experiments/exp_plugin_encoders.py
    python experiments/exp_plugin_encoders.py --dry-run  # быстрый smoke-test
"""
import sys
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import random

# ─── Путь к корню проекта ────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cida_plugin import CIDAPlugin, CIDAPluginConfig, OmegaLossSystem, DebateDiagnostics


# ─── Конфигурация эксперимента ────────────────────────────────────────────────

ENCODERS_TO_TEST = [
    {
        "name": "bert-tiny (local)",
        "hf_name": "prajjwal1/bert-tiny",   # 128-dim, ~4.4M params
        "d_model": 128,
    },
    {
        "name": "distilbert-base-uncased",
        "hf_name": "distilbert-base-uncased",  # 768-dim, ~66M params
        "d_model": 768,
    },
    {
        "name": "bert-base-uncased",
        "hf_name": "bert-base-uncased",        # 768-dim, ~110M params
        "d_model": 768,
    },
]

NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 3
MAX_SEQ_LEN = 128
DATASET_SIZE = 2000       # Подмножество SST-2 для скорости
SEED = 42


# ─── Модели ──────────────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    """Стандартный линейный классификатор — baseline."""
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc(pooled)


class EncoderWithPlugin(nn.Module):
    """Замороженный энкодер + CIDAPlugin поверх него."""
    def __init__(self, encoder, plugin: CIDAPlugin):
        super().__init__()
        self.encoder = encoder
        self.plugin = plugin
        # Замораживаем энкодер
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]   # CLS token
        seq_out = out.last_hidden_state            # (B, seq_len, d_model)
        return self.plugin(pooled, seq_output=seq_out, mask=attention_mask)


class EncoderWithLinear(nn.Module):
    """Замороженный энкодер + линейная голова."""
    def __init__(self, encoder, head: LinearHead):
        super().__init__()
        self.encoder = encoder
        self.head = head
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.head(pooled)


# ─── Утилиты ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sst2_subset(tokenizer, size: int, split: str = "train"):
    """Загружает подмножество SST-2 через HuggingFace datasets."""
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


@torch.no_grad()
def evaluate(model, loader, device, is_plugin: bool):
    """
    Вычисляет Accuracy и ECE для модели.
    is_plugin=True — модель возвращает dict с p_final.
    """
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        if is_plugin:
            out = model(input_ids, mask)
            probs = out["p_final"]
        else:
            logits = model(input_ids, mask)
            probs = F.softmax(logits, dim=-1)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    preds = all_probs.argmax(dim=-1)
    accuracy = (preds == all_labels).float().mean().item()
    ece = DebateDiagnostics.expected_calibration_error(all_probs, all_labels)

    return {"accuracy": accuracy, "ece": ece}


def train_one_epoch(model, loader, optimizer, loss_fn_or_system, device, is_plugin: bool, epoch_frac: float):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if is_plugin:
            out = model(input_ids, mask)
            # [v2] d_schedule=None → OmegaLossSystem использует обучаемое расписание
            loss, _ = loss_fn_or_system(
                p_final=out["p_final"],
                y=labels,
                b_all=out["b_all"],
                u_all=out["u_all"],
                h_all=out["h_all"],
                l_dom_val=out["L_dom"],
                l_role_val=out["L_role"],
            )
        else:
            logits = model(input_ids, mask)
            loss = F.cross_entropy(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─── Основной запуск ─────────────────────────────────────────────────────────

def run_experiment(dry_run: bool = False):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Эксперимент: CIDA-Plugin vs Linear Head (3 энкодера)")
    print(f"  Устройство: {device}")
    print(f"  Режим: {'DRY RUN' if dry_run else 'FULL'}")
    print(f"{'='*60}\n")

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("ОШИБКА: установи transformers: pip install transformers datasets")
        return

    results = {}

    epochs = 1 if dry_run else EPOCHS
    dataset_size = 64 if dry_run else DATASET_SIZE

    for enc_cfg in ENCODERS_TO_TEST:
        enc_name = enc_cfg["name"]
        hf_name = enc_cfg["hf_name"]
        d_model = enc_cfg["d_model"]

        print(f"\n--- Энкодер: {enc_name} ---")
        print(f"    Загрузка модели '{hf_name}'...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            encoder = AutoModel.from_pretrained(hf_name).to(device)
        except Exception as e:
            print(f"    ПРОПУСК: не удалось загрузить '{hf_name}': {e}")
            continue

        print("    Загрузка SST-2...")
        try:
            train_ds = load_sst2_subset(tokenizer, size=dataset_size, split="train")
            val_ds = load_sst2_subset(tokenizer, size=dataset_size // 4, split="validation")
        except Exception as e:
            print(f"    ПРОПУСК: ошибка загрузки данных: {e}")
            continue

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        enc_results = {}

        # ── BASELINE: Linear Head ────────────────────────────────────────────
        print("    [1/2] Обучение Linear baseline...")
        head = LinearHead(d_model, NUM_CLASSES).to(device)
        baseline_model = EncoderWithLinear(encoder, head).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)

        for ep in range(epochs):
            loss = train_one_epoch(baseline_model, train_loader, optimizer, None, device, False, ep / epochs)
            if not dry_run:
                print(f"      Epoch {ep+1}/{epochs} | Loss: {loss:.4f}")

        metrics = evaluate(baseline_model, val_loader, device, is_plugin=False)
        enc_results["linear"] = metrics
        print(f"    Linear  → ACC: {metrics['accuracy']:.4f} | ECE: {metrics['ece']:.4f}")

        # ── CIDA PLUGIN v2 ───────────────────────────────────────────────────
        print("    [2/2] Обучение CIDA-Plugin v2 (Transformer Updater + EMA rho + Learnable Schedule)...")
        d_hidden = min(128, d_model)
        # num_attn_heads должен делить d_hidden
        num_heads = 4 if d_hidden % 4 == 0 else (2 if d_hidden % 2 == 0 else 1)
        plugin_cfg = CIDAPluginConfig(
            d_input=d_model,
            d_hidden=d_hidden,
            d_message=d_hidden,
            num_agents=4,
            num_classes=NUM_CLASSES,
            max_rounds=2,
            early_stop_threshold=0.90,
            num_attn_heads=num_heads,
            rho_decay=0.9,
            learnable_d_schedule=True,
        )
        plugin = CIDAPlugin(plugin_cfg).to(device)
        plugin_model = EncoderWithPlugin(encoder, plugin).to(device)
        # [v2] max_rounds и learnable_d_schedule согласованы с плагином
        loss_system = OmegaLossSystem(
            lambda_task=1.0, lambda_cal=0.5, lambda_deb=0.5,
            lambda_prog=0.3, lambda_budget=0.1, lambda_role=0.1, lambda_dom=0.1,
            max_rounds=plugin_cfg.max_rounds,
            learnable_d_schedule=plugin_cfg.learnable_d_schedule,
        )
        optimizer = torch.optim.AdamW(plugin.parameters(), lr=1e-3, weight_decay=0.01)

        for ep in range(epochs):
            loss = train_one_epoch(plugin_model, train_loader, optimizer, loss_system, device, True, ep / epochs)
            if not dry_run:
                print(f"      Epoch {ep+1}/{epochs} | Loss: {loss:.4f}")

        metrics = evaluate(plugin_model, val_loader, device, is_plugin=True)
        enc_results["cida_plugin"] = metrics
        print(f"    Plugin  → ACC: {metrics['accuracy']:.4f} | ECE: {metrics['ece']:.4f}")

        # Дельта
        delta_acc = metrics["accuracy"] - enc_results["linear"]["accuracy"]
        delta_ece = metrics["ece"] - enc_results["linear"]["ece"]
        print(f"    DELTA   → ΔACC: {delta_acc:+.4f} | ΔECE: {delta_ece:+.4f}")

        # [v2] Логируем EMA надёжности агентов
        rho = plugin.rho_ema.squeeze().tolist()
        rho_str = " | ".join([f"a{i}: {v:.3f}" for i, v in enumerate(rho)])
        print(f"    EMA rho → [{rho_str}]")
        # Логируем обученное расписание разногласий
        sched = loss_system.d_targets.detach().cpu().tolist()
        sched_str = " → ".join([f"{v:.3f}" for v in sched])
        print(f"    D-Sched → [{sched_str}]")

        results[enc_name] = enc_results

    # ─── Итоговая таблица ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ИТОГОВАЯ ТАБЛИЦА")
    print(f"{'='*60}")
    print(f"{'Encoder':<30} {'Lin.ACC':>8} {'Plg.ACC':>8} {'Lin.ECE':>8} {'Plg.ECE':>8} {'ΔECE':>8}")
    print("-" * 76)
    for enc_name, res in results.items():
        lin = res.get("linear", {})
        plg = res.get("cida_plugin", {})
        delta_ece = plg.get("ece", 0) - lin.get("ece", 0)
        print(
            f"{enc_name:<30}"
            f"{lin.get('accuracy', 0):>8.4f}"
            f"{plg.get('accuracy', 0):>8.4f}"
            f"{lin.get('ece', 0):>8.4f}"
            f"{plg.get('ece', 0):>8.4f}"
            f"{delta_ece:>+8.4f}"
        )

    # Сохранение
    out_path = os.path.join(ROOT, "results", "exp_plugin_encoders.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Результаты сохранены → {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIDA-Plugin: Multi-Encoder Experiment")
    parser.add_argument("--dry-run", action="store_true", help="Быстрый smoke-test (1 epoch, 64 samples)")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run)
