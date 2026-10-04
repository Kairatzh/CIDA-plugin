"""
train_sentinel_mt.py — Industrial Multi-Turn Training Pipeline for Kairos-Sentinel-MT

Adheres strictly to the Research Honesty Contract:
- 100% Real Text Embeddings via Hugging Face AutoModel (ZERO Target Leakage).
- No synthetic feature shifts (+1.8 / -1.8) or artificial noise injection.
- Train / Val split strictly partitioned by task_id to prevent data contamination.
- Validation metrics computed strictly on held-out tasks.

Usage:
    python train_sentinel_mt.py --data_dir ./MT-AgentRisk --epochs 10 --batch_size 32
    python train_sentinel_mt.py --smoke_test
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import yaml
except ImportError:
    yaml = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from cida_plugin.sentinel_mt import (
    KairosSentinelMT,
    MT_AGENT_RISK_TAXONOMY,
    MT_MCP_TOOLS,
    INTERVENTION_POLICIES,
)
from cida_plugin.losses import CIDALoss
from cida_plugin.encoder import TextEmbeddingEngine


# ===========================================================================
# 1. Real MT-AgentRisk Task Loader (Dataset Parser)
# ===========================================================================

def load_real_mt_agentrisk_tasks(data_dir: str = "./MT-AgentRisk") -> List[Dict[str, Any]]:
    """
    Parses real multi-turn tasks from cloned CHATS-lab/MT-AgentRisk repository.
    Recursively scans for turns.yml, task-turn-*.md, and task.md.
    """
    candidates = [
        data_dir,
        os.path.join(data_dir, "workspaces"),
        os.path.join(os.getcwd(), "workspaces"),
        os.path.join(os.getcwd(), "MT-AgentRisk", "workspaces"),
        "/content/workspaces",
        "/content/MT-AgentRisk/workspaces",
    ]
    search_root = None
    for c in candidates:
        if os.path.exists(c) and (glob.glob(os.path.join(c, "*", "*")) or glob.glob(os.path.join(c, "*"))):
            search_root = c
            break

    if not search_root:
        return []

    real_tasks = []
    for root, dirs, files in os.walk(search_root):
        turn_files = sorted([f for f in files if f.startswith("task-turn-") and f.endswith(".md")])
        single_task = "task.md" in files
        turns_yaml_file = "turns.yml" if "turns.yml" in files else ("turns.yaml" if "turns.yaml" in files else None)

        if not turn_files and not single_task and not turns_yaml_file:
            continue

        task_id = os.path.basename(root)

        # Detect tool category
        tool_name = "Filesystem-MCP"
        for t in MT_MCP_TOOLS:
            prefix = t.lower().split("-")[0]
            if prefix in root.lower() or prefix in task_id.lower():
                tool_name = t
                break

        turns_text = []

        # 1. Parse turns.yml
        if turns_yaml_file and yaml is not None:
            try:
                with open(os.path.join(root, turns_yaml_file), "r", encoding="utf-8", errors="ignore") as f:
                    ydata = yaml.safe_load(f)
                    if isinstance(ydata, list):
                        for item in ydata:
                            if isinstance(item, dict):
                                ins = item.get("instruction") or item.get("prompt") or item.get("content") or item.get("text")
                                if ins:
                                    turns_text.append(str(ins).strip())
                            elif isinstance(item, str):
                                turns_text.append(item.strip())
                    elif isinstance(ydata, dict):
                        turns_list = ydata.get("turns") or ydata.get("instructions")
                        if isinstance(turns_list, list):
                            for item in turns_list:
                                if isinstance(item, dict):
                                    ins = item.get("instruction") or item.get("prompt") or item.get("content")
                                    if ins:
                                        turns_text.append(str(ins).strip())
                                elif isinstance(item, str):
                                    turns_text.append(item.strip())
            except Exception:
                pass

        # 2. Parse task-turn-*.md
        if not turns_text and turn_files:
            for tf in turn_files:
                try:
                    with open(os.path.join(root, tf), "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if content:
                            turns_text.append(content)
                except Exception:
                    pass

        # 3. Parse task.md
        if not turns_text and single_task:
            try:
                with open(os.path.join(root, "task.md"), "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if content:
                        turns_text.append(content)
            except Exception:
                pass

        if not turns_text:
            continue

        # Determine if benign or attack
        is_attack = True
        if "benign" in task_id.lower() or "benign" in root.lower():
            is_attack = False

        # Accurate taxonomy mapping from folder path
        clean_root = root.replace("\\", "/").lower()
        taxonomy = "Decomposition-MethodA-DataFiles" if is_attack else "Benign-Developer-Task"
        if is_attack:
            for tax in MT_AGENT_RISK_TAXONOMY:
                tax_norm = tax.lower().replace("-", "")
                root_norm = clean_root.replace("-", "").replace("_", "")
                if tax_norm in root_norm:
                    taxonomy = tax
                    break
            else:
                method = "MethodA" if "methoda" in clean_root else ("MethodB" if "methodb" in clean_root else "MethodA")
                target = "EnvironmentStates" if ("environment" in clean_root or "state" in clean_root) else "DataFiles"
                family = "Addition" if "addition" in clean_root else ("Decomposition" if "decomposition" in clean_root else "Decomposition")
                taxonomy = f"{family}-{method}-{target}"

        harm_type = "Multi-Turn Harmful Task" if is_attack else "None (Benign Developer Interaction)"

        chain_path = os.path.join(root, "attack_chain.json")
        if os.path.exists(chain_path):
            try:
                with open(chain_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    taxonomy = meta.get("taxonomy", meta.get("attack_category", taxonomy))
                    harm_type = meta.get("harm_type", meta.get("goal", harm_type))
            except Exception:
                pass

        real_tasks.append({
            "id": task_id,
            "tool": tool_name,
            "taxonomy": taxonomy,
            "harm_type": harm_type,
            "turns": turns_text,
            "is_attack": is_attack,
            "critical_turn": len(turns_text),
        })

    return real_tasks


# ===========================================================================
# 2. Multi-Turn Trajectory Dataset Class
# ===========================================================================

class MultiTurnTrajectoryDataset(Dataset):
    """
    Genuine dataset of encoded multi-turn tool interaction trajectories.
    Inputs are encoded real texts.
    Labels are strictly targets for the loss function.
    """
    def __init__(self, samples: List[Dict[str, Any]], d_input: int, max_turns: int = 8):
        self.samples = samples
        self.d_input = d_input
        self.max_turns = max_turns

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        turns = s["turn_embeddings"]  # (T_actual, D)
        t_actual = turns.size(0)

        padded_turns = torch.zeros(self.max_turns, self.d_input)
        turn_mask = torch.zeros(self.max_turns, dtype=torch.bool)

        n_copy = min(t_actual, self.max_turns)
        padded_turns[:n_copy] = turns[:n_copy]
        turn_mask[:n_copy] = True

        return {
            "turn_embeddings": padded_turns,
            "turn_mask": turn_mask,
            "risk_label": torch.tensor(s["risk_label"], dtype=torch.long),
            "taxonomy_label": torch.tensor(s.get("taxonomy_label", 0), dtype=torch.long),
            "mcp_label": torch.tensor(s.get("mcp_label", 0), dtype=torch.long),
            "task_id": s["id"],
        }


# ===========================================================================
# 3. Real Dataset Preparation (Zero Target Leakage)
# ===========================================================================

def prepare_real_trajectory_samples(
    tasks: List[Dict[str, Any]],
    encoder: TextEmbeddingEngine,
    desc: str = "Processing",
) -> List[Dict[str, Any]]:
    """
    Encodes real task turn texts using TextEmbeddingEngine with disk caching.
    Zero target leakage: encoder only receives turn strings.
    """
    samples = []
    total = len(tasks)
    start_time = time.time()

    tax_map = {t: idx for idx, t in enumerate(MT_AGENT_RISK_TAXONOMY)}
    mcp_map = {m: idx for idx, m in enumerate(MT_MCP_TOOLS)}

    for i, t in enumerate(tasks):
        turns_text = t["turns"]
        # Strictly encode text without any label access
        embs = encoder.encode_trajectory(turns_text, task_id=t["id"])

        tax_idx = tax_map.get(t["taxonomy"], 0)
        mcp_idx = mcp_map.get(t["tool"], 0)
        risk_label = 1 if t["is_attack"] else 0

        samples.append({
            "id": t["id"],
            "turn_embeddings": embs,
            "risk_label": risk_label,
            "taxonomy_label": tax_idx,
            "mcp_label": mcp_idx,
        })

        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / max(elapsed, 0.1)
            print(f"[{desc}] Encoded {i + 1}/{total} tasks ({rate:.1f} tasks/sec)...")

    return samples


# ===========================================================================
# 4. Smoke Test Synthetic Generator (Explicitly Marked)
# ===========================================================================

def generate_synthetic_smoke_dataset(
    n_samples: int = 40,
    d_input: int = 768,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Synthetic generator strictly for shape verification during smoke tests.
    Contract 2.4: Results MUST NOT be reported as empirical benchmark measurements.
    """
    print("\n" + "!" * 80)
    print("  [SMOKE TEST] Results are synthetic and MUST NOT be reported.")
    print("  Used exclusively to verify tensor shapes, loss gradients, and checkpointing.")
    print("!" * 80 + "\n")

    samples = []
    for i in range(n_samples):
        t_len = random.randint(2, 5)
        embs = torch.randn(t_len, d_input)
        is_attack = (i % 2 == 1)
        samples.append({
            "id": f"smoke_{i:04d}",
            "turn_embeddings": embs,
            "risk_label": 1 if is_attack else 0,
            "taxonomy_label": i % 8,
            "mcp_label": i % 5,
        })

    return samples[:30], samples[30:]


# ===========================================================================
# 5. Training Loop with Held-out Evaluation
# ===========================================================================

def train_sentinel_mt(
    data_dir: str = "./MT-AgentRisk",
    encoder_model: str = "bert-base-uncased",
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 2e-4,
    val_ratio: float = 0.2,
    seed: int = 42,
    smoke_test: bool = False,
    out_dir: str = "./kairos_checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n" + "=" * 80)
    print("  KAIROS-SENTINEL-MT INDUSTRIAL MULTI-TURN TRAINING")
    print(f"  Target: MT-AgentRisk (ICML 2026) | Device: {device} | Epochs: {epochs}")
    print(f"  Encoder: {encoder_model} | Batch Size: {batch_size} | Seed: {seed}")
    print("=" * 80 + "\n")

    if smoke_test:
        d_in = 768
        train_samples, val_samples = generate_synthetic_smoke_dataset(n_samples=40, d_input=d_in)
    else:
        # Load real tasks
        real_tasks = load_real_mt_agentrisk_tasks(data_dir=data_dir)
        if not real_tasks:
            raise FileNotFoundError(
                f"Real MT-AgentRisk dataset not found at '{data_dir}'! "
                "Per the Research Honesty Contract, training requires a real dataset. "
                "Clone CHATS-lab/MT-AgentRisk or pass --smoke_test for shape testing only."
            )

        print(f"[Dataset] Found {len(real_tasks)} real tasks in '{data_dir}'.")
        n_attacks = sum(1 for t in real_tasks if t["is_attack"])
        n_benign = len(real_tasks) - n_attacks
        print(f"[Dataset] Distribution: {n_attacks} attacks, {n_benign} benign interactions.")

        # Contract 4.1: Train / Val split strictly by task_id
        unique_ids = sorted(list(set(t["id"] for t in real_tasks)))
        random.shuffle(unique_ids)
        split_point = int(len(unique_ids) * (1.0 - val_ratio))
        train_id_set = set(unique_ids[:split_point])
        val_id_set = set(unique_ids[split_point:])

        train_tasks = [t for t in real_tasks if t["id"] in train_id_set]
        val_tasks = [t for t in real_tasks if t["id"] in val_id_set]
        print(f"[Dataset] Partitioned into {len(train_tasks)} train tasks and {len(val_tasks)} validation tasks.")

        # Initialize text encoder
        encoder = TextEmbeddingEngine(model_name=encoder_model, device=device)
        d_in = encoder.d_emb

        print("\n[*] Encoding training set trajectories...")
        train_samples = prepare_real_trajectory_samples(train_tasks, encoder, desc="Train")
        print("[*] Encoding validation set trajectories...")
        val_samples = prepare_real_trajectory_samples(val_tasks, encoder, desc="Val")

    ds_train = MultiTurnTrajectoryDataset(train_samples, d_input=d_in)
    ds_val = MultiTurnTrajectoryDataset(val_samples, d_input=d_in)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = KairosSentinelMT(
        d_input=d_in,
        d_hidden=512,
        d_message=512,
        d_proj=256,
        num_agents=4,
        max_delib_rounds=3,
        num_taxonomy_classes=len(MT_AGENT_RISK_TAXONOMY),
        num_mcp_tools=len(MT_MCP_TOOLS),
        num_policy_actions=len(INTERVENTION_POLICIES),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[SentinelMT] Trainable Parameters: {n_params:,} (d_in={d_in})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_risk_acc = -1.0
    best_ckpt_path = os.path.join(out_dir, "kairos_sentinel_mt.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in loader_train:
            embs = batch["turn_embeddings"].to(device)
            mask = batch["turn_mask"].to(device)
            risk_y = batch["risk_label"].to(device)
            tax_y = batch["taxonomy_label"].to(device)
            mcp_y = batch["mcp_label"].to(device)

            optimizer.zero_grad()
            out = model(embs, turn_mask=mask, task=None)

            # Risk loss (CrossEntropy)
            loss_risk = F.cross_entropy(out["risk"]["calibrated_logits"], risk_y)
            # Taxonomy loss
            loss_tax = F.cross_entropy(out["taxonomy"]["calibrated_logits"], tax_y)
            # MCP loss
            loss_mcp = F.cross_entropy(out["mcp"]["calibrated_logits"], mcp_y)

            # O-LoRA orthogonality loss from model
            loss_ortho = out["orth_loss"]

            total_loss = loss_risk + 0.5 * loss_tax + 0.5 * loss_mcp + 0.01 * loss_ortho
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()

        # Validation phase on held-out tasks
        model.eval()
        val_risk_correct = 0
        val_tax_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in loader_val:
                embs = batch["turn_embeddings"].to(device)
                mask = batch["turn_mask"].to(device)
                risk_y = batch["risk_label"].to(device)
                tax_y = batch["taxonomy_label"].to(device)

                out = model(embs, turn_mask=mask, task=None)

                pred_risk = out["risk"]["calibrated_logits"].argmax(dim=-1)
                pred_tax = out["taxonomy"]["calibrated_logits"].argmax(dim=-1)

                val_risk_correct += (pred_risk == risk_y).sum().item()
                val_tax_correct += (pred_tax == tax_y).sum().item()
                val_total += risk_y.size(0)

        risk_acc = (val_risk_correct / max(val_total, 1)) * 100.0
        tax_acc = (val_tax_correct / max(val_total, 1)) * 100.0
        elapsed = time.time() - t0

        is_best = risk_acc > best_val_risk_acc
        mark = " * (Best)" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss/len(loader_train):.4f} | Val Risk Acc: {risk_acc:.2f}% | Val Tax Acc: {tax_acc:.2f}% | Time: {elapsed:.1f}s{mark}")

        if is_best:
            best_val_risk_acc = risk_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "d_input": d_in,
                    "d_hidden": 512,
                    "d_message": 512,
                    "d_proj": 256,
                    "num_agents": 4,
                    "encoder_model": encoder_model if not smoke_test else "synthetic",
                },
                "val_risk_acc": risk_acc,
                "val_tax_acc": tax_acc,
                "is_smoke_test": smoke_test,
                "epoch": epoch,
            }, best_ckpt_path)

    print(f"\n[SentinelMT:Save] Best production checkpoint saved to {best_ckpt_path} (Val Risk Acc: {best_val_risk_acc:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Train KairosSentinelMT on MT-AgentRisk")
    parser.add_argument("--data_dir", type=str, default="./MT-AgentRisk", help="Path to MT-AgentRisk dataset")
    parser.add_argument("--encoder_model", type=str, default="bert-base-uncased", help="Hugging Face encoder model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick synthetic smoke test (explicitly marked)")
    parser.add_argument("--out_dir", type=str, default="./kairos_checkpoints", help="Output directory")

    args = parser.parse_args()
    train_sentinel_mt(
        data_dir=args.data_dir,
        encoder_model=args.encoder_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke_test=args.smoke_test,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
