"""
train_kairos_multitask.py — Production Multi-Task Training, Calibration & Conformal Verification Pipeline

Trains Multi-Task Kairos (MT-Kairos, 9.4M parameters) across 3 core production tracks:
1. Track 'tools': Salesforce xLAM (Top-64 tools + Class 0 = NO_TOOL / PASS_THROUGH)
2. Track 'intent': Banking77 (77 customer & financial intents)
3. Track 'safety': Jailbreak & Prompt-Injection Guardrail (Binary safe vs attack)

Features:
- Automated Latent Caching for all 3 tracks with Laya ModernBERT-large backbone.
- Scaled CIDA Deliberation Core (d_hidden=512, d_message=512, M=4 agents).
- Interleaved Multi-Task Training with scale-balanced Brier loss and Label Smoothing (0.1).
- Cosine Annealing learning rate schedule.
- Per-task Temperature Scaling (Guo et al., 2017) via L-BFGS.
- Conformal Prediction Gates (1 - alpha = 95%) with Honest Abstain verification.
- Automatic best checkpoint saving and comprehensive multi-task report generation.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from cida_plugin.multitask import MultiTaskCIDAPlugin
from cida_plugin.losses import CIDALoss


# ===========================================================================
# 1. Environment & Path Resolution
# ===========================================================================

def resolve_dir(path_str: str) -> str:
    """Finds directory whether invoked from repo root, parent directory, or Colab /content."""
    if os.path.exists(path_str) and (glob.glob(os.path.join(path_str, "shard_*.pt")) or glob.glob(os.path.join(path_str, "*.json"))):
        return path_str
    candidate1 = os.path.join(SCRIPT_DIR, path_str.lstrip("./").lstrip(".\\"))
    if os.path.exists(candidate1) and (glob.glob(os.path.join(candidate1, "shard_*.pt")) or glob.glob(os.path.join(candidate1, "*.json"))):
        return candidate1
    candidate2 = os.path.join(os.getcwd(), "CIDA-plugin", path_str.lstrip("./").lstrip(".\\"))
    if os.path.exists(candidate2) and (glob.glob(os.path.join(candidate2, "shard_*.pt")) or glob.glob(os.path.join(candidate2, "*.json"))):
        return candidate2
    if os.path.exists(candidate1):
        return candidate1
    return path_str


# ===========================================================================
# 2. Caching Engine for All 3 Tracks
# ===========================================================================

def load_laya_backbone(cache_dir: str = "./weights") -> Dict[str, Any]:
    os.makedirs(cache_dir, exist_ok=True)
    from transformers import AutoModel, AutoTokenizer, AutoConfig, ModernBertModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    repo = "convaiinnovations/laya"
    print(f"[backbone] Fetching tokenizer for {repo}...")
    tok = None
    for kwargs in [{"trust_remote_code": True}, {"use_fast": False}]:
        try:
            tok = AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir, **kwargs)
            break
        except Exception:
            pass
    if tok is None:
        tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large", cache_dir=cache_dir)

    print(f"[backbone] Fetching model for {repo}...")
    try:
        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-large", cache_dir=cache_dir)
        model = ModernBertModel(config)
        weights_path = hf_hub_download(repo_id=repo, filename="model.safetensors", cache_dir=cache_dir)
        raw_sd = load_file(weights_path)
        backbone_sd = {k[len("encoder."):]: v for k, v in raw_sd.items() if k.startswith("encoder.")}
        if backbone_sd:
            model.load_state_dict(backbone_sd, strict=False)
        else:
            model = AutoModel.from_pretrained("answerdotai/ModernBERT-large", cache_dir=cache_dir)
    except Exception:
        model = AutoModel.from_pretrained("answerdotai/ModernBERT-large", cache_dir=cache_dir)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return {"tokenizer": tok, "model": model}


def cache_all_tracks(cache_dir: str = "./weights", out_base: str = "./latent_cache", max_samples_per_track: Optional[int] = None):
    """Caches latent representations for Tools, Intent, and Safety tracks."""
    import datasets
    from collections import Counter

    backbone = load_laya_backbone(cache_dir=cache_dir)
    tok = backbone["tokenizer"]
    model = backbone["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def _encode_and_shard(data_pairs: List[Tuple[str, int]], track_name: str, split: str):
        out_dir = os.path.join(out_base, track_name, split)
        os.makedirs(out_dir, exist_ok=True)
        batch_texts, batch_labels = [], []
        shard_idx, in_shard = 0, 0
        buf_pooled, buf_seq, buf_mask, buf_lbl = [], [], [], []

        def flush():
            nonlocal shard_idx, buf_pooled, buf_seq, buf_mask, buf_lbl, in_shard
            if not buf_pooled:
                return
            p = torch.cat(buf_pooled, dim=0)
            s = torch.cat(buf_seq, dim=0)
            m = torch.cat(buf_mask, dim=0)
            l = torch.cat(buf_lbl, dim=0)
            spath = os.path.join(out_dir, f"shard_{shard_idx:05d}.pt")
            torch.save({"pooled": p, "seq": s, "mask": m, "label": l}, spath)
            print(f"[cache:{track_name}/{split}] Wrote {spath} ({p.size(0)} samples)")
            shard_idx += 1
            in_shard = 0
            buf_pooled, buf_seq, buf_mask, buf_lbl = [], [], [], []

        for i, (q, lbl) in enumerate(data_pairs):
            if max_samples_per_track and i >= max_samples_per_track:
                break
            batch_texts.append(q)
            batch_labels.append(lbl)

            if len(batch_texts) == 32:
                enc = tok(batch_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**enc)
                    seq = out.last_hidden_state
                    pooled = seq[:, 0, :]
                buf_pooled.append(pooled.cpu())
                buf_seq.append(seq.cpu())
                buf_mask.append(enc["attention_mask"].cpu())
                buf_lbl.append(torch.tensor(batch_labels, dtype=torch.long))
                in_shard += len(batch_texts)
                batch_texts, batch_labels = [], []

            if in_shard >= 1000:
                flush()

        if batch_texts:
            enc = tok(batch_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc)
                seq = out.last_hidden_state
                pooled = seq[:, 0, :]
            buf_pooled.append(pooled.cpu())
            buf_seq.append(seq.cpu())
            buf_mask.append(enc["attention_mask"].cpu())
            buf_lbl.append(torch.tensor(batch_labels, dtype=torch.long))
            flush()

    # ── Track 1: Tools (xLAM with NO_TOOL Class 0) ───────────────────────────
    print("\n[cache] Processing Track 1: Tools (Salesforce xLAM)...")
    try:
        ds_xlam = datasets.load_dataset("minpeter/xlam-function-calling-60k-parsed", split="train")
    except Exception:
        ds_xlam = datasets.load_dataset("Salesforce/xlam-function-calling-60k", split="train")

    records, tool_counter = [], Counter()
    for row in ds_xlam:
        query, tool_name = "", None
        if "messages" in row and row["messages"]:
            for m in row["messages"]:
                if isinstance(m, dict):
                    if m.get("role") == "user" and not query:
                        query = m.get("content", "")
                    elif m.get("role") == "assistant":
                        tcs = m.get("tool_calls")
                        if tcs and isinstance(tcs, list) and len(tcs) > 0 and isinstance(tcs[0], dict):
                            fn = tcs[0].get("function")
                            tool_name = fn.get("name") if isinstance(fn, dict) else tcs[0].get("name")
        if not query and "query" in row:
            query = row.get("query", "")
        if not tool_name and "answers" in row:
            ans = row.get("answers", "")
            if isinstance(ans, str):
                try:
                    parsed = json.loads(ans)
                    if isinstance(parsed, list) and len(parsed) > 0 and "name" in parsed[0]:
                        tool_name = parsed[0]["name"]
                except Exception:
                    pass
            elif isinstance(ans, list) and len(ans) > 0 and "name" in ans[0]:
                tool_name = ans[0]["name"]

        if tool_name and query:
            records.append((query, tool_name))
            tool_counter[tool_name] += 1

    top_64_tools = [t for t, _ in tool_counter.most_common(64)]
    # Tool index 1..64. Index 0 is reserved for NO_TOOL / PASS_THROUGH
    tool_to_idx = {t: i + 1 for i, t in enumerate(top_64_tools)}

    # Negative non-tool samples for Class 0
    negative_prompts = [
        ("Tell me a creative joke about a programmer.", 0),
        ("Write a short poem about the ocean and stars.", 0),
        ("What is the meaning of life according to Socrates?", 0),
        ("Explain how photosynthesis works in plants.", 0),
        ("Hello, how are you doing today?", 0),
        ("Can you help me brainstorm names for a coffee shop?", 0),
        ("Summarize the plot of Hamlet in two sentences.", 0),
        ("What are the main causes of the French Revolution?", 0),
        ("Give me five motivational quotes for studying.", 0),
        ("Translate 'good morning' to five different languages.", 0),
    ] * 100

    tool_samples = [(q, tool_to_idx[t]) for q, t in records if t in tool_to_idx]
    tool_samples.extend(negative_prompts)
    random.seed(42)
    random.shuffle(tool_samples)

    n_tool_train = int(len(tool_samples) * 0.8)
    _encode_and_shard(tool_samples[:n_tool_train], "tools", "train")
    _encode_and_shard(tool_samples[n_tool_train:], "tools", "eval")

    # ── Track 2: Intent (Banking77) ──────────────────────────────────────────
    print("\n[cache] Processing Track 2: Intent (Banking77)...")
    ds_bank_train = datasets.load_dataset("mteb/banking77", split="train")
    ds_bank_eval = datasets.load_dataset("mteb/banking77", split="test")

    intent_names = ds_bank_train.features["label"].names if hasattr(ds_bank_train.features["label"], "names") else [str(i) for i in range(77)]
    intent_map = {i: name for i, name in enumerate(intent_names)}

    bank_train = [(r["text"], int(r["label"])) for r in ds_bank_train]
    bank_eval = [(r["text"], int(r["label"])) for r in ds_bank_eval]
    _encode_and_shard(bank_train, "intent", "train")
    _encode_and_shard(bank_eval, "intent", "eval")

    # ── Track 3: Safety Guardrail (Jailbreak Detection) ──────────────────────
    print("\n[cache] Processing Track 3: Safety Guardrail...")
    safety_samples = []
    try:
        ds_jail = datasets.load_dataset("jackhhao/jailbreak-classification", split="train")
        for r in ds_jail:
            text = r.get("text", "") or r.get("prompt", "")
            lbl = int(r.get("label", 0))
            if text:
                safety_samples.append((text, lbl))
    except Exception:
        # Fallback to deepset prompt-injections
        ds_inj = datasets.load_dataset("deepset/prompt-injections", split="train")
        for r in ds_inj:
            safety_samples.append((r["text"], int(r["label"])))

    random.seed(42)
    random.shuffle(safety_samples)
    n_safe_train = int(len(safety_samples) * 0.8)
    _encode_and_shard(safety_samples[:n_safe_train], "safety", "train")
    _encode_and_shard(safety_samples[n_safe_train:], "safety", "eval")

    # Save Unified Label Maps for Production SDK
    tool_map = {0: "NO_TOOL / PASS_THROUGH", **{i + 1: t for i, t in enumerate(top_64_tools)}}
    safety_map = {0: "SAFE", 1: "ATTACK_OR_INJECTION"}
    label_maps = {
        "tools": tool_map,
        "intent": intent_map,
        "safety": safety_map,
    }
    label_path = os.path.join(out_base, "label_maps.json")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2, ensure_ascii=False)
    print(f"\n[cache] Saved unified multi-task label maps to {label_path}")
    print("[cache] Done caching all 3 tracks successfully!")


# ===========================================================================
# 3. Multi-Task Dataset & Interleaved Batching
# ===========================================================================

class ShardedLatentDataset(Dataset):
    def __init__(self, shard_dir: str):
        resolved = resolve_dir(shard_dir)
        self.paths = sorted(glob.glob(os.path.join(resolved, "shard_*.pt")))
        if not self.paths:
            raise FileNotFoundError(f"No shards found in {shard_dir} ({resolved})")
        self.index = []
        self._cache = {}
        for si, p in enumerate(self.paths):
            s = torch.load(p, map_location="cpu", weights_only=True)
            n = s["label"].size(0)
            self.index.extend([(si, j) for j in range(n)])
            self._cache[si] = s

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, j = self.index[idx]
        s = self._cache[si]
        return {
            "pooled": s["pooled"][j],
            "seq": s["seq"][j],
            "mask": s["mask"][j],
            "label": s["label"][j],
        }


# ===========================================================================
# 4. Calibration & Conformal Verification
# ===========================================================================

def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = torch.max(probs, dim=1)
    correct = (pred == targets).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    total_samples = probs.size(0)
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf <= bins[i + 1]) if i == 0 else (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            bin_weight = m.float().sum() / total_samples
            bin_acc = correct[m].mean()
            bin_conf = conf[m].mean()
            ece += bin_weight * (bin_conf - bin_acc).abs()
    return ece.item()


class TaskTemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def calibrate(self, val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.05, max_iter=50)
        criterion = nn.CrossEntropyLoss()
        def _eval():
            optimizer.zero_grad()
            l = criterion(self.forward(val_logits), val_labels)
            l.backward()
            return l
        optimizer.step(_eval)
        return float(self.temperature.item())


class TaskConformalGate:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = None

    def fit(self, val_probs: torch.Tensor, val_labels: torch.Tensor):
        n = val_probs.size(0)
        true_probs = val_probs.gather(1, val_labels.unsqueeze(1)).squeeze(1)
        scores = 1.0 - true_probs.cpu().numpy()
        p_val = min(1.0, max(0.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n))
        self.q_hat = float(np.quantile(scores, p_val))
        return self.q_hat

    def evaluate(self, test_probs: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, Any]:
        threshold = 1.0 - self.q_hat
        mask = (test_probs >= threshold)
        sets = [torch.where(mask[i])[0].tolist() for i in range(mask.size(0))]
        y_true = test_labels.cpu().tolist()
        covered = sum(y in s for y, s in zip(y_true, sets))
        n = len(y_true)
        sizes = [len(s) for s in sets]
        return {
            "empirical_coverage": covered / max(n, 1),
            "singleton_rate": sum(1 for sz in sizes if sz == 1) / n,
            "honest_abstain_rate": sum(1 for sz in sizes if sz == 0) / n,
            "avg_set_size": float(np.mean(sizes)),
        }


# ===========================================================================
# 5. Multi-Task Training Loop
# ===========================================================================

def train_multitask(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    out_dir: str = "./kairos_checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("\n" + "=" * 70)
    print("  KAIROS MULTI-TASK PRODUCTION TRAINING PIPELINE (MT-Kairos)")
    print(f"  Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 70 + "\n")

    # Load datasets for all 3 tracks
    ds_tools_train = ShardedLatentDataset("./latent_cache/tools/train")
    ds_tools_eval = ShardedLatentDataset("./latent_cache/tools/eval")

    ds_intent_train = ShardedLatentDataset("./latent_cache/intent/train")
    ds_intent_eval = ShardedLatentDataset("./latent_cache/intent/eval")

    ds_safety_train = ShardedLatentDataset("./latent_cache/safety/train")
    ds_safety_eval = ShardedLatentDataset("./latent_cache/safety/eval")

    print(f"[MT-Kairos:Data] Tools:  {len(ds_tools_train):,} train,  {len(ds_tools_eval):,} eval (65 classes)")
    print(f"[MT-Kairos:Data] Intent: {len(ds_intent_train):,} train, {len(ds_intent_eval):,} eval (77 classes)")
    print(f"[MT-Kairos:Data] Safety: {len(ds_safety_train):,} train, {len(ds_safety_eval):,} eval (2 classes)")

    loader_tools = DataLoader(ds_tools_train, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_intent = DataLoader(ds_intent_train, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_safety = DataLoader(ds_safety_train, batch_size=batch_size, shuffle=True, drop_last=True)

    eval_tools = DataLoader(ds_tools_eval, batch_size=batch_size, shuffle=False)
    eval_intent = DataLoader(ds_intent_eval, batch_size=batch_size, shuffle=False)
    eval_safety = DataLoader(ds_safety_eval, batch_size=batch_size, shuffle=False)

    d_in = ds_tools_train[0]["pooled"].size(-1)

    # Initialize Scaled Multi-Task Model (d_hidden=512, ~9.4M params)
    model = MultiTaskCIDAPlugin(
        d_input=d_in,
        d_hidden=512,
        d_message=512,
        num_agents=4,
        max_rounds=3,
        num_tools=65,
        num_intents=77,
        num_safety=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MT-Kairos:Init] Trainable Deliberation Parameters: {n_params:,} (~9.4M PTRM Scale)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = CIDALoss(lambda_cal=0.4, label_smoothing=0.1)

    best_mean_acc = 0.0
    best_ckpt = os.path.join(out_dir, "kairos_multitask_best.pt")

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        iter_tools = iter(loader_tools)
        iter_intent = iter(loader_intent)
        iter_safety = iter(loader_safety)

        n_steps = max(len(loader_tools), len(loader_intent), len(loader_safety))
        epoch_loss = 0.0

        for _ in range(n_steps):
            optimizer.zero_grad()
            total_step_loss = torch.tensor(0.0, device=device)

            # 1. Tools Step
            try:
                b_t = next(iter_tools)
            except StopIteration:
                iter_tools = iter(loader_tools)
                b_t = next(iter_tools)
            out_t = model(b_t["pooled"].to(device), seq_output=b_t["seq"].to(device), mask=b_t["mask"].to(device), task="tools")
            l_t, _ = loss_fn(out_t["p_final"], b_t["label"].to(device), out_t["b_all"])
            total_step_loss = total_step_loss + l_t

            # 2. Intent Step
            try:
                b_i = next(iter_intent)
            except StopIteration:
                iter_intent = iter(loader_intent)
                b_i = next(iter_intent)
            out_i = model(b_i["pooled"].to(device), seq_output=b_i["seq"].to(device), mask=b_i["mask"].to(device), task="intent")
            l_i, _ = loss_fn(out_i["p_final"], b_i["label"].to(device), out_i["b_all"])
            total_step_loss = total_step_loss + l_i

            # 3. Safety Step
            try:
                b_s = next(iter_safety)
            except StopIteration:
                iter_safety = iter(loader_safety)
                b_s = next(iter_safety)
            out_s = model(b_s["pooled"].to(device), task="safety")
            l_s, _ = loss_fn(out_s["p_final"], b_s["label"].to(device), out_s["b_all"])
            total_step_loss = total_step_loss + 0.5 * l_s

            total_step_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total_step_loss.item()

        scheduler.step()

        # Validation across all 3 tracks
        model.eval()
        def _eval_loader(loader, task_name):
            cor, tot = 0, 0
            with torch.no_grad():
                for b in loader:
                    out = model(b["pooled"].to(device), task=task_name)
                    pred = out["p_final"].argmax(dim=1)
                    cor += (pred == b["label"].to(device)).sum().item()
                    tot += b["label"].size(0)
            return cor / max(tot, 1)

        acc_tools = _eval_loader(eval_tools, "tools")
        acc_intent = _eval_loader(eval_intent, "intent")
        acc_safety = _eval_loader(eval_safety, "safety")
        mean_acc = (acc_tools + acc_intent + acc_safety) / 3.0

        elapsed = time.time() - t0
        is_best = mean_acc > best_mean_acc
        if is_best:
            best_mean_acc = mean_acc
            torch.save({"epoch": epoch, "model": model.state_dict(), "mean_acc": mean_acc}, best_ckpt)

        star = " ★ (Best)" if is_best else ""
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Tools: {acc_tools*100:5.2f}% | Intent: {acc_intent*100:5.2f}% | Safety: {acc_safety*100:5.2f}% | Mean: {mean_acc*100:5.2f}% | Time: {elapsed:4.1f}s{star}")

    # Load best checkpoint and perform final calibration & conformal verification
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"\n[MT-Kairos:Eval] Loaded Best Checkpoint (Epoch {ckpt['epoch']+1}) with Mean Acc: {ckpt['mean_acc']*100:.2f}%\n")

    def _full_eval(loader, task_name):
        all_p, all_y = [], []
        with torch.no_grad():
            for b in loader:
                out = model(b["pooled"].to(device), task=task_name)
                all_p.append(out["p_final"].cpu())
                all_y.append(b["label"].cpu())
        p = torch.cat(all_p)
        y = torch.cat(all_y)
        acc = (p.argmax(dim=1) == y).float().mean().item()
        raw_ece = compute_ece(p, y)

        # Temperature Scaling on 50% split
        n_half = p.size(0) // 2
        cal_p, cal_y = p[:n_half], y[:n_half]
        tst_p, tst_y = p[n_half:], y[n_half:]

        scaler = TaskTemperatureScaler()
        T = scaler.calibrate(cal_p.log().clamp(-10, 10), cal_y)
        calib_p = F.softmax(tst_p.log().clamp(-10, 10) / T, dim=-1)
        cal_ece = compute_ece(calib_p, tst_y)

        # Conformal Gate
        gate = TaskConformalGate(alpha=0.05)
        gate.fit(F.softmax(cal_p.log().clamp(-10, 10) / T, dim=-1), cal_y)
        conf_res = gate.evaluate(calib_p, tst_y)

        return {
            "accuracy": acc,
            "raw_ece": raw_ece,
            "temperature": T,
            "calibrated_ece": cal_ece,
            "conformal": conf_res,
            "q_val": gate.q_hat,
        }

    res_tools = _full_eval(eval_tools, "tools")
    res_intent = _full_eval(eval_intent, "intent")
    res_safety = _full_eval(eval_safety, "safety")

    print("=" * 70)
    print("  FINAL MULTI-TASK EVALUATION REPORT")
    print(f"  1. Tools (xLAM):  Acc: {res_tools['accuracy']*100:5.2f}% | Calibrated ECE: {res_tools['calibrated_ece']:.4f} | Coverage: {res_tools['conformal']['empirical_coverage']*100:.1f}%")
    print(f"  2. Intent (Bank): Acc: {res_intent['accuracy']*100:5.2f}% | Calibrated ECE: {res_intent['calibrated_ece']:.4f} | Coverage: {res_intent['conformal']['empirical_coverage']*100:.1f}%")
    print(f"  3. Safety Guard:  Acc: {res_safety['accuracy']*100:5.2f}% | Calibrated ECE: {res_safety['calibrated_ece']:.4f} | Coverage: {res_safety['conformal']['empirical_coverage']*100:.1f}%")
    print("=" * 70 + "\n")

    # Load label maps if saved during caching
    label_maps = None
    label_file = resolve_dir(os.path.join(latent_dir, "label_maps.json"))
    if os.path.exists(label_file):
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                label_maps = json.load(f)
        except Exception:
            pass

    prod_ckpt = os.path.join(out_dir, "kairos_multitask_production.pt")
    torch.save({
        "epoch": ckpt["epoch"],
        "model": model.state_dict(),
        "mean_acc": ckpt["mean_acc"],
        "temperatures": {
            "tools": res_tools["temperature"],
            "intent": res_intent["temperature"],
            "safety": res_safety["temperature"],
        },
        "conformal_q": {
            "tools": res_tools["q_val"],
            "intent": res_intent["q_val"],
            "safety": res_safety["q_val"],
        },
        "label_maps": label_maps,
        "config": {
            "d_input": d_in,
            "d_hidden": 512,
            "d_message": 512,
            "num_agents": 4,
            "max_rounds": 3,
            "num_tools": 65,
            "num_intents": 77,
            "num_safety": 2,
        },
        "report": {
            "tools": res_tools,
            "intent": res_intent,
            "safety": res_safety,
        }
    }, prod_ckpt)
    print(f"[MT-Kairos:Save] Saved production calibrated model to {prod_ckpt}\n")

    report = {
        "parameters": n_params,
        "tools": res_tools,
        "intent": res_intent,
        "safety": res_safety,
    }
    with open(os.path.join(out_dir, "kairos_multitask_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report


# ===========================================================================
# 6. CLI Entrypoint
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Kairos Training Pipeline")
    parser.add_argument("--cache_all", action="store_true", help="Cache datasets for all 3 tracks")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--out_dir", type=str, default="./kairos_checkpoints", help="Output directory")
    parser.add_argument("--smoke_test", action="store_true", help="Quick sanity check on synthetic data")

    args = parser.parse_args()

    if args.smoke_test:
        print("[SmokeTest] Running MultiTask sanity check...")
        m = MultiTaskCIDAPlugin(d_input=128, d_hidden=64, d_message=64, num_agents=4, num_tools=10, num_intents=12, num_safety=2)
        x = torch.randn(2, 128)
        out = m(x)
        print(f"[SmokeTest] MultiTask forward passed! Tools: {out['tools']['p_final'].shape}, Intent: {out['intent']['p_final'].shape}, Safety: {out['safety']['p_final'].shape}")
        return

    if args.cache_all:
        cache_all_tracks()
        print("\n[Kairos:Cache] All 3 tracks cached successfully! Now run training.\n")
        return

    train_multitask(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
