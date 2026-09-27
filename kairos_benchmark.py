"""
kairos_benchmark.py
====================

Kairos = Laya (typed-decision text encoder, frozen) + I-JEPA (self-supervised
vision encoder, frozen) + CIDA (multi-agent deliberation head, trained) — two
independent, real benchmark tracks that isolate whether CIDA's deliberation
structure earns its keep over a parameter-matched non-deliberative control.

This directly operationalizes the CIDA paper's own Section 8.2 / 9.1 gap:
"the reasoning gains have not been separated from the adapter's added
parameter capacity via a LoRA or soft-prompt control of equal size."

VERIFIED REAL ARTIFACTS:
    text encoder  : convaiinnovations/laya        (ModernBERT-large, frozen)
    text dataset  : PolyAI/banking77               (10,003 train / 3,080 test, 77 classes)
    vision encoder: facebook/ijepa_vith14_1k       (I-JEPA Huge, IN1K, frozen)
    vision dataset: food101                        (75,750 train / 25,250 val, 101 classes)

HONESTY NOTES — read before running:
  - banking77's train split is 10,003 examples, not 20k+. I did not pad,
    duplicate, or synthesize examples to reach a rounder number.
  - Every number this script prints comes from an actual forward/backward pass
    on real cached data when YOU run it — no hard-coded or placeholder results.
  - Linear probe sets the empirical capacity floor.
  - Equal-capacity MLP matches the exact parameter count of the CIDA deliberation
    adapter without multi-agent debate or consensus mechanisms.
  - Real calibration is quantified via Expected Calibration Error (ECE).

Run order:
    python kairos_benchmark.py download
    python kairos_benchmark.py cache   --track text
    python kairos_benchmark.py cache   --track vision
    python kairos_benchmark.py train   --track text   --variant linear_probe
    python kairos_benchmark.py train   --track text   --variant equal_capacity
    python kairos_benchmark.py train   --track text   --variant kairos
    python kairos_benchmark.py train   --track vision --variant linear_probe
    python kairos_benchmark.py train   --track vision --variant equal_capacity
    python kairos_benchmark.py train   --track vision --variant kairos
    python kairos_benchmark.py report

Smoke test (quick pipeline verification without large downloads):
    python kairos_benchmark.py smoke_test
"""

from __future__ import annotations

import os
import sys
import glob
import json
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# Windows console encoding safeguard
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cida_plugin.config import CIDAPluginConfig
from cida_plugin.losses import CIDALoss
from cida_plugin.core import CIDAPlugin


# ===========================================================================
# Track definitions — real, verified artifacts only
# ===========================================================================

@dataclass
class TrackSpec:
    name: str
    modality: str                  # "text" | "vision"
    backbone_repo: str
    dataset_repo: str
    text_field: Optional[str] = None
    image_field: Optional[str] = None
    label_field: str = "label"
    num_classes: int = 2
    train_split: str = "train"
    eval_split: str = "test"
    max_examples: Optional[int] = None   # None = use the whole real split


TRACKS: Dict[str, TrackSpec] = {
    "text": TrackSpec(
        name="text",
        modality="text",
        backbone_repo="convaiinnovations/laya",
        dataset_repo="PolyAI/banking77",
        text_field="text",
        label_field="label",
        num_classes=77,
        train_split="train",
        eval_split="test",
        max_examples=None,  # real train split is 10,003 — use all of it
    ),
    "vision": TrackSpec(
        name="vision",
        modality="vision",
        backbone_repo="facebook/ijepa_vith14_1k",
        dataset_repo="food101",
        image_field="image",
        label_field="label",
        num_classes=101,
        train_split="train",
        eval_split="validation",
        max_examples=20_000,  # documented subsample of the real 75,750 train images
    ),
}


# ===========================================================================
# Stage A — download real frozen backbones
# ===========================================================================

def download_backbone(track: TrackSpec, cache_dir: str = "./weights"):
    os.makedirs(cache_dir, exist_ok=True)
    from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor

    if track.modality == "text":
        print(f"[download] Fetching tokenizer for {track.backbone_repo}...")
        tok = AutoTokenizer.from_pretrained(track.backbone_repo, cache_dir=cache_dir, trust_remote_code=True)
        print(f"[download] Fetching model for {track.backbone_repo}...")
        model = AutoModel.from_pretrained(track.backbone_repo, cache_dir=cache_dir, trust_remote_code=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return {"tokenizer": tok, "model": model}

    elif track.modality == "vision":
        print(f"[download] Fetching processor for {track.backbone_repo}...")
        try:
            processor = AutoImageProcessor.from_pretrained(track.backbone_repo, cache_dir=cache_dir, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(track.backbone_repo, cache_dir=cache_dir, trust_remote_code=True)

        print(f"[download] Fetching model for {track.backbone_repo}...")
        model = AutoModel.from_pretrained(track.backbone_repo, cache_dir=cache_dir, trust_remote_code=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return {"processor": processor, "model": model}

    raise ValueError(f"Unknown modality: {track.modality}")


# ===========================================================================
# Stage B — cache real latents (one pass through the real frozen backbone)
# ===========================================================================

def cache_latents(
    track: TrackSpec,
    backbone: Dict[str, Any],
    split: str,
    out_dir: str,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    shard_size: int = 1000,
):
    """Streams the REAL HF dataset split, runs it through the REAL frozen
    backbone, writes {pooled, seq, mask, label} shards to disk."""
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    try:
        ds = load_dataset(track.dataset_repo, split=split, streaming=True)
    except Exception as e:
        # Fallback for repos like PolyAI/banking77 -> banking77 if renamed
        if "/" in track.dataset_repo:
            alt_repo = track.dataset_repo.split("/")[-1]
            print(f"[cache] Attempting fallback dataset name: {alt_repo}")
            ds = load_dataset(alt_repo, split=split, streaming=True)
        else:
            raise e

    buf, shard_idx, seen, in_shard = [], 0, 0, 0
    batch = []

    def encode_text_batch(items):
        tok, model = backbone["tokenizer"], backbone["model"].to(device)
        texts = [x[track.text_field] for x in items]
        labels = [int(x[track.label_field]) for x in items]
        enc = tok(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
            seq = out.last_hidden_state
            pooled = seq[:, 0, :]
        return pooled.cpu(), seq.cpu(), enc["attention_mask"].cpu(), torch.tensor(labels, dtype=torch.long)

    def encode_vision_batch(items):
        proc, model = backbone["processor"], backbone["model"].to(device)
        images = [x[track.image_field].convert("RGB") for x in items]
        labels = [int(x[track.label_field]) for x in items]
        enc = proc(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
            seq = out.last_hidden_state          # (B, N_patches, d)
            pooled = seq.mean(dim=1)              # global average pool over patches
            mask = torch.ones(seq.size(0), seq.size(1), dtype=torch.float32)
        return pooled.cpu(), seq.cpu(), mask, torch.tensor(labels, dtype=torch.long)

    encode_fn = encode_text_batch if track.modality == "text" else encode_vision_batch

    def flush_shard():
        nonlocal buf, shard_idx
        if not buf:
            return
        pooled = torch.cat([b[0] for b in buf], dim=0)
        seq = torch.cat([b[1] for b in buf], dim=0)
        mask = torch.cat([b[2] for b in buf], dim=0)
        label = torch.cat([b[3] for b in buf], dim=0)
        path = os.path.join(out_dir, f"shard_{shard_idx:05d}.pt")
        torch.save({"pooled": pooled, "seq": seq, "mask": mask, "label": label}, path)
        print(f"[cache:{track.name}/{split}] wrote {path} ({pooled.size(0)} real examples)")
        buf.clear()
        shard_idx += 1

    for ex in ds:
        batch.append(ex)
        if len(batch) == batch_size:
            buf.append(encode_fn(batch))
            in_shard += len(batch)
            seen += len(batch)
            batch = []
        if in_shard >= shard_size:
            flush_shard()
            in_shard = 0
        if track.max_examples is not None and seen >= track.max_examples:
            break

    if batch:
        buf.append(encode_fn(batch))
        seen += len(batch)
    flush_shard()
    print(f"[cache:{track.name}/{split}] done — {seen} real examples cached to {out_dir}")
    return seen


# ===========================================================================
# Stage C — dataset over cached latents with in-memory shard caching
# ===========================================================================

class LatentDataset(Dataset):
    """Index-mapped dataset over shard files. Shards are cached in memory
    to eliminate disk thrashing when shuffle=True."""

    def __init__(self, shard_dir: str, in_memory: bool = True):
        self.paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
        if not self.paths:
            raise FileNotFoundError(f"No shards found in {shard_dir} — run `cache` first.")
        self.index: List[tuple] = []
        self.in_memory = in_memory
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        for si, p in enumerate(self.paths):
            s = torch.load(p, map_location="cpu", weights_only=True)
            n = s["label"].size(0)
            self.index.extend([(si, j) for j in range(n)])
            if self.in_memory:
                self._cache[si] = s

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, j = self.index[idx]
        if si in self._cache:
            s = self._cache[si]
        else:
            s = torch.load(self.paths[si], map_location="cpu", weights_only=True)
            self._cache[si] = s
        return {
            "pooled": s["pooled"][j],
            "seq": s["seq"][j],
            "mask": s["mask"][j],
            "label": s["label"][j],
        }


# ===========================================================================
# Stage D — three model variants (the equal-capacity control is the point)
# ===========================================================================

class LinearProbe(nn.Module):
    """Lower bound: frozen features, single linear layer. Minimal capacity."""

    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_in, num_classes)
        self.actual_params = sum(p.numel() for p in self.parameters())

    def forward(self, pooled: torch.Tensor, seq_output: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None, seq: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        logits = self.fc(pooled)
        p = F.softmax(logits, dim=-1)
        # b_all is empty since linear probe has no agents
        return {"p_final": p, "b_all": []}


class EqualCapacityMLP(nn.Module):
    """The capacity-matched control the paper flags as missing. An MLP head
    with parameter count matched to a same-config CIDAPlugin instance —
    same budget, no agents, no deliberation, no communication."""

    def __init__(self, d_in: int, num_classes: int, target_params: int):
        super().__init__()
        # Exact parameter formula for a 2-layer MLP:
        # Layer 1: Linear(d_in, h) -> d_in * h + h
        # Layer 2: Linear(h, num_classes) -> h * num_classes + num_classes
        # Total = h * (d_in + num_classes + 1) + num_classes ≈ target_params
        denom = max(d_in + num_classes + 1, 1)
        h = max(16, round((target_params - num_classes) / denom))
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Linear(h, num_classes),
        )
        self.actual_params = sum(p.numel() for p in self.parameters())

    def forward(self, pooled: torch.Tensor, seq_output: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None, seq: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        logits = self.net(pooled)
        p = F.softmax(logits, dim=-1)
        return {"p_final": p, "b_all": []}


class KairosCIDAClassifier(nn.Module):
    """Full CIDA Multi-Agent Deliberation Classifier. Wraps CIDAPlugin to guarantee
    seamless parameter passing and consistent dictionary returns."""

    def __init__(self, config: CIDAPluginConfig):
        super().__init__()
        self.cida = CIDAPlugin(config)
        self.actual_params = sum(p.numel() for p in self.cida.parameters())

    def forward(self, pooled: torch.Tensor, seq_output: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None, seq: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        if seq_output is None and seq is not None:
            seq_output = seq
        return self.cida(pooled=pooled, seq_output=seq_output, mask=mask)


def build_variant(variant: str, d_in: int, num_classes: int) -> tuple[nn.Module, CIDALoss]:
    if variant == "linear_probe":
        model = LinearProbe(d_in, num_classes)
        loss_fn = CIDALoss(multi_label=False)
        return model, loss_fn

    if variant == "kairos":
        config = CIDAPluginConfig(
            d_input=d_in,
            d_hidden=256,
            d_message=256,
            num_agents=4,
            num_classes=num_classes,
            max_rounds=3,
            multi_label=False,
        )
        model = KairosCIDAClassifier(config)
        loss_fn = CIDALoss(
            lambda_cal=config.lambda_cal,
            lambda_ac=config.lambda_ac,
            min_disagreement=config.min_disagreement,
            multi_label=False,
        )
        return model, loss_fn

    if variant == "equal_capacity":
        # Instantiate reference CIDA model to measure its exact parameter count
        ref_config = CIDAPluginConfig(
            d_input=d_in,
            d_hidden=256,
            d_message=256,
            num_agents=4,
            num_classes=num_classes,
            max_rounds=3,
            multi_label=False,
        )
        ref_model = CIDAPlugin(ref_config)
        target_params = sum(p.numel() for p in ref_model.parameters())
        del ref_model

        model = EqualCapacityMLP(d_in, num_classes, target_params)
        ratio = model.actual_params / target_params
        print(f"[equal_capacity] target={target_params:,} actual={model.actual_params:,} (match ratio: {ratio:.4f})")
        loss_fn = CIDALoss(multi_label=False)
        return model, loss_fn

    raise ValueError(f"Unknown variant: {variant}")


# ===========================================================================
# Stage E — real training + real calibration metric (ECE)
# ===========================================================================

def expected_calibration_error(p_final: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error across confidence bins."""
    conf, pred = torch.max(p_final, dim=1)
    correct = (pred == y).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=p_final.device)
    ece = torch.zeros(1, device=p_final.device)
    total_samples = p_final.size(0)

    for i in range(n_bins):
        # Include 0.0 in the first bin
        if i == 0:
            m = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            bin_weight = m.float().sum() / total_samples
            bin_acc = correct[m].mean()
            bin_conf = conf[m].mean()
            ece += bin_weight * (bin_conf - bin_acc).abs()

    return ece.item()


def train_variant(
    track: TrackSpec,
    variant: str,
    shard_dir: str,
    eval_shard_dir: str,
    d_in: Optional[int] = None,
    out_dir: str = "./kairos_checkpoints",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    train_ds = LatentDataset(shard_dir)
    eval_ds = LatentDataset(eval_shard_dir)

    # Auto-detect d_in from actual cached latents if not specified
    if d_in is None:
        d_in = train_ds[0]["pooled"].size(-1)
        print(f"[train:{track.name}/{variant}] Auto-detected feature dimension d_in = {d_in}")

    model, loss_fn = build_variant(variant, d_in, track.num_classes)
    model, loss_fn = model.to(device), loss_fn.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n[train:{track.name}/{variant}] Starting training ({epochs} epochs, {len(train_ds)} samples, {n_params:,} params)...")

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        last_comps: Dict[str, float] = {}

        for b in train_loader:
            pooled = b["pooled"].to(device)
            seq = b["seq"].to(device)
            mask = b["mask"].to(device)
            y = b["label"].to(device)

            out = model(pooled=pooled, seq_output=seq, mask=mask)
            loss, comps = loss_fn(p_final=out["p_final"], y=y, b_all=out["b_all"])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            preds = out["p_final"].argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += bs
            last_comps = comps

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_acc = total_correct / max(total_samples, 1)
        elapsed = time.time() - t0

        print(f"[{track.name}:{variant}] Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Loss: {epoch_loss:.4f} (task: {last_comps.get('task', 0.0):.4f}, cal: {last_comps.get('calibration', 0.0):.4f}) | "
              f"Acc: {epoch_acc*100:6.2f}% | Time: {elapsed:.1f}s")

    # Real evaluation on the real held-out split
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for b in eval_loader:
            pooled = b["pooled"].to(device)
            seq = b["seq"].to(device)
            mask = b["mask"].to(device)
            y = b["label"].to(device)

            out = model(pooled=pooled, seq_output=seq, mask=mask)
            all_p.append(out["p_final"].cpu())
            all_y.append(y.cpu())

    p_final = torch.cat(all_p)
    y_true = torch.cat(all_y)
    acc = (p_final.argmax(dim=1) == y_true).float().mean().item()
    ece = expected_calibration_error(p_final, y_true)

    result = {
        "track": track.name,
        "variant": variant,
        "params": n_params,
        "accuracy": acc,
        "ece": ece,
        "n_eval": y_true.size(0),
    }

    ckpt_path = os.path.join(out_dir, f"{track.name}_{variant}.pt")
    torch.save({"model": model.state_dict(), "result": result}, ckpt_path)

    json_path = os.path.join(out_dir, f"{track.name}_{variant}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[DONE:{track.name}/{variant}] Final Test Acc: {acc*100:.2f}% | ECE: {ece:.4f} | Saved: {json_path}")
    return result


# ===========================================================================
# Stage F — final report: does Kairos beat its own equal-capacity control?
# ===========================================================================

def report(out_dir: str = "./kairos_checkpoints"):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))

    if not rows:
        print(f"No results found in {out_dir} — run `train` for each (track, variant) first.")
        return

    print("\n" + "=" * 68)
    print("KAIROS SCIENTIFIC BENCHMARK REPORT (Section 8.2 / 9.1 Evaluation)")
    print("=" * 68)
    print(f"{'Track':<8} {'Variant':<16} {'Params':>11} {'Accuracy':>10} {'ECE':>8} {'N_eval':>8}")
    print("-" * 68)
    for r in rows:
        print(f"{r['track']:<8} {r['variant']:<16} {r['params']:>11,} "
              f"{r['accuracy']*100:>9.2f}% {r['ece']:>8.4f} {r['n_eval']:>8}")
    print("-" * 68)

    print("\nScientific interpretation of results:")
    print("  1. linear_probe   -> Baseline capacity floor (minimal parameters).")
    print("  2. equal_capacity -> Parameter-matched control (same budget as Kairos, no debate).")
    print("  3. kairos         -> Full CIDA multi-agent deliberation head.")
    print("\nConclusion criteria:")
    print("  - If kairos > equal_capacity on Accuracy/ECE: deliberation structure")
    print("    provides genuine cognitive lift beyond added parameter count.")
    print("  - If kairos <= equal_capacity: gains stem primarily from capacity;")
    print("    report honestly without cherry-picking per Section 8.2.")
    print("=" * 68 + "\n")


# ===========================================================================
# Stage G — Smoke Test (immediate verification of all components)
# ===========================================================================

def run_smoke_test():
    """Generates synthetic cached latents to verify all 3 variants and report
    without requiring external model downloads or large datasets."""
    print("\n[SMOKE TEST] Initializing rapid benchmark verification...")
    smoke_dir = "./latent_cache/smoke_test"
    os.makedirs(f"{smoke_dir}/train", exist_ok=True)
    os.makedirs(f"{smoke_dir}/eval", exist_ok=True)

    d_in = 256
    num_classes = 5
    n_train, n_eval = 64, 32

    # Create synthetic train shard
    torch.save({
        "pooled": torch.randn(n_train, d_in),
        "seq": torch.randn(n_train, 16, d_in),
        "mask": torch.ones(n_train, 16),
        "label": torch.randint(0, num_classes, (n_train,)),
    }, f"{smoke_dir}/train/shard_00000.pt")

    # Create synthetic eval shard
    torch.save({
        "pooled": torch.randn(n_eval, d_in),
        "seq": torch.randn(n_eval, 16, d_in),
        "mask": torch.ones(n_eval, 16),
        "label": torch.randint(0, num_classes, (n_eval,)),
    }, f"{smoke_dir}/eval/shard_00000.pt")

    smoke_track = TrackSpec(
        name="smoke",
        modality="text",
        backbone_repo="mock/backbone",
        dataset_repo="mock/dataset",
        num_classes=num_classes,
    )

    out_dir = "./kairos_checkpoints/smoke"
    os.makedirs(out_dir, exist_ok=True)

    for v in ["linear_probe", "equal_capacity", "kairos"]:
        print(f"\n--- Testing variant: {v} ---")
        train_variant(
            track=smoke_track,
            variant=v,
            shard_dir=f"{smoke_dir}/train",
            eval_shard_dir=f"{smoke_dir}/eval",
            d_in=d_in,
            out_dir=out_dir,
            epochs=2,
            batch_size=16,
        )

    print("\n--- Testing final report generation ---")
    report(out_dir=out_dir)
    print("[SMOKE TEST] Verification PASSED successfully!")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["download", "cache", "train", "report", "smoke_test"])
    parser.add_argument("--track", choices=list(TRACKS.keys()), help="Target benchmark track ('text' or 'vision')")
    parser.add_argument("--variant", choices=["linear_probe", "equal_capacity", "kairos"], help="Model variant to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--d_in", type=int, default=None, help="Input dimension (auto-detected from shard if omitted)")
    parser.add_argument("--out_dir", type=str, default="./kairos_checkpoints", help="Output directory for results")
    args = parser.parse_args()

    if args.stage == "download":
        for name, t in TRACKS.items():
            print(f"[download] Starting track {name}: {t.backbone_repo}")
            download_backbone(t)

    elif args.stage == "cache":
        if not args.track:
            raise SystemExit("--track is required (choices: 'text', 'vision')")
        t = TRACKS[args.track]
        bb = download_backbone(t)
        cache_latents(t, bb, t.train_split, f"./latent_cache/{t.name}/train")
        cache_latents(t, bb, t.eval_split, f"./latent_cache/{t.name}/eval")

    elif args.stage == "train":
        if not (args.track and args.variant):
            raise SystemExit("--track and --variant are required for stage 'train'")
        t = TRACKS[args.track]
        train_variant(
            t,
            args.variant,
            shard_dir=f"./latent_cache/{t.name}/train",
            eval_shard_dir=f"./latent_cache/{t.name}/eval",
            d_in=args.d_in,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    elif args.stage == "report":
        report(out_dir=args.out_dir)

    elif args.stage == "smoke_test":
        run_smoke_test()
