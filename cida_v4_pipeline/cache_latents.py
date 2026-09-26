"""
cache_latents.py — Offline Latent Cacher for CIDA v4

Solves the Compute Bottleneck:
1. Runs prompts through frozen backbone (70B / 7B) ONCE in inference mode (FP16 / 4-bit).
2. Extracts sequence hidden states (H_seq) and teacher solution representations.
3. Saves shards to NVMe disk (.pt or .safetensors) with manifest index.
4. Unloads the LLM! Subsequent CIDA training runs at 500+ samples/sec on consumer GPUs.
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class LatentCacheDataset(Dataset):
    """PyTorch Dataset that streams cached latent tensors from disk shards."""
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found in {cache_dir}. Run cache_latents.py first.")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
            
        self.samples = self.manifest["samples"]
        self.d_llm = self.manifest["d_llm"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.samples[idx]
        shard_path = os.path.join(self.cache_dir, info["shard_file"])
        data = torch.load(shard_path, map_location="cpu")
        item = data[info["shard_idx"]]
        return item


class OfflineLatentCacher:
    """
    Drives the offline caching process for large frozen backbones.
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cache_dir: str = "./latent_cache",
        load_in_4bit: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        shard_size: int = 100
    ):
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.shard_size = shard_size
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.d_llm = 4096

    def load_model(self):
        """Loads tokenizer and backbone in inference mode."""
        print(f"[LOAD] Loading frozen backbone: {self.model_name_or_path} (device: {self.device}, 4bit: {self.load_in_4bit})...")
        if not _HAS_TRANSFORMERS:
            raise RuntimeError("transformers library required for latent caching.")
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        kwargs = {"trust_remote_code": True}
        if self.load_in_4bit and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            kwargs["device_map"] = "auto"
        else:
            if torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.float16
                kwargs["device_map"] = "auto"
                
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.d_llm = self.model.config.hidden_size
        print(f"[OK] Loaded backbone. Hidden dimension: {self.d_llm}")

    def cache_dataset(self, jsonl_path: str, max_samples: Optional[int] = None) -> str:
        """
        Reads curated JSONL, extracts H_seq representations, and writes shards.
        """
        if self.model is None:
            self.load_model()
            
        print(f"🚀 Starting offline latent caching for: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if max_samples:
            lines = lines[:max_samples]
            
        manifest_samples = []
        current_shard = []
        shard_idx = 0
        
        t0 = time.time()
        for i, line in enumerate(lines):
            record = json.loads(line)
            prompt = record["prompt"]
            solution = record.get("solution", "")
            
            # Tokenize prompt
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.model.device)
            
            with torch.no_grad():
                out = self.model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                # Last hidden state: (1, L, D_llm)
                h_seq = out.hidden_states[-1].squeeze(0).cpu().to(torch.float32)
                mask = enc.attention_mask.squeeze(0).cpu()
                
                # Optional target teacher representation from solution
                target_state = None
                if solution:
                    enc_sol = self.tokenizer(
                        solution,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256
                    ).to(self.model.device)
                    out_sol = self.model(
                        input_ids=enc_sol.input_ids,
                        attention_mask=enc_sol.attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    # Mean pooled teacher plan representation: (d_llm,)
                    target_state = out_sol.hidden_states[-1].mean(dim=1).squeeze(0).cpu().to(torch.float32)

            item = {
                "id": record["id"],
                "h_seq": h_seq,
                "mask": mask,
                "target_state": target_state,
                "verifier_spec": record.get("verifier_spec", {}),
                "prompt": prompt,
                "solution": solution
            }
            
            current_shard.append(item)
            
            # Register in manifest
            shard_filename = f"shard_{shard_idx:04d}.pt"
            manifest_samples.append({
                "id": record["id"],
                "shard_file": shard_filename,
                "shard_idx": len(current_shard) - 1,
                "seq_len": h_seq.size(0)
            })
            
            # Save shard if full
            if len(current_shard) >= self.shard_size or i == len(lines) - 1:
                shard_out_path = os.path.join(self.cache_dir, shard_filename)
                torch.save(current_shard, shard_out_path)
                shard_idx += 1
                current_shard = []
                
            if (i + 1) % 25 == 0 or (i + 1) == len(lines):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"[CACHE] Cached {i + 1}/{len(lines)} samples ({rate:.1f} samples/sec)...")
                
        # Write manifest
        manifest = {
            "model_name_or_path": self.model_name_or_path,
            "d_llm": self.d_llm,
            "total_samples": len(manifest_samples),
            "samples": manifest_samples
        }
        manifest_path = os.path.join(self.cache_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            
        print(f"[SUCCESS] Offline caching complete! {len(manifest_samples)} samples saved in {self.cache_dir}")
        
        # Unload model
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[CLEANUP] Backbone model unloaded from memory. GPU is free for training CIDA!")
        return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Latent Caching for CIDA v4")
    parser.add_argument("--data", type=str, required=True, help="Path to curated JSONL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--cache_dir", type=str, default="./latent_cache")
    parser.add_argument("--4bit", action="store_true", dest="load_in_4bit")
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    
    cacher = OfflineLatentCacher(
        model_name_or_path=args.model,
        cache_dir=args.cache_dir,
        load_in_4bit=args.load_in_4bit
    )
    cacher.cache_dataset(args.data, max_samples=args.samples)
