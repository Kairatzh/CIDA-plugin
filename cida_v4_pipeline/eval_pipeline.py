"""
eval_pipeline.py — Live Side-by-Side Benchmark for CIDA v4 vs Base LLM & CoT

Evaluates:
1. Base Backbone (Direct Greedy, 0 CoT)
2. Base Backbone (Text Chain-of-Thought)
3. CIDA v4 Co-Processor (Latent KV-Injected, 0 CoT)

Reports:
- Accuracy / Pass Rate (%)
- Generated Tokens
- Latency (sec)
- Compute Savings (%)
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .core_v4 import CIDAv4Core
from .kv_steerer import PrefixKVSteerer
from .train_stage2_grpo import verify_answer

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class PipelineEvaluator:
    def __init__(
        self,
        backbone_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cida_checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.backbone_name = backbone_name_or_path
        
        print(f"[LOAD] Loading Backbone: {backbone_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        self.cida = None
        self.steerer = None
        if cida_checkpoint_path and os.path.exists(cida_checkpoint_path):
            print(f"[MODEL] Loading CIDA v4 Core: {cida_checkpoint_path}...")
            self.cida = CIDAv4Core.from_pretrained(cida_checkpoint_path, device=self.device)
            self.cida.eval()
            self.steerer = PrefixKVSteerer.from_model_config(self.backbone.config).to(self.device)

    def generate_base_direct(self, prompt: str, max_tokens: int = 200) -> Tuple[str, int, float]:
        """Direct answer without thinking prompt."""
        t0 = time.time()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.backbone.device)
        with torch.no_grad():
            gen_ids = self.backbone.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        elapsed = time.time() - t0
        new_tokens = gen_ids.size(1) - enc.input_ids.size(1)
        text = self.tokenizer.decode(gen_ids[0][enc.input_ids.size(1):], skip_special_tokens=True).strip()
        return text, new_tokens, elapsed

    def generate_base_cot(self, prompt: str, max_tokens: int = 500) -> Tuple[str, int, float]:
        """Chain-of-thought prompt."""
        cot_prompt = prompt + "\nLet's think step by step to solve this thoroughly and output the final answer."
        t0 = time.time()
        enc = self.tokenizer(cot_prompt, return_tensors="pt").to(self.backbone.device)
        with torch.no_grad():
            gen_ids = self.backbone.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        elapsed = time.time() - t0
        new_tokens = gen_ids.size(1) - enc.input_ids.size(1)
        text = self.tokenizer.decode(gen_ids[0][enc.input_ids.size(1):], skip_special_tokens=True).strip()
        return text, new_tokens, elapsed

    def generate_cida(self, prompt: str, max_tokens: int = 200) -> Tuple[str, int, float, Dict[str, Any]]:
        """CIDA v4 Co-Processor execution: plan in latent space, then direct answer."""
        t0 = time.time()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.backbone.device)
        
        with torch.no_grad():
            # 1. Extract prompt hidden states
            bb_out = self.backbone(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            h_seq = bb_out.hidden_states[-1].to(self.device).float()
            
            # 2. CIDA Deliberation & Plan Generation
            cida_out = self.cida(h_seq, sample_plan=False)
            plan = cida_out["plan_mu"] # deterministic inference
            
            # 3. Inject plan via steerer
            p_emb = self.backbone.get_input_embeddings()(enc.input_ids)
            comb_embeds, comb_mask = self.steerer.inject_inputs_embeds_fallback(
                p_emb, plan.to(p_emb.dtype), enc.attention_mask
            )
            
            # 4. Generate answer directly (0 text reasoning tokens)
            gen_ids = self.backbone.generate(
                inputs_embeds=comb_embeds,
                attention_mask=comb_mask,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
            
        elapsed = time.time() - t0
        # Tokens generated past prefix + prompt
        P = plan.size(1)
        new_tokens = gen_ids.size(1) - (enc.input_ids.size(1) + P)
        text = self.tokenizer.decode(gen_ids[0][enc.input_ids.size(1) + P:], skip_special_tokens=True).strip()
        
        telemetry = {
            "rounds_executed": cida_out["rounds_executed"],
            "uncertainty": cida_out["uncertainty"].mean().item(),
            "disagreement": cida_out["disagreement"].mean().item(),
            "rollback_count": cida_out["rollback_count"]
        }
        return text, max(1, new_tokens), elapsed, telemetry

    def run_benchmark(self, test_data_path: str, max_samples: int = 20) -> Dict[str, Any]:
        """Runs comparative benchmark on test samples."""
        with open(test_data_path, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()][:max_samples]
            
        print(f"\n[BENCHMARK] Starting Comparative Benchmark on {len(samples)} problems...")
        
        results = {
            "base_direct": {"correct": 0, "tokens": 0, "latency": 0.0},
            "base_cot": {"correct": 0, "tokens": 0, "latency": 0.0},
            "cida_v4": {"correct": 0, "tokens": 0, "latency": 0.0}
        }
        
        for i, s in enumerate(samples):
            prompt = s["prompt"]
            spec = s.get("verifier_spec", {})
            task_type = s.get("task_type", "math")
            print(f"\n[{i+1}/{len(samples)}] Task: {prompt[:60]}...")
            
            # 1. Base Direct
            txt_dir, tok_dir, lat_dir = self.generate_base_direct(prompt)
            r_dir = verify_answer(task_type, txt_dir, spec)
            results["base_direct"]["correct"] += r_dir
            results["base_direct"]["tokens"] += tok_dir
            results["base_direct"]["latency"] += lat_dir
            print(f"  • Base Direct: {'PASS' if r_dir > 0 else 'FAIL'} | {tok_dir} tokens | {lat_dir:.2f}s")
            
            # 2. Base CoT
            txt_cot, tok_cot, lat_cot = self.generate_base_cot(prompt)
            r_cot = verify_answer(task_type, txt_cot, spec)
            results["base_cot"]["correct"] += r_cot
            results["base_cot"]["tokens"] += tok_cot
            results["base_cot"]["latency"] += lat_cot
            print(f"  • Base + CoT:  {'PASS' if r_cot > 0 else 'FAIL'} | {tok_cot} tokens | {lat_cot:.2f}s")
            
            # 3. CIDA v4
            if self.cida is not None:
                txt_cida, tok_cida, lat_cida, telem = self.generate_cida(prompt)
                r_cida = verify_answer(task_type, txt_cida, spec)
                results["cida_v4"]["correct"] += r_cida
                results["cida_v4"]["tokens"] += tok_cida
                results["cida_v4"]["latency"] += lat_cida
                print(f"  • CIDA v4:     {'PASS' if r_cida > 0 else 'FAIL'} | {tok_cida} tokens | {lat_cida:.2f}s (Rounds: {telem['rounds_executed']})")
                
        N = len(samples)
        summary = {
            "num_samples": N,
            "base_direct": {
                "accuracy": f"{(results['base_direct']['correct'] / N) * 100:.1f}%",
                "avg_tokens": f"{results['base_direct']['tokens'] / N:.1f}",
                "avg_latency": f"{results['base_direct']['latency'] / N:.2f}s"
            },
            "base_cot": {
                "accuracy": f"{(results['base_cot']['correct'] / N) * 100:.1f}%",
                "avg_tokens": f"{results['base_cot']['tokens'] / N:.1f}",
                "avg_latency": f"{results['base_cot']['latency'] / N:.2f}s"
            },
            "cida_v4": {
                "accuracy": f"{(results['cida_v4']['correct'] / N) * 100:.1f}%" if self.cida else "N/A",
                "avg_tokens": f"{results['cida_v4']['tokens'] / N:.1f}" if self.cida else "N/A",
                "avg_latency": f"{results['cida_v4']['latency'] / N:.2f}s" if self.cida else "N/A"
            }
        }
        
        self.print_summary_table(summary)
        return summary

    @staticmethod
    def print_summary_table(s: Dict[str, Any]):
        print("\n" + "=" * 75)
        print("[SUMMARY] BENCHMARK RESULTS: CIDA v4 vs BASELINE")
        print("=" * 75)
        print(f"{'Method':<20} | {'Accuracy':<12} | {'Avg Tokens':<14} | {'Avg Latency':<12}")
        print("-" * 75)
        print(f"{'Base (No CoT)':<20} | {s['base_direct']['accuracy']:<12} | {s['base_direct']['avg_tokens']:<14} | {s['base_direct']['avg_latency']:<12}")
        print(f"{'Base + Text CoT':<20} | {s['base_cot']['accuracy']:<12} | {s['base_cot']['avg_tokens']:<14} | {s['base_cot']['avg_latency']:<12}")
        print(f"{'CIDA v4 (Latent)':<20} | {s['cida_v4']['accuracy']:<12} | {s['cida_v4']['avg_tokens']:<14} | {s['cida_v4']['avg_latency']:<12}")
        print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CIDA v4 vs Base LLM")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--cida_model", type=str, default=None)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    
    evaluator = PipelineEvaluator(
        backbone_name_or_path=args.backbone,
        cida_checkpoint_path=args.cida_model
    )
    evaluator.run_benchmark(args.test_data, max_samples=args.samples)
