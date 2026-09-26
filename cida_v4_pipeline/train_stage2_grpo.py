"""
train_stage2_grpo.py — Stage 2 Latent GRPO with Rule-Based Verification

Reinforcement Learning on Latent Plan Trajectories:
1. Samples G candidate plans from CIDA's Variational Projector: z_1, ..., z_G ~ N(mu, sigma)
2. Injects plans into frozen LLM KV-cache and generates answers
3. Evaluates answers with strict Rule-Based Verifiers (SymPy for Math, unit test runner for Code)
4. Computes normalized group relative advantages: A_i = (R_i - mean(R)) / (std(R) + eps)
5. Updates CIDA Core parameters via clipped surrogate objective with KL penalty against Stage 1 reference model.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .core_v4 import CIDAv4Core, CIDAv4Config
from .kv_steerer import PrefixKVSteerer

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False


# ─── RULE-BASED VERIFIERS ───────────────────────────────────────────────────────

class MathRuleVerifier:
    """Verifies math solutions using SymPy symbolic equivalence and numerical normalization."""
    @staticmethod
    def clean_str(s: str) -> str:
        s = s.strip().replace(" ", "").replace("$", "").replace("\\", "")
        return s

    @classmethod
    def verify(cls, generated_text: str, target_answer: str) -> float:
        if not target_answer:
            return 0.0
            
        # 1. Direct string match in generated text
        clean_target = cls.clean_str(target_answer)
        if clean_target in cls.clean_str(generated_text):
            return 1.0
            
        # 2. Extract boxed or terminal numbers
        import re
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", generated_text)
        candidate = boxed[-1].strip() if boxed else None
        
        if not candidate:
            nums = re.findall(r"[-+]?\d*\.?\d+", generated_text)
            candidate = nums[-1] if nums else None
            
        if not candidate:
            return 0.0
            
        if cls.clean_str(candidate) == clean_target:
            return 1.0
            
        # 3. SymPy symbolic check
        if _HAS_SYMPY:
            try:
                sym_cand = sympy.sympify(candidate)
                sym_target = sympy.sympify(target_answer)
                if sympy.simplify(sym_cand - sym_target) == 0:
                    return 1.0
            except Exception:
                pass
                
        return 0.0


class CodeUnitTester:
    """Executes Python code solutions against unit test cases in a subprocess sandbox."""
    @classmethod
    def verify(cls, generated_code: str, test_cases: List[Dict[str, str]], timeout: float = 2.0) -> float:
        if not test_cases:
            return 1.0 if "def " in generated_code else 0.0
            
        # Extract python code block if markdown formatting is present
        import re
        code_blocks = re.findall(r"```python(.*?)```", generated_code, re.DOTALL)
        code = code_blocks[0] if code_blocks else generated_code
        
        # Build harness
        harness = [code, "\n# Auto-generated verification harness:"]
        for i, tc in enumerate(test_cases):
            inp = tc["in"]
            expected = tc["out"]
            harness.append(f"try:\n    assert str(eval('{inp}')) == '{expected}'\nexcept Exception as e:\n    sys.exit(1)")
            
        harness_code = "\n".join(harness)
        
        try:
            res = subprocess.run(
                [sys.executable, "-c", harness_code],
                capture_output=True,
                timeout=timeout
            )
            return 1.0 if res.returncode == 0 else 0.0
        except Exception:
            return 0.0


def verify_answer(task_type: str, generated_text: str, spec: Dict[str, Any]) -> float:
    """Master rule-based verifier router."""
    if task_type == "math":
        target = spec.get("target_answer", "")
        return MathRuleVerifier.verify(generated_text, target)
    elif task_type == "code":
        test_cases = spec.get("test_cases", [])
        return CodeUnitTester.verify(generated_text, test_cases)
    else:
        target = spec.get("target_answer", "")
        return 1.0 if target.strip() in generated_text else 0.0


# ─── LATENT GRPO TRAINER ───────────────────────────────────────────────────────

class LatentGRPOTrainer:
    def __init__(
        self,
        cida_model: CIDAv4Core,
        ref_model: CIDAv4Core,
        backbone_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        beta_kl: float = 0.02,
        lr: float = 5e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.cida = cida_model.to(device)
        self.ref_cida = ref_model.to(device)
        self.ref_cida.eval()
        for p in self.ref_cida.parameters():
            p.requires_grad = False
            
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.beta_kl = beta_kl
        self.device = device
        
        # Optimizer for CIDA Core
        self.optimizer = torch.optim.AdamW(self.cida.parameters(), lr=lr, weight_decay=0.01)
        
        # Load backbone
        print(f"[LOAD] Loading frozen backbone for GRPO generation: {backbone_name_or_path}...")
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
            
        self.steerer = PrefixKVSteerer.from_model_config(self.backbone.config).to(device)

    def compute_log_prob(self, z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Computes log-probability of sample z under Gaussian N(mu, exp(log_var)):
        log p(z) = -0.5 * sum [ log(2*pi) + log_var + (z - mu)^2 / exp(log_var) ]
        """
        var = torch.exp(log_var)
        log_prob = -0.5 * torch.sum(
            log_var + (z - mu).pow(2) / var + 1.8378770664093453, # log(2*pi)
            dim=(-1, -2)
        )
        return log_prob # (B,)

    def train_step(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """Single GRPO optimization step on one prompt."""
        self.cida.train()
        prompt = sample["prompt"]
        spec = sample.get("verifier_spec", {})
        task_type = sample.get("task_type", "math")
        
        # 1. Encode prompt through backbone
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.backbone.device)
        with torch.no_grad():
            bb_out = self.backbone(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            h_seq = bb_out.hidden_states[-1].to(self.device).float() # (1, L, D_llm)
            
        # 2. Get CIDA policy parameters and sample G candidate plans
        cida_out = self.cida(h_seq, sample_plan=False)
        mu = cida_out["plan_mu"] # (1, P, D_llm)
        log_var = cida_out["plan_logvar"] # (1, P, D_llm)
        std = torch.exp(0.5 * log_var)
        
        with torch.no_grad():
            ref_out = self.ref_cida(h_seq, sample_plan=False)
            ref_mu = ref_out["plan_mu"]
            ref_logvar = ref_out["plan_logvar"]
            
        # Sample G plans
        G = self.group_size
        sampled_plans = []
        old_log_probs = []
        
        for g in range(G):
            eps = torch.randn_like(std)
            z_g = mu + eps * std # (1, P, D_llm)
            log_p_old = self.compute_log_prob(z_g, mu.detach(), log_var.detach())
            sampled_plans.append(z_g)
            old_log_probs.append(log_p_old)
            
        # 3. Generate answers from frozen LLM and score rewards
        rewards = []
        generated_texts = []
        
        p_emb = self.backbone.get_input_embeddings()(enc.input_ids)
        
        for g in range(G):
            z_g = sampled_plans[g].to(p_emb.dtype)
            # Inject plan via prefix embeddings fallback or KV cache
            comb_embeds, comb_mask = self.steerer.inject_inputs_embeds_fallback(p_emb, z_g, enc.attention_mask)
            
            with torch.no_grad():
                gen_ids = self.backbone.generate(
                    inputs_embeds=comb_embeds,
                    attention_mask=comb_mask,
                    max_new_tokens=150,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False # greedy after plan steering
                )
                
            ans_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            # Remove prompt from answer
            if ans_text.startswith(prompt):
                ans_text = ans_text[len(prompt):].strip()
                
            r = verify_answer(task_type, ans_text, spec)
            rewards.append(r)
            generated_texts.append(ans_text)
            
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std()
        
        # 4. Compute group relative advantages
        if std_r > 1e-4:
            advantages = (rewards_tensor - mean_r) / (std_r + 1e-6)
        else:
            advantages = rewards_tensor - mean_r # all zero or identical
            
        # 5. Policy gradient update on CIDA Core
        self.optimizer.zero_grad()
        
        # Forward pass with gradient tracking
        curr_out = self.cida(h_seq, sample_plan=False)
        curr_mu = curr_out["plan_mu"]
        curr_logvar = curr_out["plan_logvar"]
        
        surrogate_loss = 0.0
        for g in range(G):
            z_g = sampled_plans[g]
            curr_lp = self.compute_log_prob(z_g, curr_mu, curr_logvar)
            ratio = torch.exp(curr_lp - old_log_probs[g])
            adv = advantages[g]
            
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv
            surrogate_loss = surrogate_loss - torch.min(surr1, surr2).mean()
            
        surrogate_loss = surrogate_loss / G
        
        # KL Divergence penalty against reference Stage 1 policy
        kl_loss = self.cida.projector.kl_divergence(curr_mu, curr_logvar, ref_mu, ref_logvar)
        total_loss = surrogate_loss + self.beta_kl * kl_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cida.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "mean_reward": mean_r.item(),
            "kl": kl_loss.item(),
            "best_answer": generated_texts[torch.argmax(rewards_tensor).item()]
        }


def run_grpo_training(
    stage1_checkpoint: str,
    data_path: str,
    backbone_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./checkpoints_stage2_grpo",
    steps: int = 50,
    group_size: int = 4,
    lr: float = 5e-5
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[START] Initializing Stage 2 Latent GRPO on: {data_path}")
    
    # Load Stage 1 models (active and reference)
    cida_model = CIDAv4Core.from_pretrained(stage1_checkpoint, device=device)
    ref_model = CIDAv4Core.from_pretrained(stage1_checkpoint, device=device)
    
    trainer = LatentGRPOTrainer(
        cida_model=cida_model,
        ref_model=ref_model,
        backbone_name_or_path=backbone_model,
        group_size=group_size,
        lr=lr,
        device=device
    )
    
    with open(data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
        
    print(f"[DATA] Loaded {len(samples)} training prompts. Starting GRPO loop...\n")
    
    for step in range(1, steps + 1):
        sample = samples[(step - 1) % len(samples)]
        t0 = time.time()
        metrics = trainer.train_step(sample)
        elapsed = time.time() - t0
        
        if step % 5 == 0 or step == 1:
            print(f"Step {step:03d}/{steps:03d} | Reward: {metrics['mean_reward']:.2f} | Loss: {metrics['loss']:.4f} | KL: {metrics['kl']:.4f} | Time: {elapsed:.2f}s")
            
    final_path = os.path.join(output_dir, "cida_v4_grpo_final")
    cida_model.save_pretrained(final_path)
    print(f"\n[DONE] Stage 2 Latent GRPO Complete! Final model saved to: {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Latent GRPO for CIDA v4")
    parser.add_argument("--stage1_model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--group_size", type=int, default=4)
    args = parser.parse_args()
    
    run_grpo_training(
        stage1_checkpoint=args.stage1_model,
        data_path=args.data,
        backbone_model=args.backbone,
        steps=args.steps,
        group_size=args.group_size
    )
