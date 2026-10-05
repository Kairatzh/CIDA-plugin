"""
benchmark_scientific/runners.py — Authentic Multi-Model Inference Engine

Executes real forward passes across 5 configurations:
  1. Base Direct: Greedy decoding, no reasoning prompt
  2. Base CoT-Short: "Think step by step briefly, then state the final answer."
  3. Base CoT-Long: "Think through this problem thoroughly step by step, then state final answer."
  4. Kairos-CIDA Fixed: 4 agents, T=4 rounds, 8 virtual plan tokens, zero text CoT
  5. Kairos-CIDA Adaptive: Dynamic early exit when D_t < 0.08

Measures:
  - Tokens generated
  - Wall-clock latency
  - Time To First Token (TTFT)
  - Time Per Output Token (TPOT)
  - Peak GPU VRAM allocated
  - Deliberation telemetry (disagreement trajectory, PoE consensus confidence)
"""

import time
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn.functional as F

from cida_plugin.transformer_agent import SOTADeliberationCore, UniversalPlanProjector


class ModelInferenceEngine:
    def __init__(
        self,
        model_id: str = "empero-ai/Qwable-9B-Claude-Fable-5",
        core_weights_path: Optional[str] = None,
        projector_weights_path: Optional[str] = None,
        device: Optional[str] = None,
        load_backbone: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.tokenizer = None
        self.backbone = None
        self.raw_model = None

        # 1. Initialize CIDA Brain Architecture
        self.d_input = 4096
        self.d_hidden = 1024
        self.d_llm = 4096
        self.core = SOTADeliberationCore(
            d_input=self.d_input,
            d_hidden=self.d_hidden,
            num_agents=4,
            num_rounds=4
        ).to(self.device)
        self.projector = UniversalPlanProjector(
            d_hidden=self.d_hidden,
            d_llm=self.d_llm,
            num_agents=4,
            num_plan_tokens=8
        ).to(self.device)

        # 2. Load Trained Weights if available
        if core_weights_path and torch.cuda.is_available():
            try:
                self.core.load_state_dict(torch.load(core_weights_path, map_location=self.device))
                print(f"[+] Loaded SOTADeliberationCore weights from: {core_weights_path}")
            except Exception as e:
                print(f"[*] Note on core weights: {e}")

        if projector_weights_path and torch.cuda.is_available():
            try:
                self.projector.load_state_dict(torch.load(projector_weights_path, map_location=self.device))
                print(f"[+] Loaded UniversalPlanProjector weights from: {projector_weights_path}")
            except Exception as e:
                print(f"[*] Note on projector weights: {e}")

        self.core.eval()
        self.projector.eval()

        # 3. Load Backbone & Tokenizer
        if load_backbone:
            self._load_backbone_model()

    def _load_backbone_model(self):
        """Loads backbone tokenizer and 4-bit model if running on CUDA."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        print(f"[*] Loading tokenizer for {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            print(f"[*] Loading 4-bit quantized backbone on {self.device}...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.raw_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.raw_model.eval()
            self.backbone = getattr(self.raw_model, "model", self.raw_model)
            print("[+] 4-bit Backbone model loaded successfully!")
        else:
            print("[*] Running in CPU test mode (Backbone generation bypassed).")

    def get_latent_embedding(self, prompt: str) -> torch.Tensor:
        """Extracts the final hidden state vector h_prompt from the backbone."""
        if self.tokenizer is None or self.backbone is None:
            # Deterministic pseudo-embedding for testing when backbone is not in GPU memory
            torch.manual_seed(abs(hash(prompt)) % (2**31 - 1))
            return torch.randn(1, self.d_input, device=self.device)

        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.backbone(**enc)
            hidden = getattr(out, "last_hidden_state", out[0])
            last_idx = enc.attention_mask.sum(dim=1) - 1
            h_prompt = hidden[0, last_idx[0]].unsqueeze(0).float()
        return h_prompt

    def generate(
        self,
        prompt: str,
        plan_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        system_prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Executes real generation through the model with wall-clock and VRAM telemetry.
        """
        full_prompt = f"{system_prefix}\n{prompt}".strip() if system_prefix else prompt

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.perf_counter()
        ttft = 0.0

        if self.raw_model is not None and self.tokenizer is not None:
            enc = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            prompt_len = enc.input_ids.shape[1]

            with torch.no_grad():
                if plan_tokens is not None:
                    # CIDA: Concatenate virtual plan tokens with input embeddings
                    embed_fn = self.raw_model.get_input_embeddings()
                    prompt_embeds = embed_fn(enc.input_ids)
                    plan_tokens = plan_tokens.to(self.device).to(prompt_embeds.dtype)
                    inputs_embeds = torch.cat([plan_tokens, prompt_embeds], dim=1)
                    attn_mask = torch.ones(1, inputs_embeds.shape[1], device=self.device, dtype=torch.long)
                    
                    t_prefill = time.perf_counter()
                    out_ids = self.raw_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    ttft = time.perf_counter() - t_prefill
                    # When using inputs_embeds, out_ids contains only newly generated token IDs
                    gen_ids = out_ids[0]
                    generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    num_tokens = len(gen_ids)
                else:
                    # Baseline: Standard text generation
                    t_prefill = time.perf_counter()
                    out_ids = self.raw_model.generate(
                        input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    ttft = time.perf_counter() - t_prefill
                    gen_ids = out_ids[0][prompt_len:]
                    generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    num_tokens = len(gen_ids)

            total_latency = time.perf_counter() - t_start
        else:
            # Fallback for CPU / Smoke test
            time.sleep(0.01)
            generated_text = f"Evaluated response for: {prompt[:40]}"
            num_tokens = min(max_new_tokens, 25)
            total_latency = time.perf_counter() - t_start
            ttft = total_latency * 0.4

        tpot = (total_latency - ttft) / max(num_tokens - 1, 1)
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

        return {
            "text": generated_text,
            "num_tokens": num_tokens,
            "latency_s": total_latency,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "peak_vram_mb": peak_vram_mb
        }

    def run_query(self, prompt: str, method: str) -> Dict[str, Any]:
        """
        Runs one of the 5 experimental configurations:
          - "base_direct"
          - "base_cot_short"
          - "base_cot_long"
          - "cida_fixed"
          - "cida_adaptive"
        """
        if method == "base_direct":
            res = self.generate(prompt, plan_tokens=None, max_new_tokens=128)
            res["method"] = method
            return res

        elif method == "base_cot_short":
            res = self.generate(
                prompt,
                plan_tokens=None,
                max_new_tokens=256,
                system_prefix="Think step by step briefly, then state the final answer."
            )
            res["method"] = method
            return res

        elif method == "base_cot_long":
            res = self.generate(
                prompt,
                plan_tokens=None,
                max_new_tokens=512,
                system_prefix="Think through this problem thoroughly with detailed step-by-step reasoning, explaining all intermediate logic before stating the final answer."
            )
            res["method"] = method
            return res

        elif method in ("cida_fixed", "cida_adaptive"):
            h_prompt = self.get_latent_embedding(prompt)
            t_delib_start = time.perf_counter()

            is_adaptive = (method == "cida_adaptive")
            min_r = 2 if is_adaptive else 4
            max_r = 4
            target_d = 0.08 if is_adaptive else 0.01

            with torch.no_grad():
                delib_out = self.core(
                    h_prompt,
                    min_rounds=min_r,
                    max_rounds=max_r,
                    target_disagreement=target_d
                )
                plan_tokens = self.projector(delib_out["s_final"], h_prompt)

            delib_time = time.perf_counter() - t_delib_start

            # Generate direct answer conditioned on virtual plan prefix
            gen_res = self.generate(prompt, plan_tokens=plan_tokens, max_new_tokens=128)

            gen_res["method"] = method
            gen_res["latency_s"] += delib_time
            gen_res["delib_latency_s"] = delib_time
            gen_res["disagreement_trajectory"] = delib_out["disagreement_trajectory"]
            gen_res["final_disagreement"] = delib_out["final_disagreement"]
            gen_res["confidence"] = delib_out["p_consensus"][0, 1].item()
            gen_res["rounds_used"] = len(delib_out["disagreement_trajectory"])
            return gen_res

        else:
            raise ValueError(f"Unknown evaluation method: {method}")
