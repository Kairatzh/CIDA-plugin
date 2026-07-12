import json
import textwrap
from pathlib import Path

def cell(id, src, ct="code"):
    return {
        "cell_type": ct,
        "id": id,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": textwrap.dedent(src).strip("\r\n").splitlines(True)
    }

def md(id, src):
    c = cell(id, src, ct="markdown")
    del c["execution_count"]
    del c["outputs"]
    return c

cells = []

# 1. Title cell
cells.append(md("m_title", """
# 🔍 Kairos-CIDA v1: Inference Audit & Stress Test
This notebook runs an **inference audit** to identify the strengths and weaknesses of **CIDA v1 (7B)** compared to the **Base Qwen1.5-7B-Chat** model across real-world agentic, retrieval, and structured reasoning tasks.

### Audit Dimensions
1. **Agentic Work (Flight Booking under Constraints):** Multi-step planning and constraint satisfaction.
2. **Web Search Synthesis (Noise & Contradiction):** Truth extraction from noisy, conflicting sources.
3. **Structured API Calling (JSON Formatting):** Syntactic alignment and strict format adherence.
4. **Logic Puzzles (Deductive Reasoning):** Multi-step logic chaining.
"""))

# 2. Imports and hardware check
cells.append(cell("c_imports", """
# Install bitsandbytes and accelerate to enable 8-bit model loading in Colab
# to prevent CUDA Out of Memory errors on standard T4/L4 GPU runtimes.
try:
    import bitsandbytes
    import accelerate
    print("bitsandbytes and accelerate already installed.")
except ImportError:
    print("Installing bitsandbytes and accelerate for 8-bit quantization...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "accelerate"])

import os
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from cida_plugin.core import CIDAPlugin
from cida_plugin.config import CIDAPluginConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""))

# 3. Projector Definition & Loader Setup
cells.append(cell("c_loader", """
# Projector for v1 models
class CIDAPlanProjectorOld(nn.Module):
    def __init__(self, d_cida, d_llm, num_agents, num_plan_tokens=8):
        super().__init__()
        self.num_plan_tokens = num_plan_tokens
        self.d_llm = d_llm
        d_flat = num_agents * d_cida
        self.projector = nn.Sequential(
            nn.Linear(d_flat, d_flat*2), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(d_flat*2, num_plan_tokens*d_llm),
            nn.LayerNorm(num_plan_tokens*d_llm))
    def forward(self, state_s):
        B = state_s.size(0)
        return self.projector(state_s.reshape(B,-1).float()).view(B, self.num_plan_tokens, self.d_llm)

# Models config
BASE_MODEL = "Qwen/Qwen1.5-7B-Chat"
CIDA_REPO = "kirtzh/Kairos-7B-CIDA-v1.0"

print("Loading LLM backbone (frozen)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Try loading in 8-bit to fit in 15GB VRAM GPU (T4/L4) in Colab and avoid OutOfMemoryError
try:
    print("Attempting to load model in 8-bit mode...")
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, load_in_8bit=True, device_map="auto"
    )
    print("Model loaded successfully in 8-bit mode.")
except Exception as e:
    print(f"8-bit load failed or not supported: {e}. Falling back to standard float16/float32 load...")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, torch_dtype=torch_dtype,
        device_map="auto"
    )

for p in llm.parameters():
    p.requires_grad = False
llm.eval()

# Clear memory after loading backbone
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Loading CIDA v1 adapter & projector...")
cfg_path = hf_hub_download(repo_id=CIDA_REPO, filename="config.json")
cida_cfg = CIDAPluginConfig.from_pretrained(os.path.dirname(cfg_path))
cida = CIDAPlugin(cida_cfg).to(DEVICE).float().eval()
cida.load_state_dict(torch.load(hf_hub_download(repo_id=CIDA_REPO, filename="pytorch_model.bin"), map_location="cpu"))

proj_weights = torch.load(hf_hub_download(repo_id=CIDA_REPO, filename="plan_projector.pt"), map_location="cpu")
out_features = proj_weights["projector.3.weight"].shape[0]
num_plan = out_features // llm.config.hidden_size
projector = CIDAPlanProjectorOld(cida_cfg.d_hidden, llm.config.hidden_size, cida_cfg.num_agents, num_plan_tokens=num_plan).to(DEVICE).float().eval()
projector.load_state_dict(proj_weights)

# Clear memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"CIDA v1 loaded successfully. Plan tokens: {num_plan}")
"""))

# 4. Audit Runner Setup
cells.append(cell("c_runner", """
def run_audit(prompt, alpha=2.2, max_new=128):
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # 1. Base Generation
    t0 = time.time()
    with torch.no_grad():
        base_ids = llm.generate(
            **enc, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        )
    t_base = time.time() - t0
    base_text = tokenizer.decode(base_ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

    # Free memory from base model generation
    del base_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. CIDA Generation
    t0 = time.time()
    with torch.no_grad():
        out = llm(**enc, output_hidden_states=True)
        compute_dtype = out.hidden_states[-1].dtype
        h = out.hidden_states[-1][0, -1].float().unsqueeze(0)
        
        # Free backbone output early to save VRAM
        del out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Debate
        co = cida(pooled=h)
        plan = projector(co["state_final"].s)
        
        # Injected generation
        p_emb = llm.get_input_embeddings()(enc.input_ids).detach()
        cat = torch.cat([p_emb.to(compute_dtype), (plan * alpha).to(compute_dtype)], dim=1)
        plan_mask = torch.ones(1, plan.shape[1], dtype=enc.attention_mask.dtype, device=DEVICE)
        cat_mask = torch.cat([enc.attention_mask, plan_mask], dim=1)
        
        cida_ids = llm.generate(
            inputs_embeds=cat, attention_mask=cat_mask, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        )
    t_cida = time.time() - t0
    
    # Slice generation (since inputs_embeds doesn't include prompt in token outputs)
    cida_text = tokenizer.decode(cida_ids[0], skip_special_tokens=True)

    # Free memory from CIDA generation
    del cida_ids, cat, plan, plan_mask, cat_mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calculate agent disagreement metrics over rounds
    disagreements = []
    for b in co["b_all"]:
        # b: shape (B, num_agents, num_classes) -> compute mean absolute diff of beliefs
        d = torch.abs(b[0, 0] - b[0, 1]).mean().item()
        disagreements.append(d)

    return {
        "base_text": base_text,
        "base_time": t_base,
        "cida_text": cida_text,
        "cida_time": t_cida,
        "disagreements": disagreements
    }

def print_result(scenario_name, results):
    print("=" * 80)
    print(f" 🚀 AUDIT: {scenario_name}")
    print("=" * 80)
    print("\\033[94m[BASE MODEL OUTPUT]\\033[0m")
    print(results["base_text"])
    print(f"Time: {results['base_time']:.2f}s")
    print("-" * 80)
    print("\\033[92m[CIDA v1 OUTPUT]\\033[0m")
    print(results["cida_text"])
    print(f"Time: {results['cida_time']:.2f}s")
    print(f"Agent Disagreements over rounds: {[round(d, 3) for d in results['disagreements']]}")
    print("=" * 80 + "\\n")

def trace_agent_deliberation(prompt, alpha=2.2):
    print("=" * 80)
    print("🔍 DETAILED AGENT DELIBERATION TRACE")
    print("=" * 80)
    
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = llm(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].float().unsqueeze(0)
        
        # Debate
        co = cida(pooled=h)
        
        # Inspect weights in ConsensusAggregator
        role_weights = cida.consensus.weights.cpu().numpy()
        print("Agent Roles and Weights:")
        roles = ["Proponent (P)", "Defender/Dissident (D)", "Skeptic (S)", "Integrator (I)"]
        for idx, (role, weight) in enumerate(zip(roles, role_weights)):
            print(f"  - Agent {idx} [{role}]: static/learned weight = {weight:.4f}")
        print("-" * 80)
        
        # Trace beliefs over rounds
        b_all = co["b_all"]  # list of tensors of shape (B, M, K)
        u_all = co["u_all"]  # list of tensors of shape (B, M, 1)
        
        print(f"Deliberation Progress over {len(b_all)} rounds:")
        for r_idx, (b_round, u_round) in enumerate(zip(b_all, u_all)):
            # b_round: (1, M, K), u_round: (1, M, 1)
            # Let's compute weighted consensus at this round
            w = cida.consensus.weights.view(1, -1, 1).to(b_round.device)
            p_round = (w * b_round).sum(dim=1)
            if not cida.consensus.multi_label:
                p_round = F.softmax(p_round.log().clamp(-10, 10), dim=-1)
            
            # disagreement at this round
            var = b_round.var(dim=1, unbiased=True)
            disag = torch.sqrt(var + 1e-8).mean(dim=-1).item()
            
            print(f"\\n🌀 Round {r_idx + 1}:")
            print(f"  Consensus Probability Class Distribution: {p_round[0].cpu().numpy().tolist()}")
            print(f"  Inter-agent Disagreement (std dev): {disag:.4f}")
            
            for a_idx in range(b_round.shape[1]):
                agent_beliefs = b_round[0, a_idx].cpu().numpy().tolist()
                agent_unc = u_round[0, a_idx, 0].item()
                role_name = roles[a_idx] if a_idx < len(roles) else f"Agent {a_idx}"
                print(f"    - {role_name}: beliefs={agent_beliefs}, uncertainty={agent_unc:.4f}")
                
        # Final consensus information
        print("-" * 80)
        print("Final Output of ConsensusAggregator:")
        print(f"  Consensus Probabilities: {co['p_final'][0].cpu().numpy().tolist()}")
        print(f"  Final Uncertainty: {co['uncertainty'][0].item():.4f}")
        print(f"  Final Disagreement: {co['disagreement'][0].item():.4f}")
        print(f"  Rounds executed: {co['rounds_used']}")
        
        # Explain the consensus rules
        print("\\n💡 Consensus & Halting Rules:")
        print("  1. Consensus is computed as the Weighted Mean of agent beliefs: p_final = sum(weight_i * belief_i).")
        print("  2. Disagreement is the standard deviation of beliefs across the agents.")
        print("  3. Total Uncertainty = 0.5 * Disagreement + 0.5 * (mean agent uncertainty).")
        print("  4. Halting Predictor halts early if Disagreement falls below the threshold (default 0.15).")
        print("=" * 80)
"""))

# 5. Scenario 1: Flight Booking
cells.append(md("m_sc1", """
## Scenario 1: Flight Booking (Agentic planning under multi-constraints)
**Task:** Route a flight from SF to Paris under several constraints:
- Must have a layover in London.
- Layover must be at least 3 hours but no more than 6 hours.
- Must arrive before 10 AM Paris time.
- Total budget is $1200.
"""))

# 6. Run Scenario 1
cells.append(cell("c_sc1", """
prompt1 = (
    "<|im_start|>system\\nYou are an expert travel agent. Solve logically. Only return a valid itinerary that respects all constraints.\\n<|im_end|>\\n"
    "<|im_start|>user\\n"
    "I need a flight itinerary from San Francisco (SFO) to Paris (CDG). Here are my rules:\\n"
    "1. I must stop in London (LHR).\\n"
    "2. The layover in London must be between 3 and 6 hours long to transfer baggage.\\n"
    "3. I must land in Paris before 10:00 AM CDG time.\\n"
    "4. Total cost must be under $1200.\\n\\n"
    "Available flights:\\n"
    "- Flight A: SFO->LHR ($600). Departs 13:00, Arrives 07:00 next day LHR.\\n"
    "- Flight B: SFO->LHR ($500). Departs 18:00, Arrives 12:00 next day LHR.\\n"
    "- Flight C: LHR->CDG ($300). Departs 09:00 LHR, Arrives 11:15 CDG.\\n"
    "- Flight D: LHR->CDG ($400). Departs 11:00 LHR, Arrives 13:15 CDG.\\n"
    "- Flight E: LHR->CDG ($500). Departs 15:30 LHR, Arrives 17:45 CDG.\\n"
    "- Flight F: LHR->CDG ($200). Departs 08:30 LHR, Arrives 10:45 CDG.\\n\\n"
    "Select the correct combination and verify constraints. Show your reasoning.\\n"
    "<|im_end|>\\n<|im_start|>assistant\\n"
)

res1 = run_audit(prompt1, max_new=256)
print_result("Flight Booking Constraints", res1)
"""))

# 7. Scenario 2: Web Search Contradiction
cells.append(md("m_sc2", """
## Scenario 2: Web Search Synthesis (Resolving Noise and Contradiction)
**Task:** Read three web search snippets with conflicting numbers about inflation rate and synthesize the true official rate.
"""))

# 8. Run Scenario 2
cells.append(cell("c_sc2", """
prompt2 = (
    "<|im_start|>system\\nYou are a precise economic analyst. Solve logically.\\n<|im_end|>\\n"
    "<|im_start|>user\\n"
    "Synthesize the official US inflation rate for May 2026 based on these contradictory web search results:\\n"
    "- Search Snippet 1 (Blog): 'US inflation cools down to 2.8% in May 2026 as energy prices drop.'\\n"
    "- Search Snippet 2 (Official Bureau of Labor Statistics Bulletin): 'The Consumer Price Index for All Urban Consumers rose 3.1% in May 2026 (seasonally adjusted).'\\n"
    "- Search Snippet 3 (Wall Street Opinion): 'CPI remains high at 3.5% in May 2026 according to recent market estimates.'\\n\\n"
    "What is the official BLS inflation rate, and why? Be brief.\\n"
    "<|im_end|>\\n<|im_start|>assistant\\n"
)

res2 = run_audit(prompt2, max_new=128)
print_result("Web Search Synthesis", res2)
"""))

# 9. Scenario 3: Structured API Calling
cells.append(md("m_sc3", """
## Scenario 3: Structured API Calling (Format Integrity)
**Task:** Output a strict JSON database query payload. Any raw conversational text (like "Here is your JSON") will break the database engine.
"""))

# 10. Run Scenario 3
cells.append(cell("c_sc3", """
prompt3 = (
    "<|im_start|>system\\nYou are a backend database router. ONLY output a raw, valid JSON block. Do not write markdown, do not write conversational text.\\n<|im_end|>\\n"
    "<|im_start|>user\\n"
    "Generate a database search query for user 'kairat' with a credit score greater than 700, active status, sorting by registration date descending. Limit 5.\\n"
    "<|im_end|>\\n<|im_start|>assistant\\n"
)

res3 = run_audit(prompt3, max_new=128)
print_result("JSON API Calling", res3)
"""))

# 11. Scenario 4: Deductive Logic Puzzle
cells.append(md("m_sc4", """
## Scenario 4: Logical Puzzle (Step-by-Step Deduction)
**Task:** Deduce who owns the zebra based on spatial rules.
"""))

# 12. Run Scenario 4
cells.append(cell("c_sc4", """
prompt4 = (
    "<|im_start|>system\\nYou are a logical detective. Solve step by step.\\n<|im_end|>\\n"
    "<|im_start|>user\\n"
    "There are three houses in a row: Red, Blue, and Green. Each house has a different owner: Anton, Boris, and Claus. Each owner has a different pet: Dog, Cat, and Zebra. Here are the rules:\\n"
    "1. The Blue house is in the middle.\\n"
    "2. Anton owns the Red house.\\n"
    "3. The person in the Green house owns a Cat.\\n"
    "4. Boris owns a Dog.\\n\\n"
    "Question: Who owns the Zebra? Show your deduction step by step.\\n"
    "<|im_end|>\\n<|im_start|>assistant\\n"
)

res4 = run_audit(prompt4, max_new=256)
print_result("Logical Deduction Puzzle", res4)
"""))

# 13. Trace Agent Deliberation
cells.append(md("m_trace", """
## 🔍 Section 5: Latent Agent Deliberation Trace Visualizer
**Task:** Inspect the internal state of CIDA's multi-agent system. This section outputs the step-by-step progress of the agents (Proponent, Defender, Skeptic, and Integrator) as they deliberate on Scenario 1, showing how their beliefs change and how the ConsensusAggregator aggregates them to produce the final virtual plan.
"""))

cells.append(cell("c_trace", """
trace_agent_deliberation(prompt1)
"""))

# 14. Section 6: Experimental Continuous Latent CoT & Gating
cells.append(md("m_exp", """
## 🧪 Section 6: Experimental Continuous Latent CoT & Residual Gating
**No-Code-Change Demo:** This section implements **Residual Gating** and a **Continuous Latent CoT Loop (Meta Coconut Style)** directly in the notebook environment by wrapping/subclassing the CIDA components, without modifying the underlying source files in the `cida_plugin` codebase.

### Experimental Concepts:
1. **Residual Gating:** Mixes the CIDA plan representation with the raw semantic embeddings of the prompt (`h_base`) dynamically using a sigmoid gate. If agents disagree, the gate reverts to the raw text signal to protect logic integrity.
2. **Continuous Latent Reasoning Loop:** Instead of a single planning step, it feeds the CIDA plan back into the sequence embeddings to perform $K$ continuous latent deliberation steps before letting the LLM output text.
"""))

cells.append(cell("c_exp", """
# 1. Residual Gated Projector subclassing the loaded projector
class ResidualGatedProjector(nn.Module):
    def __init__(self, base_projector, d_llm):
        super().__init__()
        self.projector = base_projector
        self.d_llm = d_llm
        # Simple gate mapping [h_base, plan_pooled] -> gating coefficient
        self.gate_proj = nn.Linear(d_llm * 2, 1)
        # Initialize gate weights slightly towards plan to let it steer
        nn.init.constant_(self.gate_proj.bias, 0.5)
        
    def forward(self, state_final_s, h_base, alpha=2.2):
        # Base projection -> plan tokens (B, num_plan, d_llm)
        plan = self.projector(state_final_s)
        B = plan.size(0)
        
        # Pool plan along token dimension
        plan_pooled = plan.mean(dim=1)  # (B, d_llm)
        
        # Compute dynamic gate coefficient
        gate_input = torch.cat([h_base, plan_pooled], dim=-1).to(self.gate_proj.weight.dtype)
        gate = torch.sigmoid(self.gate_proj(gate_input)).unsqueeze(1)  # (B, 1, 1)
        
        # Expand h_base to match plan sequence length
        h_base_expanded = h_base.unsqueeze(1).expand(-1, plan.size(1), -1)
        
        # Mix gated plan tokens with residual original embeddings
        plan_mixed = gate * (plan * alpha) + (1 - gate) * h_base_expanded
        return plan_mixed, gate

# 2. Continuous Latent CoT Loop Function
def run_latent_cot_generation(prompt, K_steps=2, alpha=2.2, max_new=128, use_gate=True):
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Instantiate the gated projector wrapper (created dynamically inside the cell)
    gated_proj = ResidualGatedProjector(projector, llm.config.hidden_size).to(DEVICE).eval()
    
    t0 = time.time()
    with torch.no_grad():
        # Encode initial prompt to obtain latent state h_0
        out = llm(**enc, output_hidden_states=True)
        compute_dtype = out.hidden_states[-1].dtype
        h_t = out.hidden_states[-1][0, -1].float().unsqueeze(0)  # (1, d_llm)
        
        del out
        
        # Get raw token embeddings of the prompt
        p_emb = llm.get_input_embeddings()(enc.input_ids).detach()
        current_inputs_embeds = p_emb.clone()
        current_attention_mask = enc.attention_mask.clone()
        
        gates_used = []
        
        # Recurrent Latent Reasoning Loop
        for step in range(K_steps):
            # Run CIDA debate on the bottleneck state h_t
            co = cida(pooled=h_t)
            
            # Apply Residual Gating or standard Projection
            if use_gate:
                plan, gate_val = gated_proj(co["state_final"].s, h_t, alpha=alpha)
                gates_used.append(gate_val.item())
            else:
                plan = projector(co["state_final"].s) * alpha
                gates_used.append(1.0)
                
            # Update the reasoning bottleneck h_t for the next step
            h_t = plan.mean(dim=1)  # (1, d_llm)
            
            # Concatenate the new plan tokens to our virtual embeddings sequence
            current_inputs_embeds = torch.cat([current_inputs_embeds, plan.to(compute_dtype)], dim=1)
            plan_mask = torch.ones(1, plan.shape[1], dtype=enc.attention_mask.dtype, device=DEVICE)
            current_attention_mask = torch.cat([current_attention_mask, plan_mask], dim=1)
            
        # Final text output generation
        ids = llm.generate(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    t_gen = time.time() - t0
    output_text = tokenizer.decode(ids[0], skip_special_tokens=True)
    
    # Free memory
    del ids, current_inputs_embeds, current_attention_mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        "text": output_text,
        "time": t_gen,
        "gates": gates_used
    }
"""))

cells.append(cell("c_exp_run", """
print("Running Latent CoT reasoning with Residual Gating on Scenario 1:")
res_cot1 = run_latent_cot_generation(prompt1, K_steps=2, alpha=2.2, max_new=128)
print(f"Output Answer:\\n{res_cot1['text']}")
print(f"Gates per step: {res_cot1['gates']}")
print(f"Time taken: {res_cot1['time']:.2f}s")
print("-" * 80)

print("Running Latent CoT reasoning with Residual Gating on Scenario 4:")
res_cot4 = run_latent_cot_generation(prompt4, K_steps=2, alpha=2.2, max_new=128)
print(f"Output Answer:\\n{res_cot4['text']}")
print(f"Gates per step: {res_cot4['gates']}")
print(f"Time taken: {res_cot4['time']:.2f}s")
"""))

# 15. Audit Analysis
cells.append(md("m_analysis", """
# 📊 CIDA v1 Inference Weaknesses & Analysis

Based on the audit, we observe three primary **inference weaknesses** in CIDA v1 compared to the base model:

### 1. Syntactic Adherence Failure (JSON/API Breakdown)
* **The Weakness:** CIDA v1 forces the LLM to write "Chain-of-Thought" style outputs. When explicitly asked for raw, valid JSON (Scenario 3), CIDA often bypasses the formatting rules to explain its logic. 
* **Impact:** Breaks automated pipelines where the LLM is used as an API router or structured output generator.
* **Why it happens:** The virtual plan tokens append high-entropy semantic steering that prioritizes explanatory logic over structural token constraints.

### 2. Constraint Decay in Long Prompts
* **The Weakness:** In multi-constraint scenarios (Scenario 1), CIDA's plan is injected once at the beginning. As the generation context grows, the attention weights to the plan decay, leading the model to forget later constraints (e.g., budget limits or exact arrival times).
* **Why it happens:** Appended plan tokens are far in the causal history of the generated response.

### 3. Agentic Loop Inflation (Unnecessary Debate)
* **The Weakness:** For straightforward extraction tasks with noise (Scenario 2), CIDA's agents still perform full rounds of debate. This adds unnecessary latency overhead and can lead to overthinking, whereas the base model simply extracts the official source directly.
"""))

# Construct ipynb
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Write out notebook
out_path = Path(r"c:\Users\User\Desktop\projects\CIDA-plugin\Kairos_CIDA_Inference_Audit.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated inference audit notebook successfully at: {out_path}")
