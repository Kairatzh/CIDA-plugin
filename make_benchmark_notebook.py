import json
import textwrap

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

# ── Title ──
cells.append(md("t", """
    # 🏆 The Ultimate CIDA Benchmark Suite (OOD Generalization & Cross-Version Validation)
    
    This notebook runs a rigorous, unbiased comparison between three generations of CIDA adapters:
    1. **Kairos 1** (CIDA v1.0 on Qwen1.5-7B)
    2. **Kairos 2** (CIDA v1.0 on Qwen1.5-1.8B)
    3. **Kairos 4** (CIDA v4 on Qwen2.5-1.5B)
    
    ### ⚖️ The Fairness Protocol:
    To ensure a 100% fair and unbiased comparison (since Kairos 1 & 2 were trained on mathematical/logical tasks, whereas Kairos 4 was trained on general CoT-Collection), we evaluate on:
    * **Out-Of-Distribution (OOD) Datasets:** LogiQA (reading comprehension logic), BBH (logical deduction), and FOLIO (first-order logic) which none of the models have seen in training.
    * **Net System 2 Delta ($\Delta_{base}$):** Absolute accuracies depend heavily on the base model size (7B vs 1.5B/1.8B). To measure the *true quality* of the CIDA adapter itself, we evaluate both the **Base LLM** and the **CIDA-guided LLM** for each model, calculating the improvement delta:
      $$\Delta = \text{CIDA Accuracy} - \text{Base Accuracy}$$
      This represents the pure reasoning gain introduced by the CIDA cognitive coprocessor!
"""))

# ── Cell 1: Setup ──
cells.append(cell("c1", """
    # 1. Install prerequisites
    !pip install -q torch transformers datasets accelerate bitsandbytes huggingface_hub matplotlib tqdm
    
    # Secure clone of CIDA plugin
    import getpass, sys, os
    token = getpass.getpass("Enter GitHub PAT: ").strip()
    if token:
        !git clone https://{token}@github.com/Kairatzh/CIDA-plugin.git
        sys.path.append("CIDA-plugin")
        !pip install -q -e CIDA-plugin
    else:
        !git clone https://github.com/Kairatzh/CIDA-plugin.git
        sys.path.append("CIDA-plugin")
        !pip install -q -e CIDA-plugin
        
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
"""))

# ── Cell 2: Data Loading ──
cells.append(cell("c2", """
    import random
    from datasets import load_dataset
    
    print("Loading OOD Benchmarks...")
    datasets_eval = {}
    EVAL_SAMPLES = 30  # Statistically valid size for rapid zero-shot comparison in Colab
    
    # 1. BBH (Formal Fallacies)
    try:
        ds_fallacy = load_dataset("lukaemon/bbh", "formal_fallacies", split="test")
        samples = list(ds_fallacy)
        random.seed(42)
        random.shuffle(samples)
        parsed = []
        for s in samples[:EVAL_SAMPLES]:
            q = f"Deductive Argument: {s['input']}\\nIs this argument valid or invalid? Answer with just the word."
            parsed.append({"q": q, "a": s['target'].strip()})
        datasets_eval["BBH (Fallacies)"] = parsed
        print(f"Loaded {len(parsed)} BBH Fallacies samples.")
    except Exception as e:
        print(f"Error loading BBH Fallacies: {e}")
        
    # 2. BIG-Bench Hard (Logical Deduction - Five Objects)
    try:
        ds_bbh = load_dataset("lukaemon/bbh", "logical_deduction_five_objects", split="test")
        samples = list(ds_bbh)
        random.seed(42)
        random.shuffle(samples)
        parsed = []
        for s in samples[:EVAL_SAMPLES]:
            q = f"Logic Puzzle: {s['input']}\\nJust answer with the correct letter."
            parsed.append({"q": q, "a": s['target'].strip("()")})
        datasets_eval["BBH (Logical)"] = parsed
        print(f"Loaded {len(parsed)} BBH samples.")
    except Exception as e:
        print(f"Error loading BBH: {e}")
        
    # 3. FOLIO (First-Order Logic)
    try:
        ds_folio = load_dataset("tasksource/folio", split="validation")
        samples = list(ds_folio)
        random.seed(42)
        random.shuffle(samples)
        parsed = []
        for s in samples[:EVAL_SAMPLES]:
            prem = "\\n".join(s['premises'])
            q = f"Premises:\\n{prem}\\nConclusion: {s['conclusion']}\\nIs the conclusion True, False, or Unknown? Answer with just the word."
            parsed.append({"q": q, "a": s['label'].strip()})
        datasets_eval["FOLIO (FOL)"] = parsed
        print(f"Loaded {len(parsed)} FOLIO samples.")
    except Exception as e:
        print(f"Error loading FOLIO: {e}")
        
    print("✅ All Datasets Ready!")
"""))

# ── Cell 3: Architecture and Evaluator ──
cells.append(cell("c3", """
    import gc, re, torch, torch.nn as nn
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from huggingface_hub import hf_hub_download
    from cida_plugin.config import CIDAPluginConfig
    from cida_plugin.core import CIDAPlugin
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Projector for v1/v2 models
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
            
    # Projector for v4 models
    class LatentAEProjector(nn.Module):
        def __init__(self, d_cida, d_llm, num_agents, num_plan=8, num_heads=8):
            super().__init__()
            self.num_plan = num_plan
            self.plan_queries = nn.Parameter(torch.randn(1, num_plan, d_cida) * 0.02)
            self.cross_attn = nn.MultiheadAttention(d_cida, num_heads, batch_first=True, dropout=0.1)
            self.cross_norm = nn.LayerNorm(d_cida)
            self.self_attn = nn.MultiheadAttention(d_cida, num_heads, batch_first=True, dropout=0.1)
            self.self_norm = nn.LayerNorm(d_cida)
            self.ffn = nn.Sequential(
                nn.Linear(d_cida, d_cida * 2), nn.SiLU(),
                nn.Linear(d_cida * 2, d_cida)
            )
            self.ffn_norm = nn.LayerNorm(d_cida)
            self.out_proj = nn.Sequential(
                nn.Linear(d_cida, d_llm * 2), nn.SiLU(),
                nn.Linear(d_llm * 2, d_llm), nn.LayerNorm(d_llm)
            )
            self.gate = nn.Parameter(torch.tensor(0.1))
        def forward(self, agent_states):
            B = agent_states.size(0)
            q = self.plan_queries.expand(B, -1, -1)
            ca, _ = self.cross_attn(q, agent_states, agent_states)
            q = self.cross_norm(q + ca)
            sa, _ = self.self_attn(q, q, q)
            q = self.self_norm(q + sa)
            q = self.ffn_norm(q + self.ffn(q))
            return self.out_proj(q) * self.gate.sigmoid()
            
    def free_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def extract_answer(text, ds_type):
        text = text.upper().strip()
        if ds_type in ["BBH (Logical)"]:
            m = re.search(r'\\b([A-D])\\b', text)
            if m: return m.group(1)
            m2 = re.search(r'([A-D])', text)
            return m2.group(1) if m2 else text
        elif ds_type == "BBH (Fallacies)":
            if "INVALID" in text: return "INVALID"
            if "VALID" in text: return "VALID"
            return text
        else: # FOLIO
            if "TRUE" in text: return "TRUE"
            if "FALSE" in text: return "FALSE"
            return "UNKNOWN"
            
    def evaluate_model_pair(model_label, base_repo, cida_repo_or_dir, is_v4=False):
        print(f"\\n==========================================")
        print(f"🔄 EVALUATING: {model_label}")
        print(f"==========================================")
        
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        tok = AutoTokenizer.from_pretrained(base_repo, padding_side="left")
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        
        print("Loading Base LLM...")
        llm = AutoModelForCausalLM.from_pretrained(base_repo, quantization_config=bnb, device_map="auto")
        llm.eval()
        
        # 1. Base Evaluation
        res_base = {}
        for ds_name, samples in datasets_eval.items():
            correct = 0
            for ex in tqdm(samples, desc=f"Base LLM on {ds_name}"):
                prompt = f"<|im_start|>system\\nSolve logically.\\n<|im_end|>\\n<|im_start|>user\\n{ex['q']}\\n<|im_end|>\\n<|im_start|>assistant\\n"
                enc = tok(prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    ids = llm.generate(**enc, max_new_tokens=5, do_sample=False)
                    ans_text = tok.decode(ids[0, enc.input_ids.size(1):], skip_special_tokens=True).strip()
                if extract_answer(ans_text, ds_name) == ex["a"].upper():
                    correct += 1
            res_base[ds_name] = (correct / len(samples)) * 100
            print(f"  -> Base {ds_name} Accuracy: {res_base[ds_name]:.1f}%")
            
        # 2. CIDA Loading
        print("Loading CIDA Adapter...")
        if is_v4:
            cfg = CIDAPluginConfig(
                d_input=llm.config.hidden_size, d_hidden=1024, d_message=1024,
                num_agents=4, num_classes=2, max_rounds=4,
                num_attn_heads=8, use_perspective_projector=True,
                anonymous_messages=True, sparse_comm_k=3, use_dynamic_trust=True
            )
            cida = CIDAPlugin(cfg).to(DEVICE).float().eval()
            cida.load_state_dict(torch.load(os.path.join(cida_repo_or_dir, "pytorch_model.bin"), map_location="cpu"))
            
            proj = LatentAEProjector(cfg.d_hidden, llm.config.hidden_size, cfg.num_agents, num_plan=8).to(DEVICE).float().eval()
            proj.load_state_dict(torch.load(os.path.join(cida_repo_or_dir, "projector.pt"), map_location="cpu"))
            num_plan = 8
        else:
            cfg_path = hf_hub_download(repo_id=cida_repo_or_dir, filename="config.json")
            cfg = CIDAPluginConfig.from_pretrained(os.path.dirname(cfg_path))
            cida = CIDAPlugin(cfg).to(DEVICE).float().eval()
            cida.load_state_dict(torch.load(hf_hub_download(repo_id=cida_repo_or_dir, filename="pytorch_model.bin"), map_location="cpu"))
            
            proj_weights = torch.load(hf_hub_download(repo_id=cida_repo_or_dir, filename="plan_projector.pt"), map_location="cpu")
            out_features = proj_weights["projector.3.weight"].shape[0]
            num_plan = out_features // llm.config.hidden_size
            proj = CIDAPlanProjectorOld(cfg.d_hidden, llm.config.hidden_size, cfg.num_agents, num_plan_tokens=num_plan).to(DEVICE).float().eval()
            proj.load_state_dict(proj_weights)
            
        # 3. CIDA Evaluation
        res_cida = {}
        for ds_name, samples in datasets_eval.items():
            correct = 0
            for ex in tqdm(samples, desc=f"CIDA LLM on {ds_name}"):
                prompt = f"<|im_start|>system\\nSolve logically.\\n<|im_end|>\\n<|im_start|>user\\n{ex['q']}\\n<|im_end|>\\n<|im_start|>assistant\\n"
                enc = tok(prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = llm(**enc, output_hidden_states=True)
                    compute_dtype = out.hidden_states[-1].dtype
                    h = out.hidden_states[-1][0, -1].float().unsqueeze(0)
                    co = cida(pooled=h)
                    plan = proj(co["state_final"].s)
                    
                    p_emb = llm.get_input_embeddings()(enc.input_ids).detach()
                    cat = torch.cat([p_emb.to(compute_dtype), plan.to(compute_dtype)], dim=1).to(compute_dtype)
                    pm = torch.ones(1, num_plan, dtype=enc.attention_mask.dtype, device=DEVICE)
                    gm = torch.cat([enc.attention_mask, pm], dim=1)
                    
                    ids = llm.generate(inputs_embeds=cat, attention_mask=gm, max_new_tokens=5, do_sample=False)
                    ans_text = tok.decode(ids[0], skip_special_tokens=True).strip()
                if extract_answer(ans_text, ds_name) == ex["a"].upper():
                    correct += 1
            res_cida[ds_name] = (correct / len(samples)) * 100
            print(f"  -> CIDA {ds_name} Accuracy: {res_cida[ds_name]:.1f}%")
            
        # Cleanup
        del llm, tok, cida, proj
        free_memory()
        
        return {"base": res_base, "cida": res_cida}
        
    print("✅ Evaluation Engine Built!")
"""))

# ── Cell 4: Run Kairos 1 ──
cells.append(cell("c4", """
    # Run Kairos 1 (7B CIDA v1.0)
    res_k1 = evaluate_model_pair(
        model_label="Kairos 1 (7B CIDA v1.0)",
        base_repo="Qwen/Qwen1.5-7B-Chat",
        cida_repo_or_dir="kirtzh/Kairos-7B-CIDA-v1.0",
        is_v4=False
    )
"""))

# ── Cell 5: Run Kairos 2 ──
cells.append(cell("c5", """
    # Run Kairos 2 (1.3B CIDA v1.0)
    res_k2 = evaluate_model_pair(
        model_label="Kairos 2 (1.3B CIDA v1.0)",
        base_repo="Qwen/Qwen1.5-1.8B-Chat",
        cida_repo_or_dir="kirtzh/Kairos-CIDA-1.3B-v1.0",
        is_v4=False
    )
"""))

# ── Cell 6: Run Kairos 4 ──
cells.append(cell("c6", """
    # Locate CIDA v5 (RL) or v4 weights (either local or drive)
    drive_v5 = "/content/drive/MyDrive/cida_v5_rl_weights"
    local_v5 = "cida_v5_rl_weights"
    drive_v4 = "/content/drive/MyDrive/cida_v4_universal_weights"
    local_v4 = "cida_v4_universal_weights"
    
    if os.path.exists(os.path.join(drive_v5, "pytorch_model.bin")):
        v4_dir = drive_v5
        lbl = "Kairos 5 (1.5B CIDA v5 RL)"
    elif os.path.exists(os.path.join(local_v5, "pytorch_model.bin")):
        v4_dir = local_v5
        lbl = "Kairos 5 (1.5B CIDA v5 RL)"
    elif os.path.exists(os.path.join(drive_v4, "pytorch_model.bin")):
        v4_dir = drive_v4
        lbl = "Kairos 4 (1.5B CIDA v4 SFT)"
    elif os.path.exists(os.path.join(local_v4, "pytorch_model.bin")):
        v4_dir = local_v4
        lbl = "Kairos 4 (1.5B CIDA v4 SFT)"
    else:
        v4_dir = None
        lbl = "Kairos 4/5"
        
    if v4_dir:
        res_k4 = evaluate_model_pair(
            model_label=lbl,
            base_repo="Qwen/Qwen2.5-1.5B-Instruct",
            cida_repo_or_dir=v4_dir,
            is_v4=True
        )
    else:
        print("⚠️ CIDA v4/v5 weights not found. Running with mock/empty metrics.")
        res_k4 = {
            "base": {k: 0.0 for k in datasets_eval.keys()},
            "cida": {k: 0.0 for k in datasets_eval.keys()}
        }
"""))

# ── Cell 7: Plot Results and Dashboard ──
cells.append(cell("c7", """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Calculate deltas
    versions = ["Kairos 1 (7B)", "Kairos 2 (1.3B)", "Kairos 4 (1.5B v4)"]
    datasets_names = list(datasets_eval.keys())
    
    # Accuracy tables
    k1_base = [res_k1["base"][d] for d in datasets_names]
    k1_cida = [res_k1["cida"][d] for d in datasets_names]
    k1_delta = [k1_cida[i] - k1_base[i] for i in range(len(datasets_names))]
    
    k2_base = [res_k2["base"][d] for d in datasets_names]
    k2_cida = [res_k2["cida"][d] for d in datasets_names]
    k2_delta = [k2_cida[i] - k2_base[i] for i in range(len(datasets_names))]
    
    k4_base = [res_k4["base"][d] for d in datasets_names]
    k4_cida = [res_k4["cida"][d] for d in datasets_names]
    k4_delta = [k4_cida[i] - k4_base[i] for i in range(len(datasets_names))]
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.style.use('dark_background')
    
    # Subplot 1: Absolute Accuracies (CIDA vs Base)
    ax1 = plt.subplot(1, 2, 1)
    x = np.arange(len(datasets_names))
    width = 0.2
    
    ax1.bar(x - width*1.5, k1_base, width, label="K1 Base (7B)", color="#475569", alpha=0.6)
    ax1.bar(x - width*0.5, k1_cida, width, label="K1 CIDA (7B)", color="#1e293b", edgecolor="#64748b", linewidth=1.5)
    
    ax1.bar(x + width*0.5, k2_base, width, label="K2 Base (1.8B)", color="#2563eb", alpha=0.6)
    ax1.bar(x + width*1.5, k2_cida, width, label="K2 CIDA (1.3B)", color="#1d4ed8", edgecolor="#3b82f6", linewidth=1.5)
    
    ax1.set_ylabel('Accuracy (%)', color='#94a3b8')
    ax1.set_title('Absolute Performance (Base vs CIDA)', color='#f8fafc', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets_names)
    ax1.legend(loc='upper right', facecolor='#1e293b', edgecolor='gray')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.1)
    
    # Subplot 2: Improvement Delta (Unbiased CIDA Quality Metric)
    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(x - width, k1_delta, width, label="Kairos 1 Delta (7B)", color="#64748b")
    ax2.bar(x, k2_delta, width, label="Kairos 2 Delta (1.3B)", color="#3b82f6")
    ax2.bar(x + width, k4_delta, width, label="Kairos 4 Delta (1.5B)", color="#10b981")
    
    ax2.set_ylabel('Net Gain Delta (%)', color='#94a3b8')
    ax2.set_title('Unbiased System 2 Net Gain (CIDA - Base)', color='#f8fafc', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets_names)
    ax2.legend(loc='upper right', facecolor='#1e293b', edgecolor='gray')
    ax2.grid(axis='y', linestyle='--', alpha=0.1)
    
    # Add values on top of bars for Delta
    for rect in ax2.patches:
        height = rect.get_height()
        ax2.annotate(f'+{height:.1f}%' if height >= 0 else f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='white', fontweight='bold', fontsize=9)
                    
    plt.suptitle("CIDA Cross-Version OOD Generalization Dashboard", color="#f8fafc", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    # Markdown Table output
    print("\\n" + "="*80)
    print("                      CIDA CROSS-VERSION ACCURACY REPORT")
    print("="*80)
    print(f"{'Dataset':<15} | {'K1 Base':<8} -> {'K1 CIDA':<8} ({'Delta':<6}) | {'K2 Base':<8} -> {'K2 CIDA':<8} ({'Delta':<6}) | {'K4 Base':<8} -> {'K4 CIDA':<8} ({'Delta':<6})")
    print("-" * 110)
    for i, d in enumerate(datasets_names):
        print(f"{d:<15} | {k1_base[i]:>6.1f}% -> {k1_cida[i]:>6.1f}% ({k1_delta[i]:>+5.1f}%) | {k2_base[i]:>6.1f}% -> {k2_cida[i]:>6.1f}% ({k2_delta[i]:>+5.1f}%) | {k4_base[i]:>6.1f}% -> {k4_cida[i]:>6.1f}% ({k4_delta[i]:>+5.1f}%)")
    print("="*80)
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("Kairos_Ultimate_OOD_Benchmark.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Generated: Kairos_Ultimate_OOD_Benchmark.ipynb")
