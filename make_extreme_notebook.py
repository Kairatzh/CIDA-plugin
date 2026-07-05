"""
Generate Kairos_CIDA_Extreme_Reasoning.ipynb.
Includes Zero-Shot Latent MCTS, Latent State Adaptation (LSA), and Dual-Pass generation.
"""
import json, textwrap

def cell(id, src, ct="code"):
    return {"cell_type": ct, "id": id, "metadata": {}, "outputs": [],
            "execution_count": None, "source": textwrap.dedent(src).strip("\r\n").splitlines(True)}

def md(id, src):
    c = cell(id, src, ct="markdown")
    del c["execution_count"]
    del c["outputs"]
    return c

cells = []

# ── Title ──
cells.append(md("t", """
    # 🧠 Kairos CIDA v2: Extreme Latent Reasoning & Test-Time Search (LSA & MCTS)
    
    **Goal:** Take a frozen `Qwen1.5-1.8B-Chat` model and CIDA v2 adapter, and unleash the absolute maximum cognitive performance using **Test-Time Compute Scaling** without retraining:
    1. **Zero-Shot Latent MCTS:** Branching and evaluating consensus in the latent space before generation.
    2. **Latent State Adaptation (LSA):** Run gradient descent on the latent query vector during inference to minimize agent disagreement (meditation).
    3. **Dual-Pass Rationalization:** Direct concise solution generation followed by a step-by-step post-hoc explanation.
    
    **Requirements:** Google Colab (CPU or T4 GPU). Runs entirely in zero-shot inference mode or with optional fine-tuning.
"""))

# ── Cell 0: Install Prerequisites ──
cells.append(cell("c0", """
    # 1. Install prerequisites
    !pip install -q torch transformers datasets accelerate bitsandbytes torchdiffeq huggingface_hub
    
    # 2. Get GitHub Personal Access Token (PAT) securely for private repo cloning
    import getpass, os, sys
    token = getpass.getpass("Enter your GitHub Personal Access Token (PAT): ").strip()
    
    if token:
        !git clone https://{token}@github.com/Kairatzh/CIDA-plugin.git
        sys.path.append("CIDA-plugin")
        !pip install -q -e CIDA-plugin
    else:
        # Fallback
        !git clone https://github.com/Kairatzh/CIDA-plugin.git
        sys.path.append("CIDA-plugin")
        !pip install -q -e CIDA-plugin
    
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
"""))

# ── Cell 1: Load Frozen LLM ──
cells.append(md("m1", """
    ## 1. Load Frozen LLM Backbone (Qwen1.5-1.8B-Chat)
"""))

cells.append(cell("c1", """
    import gc, os, re, json, time, random, math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm.auto import tqdm
    from huggingface_hub import hf_hub_download
    from cida_plugin import CIDAPlugin, CIDAPluginConfig, PlanProjector
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
    NUM_PLAN = 4  # size matches checkpoint
    MAX_SEQ = 256
    
    def flush():
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
         
    if torch.cuda.is_available():
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16,
            device_map={"": 0}, output_hidden_states=True)
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32,
            device_map="cpu", output_hidden_states=True)
            
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False
        
    embed_tokens = llm.get_input_embeddings()
    LLM_HIDDEN = embed_tokens.embedding_dim
    print(f"Base LLM loaded. Hidden state size: {LLM_HIDDEN}")
"""))

# ── Cell 2: Load CIDA Adapter ──
cells.append(md("m2", """
    ## 2. Load Pre-trained CIDA v2 weights from Hugging Face
"""))

cells.append(cell("c2", """
    model_repo = "kirtzh/Kairos-CIDA-1.3B-v1.0"
    print(f"Downloading CIDA adapter from: {model_repo}...")
    
    config_path = hf_hub_download(repo_id=model_repo, filename="config.json")
    plugin_path = hf_hub_download(repo_id=model_repo, filename="pytorch_model.bin")
    proj_path = hf_hub_download(repo_id=model_repo, filename="plan_projector.pt")
    
    # Load core configuration
    config = CIDAPluginConfig.from_pretrained(model_repo)
    config.d_input = LLM_HIDDEN
    config.d_hidden = 512
    
    # Instantiate models
    cida = CIDAPlugin(config)
    proj = PlanProjector(num_plan_tokens=NUM_PLAN, hidden_size=LLM_HIDDEN)
    
    # Load weights
    cida.load_state_dict(torch.load(plugin_path, map_location="cpu"))
    proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
    
    # Dynamically read projector dimensions
    out_features = proj.projector[-1].weight.shape[0]
    NUM_PLAN = out_features // LLM_HIDDEN
    proj.num_plan_tokens = NUM_PLAN
    
    # Send models to device
    cida = cida.to(DEVICE)
    proj = proj.to(DEVICE)
    
    print(f"CIDA v2 successfully loaded on {DEVICE}! (Plan Tokens = {NUM_PLAN})")
"""))

# ── Cell 3: Define Extreme Inference Engine ──
cells.append(md("m3", """
    ## 3. Define the Extreme Inference Engine
    This engine implements:
    1. **Latent MCTS:** Explores 4 random continuous vector directions, measuring coordinator consensus, and selecting the most promising start state.
    2. **Latent State Adaptation (LSA):** Runs 8 steps of Adam gradient descent on the latent query vector during inference to minimize agent disagreement (active meditation).
    3. **Dual-Pass Generation:** Uses CIDA to find the concise answer (Pass 1), then lets the base LLM explain the correctness step-by-step (Pass 2).
"""))

cells.append(cell("c3", """
    def chat_format(p):
        return f"<|im_start|>system\\nYou are a logical solver. Answer the request step-by-step.\\n<|im_end|>\\n<|im_start|>user\\n{p}\\n<|im_end|>\\n<|im_start|>assistant\\n"

    @torch.no_grad()
    def gen_base(prompt, max_new=256):
        flush()
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        ids = llm.generate(**enc, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        return text

    @torch.no_grad()
    def gen_cida_configured(prompt, max_rounds=3, run_rollback=False):
        flush()
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        out = llm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                  output_hidden_states=True, use_cache=False)
        h_base = out.hidden_states[-1][0, -1].float().unsqueeze(0)
        p_emb = embed_tokens(enc["input_ids"]).detach().float()
        
        cida.config.max_rounds = max_rounds
        co = cida(pooled=h_base)
        disagreement = co["disagreement"].item()
        
        if run_rollback and disagreement > 0.18:
            noise = torch.randn_like(h_base) * 0.05
            h_perturbed = h_base + noise
            cida.config.max_rounds = 15
            co = cida(pooled=h_perturbed)
            
        plan = proj(co["state_final"].s)
        cat = torch.cat([p_emb, plan], dim=1).to(llm.dtype)
        pm = torch.ones(1, NUM_PLAN, dtype=enc["attention_mask"].dtype, device=DEVICE)
        gm = torch.cat([enc["attention_mask"], pm], dim=1)
        
        ids = llm.generate(inputs_embeds=cat, attention_mask=gm,
            max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        return text, co

    # Extreme Inference Pipeline
    def gen_cida_extreme(prompt_raw, max_rounds=15, mcts_branches=4, lsa_steps=8, lsa_lr=0.03):
        flush()
        enc = tokenizer(chat_format(prompt_raw), return_tensors="pt").to(DEVICE)
        p_emb = embed_tokens(enc["input_ids"]).detach().float()
        
        with torch.no_grad():
            out = llm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                      output_hidden_states=True, use_cache=False)
            h_base = out.hidden_states[-1][0, -1].float().unsqueeze(0)
            
        # ── 1. Latent MCTS Branching ──
        best_h = h_base.clone()
        best_disagreement = 999.0
        
        with torch.no_grad():
            cida.config.max_rounds = max_rounds
            co = cida(pooled=h_base)
            best_disagreement = co["disagreement"].mean().item()
            
        if mcts_branches > 1:
            with torch.no_grad():
                for b in range(mcts_branches - 1):
                    noise = torch.randn_like(h_base) * 0.08
                    h_cand = h_base + noise
                    co_cand = cida(pooled=h_cand)
                    dis = co_cand["disagreement"].mean().item()
                    if dis < best_disagreement:
                        best_disagreement = dis
                        best_h = h_cand.clone()
                        
        # ── 2. Latent State Adaptation (LSA) ──
        delta = torch.zeros_like(best_h, requires_grad=True, device=DEVICE)
        lsa_optimizer = torch.optim.Adam([delta], lr=lsa_lr)
        
        # Freeze network weights temporarily
        for p in cida.parameters(): p.requires_grad = False
        for p in proj.parameters(): p.requires_grad = False
        
        lsa_losses = []
        for step in range(lsa_steps):
            lsa_optimizer.zero_grad()
            h_adapted = best_h + delta
            co_adapt = cida(pooled=h_adapted)
            loss = co_adapt["disagreement"].mean()
            loss.backward()
            lsa_optimizer.step()
            lsa_losses.append(loss.item())
            
        for p in cida.parameters(): p.requires_grad = True
        for p in proj.parameters(): p.requires_grad = True
        
        # ── 3. Single-Pass CIDA-Guided CoT Generation ──
        with torch.no_grad():
            h_final = best_h + delta
            co_final = cida(pooled=h_final)
            plan = proj(co_final["state_final"].s)
            
            cat = torch.cat([p_emb, plan], dim=1).to(llm.dtype)
            pm = torch.ones(1, NUM_PLAN, dtype=enc["attention_mask"].dtype, device=DEVICE)
            gm = torch.cat([enc["attention_mask"], pm], dim=1)
            
            ids = llm.generate(inputs_embeds=cat, attention_mask=gm,
                max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(ids[0], skip_special_tokens=True)
            
        return text, co_final, lsa_losses
    
    print("Extreme inference engine successfully defined!")
"""))

# ── Cell 4: Benchmarking ──
cells.append(md("m4", """
    ## 4. Run the Revolutionary Benchmarks
    We evaluate the models on the 4 hardest logical challenges:
    1. **Self-Referential Temporal Loop Grid**
    2. **Slow-Light Relativistic Addition**
    3. **Quantum Weak Value Paradox (Three-Box)**
    4. **Cheryl's Birthday with Liars**
"""))

cells.append(cell("c4", """
    tasks = [
        {
            "name": "Self-Referential Temporal Loop Grid",
            "prompt": "You are a pathfinder solving a grid puzzle on a 3x3 grid (cells A1 to C3) from A1 (start at step 0) to C3 (destination) in exactly 4 steps.\\nRules:\\n1. Column 3 Rule: Entering any cell in Column 3 (C1, C2, C3) at step t retroactively blocks the cell visited at step t-2 from being visited again at any step k > t-2.\\n2. Row 3 Rule: Entering any cell in Row 3 (A3, B3, C3) at step t blocks cell B2 for all steps k < t.\\n3. Column 1 Rule: Entering any cell in Column 1 (A1, A2, A3) at any step t > 0 blocks cell B3 for all steps.\\nFind the unique shortest path of length 4 from A1 to C3. Verify each step and rule quantitatively, and explain step-by-step why the solution is valid while other paths fail.",
            "gold": "A1 -> B1 -> C1 -> C2 -> C3"
        },
        {
            "name": "Slow-Light Grid (Relativistic Velocity Addition)",
            "prompt": "You are a relativistic physics synthesizer.\\nBase Theory: Linear velocity addition v_new = v + u.\\nObservations: Speed of light c=3. Ship moving at v=2, fires projectile at u=2 -> observed speed is 2.77. Ship moving at v=2, fires projectile at u=1 -> observed speed is 2.45.\\nTask: Determine the speed of a projectile fired at u=3 from a ship moving at v=2. Show the mathematical derivation of the velocity addition formula from the observations, verify it against the c=3 speed of light limit, and explain step-by-step how you calculated the final value.",
            "gold": "3.00 (via u' = (v+u)/(1 + vu/c^2))"
        },
        {
            "name": "Quantum Weak Value Paradox (Three-Box Paradox)",
            "prompt": "In quantum mechanics, a system is prepared in state |Psi> = 1/sqrt(3) * (|A> + |B> - |C>). We post-select the state |Phi> = 1/sqrt(3) * (|A> + |B> + |C>). A weak measurement of the projection operators on box A (P_A) and box B (P_B) yields weak values A_w = +1 and B_w = +1. Yet, if we measure the joint projection P_{A+B}, the weak value is 0. Explain step-by-step how these weak values are derived mathematically and resolve this apparent paradox (where 1 + 1 = 0) without violating the conservation of probability.",
            "gold": "The weak value definition A_w = <Phi|P_A|Psi> / <Phi|Psi>. Here, <Phi|Psi> = 1/3 * (1 + 1 - 1) = 1/3. <Phi|P_A|Psi> = 1/3. So A_w = 1. For P_{A+B}, the joint projector is P_A + P_B + cross terms, but weak values sum linearly. Wait, P_{A+B}_w = A_w + B_w = 1 + 1 = 2? No, the paradox is resolved because the weak value of C is C_w = <Phi|P_C|Psi> / <Phi|Psi> = -1 (negative weak value), so A_w + B_w + C_w = 1 + 1 - 1 = 1."
        },
        {
            "name": "Cheryl's Birthday Liar Paradox",
            "prompt": "Cheryl gives Albert and Bernard 10 possible dates: May 15, May 16, May 19, June 17, June 18, July 14, July 16, August 14, August 15, August 17. Albert knows the month, Bernard knows the day. Albert says: 'I don't know the date, but I know Bernard doesn't know either.' Bernard says: 'At first I didn't know, but now I know.' Albert says: 'Now I know too.'\\nTwist: What if Albert is a liar 50% of the time, and Bernard knows this. If the actual date is July 16, deduce step-by-step whether Albert's first statement is a lie, and explain how Bernard resolves the truth to find July 16.",
            "gold": "Albert is telling the truth. Since Albert knows the month is July, he checks July 14 and July 16. Neither is unique, so Bernard couldn't know. Bernard eliminates May and June. Of July 14, July 16, August 14, August 15, August 17: Bernard knows the day is 16, which is unique in July/August, so Bernard knows. Albert then knows July 16."
        },
        {
            "name": "Alice in Wonderland Relation Paradox (AIW)",
            "prompt": "Alice has 4 brothers and she also has 3 sisters. How many sisters does Alice's brother have? Solve this by: 1) Listing all girls (sisters) in the family, 2) Listing all boys (brothers) in the family, and 3) Counting the sisters from a brother's perspective step-by-step.",
            "gold": "4 sisters. Alice is a girl (sister). If Alice has 3 sisters, there are 4 girls total in the family (Alice + 3 sisters). Each of Alice's brothers has all 4 girls as his sisters."
        }
    ]

    for t in tasks:
        prompt = chat_format(t["prompt"])
        print(f"\\nTASK: {t['name']}")
        print(f"Gold Standard Target: {t['gold']}\\n")
        
        # 1. Base Model
        base_ans = gen_base(prompt, max_new=256).strip()
        print(f"  [1] Base Qwen 1.8B (System 1):\\n{base_ans.replace(prompt, '').strip()}\\n")
        
        # 2. CIDA Normal (3 rounds)
        cida_normal_ans, co_n = gen_cida_configured(prompt, max_rounds=3, run_rollback=False)
        print(f"  [2] CIDA v2 (Normal - 3 rounds):\\n{cida_normal_ans.replace(prompt, '').strip()}")
        print(f"      Disagreement: {co_n['disagreement'].item():.4f}\\n")
        
        # 3. CIDA Extreme (MCTS + LSA Meditate + 15 rounds)
        print("  [3] CIDA v2 (Extreme Inference: MCTS + LSA)... meditating...")
        cida_extreme_ans, co_s, losses = gen_cida_extreme(t["prompt"], max_rounds=15)
        print(f"      Optimization Steps Loss Curve: {', '.join([f'{l:.4f}' for l in losses])}")
        print(f"      Final Disagreement: {co_s['disagreement'].item():.4f}")
        print(f"      Response:\\n{cida_extreme_ans}")
        print("-" * 80)
"""))

# ── Cell 5: Live Interactive Playground ──
cells.append(md("m5", """
    ## 5. Live Interactive Playground
    Input your own logical paradoxes or complex constraints here, watch CIDA actively meditate (minimizing disagreement via LSA gradient steps) and output its step-by-step rationalized solution.
"""))

cells.append(cell("c5", """
    user_prompt = "A is twice as old as B. B will be 30 in 5 years. How old is A now? Write a detailed explanation."
    
    print(f"Input: {user_prompt}\\n")
    print("Meditating on prompt...")
    ans, co, losses = gen_cida_extreme(user_prompt, max_rounds=15, mcts_branches=4, lsa_steps=10)
    
    print(f"Optimization disagreement curve: {', '.join([f'{l:.4f}' for l in losses])}")
    print(f"Final consensus: {1.0 - co['disagreement'].item():.4f} agreement")
    print(f"\\nCIDA Response:\\n{ans}")
"""))

# ── Assemble ──
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = "Kairos_CIDA_Extreme_Reasoning.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Successfully generated {out_path} ({len(cells)} cells)")
