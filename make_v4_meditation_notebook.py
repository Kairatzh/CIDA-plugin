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
    # 🧠 Kairos CIDA v4: Active Latent Meditation Playground (Test-Time Compute Scaling)
    
    This notebook implements **Latent State Adaptation (LSA)** on top of your pre-trained **CIDA v4 (1.5B)** weights. 
    
    ### 💡 The Goal:
    Small models (1.5B/1.8B) often lack the raw capacity to trace complex logical rules in a single feedforward pass. To bypass this bottleneck and achieve **CoT-level logical precision** without generating any reasoning tokens, we implement **Active Latent Meditation**:
    
    1. **Freeze** the LLM, CIDA, and Projector weights.
    2. Define a small, trainable latent shift vector $\delta$ initialized to zeros.
    3. Run 8-12 steps of gradient descent (Adam) during inference to **minimize agent disagreement**:
       $$\min_{\delta} \text{disagreement}(h_{base} + \delta)$$
    4. Inject the optimized $\delta$-adapted state through the projector to guide the LLM.
    
    This forces the agents to debate and resolve logical contradictions *internally* in the hidden layers before outputting a single character!
"""))

# ── Cell 1: Setup ──
cells.append(cell("c1", """
    # Install dependencies
    !pip install -q torch transformers datasets accelerate bitsandbytes huggingface_hub matplotlib tqdm
    
    import getpass, sys, os
    token = getpass.getpass("GitHub PAT: ").strip()
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

# ── Cell 2: Load Models ──
cells.append(cell("c2", """
    import os, gc, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from cida_plugin.config import CIDAPluginConfig
    from cida_plugin.core import CIDAPlugin
    from cida_plugin.projector import LatentAEProjector
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    
    print("Loading Base LLM...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    llm.eval()
    
    # Locate CIDA v4 weights (either local or drive)
    drive_path = "/content/drive/MyDrive/cida_v4_universal_weights"
    local_path = "cida_v4_universal_weights"
    v4_dir = drive_path if os.path.exists(os.path.join(drive_path, "pytorch_model.bin")) else (local_path if os.path.exists(os.path.join(local_path, "pytorch_model.bin")) else None)
    
    if not v4_dir:
        raise FileNotFoundError("CIDA v4 weights not found. Please upload them or mount Drive.")
        
    print(f"Loading CIDA v4 weights from: {v4_dir}")
    cfg = CIDAPluginConfig(
        d_input=llm.config.hidden_size, d_hidden=1024, d_message=1024,
        num_agents=4, num_classes=2, max_rounds=4,
        num_attn_heads=8, use_perspective_projector=True,
        anonymous_messages=True, sparse_comm_k=3, use_dynamic_trust=True
    )
    cida = CIDAPlugin(cfg).to(DEVICE).float().eval()
    cida.load_state_dict(torch.load(os.path.join(v4_dir, "pytorch_model.bin"), map_location="cpu"))
    
    proj = LatentAEProjector(cfg.d_hidden, llm.config.hidden_size, cfg.num_agents, num_plan=8).to(DEVICE).float().eval()
    proj.load_state_dict(torch.load(os.path.join(v4_dir, "projector.pt"), map_location="cpu"))
    
    print("✅ All Models & Weights Loaded Successfully!")
"""))

# ── Cell 3: Meditation Engine ──
cells.append(cell("c3", """
    def make_prompt(q):
        return f"<|im_start|>system\\nSolve logically.\\n<|im_end|>\\n<|im_start|>user\\n{q}\\n<|im_end|>\\n<|im_start|>assistant\\n"
        
    def run_latent_meditation(question, steps=10, lr=0.03):
        # 1. Encode prompt
        prompt = make_prompt(question)
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # 2. Extract baseline prompt hidden state
        with torch.no_grad():
            out = llm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
            h_base = out.hidden_states[-1][0, -1].float().unsqueeze(0) # (1, d_llm)
            p_emb = llm.get_input_embeddings()(enc.input_ids).detach().float()
            
        # 3. Create optimization variable (latent shift)
        delta = torch.zeros_like(h_base, requires_grad=True, device=DEVICE)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        # Freeze network parameters to avoid unintended shifts
        for p in cida.parameters(): p.requires_grad = False
        for p in proj.parameters(): p.requires_grad = False
        
        # 4. Latent State Adaptation (Meditation Loop with Stochastic Restarts)
        losses = []
        best_loss = float('inf')
        best_delta = torch.zeros_like(delta)
        
        for i in range(steps):
            optimizer.zero_grad()
            h_adapted = h_base + delta
            co = cida(pooled=h_adapted)
            loss = co["disagreement"].mean()
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            losses.append(loss_val)
            
            # Track best delta
            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.clone()
                
            # Latent Annealing Pivot: If stuck in a local minimum, trigger a stochastic jump
            if i > 2 and loss_val > 0.15 and (losses[-2] - loss_val) < 0.005:
                # Disagreement is high and we are not making progress. Jump!
                temp = 0.05 * loss_val
                with torch.no_grad():
                    delta.add_(torch.randn_like(delta) * temp)
                # Re-initialize optimizer to clear momentum states
                optimizer = torch.optim.Adam([delta], lr=lr)
            
        # Unfreeze network parameters
        for p in cida.parameters(): p.requires_grad = True
        for p in proj.parameters(): p.requires_grad = True
        
        # 5. Generate Answer using optimized best_delta
        with torch.no_grad():
            h_final = h_base + best_delta
            co_final = cida(pooled=h_final)
            plan = proj(co_final["state_final"].s)
            
            cat = torch.cat([p_emb, plan.to(llm.dtype)], dim=1)
            pm = torch.ones(1, 8, dtype=enc.attention_mask.dtype, device=DEVICE)
            gm = torch.cat([enc.attention_mask, pm], dim=1)
            
            ids = llm.generate(inputs_embeds=cat, attention_mask=gm, max_new_tokens=150, do_sample=False)
            ans = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            
        return ans, losses, best_loss
        
    print("✅ Latent Meditation Engine Compiled!")
"""))

# ── Cell 4: Live Demonstration ──
cells.append(cell("c4", """
    # Let's test on our Agentic flight booking problem!
    booking_question = (
        "An AI travel agent is booking a flight from Astana to Dubai.\\n"
        "Strict constraints:\\n"
        "1. The ticket price must be under $1,000.\\n"
        "2. The flight must depart on Friday.\\n\\n"
        "Available flights:\\n"
        "- Flight A: Departs Wednesday, Cost: $800.\\n"
        "- Flight B: Departs Friday, Cost: $950.\\n"
        "- Flight C: Departs Friday, Cost: $1,200.\\n\\n"
        "Which flight meets all constraints? State the correct flight and briefly explain why the others are incorrect."
    )
    
    print("=================================================================")
    print("🔍 TESTING ON AGENTIC FLIGHT BOOKING TRAP:")
    print("=================================================================")
    
    # 1. Base Model Answer
    enc = tokenizer(make_prompt(booking_question), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids_base = llm.generate(**enc, max_new_tokens=150, do_sample=False)
        ans_base = tokenizer.decode(ids_base[0, enc.input_ids.size(1):], skip_special_tokens=True).strip()
    print(f"🔴 Base LLM (System 1) Answer:\\n{ans_base}\\n")
    
    # 2. CIDA Standard (1-pass) Answer
    with torch.no_grad():
        out = llm(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].float().unsqueeze(0)
        co = cida(pooled=h)
        plan = proj(co["state_final"].s)
        p_emb = llm.get_input_embeddings()(enc.input_ids)
        cat = torch.cat([p_emb, plan.to(llm.dtype)], dim=1)
        pm = torch.ones(1, 8, dtype=enc.attention_mask.dtype, device=DEVICE)
        gm = torch.cat([enc.attention_mask, pm], dim=1)
        ids_std = llm.generate(inputs_embeds=cat, attention_mask=gm, max_new_tokens=150, do_sample=False)
        ans_std = tokenizer.decode(ids_std[0], skip_special_tokens=True).strip()
    print(f"🟡 CIDA Standard (1-pass) Answer:\\n{ans_std}\\n")
    
    # 3. CIDA Meditated (Active LSA) Answer
    ans_med, losses, final_dis = run_latent_meditation(booking_question, steps=10, lr=0.03)
    print(f"🟢 CIDA Meditated (LSA 10 steps) Answer:\\n{ans_med}\\n")
    print(f"🧘 Meditation Disagreement Curve: {', '.join([f'{l:.4f}' for l in losses])}")
    print("=================================================================")
"""))

# ── Cell 5: Interactive Playground ──
cells.append(cell("c5", """
    # Interactive Playground: Write your own logical riddle and watch CIDA meditate!
    custom_question = "If A is taller than B, and B is taller than C. Is C taller than A? Answer with Yes or No."
    
    print(f"User Query: {custom_question}\\n")
    print("Meditating on prompt...")
    ans, losses, final_dis = run_latent_meditation(custom_question, steps=12, lr=0.03)
    
    print(f"🧘 Optimization disagreement curve: {', '.join([f'{l:.4f}' for l in losses])}")
    print(f"✨ Final Disagreement: {final_dis:.4f}")
    print(f"\\nCIDA Response: '{ans}'")
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

with open("Kairos_CIDA_v4_Latent_Meditation.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Generated: Kairos_CIDA_v4_Latent_Meditation.ipynb")
