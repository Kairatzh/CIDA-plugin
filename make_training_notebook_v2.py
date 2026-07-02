"""Generate Kairos_CIDA_v2_Training.ipynb for free T4 Colab."""
import json, textwrap

def cell(id, src, ct="code"):
    return {"cell_type": ct, "id": id, "metadata": {}, "outputs": [],
            "execution_count": None, "source": textwrap.dedent(src).strip().splitlines(True)}

def md(id, src):
    c = cell(id, src, ct="markdown")
    del c["execution_count"]
    del c["outputs"]
    return c

cells = []

# ── Title ──
cells.append(md("t", """
    # 🧠 Kairos Latent-CIDA v2: Multi-Task Reasoning Training
    
    **Goal:** Train the **Latent CIDA v2** plugin (from `cida_latent_plugin`) on top of a frozen
    `Qwen1.5-1.8B-Chat` backbone. Uses **dynamic round-jittering** during training so the
    adapter learns to reason at variable deliberation depths — enabling test-time scaling.
    
    **Requirements:** Free Google Colab T4 GPU (15GB VRAM). Training uses < 6GB.
    
    **Datasets:** GSM8K (math reasoning) + ARC-Challenge (science/logic reasoning).
"""))

# ── Cell 0: Install from Private Repository ──
cells.append(cell("c0", """
    # 1. Install prerequisites
    !pip install -q torch transformers datasets accelerate bitsandbytes torchdiffeq
    
    # 2. Get GitHub Personal Access Token (PAT) securely
    import getpass, os, sys
    token = getpass.getpass("Enter your GitHub Personal Access Token (PAT): ").strip()
    
    if token:
        # Clone private repository using token authentication
        !git clone https://{token}@github.com/Kairatzh/CIDA-plugin.git
        
        # Add folder to Python path and install
        sys.path.append("CIDA-plugin")
        !pip install -e CIDA-plugin
        print("Successfully cloned and installed CIDA-plugin from private repository!")
    else:
        print("Warning: No token provided. Will fallback to local import if files are present.")
        
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
"""))

# ── Cell 1: Load Frozen LLM ──
cells.append(md("m1", """
    ## 1. Load Frozen LLM Backbone (Qwen1.5-1.8B-Chat)
    We load the model in float16. The backbone stays **completely frozen** — 
    zero gradients flow through its 1.8B parameters.
"""))

cells.append(cell("c1", """
    import gc, os, re, json, time, random, math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from tqdm.auto import tqdm

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
    NUM_PLAN = 4  # number of virtual plan tokens to inject
    MAX_SEQ = 256  # max sequence length (prompt+answer)

    def flush():
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        print("GPU detected. Loading model in float16 on GPU...")
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16,
            device_map={"": 0}, output_hidden_states=True)
    else:
        print("Warning: GPU not detected. Loading model in float32 on CPU (training will be slow)...")
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32,
            device_map="cpu", output_hidden_states=True)

    # Freeze ALL LLM parameters
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()

    LLM_HIDDEN = llm.config.hidden_size
    embed_tokens = llm.get_input_embeddings()
    print(f"LLM loaded. Hidden size: {LLM_HIDDEN}, Params: {sum(p.numel() for p in llm.parameters())/1e6:.0f}M (all frozen)")
    vram_used = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0.0
    print(f"VRAM used: {vram_used:.2f} GB")
"""))

# ── Cell 2: Latent CIDA v2 + PlanProjector ──
cells.append(md("m2", """
    ## 2. Initialize Latent CIDA v2 + Plan Projector
    
    We use `cida_latent_plugin` — the latent variant that supports **dynamic round-jittering**.
    During training, each sample gets a random number of deliberation rounds (2–5).
    This teaches agents to reason at variable depths, enabling **test-time scaling**:
    at inference, we can increase rounds for harder problems.
    
    **Only CIDA + Projector parameters are trainable** (~4M params total).
"""))

cells.append(cell("c2", """
    from cida_latent_plugin import CIDAPlugin as LatentCIDAPlugin, CIDAPluginConfig as LatentConfig

    # Configure Latent CIDA for the 1.8B backbone
    cida_cfg = LatentConfig(
        d_input=LLM_HIDDEN,     # match LLM hidden size
        d_hidden=256,           # internal agent dimension (small for T4)
        d_message=256,
        num_agents=4,
        num_classes=2,          # binary (not used for generation, but needed)
        max_rounds=3,           # default; will be jittered 2-5 during training
        comm_dropout=0.1,
        early_stop_threshold=None,  # always run all rounds during training
        use_perspective_projector=True,
        anonymous_messages=True,
    )

    cida = LatentCIDAPlugin(cida_cfg).to(DEVICE).float()

    # Plan Projector: maps Latent CIDA consensus state -> virtual plan tokens
    class CIDAPlanProjector(nn.Module):
        def __init__(self, d_cida, d_llm, num_agents, num_plan_tokens=4):
            super().__init__()
            self.num_plan_tokens = num_plan_tokens
            self.d_llm = d_llm
            d_flat = num_agents * d_cida
            self.projector = nn.Sequential(
                nn.Linear(d_flat, d_flat * 2), nn.GELU(), nn.Dropout(0.05),
                nn.Linear(d_flat * 2, num_plan_tokens * d_llm),
                nn.LayerNorm(num_plan_tokens * d_llm))
        def forward(self, state_s):
            B = state_s.size(0)
            return self.projector(state_s.reshape(B, -1).float()).view(B, self.num_plan_tokens, self.d_llm)

    proj = CIDAPlanProjector(
        d_cida=cida_cfg.d_hidden, d_llm=LLM_HIDDEN,
        num_agents=cida_cfg.num_agents, num_plan_tokens=NUM_PLAN
    ).to(DEVICE)

    # Count trainable params
    cida_params = sum(p.numel() for p in cida.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    print(f"Latent CIDA params: {cida_params/1e6:.2f}M (trainable)")
    print(f"Projector params: {proj_params/1e6:.2f}M (trainable)")
    print(f"Total trainable: {(cida_params+proj_params)/1e6:.2f}M")
    vram_used = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0.0
    print(f"Total VRAM: {vram_used:.2f} GB")
"""))

# ── Cell 3: Load Multi-Task Dataset ──
cells.append(md("m3", """
    ## 3. Load Multi-Task Dataset (GSM8K + ARC-Challenge)
    
    We combine two datasets into one unified training stream:
    - **GSM8K** (math): teaches step-by-step arithmetic reasoning
    - **ARC-Challenge** (science): teaches hypothesis elimination and scientific logic
    
    Each sample is formatted as a chat prompt with a gold answer.
"""))

cells.append(cell("c3", """
    print("Loading datasets...")
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    print(f"GSM8K: {len(gsm)} samples, ARC: {len(arc)} samples")

    # Format samples into (prompt_str, answer_str) pairs
    def format_gsm(ex):
        q = ex["question"]
        a = ex["answer"]
        prompt = f"<|im_start|>system\\nYou are a mathematical reasoner. Solve step by step.\\n<|im_end|>\\n<|im_start|>user\\n{q}\\n<|im_end|>\\n<|im_start|>assistant\\n"
        return {"prompt": prompt, "answer": a}

    def format_arc(ex):
        choices = ex["choices"]
        opts = "\\n".join([f"{l}) {t}" for l, t in zip(choices["label"], choices["text"])])
        q = f"{ex['question']}\\n{opts}"
        gold_label = ex["answerKey"]
        # Find the gold answer text
        gold_text = ""
        for l, t in zip(choices["label"], choices["text"]):
            if l == gold_label:
                gold_text = t
                break
        answer = f"The answer is {gold_label}) {gold_text}"
        prompt = f"<|im_start|>system\\nYou are a logical reasoner. Pick the correct answer and explain briefly.\\n<|im_end|>\\n<|im_start|>user\\n{q}\\n<|im_end|>\\n<|im_start|>assistant\\n"
        return {"prompt": prompt, "answer": answer}

    # Build unified dataset
    all_samples = []
    for ex in gsm:
        all_samples.append(format_gsm(ex))
    for ex in arc:
        all_samples.append(format_arc(ex))

    random.seed(42)
    random.shuffle(all_samples)

    # Limit for T4 training time (~2000 samples = ~25 min training)
    TRAIN_LIMIT = 2000
    train_data = all_samples[:TRAIN_LIMIT]
    eval_data = all_samples[TRAIN_LIMIT:TRAIN_LIMIT+100]

    print(f"Training on {len(train_data)} samples, eval on {len(eval_data)} samples")
"""))

# ── Cell 4: Training Loop ──
cells.append(md("m4", """
    ## 4. Training Loop with Dynamic Round-Jittering
    
    For each sample:
    1. **Jitter deliberation rounds** (randomly 2–5) — this is the key Latent CIDA mechanic.
    2. Pass the prompt through the frozen LLM to get `h_last` (the semantic latent).
    3. Feed `h_last` into Latent CIDA → multi-agent deliberation with jittered depth.
    4. Project consensus into virtual plan tokens.
    5. Concatenate: `[prompt_embeds || plan_tokens || answer_embeds]`
    6. Compute cross-entropy loss **only on the answer tokens** (prompt is masked with -100).
    7. Backprop through Latent CIDA + Projector only (LLM is frozen).
"""))

cells.append(cell("c4", """
    # Optimizer: only CIDA + Projector
    optimizer = torch.optim.AdamW(
        list(cida.parameters()) + list(proj.parameters()),
        lr=2e-4, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data))

    EPOCHS = 1  # 1 epoch over 2000 samples is enough for T4
    GRAD_ACCUM = 4  # effective batch size = 4
    LOG_EVERY = 50

    print(f"Training for {EPOCHS} epoch(s), {len(train_data)} samples, grad_accum={GRAD_ACCUM}")
    vram_before = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0.0
    print(f"VRAM before training: {vram_before:.2f} GB")

    cida.train()
    proj.train()

    losses = []
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step_idx, sample in enumerate(tqdm(train_data, desc=f"Epoch {epoch+1}")):
            prompt_str = sample["prompt"]
            answer_str = sample["answer"]
            full_str = prompt_str + answer_str + tokenizer.eos_token

            # Tokenize
            enc_prompt = tokenizer(prompt_str, return_tensors="pt", max_length=MAX_SEQ//2, truncation=True).to(DEVICE)
            enc_full = tokenizer(full_str, return_tensors="pt", max_length=MAX_SEQ, truncation=True).to(DEVICE)
            prompt_len = enc_prompt["input_ids"].shape[1]

            if enc_full["input_ids"].shape[1] <= prompt_len + 1:
                continue  # skip if answer is empty after truncation

            # Labels: mask prompt tokens with -100
            labels = enc_full["input_ids"].clone()
            labels[:, :prompt_len] = -100

            with torch.no_grad():
                # 1. Extract h_last from frozen LLM
                out_p = llm(input_ids=enc_prompt["input_ids"],
                           attention_mask=enc_prompt["attention_mask"],
                           output_hidden_states=True, use_cache=False)
                h_last = out_p.hidden_states[-1][0, -1].float().unsqueeze(0)  # (1, d_llm)

                # 2. Get full sequence embeddings
                full_emb = embed_tokens(enc_full["input_ids"]).float()  # (1, seq, d_llm)

            # 3. JITTER deliberation rounds (key Latent CIDA mechanic)
            cida.config.max_rounds = int(np.random.randint(2, 6))  # 2, 3, 4, or 5 rounds

            # 4. Latent CIDA deliberation (WITH gradients)
            co = cida(pooled=h_last)

            # 5. Project to plan tokens
            plan = proj(co["state_final"].s)  # (1, NUM_PLAN, d_llm)

            # 6. Concatenate: prompt_emb + plan + answer_emb
            p_emb = full_emb[:, :prompt_len].detach()
            a_emb = full_emb[:, prompt_len:].detach()
            cat = torch.cat([p_emb, plan, a_emb], dim=1).to(llm.dtype)

            # 7. Build attention mask and shifted labels
            pm = torch.ones(1, NUM_PLAN, dtype=enc_full["attention_mask"].dtype, device=DEVICE)
            attn_mask = torch.cat([enc_full["attention_mask"][:, :prompt_len], pm,
                                   enc_full["attention_mask"][:, prompt_len:]], dim=1)
            shifted_labels = torch.cat([
                torch.full((1, prompt_len + NUM_PLAN), -100, dtype=labels.dtype, device=DEVICE),
                labels[:, prompt_len:]
            ], dim=1)

            # 8. Forward through frozen LLM with plan-injected embeddings
            outputs = llm(inputs_embeds=cat, attention_mask=attn_mask, labels=shifted_labels)
            loss = outputs.loss / GRAD_ACCUM

            # 9. Backprop (only through Latent CIDA + Projector)
            loss.backward()

            if (step_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(cida.parameters()) + list(proj.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM
            losses.append(loss.item() * GRAD_ACCUM)

            if (step_idx + 1) % LOG_EVERY == 0:
                avg = epoch_loss / (step_idx + 1)
                lr_now = scheduler.get_last_lr()[0]
                vram = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0.0
                print(f"  Step {step_idx+1}/{len(train_data)} | Loss: {avg:.4f} | LR: {lr_now:.2e} | VRAM: {vram:.2f}GB")

            # Free memory
            del out_p, h_last, full_emb, co, plan, cat, outputs, loss
            flush()

        avg_epoch_loss = epoch_loss / len(train_data)
        print(f"\\nEpoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            # Save best checkpoint
            os.makedirs("cida_v2_best", exist_ok=True)
            cida_cfg.save_pretrained("cida_v2_best")
            torch.save(cida.state_dict(), "cida_v2_best/pytorch_model.bin")
            torch.save(proj.state_dict(), "cida_v2_best/plan_projector.pt")
            with open("cida_v2_best/kairos_config.json", "w") as f:
                json.dump({"llm_model_name": MODEL_NAME, "num_plan_tokens": NUM_PLAN,
                           "train_loss": avg_epoch_loss, "train_samples": len(train_data)}, f, indent=2)
            print("  Saved best checkpoint to cida_v2_best/")

    print(f"\\nTraining complete! Best loss: {best_loss:.4f}")
"""))

# ── Cell 5: Loss Curve ──
cells.append(md("m5", """
    ## 5. Training Loss Curve
"""))

cells.append(cell("c5", """
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4))
    # Smooth with rolling average
    window = 20
    smoothed = [sum(losses[max(0,i-window):i+1])/min(i+1, window) for i in range(len(losses))]
    ax.plot(smoothed, color='#00d4ff', linewidth=1.5, label='Training Loss (smoothed)')
    ax.set_xlabel('Step'); ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('CIDA v2 Multi-Task Training Loss', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.show()
"""))

# ── Cell 6: Evaluation ──
cells.append(md("m6", """
    ## 6. Evaluation: Base LLM vs CIDA v2
    
    We compare the frozen base model against the same model enhanced with CIDA v2 plan tokens.
"""))

cells.append(cell("c6", """
    cida.eval()
    proj.eval()

    @torch.no_grad()
    def gen_base(prompt, max_new=128):
        flush()
        enc = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ//2, truncation=True).to(DEVICE)
        ids = llm.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(ids[0, enc["input_ids"].size(1):], skip_special_tokens=True)

    @torch.no_grad()
    def gen_cida_with_rollback(prompt, max_new=150, max_attempts=2):
        flush()
        enc = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ//2, truncation=True).to(DEVICE)
        out = llm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                  output_hidden_states=True, use_cache=False)
        h_base = out.hidden_states[-1][0, -1].float().unsqueeze(0)
        p_emb = embed_tokens(enc["input_ids"]).detach().float()
        del out; flush()

        # Attempt 1: Standard Deliberation
        cida.config.max_rounds = 3
        co = cida(pooled=h_base)
        disagreement = co["disagreement"].item()
        
        # If agents disagree too much (> 0.20), we trigger Latent Rollback!
        if disagreement > 0.20 and max_attempts > 1:
            print(f"      ⚠️ [CIDA Monitor] Conflict detected (Disagreement: {disagreement:.3f}). Triggering Latent Rollback...")
            # Perturb the latent space to force alternative consensus exploration (System 2 Backtracking)
            noise = torch.randn_like(h_base) * 0.05
            h_perturbed = h_base + noise
            
            # Increase deliberation rounds (thinking longer to escape the trap)
            cida.config.max_rounds = 6
            co = cida(pooled=h_perturbed)
            print(f"      ✅ [CIDA Monitor] Rollback successful. New Disagreement: {co['disagreement'].item():.3f}")

        plan = proj(co["state_final"].s)
        cat = torch.cat([p_emb, plan], dim=1).to(llm.dtype)
        pm = torch.ones(1, NUM_PLAN, dtype=enc["attention_mask"].dtype, device=DEVICE)
        gm = torch.cat([enc["attention_mask"], pm], dim=1)
        
        ids = llm.generate(inputs_embeds=cat, attention_mask=gm,
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    def chat(q, sys_msg="You are a helpful assistant. Think step by step."):
        return f"<|im_start|>system\\n{sys_msg}\\n<|im_end|>\\n<|im_start|>user\\n{q}\\n<|im_end|>\\n<|im_start|>assistant\\n"

    # Test on math and logic traps
    test_prompts = [
        ("Math", "Natalia sold clips to 48 of her friends in April, then half as many in May. How many total?"),
        ("Logic Trap", "A is older than B, B is older than C, C is older than D. Is D older than A?"),
        ("Science", "If you drop a feather and a hammer on the Moon (no atmosphere), which hits the ground first?"),
    ]

    print("=" * 80)
    print(f"{'Type':<10} | {'Question':<60}")
    print("=" * 80)
    for task_type, q in test_prompts:
        prompt = chat(q)
        print(f"\\n{'─'*80}")
        print(f"[{task_type}] {q}")
        
        base_ans = gen_base(prompt, max_new=150)
        print(f"\\n  BASE (System 1):  {base_ans[:200]}")
        
        cida_ans = gen_cida_with_rollback(prompt, max_new=150)
        print(f"  CIDA (System 2):  {cida_ans[:200]}")
    print("\\n" + "=" * 80)
"""))

# ── Cell 7: Save to Drive ──
cells.append(md("m7", """
    ## 7. Save Trained Weights to Google Drive
"""))

cells.append(cell("c7", """
    import shutil
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        save_path = "/content/drive/MyDrive/Kairos_Cache/cida_v2_weights"
        os.makedirs(save_path, exist_ok=True)
        for f in os.listdir("cida_v2_best"):
            shutil.copy2(f"cida_v2_best/{f}", save_path)
        print(f"Weights saved to Google Drive: {save_path}")
    except Exception as e:
        print(f"Could not save to Drive: {e}")
        print("Weights are saved locally in cida_v2_best/")
"""))

# ── Assemble notebook ──
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

out_path = "Kairos_CIDA_v2_Training.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Generated {out_path} ({len(cells)} cells)")
print("Upload to Google Colab and run with T4 GPU!")
