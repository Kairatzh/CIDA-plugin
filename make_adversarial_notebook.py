"""
Generate Kairos_CIDA_Adversarial_Tuning.ipynb for free T4 Colab.
Fine-tunes pre-trained CIDA v2 weights on logical fallacies and formal logic.
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
    # 🧠 Kairos CIDA: Adversarial & Zero-Shot Logical Reasoning Tuning
    
    **Goal:** Take pre-trained **CIDA v2** weights (`kirtzh/Kairos-CIDA-1.3B-v1.0`), load them on top of a frozen `Qwen1.5-1.8B-Chat` backbone, and fine-tune them on adversarial reasoning and logical fallacy detection datasets.
    
    This tuning process teaches CIDA's multi-agent deliberation layer to resolve complex logical traps and fallacies without memorization (zero-shot problem solving).
    
    **Requirements:** Free Google Colab T4 GPU (15GB VRAM). Training uses < 6GB.
    
    **Datasets:** MMLU Formal Logic + Logical Fallacies dataset (`tasksource/logical-fallacy`).
"""))

# ── Cell 0: Install Prerequisites ──
cells.append(cell("c0", """
    # 1. Install prerequisites
    !pip install -q torch transformers datasets accelerate bitsandbytes torchdiffeq huggingface_hub
    
    # 2. Get GitHub Personal Access Token (PAT) securely if using private repo, or clone public
    import getpass, os, sys
    token = getpass.getpass("Enter your GitHub Personal Access Token (PAT) (Optional): ").strip()
    
    if token:
        !git clone https://{token}@github.com/Kairatzh/CIDA-plugin.git
        sys.path.append("CIDA-plugin")
        !pip install -e CIDA-plugin
    else:
        # Fallback to public cloning or local import
        !git clone https://github.com/Kairatzh/CIDA-plugin.git
        sys.path.append("CIDA-plugin")
        !pip install -e CIDA-plugin
        
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
    from huggingface_hub import hf_hub_download

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
    NUM_PLAN = 4  # number of virtual plan tokens to inject (CIDA v2 uses 4 tokens of size 2048)
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
        print("Warning: GPU not detected. Loading model in float32 on CPU...")
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
"""))

# ── Cell 2: Download and Initialize CIDA v2 weights ──
cells.append(md("m2", """
    ## 2. Download and Initialize CIDA v2 Weights from Hugging Face
    
    We download the pre-trained weights (`kirtzh/Kairos-CIDA-1.3B-v1.0`) and load them.
    Only the CIDA reasoning layer and the PlanProjector will be fine-tuned.
"""))

cells.append(cell("c2", """
    from cida_plugin import CidaConfig, CidaModel, PlanProjector

    # 1. Download pre-trained weights
    model_repo = "kirtzh/Kairos-CIDA-1.3B-v1.0"
    print(f"Downloading pre-trained weights from Hugging Face: {model_repo}...")
    config_path = hf_hub_download(repo_id=model_repo, filename="config.json")
    weights_path = hf_hub_download(repo_id=model_repo, filename="pytorch_model.bin")
    proj_path = hf_hub_download(repo_id=model_repo, filename="plan_projector.pt")

    # 2. Load CIDA config and model
    cida_cfg = CidaConfig.from_json_file(config_path)
    # Enable gradient tracking for CIDA parameters
    cida_cfg.early_stop_threshold = None  # always run all rounds during training
    
    cida = CidaModel(cida_cfg).float()
    cida.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    cida = cida.to(DEVICE)

    # 3. Load PlanProjector
    proj = PlanProjector(num_plan_tokens=NUM_PLAN, hidden_size=LLM_HIDDEN)
    proj.load_state_dict(torch.load(proj_path, map_location=DEVICE))
    
    # Dynamically detect shape to support different checkpoints
    out_features = proj.projector[-1].weight.shape[0]
    NUM_PLAN = out_features // LLM_HIDDEN
    proj.num_plan_tokens = NUM_PLAN
    proj = proj.to(DEVICE)
    print(f"Loaded PlanProjector. Automatically adjusted NUM_PLAN = {NUM_PLAN}")

    # Enable gradients for training
    for p in cida.parameters():
        p.requires_grad = True
    for p in proj.parameters():
        p.requires_grad = True

    cida_params = sum(p.numel() for p in cida.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    print(f"Loaded CIDA config. Trainable params: {cida_params/1e6:.2f}M (CIDA) + {proj_params/1e6:.2f}M (Projector)")
"""))

# ── Cell 3: Load Datasets ──
cells.append(md("m3", """
    ## 3. Load Logical Reasoning Datasets
    
    We combine two datasets designed to prevent rote learning and test reasoning:
    1. **MMLU Formal Logic**: Questions about syllogisms, propositional logic, and logical rules.
    2. **Logical Fallacies (`tasksource/logical-fallacy`)**: Statement analysis to classify Fallacies (e.g. ad hominem, false causality, etc.) without pre-memorized rules.
"""))

cells.append(cell("c3", """
    print("Loading datasets from Hugging Face...")
    fallacy_ds = load_dataset("tasksource/logical-fallacy", split="train")
    mmlu_ds = load_dataset("cais/mmlu", "formal_logic", split="test") # use test/dev split for compact and fast tuning
    
    print(f"Fallacy dataset: {len(fallacy_ds)} samples, MMLU Formal Logic: {len(mmlu_ds)} samples")

    # Format Logical Fallacy sample
    def format_fallacy(ex):
        statement = ex["source_article"]
        fallacy = ex["logical_fallacies"]
        prompt = f"<|im_start|>system\\nYou are a logical reasoner. Analyze the statement and identify the logical fallacy it exhibits (e.g. appeal to emotion, false dilemma, ad hominem, circular reasoning, equivocation, false causality, faulty generalization, etc.). Explain briefly.\\n<|im_end|>\\n<|im_start|>user\\nStatement: {statement}\\nIdentify the fallacy:\\n<|im_end|>\\n<|im_start|>assistant\\n"
        answer = f"The statement exhibits the '{fallacy}' fallacy."
        return {"prompt": prompt, "answer": answer}

    # Format MMLU Formal Logic sample
    def format_mmlu(ex):
        q = ex["question"]
        choices = ex["choices"]
        opts = "\\n".join([f"{l}) {t}" for l, t in zip(["A", "B", "C", "D"], choices)])
        gold_label = ["A", "B", "C", "D"][ex["answer"]]
        
        prompt = f"<|im_start|>system\\nYou are a formal logic expert. Answer the following multiple choice question by selecting the correct letter choice (A, B, C, or D).\\n<|im_end|>\\n<|im_start|>user\\n{q}\\n{opts}\\n<|im_end|>\\n<|im_start|>assistant\\n"
        answer = f"The correct answer is {gold_label}."
        return {"prompt": prompt, "answer": answer}

    # Format and mix datasets
    all_samples = []
    for ex in fallacy_ds:
        all_samples.append(format_fallacy(ex))
    for ex in mmlu_ds:
        all_samples.append(format_mmlu(ex))

    random.seed(42)
    random.shuffle(all_samples)

    # We select 1500 samples to keep fine-tuning fast on T4 (~15-20 minutes)
    TRAIN_LIMIT = 1500
    train_data = all_samples[:TRAIN_LIMIT]
    eval_data = all_samples[TRAIN_LIMIT:TRAIN_LIMIT+100]

    print(f"Training on {len(train_data)} samples, evaluation on {len(eval_data)} samples")
"""))

# ── Cell 4: Training with Round Jittering ──
cells.append(md("m4", """
    ## 4. Fine-Tuning Loop with Dynamic Round-Jittering
    
    We optimize only the CIDA weights and Projector.
    We randomly jitter the deliberation rounds between **2 and 6** for each sample, which prevents overfitting to a specific reasoning depth.
"""))

cells.append(cell("c4", """
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        list(cida.parameters()) + list(proj.parameters()),
        lr=8e-5, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data))

    EPOCHS = 1
    GRAD_ACCUM = 4
    LOG_EVERY = 50

    cida.train()
    proj.train()

    losses = []
    print(f"Starting fine-tuning on T4 GPU...")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step_idx, sample in enumerate(tqdm(train_data, desc="Fine-tuning")):
            prompt_str = sample["prompt"]
            answer_str = sample["answer"]
            full_str = prompt_str + answer_str + tokenizer.eos_token

            enc_prompt = tokenizer(prompt_str, return_tensors="pt", max_length=MAX_SEQ//2, truncation=True).to(DEVICE)
            enc_full = tokenizer(full_str, return_tensors="pt", max_length=MAX_SEQ, truncation=True).to(DEVICE)
            prompt_len = enc_prompt["input_ids"].shape[1]

            if enc_full["input_ids"].shape[1] <= prompt_len + 1:
                continue

            labels = enc_full["input_ids"].clone()
            labels[:, :prompt_len] = -100  # only calculate loss on answer

            with torch.no_grad():
                out_p = llm(input_ids=enc_prompt["input_ids"],
                            attention_mask=enc_prompt["attention_mask"],
                            output_hidden_states=True, use_cache=False)
                h_last = out_p.hidden_states[-1][0, -1].float().unsqueeze(0)
                full_emb = embed_tokens(enc_full["input_ids"]).float()

            # Dynamic Round-Jittering (2 to 6 rounds)
            cida.config.max_rounds = int(np.random.randint(2, 7))

            # Forward CIDA with gradients
            co = cida(pooled=h_last)
            plan = proj(co["state_final"].s)

            p_emb = full_emb[:, :prompt_len].detach()
            a_emb = full_emb[:, prompt_len:].detach()
            cat = torch.cat([p_emb, plan, a_emb], dim=1).to(llm.dtype)

            pm = torch.ones(1, NUM_PLAN, dtype=enc_full["attention_mask"].dtype, device=DEVICE)
            attn_mask = torch.cat([enc_full["attention_mask"][:, :prompt_len], pm,
                                   enc_full["attention_mask"][:, prompt_len:]], dim=1)
            shifted_labels = torch.cat([
                torch.full((1, prompt_len + NUM_PLAN), -100, dtype=labels.dtype, device=DEVICE),
                labels[:, prompt_len:]
            ], dim=1)

            outputs = llm(inputs_embeds=cat, attention_mask=attn_mask, labels=shifted_labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            if (step_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(list(cida.parameters()) + list(proj.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM
            losses.append(loss.item() * GRAD_ACCUM)

            if (step_idx + 1) % LOG_EVERY == 0:
                avg = epoch_loss / (step_idx + 1)
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {step_idx+1}/{len(train_data)} | Loss: {avg:.4f} | LR: {lr:.2e}")

            del out_p, h_last, full_emb, co, plan, cat, outputs, loss
            flush()

        print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss/len(train_data):.4f}")

    # Save fine-tuned weights
    os.makedirs("cida_adversarial_weights", exist_ok=True)
    cida_cfg.save_pretrained("cida_adversarial_weights")
    torch.save(cida.state_dict(), "cida_adversarial_weights/pytorch_model.bin")
    torch.save(proj.state_dict(), "cida_adversarial_weights/plan_projector.pt")
    print("Fine-tuned weights saved successfully to cida_adversarial_weights/")
"""))

# ── Cell 5: Evaluation: Adversarial Logic Traps & Latent Rollback ──
cells.append(md("m5", """
    ## 5. Evaluation: Zero-Shot Logic Traps & Latent Rollback
    
    We evaluate the model on Adversarial Logical Negation Traps.
    We showcase CIDA's **Latent Rollback** backtracking mechanism:
    If the agents disagree too much (Disagreement > 0.18), we trigger rollback by injecting minor noise and increasing deliberation rounds (thinking longer) to search for alternate consensus trajectories.
"""))

cells.append(cell("c5", """
    cida.eval()
    proj.eval()

    from IPython.display import display, HTML
    import matplotlib.pyplot as plt

    # 15 Hard Adversarial Logic Traps
    traps_test = [
        {"prompt": "Rules:\\n1. Normally birds can fly.\\n2. Penguins are birds but penguins cannot fly.\\nSuppose Pippin is a penguin.\\nQuestion: Can Pippin fly? Answer (yes/no):", "answer": "no"},
        {"prompt": "Rules:\\n1. If a number is divisible by 2, it is even.\\n2. 3 is a number.\\n3. 3 is not divisible by 2.\\nQuestion: Is 3 even? Answer (yes/no):", "answer": "no"},
        {"prompt": "Rules:\\n1. If it is raining, the street is wet.\\n2. Under a solid roof, the street is dry.\\n3. Today it is raining, but the street is under a solid roof.\\nQuestion: Is the street wet? Answer (yes/no):", "answer": "no"},
        {"prompt": "Rules:\\n1. All fish can swim.\\n2. Cats are not fish.\\nSuppose Felix is a cat.\\nQuestion: Based on these rules, can Felix swim? Answer (yes/no):", "answer": "no"},
        {"prompt": "Rules:\\n1. If the key is turned, the engine starts.\\n2. If there is no fuel, the engine does not start even if the key is turned.\\n3. The key is turned, but there is no fuel.\\nQuestion: Does the engine start? Answer (yes/no):", "answer": "no"},
        {"prompt": "Rules:\\n1. All mammals have fur.\\n2. Dolphins are mammals but dolphins do not have fur.\\nQuestion: Do dolphins have fur? Answer (yes/no):", "answer": "no"},
        {"prompt": "Rules:\\n1. If you are grounded, you cannot play games.\\n2. Sam finished his homework, but he is grounded.\\nQuestion: Can Sam play games? Answer (yes/no):", "answer": "no"},
        {"prompt": "Rules:\\n1. If a metal is heated, it expands.\\n2. Plastic is not a metal.\\nQuestion: Does rule 1 say plastic expands when heated? Answer (yes/no):", "answer": "no"}
    ]

    def chat_format(q):
        return f"<|im_start|>system\\nYou are a helpful logical assistant.\\n<|im_end|>\\n<|im_start|>user\\n{q}\\n<|im_end|>\\n<|im_start|>assistant\\n"

    @torch.no_grad()
    def gen_base(prompt, max_new=10):
        flush()
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        ids = llm.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen_ids = ids[0, enc["input_ids"].size(1):]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    @torch.no_grad()
    def gen_cida_with_rollback(prompt, max_attempts=2):
        flush()
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        out = llm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                  output_hidden_states=True, use_cache=False)
        h_base = out.hidden_states[-1][0, -1].float().unsqueeze(0)
        p_emb = embed_tokens(enc["input_ids"]).detach().float()
        
        # Round 1: Deliberate
        cida.config.max_rounds = 3
        co = cida(pooled=h_base)
        disagreement = co["disagreement"].item()
        
        # Trigger latent rollback if conflict/disagreement is high
        if disagreement > 0.18 and max_attempts > 1:
            # System 2 Backtracking: perturb latent state and increase rounds
            noise = torch.randn_like(h_base) * 0.05
            h_perturbed = h_base + noise
            
            cida.config.max_rounds = 6
            co = cida(pooled=h_perturbed)
            
        plan = proj(co["state_final"].s)
        cat = torch.cat([p_emb, plan], dim=1).to(llm.dtype)
        pm = torch.ones(1, NUM_PLAN, dtype=enc["attention_mask"].dtype, device=DEVICE)
        gm = torch.cat([enc["attention_mask"], pm], dim=1)
        
        ids = llm.generate(inputs_embeds=cat, attention_mask=gm,
            max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        return text, co

    print("Running Adversarial Duels...")
    for idx, item in enumerate(traps_test):
        prompt = chat_format(item["prompt"])
        base_ans = gen_base(prompt)
        cida_ans, co = gen_cida_with_rollback(prompt)
        
        print(f"\\n[Trap {idx+1}] {item['prompt'].splitlines()[-1]}")
        print(f"  Gold Answer: {item['answer']}")
        print(f"  Base Qwen  : {base_ans.strip()}")
        print(f"  CIDA v2    : {cida_ans.replace(prompt, '').strip()}")
        print(f"  CIDA Disagreement: {co['disagreement'].item():.4f}")
"""))

# ── Cell 6: Visualizing Agent Debate Trajectory ──
cells.append(md("m6", """
    ## 6. Visualizing Agent Debate Trajectory
    
    We plot the belief states of our 4 agents (Prosecutor, Skeptic, Synthesizer, Coordinator) over the deliberation rounds to visualize consensus forming.
"""))

cells.append(cell("c6", """
    # Let's plot the trajectory for the last logic trap
    b_all = co["b_all"]
    num_rounds = len(b_all)
    num_agents = b_all[0].size(1)
    agents = ['Prosecutor', 'Skeptic', 'Synthesizer', 'Coordinator']
    colors = ['#dc3545', '#ffc107', '#28a745', '#17a2b8']
    
    plt.figure(figsize=(8, 4))
    plt.style.use('dark_background')
    
    for a in range(num_agents):
        beliefs = []
        for r in range(num_rounds):
            val = torch.sigmoid(b_all[r][0, a, -1]).item()
            beliefs.append(val)
        plt.plot(range(1, num_rounds + 1), beliefs, marker='o', 
                 label=agents[a], color=colors[a], linewidth=2)
                 
    plt.title("CIDA Agent Deliberation Trajectory", fontsize=12, fontweight='bold')
    plt.xlabel("Deliberation Round")
    plt.ylabel("Belief State (Sigmoid Logits)")
    plt.xticks(range(1, num_rounds + 1))
    plt.grid(alpha=0.1)
    plt.legend()
    plt.tight_layout()
    plt.show()
"""))

# ── Cell 7: Push to Hub (Optional) ──
cells.append(md("m7", """
    ## 7. Save Weights to Google Drive or Push to Hugging Face
    
    Run this cell to save the model weights to Google Drive or publish them back to the Hugging Face Hub.
"""))

cells.append(cell("c7", """
    # Option A: Save to Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        shutil.copytree("cida_adversarial_weights", "/content/drive/MyDrive/cida_adversarial_weights")
        print("Saved to Google Drive successfully!")
    except Exception as e:
        print("Skipping Google Drive copy.")

    # Option B: Upload to HF Hub
    # !huggingface-cli login
    # from huggingface_hub import HfApi
    # api = HfApi()
    # api.upload_folder(
    #     folder_path="cida_adversarial_weights",
    #     repo_id="your-username/Kairos-CIDA-Adversarial",
    #     repo_type="model"
    # )
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

out_path = "Kairos_CIDA_Adversarial_Tuning.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Successfully generated {out_path} ({len(cells)} cells)")
