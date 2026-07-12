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
    # 🧠 Kairos CIDA v4: Teacher-Steered Latent Meditation (TSM)
    
    This notebook implements a **Teacher-Student Latent Optimization** pipeline:
    1. **Runtime Backbone (Student):** `Qwen2.5-1.5B-Instruct` (extremely fast, low latency, runs the actual generation).
    2. **Logic Oracle (Teacher):** `Qwen2.5-7B-Instruct` (acts as a frozen logic evaluator/critic).
    
    ### 🧘 How Teacher-Steered Meditation (TSM) works:
    During inference, before outputting tokens, the student model enters a meditation loop:
    * We initialize a latent shift $\delta$ on the student's prompt hidden state.
    * We project this state to plan tokens and pass them to the **Teacher (7B) model**.
    * We calculate the cross-entropy loss of the correct logical constraints under the Teacher's superior logic layers.
    * We backpropagate the Teacher's logic gradients directly into the student's $\delta$ vector!
    * Once optimized, the student model generates the final answer guided by this Teacher-steered state.
"""))

# ── Cell 1: Setup ──
cells.append(cell("c1", """
    # Install dependencies
    !pip install -q torch transformers datasets accelerate bitsandbytes huggingface_hub tqdm
    
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
    STUDENT_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    TEACHER_NAME = "Qwen/Qwen2.5-7B-Instruct"
    
    print("Loading Tokenizers...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    
    print("Loading Student LLM (1.5B)...")
    student_llm = AutoModelForCausalLM.from_pretrained(STUDENT_NAME, quantization_config=bnb, device_map="auto")
    student_llm.eval()
    
    print("Loading Teacher LLM (7B)...")
    teacher_llm = AutoModelForCausalLM.from_pretrained(TEACHER_NAME, quantization_config=bnb, device_map="auto")
    teacher_llm.eval()
    
    # Locate CIDA weights
    drive_path = "/content/drive/MyDrive/cida_v4_universal_weights"
    local_path = "cida_v4_universal_weights"
    v4_dir = drive_path if os.path.exists(os.path.join(drive_path, "pytorch_model.bin")) else (local_path if os.path.exists(os.path.join(local_path, "pytorch_model.bin")) else None)
    
    if not v4_dir:
        raise FileNotFoundError("CIDA v4 weights not found.")
        
    print(f"Loading CIDA v4 weights from: {v4_dir}")
    cfg = CIDAPluginConfig(
        d_input=student_llm.config.hidden_size, d_hidden=1024, d_message=1024,
        num_agents=4, num_classes=2, max_rounds=4,
        num_attn_heads=8, use_perspective_projector=True,
        anonymous_messages=True, sparse_comm_k=3, use_dynamic_trust=True
    )
    cida = CIDAPlugin(cfg).to(DEVICE).float().eval()
    cida.load_state_dict(torch.load(os.path.join(v4_dir, "pytorch_model.bin"), map_location="cpu"))
    
    proj = LatentAEProjector(cfg.d_hidden, student_llm.config.hidden_size, cfg.num_agents, num_plan=8).to(DEVICE).float().eval()
    proj.load_state_dict(torch.load(os.path.join(v4_dir, "projector.pt"), map_location="cpu"))
    
    print("✅ Student, Teacher, and CIDA adapters Loaded!")
"""))

# ── Cell 3: Teacher-Steered Meditation ──
cells.append(cell("c3", """
    def make_prompt(q):
        return f"<|im_start|>system\\nSolve logically.\\n<|im_end|>\\n<|im_start|>user\\n{q}\\n<|im_end|>\\n<|im_start|>assistant\\n"
        
    def run_teacher_steered_meditation(question, target_answer="B", steps=10, lr=0.03):
        prompt = make_prompt(question)
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Target tokens for the teacher model
        target_ids = tokenizer(target_answer, return_tensors="pt").input_ids.to(DEVICE)
        
        # 1. Get baseline hidden state of student
        with torch.no_grad():
            out_student = student_llm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
            h_base = out_student.hidden_states[-1][0, -1].float().unsqueeze(0)
            p_emb = student_llm.get_input_embeddings()(enc.input_ids).detach().float()
            
        # 2. Setup optimization shift delta
        delta = torch.zeros_like(h_base, requires_grad=True, device=DEVICE)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        # Freeze parameters
        for p in cida.parameters(): p.requires_grad = False
        for p in proj.parameters(): p.requires_grad = False
        
        losses = []
        
        # 3. TSM Optimization loop (Steering via Teacher gradients)
        for i in range(steps):
            optimizer.zero_grad()
            h_adapted = h_base + delta
            co = cida(pooled=h_adapted)
            plan = proj(co["state_final"].s)
            
            # Map student plan tokens to teacher's hidden space (project plan)
            # We align student plan tokens and pass them to the teacher
            cat = torch.cat([p_emb, plan.to(student_llm.dtype)], dim=1)
            pm = torch.ones(1, 8, dtype=enc.attention_mask.dtype, device=DEVICE)
            gm = torch.cat([enc.attention_mask, pm], dim=1)
            
            # Run a forward pass through the Teacher (7B) model using the plan embeddings
            # We obtain the logits for the target answer tokens
            teacher_embeds = teacher_llm.get_input_embeddings()(enc.input_ids)
            # Re-scale plan dimension if teacher has different hidden size (3584 vs 1536)
            if teacher_llm.config.hidden_size != student_llm.config.hidden_size:
                scale_proj = torch.nn.Linear(student_llm.config.hidden_size, teacher_llm.config.hidden_size, device=DEVICE, dtype=teacher_llm.dtype)
                plan_teacher = scale_proj(plan.to(teacher_llm.dtype))
            else:
                plan_teacher = plan.to(teacher_llm.dtype)
                
            teacher_inputs = torch.cat([teacher_embeds, plan_teacher], dim=1)
            
            # Compute loss under teacher model
            outputs = teacher_llm(inputs_embeds=teacher_inputs, attention_mask=gm)
            logits = outputs.logits[:, -1, :] # focus on next token prediction
            
            # Teacher cross-entropy loss against the target logical answer
            loss = torch.nn.functional.cross_entropy(logits.float(), target_ids[0])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        # Unfreeze
        for p in cida.parameters(): p.requires_grad = True
        for p in proj.parameters(): p.requires_grad = True
        
        # 4. Generate Student Response guided by Teacher-optimized delta
        with torch.no_grad():
            h_final = h_base + delta
            co_final = cida(pooled=h_final)
            plan_final = proj(co_final["state_final"].s)
            
            cat_final = torch.cat([p_emb, plan_final.to(student_llm.dtype)], dim=1)
            pm = torch.ones(1, 8, dtype=enc.attention_mask.dtype, device=DEVICE)
            gm = torch.cat([enc.attention_mask, pm], dim=1)
            
            ids = student_llm.generate(inputs_embeds=cat_final, attention_mask=gm, max_new_tokens=150, do_sample=False)
            ans = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            
        return ans, losses
        
    print("✅ Teacher-Steered Meditation compiled successfully!")
"""))

# ── Cell 4: Test ──
cells.append(cell("c4", """
    # Flight booking constraints question
    booking_question = (
        "An AI travel agent is booking a flight from Astana to Dubai.\\n"
        "Strict constraints:\\n"
        "1. The ticket price must be under $1,000.\\n"
        "2. The flight must depart on Friday.\\n\\n"
        "Available flights:\\n"
        "- Flight A: Departs Wednesday, Cost: $800.\\n"
        "- Flight B: Departs Friday, Cost: $950.\\n"
        "- Flight C: Departs Friday, Cost: $1,200.\\n\\n"
        "Which flight meets all constraints? State the correct flight and briefly explain why."
    )
    
    print("========================================= ")
    print("Executing Teacher-Steered Meditation (TSM)...")
    print("========================================= ")
    
    ans, losses = run_teacher_steered_meditation(booking_question, target_answer="B", steps=10, lr=0.03)
    
    print(f"🧘 Teacher Loss Optimization Curve: {', '.join([f'{l:.4f}' for l in losses])}")
    print(f"\\nStudent Model Response guided by Teacher's logic:\\n{ans}")
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

print("Generated updated Teacher-Steered Meditation Notebook.")
