"""Generate Kairos_CIDA_Showcase.ipynb — fixed: pip install cida-plugin instead of git clone."""
import json, textwrap

def cell(id, src, ct="code"):
    return {"cell_type": ct, "id": id, "metadata": {}, "outputs": [],
            "execution_count": None, "source": textwrap.dedent(src).strip().splitlines(True)}

cells = []

# ── Cell 0: Install ──
cells.append(cell("c0", """
    !pip install -q torch transformers datasets accelerate torchdiffeq bitsandbytes cida-plugin
    from google.colab import drive
    drive.mount('/content/drive')
    print("Ready!")
"""))

# ── Cell 1: Load Model (8-bit) + CIDA ──
cells.append(cell("c1", """
    import gc, re, json, time, random, torch, torch.nn as nn
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from IPython.display import display, HTML, clear_output
    from cida_plugin import CIDAPlugin, CIDAPluginConfig

    DEVICE = "cuda"
    WEIGHTS_PATH = 'Kairatzh/Kairos-7B-CIDA-v2'

    def flush():
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    try:
        from huggingface_hub import hf_hub_download
        kcfg_path = hf_hub_download(repo_id=WEIGHTS_PATH, filename="kairos_config.json")
        with open(kcfg_path) as f: info = json.load(f)
    except Exception:
        info = {}

    MODEL_NAME = info.get("llm_model_name", "Qwen/Qwen1.5-7B-Chat")
    NUM_PLAN = info.get("num_plan_tokens", 8)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {MODEL_NAME} in 8-bit (this takes 2-3 minutes on first run)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, dtype=torch.float16,
        device_map="auto", quantization_config=bnb_config, output_hidden_states=True)
    for p in llm.parameters(): p.requires_grad = False
    llm.eval()

    LLM_HIDDEN = llm.config.hidden_size
    embed_tokens = llm.get_input_embeddings()

    # CIDA
    cida_cfg = CIDAPluginConfig.from_pretrained(WEIGHTS_PATH)
    cida = CIDAPlugin(cida_cfg)
    cida_bin_path = hf_hub_download(repo_id=WEIGHTS_PATH, filename="pytorch_model.bin")
    sd = torch.load(cida_bin_path, map_location="cpu")
    cida.load_state_dict(sd); del sd; flush()
    cida = cida.to(DEVICE).float().eval()

    # PlanProjector
    class CIDAPlanProjector(nn.Module):
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

    proj = CIDAPlanProjector(d_cida=cida_cfg.d_hidden, d_llm=LLM_HIDDEN,
        num_agents=cida_cfg.num_agents, num_plan_tokens=NUM_PLAN).to(DEVICE)
    proj_path = hf_hub_download(repo_id=WEIGHTS_PATH, filename="plan_projector.pt")
    proj.load_state_dict(torch.load(proj_path, map_location=DEVICE))
    proj.eval(); flush()
    print(f"Loaded: {MODEL_NAME} (8-bit) + CIDA ({NUM_PLAN} plan tokens)")
"""))

# ── Cell 2: Generation functions ──
cells.append(cell("c2", """
    @torch.no_grad()
    def gen_base(prompt, max_new=256):
        chat = [{"role": "user", "content": prompt}]
        prompt_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt_chat, return_tensors="pt").to(DEVICE)
        t0 = time.perf_counter()
        ids = llm.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                           max_new_tokens=max_new, do_sample=False, temperature=1.0)
        dt = time.perf_counter() - t0
        out_ids = ids[0][enc["input_ids"].shape[1]:]
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        return text, len(out_ids), dt

    @torch.no_grad()
    def gen_kairos(prompt, max_new=256):
        chat = [{"role": "user", "content": prompt}]
        prompt_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt_chat, return_tensors="pt").to(DEVICE)
        t0 = time.perf_counter()
        out = llm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                  output_hidden_states=True)
        pooled = out.hidden_states[-1][0, -1].unsqueeze(0).float()
        p_emb = embed_tokens(enc["input_ids"]).float()
        co = cida(pooled=pooled)
        plan = proj(co["state_final"].s)
        cat = torch.cat([p_emb, plan], dim=1).to(torch.float16)
        pm = torch.ones(1, NUM_PLAN, dtype=enc["attention_mask"].dtype, device=DEVICE)
        gm = torch.cat([enc["attention_mask"], pm], dim=1)
        ids = llm.generate(inputs_embeds=cat, attention_mask=gm,
                           max_new_tokens=max_new, do_sample=False, temperature=1.0)
        dt = time.perf_counter() - t0
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        return text, len(ids[0]), dt, co
    print("Generation functions ready.")
"""))

# ── Cell 3: HTML Renderer ──
cells.append(cell("c3", r"""
    def parse_gsm_answer(text):
        nums = re.findall(r'-?[\d,]+\.?\d*', text.replace(',',''))
        return float(nums[-1]) if nums else None

    def render_split(prompt, gold_answer, base_text, base_toks, base_time,
                     kairos_text, kairos_toks, kairos_time, cida_out=None):
        base_correct = False
        kairos_correct = False
        if gold_answer is not None:
            bp = parse_gsm_answer(base_text)
            kp = parse_gsm_answer(kairos_text)
            if bp is not None: base_correct = (abs(bp - gold_answer) < 0.01)
            if kp is not None: kairos_correct = (abs(kp - gold_answer) < 0.01)

        b_speed = base_toks / base_time if base_time > 0 else 0
        k_speed = kairos_toks / kairos_time if kairos_time > 0 else 0
        speedup = base_time / kairos_time if kairos_time > 0 else 0

        b_border = "#22c55e" if base_correct else "#ef4444"
        k_border = "#22c55e" if kairos_correct else "#ef4444"
        b_badge = "CORRECT" if base_correct else "WRONG"
        k_badge = "CORRECT" if kairos_correct else "WRONG"

        if gold_answer is None:
            b_border = "#64748b"; k_border = "#64748b"
            b_badge = "MANUAL CHECK"; k_badge = "MANUAL CHECK"

        b_text_esc = base_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        k_text_esc = kairos_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

        disagree = ""
        if cida_out and "disagreement" in cida_out:
            d = cida_out["disagreement"].item()
            disagree = f'<div style="margin-top:8px;font-size:12px;color:#94a3b8">Agent Disagreement: {d:.4f}</div>'

        gold_str = str(gold_answer) if gold_answer is not None else "N/A (code/logic)"

        html = f'''
        <div style="font-family:'Inter',sans-serif;margin:16px 0">
          <div style="background:#0f172a;border-radius:12px;padding:12px 16px;margin-bottom:12px">
            <div style="color:#f8fafc;font-size:14px;font-weight:600">Prompt</div>
            <div style="color:#cbd5e1;font-size:13px;margin-top:6px;white-space:pre-wrap">{prompt[:500]}</div>
            <div style="color:#fbbf24;font-size:13px;margin-top:4px">Gold Answer: <b>{gold_str}</b></div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div style="background:#1e293b;border:2px solid {b_border};border-radius:12px;padding:14px">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:#f8fafc;font-size:15px;font-weight:700">Base Qwen-7B</span>
                <span style="background:{b_border};color:#fff;padding:2px 10px;border-radius:20px;font-size:12px">{b_badge}</span>
              </div>
              <div style="color:#94a3b8;font-size:12px;margin-top:8px">
                Time: {base_time:.2f}s | Tokens: {base_toks} | Speed: {b_speed:.1f} tok/s
              </div>
              <div style="color:#e2e8f0;font-size:13px;margin-top:10px;max-height:300px;overflow-y:auto;
                          background:#0f172a;padding:10px;border-radius:8px;line-height:1.5;
                          white-space:pre-wrap;font-family:monospace;">{b_text_esc}</div>
            </div>
            <div style="background:#1e293b;border:2px solid {k_border};border-radius:12px;padding:14px">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:#f8fafc;font-size:15px;font-weight:700">Kairos-CIDA-7B</span>
                <span style="background:{k_border};color:#fff;padding:2px 10px;border-radius:20px;font-size:12px">{k_badge}</span>
              </div>
              <div style="color:#94a3b8;font-size:12px;margin-top:8px">
                Time: {kairos_time:.2f}s | Tokens: {kairos_toks} | Speed: {k_speed:.1f} tok/s
                | <span style="color:#22d3ee">{speedup:.1f}x vs base</span>
              </div>{disagree}
              <div style="color:#e2e8f0;font-size:13px;margin-top:10px;max-height:300px;overflow-y:auto;
                          background:#0f172a;padding:10px;border-radius:8px;line-height:1.5;
                          white-space:pre-wrap;font-family:monospace;">{k_text_esc}</div>
            </div>
          </div>
        </div>'''
        display(HTML(html))
    print("Renderer ready.")
"""))

# ── Cell 4: Problems ──
cells.append(cell("c4", """
    PROBLEMS = [
        {"category": "Math (GSM8K-style)",
         "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
         "gold": 72},
        {"category": "Math (Multi-step)",
         "prompt": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
         "gold": 10},
        {"category": "Math (Fractions)",
         "prompt": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
         "gold": 5},
        {"category": "Math (Complex)",
         "prompt": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
         "gold": 48},
        {"category": "Code (Function)",
         "prompt": "Write a Python function that checks if a given number is prime. Return True if prime, False otherwise.",
         "gold": None},
        {"category": "Code (Algorithm)",
         "prompt": "Write a Python function to compute the nth Fibonacci number using iteration (not recursion).",
         "gold": None},
        {"category": "Logic (ARC-style)",
         "prompt": "What happens to the speed of sound as the temperature of the medium increases? A) decreases B) increases C) stays the same D) becomes zero. Answer with just the letter.",
         "gold": None},
        {"category": "Logic (Reasoning)",
         "prompt": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Answer Yes or No and explain briefly.",
         "gold": None},
        {"category": "Code (World Model / Physics)",
         "prompt": "Write the Python logic for updating the vertical position `y` and `velocity` of a bouncing ball dropping to the floor (`y=0`). Write only the update function, considering realistic physical constraints like energy loss.",
         "gold": None},
        {"category": "Logic (World Model / Thermodynamics)",
         "prompt": "A refrigerator is placed in a completely sealed, thermally insulated room. The refrigerator door is left wide open, and it is plugged in and running at full power. After 3 hours, what happens to the average temperature of the room? Answer briefly.",
         "gold": None},
    ]
    print(f"Loaded {len(PROBLEMS)} benchmark problems.")
"""))

# ── Cell 5: Run ──
cells.append(cell("c5", """
    display(HTML('''
    <div style="font-family:'Inter',sans-serif;text-align:center;padding:24px;
                background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:16px;margin-bottom:20px">
      <h1 style="color:#f8fafc;margin:0;font-size:28px">Kairos-CIDA-7B Showcase</h1>
      <p style="color:#94a3b8;margin:6px 0 0">Base Qwen-7B vs Kairos-CIDA-7B Side by Side</p>
    </div>'''))

    total_base_time = 0; total_kairos_time = 0
    total_base_toks = 0; total_kairos_toks = 0
    base_correct = 0; kairos_correct = 0; math_count = 0

    for i, prob in enumerate(PROBLEMS):
        display(HTML(f'<h3 style="font-family:Inter,sans-serif;color:#f8fafc;background:#334155;'
                     f'padding:8px 16px;border-radius:8px;margin:16px 0 4px">'
                     f'Test {i+1}/{len(PROBLEMS)}: {prob["category"]}</h3>'))
        bt, bn, btime = gen_base(prob["prompt"], max_new=256)
        kt, kn, ktime, co = gen_kairos(prob["prompt"], max_new=256)
        total_base_time += btime; total_kairos_time += ktime
        total_base_toks += bn; total_kairos_toks += kn
        if prob["gold"] is not None:
            math_count += 1
            bp = parse_gsm_answer(bt)
            kp = parse_gsm_answer(kt)
            if bp is not None and abs(bp - prob["gold"]) < 0.01: base_correct += 1
            if kp is not None and abs(kp - prob["gold"]) < 0.01: kairos_correct += 1
        render_split(prob["prompt"], prob["gold"], bt, bn, btime, kt, kn, ktime, co)
        flush()

    avg_b = total_base_toks/total_base_time if total_base_time>0 else 0
    avg_k = total_kairos_toks/total_kairos_time if total_kairos_time>0 else 0
    sp = total_base_time/total_kairos_time if total_kairos_time>0 else 0

    display(HTML(f'''
    <div style="font-family:'Inter',sans-serif;margin-top:24px;padding:20px;
                background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:16px">
      <h2 style="color:#f8fafc;text-align:center;margin:0 0 16px">Final Scoreboard</h2>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div style="background:#1e293b;border-radius:12px;padding:16px;text-align:center">
          <div style="color:#94a3b8;font-size:13px">Base Qwen-7B</div>
          <div style="color:#f8fafc;font-size:24px;font-weight:700;margin:8px 0">{total_base_time:.1f}s total</div>
          <div style="color:#94a3b8;font-size:12px">{total_base_toks} tokens | {avg_b:.1f} tok/s</div>
          <div style="color:#ef4444;font-size:14px;margin-top:8px">Math: {base_correct}/{math_count}</div>
        </div>
        <div style="background:#1e293b;border:2px solid #22d3ee;border-radius:12px;padding:16px;text-align:center">
          <div style="color:#22d3ee;font-size:13px;font-weight:600">Kairos-CIDA-7B</div>
          <div style="color:#f8fafc;font-size:24px;font-weight:700;margin:8px 0">{total_kairos_time:.1f}s total</div>
          <div style="color:#94a3b8;font-size:12px">{total_kairos_toks} tokens | {avg_k:.1f} tok/s</div>
          <div style="color:#22c55e;font-size:14px;margin-top:8px">Math: {kairos_correct}/{math_count}</div>
          <div style="color:#22d3ee;font-size:16px;font-weight:700;margin-top:8px">{sp:.1f}x faster overall</div>
        </div>
      </div>
    </div>'''))
    print("Showcase complete!")
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
        "accelerator": "GPU", "gpuClass": "standard"
    },
    "cells": cells
}

with open("Kairos_CIDA_Showcase.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Created: Kairos_CIDA_Showcase.ipynb")
