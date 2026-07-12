import os
import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin
from cida_plugin.projector import LatentAEProjector
from datasets import load_dataset
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STUDENT_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
student_llm = AutoModelForCausalLM.from_pretrained(STUDENT_NAME, torch_dtype=torch.bfloat16, device_map="auto")

cfg = CIDAPluginConfig(
    d_input=student_llm.config.hidden_size, d_hidden=1024, d_message=1024,
    num_agents=4, num_classes=2, max_rounds=4,
    num_attn_heads=8, use_perspective_projector=True,
    anonymous_messages=True, sparse_comm_k=3, use_dynamic_trust=True
)

# Load CIDA weights
v5_dir = "/content/drive/MyDrive/Kairos_CIDA_v5_AETHER_RL"
if not os.path.exists(v5_dir):
    v5_dir = "./drive_mirror/Kairos_CIDA_v5_AETHER_RL"
if not os.path.exists(v5_dir):
    # Try finding weights in drive_mirror or other dirs
    v5_dir = "C:/Users/User/Desktop/projects/CIDA-plugin/downloaded_weights/Kairos_CIDA_v5_AETHER_RL"

print(f"Checking for weights in: {v5_dir}")
cida = CIDAPlugin(cfg).to(DEVICE).float()
proj = LatentAEProjector(cfg.d_hidden, student_llm.config.hidden_size, cfg.num_agents, num_plan=8).to(DEVICE)

# Look in common local dirs
for path in [
    "c:/Users/User/.gemini/antigravity/brain/1311f0cd-f650-4658-9d6f-be2aea550617/cida_v5_rl_weights",
    "c:/Users/User/Desktop/projects/CIDA-plugin/cida_v5_rl_weights",
    "c:/Users/User/Desktop/projects/CIDA-plugin/cida_v5_rl_weights/best"
]:
    if os.path.exists(os.path.join(path, "pytorch_model.bin")):
        v5_dir = path
        break

if os.path.exists(os.path.join(v5_dir, "pytorch_model.bin")):
    cida.load_state_dict(torch.load(os.path.join(v5_dir, "pytorch_model.bin"), map_location=DEVICE))
    proj.load_state_dict(torch.load(os.path.join(v5_dir, "projector.pt"), map_location=DEVICE))
    print("Loaded CIDA weights from", v5_dir)
else:
    print("Weights not found.")

def build_prompt(q):
    return f"<|im_start|>system\nSolve logically.\n<|im_end|>\n<|im_start|>user\n{q}\n<|im_end|>\n<|im_start|>assistant\n"

@torch.no_grad()
def test_sample(q):
    enc = tokenizer(build_prompt(q), return_tensors="pt").to(DEVICE)
    
    # Base
    ids_base = student_llm.generate(**enc, max_new_tokens=80, do_sample=False)
    text_base = tokenizer.decode(ids_base[0, enc.input_ids.size(1):], skip_special_tokens=True).strip()
    
    # CIDA
    out = student_llm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
    compute_dtype = out.hidden_states[-1].dtype
    h_base = out.hidden_states[-1][0, -1].float().unsqueeze(0)
    co = cida(pooled=h_base)
    plan = proj(co["state_final"].s)
    
    p_emb = student_llm.get_input_embeddings()(enc.input_ids).detach()
    cat = torch.cat([p_emb.to(compute_dtype), plan.to(compute_dtype)], dim=1)
    pm = torch.ones(1, plan.size(1), dtype=enc.attention_mask.dtype, device=DEVICE)
    gm = torch.cat([enc.attention_mask, pm], dim=1)
    
    ids_cida = student_llm.generate(inputs_embeds=cat, attention_mask=gm, max_new_tokens=80, do_sample=False)
    text_cida = tokenizer.decode(ids_cida[0], skip_special_tokens=True).strip()
    
    print("\n--- TEST SAMPLE ---")
    print(f"Prompt: {q}")
    print(f"Base LLM output: {repr(text_base)}")
    print(f"CIDA output: {repr(text_cida)}")

# Run 2 samples from BBH Fallacies
try:
    ds = load_dataset("lukaemon/bbh", "formal_fallacies", split="test")
    for i in range(2):
        test_sample(ds[i]["input"])
except Exception as e:
    print("Failed to run test:", e)
