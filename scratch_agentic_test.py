import os, torch, re, gc
import torch.nn as nn
from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Упрощенная задача планирования для 1.5B моделей
simplified_problem = {
    "context": (
        "An AI travel agent is booking a flight from Astana to Dubai.\n"
        "Strict constraints:\n"
        "1. The ticket price must be under $1,000.\n"
        "2. The flight must depart on Friday.\n\n"
        "Available flights:\n"
        "- Flight A: Departs Wednesday, Cost: $800.\n"
        "- Flight B: Departs Friday, Cost: $950.\n"
        "- Flight C: Departs Friday, Cost: $1,200."
    ),
    "query": "Which flight meets all constraints?",
    "options": ["A", "B", "C"],
    "correct": "B"
}

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

def extract_letter(text):
    m = re.search(r'\b[A-C]\b', text.upper())
    if m: return m.group(0)
    m2 = re.search(r'[A-C]', text.upper())
    return m2.group(0) if m2 else text.strip().upper()

# Загружаем только Qwen 1.3B/1.8B на CPU (без accelerate) для быстрой проверки логики
print("Loading Qwen-1.8B locally on CPU...")
model_1_3 = "Qwen/Qwen1.5-1.8B-Chat"
tok_1_3 = AutoTokenizer.from_pretrained(model_1_3)
llm_1_3 = AutoModelForCausalLM.from_pretrained(model_1_3, torch_dtype=torch.float32)

repo_1_3 = "kirtzh/Kairos-CIDA-1.3B-v1.0"
cfg_path_1_3 = hf_hub_download(repo_id=repo_1_3, filename="config.json")
cfg_1_3 = CIDAPluginConfig.from_pretrained(os.path.dirname(cfg_path_1_3))
cida_1_3 = CIDAPlugin(cfg_1_3).to(DEVICE).float().eval()
cida_1_3.load_state_dict(torch.load(hf_hub_download(repo_id=repo_1_3, filename="pytorch_model.bin"), map_location="cpu"))

proj_weights_1_3 = torch.load(hf_hub_download(repo_id=repo_1_3, filename="plan_projector.pt"), map_location="cpu")
out_features_1_3 = proj_weights_1_3["projector.3.weight"].shape[0]
num_plan_1_3 = out_features_1_3 // llm_1_3.config.hidden_size

proj_1_3 = CIDAPlanProjectorOld(cfg_1_3.d_hidden, llm_1_3.config.hidden_size, cfg_1_3.num_agents, num_plan_tokens=num_plan_1_3).to(DEVICE).float().eval()
proj_1_3.load_state_dict(proj_weights_1_3)

p = simplified_problem
prompt = f"<|im_start|>system\nSolve logically.\n<|im_end|>\n<|im_start|>user\nContext: {p['context']}\nQuestion: {p['query']}\nOptions: A, B, C\nJust answer with the letter only.\n<|im_end|>\n<|im_start|>assistant\n"
enc = tok_1_3(prompt, return_tensors='pt').to(DEVICE)

with torch.no_grad():
    ids_base = llm_1_3.generate(**enc, max_new_tokens=5, do_sample=False)
    ans_base = tok_1_3.decode(ids_base[0, enc.input_ids.size(1):], skip_special_tokens=True).strip()
    
    out = llm_1_3(**enc, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1].float().unsqueeze(0)
    co = cida_1_3(pooled=h)
    plan = proj_1_3(co['state_final'].s)
    p_emb = llm_1_3.get_input_embeddings()(enc.input_ids)
    cat = torch.cat([p_emb, plan.to(llm_1_3.dtype)], dim=1)
    pm = torch.ones(1, num_plan_1_3, dtype=enc.attention_mask.dtype, device=DEVICE)
    gm = torch.cat([enc.attention_mask, pm], dim=1)
    ids_cida = llm_1_3.generate(inputs_embeds=cat, attention_mask=gm, max_new_tokens=5, do_sample=False)
    ans_cida = tok_1_3.decode(ids_cida[0], skip_special_tokens=True).strip()

print("RESULTS:")
print(f"Base Qwen 1.8B: '{extract_letter(ans_base)}'")
print(f"CIDA 1.3B     : '{extract_letter(ans_cida)}'")
