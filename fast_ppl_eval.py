import torch
import torch.nn as nn
import math
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Configuration & Constants
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
FALLBACK_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PLAN = 8

# Math problems from our showcase dataset (with step-by-step gold solutions)
EVAL_DATA = [
    {
        "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "solution": "Natalia sold 48 / 2 = 24 clips in May. Natalia sold 48 + 24 = 72 clips altogether."
    },
    {
        "prompt": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "solution": "Weng earns $12 / 60 = $0.20 per minute. Yesterday she earned 50 * $0.20 = $10."
    },
    {
        "prompt": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "solution": "Betty has 100 / 2 = 50 dollars. Her grandparents give her 15 * 2 = 30 dollars. In total, Betty has 50 + 15 + 30 = 95 dollars. She needs 100 - 95 = 5 dollars more."
    },
    {
        "prompt": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
        "solution": "Albert buys 2 * 16 = 32 large slices. Albert buys 2 * 8 = 16 small slices. In total, he eats 32 + 16 = 48 slices."
    }
]

print("Initializing Models & Tokenizers...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    print("GPU detected. Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16,
            device_map={"": 0}, quantization_config=bnb_config, output_hidden_states=True)
    except Exception as e:
        print(f"Failed to load {MODEL_NAME} on GPU: {e}")
        print(f"Attempting fallback to smaller model {FALLBACK_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL, trust_remote_code=True, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL, trust_remote_code=True, torch_dtype=torch.float16,
            device_map={"": 0}, quantization_config=bnb_config, output_hidden_states=True)
        MODEL_NAME = FALLBACK_MODEL
else:
    print("Warning: GPU not detected. Loading fallback model on CPU in float32...")
    MODEL_NAME = FALLBACK_MODEL
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32,
        device_map="cpu", output_hidden_states=True)

llm.eval()
embed_tokens = llm.get_input_embeddings()

# Load CIDA
from cida_plugin import CIDAPlugin, CIDAPluginConfig
from huggingface_hub import hf_hub_download
import os

print("Loading CIDA weights...")
config_path = hf_hub_download(repo_id="kirtzh/Kairos-7B-CIDA-v1.0", filename="config.json")
cida_cfg = CIDAPluginConfig.from_pretrained(os.path.dirname(config_path))
cida = CIDAPlugin(cida_cfg).to(DEVICE)
cida_bin_path = hf_hub_download(repo_id="kirtzh/Kairos-7B-CIDA-v1.0", filename="pytorch_model.bin")
cida.load_state_dict(torch.load(cida_bin_path, map_location=DEVICE))
cida.eval()

# Projector
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

LLM_HIDDEN = llm.config.hidden_size
proj = CIDAPlanProjector(d_cida=cida_cfg.d_hidden, d_llm=LLM_HIDDEN, num_agents=cida_cfg.num_agents, num_plan_tokens=NUM_PLAN).to(DEVICE)
proj_path = hf_hub_download(repo_id="kirtzh/Kairos-7B-CIDA-v1.0", filename="plan_projector.pt")
proj.load_state_dict(torch.load(proj_path, map_location=DEVICE))
proj.eval()

# Cross-entropy loss computation function over answer tokens only
def compute_loss_and_perplexity(prompt, target_text, use_cida=False):
    # Formulate inputs using chat templates
    chat_prompt = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
    
    full_str = prompt_str + target_text + tokenizer.eos_token
    
    enc_prompt = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)
    enc_full = tokenizer(full_str, return_tensors="pt").to(DEVICE)
    
    prompt_len = enc_prompt["input_ids"].shape[1]
    
    # We want to measure the model's loss on the targets (solution) only
    target_ids = enc_full["input_ids"].clone()
    target_ids[:, :prompt_len] = -100  # Mask out the prompt tokens in cross-entropy loss computation
    
    with torch.no_grad():
        if not use_cida:
            outputs = llm(input_ids=enc_full["input_ids"], attention_mask=enc_full["attention_mask"], labels=target_ids)
            loss = outputs.loss.item()
        else:
            # 1. Pass only prompt through LLM to extract pooled representations
            out_prompt = llm(input_ids=enc_prompt["input_ids"], attention_mask=enc_prompt["attention_mask"], output_hidden_states=True)
            pooled = out_prompt.hidden_states[-1][0, -1].unsqueeze(0).float()
            
            # 2. Let CIDA deliberate
            old_stop = getattr(cida.config, 'early_stop_threshold', None)
            cida.config.early_stop_threshold = None  # force full reasoning rounds
            co = cida(pooled=pooled)
            cida.config.early_stop_threshold = old_stop
            
            plan = proj(co["state_final"].s)  # Shape: (1, NUM_PLAN, d_llm)
            
            # 3. Embed full sequence (Prompt + Solution)
            full_embeddings = embed_tokens(enc_full["input_ids"]).float() # (1, seq_len, d_llm)
            
            # 4. Inject CIDA deliberation plan tokens directly after the prompt
            p_emb = full_embeddings[:, :prompt_len]
            sol_emb = full_embeddings[:, prompt_len:]
            
            # Concatenate plan tokens: Prompt + Plan + Solution
            cat = torch.cat([p_emb, plan, sol_emb], dim=1).to(llm.dtype)
            
            # Formulate the updated attention mask
            pm = torch.ones(1, NUM_PLAN, dtype=enc_full["attention_mask"].dtype, device=DEVICE)
            gm = torch.cat([enc_full["attention_mask"][:, :prompt_len], pm, enc_full["attention_mask"][:, prompt_len:]], dim=1)
            
            # Update target label positions to align with plan injection
            updated_target_ids = torch.cat([
                torch.full((1, prompt_len + NUM_PLAN), -100, dtype=target_ids.dtype, device=DEVICE),
                target_ids[:, prompt_len:]
            ], dim=1)
            
            outputs = llm(inputs_embeds=cat, attention_mask=gm, labels=updated_target_ids)
            loss = outputs.loss.item()
            
    perplexity = math.exp(loss)
    return loss, perplexity

print("\nRunning Fast Perplexity evaluation on GSM8K Subset...")
print("-" * 80)
print(f"{'Problem ID':<10} | {'Base Loss':<10} | {'Base PPL':<10} | {'CIDA Loss':<10} | {'CIDA PPL':<10}")
print("-" * 80)

total_base_loss = 0.0
total_cida_loss = 0.0

for idx, item in enumerate(EVAL_DATA):
    b_loss, b_ppl = compute_loss_and_perplexity(item["prompt"], item["solution"], use_cida=False)
    c_loss, c_ppl = compute_loss_and_perplexity(item["prompt"], item["solution"], use_cida=True)
    
    total_base_loss += b_loss
    total_cida_loss += c_loss
    
    print(f"Prob {idx+1:<5} | {b_loss:<10.4f} | {b_ppl:<10.2f} | {c_loss:<10.4f} | {c_ppl:<10.2f}")

avg_b_loss = total_base_loss / len(EVAL_DATA)
avg_c_loss = total_cida_loss / len(EVAL_DATA)

print("-" * 80)
print(f"{'AVERAGE':<10} | {avg_b_loss:<10.4f} | {math.exp(avg_b_loss):<10.2f} | {avg_c_loss:<10.4f} | {math.exp(avg_c_loss):<10.2f}")
print("-" * 80)
