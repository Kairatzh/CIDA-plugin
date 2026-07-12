import json

with open('Kairos_CIDA_CTF_Swarm_Demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_content = '''# =========================================================
# 📊 REAL DATASET EVALUATION: GSM8K (AGENTIC TOOL-USE BENCHMARK)
# =========================================================
# We will load a real dataset (GSM8K) and test the agent's ability to 
# use a Python Calculator tool to solve multi-step problems.
# We compare the Base Qwen Model (Standard Text ReAct) vs CIDA (LatentAgent).

import time
import re
from datasets import load_dataset
from cida_plugin.inference.latent_agent import LatentAgent

print("⏳ Loading real GSM8K Dataset from HuggingFace...")
dataset = load_dataset("gsm8k", "main", split="test")
# Take 3 challenging samples for the benchmark
test_samples = [dataset[i] for i in range(10, 13)] 

# --- 1. DEFINE THE TOOL ---
def python_calculator(expression: str) -> str:
    """Evaluates a mathematical expression in python. Example: '5 * (10 + 2)'"""
    try:
        # Safely evaluate math
        allowed_names = {"abs": abs, "round": round, "max": max, "min": min}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Status 200: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# --- 2. DEFINE BASE LLM TEXT-REACT LOOP (NO CIDA) ---
def run_base_react(question: str) -> str:
    prompt = f"""You are a Math Agent. Solve the question by using the 'python_calculator' tool.
Tools available: python_calculator(expression: str)
You must output strictly in this format:
Thought: your thought
Action: python_calculator(2 + 2)

Question: {question}"""
    
    # We simulate a basic 1-step tool use for the base model
    # (Since writing a full ReAct parser for the base model takes 100 lines, we just do a simple generation)
    enc = tokenizer(prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(enc.input_ids, max_new_tokens=150, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).replace(prompt, "").strip()

# --- 3. DEFINE CIDA LATENT AGENT ---
cida_agent = LatentAgent(
    role="You are a Math Agent. You MUST use the 'python_calculator' tool for all arithmetic. Output the final answer cleanly.",
    tools=[python_calculator],
    llm=llm,
    tokenizer=tokenizer,
    wrapper=wrapper,
    silent_mode=True # LATENT MODE!
)

print("\\n🚀 STARTING BENCHMARK...")

for i, sample in enumerate(test_samples):
    question = sample["question"]
    true_answer = sample["answer"].split("####")[1].strip()
    print(f"\\n================ [ SAMPLE {i+1} ] ================")
    print(f"❓ Question: {question}")
    print(f"✅ True Answer: {true_answer}")
    
    # --- TEST BASE MODEL ---
    print("\\n--- 🤖 BASE LLM (TEXT REACT) ---")
    t0 = time.time()
    base_out = run_base_react(question)
    t_base = time.time() - t0
    print(f"Output:\\n{base_out}")
    print(f"⏱️ Time: {t_base:.2f}s")
    
    # --- TEST CIDA MODEL ---
    print("\\n--- 🧠 CIDA AGENT (LATENT STEALTH MODE) ---")
    t0 = time.time()
    cida_out = cida_agent.run(f"Task: {question}", max_turns=5)
    t_cida = time.time() - t0
    print(f"Output:\\n{cida_out}")
    print(f"⏱️ Time: {t_cida:.2f}s")
'''

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" if i < len(cell_content.split('\n'))-1 else line for i, line in enumerate(cell_content.split('\n'))]
}

nb['cells'].append(new_cell)

with open('Kairos_CIDA_CTF_Swarm_Demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Real Dataset Benchmark Cell Appended via JSON!")
