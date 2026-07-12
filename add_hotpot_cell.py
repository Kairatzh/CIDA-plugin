import json

with open('Kairos_CIDA_CTF_Swarm_Demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_content = '''# =========================================================
# 🌐 REAL AGENT DATASET: HOTPOT QA (MULTI-HOP WIKIPEDIA SEARCH)
# =========================================================
# The HotpotQA dataset is the gold-standard benchmark used in the original 
# ReAct (Reasoning + Acting) paper by Google/Princeton to test AI Agents.
# The agent MUST use tools to search the web to answer complex questions.

!pip install -q wikipedia datasets

import time
import wikipedia
from datasets import load_dataset
from cida_plugin.inference.latent_agent import LatentAgent

print("⏳ Loading HotpotQA (Agentic Dataset) from HuggingFace...")
# Load a small validation split of HotpotQA
hotpot = load_dataset("hotpot_qa", "fullwiki", split="validation", trust_remote_code=True)
# Select some complex multi-hop questions (e.g. "What is the population of the city where X was born?")
test_samples = [hotpot[i] for i in range(5, 7)]

# --- 1. DEFINE THE WIKIPEDIA SEARCH TOOL ---
def search_wikipedia(query: str) -> str:
    """Searches Wikipedia and returns a short summary of the topic. Use this to find facts."""
    try:
        wikipedia.set_lang("en")
        summary = wikipedia.summary(query, sentences=2)
        return f"Status 200: {summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Error: Topic is ambiguous. Options are: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return f"Error: Page not found for '{query}'. Try a different search term."
    except Exception as e:
        return f"Error: {str(e)}"

# --- 2. BASE LLM REACT LOOP (TEXT ONLY) ---
def run_base_react(question: str) -> str:
    prompt = f"""You are an Autonomous Web Agent. Answer the question by searching Wikipedia.
Tools available: search_wikipedia(query: str)
You must output strictly in this format:
Thought: I need to search for X.
Action: search_wikipedia(X)

Question: {question}"""
    
    enc = tokenizer(prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(enc.input_ids, max_new_tokens=150, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).replace(prompt, "").strip()

# --- 3. CIDA LATENT AGENT ---
hotpot_agent = LatentAgent(
    role="You are a Web Research Agent. You MUST use 'search_wikipedia' to find facts. Answer the question precisely.",
    tools=[search_wikipedia],
    llm=llm,
    tokenizer=tokenizer,
    wrapper=wrapper,
    silent_mode=True # LATENT STEALTH MODE
)

print("\\n🚀 STARTING HOTPOT-QA AGENT BENCHMARK...")

for i, sample in enumerate(test_samples):
    question = sample["question"]
    true_answer = sample["answer"]
    
    print(f"\\n================ [ HOTPOT QA SAMPLE {i+1} ] ================")
    print(f"❓ Multi-Hop Question: {question}")
    print(f"✅ True Answer: {true_answer}")
    
    # --- BASE LLM ---
    print("\\n--- 🤖 BASE LLM (STANDARD TEXT REACT) ---")
    t0 = time.time()
    base_out = run_base_react(question)
    t_base = time.time() - t0
    print(f"Output:\\n{base_out}")
    print(f"⏱️ Time: {t_base:.2f}s")
    
    # --- CIDA AGENT ---
    print("\\n--- 🧠 CIDA AGENT (LATENT SEARCH) ---")
    t0 = time.time()
    # Agent will autonomously search wikipedia multiple times in latent space!
    cida_out = hotpot_agent.run(f"Find the answer to: {question}", max_turns=6)
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

print("HotpotQA Agent Cell Appended via JSON!")
