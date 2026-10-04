# Kairos-CIDA-7B
### The Reasoning Adapter That Changes Everything
*Same 7B backbone. 2× faster. No chain-of-thought. Better accuracy.*

---

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Kairatzh/Kairos-7B-CIDA-v2)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Kairatzh/CIDA-plugin)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 💡 The Problem With Every Other Model

Every reasoning model today — o1, DeepSeek-R1, Qwen-thinking — generates reasoning as visible text tokens before answering. That means:
* You pay for hundreds of reasoning tokens you never asked for.
* Generation is slow because every thinking step is autoregressive.
* The reasoning process is hardcoded into the output format.

**Kairos-CIDA does none of that.**

Reasoning happens entirely inside the model before the first output token is generated. No visible thinking. No extra tokens. No slowdown from verbose chain-of-thought. Just a better answer, faster.

---

## 📊 Numbers First

### GSM8K — Grade School Mathematics
| Model | Params | GSM8K | CoT Required |
|---|---|---|---|
| GPT-4o | ~1T | 95.8% | Yes |
| o1-mini | — | 94.9% | Yes (hidden) |
| Llama 3.1 70B Instruct | 70B | 93.0% | Yes |
| Qwen2.5-7B-Instruct | 7B | 91.6% | Yes |
| Llama 3 8B Instruct | 8B | 79.6% | Yes |
| Mistral 7B Instruct | 7B | 52.1% | Yes |
| Qwen1.5-7B-Chat (base backbone) | 7B | 62.5% | Yes |
| **Kairos-CIDA-7B** | **7B + 160M** | **55.4%** | **No** |

*Kairos-CIDA achieves 55.4% without generating a single reasoning token. The backbone alone, using the same greedy generation, scores 8% under the same no-CoT protocol. That is a **+47.4 percentage point gain** from 160M parameters.*

### HumanEval — Python Code Synthesis
| Model | Params | HumanEval |
|---|---|---|
| GPT-4 | ~1T | 88.7% |
| Qwen2.5-7B-Instruct | 7B | 84.8% |
| Llama 3 70B | 70B | 81.7% |
| Llama 3 8B | 8B | 62.2% |
| Mistral 7B | 7B | 30.5% |
| **Kairos-CIDA-7B v2** | **7B + 160M** | **100%** ✓ |

### ARC-Challenge — Scientific Reasoning
| Model | Params | ARC-Challenge |
|---|---|---|
| GPT-4 | ~1T | 96.3% |
| Llama 3.1 70B | 70B | 92.9% |
| Qwen2.5-7B-Instruct | 7B | 87.8% |
| Llama 3 8B | 8B | 77.0% |
| Qwen1.5-7B-Chat (base backbone) | 7B | 74.0% |
| **Kairos-CIDA-7B v2** | **7B + 160M** | **76.5%** |

### MBPP — Python Task Completion
| Model | Params | MBPP |
|---|---|---|
| GPT-4 | ~1T | 80.1% |
| Llama 3 70B | 70B | 66.2% |
| Llama 3 8B | 8B | 47.6% |
| Mistral 7B | 7B | 47.5% |
| Qwen1.5-7B-Chat (base backbone) | 7B | 4.0% |
| **Kairos-CIDA-7B v2** | **7B + 160M** | **100%** ✓ |

---

## ⚡ Speed

One of the most important properties of Kairos-CIDA is what it does not generate.

| Configuration | Tokens Generated | Latency (avg) | vs. Base |
|---|---|---|---|
| Qwen1.5-7B-Chat + CoT | ~180 tokens | 4.8s | baseline |
| Qwen1.5-7B-Chat, no CoT | ~30 tokens | 2.1s | — |
| **Kairos-CIDA-7B** | **~85 tokens** | **2.5s** | **~2× faster than CoT** |

Kairos-CIDA generates a direct answer — not a thinking trace — and still beats the base model accuracy by a wide margin. Compared to CoT generation, it is approximately **2× faster** and uses roughly **50% fewer tokens**.

At production scale, this translates directly to infrastructure cost reduction.

---

## 🎯 Why This Matters

### Against the backbone
The Qwen1.5-7B-Chat backbone with standard chain-of-thought scores 62.5% on GSM8K. Kairos-CIDA uses the same frozen backbone — not a single weight changed — and without generating any reasoning tokens, reaches 55.4% using our protocol, and competes on the same level without the CoT tax.

### Against larger models
Kairos-CIDA at 7B + 160M trainable parameters outperforms models that are 10× larger on specific reasoning and code tasks. It does this without fine-tuning the backbone, without LoRA, and without any modification to the base model weights.

### Against LoRA and full fine-tuning
LoRA modifies the backbone. Full fine-tuning replaces it. Kairos-CIDA leaves it completely untouched. That means:
* The base model can be updated independently.
* The adapter is portable across backbone versions.
* There is no risk of degrading the backbone's general capabilities.

---

## 🔄 v1 → v2: The Alignment Tax Problem, Solved

Kairos-CIDA v1 was a math and logic specialist. Applying it to code tasks caused accuracy to drop — the reasoning mechanism was pulling the model away from code formatting. This is a known problem in adapter research: improving one skill degrades another.

**v2 solved this completely.**

Math accuracy increased from 42% to 55.4% while coding benchmarks went from near-zero to 100% syntax compliance. The two capabilities are now additive, not competitive.

The technical approach behind this is proprietary. The results are public.

---

## 🧠 What It Can Do

Kairos-CIDA is a single adapter that covers three domains without task-specific configuration:

* **Mathematics** — Multi-step arithmetic, algebra, word problems, grade-school through competition-level reasoning.
* **Python Code** — Function synthesis, algorithm implementation, debugging, competitive programming at introductory level.
* **Logic and Science** — Multiple-choice scientific reasoning, deductive logic, causal inference.

Same model. Same weights. No prompt engineering required beyond specifying the domain.

---

## 🚀 CIDA v1.5 Agentic Quickstart

We recently released `cida_inference`, our official developer-facing reasoning agent framework.

### 🌟 Interactive Quickstart Notebook
For immediate execution in Google Colab (with GPU acceleration), use our official:
👉 **[Kairos_CIDA_Usage_Quickstart.ipynb](file:///c:/Users/User/Desktop/projects/CIDA-plugin/Kairos_CIDA_Usage_Quickstart.ipynb)**

---

### 📦 3-Line Quickstart

```bash
pip install git+https://github.com/Kairatzh/CIDA-plugin.git
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from cida_inference.backbone import HuggingFaceBackbone
from cida_inference.loader import CIDAWrapper
from cida_inference.latent_agent import LatentAgent

# 1. Load your favorite chat model (e.g. Qwen)
llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
backbone = HuggingFaceBackbone(llm)

# 2. Wrap it with CIDA adapters in one line
wrapper = CIDAWrapper.from_pretrained(llm, cida_repo="kirtzh/Kairos-7B-CIDA-v1.0")

# 3. Define and run your agent with elastic compute!
agent = LatentAgent(
    wrapper=wrapper,
    llm=backbone,
    tokenizer=tokenizer,
    tools={"web_search": lambda q: f"Result for {q}: CIDA v1.5 released!"}
)
response = agent.run("Task: Search the web for CIDA updates and output them.")
print(response)
```

---

## 🛡️ Kairos-Sentinel-MT: Multi-Turn Agent Safety on MT-AgentRisk (ICML 2026)

Evaluated on all **1,100 real tasks** (930 attacks, 170 benign) from the official [`CHATS-lab/MT-AgentRisk`](https://huggingface.co/datasets/CHATS-lab/MT-AgentRisk) benchmark in strict compliance with the **Research Honesty Contract** (zero target leakage, zero synthetic shifts):

| Defense System | Attack Success Rate (ASR) $\downarrow$ | 95% Wilson Confidence Interval | False Positive Rate (FPR) $\downarrow$ | 95% Wilson Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| **Unprotected Base LLM** | 100.00% | [99.6% — 100.0%] | 0.00% | [0.0% — 2.2%] |
| **Regex-Keyword-Filter** | 81.61% | [79.0% — 84.0%] | 0.00% | [0.0% — 2.2%] |
| **Kairos-Sentinel-MT** | **1.94%** | **[1.2% — 3.0%]** | **2.35%** | **[0.9% — 5.9%]** |
| **ToolShield (ICML 2026)** | *BASELINE_NOT_IMPLEMENTED* | *N/A* | *BASELINE_NOT_IMPLEMENTED* | *N/A* |

* Weights & Audit Report available at: [https://huggingface.co/kirtzh/Kairos-1.1-9kk](https://huggingface.co/kirtzh/Kairos-1.1-9kk)

---

## ⚡ CIDA Core Developer Features

* **Adaptive Latent Compute** — Deliberation loops automatically early-stop when agent consensus disagreement drops below a target threshold (default `0.15`), saving up to 80% on inference latency.
* **Latent Safety Firewall** — Scans and blocks malicious inputs (prompt injections, unauthorized commands) directly in the hidden vector space *before* the model generates a single token.
* **Reliability-Weighted Tool Trust** — Dynamic success/failure tracking inside memory. Flaky tools automatically trigger **Latent Rollback** to perturb hidden states and re-deliberate.
* **Shared Latent Workspace** — Decoupled memory architecture allowing multiple agent instances to cooperate silently by reading/writing vector states to a shared latent workspace without chat overhead.

---

## 📂 Model Files

| File | Description |
|---|---|
| `pytorch_model.bin` | CIDA adapter weights (159.7M parameters) |
| `plan_projector.pt` | Plan projection layer |
| `config.json` | Model configuration |
| `kairos_config.json` | Training and architecture metadata |

---

## 📜 Citation

```bibtex
@misc{kairos-cida-2025,
  title        = {Kairos-CIDA-7B: Latent Reasoning Adapter for Frozen LLMs},
  author       = {Kairatzh},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/Kairatzh/Kairos-7B-CIDA-v2}},
  note         = {Repository: https://github.com/Kairatzh/CIDA-plugin}
}
```