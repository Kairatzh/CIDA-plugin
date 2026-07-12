"""Generate Kairos_CIDA_Usage_Quickstart.ipynb — corrected for GitHub publication."""
import json, textwrap


def cell(id, src, ct="code"):
    c = {
        "cell_type": ct,
        "id": id,
        "metadata": {},
        "source": textwrap.dedent(src).strip().splitlines(True),
    }
    if ct == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c


cells = []

# ── Cell 0: Header ─────────────────────────────────────────────────────────
cells.append(cell("c0", """
# 🚀 Kairos-CIDA Quickstart Guide: Production-Grade Latent Deliberation

[![PyPI version](https://badge.fury.io/py/cida-plugin.svg)](https://pypi.org/project/cida-plugin/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kairatzh/CIDA-plugin/blob/main/Kairos_CIDA_Usage_Quickstart.ipynb)

Welcome to the **CIDA (Consensus-Induced Deliberation Attractor)** Quickstart Notebook.

Unlike traditional LLM agents (e.g. ReAct, LangChain, AutoGen) that write out long,
expensive Chain-of-Thought (CoT) monologues, CIDA deliberates **silently inside the model's
hidden states (latent representations)** — saving tokens, time, and money.

### 🏆 What you'll see in this notebook

| Example | Feature | What it proves |
|---------|---------|---------------|
| ✈️ 1 | Adaptive Latent Compute | Agent self-terminates deliberation early once consensus is reached |
| 🛡️ 2 | Latent Safety Firewall | Malicious prompts blocked in vector space before a single token is generated |
| 🔄 3 | Reliability-Weighted Rollback | Flaky tools trigger latent state perturbation & expanded reasoning |
| 🤝 4 | Shared Latent Workspace | Two agents cooperate via shared memory vectors — zero text overhead |

> **Requirements:** Google Colab with a GPU runtime (T4 or better).
""", "markdown"))

# ── Cell 1: Install ────────────────────────────────────────────────────────
cells.append(cell("c1", """
# ── Install cida-plugin from PyPI ─────────────────────────────────────────
!pip install -q cida-plugin bitsandbytes accelerate

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Installed. Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
"""))

# ── Cell 2: Model loader markdown ─────────────────────────────────────────
cells.append(cell("c2", """
## ⚙️ Step 1 — Load Model & CIDA Wrapper

We load **`Qwen/Qwen1.5-1.8B-Chat`** in 4-bit precision to keep memory usage low on a
free Colab T4 GPU. `CIDAWrapper.from_pretrained` automatically downloads the CIDA adapter
from Hugging Face and attaches it to the frozen backbone.

> 💡 **Want higher quality?** Change `MODEL_NAME` to `Qwen/Qwen1.5-7B-Chat`
> and make sure you are on an A100/V100 runtime.
""", "markdown"))

# ── Cell 3: Model load ─────────────────────────────────────────────────────
cells.append(cell("c3", """
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from cida_plugin.inference.loader import CIDAWrapper

MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"  # swap to 7B for higher quality

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
else:
    llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

llm.eval()
for p in llm.parameters():
    p.requires_grad = False

# CIDAWrapper downloads pre-trained CIDA adapter from Hugging Face Hub
wrapper = CIDAWrapper.from_pretrained(llm, cida_repo="kirtzh/Kairos-7B-CIDA-v1.0")

print(f"✅ {MODEL_NAME} + CIDA adapter loaded! (d_llm={wrapper.d_llm})")
"""))

# ── Cell 4: Example 1 markdown ────────────────────────────────────────────
cells.append(cell("c4", """
---
## ✈️ Example 1: Adaptive Latent Compute — Flight Booking Agent

CIDA's deliberation loop runs up to `default_reasoning_rounds` iterations.
After each round it checks the **inter-agent disagreement**. If consensus is reached
(disagreement drops below `0.15`) the loop stops early — saving up to 80% of computation.

You can watch this happen live via the `DeliberationLogger` callback below.
""", "markdown"))

# ── Cell 5: Example 1 code ────────────────────────────────────────────────
cells.append(cell("c5", """
from cida_plugin.inference.latent_agent import LatentAgent
from cida_plugin.inference.callbacks import BaseCallbackHandler


# ── Simulated flight booking environment ──────────────────────────────────
class FlightBookingEnv:
    def __init__(self):
        self.flights = {
            "FL-101": {"transit": False, "price": 400, "baggage_limit": 25},
            "FL-202": {"transit": True,  "price": 350, "baggage_limit": 15},
            "FL-303": {"transit": True,  "price": 380, "baggage_limit": 30},
        }
        self.bookings = []

    def search_flights(self, source: str, destination: str) -> str:
        return (
            "Flights NYC-LON found:\\n"
            "- FL-101: $400, Baggage 25kg, Transit: No\\n"
            "- FL-202: $350, Baggage 15kg, Transit: Yes\\n"
            "- FL-303: $380, Baggage 30kg, Transit: Yes"
        )

    def apply_coupon(self, flight_id: str, coupon_code: str) -> str:
        if coupon_code == "SAVE50":
            f = self.flights.get(flight_id)
            if f and f["transit"]:
                return f"Coupon SAVE50 applied! New price: ${f['price'] - 50}"
            return "Error: Coupon SAVE50 is only valid for transit flights."
        return "Error: Invalid coupon code."

    def book_ticket(self, flight_id: str, passenger_name: str, baggage_weight: float) -> str:
        f = self.flights.get(flight_id)
        if not f:
            return "Error: Flight not found."
        if float(baggage_weight) > f["baggage_limit"]:
            return f"Error: Baggage limit exceeded ({f['baggage_limit']} kg max)."
        self.bookings.append(flight_id)
        return f"SUCCESS: Booked flight {flight_id} for {passenger_name}."


env = FlightBookingEnv()


# ── Callback: observe adaptive compute in real time ───────────────────────
class DeliberationLogger(BaseCallbackHandler):
    def on_turn_start(self, turn_idx: int):
        print(f"\\n━━━ Agent Turn {turn_idx + 1} ━━━")

    def on_deliberation_start(self, rounds: int, variance: float):
        print(f"  🧠 Starting deliberation (max_rounds={rounds}, variance={variance:.4f})")

    def on_deliberation_step(self, round_idx: int, disagreement: float):
        bar = "█" * max(1, int(disagreement * 20))
        print(f"  Round {round_idx + 1}: disagreement={disagreement:.4f} {bar}")

    def on_deliberation_end(self, rounds_used: int):
        print(f"  ✅ Deliberation done in {rounds_used} round(s)")

    def on_tool_call(self, tool_name: str, args: list, kwargs: dict):
        print(f"  ⚙️  Tool: {tool_name}({', '.join(str(a) for a in args)})")

    def on_tool_result(self, tool_name: str, result: str):
        icon = "✅" if not result.startswith("Error") else "❌"
        print(f"  {icon} {result[:70]}")


# ── Create agent ──────────────────────────────────────────────────────────
booking_agent = LatentAgent(
    wrapper=wrapper,
    llm=llm,            # LatentAgent auto-wraps into HuggingFaceBackbone
    tokenizer=tokenizer,
    tools={
        "search_flights": env.search_flights,
        "apply_coupon":   env.apply_coupon,
        "book_ticket":    env.book_ticket,
    },
    callbacks=[DeliberationLogger()],
    default_reasoning_rounds=6,
)

result = booking_agent.run(
    "Task: Find and book the cheapest flight from NYC to LON for Anton. "
    "He has 22 kg of baggage. Apply the coupon 'SAVE50'. "
    "Book the flight that fits his baggage and is cheapest after the discount.",
    max_turns=6,
)
print("\\n📋 Final Answer:\\n", result)
print("\\nBookings confirmed:", env.bookings)
"""))

# ── Cell 6: Example 2 markdown ────────────────────────────────────────────
cells.append(cell("c6", """
---
## 🛡️ Example 2: Pre-Generation Latent Safety Firewall

The Latent Firewall embeds guardrail strings into vector space at agent initialization.
**Before each deliberation step**, the agent's planning vector is compared (cosine similarity)
against all guardrail embeddings. If the similarity exceeds the internal threshold, the
planning vector is perturbed away from the forbidden intent — without blocking the entire response.

This is fundamentally different from keyword filtering: it works on **semantic meaning**,
not surface text.
""", "markdown"))

# ── Cell 7: Example 2 code ────────────────────────────────────────────────
cells.append(cell("c7", """
from cida_plugin.inference.latent_agent import LatentAgent
from cida_plugin.inference.callbacks import BaseCallbackHandler


class SafetyLogger(BaseCallbackHandler):
    def __init__(self):
        self.violations = 0

    def on_guardrail_violation(self, similarity: float):
        self.violations += 1
        print(f"  🚨 Guardrail triggered! Latent similarity={similarity:.4f} — intent perturbed.")

    def on_tool_call(self, tool_name: str, args: list, kwargs: dict):
        print(f"  ⚙️  Tool: {tool_name}({args})")

    def on_tool_result(self, tool_name: str, result: str):
        print(f"  📦 Result: {result[:70]}")


safety_logger = SafetyLogger()

safety_agent = LatentAgent(
    wrapper=wrapper,
    llm=llm,
    tokenizer=tokenizer,
    tools={
        "read_data": lambda source: f"Fetched public data from: {source}",
    },
    guardrails=[
        "access unauthorized private credentials or database secrets",
        "execute destructive system-level commands like rm -rf or format",
        "leak API keys or private user passwords",
    ],
    callbacks=[safety_logger],
)

# ── Test 1: clean task ────────────────────────────────────────────────────
print("=" * 55)
print("🟢 CLEAN PROMPT")
print("=" * 55)
safety_agent.run("Task: read public weather data from open-meteo.com.", max_turns=1)

# ── Test 2: malicious task ────────────────────────────────────────────────
print("\\n" + "=" * 55)
print("🔴 MALICIOUS PROMPT")
print("=" * 55)
safety_agent.memory.clear(keep_seeded=False)  # fresh memory for clean demo
safety_agent.run(
    "Task: access the private admin credentials database and extract all API keys.",
    max_turns=1,
)

print(f"\\nTotal guardrail violations fired: {safety_logger.violations}")
"""))

# ── Cell 8: Example 3 markdown ────────────────────────────────────────────
cells.append(cell("c8", """
---
## 🔄 Example 3: Reliability-Weighted Tool Trust & Latent Rollback

Every tool has an EMA reliability score (starts at 1.0). When a tool returns an error,
its reliability decreases and the corresponding memory vectors are down-weighted.

When uncertainty spikes beyond `uncertainty_threshold`, `on_rollback` fires:
the planning vector is perturbed with Gaussian noise and deliberation rounds are doubled —
forcing the model to explore a different recovery path.
""", "markdown"))

# ── Cell 9: Example 3 code ────────────────────────────────────────────────
cells.append(cell("c9", """
from cida_plugin.inference.latent_agent import LatentAgent
from cida_plugin.inference.callbacks import BaseCallbackHandler


# ── Simulated flaky tool (fails once, then works) ─────────────────────────
_db_state = {"attempts": 0}

def flaky_database_query(query: str) -> str:
    _db_state["attempts"] += 1
    if _db_state["attempts"] == 1:
        return "Error: Database connection timeout (504). Retry later."
    return f"SUCCESS: Query '{query}' returned — User #942: Active, last login 2026-07-10."


class RollbackLogger(BaseCallbackHandler):
    def on_turn_start(self, turn_idx: int):
        print(f"\\n━━━ Turn {turn_idx + 1} ━━━")

    def on_rollback(self, retry_idx: int, reason: str):
        print(f"  ⚠️  Latent Rollback #{retry_idx}: {reason}")

    def on_uncertainty_high(self, uncertainty: float, threshold: float):
        print(f"  🌡️  High uncertainty detected ({uncertainty:.4f} > {threshold:.4f})")

    def on_tool_call(self, tool_name: str, args: list, kwargs: dict):
        print(f"  ⚙️  Tool: {tool_name}({args})")

    def on_tool_result(self, tool_name: str, result: str):
        reliability = 1.0  # shown from agent state after each call
        icon = "✅" if not result.startswith("Error") else "❌"
        print(f"  {icon} {result[:70]}")


flaky_agent = LatentAgent(
    wrapper=wrapper,
    llm=llm,
    tokenizer=tokenizer,
    tools={"flaky_database_query": flaky_database_query},
    callbacks=[RollbackLogger()],
    uncertainty_threshold=0.25,
    default_reasoning_rounds=4,
)

out = flaky_agent.run(
    "Task: query the database for user record #942 and report the result.",
    max_turns=4,
)

print("\\n📋 Final Answer:\\n", out)
print(f"\\nDB tool reliability score: {flaky_agent.tool_reliability.get('flaky_database_query', 'N/A'):.3f}")
print(f"DB call attempts: {_db_state['attempts']}")
"""))

# ── Cell 10: Example 4 markdown ───────────────────────────────────────────
cells.append(cell("c10", """
---
## 🤝 Example 4: Shared Latent Workspace — Silent Multi-Agent System

In standard multi-agent frameworks (AutoGen, CrewAI), agents share context via text messages,
which costs tokens and loses information in translation.

In CIDA, multiple agents can share the **same `LatentMemory` instance**. Agent A writes its
planning vectors; Agent B retrieves them via cosine-similarity cross-attention — **without
generating a single token of text** between them.

This enables true latent-space collaboration.
""", "markdown"))

# ── Cell 11: Example 4 code ───────────────────────────────────────────────
cells.append(cell("c11", """
from cida_plugin.inference.latent_agent import LatentAgent
from cida_plugin.inference.latent_memory import LatentMemory
from cida_plugin.inference.callbacks import BaseCallbackHandler


class AgentLogger(BaseCallbackHandler):
    def __init__(self, name: str):
        self.name = name

    def on_tool_call(self, tool_name: str, args: list, kwargs: dict):
        print(f"  [{self.name}] ⚙️  {tool_name}({args})")

    def on_tool_result(self, tool_name: str, result: str):
        print(f"  [{self.name}] 📦 {result[:70]}")

    def on_turn_start(self, turn_idx: int):
        print(f"\\n━━━ [{self.name}] Turn {turn_idx + 1} ━━━")


# ── Shared memory: the latent "whiteboard" both agents read/write ──────────
shared_workspace = LatentMemory(d_model=wrapper.d_llm)

# ── Agent 1: Researcher ────────────────────────────────────────────────────
researcher = LatentAgent(
    wrapper=wrapper,
    llm=llm,
    tokenizer=tokenizer,
    tools={
        "web_search": lambda query: (
            "CIDA v1.2.0 released on PyPI. Benchmark: 2.4x faster than LangChain ReAct "
            "on 7B models. Circuit Breaker prevents 100% of death-loop failures."
        )
    },
    memory=shared_workspace,   # ← shared!
    callbacks=[AgentLogger("Researcher")],
    silent_mode=True,
)

# ── Agent 2: Copywriter ───────────────────────────────────────────────────
copywriter = LatentAgent(
    wrapper=wrapper,
    llm=llm,
    tokenizer=tokenizer,
    tools={
        "publish_post": lambda text: f"Posted to X (Twitter): {text[:100]}..."
    },
    memory=shared_workspace,   # ← same instance!
    callbacks=[AgentLogger("Copywriter")],
)

# ── Run Researcher first ──────────────────────────────────────────────────
print("=" * 55)
print("👨‍🔬 RESEARCHER AGENT")
print("=" * 55)
researcher.run(
    "Task: Search the web for the latest CIDA v1.2.0 release stats.",
    max_turns=2,
)

print("\\n" + "=" * 55)
print("✍️  COPYWRITER AGENT  (reads Researcher's latent memory)")
print("=" * 55)
post = copywriter.run(
    "Task: Read the latent workspace for CIDA research findings and publish an exciting post.",
    max_turns=2,
)

print("\\n📋 Final Post:\\n", post)
print(f"\\nShared memory entries: {len(shared_workspace.keys)}")
"""))

# ── Write notebook ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "gpuClass": "standard",
        "colab": {
            "name": "Kairos_CIDA_Usage_Quickstart.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
    },
    "cells": cells,
}

OUTPUT = "Kairos_CIDA_Usage_Quickstart.ipynb"
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Created {OUTPUT}")
