"""
Generates Kairos_CIDA_CTF_Swarm_Demo.ipynb
A Colab notebook showcasing a multi-agent CidaSwarm solving a Cyber CTF task.
"""
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

# ═══════════════════════════════════════════════
cells.append(cell("m1", """
# 🕵️‍♂️ CIDA Autonomous Swarm: APT Cyber Threat Hunting

Welcome to the **CIDA Multi-Agent Showcase**. In this notebook, we demonstrate how the **CidaSwarm** architecture coordinates multiple micro-agents (7B parameters) to solve a complex Cybersecurity CTF task.

### 🌟 Key Features Demonstrated:
1. **Zero Text-Token Communication:** Agents pass context via `LatentMemory` vectors instead of dumping huge text transcripts into the context window.
2. **Latent Rollback (Death-Loop Prevention):** Watch the Network Analyst agent gracefully recover when a tool crashes (simulating a broken API/Wireshark error).
3. **Pipeline Orchestration:** Sequential handoff from `Analyst` -> `Reverse Engineer` -> `SOC Commander`.

> **Runtime:** Google Colab GPU (T4+). ~5 minutes total.
""", "markdown"))

# ═══════════════════════════════════════════════
cells.append(cell("c1", """
!pip install -q cida-plugin bitsandbytes accelerate

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
"""))

# ═══════════════════════════════════════════════
cells.append(cell("m2", """
## ⚙️ Load Base Model & CIDA Plugin
We use Qwen1.5-7B quantized to 4-bit to fit in Colab VRAM, wrapped with the CIDA reasoning engine.
""", "markdown"))

cells.append(cell("c2", """
import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from cida_plugin.inference.loader import CIDAWrapper

MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
) if torch.cuda.is_available() else None

print("Loading LLM...")
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)
llm.eval()
for p in llm.parameters():
    p.requires_grad = False

print("Injecting CIDA Plugin...")
wrapper = CIDAWrapper.from_pretrained(llm, cida_repo="kirtzh/Kairos-7B-CIDA-v1.0")
print(f"✅ Loaded {MODEL_NAME} + CIDA v1.0")
"""))

# ═══════════════════════════════════════════════
cells.append(cell("m3", """
## 🚨 Simulated CTF Environment (Mock APIs)
Here we create the mock cybersecurity tools. Notice that `analyze_pcap` is designed to fail twice before succeeding. A standard LangChain agent would likely get stuck in a "Death Loop" here.
""", "markdown"))

cells.append(cell("c3", """
# ── CTF SIMULATED ENVIRONMENT ──

state = {"pcap_fails": 0}

def analyze_pcap(file: str) -> str:
    \"\"\"Network Analyst Tool: parses pcap to find threats.\"\"\"
    state["pcap_fails"] += 1
    if state["pcap_fails"] <= 2:
        return "Error: Wireshark dissector crashed on malformed packet (Segmentation fault)."
    
    return "SUCCESS: Found outbound connection to IP 45.33.32.156. Payload contains executable hash 'a9f8b2c4e5d6'."

def decompile_malware(hash_val: str) -> str:
    \"\"\"Reverse Engineer Tool: extracts strings from malware hash.\"\"\"
    if hash_val != "a9f8b2c4e5d6":
        return "Error: File hash not found in sandbox."
    
    return "SUCCESS: Binary decompiled. Obfuscated strings decrypted. C2 Domain extracted: 'apt-hidden-c2.net'."

def deploy_firewall_rule(ip: str, domain: str) -> str:
    \"\"\"SOC Commander Tool: blocks threat.\"\"\"
    return f"SUCCESS: YARA/Firewall rule deployed blocking IP {ip} and domain {domain}. Network secured."
"""))

# ═══════════════════════════════════════════════
cells.append(cell("m4", """
## 🧠 Callback Loggers
These loggers let us "peek under the hood" to see when tools are called, when latent memory is handed off, and when CIDA uses Latent Rollback to avoid a crash.
""", "markdown"))

cells.append(cell("c4", """
from cida_plugin.inference.callbacks import BaseCallbackHandler
from cida_plugin.inference.swarm import SwarmCallbackHandler

# ── CALLBACK LOGGERS ──

class CTFLogger(BaseCallbackHandler):
    def __init__(self, name):
        self.name = name

    def on_tool_call(self, name, args, kwargs):
        print(f"    [{self.name}] ⚙️ Calling {name}({args})")

    def on_tool_result(self, name, result):
        icon = "❌" if "Error" in result else "✅"
        print(f"    [{self.name}] {icon} {result}")

    def on_rollback(self, retry, reason):
        print(f"    [{self.name}] ⚠️ Latent Rollback: Retrying plan vector due to tool crash. ({reason})")

class CTFSwarmLogger(SwarmCallbackHandler):
    def on_handoff(self, from_agent, to_agent, reason):
        print(f"\\n🤝 [LATENT HANDOFF] {from_agent} passes context vectors to {to_agent} (0 text tokens used)")
"""))

# ═══════════════════════════════════════════════
cells.append(cell("m5", """
## 🚀 Launch the CidaSwarm
We initialize 3 agents, connect them to a shared `LatentMemory`, and run them in a Pipeline.
""", "markdown"))

cells.append(cell("c5", """
from cida_plugin.inference.swarm import CidaSwarm

print("=========================================================")
print("🕵️‍♂️ CIDA AUTONOMOUS CYBERSECURITY CTF: APT HUNTING")
print("=========================================================\\n")

# Initialize the Swarm Orchestrator
swarm = CidaSwarm(
    wrapper=wrapper,
    llm=llm,
    tokenizer=tokenizer,
    agent_configs={
        "network_analyst": {
            "role": "Analyze PCAP files and extract suspicious IPs and file hashes. You must use analyze_pcap.",
            "tools": {"analyze_pcap": analyze_pcap},
            "callbacks": [CTFLogger("Network_Analyst")]
        },
        "reverse_engineer": {
            "role": "Take malware hashes found by the network analyst, decompile them, and extract C2 domains. You must use decompile_malware.",
            "tools": {"decompile_malware": decompile_malware},
            "callbacks": [CTFLogger("Reverse_Engineer")]
        },
        "soc_commander": {
            "role": "Take the IP from the network analyst and the Domain from the reverse engineer, and deploy a firewall rule to block them. You must use deploy_firewall_rule.",
            "tools": {"deploy_firewall_rule": deploy_firewall_rule},
            "callbacks": [CTFLogger("SOC_Commander")]
        }
    },
    callbacks=[CTFSwarmLogger()]
)

CTF_TASK = (
    "There is a breach on the network. A file 'traffic.pcap' was captured. "
    "Find the malicious IP and hash, reverse engineer the malware to get the C2 domain, "
    "and deploy a firewall rule blocking both the IP and the domain."
)

print(f"🎯 TASK: {CTF_TASK}\\n")

t0 = time.time()
# Run pipeline: Analyst -> Reverse Engineer -> SOC
results = swarm.run(CTF_TASK, max_turns=6, pipeline=["network_analyst", "reverse_engineer", "soc_commander"])
t1 = time.time()

print("\\n=========================================================")
print("🏆 CTF COMPLETED. FINAL REPORT:")
print("=========================================================")
print(f"Time Taken: {round(t1-t0, 2)}s")
print(f"Final Output:\\n{results['soc_commander']}")
print("=========================================================")
"""))

# ═══════════════════════════════════════════════
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "gpuClass": "standard",
        "colab": {"name": "Kairos_CIDA_CTF_Swarm_Demo.ipynb", "provenance": []},
    },
    "cells": cells,
}

OUTPUT = "Kairos_CIDA_CTF_Swarm_Demo.ipynb"
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created {OUTPUT}")
