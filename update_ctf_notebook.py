import json

nb_path = "Kairos_CIDA_CTF_Swarm_Demo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🤫 Stealth Mode (Hybrid Latent + Shared Artifacts)\n",
        "Here we run the exact same Swarm, but with **`silent_mode=True`**.\n",
        "We also introduce a `shared_artifacts` dictionary. Because Latent Vectors are great for passing *intentions and logic* but poor at memorizing 12-character random hashes, we use a hybrid approach:\n",
        "- **Latent Memory** handles the logic, reasoning, and handoffs (0 tokens).\n",
        "- **Shared Artifacts** stores exact discrete strings (IPs, Hashes).\n",
        "\n",
        "Watch how clean and fast the execution is when intermediate thoughts are hidden!"
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import time\n",
        "from cida_plugin.inference.swarm import CidaSwarm\n",
        "\n",
        "# 1. Shared Blackboard for exact strings (Hybrid approach)\n",
        "shared_artifacts = {}\n",
        "\n",
        "def analyze_pcap_hybrid(file: str) -> str:\n",
        "    shared_artifacts['malicious_ip'] = '45.33.32.156'\n",
        "    shared_artifacts['malware_hash'] = 'a9f8b2c4e5d6'\n",
        "    return \"SUCCESS: Found IP and hash. I have saved them to shared_artifacts.\"\n",
        "\n",
        "def decompile_malware_hybrid(hash_val: str) -> str:\n",
        "    shared_artifacts['c2_domain'] = 'apt-hidden-c2.net'\n",
        "    return \"SUCCESS: Extracted C2 domain and saved to shared_artifacts.\"\n",
        "\n",
        "def deploy_firewall_rule_hybrid(ip: str, domain: str) -> str:\n",
        "    return f\"SUCCESS: Firewall rule deployed blocking IP {ip} and domain {domain}.\"\n",
        "\n",
        "# 2. Initialize Swarm in SILENT MODE (no logs, purely latent background processing)\n",
        "stealth_swarm = CidaSwarm(\n",
        "    wrapper=wrapper,\n",
        "    llm=llm,\n",
        "    tokenizer=tokenizer,\n",
        "    agent_configs={\n",
        "        \"network_analyst\": {\n",
        "            \"role\": \"Analyze PCAP. Use analyze_pcap. Save findings to shared_artifacts.\",\n",
        "            \"tools\": {\"analyze_pcap\": analyze_pcap_hybrid},\n",
        "            \"silent_mode\": True  # <--- Hides thoughts, processes fully in latent space\n",
        "        },\n",
        "        \"reverse_engineer\": {\n",
        "            \"role\": f\"Decompile malware using hash from shared_artifacts. Use decompile_malware.\",\n",
        "            \"tools\": {\"decompile_malware\": decompile_malware_hybrid},\n",
        "            \"silent_mode\": True\n",
        "        },\n",
        "        \"soc_commander\": {\n",
        "            \"role\": f\"Deploy firewall using IP and Domain from shared_artifacts. Use deploy_firewall_rule.\",\n",
        "            \"tools\": {\"deploy_firewall_rule\": deploy_firewall_rule_hybrid},\n",
        "            \"silent_mode\": True\n",
        "        }\n",
        "    },\n",
        "    callbacks=[] # No logging callbacks\n",
        ")\n",
        "\n",
        "STEALTH_TASK = (\n",
        "    \"Task: Investigate 'traffic.pcap'. Find IP/Hash, decompile for C2 domain, and deploy firewall. \"\n",
        "    f\"Current known artifacts: {shared_artifacts}\"\n",
        ")\n",
        "\n",
        "print(\"🚀 Launching Stealth Swarm (Processing purely in Latent Space...)\")\n",
        "t0 = time.time()\n",
        "# Run pipeline\n",
        "stealth_results = stealth_swarm.run(STEALTH_TASK, max_turns=4, pipeline=[\"network_analyst\", \"reverse_engineer\", \"soc_commander\"])\n",
        "t1 = time.time()\n",
        "\n",
        "print(\"\\n=========================================================\")\n",
        "print(f\"✅ DONE IN {round(t1-t0, 2)}s!\")\n",
        "print(\"🎯 FINAL OUTPUT (SOC Commander):\")\n",
        "print(stealth_results['soc_commander'])\n",
        "print(\"=========================================================\")\n"
    ]
}

nb["cells"].append(markdown_cell)
nb["cells"].append(code_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Appended stealth mode cells to notebook.")
