import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from cida_plugin.inference.loader import CIDAWrapper
from cida_plugin.inference.swarm import CidaSwarm, SwarmCallbackHandler
from cida_plugin.inference.callbacks import BaseCallbackHandler

# ── CTF SIMULATED ENVIRONMENT ──

state = {"pcap_fails": 0}

def analyze_pcap(file: str) -> str:
    """Network Analyst Tool: parses pcap to find threats."""
    state["pcap_fails"] += 1
    if state["pcap_fails"] <= 2:
        return "Error: Wireshark dissector crashed on malformed packet (Segmentation fault)."
    
    return "SUCCESS: Found outbound connection to IP 45.33.32.156. Payload contains executable hash 'a9f8b2c4e5d6'."

def decompile_malware(hash_val: str) -> str:
    """Reverse Engineer Tool: extracts strings from malware hash."""
    if hash_val != "a9f8b2c4e5d6":
        return "Error: File hash not found in sandbox."
    
    return "SUCCESS: Binary decompiled. Obfuscated strings decrypted. C2 Domain extracted: 'apt-hidden-c2.net'."

def deploy_firewall_rule(ip: str, domain: str) -> str:
    """SOC Commander Tool: blocks threat."""
    return f"SUCCESS: YARA/Firewall rule deployed blocking IP {ip} and domain {domain}. Network secured."


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
    def __init__(self):
        super().__init__()
    
    def on_handoff(self, from_agent, to_agent, reason):
        print(f"\n🤝 [LATENT HANDOFF] {from_agent} passes context vectors to {to_agent} (0 text tokens used)")

# ── MAIN DEMO ──

def run_ctf():
    print("=========================================================")
    print("🕵️‍♂️ CIDA AUTONOMOUS CYBERSECURITY CTF: APT HUNTING")
    print("=========================================================\n")
    print("Loading 7B Model + CIDA Plugin (Quantized 4-bit)...")
    
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

    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    wrapper = CIDAWrapper.from_pretrained(llm, cida_repo="kirtzh/Kairos-7B-CIDA-v1.0")
    
    print("\n✅ Setup complete. Initializing Swarm...")
    
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
    
    print("\n🚀 STARTING CTF PIPELINE...")
    print(f"Task: {CTF_TASK}\n")
    
    results = swarm.run(CTF_TASK, max_turns=4, pipeline=["network_analyst", "reverse_engineer", "soc_commander"])
    
    print("\n=========================================================")
    print("🏆 CTF COMPLETED. FINAL REPORT:")
    print("=========================================================")
    print(f"SOC Commander Final Output:\n{results['soc_commander']}")
    print("=========================================================")

if __name__ == "__main__":
    run_ctf()
