import json

with open('Kairos_CIDA_CTF_Swarm_Demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_content = '''# =========================================================
# 🏆 AGENTBENCH MINI: OS TERMINAL CHALLENGE (REAL BASH)
# =========================================================
# This cell uses the ALREADY LOADED LLM weights (no redownloading).
# It tests CIDA's agentic ability to autonomously navigate a Linux filesystem.

import os
import subprocess
from cida_plugin.inference.latent_agent import LatentAgent

# 1. Setup a realistic filesystem challenge for the Agent
env_dir = "agent_test_env"
os.makedirs(f"{env_dir}/hidden_configs", exist_ok=True)

with open(f"{env_dir}/README.md", "w") as f:
    f.write("System Info: The database credentials have been moved to a hidden file inside 'hidden_configs' directory.")

with open(f"{env_dir}/hidden_configs/.db_pass.txt", "w") as f:
    f.write("DB_PASSWORD=cida_super_agent_2026")

# 2. Give the Agent a REAL Bash Terminal Tool
def execute_bash(command: str) -> str:
    """Executes a bash command in the terminal. ALWAYS use this to run 'ls -la', 'cat', etc."""
    try:
        # Run command inside our sandbox dir
        result = subprocess.run(command, shell=True, cwd=env_dir, text=True, capture_output=True, timeout=5)
        if result.returncode == 0:
            out = result.stdout.strip()
            return f"Status 200: {out if out else 'Success (No output)'}"
        else:
            return f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Error: {str(e)}"

# 3. Initialize the OS Agent (Reusing the already loaded weights in RAM!)
os_agent = LatentAgent(
    role="""You are an autonomous Linux DevOps agent. 
You must explore the filesystem using 'execute_bash'.
Start by running 'ls -la' to see what files exist. Then read files to find clues.
""",
    tools=[execute_bash],
    llm=llm,
    tokenizer=tokenizer,
    wrapper=wrapper
)

print("🚀 Launching AgentBench OS Challenge (Autonomous Shell Navigation)...")

task = "Explore the system, read the README, find the hidden database password, and tell me what it is. Use the execute_bash tool to run bash commands."

# Run the agent! (It will loop, make mistakes, do rollbacks, and navigate the filesystem)
final_answer = os_agent.run(task, max_turns=8)

print("\\n=========================================================")
print(f"🎯 AGENTBENCH RESULT:\\n{final_answer}")
print("=========================================================")
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

print("AgentBench Cell Appended via JSON!")
