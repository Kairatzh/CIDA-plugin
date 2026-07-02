# Kairos-CIDA-1.3B-v1.0

This repository contains the weights for the **CIDA v2 (System 2)** adapter trained on top of the frozen `Qwen1.5-1.8B-Chat` backbone model.
It utilizes **dynamic round-jittering** (2 to 5 rounds) during training to enable **Test-Time Compute Scaling** and a **Latent Rollback** backtracking mechanism at inference.

---

##  Comprehensive 4-Way Evaluation Scoreboard

The following table compares the performance of CIDA v2 against the base models and CIDA v1 on GSM8K (math), ARC-Challenge (science), Logic Traps (adversarial negation), and our OOD Creativity Suite:

| Model Configuration | GSM8K Acc | ARC-Ch. Acc | Logic Traps Acc | Creativity Score | Average Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Base Qwen 1.8B (System 1) | 10.00% | 70.00% | 10.00% | 0.00% | **22.50%** |
| 2. **CIDA v2 (1.8B) [Our Newly Trained]** | 16.67% | 76.67% | 93.33% | 100.00% | **71.67%** |
| 3. Base Qwen 7B (System 1) | 3.33% | 60.00% | 23.33% | 75.00% | **40.41%** |
| 4. **Kairos 7B (HuggingFace) [CIDA v1]** | 33.33% | 86.67% | 30.00% | 100.00% | **62.50%** |

---

##  Quickstart Tutorial

To load and run this model with the CIDA Latent Deliberation plugin:

### 1. Installation
Clone and install the CIDA-plugin library:
```bash
git clone https://github.com/Kairatzh/CIDA-plugin.git
cd CIDA-plugin
pip install -e .
```

### 2. Loading the Model and CIDA Adapter
Use the following Python snippet to initialize the model and load the CIDA weights from Hugging Face:

```python
import torch
from cida_plugin import CidaConfig, CidaModel, PlanProjector
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Qwen base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-1.8B-Chat", 
    torch_dtype=torch.float16
).to(DEVICE)

# 2. Download CIDA adapter weights from Hugging Face
from huggingface_hub import hf_hub_download

model_repo = "kirtzh/Kairos-CIDA-1.3B-v1.0"
config_path = hf_hub_download(repo_id=model_repo, filename="config.json")
weights_path = hf_hub_download(repo_id=model_repo, filename="pytorch_model.bin")
proj_path = hf_hub_download(repo_id=model_repo, filename="plan_projector.pt")

# 3. Initialize CIDA system 2 layers
config = CidaConfig.from_json_file(config_path)
cida_model = CidaModel(config).to(DEVICE)
cida_model.load_state_dict(torch.load(weights_path))

# 4. Load projection layer
proj = PlanProjector(num_plan_tokens=8, hidden_dim=4096).to(DEVICE)
proj.load_state_dict(torch.load(proj_path))

print("Successfully loaded CIDA v2 weights!")
```
