"""Convert train_foundation.py into a Jupyter notebook."""
import json, re, os

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":src})
def code(src): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src})

md(["# CIDA Role Pre-Training v2 (Foundation Edition)\n",
    "This notebook contains the complete pipeline for pretraining **RoleCIDA** to foundation-grade specialization using `microsoft/deberta-v3-base` (768 dimensions) and official Stanford/NYU NLI datasets (bypassing HF datasets API completely).\n",
    "\n",
    "## Key Upgrades\n",
    "1. **DeBERTa-v3-base**: Strong encoder for logical/semantic relationships.\n",
    "2. **Direct Academic Downloads**: Downloads SNLI from Stanford and MultiNLI from NYU via direct ZIP URLs to bypass HuggingFace datasets library issues.\n",
    "3. **Connected Gradient Graph**: Fixed backprop flow through anchors.\n",
    "4. **CIDAInferenceEngine**: High-level wrapper class to run predictions and get structured outputs immediately after training.\n"])

md(["## 0. Setup and Dependencies"])
code(["!pip install torchdiffeq transformers sentence-transformers scikit-learn -q\n",
      "import os\n",
      "if not os.path.exists('/content/CIDA-plugin'):\n",
      "    !git clone https://github.com/Kairatzh/CIDA-plugin.git /content/CIDA-plugin\n",
      "%cd /content/CIDA-plugin\n",
      "!git pull"])

# Read source files
base = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base, "role_identity.py"), "r") as f:
    ri = f.read()
with open(os.path.join(base, "model_and_losses.py"), "r") as f:
    ml = f.read()
with open(os.path.join(base, "train_foundation.py"), "r") as f:
    tf = f.read()

md(["## 1. Role Identity Layer v2"])
code(["!mkdir -p cida_role_pretrain_v2"])
code(["%%writefile cida_role_pretrain_v2/role_identity.py\n" + ri])

md(["## 2. Model & Losses v2"])
code(["%%writefile cida_role_pretrain_v2/model_and_losses.py\n" + ml])

md(["## 3. Foundation Training Script"])
code(["%%writefile cida_role_pretrain_v2/train_foundation.py\n" + tf])

md(["## 4. Run Foundation Pretraining"])
code(["!python cida_role_pretrain_v2/train_foundation.py"])

md(["## 5. Production Inference Demo\n",
    "Use the pre-trained foundation CIDA model directly on custom texts:"])
code(["# Import engine from the script\n",
      "import sys\n",
      "sys.path.insert(0, './cida_role_pretrain_v2')\n",
      "from train_foundation import CIDAInferenceEngine\n",
      "\n",
      "# Initialize engine with the best checkpoint\n",
      "engine = CIDAInferenceEngine('ckpt_role_foundation/best_foundation.pt')\n",
      "\n",
      "# Real-world diagnostic/logical test texts\n",
      "test_cases = [\n",
      "    'The patient shows clear ground glass opacities in both lungs. | This indicates a severe case of viral pneumonia.',\n",
      "    'The patient is coughing. | The patient definitely has tuberculosis.',\n",
      "]\n",
      "\n",
      "# Run deliberation\n",
      "analyses = engine.analyze(test_cases)\n",
      "\n",
      "for i, res in enumerate(analyses):\n",
      "    print(f'\\nCase {i+1}: \"{res[\"text\"]}\"')\n",
      "    print(f'  - Predicted Class: {res[\"predicted_class\"]} (0=Entailment, 1=Neutral, 2=Contradiction)')\n",
      "    print(f'  - Disagreement Score: {res[\"disagreement\"]:.4f}')\n",
      "    print(f'  - Uncertainty: {res[\"uncertainty\"]:.4f}')\n",
      "    print('  - Deliberation Trace (Integrator Consensus):')\n",
      "    for r in res['deliberation_trace']:\n",
      "        print(f'    Round {r[\"round\"]}: P={r[\"prosecutor_belief\"]}, D={r[\"defender_belief\"]}, S={r[\"skeptic_belief\"]}, I={r[\"integrator_belief\"]}')\n"])

nb = {"nbformat":4,"nbformat_minor":0,
      "metadata":{"colab":{"provenance":[],"gpuType":"A100"},
                   "kernelspec":{"name":"python3","display_name":"Python 3"},
                   "accelerator":"GPU"},
      "cells": cells}

out_path = os.path.join(base, "..", "CIDA_RolePretrain_Foundation.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Foundation Notebook saved to {out_path}")
