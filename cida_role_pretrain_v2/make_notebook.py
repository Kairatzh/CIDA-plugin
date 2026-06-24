"""Convert train_v2.py into a Jupyter notebook."""
import json, re, os

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":src})
def code(src): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src})

md(["# CIDA Role Pre-Training v2\n",
    "## Fixes + Improvements over v1\n",
    "1. **FIX**: RoleIdentityLayer now feeds INTO CIDAPlugin (connected gradient graph)\n",
    "2. **FIX**: Anchor Diversity Loss on RAW anchors (no input contamination)\n",
    "3. **FIX**: KL divergence target normalization\n",
    "4. **NEW**: Role-Conditioned Messaging (role embed at every round)\n",
    "5. **NEW**: Contrastive Message Loss (InfoNCE — agents must be identifiable)\n",
    "6. **NEW**: Role Time Dynamics (Skeptic=viscous, Integrator=fluid)\n",
    "7. **NEW**: EMA weight averaging for stable eval\n",
    "8. **NEW**: Gradient flow verification\n",
    "9. **NEW**: t-SNE visualization of message clusters\n"])

md(["## 0. Setup"])
code(["!pip install torchdiffeq 'datasets==2.19.1' 'huggingface-hub==0.23.4' transformers sentence-transformers scikit-learn -q\n",
      "import os\n",
      "if not os.path.exists('/content/CIDA-plugin'):\n",
      "    !git clone https://github.com/Kairatzh/CIDA-plugin.git /content/CIDA-plugin\n",
      "%cd /content/CIDA-plugin\n",
      "!git pull"])

# Read the two source files
base = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base, "role_identity.py"), "r") as f:
    ri = f.read()
with open(os.path.join(base, "model_and_losses.py"), "r") as f:
    ml = f.read()
with open(os.path.join(base, "train_v2.py"), "r") as f:
    tv = f.read()

md(["## 1. Role Identity Layer v2 (Fixed)"])
code(["!mkdir -p cida_role_pretrain_v2"])
code(["%%writefile cida_role_pretrain_v2/role_identity.py\n" + ri])

md(["## 2. Model + Losses v2 (Fixed)"])
code(["%%writefile cida_role_pretrain_v2/model_and_losses.py\n" + ml])

md(["## 3. Training"])
# Split train_v2.py into logical cells
sections = tv.split("\n# ── ")
for i, sec in enumerate(sections):
    if i == 0:
        code([sec])
    else:
        header = sec.split("\n")[0].strip().rstrip("─").strip()
        md([f"### {header}"])
        code(["# ── " + sec])

nb = {"nbformat":4,"nbformat_minor":0,
      "metadata":{"colab":{"provenance":[],"gpuType":"A100"},
                   "kernelspec":{"name":"python3","display_name":"Python 3"},
                   "accelerator":"GPU"},
      "cells": cells}

out_path = os.path.join(base, "..", "CIDA_RolePretrain_v2.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook saved to {out_path}")
