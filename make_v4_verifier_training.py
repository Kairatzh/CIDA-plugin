import json

def cc(src, cid):
    return {"cell_type":"code","execution_count":None,"id":cid,"metadata":{},"outputs":[],"source":src}
def mc(src, cid):
    return {"cell_type":"markdown","id":cid,"metadata":{},"source":src}

cells = []

# TITLE
cells.append(mc(
"# 🛡️ CIDA v4 Plan B: Anti-Hallucination Verifier\n\n"
"**Concept Pivot:** Instead of generating Chain-of-Thought planning tokens, CIDA acts as an independent auditor to verify LLM responses and detect hallucinations.\n\n"
"**Mechanism:**\n"
"1. Input: Prompt + LLM Response -> Frozen Embedding Model.\n"
"2. CIDA Agents: Prosecutor, Defender, Skeptic, and Integrator debate the response validity.\n"
"3. Output: Factual validation score (0.0 to 1.0) and uncertainty metrics derived from agent disagreement.",
"title"))

# CELL 0: SETUP
cells.append(mc("## 0 · Setup", "s0"))
cells.append(cc(
"!pip install -q torch transformers datasets accelerate torchdiffeq bitsandbytes\n"
"import subprocess, sys, os\n"
"if not os.path.exists('CIDA-plugin'):\n"
"    subprocess.run(['git','clone','--depth','1',\n"
"                   'https://github.com/Kairatzh/CIDA-plugin.git'], check=True)\n"
"sys.path.insert(0, os.path.abspath('CIDA-plugin'))\n"
"from google.colab import drive\n"
"drive.mount('/content/drive')\n"
"print('Setup complete.')\n",
"c0"))

# CELL 1: DATASET
cells.append(mc("## 1 · Dataset (HaluEval - QA & Summarization Hallucinations)", "s1"))
cells.append(cc(
"from datasets import load_dataset\n"
"import random\n"
"random.seed(42)\n\n"
"print('Loading HaluEval dataset...')\n"
"try:\n"
"    # Load QA subset of HaluEval\n"
"    dataset = load_dataset('jiazhengli/HaluEval', 'qa', split='data')\n"
"    all_samples = []\n"
"    for ex in dataset:\n"
"        q = ex['question']\n"
"        # Positive sample (Correct)\n"
"        all_samples.append({'question': q, 'response': ex['right_answer'], 'label': 1})\n"
"        # Negative sample (Hallucinated)\n"
"        all_samples.append({'question': q, 'response': ex['hallucinated_answer'], 'label': 0})\n"
"except Exception as e:\n"
"    print(f'HaluEval failed: {e}, falling back to synthetic verification data')\n"
"    all_samples = [\n"
"        {'question': 'What is the capital of France?', 'response': 'Paris is the capital of France.', 'label': 1},\n"
"        {'question': 'What is the capital of France?', 'response': 'London is the capital of France.', 'label': 0},\n"
"    ] * 500\n\n"
"random.shuffle(all_samples)\n"
"all_samples = all_samples[:20000]  # Take 20K samples (10K QA pairs)\n"
"split = int(len(all_samples) * 0.9)\n"
"train_data, val_data = all_samples[:split], all_samples[split:]\n"
"print(f'Train size: {len(train_data)} | Val size: {len(val_data)}')\n",
"c1"))

# CELL 2: ENCODER
cells.append(mc("## 2 · Frozen Sentence Encoder (bge-large-en-v1.5)", "s2"))
cells.append(cc(
"from transformers import AutoTokenizer, AutoModel\n"
"import torch\n\n"
"MODEL_NAME = 'BAAI/bge-large-en-v1.5'\n"
"tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
"encoder = AutoModel.from_pretrained(MODEL_NAME).to('cuda')\n"
"for p in encoder.parameters(): p.requires_grad = False\n"
"encoder.eval()\n\n"
"D_EMBED = encoder.config.hidden_size\n"
"print(f'Encoder loaded. Embedding dimension: {D_EMBED}')\n",
"c2"))

# CELL 3: CACHING
cells.append(mc("## 3 · Cache Encoded Representations", "s3"))
cells.append(cc(
"from tqdm.auto import tqdm\n"
"import os\n\n"
"def cache_verifications(data, name):\n"
"    os.makedirs('cache_verifier', exist_ok=True)\n"
"    cached = []\n"
"    for sample in tqdm(data, desc=f'Caching {name}'):\n"
"        # Combine question and response into auditor format\n"
"        text = f\"Question: {sample['question']}\\nResponse: {sample['response']}\"\n"
"        enc = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')\n"
"        \n"
"        with torch.no_grad():\n"
"            out = encoder(**enc)\n"
"            # Mean pooling\n"
"            emb = out.last_hidden_state.mean(dim=1)[0].cpu().half()\n"
"            \n"
"        cached.append({'embedding': emb, 'label': sample['label']})\n"
"        \n"
"    path = f'cache_verifier/{name}_cached.pt'\n"
"    torch.save(cached, path)\n"
"    return path\n\n"
"print('Caching Train embeddings...')\n"
"train_path = cache_verifications(train_data, 'train')\n"
"print('Caching Val embeddings...')\n"
"val_path = cache_verifications(val_data, 'val')\n",
"c3"))

# CELL 4: CIDA PIVOT CONFIG
cells.append(mc("## 4 · CIDA Verifier Configuration (Binary Classifier)", "s4"))
cells.append(cc(
"from cida_plugin.config import CIDAPluginConfig\n"
"from cida_plugin.core import CIDAPlugin\n"
"import torch.nn as nn\n\n"
"cfg = CIDAPluginConfig(\n"
"    d_input=D_EMBED,\n"
"    d_hidden=512,\n"
"    d_message=512,\n"
"    num_agents=4,\n"
"    num_classes=2,  # 0 = Hallucination, 1 = Factual\n"
"    max_rounds=4,\n"
"    num_attn_heads=8,\n"
"    use_perspective_projector=True,\n"
"    anonymous_messages=True,\n"
"    sparse_comm_k=3,\n"
"    use_dynamic_trust=True\n"
")\n\n"
"cida = CIDAPlugin(cfg).to('cuda').float()\n"
"print('CIDA Verifier initialized successfully.')\n",
"c4"))

# CELL 5: TRAINING
cells.append(mc("## 5 · Training CIDA Verifier", "s5"))
src5 = (
"import torch.nn.functional as F\n"
"from torch.utils.data import DataLoader, Dataset\n\n"
"class EmbedDataset(Dataset):\n"
"    def __init__(self, path):\n"
"        self.data = torch.load(path)\n"
"    def __len__(self):\n"
"        return len(self.data)\n"
"    def __getitem__(self, idx):\n"
"        return self.data[idx]['embedding'].float(), self.data[idx]['label']\n\n"
"train_loader = DataLoader(EmbedDataset(train_path), batch_size=64, shuffle=True)\n"
"val_loader = DataLoader(EmbedDataset(val_path), batch_size=64, shuffle=False)\n\n"
"optimizer = torch.optim.AdamW(cida.parameters(), lr=1e-3, weight_decay=0.01)\n"
"epochs = 5\n\n"
"for epoch in range(epochs):\n"
"    cida.train()\n"
"    total_loss, correct, total = 0, 0, 0\n"
"    for embeds, labels in train_loader:\n"
"        embeds, labels = embeds.to('cuda'), labels.to('cuda')\n"
"        optimizer.zero_grad()\n"
"        \n"
"        out = cida(pooled=embeds)\n"
"        logits = out['p_final']  # Consensus probability distribution\n"
"        loss_ce = F.cross_entropy(logits, labels)\n"
"        loss_orth = out['L_orth']\n"
"        \n"
"        loss = loss_ce + 0.1 * loss_orth\n"
"        loss.backward()\n"
"        optimizer.step()\n"
"        \n"
"        total_loss += loss.item() * embeds.size(0)\n"
"        preds = logits.argmax(dim=-1)\n"
"        correct += (preds == labels).sum().item()\n"
"        total += embeds.size(0)\n"
"        \n"
"    train_acc = correct / total\n"
"    train_loss = total_loss / total\n"
"    \n"
"    # Validation\n"
"    cida.eval()\n"
"    val_loss, val_correct, val_total = 0, 0, 0\n"
"    with torch.no_grad():\n"
"        for embeds, labels in val_loader:\n"
"            embeds, labels = embeds.to('cuda'), labels.to('cuda')\n"
"            out = cida(pooled=embeds)\n"
"            logits = out['p_final']\n"
"            val_loss += F.cross_entropy(logits, labels).item() * embeds.size(0)\n"
"            preds = logits.argmax(dim=-1)\n"
"            val_correct += (preds == labels).sum().item()\n"
"            val_total += embeds.size(0)\n"
"            \n"
"    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_correct/val_total*100:.2f}%')\n\n"
"# Save Weights\n"
"save_dir = 'cida_verifier_weights'\n"
"os.makedirs(save_dir, exist_ok=True)\n"
"cfg.save_pretrained(save_dir)\n"
"torch.save(cida.state_dict(), f'{save_dir}/pytorch_model.bin')\n"
"print(f'Verifier weights saved to {save_dir}/')\n"
)
cells.append(cc(src5, "s5"))

# CELL 6: BACKUP
cells.append(mc("## 6 · Backup to Google Drive", "m6"))
cells.append(cc(
"import shutil\n"
"dst = '/content/drive/MyDrive/cida_verifier_weights'\n"
"os.makedirs(dst, exist_ok=True)\n"
"for f in os.listdir(save_dir):\n"
"    shutil.copy2(f'{save_dir}/{f}', dst)\n"
"print(f'Backed up to Google Drive: {dst}')\n",
"c6"))

# BUILD NOTEBOOK
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "colab": {"provenance":[], "gpuType":"T4"},
        "accelerator": "GPU"
    },
    "cells": cells,
}
with open("Kairos_CIDA_v4_Verifier_Training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook generated: Kairos_CIDA_v4_Verifier_Training.ipynb")
