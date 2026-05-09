# CIDA-Plugin v2: Universal Evidence-Grounded Multi-Agent Deliberation Layer

> *"What if a neural network could argue with itself — and reach a better answer?"*

**CIDA-Plugin** is a drop-in architectural layer that can be added on top of **any** pre-trained Transformer encoder (BERT, DistilBERT, RoBERTa, etc.). Instead of a simple Linear Head, CIDA-Plugin introduces a **Multi-Agent Deliberation Protocol**.

It forces the model to form independent perspectives (agents), exchange arguments, and reach a consensus weighted by each agent's uncertainty.

**Result:** Massive reductions in Expected Calibration Error (ECE) and better-reasoned predictions without relying on post-hoc calibration methods like Temperature Scaling.

##  What's New in v2
- **Transformer Agent Updater**: Agents now use Cross-Attention to selectively focus on counterarguments, replacing the rigid GRU cell.
- **EMA Reliability Tracker**: Agent reliability ($\rho$) is tracked automatically during training via Exponential Moving Average based on consensus agreement.
- **Learnable Disagreement Schedule**: The system autonomously learns when agents should argue and when they should converge during the deliberation rounds.
- **Hugging Face Hub Integration**: Full support for `save_pretrained()` and `from_pretrained()`.

---

##  Performance Benchmark (SST-2)

Results comparing a standard Linear Head vs. **CIDA-Plugin v2** across 2000 samples and 3 epochs:

| Encoder | Head Type | Accuracy | ECE (Calibration) | ΔECE |
| :--- | :--- | :---: | :---: | :---: |
| DistilBERT | Linear | 81.00% | 0.1023 | - |
| **DistilBERT** | **CIDA-Plugin** | **83.00%** | **0.0431** | **-58%** ↓ |
| BERT-base | Linear | 83.20% | 0.0715 | - |
| **BERT-base** | **CIDA-Plugin** | **84.60%** | **0.0297** | **-58%** ↓ |

> **Key takeaway:** CIDA-Plugin consistently reduces calibration error by over **50%** while simultaneously providing a **1.5% - 2.0% accuracy boost** through its multi-agent deliberation process.

---

## ⚡ Confidence-based Early Stopping

CIDA-Plugin supports dynamic deliberation rounds. On simple examples, the model can halt early to save computation:

| Confidence Threshold | Avg. Rounds | Accuracy | ECE |
| :---: | :---: | :---: | :---: |
| None (Full) | 4.0 | 83.4% | 0.034 |
| 0.85 | ~2.8 | 82.9% | 0.039 |

*Model achieves ~30% faster inference on simple inputs with minimal accuracy trade-offs.*

---

## 📦 Installation
```bash
pip install .
# or if uploaded to PyPI:
# pip install cida-plugin
```

---

## ⚡ Quickstart

CIDA-Plugin is designed to be as easy to use as a standard Hugging Face model.

### 1. Training with any Encoder
```python
import torch
from transformers import AutoModel
from cida_plugin import CIDAPlugin, CIDAPluginConfig, OmegaLossSystem

# 1. Load any frozen encoder
encoder = AutoModel.from_pretrained("distilbert-base-uncased")
d_model = encoder.config.hidden_size

# 2. Initialize the plugin
config = CIDAPluginConfig(
    d_input=d_model,     # Match encoder output dimension
    d_hidden=128,        # Internal plugin dimension
    num_classes=2,
    max_rounds=3,        # Deliberation rounds
    early_stop_threshold=0.90
)
plugin = CIDAPlugin(config)

# 3. Forward pass
input_ids = torch.randint(0, 1000, (4, 128))
out = encoder(input_ids)
pooled = out.last_hidden_state[:, 0, :]

# The plugin takes the pooled representation and deliberates
plugin_out = plugin(pooled, seq_output=out.last_hidden_state)

logits = plugin_out["p_final"] # (Batch, Num_Classes)
```

### 2. Saving and Loading (Hugging Face style)
```python
# Save to disk
plugin.save_pretrained("./my-cida-plugin")

# Load from disk or Hugging Face Hub
loaded_plugin = CIDAPlugin.from_pretrained("./my-cida-plugin")
# or
# loaded_plugin = CIDAPlugin.from_pretrained("Kairatzh/cida-plugin-distilbert")
```

---

##  Architecture Overview

The plugin takes the output of your encoder and processes it through the following steps:

1. **Input Projection:** Maps the arbitrary `d_input` of the encoder to the internal `d_hidden` of the agents.
2. **Agent Initialization:** Creates $M$ distinct agents, each with a learned role embedding.
3. **Deliberation Loop ($R$ rounds):**
   - **Evidence Extraction:** Agents attend to the input sequence to gather distinct evidence.
   - **Message Formulation:** Agents compress their beliefs and evidence into theses.
   - **Cross-Attention Communication:** Agents listen to others, explicitly weighting disagreement.
   - **Gated Update:** Agents update their internal Dirichlet belief states.
4. **Consensus Aggregation:** A final Product-of-Experts (PoE) aggregation weighted by inverse uncertainty.

---

##  Evaluation Metrics

Run the built-in benchmark to compare a standard Linear Head vs. CIDA-Plugin across different encoders:
```bash
python experiments/run_plugin_benchmark.py
```

*Preliminary results show CIDA-Plugin reducing ECE by up to 90% compared to a standard Linear Head, acting as an architectural regularizer for calibration.*

---

##  Citation
```bibtex
@article{zhaksylykov2026cida,
  title     = {CIDA: Collective Intelligence via Deliberation and Aggregation},
  author    = {Zhaksylykov, Kairat},
  year      = {2026},
  month     = {May},
  note      = {K.Zhubanov Regional University}
}
```