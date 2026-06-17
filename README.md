# CIDA-Plugin v3: Universal Evidence-Grounded Multi-Agent Deliberation Layer

> *"What if a neural network could argue with itself — and reach a better answer?"*

**CIDA-Plugin** is a drop-in architectural layer that can be added on top of **any** pre-trained Transformer encoder (BERT, DistilBERT, RoBERTa, etc.) or Vision Backbone (ResNet, DenseNet, etc.). Instead of a simple Linear Head, CIDA-Plugin introduces a **Multi-Agent Deliberation Protocol**.

It forces the model to form independent perspectives (agents), exchange arguments, and reach a consensus weighted by each agent's uncertainty.

**Result:** Massive reductions in Expected Calibration Error (ECE), robust uncertainty estimation, and better-reasoned predictions without relying on post-hoc calibration methods like Temperature Scaling.

---

## ⚡ What's New in v3 (Simplified Architecture)
We transitioned from a highly complex, statistical embedding-based formulation (v2) to a streamlined, theoretically grounded Bayesian-inspired architecture (v3):

*   **Bayesian Role Priors**: Removed the hyperparameter-heavy `debate_loss`, role-specific serialization losses, and learnable `RoleEmbeddings`. Instead, agents are assigned fixed, mathematically guaranteed prior beliefs (Prosecutor, Defender, Skeptic, Integrator). This enforces structural disagreement at all times, preventing agent representation collapse.
*   **Weighted Mean Consensus**: Replaced the Product of Experts (PoE) aggregator, which incorrectly assumed agent independence and amplified shared bias (causing overconfidence). We now aggregate beliefs via a Weighted Mean, which is mathematically sound for correlated variables.
*   **Disagreement-as-Uncertainty**: Replaced the circular and unstable `ReliabilityTracker` with an observable, non-learned uncertainty quantification based on the variance (standard deviation) between expert beliefs: $U = f(\text{std}(b))$.
*   **3-Component Loss System**: Reduced the loss system from 11 components to 3 core components: Task Loss (CE/BCE), Calibration Loss (Brier Score), and Anti-Collapse Loss (used only as a safety net).

---

## ⚡ Comparison: Legacy vs. v3 Simplified

| Feature | Legacy CIDA (v2/Omega) | CIDA v3 (Simplified) |
| :--- | :--- | :--- |
| **Agent Diversity** | Additive embeddings + `debate_loss` | Fixed Role Priors + `anti_collapse_loss` |
| **Consensus Mech** | Product of Experts (PoE) | Weighted Mean (Correlation-Aware) |
| **Reliability/Uncertainty** | Learned EMA Reliability Tracker | Observed Disagreement ($std(b)$) |
| **Loss System** | 11 components (hard to tune) | 3 components (extremely stable) |
| **Hyperparameters** | 11 (lambda schedules, temp, etc.) | 2 (lambda_cal, lambda_ac) |

---

## 📦 Installation

```bash
pip install .
```

---

## ⚡ Quickstart

CIDA-Plugin is designed to be as easy to use as a standard Hugging Face model.

### 1. Training with any Encoder
```python
import torch
from transformers import AutoModel
from cida_plugin import CIDAPlugin, CIDAPluginConfig, CIDALoss

# 1. Load any frozen encoder
encoder = AutoModel.from_pretrained("distilbert-base-uncased")
d_model = encoder.config.hidden_size

# 2. Initialize the plugin config
config = CIDAPluginConfig(
    d_input=d_model,     # Match encoder output dimension
    d_hidden=128,        # Internal plugin dimension
    num_classes=2,
    max_rounds=3,        # Deliberation rounds
    early_stop_threshold=0.90
)
plugin = CIDAPlugin(config)
loss_fn = CIDALoss(lambda_cal=0.4, lambda_ac=0.2)

# 3. Forward pass
input_ids = torch.randint(0, 1000, (4, 128))
out = encoder(input_ids)
pooled = out.last_hidden_state[:, 0, :]

# The plugin takes the pooled representation and deliberates
plugin_out = plugin(pooled, seq_output=out.last_hidden_state)

logits = plugin_out["p_final"] # (Batch, Num_Classes)
loss, loss_components = loss_fn(logits, targets, plugin_out["b_all"])
```

### 2. Saving and Loading (Hugging Face style)
```python
# Save to disk
plugin.save_pretrained("./my-cida-plugin")

# Load from disk
loaded_plugin = CIDAPlugin.from_pretrained("./my-cida-plugin")
```

---

## 🛠️ Architecture Overview

The plugin takes the output of your encoder and processes it through the following steps:

1. **Input Projection:** Maps the arbitrary `d_input` of the encoder to the internal `d_hidden` of the agents.
2. **Agent Initialization:** Creates $M$ distinct agents initialized with the pooled representation.
3. **Deliberation Loop ($R$ rounds):**
   - **Evidence Extraction:** Agents attend to the input sequence to gather distinct evidence.
   - **Message Formulation:** Agents compress their beliefs and evidence into theses.
   - **Cross-Attention Communication:** Agents listen to others, explicitly weighting disagreement.
   - **Gated Update:** Agents update their internal states.
   - **Role Prior Blending:** Enforces structural roles (Prosecutor, Defender, Skeptic, Integrator) on updated beliefs.
4. **Consensus Aggregation:** A final Weighted Mean consensus calculation.

---

## 🧪 Liquid Dynamics & TTT (v5 Additions)
*   **Liquid Neural ODE**: Set `use_liquid_dynamics=True` to replace the discrete iteration loop with continuous-time deliberation solver ($ds/dt = -s/\tau(x) + F(s,r,e)$).
*   **Test-Time Training (TTT)**: Set `use_ttt=True` to allow agents to adapt their weights to a specific input using self-supervised masked state reconstruction steps before answering.

---

## 🎮 Interactive Demo (Hugging Face Spaces)

See how 4 agents deliberate before answering — with agent vote charts and uncertainty gauges.

```bash
# Train demo checkpoints (~5 min on CPU)
python demo/train_demo.py

# Launch Gradio locally
python demo/app.py
```

Deploy to [Hugging Face Spaces](https://huggingface.co/spaces): create a Gradio Space pointing to the `demo/` folder (see `demo/README.md`).

---

## ⚡ Running Tests
To verify the installation and execution:
```bash
pytest tests/ -v
```