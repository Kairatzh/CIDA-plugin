---
title: CIDA-Plugin Demo
emoji: 🤝
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multi-agent deliberation — see 4 agents debate before answering
---

# CIDA-Plugin: Interactive Multi-Agent Deliberation Demo

**CIDA** replaces a single linear classifier head with **4 agents** that deliberate before reaching a consensus.

## What you'll see

1. **Final verdict** — digit classification or sentiment label
2. **Agent vote chart** — Prosecutor, Defender, Skeptic, Integrator each vote independently
3. **Uncertainty gauge** — turns red when agents disagree → "human review needed"

## Tabs

- **MNIST** — upload or draw a handwritten digit
- **Sentiment** — enter a movie review (English, SST-2 style)

## How it differs from a normal model

A standard classifier outputs one probability. CIDA shows **structural disagreement** via role priors and measures **observable uncertainty** from agent variance — no post-hoc calibration needed.

## Deploy your own Space

```bash
# Clone repo, copy demo/ folder to a new HF Space
huggingface-cli upload your-username/cida-demo demo/ --repo-type space

# Or train checkpoints locally first:
python demo/train_demo.py
```

## Links

- [GitHub Repository](https://github.com/Kairatzh/CIDA-plugin)
- [Paper / Docs](https://github.com/Kairatzh/CIDA-plugin#readme)
