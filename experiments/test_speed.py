import torch
import time
from cida_plugin.core import CIDAPlugin
from cida_plugin.config import CIDAPluginConfig

cfg = CIDAPluginConfig(
    d_input=768,
    d_hidden=768,
    d_message=768,
    num_agents=4,
    num_classes=14,
    max_rounds=3,
    multi_label=True
)

plugin = CIDAPlugin(cfg)
plugin.train()

p = torch.randn(8, 768)
s = torch.randn(8, 980, 768)

print("Starting forward-backward tests...")
for i in range(5):
    t0 = time.time()
    out = plugin(p, seq_output=s)
    loss = out["p_final"].sum()
    loss.backward()
    print(f"Step {i}: {time.time() - t0:.4f} seconds")
