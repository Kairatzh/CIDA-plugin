import torch
from cida_plugin import CIDAPlugin, CIDAPluginConfig

def test_plugin_initialization():
    config = CIDAPluginConfig(d_input=128, num_classes=2)
    plugin = CIDAPlugin(config)
    assert plugin is not None
    assert plugin.config.d_input == 128

def test_plugin_forward_pass():
    config = CIDAPluginConfig(d_input=128, d_hidden=64, num_classes=3, max_rounds=2)
    plugin = CIDAPlugin(config)
    plugin.eval()

    batch_size = 4
    pooled = torch.randn(batch_size, 128)
    seq_out = torch.randn(batch_size, 10, 128)

    with torch.no_grad():
        out = plugin(pooled, seq_output=seq_out)

    assert "p_final" in out
    assert out["p_final"].shape == (batch_size, 3)
    assert torch.allclose(out["p_final"].sum(dim=-1), torch.ones(batch_size))
