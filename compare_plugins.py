import torch
from cida_plugin import CIDAPlugin as OriginalCIDAPlugin, CIDAPluginConfig as OriginalConfig
from cida_latent_plugin import CIDAPlugin as LatentCIDAPlugin, CIDAPluginConfig as LatentConfig

def test_parity():
    # Setup configs
    cfg_orig = OriginalConfig(d_input=128, d_hidden=64, num_classes=3, max_rounds=3, use_liquid_dynamics=True)
    cfg_latent = LatentConfig(d_input=128, d_hidden=64, num_classes=3, max_rounds=3, use_liquid_dynamics=True)
    
    # Initialize both with deterministic state
    torch.manual_seed(42)
    plugin_orig = OriginalCIDAPlugin(cfg_orig)
    
    torch.manual_seed(42)
    plugin_latent = LatentCIDAPlugin(cfg_latent)
    
    # Ensure weights are identical
    for p1, p2 in zip(plugin_orig.parameters(), plugin_latent.parameters()):
        assert torch.allclose(p1, p2), "Model parameters are not identical after seeding!"
        
    # Generate dummy input
    batch_size = 4
    pooled = torch.randn(batch_size, 128)
    seq = torch.randn(batch_size, 10, 128)
    mask = torch.ones(batch_size, 10)
    
    # Eval mode
    plugin_orig.eval()
    plugin_latent.eval()
    
    with torch.no_grad():
        out_orig = plugin_orig(pooled, seq_output=seq, mask=mask)
        out_latent = plugin_latent(pooled, seq_output=seq, mask=mask)
        
    # Compare outputs
    assert torch.allclose(out_orig["p_final"], out_latent["p_final"], atol=1e-6), "Logits do not match!"
    assert len(out_orig["b_all"]) == len(out_latent["b_all"]), "Number of debate beliefs does not match!"
    
    for b1, b2 in zip(out_orig["b_all"], out_latent["b_all"]):
        assert torch.allclose(b1, b2, atol=1e-6), "Debate beliefs do not match!"
        
    print("[SUCCESS] Parity check PASSED! Both plugins output identical results for identical inputs.")

if __name__ == "__main__":
    test_parity()
