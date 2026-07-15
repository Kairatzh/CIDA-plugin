import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin

# Decoupled projection layer from baseline
class CIDAPlanProjector(nn.Module):
    def __init__(self, d_cida, d_llm, num_agents, num_plan_tokens=8):
        super().__init__()
        self.num_plan_tokens = num_plan_tokens
        self.d_llm = d_llm
        d_flat = num_agents * d_cida
        self.projector = nn.Sequential(
            nn.Linear(d_flat, d_flat*2), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(d_flat*2, num_plan_tokens*d_llm),
            nn.LayerNorm(num_plan_tokens*d_llm))
            
    def forward(self, state_s):
        B = state_s.size(0)
        return self.projector(state_s.reshape(B,-1).float()).view(B, self.num_plan_tokens, self.d_llm)

class AlignedPlanProjector(nn.Module):
    def __init__(self, base_projector, W):
        super().__init__()
        self.base_projector = base_projector
        self.register_buffer("W", W)
        
    def forward(self, state_s):
        plan = self.base_projector(state_s) # (B, 8, d_source)
        W_dtype = next(self.base_projector.parameters()).dtype
        aligned_plan = torch.matmul(plan.to(W_dtype), self.W.to(W_dtype))
        return aligned_plan

class CIDAWrapper(nn.Module):
    """
    LangChain/LlamaIndex style wrapper for integrating CIDA with Causal LLMs.
    """
    def __init__(self, cida_core, base_projector, d_llm):
        super().__init__()
        self.cida_core = cida_core
        self.base_projector = base_projector
        self.d_llm = d_llm
        self.gate_proj = nn.Linear(d_llm * 2, 1)
        
    def forward(self, h_base, alpha=2.2):
        co = self.cida_core(pooled=h_base)
        state_s = co["state_final"].s if hasattr(co["state_final"], "s") else co["state_final"]
        plan = self.base_projector(state_s)
        plan_pooled = plan.mean(dim=1)
        gate_in = torch.cat([h_base, plan_pooled], dim=-1).to(self.gate_proj.weight.dtype)
        gate = torch.sigmoid(self.gate_proj(gate_in)).unsqueeze(1)
        h_base_exp = h_base.unsqueeze(1).expand(-1, plan.size(1), -1)
        mixed_plan = gate * (plan * alpha) + (1.0 - gate) * h_base_exp
        return mixed_plan

    @classmethod
    def from_pretrained(cls, base_model, cida_repo="kirtzh/Kairos-7B-CIDA-v1.0", fine_tuned_path=None):
        """
        One-line initializer to download CIDA adapters and wrap a base causal LLM.
        """
        device = next(base_model.parameters()).device
        d_llm = base_model.config.hidden_size
        
        print(f"📥 Fetching CIDA adapters from repository: {cida_repo}...")
        cfg_path = hf_hub_download(repo_id=cida_repo, filename="config.json")
        cida_bin = hf_hub_download(repo_id=cida_repo, filename="pytorch_model.bin")
        proj_bin = hf_hub_download(repo_id=cida_repo, filename="plan_projector.pt")
        
        # Load CIDA config and core
        cfg = CIDAPluginConfig.from_pretrained(os.path.dirname(cfg_path))
        cida_core = CIDAPlugin(cfg).to(device).float()
        cida_core.load_state_dict(torch.load(cida_bin, map_location=device))
        
        # Determine native d_source from weights
        proj_sd = torch.load(proj_bin, map_location="cpu")
        last_key = list(proj_sd.keys())[-1]
        out_features = proj_sd[last_key].shape[0]
        d_source = out_features // 8
        
        # Load base plan projector with native d_source
        base_proj = CIDAPlanProjector(cfg.d_hidden, d_source, cfg.num_agents, 8).to(device)
        base_proj.load_state_dict(proj_sd)
        
        # If target dimension matches, use directly. Otherwise align.
        if d_source == d_llm:
            final_proj = base_proj
        else:
            print(f"🔄 Dimension mismatch (CIDA source: {d_source}, LLM target: {d_llm}). "
                  f"Applying dynamic manifold alignment...")
            # Compute W alignment matrix using SVD on target embeddings
            try:
                target_embeds = base_model.get_input_embeddings().weight.detach().float()
            except Exception:
                target_embeds = torch.randn(2000, d_llm, device="cpu")
            
            U, S, V = torch.linalg.svd(target_embeds, full_matrices=False)
            W = torch.zeros(d_source, d_llm, device=device)
            min_d = min(d_source, d_llm)
            for i in range(min_d):
                W[i, i] = 1.0
            W = torch.matmul(W, V.to(device))
            final_proj = AlignedPlanProjector(base_proj, W).to(device)
            
        wrapper = cls(cida_core, final_proj, d_llm).to(device)
        
        # Optionally load fine-tuned gating layers
        if fine_tuned_path and os.path.exists(fine_tuned_path):
            wrapper.load_state_dict(torch.load(fine_tuned_path, map_location=device), strict=False)
            print(f"✅ Loaded fine-tuned gating weights from: {fine_tuned_path}")
        else:
            print("⚠️ No fine-tuned path supplied or found. Using baseline gating parameters.")
            
        return wrapper
