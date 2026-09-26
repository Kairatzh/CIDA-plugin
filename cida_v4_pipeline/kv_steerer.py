"""
kv_steerer.py — Prefix KV-Cache Injector for Hugging Face Transformers

Eliminates the "Alignment Tax" and Code Syntax Degradation:
Instead of prepending soft tokens to input_ids (which shifts positional embeddings and breaks code indentation),
PrefixKVSteerer projects CIDA plan tokens directly into Key/Value states of intermediate transformer layers.
"""

from typing import Optional, List, Tuple, Union, Dict, Any
import torch
import torch.nn as nn

try:
    from transformers.cache_utils import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except ImportError:
    _HAS_DYNAMIC_CACHE = False


class PrefixKVSteerer(nn.Module):
    """
    Transforms CIDA plan tokens (B, P, D_llm) into Key/Value prefix caches
    for target transformer layers (e.g., layers 2 through 6).
    """
    def __init__(
        self,
        d_llm: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        target_layer_indices: Optional[List[int]] = None,
        num_plan_tokens: int = 4,
    ):
        super().__init__()
        self.d_llm = d_llm
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_plan_tokens = num_plan_tokens
        
        # Default target layers: intermediate layers (e.g. layers 2 to 6)
        if target_layer_indices is None:
            max_idx = min(6, num_layers)
            self.target_layer_indices = list(range(max(0, max_idx - 4), max_idx))
        else:
            self.target_layer_indices = target_layer_indices
            
        kv_dim = num_heads * head_dim
        
        # Lightweight Key and Value projection heads for each target layer
        self.k_projs = nn.ModuleDict({
            str(layer_idx): nn.Sequential(
                nn.Linear(d_llm, kv_dim, bias=False),
                nn.LayerNorm(kv_dim)
            )
            for layer_idx in self.target_layer_indices
        })
        
        self.v_projs = nn.ModuleDict({
            str(layer_idx): nn.Sequential(
                nn.Linear(d_llm, kv_dim, bias=False),
                nn.LayerNorm(kv_dim)
            )
            for layer_idx in self.target_layer_indices
        })

    @classmethod
    def from_model_config(cls, model_config, target_layers: Optional[List[int]] = None) -> "PrefixKVSteerer":
        """Factory method to construct steerer matching the exact backbone architecture."""
        d_llm = getattr(model_config, "hidden_size", 4096)
        num_layers = getattr(model_config, "num_hidden_layers", 32)
        num_heads = getattr(model_config, "num_key_value_heads", None) or getattr(model_config, "num_attention_heads", 32)
        head_dim = getattr(model_config, "head_dim", None) or (d_llm // getattr(model_config, "num_attention_heads", 32))
        
        return cls(
            d_llm=d_llm,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            target_layer_indices=target_layers
        )

    def forward(self, plan_tokens: torch.Tensor) -> Any:
        """
        plan_tokens: (B, P, d_llm)
        Returns: HuggingFace DynamicCache or dictionary of (layer_idx -> (K_prefix, V_prefix))
        """
        B, P, _ = plan_tokens.size()
        kv_dict = {}
        
        for layer_idx in self.target_layer_indices:
            k_proj = self.k_projs[str(layer_idx)]
            v_proj = self.v_projs[str(layer_idx)]
            
            # Project to (B, P, num_heads * head_dim)
            k = k_proj(plan_tokens).view(B, P, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, P, D)
            v = v_proj(plan_tokens).view(B, P, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, P, D)
            
            kv_dict[layer_idx] = (k, v)
            
        return kv_dict

    def build_huggingface_past_key_values(
        self,
        plan_tokens: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype
    ) -> Union[Any, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """
        Constructs standard HuggingFace past_key_values structure containing the prefix KV states.
        """
        kv_dict = self.forward(plan_tokens)
        B, P = plan_tokens.size(0), plan_tokens.size(1)
        
        if _HAS_DYNAMIC_CACHE:
            cache = DynamicCache()
            for layer_idx in range(self.num_layers):
                if layer_idx in kv_dict:
                    k, v = kv_dict[layer_idx]
                    cache.update(k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype), layer_idx)
                else:
                    # Dummy empty prefix for non-target layers
                    empty_k = torch.zeros(B, self.num_heads, 0, self.head_dim, device=device, dtype=dtype)
                    empty_v = torch.zeros(B, self.num_heads, 0, self.head_dim, device=device, dtype=dtype)
                    cache.update(empty_k, empty_v, layer_idx)
            return cache
        else:
            past_list = []
            for layer_idx in range(self.num_layers):
                if layer_idx in kv_dict:
                    k, v = kv_dict[layer_idx]
                    past_list.append((k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype)))
                else:
                    empty_k = torch.zeros(B, self.num_heads, 0, self.head_dim, device=device, dtype=dtype)
                    empty_v = torch.zeros(B, self.num_heads, 0, self.head_dim, device=device, dtype=dtype)
                    past_list.append((empty_k, empty_v))
            return tuple(past_list)

    def inject_inputs_embeds_fallback(
        self,
        prompt_embeds: torch.Tensor,
        plan_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fallback mode: Prepend plan tokens directly to input embeddings if model does not support prefix cache.
        prompt_embeds: (B, L, D)
        plan_tokens: (B, P, D)
        """
        B, L, D = prompt_embeds.size()
        P = plan_tokens.size(1)
        
        combined_embeds = torch.cat([plan_tokens.to(prompt_embeds.dtype), prompt_embeds], dim=1)
        
        if attention_mask is not None:
            plan_mask = torch.ones(B, P, device=attention_mask.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([plan_mask, attention_mask], dim=1)
        else:
            combined_mask = torch.ones(B, L + P, device=prompt_embeds.device, dtype=torch.long)
            
        return combined_embeds, combined_mask
