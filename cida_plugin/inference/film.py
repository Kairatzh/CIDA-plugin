"""
FiLM — Feature-wise Linear Modulation for CIDA ↔ LLM Integration.

Instead of concatenating plan-tokens to input embeddings (rigid),
FiLM generates per-layer (γ, β) coefficients from CIDA's latent consensus state.
These coefficients dynamically modulate the hidden activations inside
frozen LLM Transformer layers via register_forward_hook.

Reference: Perez et al. 2018 — "FiLM: Visual Reasoning with a General Conditioning Layer"

Usage:
    film_gen = FiLMGenerator(d_cida=128, d_llm=4096, num_film_layers=4)
    gamma, beta = film_gen(cida_latent_state)

    film_mod = FiLMModulatedGeneration(base_model, tokenizer, film_gen)
    output_text = film_mod.generate(prompt, cida_state, max_new_tokens=256)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class FiLMGenerator(nn.Module):
    """
    Generates per-layer FiLM coefficients (γ, β) from CIDA latent state.

    Given a latent vector from CIDA's deliberation (e.g. state_final.s pooled),
    produces scaling (gamma) and shifting (beta) vectors for each target LLM layer.

    Parameters
    ----------
    d_cida : int
        Dimensionality of the CIDA latent input (e.g. d_hidden * num_agents flattened).
    d_llm : int
        Hidden size of the target LLM (e.g. 4096 for Qwen-7B).
    num_film_layers : int
        How many LLM layers to modulate (default: 4, applied to the last N layers).
    bottleneck_factor : int
        Compression factor for the shared trunk (default: 4).
    """

    def __init__(self, d_cida: int, d_llm: int, num_film_layers: int = 4, bottleneck_factor: int = 4):
        super().__init__()
        self.d_cida = d_cida
        self.d_llm = d_llm
        self.num_film_layers = num_film_layers

        bottleneck_dim = d_llm // bottleneck_factor

        # Shared trunk: compress CIDA state into a compact representation
        self.trunk = nn.Sequential(
            nn.Linear(d_cida, bottleneck_dim),
            nn.SiLU(),
            nn.LayerNorm(bottleneck_dim),
        )

        # Per-layer heads: each produces (gamma, beta) pair
        self.film_heads = nn.ModuleList([
            nn.Linear(bottleneck_dim, d_llm * 2)  # gamma + beta concatenated
            for _ in range(num_film_layers)
        ])

        # Initialize close to identity: gamma ≈ 1, beta ≈ 0
        self._init_near_identity()

    def _init_near_identity(self):
        """Initialize FiLM heads so that gamma ≈ 1.0 and beta ≈ 0.0 (no-op at start)."""
        for head in self.film_heads:
            nn.init.zeros_(head.weight)
            # Bias: first half (gamma) = 1.0, second half (beta) = 0.0
            with torch.no_grad():
                head.bias[:self.d_llm] = 1.0   # gamma init
                head.bias[self.d_llm:] = 0.0   # beta init

    def forward(self, cida_state: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate FiLM coefficients for each target layer.

        Parameters
        ----------
        cida_state : (B, d_cida) or (B, num_agents, d_hidden)
            Latent state from CIDA deliberation.

        Returns
        -------
        list of (gamma, beta) tuples, one per film layer.
            gamma : (B, d_llm) — scaling coefficients
            beta  : (B, d_llm) — shifting coefficients
        """
        B = cida_state.size(0)

        # Flatten if multi-agent state (B, num_agents, d_hidden) -> (B, d_cida)
        if cida_state.dim() > 2:
            cida_state = cida_state.reshape(B, -1)

        # Pad or truncate to match expected d_cida
        if cida_state.size(1) < self.d_cida:
            padding = torch.zeros(B, self.d_cida - cida_state.size(1),
                                  device=cida_state.device, dtype=cida_state.dtype)
            cida_state = torch.cat([cida_state, padding], dim=1)
        elif cida_state.size(1) > self.d_cida:
            cida_state = cida_state[:, :self.d_cida]

        trunk_out = self.trunk(cida_state.float())  # (B, bottleneck_dim)

        film_params = []
        for head in self.film_heads:
            gb = head(trunk_out)  # (B, d_llm * 2)
            gamma = gb[:, :self.d_llm]   # (B, d_llm)
            beta = gb[:, self.d_llm:]    # (B, d_llm)
            film_params.append((gamma, beta))

        return film_params


class FiLMModulatedGeneration:
    """
    Wraps a frozen LLM and applies FiLM modulation during generation.

    Uses PyTorch forward hooks to intercept and modulate hidden states
    at specific transformer layers. Hooks are installed only during generation
    and removed immediately after.

    Parameters
    ----------
    base_model : nn.Module
        The frozen LLM (e.g. Qwen, Llama).
    tokenizer : PreTrainedTokenizer
        The model's tokenizer.
    film_generator : FiLMGenerator
        The trained FiLM coefficient generator.
    target_layer_indices : list[int], optional
        Which transformer layer indices to modulate. If None, uses the last N layers.
    """

    def __init__(self, base_model, tokenizer, film_generator: FiLMGenerator,
                 target_layer_indices: Optional[List[int]] = None):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.film_generator = film_generator
        self.target_layer_indices = target_layer_indices
        self._hooks = []

    def _get_transformer_layers(self) -> nn.ModuleList:
        """Auto-detect the transformer layer list from common model architectures."""
        model = self.base_model

        # Common attribute paths for transformer layers
        candidates = [
            "model.layers",          # Llama, Qwen, Mistral
            "transformer.h",         # GPT-2, GPT-Neo
            "model.decoder.layers",  # OPT
            "encoder.layer",         # BERT
        ]

        for path in candidates:
            obj = model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)):
                    return obj
            except AttributeError:
                continue

        raise ValueError(
            "Could not auto-detect transformer layers. "
            "Please specify target_layer_indices manually."
        )

    def _resolve_target_indices(self, all_layers) -> List[int]:
        """Determine which layer indices to modulate."""
        if self.target_layer_indices is not None:
            return self.target_layer_indices

        num_layers = len(all_layers)
        n = self.film_generator.num_film_layers
        # Default: modulate the last N layers
        return list(range(max(0, num_layers - n), num_layers))

    def _install_hooks(self, film_params: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Install forward hooks on target layers."""
        self._remove_hooks()  # Clean up any existing hooks

        try:
            layers = self._get_transformer_layers()
        except ValueError:
            return  # Skip if layers can't be detected

        target_indices = self._resolve_target_indices(layers)

        for i, layer_idx in enumerate(target_indices):
            if i >= len(film_params):
                break

            gamma, beta = film_params[i]

            def make_hook(g, b):
                def hook_fn(module, input, output):
                    # Handle different output formats
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        # Apply FiLM: h_new = gamma * h + beta
                        g_cast = g.to(hidden_states.dtype).unsqueeze(1)  # (B, 1, d_llm)
                        b_cast = b.to(hidden_states.dtype).unsqueeze(1)  # (B, 1, d_llm)
                        
                        # Handle batch size mismatch (beam search, etc.)
                        if hidden_states.size(0) != g_cast.size(0):
                            g_cast = g_cast.expand(hidden_states.size(0), -1, -1)
                            b_cast = b_cast.expand(hidden_states.size(0), -1, -1)
                        
                        modulated = g_cast * hidden_states + b_cast
                        return (modulated,) + output[1:]
                    else:
                        g_cast = g.to(output.dtype).unsqueeze(1)
                        b_cast = b.to(output.dtype).unsqueeze(1)
                        if output.size(0) != g_cast.size(0):
                            g_cast = g_cast.expand(output.size(0), -1, -1)
                            b_cast = b_cast.expand(output.size(0), -1, -1)
                        return g_cast * output + b_cast
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook(gamma, beta))
            self._hooks.append(handle)

    def _remove_hooks(self):
        """Remove all installed hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def generate(self, prompt: str, cida_state: torch.Tensor,
                 max_new_tokens: int = 256, **generate_kwargs) -> str:
        """
        Generate text with FiLM-modulated LLM layers.

        Parameters
        ----------
        prompt : str
            The input prompt.
        cida_state : torch.Tensor
            The latent state from CIDA deliberation.
        max_new_tokens : int
            Maximum tokens to generate.

        Returns
        -------
        str : Generated text.
        """
        device = next(self.base_model.parameters(), next(self.film_generator.parameters(), torch.tensor(0))).device


        # Generate FiLM coefficients
        with torch.no_grad():
            film_params = self.film_generator(cida_state.to(device))

        # Install hooks
        self._install_hooks(film_params)

        try:
            enc = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(enc, "to"):
                enc = enc.to(device)
            elif isinstance(enc, dict):
                enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}

            input_ids = getattr(enc, "input_ids", enc["input_ids"] if isinstance(enc, dict) else enc)

            with torch.no_grad():
                if isinstance(enc, dict):
                    output_ids = self.base_model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id or getattr(self.tokenizer, "eos_token_id", 0),
                        **generate_kwargs
                    )
                else:
                    output_ids = self.base_model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id or getattr(self.tokenizer, "eos_token_id", 0),
                        **generate_kwargs
                    )
            generated = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

        finally:
            # Always clean up hooks
            self._remove_hooks()

        return generated

    def get_film_params(self, cida_state: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get the FiLM parameters without generating text (for Strategy Library)."""
        with torch.no_grad():
            return self.film_generator(cida_state)
