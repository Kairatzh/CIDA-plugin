import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from .core import CIDAPlugin, PlanProjector

class KairosOrchestrator(nn.Module):
    """
    KairosOrchestrator is the heart of CIDA-OS.
    It wraps the frozen LLM backbone and CIDA v1, enabling latent integration
    of multi-source inputs (text prompts, vision features, RAG documents)
    directly into CIDA v1's deliberation sequence loop.
    """
    def __init__(
        self,
        backbone_model_name: str,
        cida_model_path_or_repo: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype

        # 1. Load frozen LLM backbone
        print(f"[CIDA-OS] Loading frozen backbone: {backbone_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                backbone_model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map={"": device} if device == "cuda" else None,
                output_hidden_states=True,
                **kwargs
            )
        except Exception:
            self.llm = AutoModelForCausalLM.from_pretrained(
                backbone_model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                output_hidden_states=True,
                **kwargs
            ).to(device)
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()

        self.d_llm = self.llm.config.hidden_size
        self.embed_tokens = self.llm.get_input_embeddings()

        # 2. Load trained CIDA v1 weights (or initialize fresh if None)
        if cida_model_path_or_repo is not None:
            print(f"[CIDA-OS] Loading CIDA v1 adapter from: {cida_model_path_or_repo}")
            self.cida = CIDAPlugin.from_pretrained(cida_model_path_or_repo).to(device)
        else:
            print("[CIDA-OS] No CIDA adapter path provided. Initializing fresh CIDA weights for testing.")
            self.cida = CIDAPlugin(d_llm=self.d_llm).to(device)
            
        self.cida.eval()

        # Load plan projector
        print(f"[CIDA-OS] Loading PlanProjector")
        # In CIDA v1, the plan projector maps the final CIDA state to plan tokens.
        # Let's dynamically load it
        if hasattr(self.cida, 'plan_projector'):
            self.projector = self.cida.plan_projector
        else:
            # Fallback initialization matching v1 dimension
            self.projector = PlanProjector(num_plan_tokens=8, hidden_size=self.d_llm).to(device)
            # Try downloading/loading plan_projector.pt from HF repository
            try:
                import os
                if os.path.isdir(cida_model_path_or_repo):
                    pt_path = os.path.join(cida_model_path_or_repo, "plan_projector.pt")
                else:
                    from huggingface_hub import hf_hub_download
                    pt_path = hf_hub_download(repo_id=cida_model_path_or_repo, filename="plan_projector.pt")
                state_dict = torch.load(pt_path, map_location=device)
                self.projector.load_state_dict(state_dict)
                
                # Dynamically set num_plan_tokens based on loaded weights shape
                last_weight_key = [k for k in state_dict.keys() if k.endswith(".weight")][-1]
                out_features = state_dict[last_weight_key].shape[0]
                # Extract the loaded hidden size from the first layer in state_dict
                loaded_hidden_size = state_dict["projector.0.weight"].shape[1]
                self.projector.num_plan_tokens = out_features // loaded_hidden_size
                print(f"[CIDA-OS] Loaded plan_projector.pt successfully. Detected plan tokens: {self.projector.num_plan_tokens}")
            except Exception as e:
                print(f"[CIDA-OS] Warning loading plan_projector.pt, using default: {e}")

        # 3. Projection heads for external architectures to match LLM hidden size
        self.external_projections = nn.ModuleDict()

    def register_external_source(self, name: str, input_dim: int):
        """Register a projection head for an external architecture (e.g. Vision encoder)."""
        self.external_projections[name] = nn.Linear(input_dim, self.d_llm).to(self.device).to(self.torch_dtype)
        print(f"[CIDA-OS] Registered external source: {name} (dimension {input_dim} -> {self.d_llm})")

    @torch.no_grad()
    def get_prompt_context(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract frozen LLM hidden states, prompt embeddings, and the pooled prompt representation."""
        out = self.llm(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            output_hidden_states=True,
            use_cache=False
        )
        last_h = out.hidden_states[-1]
        idx = prompt_mask.sum(1) - 1
        bidx = torch.arange(prompt_ids.size(0), device=prompt_ids.device)
        pooled = last_h[bidx, idx].float()
        
        p_emb = self.embed_tokens(prompt_ids).detach().float()
        return pooled, p_emb, last_h

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        external_inputs: Optional[Dict[str, torch.Tensor]] = None,
        alpha: float = 2.2,
    ) -> Dict[str, Any]:
        """
        Orchestrates multi-source inputs, runs CIDA deliberation, and outputs virtual plan embeddings.
        """
        B = prompt_ids.size(0)
        pooled, p_emb, last_h = self.get_prompt_context(prompt_ids, prompt_mask)

        # Build seq_output for CIDA deliberation
        # We start with the prompt hidden states
        seq_list = [last_h.float()]
        mask_list = [prompt_mask]

        if external_inputs is not None:
            for name, tensor in external_inputs.items():
                # Project external features to match LLM hidden size
                if name in self.external_projections:
                    proj_tensor = self.external_projections[name](tensor.to(self.torch_dtype)).float()
                else:
                    proj_tensor = tensor.float()

                seq_list.append(proj_tensor)
                # Create a active attention mask for all external tokens
                ext_mask = torch.ones(B, proj_tensor.size(1), dtype=prompt_mask.dtype, device=prompt_mask.device)
                mask_list.append(ext_mask)

        # Concatenate everything along the sequence dimension
        seq_output = torch.cat(seq_list, dim=1)
        combined_mask = torch.cat(mask_list, dim=1)

        # Handle CIDA input dimension mismatch (e.g. 3584 backbone vs 4096 CIDA adapter)
        d_cida_input = getattr(self.cida.config, "d_input", pooled.size(-1))
        if pooled.size(-1) != d_cida_input:
            if pooled.size(-1) < d_cida_input:
                padding_size = d_cida_input - pooled.size(-1)
                pooled = torch.nn.functional.pad(pooled, (0, padding_size))
                seq_output = torch.nn.functional.pad(seq_output, (0, padding_size))
            else:
                pooled = pooled[..., :d_cida_input]
                seq_output = seq_output[..., :d_cida_input]

        # Deliberate inside CIDA v1
        cida_out = self.cida(
            pooled=pooled,
            seq_output=seq_output,
            mask=combined_mask
        )

        # Project consensus state to virtual plan tokens
        s_final = cida_out["state_final"].s
        plan = self.projector(s_final)
        plan = plan * alpha

        # Cat plan embeddings for assistant consumption
        cat_embeddings = torch.cat([p_emb.to(self.torch_dtype), plan.to(self.torch_dtype)], dim=1)
        plan_mask = torch.ones(B, plan.size(1), dtype=prompt_mask.dtype, device=prompt_mask.device)
        cat_mask = torch.cat([prompt_mask, plan_mask], dim=1)

        return {
            "inputs_embeds": cat_embeddings,
            "attention_mask": cat_mask,
            "uncertainty": cida_out["uncertainty"],
            "disagreement": cida_out["disagreement"],
            "b_all": cida_out["b_all"]
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        external_inputs: Optional[Dict[str, torch.Tensor]] = None,
        alpha: float = 2.2,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """Runs CIDA-OS latent orchestration and generates text completion."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        orch_out = self.forward(
            prompt_ids=enc["input_ids"],
            prompt_mask=enc["attention_mask"],
            external_inputs=external_inputs,
            alpha=alpha
        )

        gen_ids = self.llm.generate(
            inputs_embeds=orch_out["inputs_embeds"],
            attention_mask=orch_out["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
