import torch
from .backbone import LLMBackbone, HuggingFaceBackbone
from .callbacks import BaseCallbackHandler, AgentCallbackManager

class LatentReasoner:
    """
    Implements continuous latent-space reasoning (Coconut-style).
    Runs recursive deliberation rounds on hidden states without emitting text tokens.
    Supports Adaptive Latent Compute (elastic thinking depth) based on CIDA disagreement.
    """
    def __init__(self, wrapper, llm, tokenizer, callbacks=None):
        self.wrapper = wrapper
        
        # Check and wrap llm using HuggingFaceBackbone if it's not already an LLMBackbone
        if llm is not None and not (isinstance(llm, LLMBackbone) or hasattr(llm, "get_last_hidden")):
            self.llm = HuggingFaceBackbone(llm)
        else:
            self.llm = llm
            
        self.tokenizer = tokenizer
        
        # Callbacks
        if callbacks is None:
            self.callbacks = AgentCallbackManager([])
        elif isinstance(callbacks, list):
            self.callbacks = AgentCallbackManager(callbacks)
        else:
            self.callbacks = callbacks
            
        # Safely determine device
        self.device = "cpu"
        if self.llm is not None:
            self.device = self.llm.device
        elif wrapper is not None:
            try:
                self.device = next(wrapper.parameters()).device
            except StopIteration:
                pass

    def generate_latent(self, prompt: str, min_rounds: int = 1, max_rounds: int = 8, 
                        target_disagreement: float = 0.15, max_new_tokens: int = 250, **kwargs):
        """
        Executes adaptive continuous reasoning rounds in the CIDA latent space, then
        emits the final answer.
        """
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Safely determine compute_dtype
        compute_dtype = self.llm.dtype if self.llm is not None else torch.float32

        with torch.no_grad():
            # Get base prompt hidden states using the backbone abstraction
            h = self.llm.get_last_hidden(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            
            p_emb = self.llm.get_input_embeddings()(enc.input_ids).detach()
            
            # Recursive latent reasoning loop (Coconut-style) with Adaptive Compute
            h_latent = h.clone()
            
            # Determine initial task complexity/variance
            co_init = self.wrapper.cida_core(pooled=h_latent)
            agent_states = co_init["state_final"].s
            agent_mean = agent_states.mean(dim=1, keepdim=True)
            variance = torch.var(agent_states - agent_mean).item()
            
            self.callbacks.on_deliberation_start(max_rounds, variance)
            
            actual_rounds = 0
            
            # --- FAST PATH EARLY EXIT ---
            if variance < 0.05:
                # Task is trivial, agents are in near-perfect agreement instantly.
                self.callbacks.on_deliberation_step(0, variance)
            else:
                for r in range(max_rounds):
                    actual_rounds = r + 1
                    co = self.wrapper.cida_core(pooled=h_latent)
                    disagreement_val = co.get("disagreement", torch.tensor([0.0])).mean().item()
                    
                    self.callbacks.on_deliberation_step(r, disagreement_val)
                    
                    if r >= min_rounds and disagreement_val < target_disagreement:
                        break
                    
                    # Extract consensus state and project back to LLM space
                    plan = self.wrapper.base_projector(co["state_final"].s)
                    h_latent = plan.mean(dim=1) # Update working state recursively
            
            self.callbacks.on_deliberation_end(actual_rounds)
            
            # Project final optimized plan using the refined latent state
            final_plan = self.wrapper(h_latent)
            
            # Prepend the prompt embeddings with the 8-token latent plan
            gm = torch.cat([enc.attention_mask, torch.ones(1, 8, dtype=enc.attention_mask.dtype, device=self.device)], dim=1)
            
            ids = self.llm.generate_from_embeds(
                inputs_embeds=torch.cat([p_emb.to(compute_dtype), final_plan.to(compute_dtype)], dim=1),
                attention_mask=gm,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
            
            # Decode the generated tokens
            response = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            return response
