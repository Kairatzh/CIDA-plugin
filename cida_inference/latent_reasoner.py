import torch

class LatentReasoner:
    """
    Implements continuous latent-space reasoning (Coconut-style).
    Runs recursive deliberation rounds on hidden states without emitting text tokens.
    """
    def __init__(self, wrapper, llm, tokenizer):
        self.wrapper = wrapper
        self.llm = llm
        self.tokenizer = tokenizer
        # Safely determine device
        self.device = "cpu"
        try:
            self.device = next(wrapper.parameters()).device
        except StopIteration:
            if llm is not None:
                try:
                    self.device = next(llm.parameters()).device
                except StopIteration:
                    pass

    def generate_latent(self, prompt: str, reasoning_rounds: int = 5, max_new_tokens: int = 250, **kwargs):
        """
        Executes continuous reasoning rounds in the CIDA latent space, then
        emits the final answer.
        """
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Safely determine compute_dtype
        compute_dtype = torch.float32
        if self.llm is not None:
            try:
                compute_dtype = next(self.llm.parameters()).dtype
            except StopIteration:
                pass

        with torch.no_grad():
            # Get base prompt hidden states
            out = self.llm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1].float().unsqueeze(0) # (1, d_llm)
            p_emb = self.llm.get_input_embeddings()(enc.input_ids).detach()
            
            # Recursive latent reasoning loop (Coconut-style)
            # Instead of emitting tokens, we pass the hidden state through the CIDA core recursively
            h_latent = h.clone()
            for _ in range(reasoning_rounds):
                # Deliberate inside the multi-agent consensus pool
                co = self.wrapper.cida_core(pooled=h_latent)
                # Extract consensus state and project back to LLM space
                plan = self.wrapper.base_projector(co["state_final"].s)
                h_latent = plan.mean(dim=1) # Update working state recursively
            
            # Project final optimized plan using the refined latent state
            final_plan = self.wrapper(h_latent)
            
            # Prepend the prompt embeddings with the 8-token latent plan
            gm = torch.cat([enc.attention_mask, torch.ones(1, 8, dtype=enc.attention_mask.dtype, device=self.device)], dim=1)
            
            ids = self.llm.generate(
                inputs_embeds=torch.cat([p_emb.to(compute_dtype), final_plan.to(compute_dtype)], dim=1),
                attention_mask=gm,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
            
            # Decode the generated tokens
            response = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            return response
