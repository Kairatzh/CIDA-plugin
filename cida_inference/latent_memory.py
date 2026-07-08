import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentMemory(nn.Module):
    """
    A lightweight, tensor-based vector store that acts as a 'latent diary'
    to store past planning states and mix them back into current decisions.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.keys = []
        self.values = []
        self.texts = []
        
        # Simple cross-attention mixing projection
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, 1)

    def clear(self):
        """Reset the latent memory."""
        self.keys = []
        self.values = []
        self.texts = []

    def add_memory(self, key_tensor, value_tensor, text_desc=""):
        """
        Store a planning state key and value representation along with a description.
        Expects shapes of (1, d_model) or (d_model,).
        """
        self.keys.append(key_tensor.detach().cpu().squeeze().float())
        self.values.append(value_tensor.detach().cpu().squeeze().float())
        self.texts.append(text_desc)

    def retrieve(self, query_tensor):
        """
        Retrieves matching memory traces using cosine similarity
        and returns a gated, attention-weighted context vector.
        """
        device = query_tensor.device
        if not self.keys:
            # Return zero vector if no memories are stored yet
            return torch.zeros_like(query_tensor)

        # Convert lists of tensors to single batched tensors on current device
        K_stack = torch.stack(self.keys).to(device) # (N, d_model)
        V_stack = torch.stack(self.values).to(device) # (N, d_model)

        q = self.q_proj(query_tensor) # (1, d_model)
        k = self.k_proj(K_stack) # (N, d_model)
        v = self.v_proj(V_stack) # (N, d_model)

        # Compute cosine similarities / dot-product attention
        scores = torch.matmul(q, k.transpose(0, 1)) / (self.d_model ** 0.5) # (1, N)
        attn_weights = F.softmax(scores, dim=-1) # (1, N)

        # Retrieve weighted value representation
        context = torch.matmul(attn_weights, v) # (1, d_model)
        context = self.out_proj(context)

        # Compute dynamic gating factor between memory context and current query
        gate_val = torch.sigmoid(self.gate(torch.cat([query_tensor, context], dim=-1)))
        mixed = gate_val * context + (1.0 - gate_val) * query_tensor
        
        return mixed

    def get_active_concepts(self, query_tensor, k=3):
        """
        Retrieve the text descriptions of the top k matching memories
        for explainability logging.
        """
        if not self.keys:
            return []
        
        device = query_tensor.device
        K_stack = torch.stack(self.keys).to(device) # (N, d_model)
        
        with torch.no_grad():
            q = self.q_proj(query_tensor)
            k_proj = self.k_proj(K_stack)
            # Compute cosine similarities
            scores = torch.matmul(q, k_proj.transpose(0, 1)).squeeze(0) # (N,)
            
            top_k = min(k, len(self.keys))
            top_vals, top_idx = torch.topk(scores, top_k)
            
            active = []
            for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                active.append((self.texts[idx], val))
            return active
