import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentMemory(nn.Module):
    """
    A lightweight, tensor-based vector store that acts as a 'latent diary'
    to store past planning states and mix them back into current decisions.
    Supports tool tracking and reliability weighting.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.keys = []
        self.values = []
        self.texts = []
        self.tool_names = []
        self.decay_weights = []
        
        # Simple cross-attention mixing projection
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, 1)

        # Initialize linear layers as Identity mappings (zero-shot ready)
        with torch.no_grad():
            nn.init.eye_(self.q_proj.weight)
            nn.init.zeros_(self.q_proj.bias)
            nn.init.eye_(self.k_proj.weight)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.eye_(self.v_proj.weight)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.eye_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def clear(self, keep_seeded=True):
        """Reset the latent memory, optionally preserving seeded clues."""
        if keep_seeded:
            new_keys = []
            new_values = []
            new_texts = []
            new_tools = []
            new_decays = []
            for k, v, t, tn, d in zip(self.keys, self.values, self.texts, self.tool_names, self.decay_weights):
                if not t.startswith("Called tool"):
                    new_keys.append(k)
                    new_values.append(v)
                    new_texts.append(t)
                    new_tools.append(tn)
                    new_decays.append(d)
            self.keys = new_keys
            self.values = new_values
            self.texts = new_texts
            self.tool_names = new_tools
            self.decay_weights = new_decays
        else:
            self.keys = []
            self.values = []
            self.texts = []
            self.tool_names = []
            self.decay_weights = []

    def add_memory(self, key_tensor, value_tensor, text_desc="", tool_name=None):
        """
        Store a planning state key and value representation along with a description.
        Expects shapes of (1, d_model) or (d_model,).
        """
        self.keys.append(key_tensor.detach().cpu().squeeze().float())
        self.values.append(value_tensor.detach().cpu().squeeze().float())
        self.texts.append(text_desc)
        self.tool_names.append(tool_name)
        self.decay_weights.append(1.0)

    def retrieve(self, query_tensor, tool_reliability=None):
        """
        Retrieves matching memory traces using cosine similarity
        and returns a gated, attention-weighted context vector.
        If tool_reliability is provided (dict mapping tool_name -> float),
        the attention scores are dynamically scaled based on historical tool reliability.
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

        # L2 Normalize for true Cosine Similarity
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)

        # Compute cosine similarities (scores in range [-1.0, 1.0])
        scores = torch.matmul(q_norm, k_norm.transpose(0, 1)) # (1, N)
        
        # Scale scores by temperature factor to prevent softmax flattening
        scores_scaled = scores * 10.0
        
        # Apply Reliability-Weighted tool scaling
        if tool_reliability:
            # Construct a scaling multiplier tensor of shape (1, N)
            multipliers = []
            for tn in self.tool_names:
                multipliers.append(tool_reliability.get(tn, 1.0))
            m_tensor = torch.tensor(multipliers, dtype=scores_scaled.dtype, device=device).unsqueeze(0)
            scores_scaled = scores_scaled * m_tensor

        # Apply Temporal Decay (Latent Garbage Collector)
        d_tensor = torch.tensor(self.decay_weights, dtype=scores_scaled.dtype, device=device).unsqueeze(0)
        scores_scaled = scores_scaled * d_tensor

        attn_weights = F.softmax(scores_scaled, dim=-1) # (1, N)

        # Update decay weights: Top 2 matches get refreshed (1.0), others decay by 10%
        if len(self.keys) > 0:
            top_indices = torch.topk(scores_scaled, min(2, len(self.keys))).indices[0].tolist()
            for i in range(len(self.decay_weights)):
                if i in top_indices:
                    self.decay_weights[i] = 1.0
                else:
                    self.decay_weights[i] = max(0.1, self.decay_weights[i] * 0.9)

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
        for explainability logging using true cosine similarity.
        """
        if not self.keys:
            return []
        
        device = query_tensor.device
        K_stack = torch.stack(self.keys).to(device) # (N, d_model)
        
        with torch.no_grad():
            q = self.q_proj(query_tensor)
            k_proj = self.k_proj(K_stack)
            
            # L2 Normalize for true Cosine Similarity
            q_norm = F.normalize(q, p=2, dim=-1)
            k_norm = F.normalize(k_proj, p=2, dim=-1)
            
            # Compute cosine similarities
            scores = torch.matmul(q_norm, k_norm.transpose(0, 1)).squeeze(0) # (N,)
            
            top_k = min(k, len(self.keys))
            top_vals, top_idx = torch.topk(scores, top_k)
            
            active = []
            for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                active.append((self.texts[idx], val))
            return active

    def save_to_file(self, filepath: str):
        """
        Saves the memory keys, values, and texts to a file.
        Uses torch.save to handle tensors alongside strings safely.
        """
        state = {
            "d_model": self.d_model,
            "keys": [k.cpu() for k in self.keys],
            "values": [v.cpu() for v in self.values],
            "texts": self.texts,
            "tool_names": self.tool_names,
            "decay_weights": self.decay_weights,
            "state_dict": self.state_dict()
        }
        torch.save(state, filepath)

    def load_from_file(self, filepath: str, map_location="cpu"):
        """
        Loads the memory state from a file.
        """
        state = torch.load(filepath, map_location=map_location)
        self.d_model = state["d_model"]
        self.keys = [k.to(map_location) for k in state["keys"]]
        self.values = [v.to(map_location) for v in state["values"]]
        self.texts = state["texts"]
        self.tool_names = state.get("tool_names", [None] * len(self.keys))
        self.decay_weights = state.get("decay_weights", [1.0] * len(self.keys))
        self.load_state_dict(state["state_dict"])
