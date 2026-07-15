import torch
import torch.nn as nn
import torch.nn.functional as F

class StoreProxyList(list):
    """A list wrapper that propagates item updates back to the underlying memory store."""
    def __init__(self, items, store, field_name):
        super().__init__(items)
        self.store = store
        self.field_name = field_name

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        entries = self.store.get_all()
        # Resolve negative indexing
        if index < 0:
            index = len(entries) + index
        if 0 <= index < len(entries):
            if self.field_name == "decay_weight":
                self.store.update_decay_weight(index, value)
            else:
                entries[index][self.field_name] = value
                self.store.set_all(entries)

class BaseMemoryStore:
    """Abstract base class for CIDA LatentMemory backends."""
    def clear(self):
        raise NotImplementedError
    def add(self, key, value, text, tool_name, decay_weight=1.0):
        raise NotImplementedError
    def get_all(self):
        raise NotImplementedError
    def set_all(self, entries):
        raise NotImplementedError
    def update_decay_weight(self, index, new_weight):
        raise NotImplementedError

class InMemoryMemoryStore(BaseMemoryStore):
    """In-memory backend (default, fully backwards-compatible)."""
    def __init__(self):
        self.entries = []
    def clear(self):
        self.entries = []
    def add(self, key, value, text, tool_name, decay_weight=1.0):
        self.entries.append({
            "key": key.detach().cpu().squeeze().float(),
            "value": value.detach().cpu().squeeze().float(),
            "text": text,
            "tool_name": tool_name,
            "decay_weight": decay_weight
        })
    def get_all(self):
        return self.entries
    def set_all(self, entries):
        self.entries = list(entries)
    def update_decay_weight(self, index, new_weight):
        if 0 <= index < len(self.entries):
            self.entries[index]["decay_weight"] = new_weight

class RedisMemoryStore(BaseMemoryStore):
    """Redis-backed persistent memory backend."""
    def __init__(self, host="localhost", port=6379, db=0, key_prefix="cida:memory", client=None):
        self.host = host
        self.port = port
        self.db = db
        self.key_prefix = key_prefix
        self._client = client
    @property
    def client(self):
        if self._client is None:
            import redis
            self._client = redis.Redis(host=self.host, port=self.port, db=self.db)
        return self._client
    def clear(self):
        self.client.delete(self.key_prefix)
    def add(self, key, value, text, tool_name, decay_weight=1.0):
        import pickle
        entry = {
            "key": key.detach().cpu().squeeze().float(),
            "value": value.detach().cpu().squeeze().float(),
            "text": text,
            "tool_name": tool_name,
            "decay_weight": decay_weight
        }
        self.client.rpush(self.key_prefix, pickle.dumps(entry))
    def get_all(self):
        import pickle
        raw_items = self.client.lrange(self.key_prefix, 0, -1)
        entries = []
        for item in raw_items:
            entries.append(pickle.loads(item))
        return entries
    def set_all(self, entries):
        import pickle
        self.clear()
        if entries:
            serialized = [pickle.dumps(e) for e in entries]
            self.client.rpush(self.key_prefix, *serialized)
    def update_decay_weight(self, index, new_weight):
        import pickle
        raw_item = self.client.lindex(self.key_prefix, index)
        if raw_item:
            entry = pickle.loads(raw_item)
            entry["decay_weight"] = new_weight
            self.client.lset(self.key_prefix, index, pickle.dumps(entry))

class LatentMemory(nn.Module):
    """
    A lightweight, tensor-based vector store that acts as a 'latent diary'
    to store past planning states and mix them back into current decisions.
    Supports tool tracking, reliability weighting, and pluggable storage backends.
    """
    def __init__(self, d_model, store=None):
        super().__init__()
        self.d_model = d_model
        self.store = store if store is not None else InMemoryMemoryStore()
        
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

    @property
    def keys(self):
        return StoreProxyList([e["key"] for e in self.store.get_all()], self.store, "key")

    @property
    def values(self):
        return StoreProxyList([e["value"] for e in self.store.get_all()], self.store, "value")

    @property
    def texts(self):
        return StoreProxyList([e["text"] for e in self.store.get_all()], self.store, "text")

    @property
    def tool_names(self):
        return StoreProxyList([e["tool_name"] for e in self.store.get_all()], self.store, "tool_name")

    @property
    def decay_weights(self):
        return StoreProxyList([e["decay_weight"] for e in self.store.get_all()], self.store, "decay_weight")

    def clear(self, keep_seeded=True):
        """Reset the latent memory, optionally preserving seeded clues."""
        if keep_seeded:
            new_entries = []
            for e in self.store.get_all():
                if not e["text"].startswith("Called tool"):
                    new_entries.append(e)
            self.store.set_all(new_entries)
        else:
            self.store.clear()

    def add_memory(self, key_tensor, value_tensor, text_desc="", tool_name=None):
        """
        Store a planning state key and value representation along with a description.
        Expects shapes of (1, d_model) or (d_model,).
        """
        self.store.add(key_tensor, value_tensor, text_desc, tool_name, 1.0)

    def retrieve(self, query_tensor, tool_reliability=None):
        """
        Retrieves matching memory traces using cosine similarity
        and returns a gated, attention-weighted context vector.
        If tool_reliability is provided (dict mapping tool_name -> float),
        the attention scores are dynamically scaled based on historical tool reliability.
        """
        device = query_tensor.device
        self.to(device)
        
        entries = self.store.get_all()
        if not entries:
            # Return zero vector if no memories are stored yet
            return torch.zeros_like(query_tensor)

        # Convert lists of tensors to single batched tensors on current device
        keys = [e["key"] for e in entries]
        values = [e["value"] for e in entries]
        decay_weights = [e["decay_weight"] for e in entries]
        tool_names = [e["tool_name"] for e in entries]

        K_stack = torch.stack(keys).to(device) # (N, d_model)
        V_stack = torch.stack(values).to(device) # (N, d_model)

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
            for tn in tool_names:
                multipliers.append(tool_reliability.get(tn, 1.0))
            m_tensor = torch.tensor(multipliers, dtype=scores_scaled.dtype, device=device).unsqueeze(0)
            scores_scaled = scores_scaled * m_tensor

        # Apply Temporal Decay (Latent Garbage Collector)
        d_tensor = torch.tensor(decay_weights, dtype=scores_scaled.dtype, device=device).unsqueeze(0)
        scores_scaled = scores_scaled * d_tensor

        attn_weights = F.softmax(scores_scaled, dim=-1) # (1, N)

        # Update decay weights: Top 2 matches get refreshed (1.0), others decay by 10%
        if len(keys) > 0:
            top_indices = torch.topk(scores_scaled, min(2, len(keys))).indices[0].tolist()
            for i in range(len(decay_weights)):
                if decay_weights[i] >= 900.0:
                    continue
                if i in top_indices:
                    self.store.update_decay_weight(i, 1.0)
                else:
                    self.store.update_decay_weight(i, max(0.1, decay_weights[i] * 0.9))

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
        entries = self.store.get_all()
        if not entries:
            return []
        
        device = query_tensor.device
        keys = [e["key"] for e in entries]
        K_stack = torch.stack(keys).to(device) # (N, d_model)
        
        with torch.no_grad():
            q = self.q_proj(query_tensor)
            k_proj = self.k_proj(K_stack)
            
            # L2 Normalize for true Cosine Similarity
            q_norm = F.normalize(q, p=2, dim=-1)
            k_norm = F.normalize(k_proj, p=2, dim=-1)
            
            # Compute cosine similarities
            scores = torch.matmul(q_norm, k_norm.transpose(0, 1)).squeeze(0) # (N,)
            
            top_k = min(k, len(keys))
            top_vals, top_idx = torch.topk(scores, top_k)
            
            active = []
            for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                active.append((entries[idx]["text"], val))
            return active

    def save_to_file(self, filepath: str):
        """
        Saves the memory keys, values, and texts to a file.
        Uses torch.save to handle tensors alongside strings safely.
        """
        entries = self.store.get_all()
        state = {
            "d_model": self.d_model,
            "keys": [e["key"].cpu() for e in entries],
            "values": [e["value"].cpu() for e in entries],
            "texts": [e["text"] for e in entries],
            "tool_names": [e["tool_name"] for e in entries],
            "decay_weights": [e["decay_weight"] for e in entries],
            "state_dict": self.state_dict()
        }
        torch.save(state, filepath)

    def load_from_file(self, filepath: str, map_location="cpu"):
        """
        Loads the memory state from a file.
        """
        state = torch.load(filepath, map_location=map_location)
        self.d_model = state["d_model"]
        
        keys = state["keys"]
        values = state["values"]
        texts = state["texts"]
        tool_names = state.get("tool_names", [None] * len(keys))
        decay_weights = state.get("decay_weights", [1.0] * len(keys))
        
        entries = []
        for k, v, t, tn, d in zip(keys, values, texts, tool_names, decay_weights):
            entries.append({
                "key": k.to(map_location),
                "value": v.to(map_location),
                "text": t,
                "tool_name": tn,
                "decay_weight": d
            })
        self.store.set_all(entries)
        self.load_state_dict(state["state_dict"])
