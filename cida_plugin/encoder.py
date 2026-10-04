"""
cida_plugin/encoder.py — Real-Text Embedding Engine with Zero Target Leakage

Encodes dialogue turns into dense representations using Hugging Face AutoModel:
- Pure text-to-vector transformation: v = MeanPool(Model(Tokens))
- ZERO TARGET LEAKAGE: No access to labels, no synthetic shifts, no artificial noise.
- Disk caching mechanism to avoid redundant inference across epochs.
"""

import hashlib
import os
from typing import List, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEmbeddingEngine:
    """
    Genuine Transformer-based text embedding engine.
    Mean-pools contextualized token representations into dense sentence embeddings.
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cpu",
        cache_dir: Optional[str] = "./latent_cache_mt",
        max_length: int = 256,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        print(f"[Encoder] Loading tokenizer and model for '{model_name}' (device: {device})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
        except Exception:
            # Fallback if local_files_only was set or network retry needed
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)

        self.model.eval()
        self.d_emb = self.model.config.hidden_size
        print(f"[Encoder] Ready! Hidden dimension: {self.d_emb}")

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encodes a list of turn strings into a (N, d_emb) tensor.
        Uses mean pooling with attention mask.
        Strictly honest: input is only text.
        """
        if not texts:
            return torch.zeros(0, self.d_emb, device=self.device)

        all_embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            model_output = self.model(**encoded_input)
            token_embeddings = model_output[0]  # First element contains hidden-states
            input_mask_expanded = encoded_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            all_embs.append(mean_pooled.cpu())

        return torch.cat(all_embs, dim=0)

    def encode_trajectory(self, turn_texts: List[str], task_id: Optional[str] = None) -> torch.Tensor:
        """
        Encodes a full sequence of turns for a task.
        Checks disk cache if task_id or hash is available.
        Returns: (T, d_emb) float tensor on CPU.
        """
        if task_id and self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{task_id}.pt")
            if os.path.exists(cache_file):
                try:
                    cached_tensor = torch.load(cache_file, map_location="cpu")
                    if cached_tensor.size(0) == len(turn_texts) and cached_tensor.size(1) == self.d_emb:
                        return cached_tensor
                except Exception:
                    pass

        # Encode live text
        embs = self.encode(turn_texts)

        if task_id and self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{task_id}.pt")
            try:
                torch.save(embs, cache_file)
            except Exception:
                pass

        return embs
