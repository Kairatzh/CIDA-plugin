import torch
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMBackbone(Protocol):
    """
    Protocol defining the interface for the LLM backbones used by CIDA.
    Enables plug-and-play support for transformers, vLLM, llama.cpp, etc.
    """
    @property
    def device(self) -> str:
        ...

    @property
    def dtype(self) -> torch.dtype:
        ...

    @property
    def config(self):
        ...

    def get_last_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract the last-token hidden state of the prompt."""
        ...

    def get_input_embeddings(self):
        """Returns the model's token embedding module or getter."""
        ...

    def generate_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        """Generates tokens starting from embedded representations."""
        ...


class HuggingFaceBackbone:
    """
    HuggingFace transformers implementation of LLMBackbone.
    """
    def __init__(self, model):
        self.model = model

    @property
    def device(self) -> str:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def config(self):
        return self.model.config

    def get_last_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return out.hidden_states[-1][0, -1].float().unsqueeze(0)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def generate_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
