import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentSemanticTranslator(nn.Module):
    """
    Module 3 of Kairos OS: Latent-to-Semantic Translator.
    Translates CIDA's latent superposition (Prosecutor vs Defender beliefs)
    into structured semantic concept embeddings that an LLM can understand.
    """
    def __init__(self, d_hidden: int = 128, vocab_size: int = 50000, d_model_llm: int = 4096):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_model_llm = d_model_llm
        
        # Non-linear projection from CIDA latent space to LLM semantic space
        self.latent_to_semantic = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden * 4, d_model_llm)
        )
        
        # Attention pooling to summarize evidence
        self.evidence_attention = nn.MultiheadAttention(embed_dim=d_model_llm, num_heads=8, batch_first=True)
        
    def forward(self, superposition_state: dict, class_names: list = None):
        """
        Args:
            superposition_state: Dict containing 'prosecutor_belief' and 'defender_belief'
                                 Each tensor is of shape (B, num_classes) or (B, d_hidden)
            class_names: List of class names for semantic grounding
            
        Returns:
            semantic_embeddings: (B, Seq, d_model_llm) ready to be prepended to LLM prompt
        """
        if superposition_state is None:
            return None
            
        prosecutor_b = superposition_state.get("prosecutor_belief") # (B, ...)
        defender_b = superposition_state.get("defender_belief")     # (B, ...)
        
        # In a real implementation, b is usually (B, num_classes). 
        # For the epic pipeline, CIDA needs to output the raw hidden states (B, d_hidden) 
        # alongside the probabilities. We assume they are d_hidden here for the projection.
        
        # Protect against shape mismatch if b is just probabilities (B, num_classes)
        # We will pad or project it to d_hidden for this skeleton
        B = prosecutor_b.size(0)
        
        if prosecutor_b.size(-1) != self.d_hidden:
            # Dummy projection if we only received probabilities
            pad_p = torch.zeros(B, self.d_hidden, device=prosecutor_b.device)
            pad_p[:, :prosecutor_b.size(-1)] = prosecutor_b
            prosecutor_b = pad_p
            
            pad_d = torch.zeros(B, self.d_hidden, device=defender_b.device)
            pad_d[:, :defender_b.size(-1)] = defender_b
            defender_b = pad_d

        # Map to LLM dimension
        sem_prosecutor = self.latent_to_semantic(prosecutor_b).unsqueeze(1) # (B, 1, d_llm)
        sem_defender = self.latent_to_semantic(defender_b).unsqueeze(1)     # (B, 1, d_llm)
        
        # Combine hypotheses into a sequence
        semantic_sequence = torch.cat([sem_prosecutor, sem_defender], dim=1) # (B, 2, d_llm)
        
        # Self-attention to find conflicts
        attn_out, _ = self.evidence_attention(semantic_sequence, semantic_sequence, semantic_sequence)
        
        # Add residual
        semantic_sequence = semantic_sequence + attn_out
        
        return semantic_sequence
