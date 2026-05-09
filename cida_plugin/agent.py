import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentState:
    """
    Explicit state representation of a CIDA-Omega Agent.
    This replaces the implicit 'latent only' states with a formal structure.
    """
    s: torch.Tensor       # (B, d)   - Hidden state
    b: torch.Tensor       # (B, K)   - Belief distribution over classes
    u: torch.Tensor       # (B, 1)   - Uncertainty measure
    p: torch.Tensor       # (B, n)   - Evidence pointer (attention distribution over tokens)
    e: torch.Tensor       # (B, d)   - Evidence selected vector
    alpha: torch.Tensor   # (B, K)   - Dirichlet evidence values
    
class RoleEmbeddings(nn.Module):
    """
    Learned role embeddings for M agents to ensure specialization.
    r_i is the role embedding of agent i.
    """
    def __init__(self, num_agents: int, dim: int):
        super().__init__()
        self.num_agents = num_agents
        self.dim = dim
        self.R = nn.Parameter(torch.randn(num_agents, dim) / (dim ** 0.5))
        
    def forward(self) -> torch.Tensor:
        """
        Returns the role embeddings matrix.
        Shape: (M, d)
        """
        return self.R
        
    def get_role_loss(self) -> torch.Tensor:
        """
        Orthogonality regularization: L_role = || R_norm R_norm^T - I ||_F^2
        Ensures agents maintain fundamentally different functional roles (e.g. skeptic, supporter)
        """
        # Normalize to encourage cosine orthogonality
        R_norm = torch.nn.functional.normalize(self.R, p=2, dim=1)
        identity = torch.eye(self.num_agents, device=self.R.device)
        return torch.norm(torch.mm(R_norm, R_norm.t()) - identity, p='fro') ** 2
