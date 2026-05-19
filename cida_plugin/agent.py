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

    def get_role_specialization_loss(self, b_all: list, y_true: torch.Tensor) -> torch.Tensor:
        """
        [v3] Явные роли агентов и вспомогательные лоссы.
        
        Роли:
        0: Prosecutor (Прокурор) - ищет признаки класса 1
        1: Defender (Защитник) - ищет признаки класса 0
        2: Skeptic (Скептик) - сомневается в уверенных
        3: Integrator (Интегратор) - синтезирует
        """
        if len(b_all) == 0 or y_true is None:
            return torch.tensor(0.0, device=self.R.device)

        b_final = b_all[-1]  # (B, M, K)
        B, M, K = b_final.shape
        
        if M < 4: # Если агентов меньше, специализация ограничена
            return torch.tensor(0.0, device=self.R.device)

        if y_true.dim() == 2: # Multi-label (B, K)
            # 1. Prosecutor: должен видеть патологии (y=1)
            l_pros = (y_true * (1.0 - b_final[:, 0, :])).mean()
            # 2. Defender: должен видеть отсутствие патологий (y=0)
            l_def = ((1.0 - y_true) * b_final[:, 1, :]).mean()
        else: # Binary (B,)
            # 1. Prosecutor penalty: штраф если не видит класс 1 когда он есть
            mask_y1 = (y_true == 1).float()
            prosecutor_belief_c1 = b_final[:, 0, 1] if K > 1 else b_final[:, 0, 0]
            l_pros = (mask_y1 * (1.0 - prosecutor_belief_c1)).mean()

            # 2. Defender penalty: штраф если не видит класс 0 когда он есть
            mask_y0 = (y_true == 0).float()
            defender_belief_c0 = b_final[:, 1, 0]
            l_def = (mask_y0 * (1.0 - defender_belief_c0)).mean()

        # 3. Skeptic penalty: высокая энтропия (неуверенность)
        skeptic_b = b_final[:, 2, :]
        if K > 1:
            l_skep = - (skeptic_b * torch.log(skeptic_b + 1e-9)).sum(dim=-1).mean()
        else:
            l_skep = - (skeptic_b * torch.log(skeptic_b + 1e-9) + (1-skeptic_b)*torch.log(1-skeptic_b + 1e-9)).mean()
        
        l_skep = torch.clamp(1.0 - l_skep, min=0.0)

        return l_pros + l_def + 0.1 * l_skep
