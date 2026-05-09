import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsensusAggregator(nn.Module):
    """
    Implements Weighted Product of Experts (PoE) consensus based on 
    agent uncertainty and explicit historical reliability.
    """
    def __init__(self, beta_u: float = 1.0, beta_r: float = 1.0, h_min: float = 1.0, 
                 T_max: float = 3.0, T_min: float = 0.5, max_rho: float = 5.0):
        super().__init__()
        self.beta_u = beta_u
        self.beta_r = beta_r
        self.h_min = h_min
        self.T_max = T_max
        self.T_min = T_min
        self.max_rho = max_rho
        
    def forward(self, b: torch.Tensor, u: torch.Tensor, rho: torch.Tensor, epoch_fraction: float = 0.0):
        """
        b: (B, M, K) - agent beliefs
        u: (B, M, 1) - agent uncertainties
        rho: (B, M, 1) - agent historical EMA reliabilities
        epoch_fraction: float in [0, 1] - training progress for temperature scheduling
        Returns:
            p_final: (B, K) - final consensus distribution
            w: (B, M) - consensus weights used
            L_dom: (1) - dominance penalty averaged over batch
        """
        # Epoch-aware temperature: high early (soft), low late (hard)
        temperature = self.T_max - (self.T_max - self.T_min) * epoch_fraction
        
        # Clip reliability to prevent historical dominance (tyranny)
        rho_clipped = torch.clamp(rho, max=self.max_rho)
        # Calculate weights w_i = softmax((-beta_u * u_i + beta_r * rho_i) / T)
        scores = -self.beta_u * u + self.beta_r * rho_clipped # (B, M, 1)
        scores = scores.squeeze(-1) / temperature # (B, M)
        w = F.softmax(scores, dim=-1) # (B, M)
        
        # Product of Experts: log p_final \propto sum_i w_i \log p_i
        log_b = torch.log(b + 1e-9) # (B, M, K)
        w_expanded = w.unsqueeze(-1) # (B, M, 1)
        
        weighted_log_b = w_expanded * log_b # (B, M, K)
        log_p_final_unnorm = weighted_log_b.sum(dim=1) # (B, K)
        
        # Normalize in log space to get final probabilities
        p_final = F.softmax(log_p_final_unnorm, dim=-1) # (B, K)
        
        # Dominance loss: L_dom = max(0, H_min - H(w))
        # Scale by epoch_fraction: weak early, strong late
        H_w = -(w * torch.log(w + 1e-9)).sum(dim=-1) # (B)
        L_dom = F.relu(self.h_min - H_w) * epoch_fraction # (B)
        
        return p_final, w, L_dom.mean()

class HaltingPredictor(nn.Module):
    """
    Budget-aware halting mechanism predicting whether the network should stop thinking.
    """
    def __init__(self, d_hidden: int, num_agents: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_agents * d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1)
        )
        
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B, M, d_hidden)
        Returns h_t: (B, 1) - Probability of halting.
        """
        B, M, d = s.shape
        s_flat = s.view(B, -1)
        logits = self.proj(s_flat)
        h_t = torch.sigmoid(logits)
        return h_t
