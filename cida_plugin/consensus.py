import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsensusAggregator(nn.Module):
    """
    Weighted Product of Experts (PoE) consensus based on
    agent uncertainty and explicit historical reliability.

    [v3+] Нормализация шкал: u и ρ приводятся к [0, 1] через min-max
    по агентам внутри каждого примера, гарантируя инвариантность
    весов к аффинным преобразованиям шкал (см. доказательство в docs).
    """

    def __init__(self, beta_u: float = 1.0, beta_r: float = 1.0, h_min: float = 1.0,
                 T_max: float = 3.0, T_min: float = 0.5, max_rho: float = 5.0,
                 normalize_components: bool = True):
        super().__init__()
        self.beta_u = beta_u
        self.beta_r = beta_r
        self.h_min = h_min
        self.T_max = T_max
        self.T_min = T_min
        self.max_rho = max_rho
        self.normalize_components = normalize_components

    @staticmethod
    def _minmax_norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
        """Min-max нормализация по dim (агентам): x → [0, 1].
        Инвариантна к аффинным преобразованиям: f(ax+b) = f(x)."""
        x_min = x.min(dim=dim, keepdim=True)[0]
        x_max = x.max(dim=dim, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + eps)

    def forward(self, b: torch.Tensor, u: torch.Tensor, rho: torch.Tensor,
                epoch_fraction: float = 0.0, multi_label: bool = False):
        """
        b   : (B, M, K) — agent beliefs
        u   : (B, M, 1) — agent uncertainties
        rho : (B, M, 1) — agent historical reliabilities
        epoch_fraction : float in [0, 1]
        multi_label    : bool

        Returns:
            p_final : (B, K) — final consensus distribution
            w       : (B, M) — consensus weights
            L_dom   : scalar — dominance penalty averaged over batch
        """
        # Epoch-aware temperature: high early (soft), low late (hard)
        temperature = self.T_max - (self.T_max - self.T_min) * epoch_fraction

        # Clip reliability to prevent historical dominance (tyranny)
        rho_clipped = torch.clamp(rho, max=self.max_rho)

        # ── [v3+] Нормализация шкал ───────────────────────────────────────────
        # Без этого u ∈ [0, K] и ρ ∈ [0.1, 5.0] несопоставимы,
        # что делает beta_u и beta_r зависимыми от масштаба.
        if self.normalize_components:
            u_norm = self._minmax_norm(u, dim=1)       # (B, M, 1) → [0, 1]
            rho_norm = self._minmax_norm(rho_clipped, dim=1)  # (B, M, 1) → [0, 1]
        else:
            u_norm = u
            rho_norm = rho_clipped

        # w_i = softmax((-β_u · ũ_i + β_r · ρ̃_i) / T)
        scores = -self.beta_u * u_norm + self.beta_r * rho_norm  # (B, M, 1)
        scores = scores.squeeze(-1) / temperature                # (B, M)
        w = F.softmax(scores, dim=-1)                            # (B, M)

        # Product of Experts: log p_final ∝ Σ_i w_i log p_i
        w_expanded = w.unsqueeze(-1)  # (B, M, 1)

        if multi_label:
            # Multi-label: взвешенные log-odds → логиты для BCEWithLogitsLoss
            logits = torch.log(b / (1.0 - b + 1e-9) + 1e-9)
            weighted_logits = (w_expanded * logits).sum(dim=1)
            p_final = weighted_logits
        else:
            log_b = torch.log(b + 1e-9)                   # (B, M, K)
            weighted_log_b = w_expanded * log_b            # (B, M, K)
            log_p_final_unnorm = weighted_log_b.sum(dim=1) # (B, K)
            p_final = F.softmax(log_p_final_unnorm, dim=-1)

        # Dominance loss: L_dom = max(0, H_min - H(w))
        H_w = -(w * torch.log(w + 1e-9)).sum(dim=-1)      # (B,)
        L_dom = F.relu(self.h_min - H_w) * epoch_fraction  # (B,)

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


class ReliabilityTracker(nn.Module):
    """
    [v3] Надёжность агента = его точность на примерах где он был уверен.
    НЕ зависит от других агентов (устранение circular dependency).
    """
    def __init__(self, num_agents: int, rho_decay: float = 0.9):
        super().__init__()
        self.rho_decay = rho_decay
        self.register_buffer("rho", torch.ones(1, num_agents, 1))

    @torch.no_grad()
    def update(self, b_final: torch.Tensor, u_final: torch.Tensor, y_true: torch.Tensor):
        """
        b_final: (B, M, K)
        u_final: (B, M, 1)
        y_true: (B,) или (B, K)
        """
        if y_true is None:
            return

        confidence = 1.0 - u_final  # (B, M, 1)
        
        if y_true.dim() == 2: # Multi-label
            # Точность по каждому ярлыку
            preds = (b_final > 0.5).float()
            correct = (preds == y_true.unsqueeze(1)).float().mean(dim=-1, keepdim=True) # (B, M, 1)
        else: # Single-label
            preds = b_final.argmax(dim=-1)  # (B, M)
            correct = (preds == y_true.unsqueeze(1)).float().unsqueeze(-1)  # (B, M, 1)
        
        # Агент заслуживает доверия только когда был уверен И прав
        signal = confidence * correct - confidence * (1 - correct) * 2.0
        
        batch_rho = signal.mean(dim=0, keepdim=True)  # (1, M, 1)
        
        # Обновляем EMA
        self.rho.data = self.rho_decay * self.rho.data + (1.0 - self.rho_decay) * batch_rho
        self.rho.data = torch.clamp(self.rho.data, min=0.1, max=5.0)
