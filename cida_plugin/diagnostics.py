import torch

class DebateDiagnostics:
    """
    Validation mechanics to ensure explicit multi-agent deliberation avoids simple
    over-confidence and actually provides semantic divergence over standard training.
    """
    
    @staticmethod
    def expected_calibration_error(p_final: torch.Tensor, y: torch.Tensor, n_bins: int = 10):
        """Computes Expected Calibration Error (ECE) for final probability vector."""
        if y.dim() == 2: # Multi-label
            # If p_final contains logits (values > 1 or < 0 possible), apply sigmoid
            if p_final.max() > 1.0 or p_final.min() < 0.0:
                probs = torch.sigmoid(p_final)
            else:
                probs = p_final
            
            K = probs.size(1)
            total_ece = 0.0
            for k in range(K):
                total_ece += DebateDiagnostics._binary_ece(probs[:, k], y[:, k], n_bins)
            return total_ece / K
            
        # Single-label (original logic)
        confidences, predictions = torch.max(p_final, dim=1)
        accuracies = (predictions == y)
        return DebateDiagnostics._calculate_ece_from_bins(confidences, accuracies, n_bins, p_final.device)

    @staticmethod
    def _binary_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int):
        """Binary ECE for a single class."""
        accuracies = (probs > 0.5) == targets
        return DebateDiagnostics._calculate_ece_from_bins(probs, accuracies, n_bins, probs.device)

    @staticmethod
    def _calculate_ece_from_bins(confidences: torch.Tensor, accuracies: torch.Tensor, n_bins: int, device: torch.device):
        ece = torch.zeros(1, device=device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
        
        for bin_idx in range(n_bins):
            in_bin = (confidences > bin_boundaries[bin_idx]) & (confidences <= bin_boundaries[bin_idx + 1])
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece.item()
        
    @staticmethod
    def persuasion_matrix(b_all: list):
        """
        Calculates how much agents changed their beliefs over time.
        Allows us to track if persuasion is actually occurring.
        Returns tensor of shape (Rounds, Agents) tracking average shifts.
        """
        T = len(b_all)
        if T < 2:
            return None
            
        shifts = []
        for t in range(1, T):
            shift = torch.norm(b_all[t] - b_all[t-1], p=1, dim=-1).mean(dim=0) # (M,)
            shifts.append(shift)
            
        return torch.stack(shifts)

    @staticmethod
    def belief_diversity_curve(b_all: list):
        """
        Measures disagreement level over time. Useful to plot against d_t schedule.
        """
        divergence = []
        for b_t in b_all:
            b_i = b_t.unsqueeze(2)
            b_j = b_t.unsqueeze(1)
            dist = torch.norm(b_i - b_j, p=1, dim=-1).mean()
            divergence.append(dist.item())
        return divergence
