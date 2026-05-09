import torch

class DebateDiagnostics:
    """
    Validation mechanics to ensure explicit multi-agent deliberation avoids simple
    over-confidence and actually provides semantic divergence over standard training.
    """
    
    @staticmethod
    def expected_calibration_error(p_final: torch.Tensor, y: torch.Tensor, n_bins: int = 10):
        """Computes Expected Calibration Error (ECE) for final probability vector."""
        confidences, predictions = torch.max(p_final, dim=1)
        accuracies = (predictions == y)
        
        ece = torch.zeros(1, device=p_final.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=p_final.device)
        
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
