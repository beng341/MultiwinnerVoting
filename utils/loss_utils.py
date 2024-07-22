import torch
import torch.nn as nn

class CondorcetWinnerLoss(nn.Module):
    def __init__(self):
        super(CondorcetWinnerLoss, self).__init__()

    def forward(self, winning_committee, candidate_pairs=None, n_voters=None, num_winners=None):
        batch_size, num_candidates = winning_committee.shape
        cp = candidate_pairs.view(batch_size, num_candidates, num_candidates)
                
        threshold = n_voters / 2.0  # Use float division for smoother gradients

        def sigmoid_gt(a, b, alpha=1):
            return torch.sigmoid(alpha * (b - a))

        def is_condorcet(committee, cp):
            all_comparisons = []
            for c in range(num_candidates):
                for d in range(num_candidates):
                    if c != d:
                        P_c_d = cp[c, d]
                        c_in_committee = committee[c]
                        d_not_in_committee = 1 - committee[d]
                        comparison = sigmoid_gt(P_c_d, threshold) * c_in_committee * d_not_in_committee
                        # this logic with the torch min is not right
                        all_comparisons.append(comparison)
            
            if all_comparisons:
                return torch.min(torch.stack(all_comparisons))
            else:
                return torch.zeros(1, device=committee.device, requires_grad=True)

        condorcet_loss = []
        for i in range(batch_size):
            wc_i, cp_i = winning_committee[i], cp[i]
            loss_i = is_condorcet(wc_i, cp_i)
            condorcet_loss.append(loss_i)

        condorcet_loss = torch.stack(condorcet_loss)
        final_loss = torch.mean(1.0 - condorcet_loss)
        
        return final_loss