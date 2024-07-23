import torch
import torch.nn as nn
import itertools
import numpy as np
from utils import axiom_eval as ae
from utils import data_utils as du



class CondorcetWinnerLoss(nn.Module):
    def __init__(self):
        super(CondorcetWinnerLoss, self).__init__()

    def all_condorcet_committees(self, n_voters, num_candidates, num_winners, candidate_pairs):
        # candidate pairs is flattened
        committees = du.generate_all_committees(num_candidates, num_winners)

        does_condorcet_exist = ae.exists_condorcet_winner(committees, candidate_pairs)

        result = []

        for c in committees:
            if does_condorcet_exist and ae.eval_condorcet_winner(c, candidate_pairs) == 0:
                result.append(c)
        
        return result

    def forward(self, winning_committee, candidate_pairs, n_voters=None, num_winners=None):
        batch_size = len(winning_committee)

        # find all possible winning committees
        # for each winning committee (output from NN), calculate the loss between it and each possible winning committee
        # find the minimum loss (could try max and mean also?)
        num_candidates = len(winning_committee[0])

        losses = []

        for i in range(batch_size):
            wc = winning_committee[i]
            cp = candidate_pairs[i].detach().numpy()
            condorcet_committees = torch.tensor(self.all_condorcet_committees(n_voters, num_candidates, num_winners, cp))
            
            if len(condorcet_committees) == 0:
                all_distances = torch.abs(wc - wc)
                min_distance = torch.min(all_distances)
            else:
                all_distances = torch.abs(wc - condorcet_committees)
                all_distances = torch.sum(all_distances, dim=1)
                min_distance = torch.min(all_distances)
            
            losses.append(min_distance)
        


        losses = torch.stack(losses)
        loss = torch.mean(losses)
        return loss

    """
    # Get the top indices
    _, c_indices = torch.topk(winning_committee, num_winners)
    num_candidates = len(winning_committee[0])


    def sigmoid_gt(a, b, alpha=0.01):
        return torch.sigmoid(alpha * (a - b))
    
    # Initialize an empty list to store the sigmoid results
    sigmoid_results = []

    all_indices = torch.arange(len(winning_committee[0]), device=winning_committee.device)

    # Create a mask for indices not in c_indices
    mask = torch.ones_like(all_indices, dtype=torch.bool)
    mask[c_indices] = False
    remaining_indices = all_indices[mask]


    for i in range(c_indices.size(0)):
        # Gather the element at the current index
        c = c_indices[i]
        for d in remaining_indices:
            gathered_element = candidate_pairs[:, c * num_candidates + d]
            
            # Apply the sigmoid function
            sigmoid_value = sigmoid_gt(gathered_element, n_voters // 2 + 1)

            # Append the result to the list
            sigmoid_results.append(sigmoid_value)

    # Convert the list to a tensor
    sigmoid_results = torch.stack(sigmoid_results)

    # Find the minimum of the sigmoid results
    minimum = torch.min(sigmoid_results)

    # Calculate the mean of the minimum value
    average = torch.mean(minimum)

    # Calculate the absolute value of the mean
    absval = torch.abs(average)

    return absval
    """
    
    """
    def forward(self, winning_committee, candidate_pairs=None, n_voters=None, num_winners=None):
        batch_size, num_candidates = winning_committee.shape
                
        threshold = n_voters / 2.0

        def sigmoid_gt(a, b, alpha=1):
            return torch.sigmoid(alpha * (a - b))
        
        def check_tensor_graph_connection(tensor, tensor_name):
            print(f"Tensor: {tensor_name}")
            print(f"  requires_grad: {tensor.requires_grad}")
            print(f"  is_leaf: {tensor.is_leaf}")
            if tensor.grad_fn is not None:
                print(f"  grad_fn: {tensor.grad_fn}")
            else:
                print("  grad_fn: None")
            print()
                

        def is_condorcet(committee):
            # committee is a batch of committees

            # this will be the total loss of all the committees in the batch
            total = torch.tensor(0.0, requires_grad=True)

            check_tensor_graph_connection(total, "Total")
            
            # iterate over every committee in the batch
            for i, wc in enumerate(committee):
                # extract winning indices
                _, c_indices = torch.topk(wc, num_winners)
                
                # make a set of winning indices in the batch item wc
                c_mask = torch.zeros(num_candidates, dtype=torch.bool, device=c_indices.device)
                c_mask[c_indices] = True
                non_winning_indices = torch.where(~c_mask)[0]

                # this will be the loss of the individual committee, wc
                result = torch.tensor(float('inf'), requires_grad = True)

                # for each c in the committee
                for c in c_indices:
                    # for each d not in the committee
                    for d in non_winning_indices:
                        # calculate the index for P(c, d)
                        cd_idx = c * num_candidates + d

                        # this calculates 1/(1+e^-(P(c, d) - (n // 2 + 1)))
                        sig_score = sigmoid_gt(candidate_pairs[i][cd_idx], threshold)

                        # find the minimum of all of the comparisons
                        # this will be min(min(all d)), same as min of all c, min of all d in your code
                        result = torch.min(result + result * 1e-6, sig_score)
                
                # here we add the loss for this individual committee in the batch to the 
                # total

                total = torch.add(total, result)

                check_tensor_graph_connection(total, "Total")
                print("Looping\n")
            
            # at the end, we return the mean of the losses of the entire batch. 
            # this is the same as torch.mean(all losses of all committees in batch)
                
            result = torch.div(total, batch_size)

            check_tensor_graph_connection(result, "result")

            return result
        
        return is_condorcet(winning_committee)
    """