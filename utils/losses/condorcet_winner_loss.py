import torch
import torch.nn as nn
import itertools
import numpy as np
from utils import axiom_eval as ae
from utils import data_utils as du
import torch.nn.functional as F
from functorch import vmap, grad
from torch.autograd import Function

def softmax_w(x, w, t=0.0001):
    logw = torch.log(w.clamp(min=1e-12))
    x = (x + logw) / t
    x_max, _ = torch.max(x, dim=-1, keepdim=True)
    x = x - x_max
    exp_x = torch.exp(x)
    return exp_x / (torch.sum(exp_x, dim=-1, keepdim=True) + 1e-12)

def top_k(x, k):
    batch_size, seq_len = x.shape
    y = torch.zeros_like(x)
    for i in range(k):
        x1 = softmax_w(x, w=(1 - y))
        y = y + x1
    return y
        

class CondorcetWinnerLoss(nn.Module):
    def __init__(self):
        super(CondorcetWinnerLoss, self).__init__()

def all_condorcet_committees(candidate_pairs, committees):
    # candidate pairs is flattened

    does_condorcet_exist = ae.exists_condorcet_winner(committees, candidate_pairs)

    result = []

    for c in committees:
        if does_condorcet_exist and ae.eval_condorcet_winner(c, candidate_pairs) == 0:
            result.append(c)
    
    return result

def all_condorcet_loser_committees(candidate_pairs, committees):
    result = []

    for c in committees:
        if ae.eval_condorcet_loser(c, candidate_pairs) == 0:
            result.append(c)
    
    return result

def all_majority_committees(num_voters, ranked_choice, committees):
    result = []

    for c in committees:
        if ae.eval_majority_axiom(num_voters, c, ranked_choice):
            result.append(c)
    
    return result

def all_majority_loser_committees(num_voters, ranked_choice, committees):
    result = []

    for c in committees:
        if ae.eval_majority_loser_axiom(num_voters, c, ranked_choice):
            result.append(c)
    
    return result

def loss_calculate(output, cp, rc, num_winners, num_candidates, num_voters):
    # find all possible winning committees
    # for each winning committee (output from NN), calculate the loss between it and each possible winning committee
    # find the minimum loss (could try max and mean also?)
    condorcet_losses = []
    condorcet_loser_losses = []
    majority_losses = []
    majority_loser_losses = []

    committees = du.generate_all_committees(num_candidates, num_winners)

    for i in range(len(output)):
        wc = output[i]

        condorcet_committees = torch.tensor(all_condorcet_committees(cp[i], committees))
        condorcet_loser_committees = torch.tensor(all_condorcet_loser_committees(cp[i], committees))
        majority_committees = torch.tensor(all_majority_committees(num_voters, rc[i], committees))
        majority_loser_committees = torch.tensor(all_majority_loser_committees(num_voters, rc[i], committees))

        if len(condorcet_committees) == 0:
            all_distances = torch.abs(wc - wc)
            min_distance = torch.min(all_distances)
        else:
            all_distances = torch.abs(wc - condorcet_committees)
            all_distances = torch.sum(all_distances, dim=1)
            min_distance = torch.min(all_distances)
        
        condorcet_losses.append(min_distance)
        
        if len(condorcet_loser_committees) == 0:
            all_distances = torch.abs(wc - wc)
            min_distance = torch.min(all_distances)
        else:
            all_distances = torch.abs(wc - condorcet_loser_committees)
            all_distances = torch.sum(all_distances, dim=1)
            min_distance = torch.min(all_distances)
        
        condorcet_loser_losses.append(min_distance)

        if len(majority_committees) == 0:
            all_distances = torch.abs(wc - wc)
            min_distance = torch.min(all_distances)
        else:
            all_distances = torch.abs(wc - majority_committees)
            all_distances = torch.sum(all_distances, dim=1)
            min_distance = torch.min(all_distances)
        
        majority_losses.append(min_distance)

        if len(majority_loser_committees) == 0:
            all_distances = torch.abs(wc - wc)
            min_distance = torch.min(all_distances)
        else:
            all_distances = torch.abs(wc - majority_loser_committees)
            all_distances = torch.sum(all_distances, dim=1)
            min_distance = torch.min(all_distances)
        
        majority_loser_losses.append(min_distance)

    condorcet_losses = torch.stack(condorcet_losses)
    condorcet_loss = torch.min(condorcet_losses)

    
    condorcet_loser_losses = torch.stack(condorcet_loser_losses)
    condorcet_loser_loss = torch.min(condorcet_loser_losses)

    majority_losses = torch.stack(majority_losses)
    majority_loss = torch.min(majority_losses)

    majority_loser_losses = torch.stack(majority_loser_losses)
    majority_loser_loss = torch.min(majority_loser_losses)

    return 10*condorcet_loss + 10*condorcet_loser_loss + 10*majority_loss + 10*majority_loser_loss

def cw_loss(output, cp, rc, num_winners, num_candidates, num_voters):

    batch_losses = []

    for k in range(len(output)):
        output_softmax = torch.zeros_like(output[k]).requires_grad_(True)
        for i in range(num_winners):
            logw = torch.log((1-output_softmax).clamp(min=1e-12))
            x = (output[k] + logw) / 0.01
            x_max, _ = torch.max(x, dim=-1, keepdim=True)
            x = x - x_max
            exp_x = torch.exp(x)
            x1 = exp_x / (torch.sum(exp_x, dim=-1, keepdim=True) + 1e-12)
            output_softmax = output_softmax + x1
        
        not_output = 1 - output_softmax

        losses = []

        for i in range(num_candidates):
            for j in range(num_candidates):
                cp_i_j = cp[k, i * num_candidates + j]
                sigval = torch.sigmoid(cp_i_j - num_voters // 2 + 1)
                losses.append(sigval * not_output[j] * output_softmax[i])
        
        stackedlosses = torch.stack(losses)

        batch_losses.append(torch.max(stackedlosses))
    
    batch_losses = torch.stack(batch_losses)
    
    return torch.sum(batch_losses)
        

    """
    output_softmax = torch.zeros_like(output).requires_grad_(True)
    for i in range(num_winners):
        logw = torch.log((1-output_softmax).clamp(min=1e-12))
        x = (output + logw) / 0.0001
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        x = x - x_max
        exp_x = torch.exp(x)
        x1 = exp_x / (torch.sum(exp_x, dim=-1, keepdim=True) + 1e-12)
        output_softmax = output_softmax + x1
    

    not_output = 1 - output_softmax

    print(output_softmax.shape)

    print(output_softmax[0])

    losses = []

    for i in range(num_candidates):
        for j in range(num_candidates):
            cp_i_j = cp[:, i * num_candidates + j]
            sigval = torch.sigmoid(cp_i_j - num_voters // 2 + 1)
            losses.append(sigval * not_output[:, j] * output_softmax[:, i])
    
    stackedlosses = torch.stack(losses)
    lossres = torch.max(stackedlosses, dim=1)

    print(stackedlosses.shape)
    """
    exit(1)



    return torch.min(stackedlosses)




    """
    output_softmax = torch.zeros_like(output).requires_grad_(True)
    for i in range(n_winners):
        logw = torch.log((1-output_softmax).clamp(min=1e-12))
        x = (output + logw) / 0.0001
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        x = x - x_max
        exp_x = torch.exp(x)
        x1 = exp_x / (torch.sum(exp_x, dim=-1, keepdim=True) + 1e-12)
        output_softmax = output_softmax + x1
    
    batch_size, seq_len = output.shape
    loss = torch.zeros(batch_size, device=output.device, requires_grad=True)
    
    for i in range(seq_len):
        for j in range(seq_len):
            loss = loss + output_softmax[:, i] * (1 - output_softmax[:, j]) * F.softplus(cp[:, j * num_candidates + i] - cp[:, i * num_candidates + j])

    mean_loss = torch.mean(loss)
    print(output_softmax[0])
    print(cp[0])
    """
    
    """
    def all_condorcet_committees(self, n_voters, num_candidates, num_winners, candidate_pairs):
        # candidate pairs is flattened
        committees = du.generate_all_committees(num_candidates, num_winners)

        does_condorcet_exist = ae.exists_condorcet_winner(committees, candidate_pairs)

        result = []

        for c in committees:
            if does_condorcet_exist and ae.eval_condorcet_winner(c, candidate_pairs) == 0:
                result.append(c)
        
        return result
    """
        

    """
    def forward(self, c_indices, d_indices, input, n_voters=None, num_winners=None, batch_size=None, num_candidates=None):
        def sigmoid_gt(a, b, alpha=0.1):
            return torch.sigmoid(alpha * (a - b))

        sigmoid_results = torch.tensor(0.0, device=c_indices.device, requires_grad=True)

        for i in range(batch_size):
            c_batch = c_indices[i].to(torch.int64)
            d_batch = d_indices[i].to(torch.int64)

            cmpval = input[i, 64 + c_batch.unsqueeze(1) * num_candidates + d_batch]
            sigval = sigmoid_gt(cmpval, n_voters // 2 + 1)
            min_sigval = torch.max(sigval)
            
            sigmoid_results = torch.add(sigmoid_results, min_sigval)

        return sigmoid_results
    
        







        for i in range(c_indices.size(0)):
            # Gather the element at the current index
            c = c_indices[i]
            for j in range(d_indices.size(0)):
                d = d_indices[j]

                print(c, d)

                gathered_element = input[:, 64+(c * num_candidates + d)]
                
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