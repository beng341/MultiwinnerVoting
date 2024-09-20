import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_ops.MultiWinnerVotingRule import MultiWinnerVotingRule

from sklearn.metrics import accuracy_score
from os.path import isfile, join
import torch
import torch.nn as nn
from pref_voting.generate_profiles import generate_profile as gen_prof
from . import data_utils as du
import pandas as pd
import numpy as np
import torch.nn.functional as F
import itertools
import itertools
from torch.autograd import Function


def get_default_parameter_value_sets(m=False, n=False, train_size=False, num_winners=False, pref_dists=False,
                                     features=False, losses=False, networks_per_param=False):
    """
    Return a tuple containing all default values for all parameters used. Made to ease consistency between training
    and evaluation steps. Returns a number of values equal to the number of parameters set True so ensure values
    are set correctly upon return.
    :param m:
    :param n:
    :param train_size:
    :param num_winners:
    :param pref_dists:
    :param features:
    :param losses:
    :param networks_per_param: Only value which is not a list but is an integer instead
    :return:
    """

    results = []
    if m:
        m_all = [8]  # all numbers of candidates
        results.append(m_all)

    if n:
        n_all = [100]  # all numbers of voters
        results.append(n_all)

    if train_size:
        train_size_all = [10000]  # training size
        results.append(train_size_all)
    if num_winners:
        num_winners = [3]
        results.append(num_winners)
    if pref_dists:
        pref_dist_all = [
            #"stratification__args__weight=0.5",
            "URN-R",
            "IC",
            #"IAC",
            "MALLOWS-RELPHI-R",
            "single_peaked_conitzer",
            # "single_peaked_walsh",
            # "single_peaked_circle",
            # "euclidean__args__dimensions=2_space=uniform",
            # "euclidean__args__dimensions=3_space=uniform",
            # "euclidean__args__dimensions=2_space=ball",
            # "euclidean__args__dimensions=3_space=ball",
            # "euclidean__args__dimensions=2_space=gaussian",
            # "euclidean__args__dimensions=3_space=gaussian",
            # "euclidean__args__dimensions=2_space=sphere",
            # "euclidean__args__dimensions=3_space=sphere",
        ]
        results.append(pref_dist_all)
    if features:
        # feature_set_all = ["b", "c", "r", "br", "bc",  "cr", "bcr"]
        feature_set_all = ["bcr"]  # list of features to learn from (two letters means both features appended together)
        results.append(feature_set_all)
    if losses:
        losses_all = [
            nn.L1Loss(),
            #nn.MSELoss(),
            #nn.CrossEntropyLoss(),
            # nn.CTCLoss,                       # Doesn't work immediately
            # nn.NLLLoss,                       # Doesn't work immediately
            #nn.PoissonNLLLoss(),
            # nn.GaussianNLLLoss,               # Doesn't work immediately
            #nn.KLDivLoss(),
            # nn.BCELoss,                       # Doesn't work immediately
            #nn.BCEWithLogitsLoss(),
            # nn.MarginRankingLoss,             # Doesn't work immediately
            #nn.HingeEmbeddingLoss(),
            # nn.MultiLabelMarginLoss,          # Doesn't work immediately
            # nn.HuberLoss(),
            #nn.SmoothL1Loss(),
            #nn.SoftMarginLoss(),
            #nn.MultiLabelSoftMarginLoss(),
            # nn.CosineEmbeddingLoss,           # Doesn't work immediately
            # nn.MultiMarginLoss,               # Doesn't work immediately
            # nn.TripletMarginLoss,             # Doesn't work immediately
            # nn.TripletMarginWithDistanceLoss  # Doesn't work immediately
        ]
        results.append(losses_all)

    if networks_per_param:
        networks_per_param_set = 1  # How many networks to learn for each combination of parameters
        results.append(networks_per_param_set)

    return results


def load_model(model_path):
    """

    :param model_path:
    :return:
    """
    checkpoints = torch.load(model_path)

    adjusted_state_dict = {f"model.{k}": v for k, v in
                           checkpoints['model_state_dict'].items()}  # feels messy but seems to work

    num_candidates = checkpoints['num_candidates']
    num_voters = checkpoints['num_voters']
    num_winners = checkpoints['n_winners']
    config = checkpoints['config']
    kwargs = checkpoints['kwargs']
    
    model = MultiWinnerVotingRule(num_candidates, num_voters, num_winners, config, **kwargs)
    model.load_state_dict(adjusted_state_dict)
    model.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    model.eval()

    return model


def saved_model_paths(n, m, pref_dist, axioms, features, num_models, loss):
    """

    Generate and return list of strings containing the paths to a group of saved models (or, where they would be
    if models with those parameters exist).
    :param n:
    :param m:
    :param pref_dist:
    :param features:
    :param rule:
    :param num_models:
    :return:
    """
    base_model_path = "trained_networks"
    model_paths = []
    for idx in range(num_models):
        model_paths.append(
            f"{base_model_path}/NN-num_voters={n}-m={m}-pref_dist={pref_dist}-axioms={axioms}-features={features}-loss={str(loss)}-idx={idx}-.pt"
        )
    return model_paths


def features_from_column_names(df, column_names):
    """
    Given a dataframe containing all relevant features, extract all given columns. Append the same row of each column
    into a single list per row that serves as feature vectors.
    :param df:
    :param column_names:
    :return:
    """
    if not isinstance(column_names, list):
        column_names = [column_names]


    all_feature_cols = []
    for column in column_names:
        features = [eval(elem) for elem in df[column].tolist()]
        all_feature_cols.append(features)

    all_rows = []
    for row_idx in range(len(all_feature_cols[0])):
        # all columns had better have the same number of rows
        row = []
        for col in all_feature_cols:
            row += col[row_idx]
        all_rows.append(row)
    return all_rows


def features_from_column_abbreviations(df, abbs):
    """
    Given dataframe features corresponding to letters ("b", "c", "r"), return the matching feature values.
    :param df:
    :param abbs:
    :return:
    """
    features = {
        # "c": "candidate_pairs",
        "c": "candidate_pairs-normalized-no_diagonal",
        # "r": "rank_matrix",
        "r": "rank_matrix-normalized",
        "b": "binary_pairs-no_diagonal"
    }
    used_features = [features[c] for c in abbs]
    return features_from_column_names(df, used_features)


def feature_names_from_column_abbreviations(abbs):
    """
    Given expanded feature names corresponding to letters ("b", "c", "r"), return the matching feature values.
    :param abbs:
    :return:
    """
    features = {
        #"c": "candidate_pairs",
        "c": "candidate_pairs-normalized-no_diagonal",
        #"r": "rank_matrix",
        "r": "rank_matrix-normalized",
        "b": "binary_pairs-no_diagonal"
    }
    used_features = [features[c] for c in abbs]
    return used_features


def tied_winner_accuracy(df, y_pred, winner_column, tied_winner_column):
    """
    :param df:
    :param y_pred:
    :param winner_column:
    :param tied_winner_column:
    :return:
    """
    y_true = df[winner_column].tolist()
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    y_true_tie = df[tied_winner_column].tolist()
    correct = 0
    for idx, (y_t, y_p) in enumerate(zip(y_true_tie, y_pred)):
        possible_winners = eval(y_t)
        if y_p in possible_winners:
            correct += 1
        if y_true[idx] not in possible_winners:
            print(f"{y_true[idx]} not in {possible_winners} - SOMETHING BAD HAPPENED WITH TIED WINNERS")
            exit()
    tied_acc = correct / len(y_true_tie)
    return acc, tied_acc


def targets_from_column_names(df, target_columns="all"):
    """
    Return a list containing the names of winners and tied winners for the given target rules.
    Eventually, useful to specify target rules in more natural language (e.g. ["borda", "copeland", "plurality"]) but,
    for now, I just want to get it up and running with the default "all" argument.
    :param df:
    :param target_columns:
    :return:
    """
    # Makes the assumption that the bit before the dash in the names is the same (will have the same sort order)

    single_winners = set()
    tied_winners = set()
    if isinstance(target_columns, str) and "all".casefold() == target_columns.casefold():
        # get all columns of the form "XYZ-single_winner" and "XYZ-tied_winners"
        column_names = df.columns
        for cn in column_names:
            if "-single_winner" in cn:
                single_winners.add(cn)
            elif "-tied_winners" in cn:
                tied_winners.add(cn)
    elif isinstance(target_columns, list) and target_columns[0] == "all".casefold():
        # get all columns of the form "XYZ-single_winner" and "XYZ-tied_winners"
        column_names = df.columns
        for cn in column_names:
            if "-single_winner" in cn:
                single_winners.add(cn)
            elif "-tied_winners" in cn:
                tied_winners.add(cn)
    elif isinstance(target_columns, list):
        column_names = df.columns.tolist()
        for tc in target_columns:
            for cn in column_names:
                if f"{tc}-single_winner".casefold() == cn.casefold():
                    single_winners.add(cn)
                elif f"{tc}-tied_winners".casefold() == cn.casefold():
                    tied_winners.add(cn)
    elif isinstance(target_columns, str):
        # One rule whose winners we wish to extract
        column_names = df.columns.tolist()
        for cn in column_names:
            if f"{target_columns}-single_winner".casefold() == cn.casefold():
                single_winners.add(cn)
            elif f"{target_columns}-tied_winners".casefold() == cn.casefold():
                tied_winners.add(cn)
    single_winners = list(single_winners)
    tied_winners = list(tied_winners)
    single_winners.sort()
    tied_winners.sort()
    return single_winners, tied_winners


def generate_viol_df(profiles):
    # Generate new profiles
    data = []

    for profile in profiles:
        candidate_pairs = du.candidate_pairs_from_profiles(eval(profile))
        binary_candidate_pairs = du.binary_candidate_pairs_from_profiles(eval(profile))
        rank_counts = du.rank_counts_from_profiles(eval(profile))

        data.append([profile, candidate_pairs, binary_candidate_pairs, rank_counts])

    df = pd.DataFrame(data, columns=[
        'profiles',
        'candidate_pairs',
        'binary_pairs-no_diagonal',
        'rank_matrix'
    ])

    return df

def dgt(a, b, alpha=10, buf=1e-10):
        return torch.sigmoid(alpha * (a - b + buf))


def majority_winner_loss(winning_committee, n_voters, num_winners, rank_counts):    
    batch_size = winning_committee.shape[0]
    num_candidates = int(rank_counts.shape[1]**0.5)
    rank_counts_matrix = rank_counts.view(batch_size, num_candidates, num_candidates)
    rank_counts_matrix = rank_counts_matrix.clone().detach().requires_grad_(True)
    
    # Get the winning committee in one-hot encoded format
    _, topk_indices = torch.topk(winning_committee, num_winners, dim=1)
    winners = torch.zeros_like(winning_committee)
    W = torch.scatter(winners, 1, topk_indices, 1.0)
    
    all_c_terms = []
    for c in range(num_candidates):
        # Calculate a for each candidate c
        a = 1 - dgt(rank_counts_matrix[:, c, 0], n_voters // 2 + 1)
        
        # Calculate b for each candidate c
        c_h = torch.zeros((batch_size, num_candidates), dtype=torch.float32, device=winning_committee.device)
        c_h[:, c] = 1.0
        b = torch.sum(c_h * W, dim=1)
        
        # Expand dimensions of a and b for correct broadcasting
        a = a.unsqueeze(1)
        b = b.unsqueeze(1)

        # Max of a and b for each candidate c
        c_term = torch.max(a, b)
        all_c_terms.append(c_term)
    
    all_c_terms = torch.cat(all_c_terms, dim=1)
    loss = 1 - torch.min(all_c_terms, dim=1)[0]
    return loss.mean()


def majority_loser_loss(winning_committee, n_voters, num_winners, rank_counts):
    batch_size = winning_committee.shape[0]
    num_candidates = int(rank_counts.shape[1]**0.5)
    rank_counts_matrix = rank_counts.view(batch_size, num_candidates, num_candidates)
    
    _, topk_indices = torch.topk(winning_committee, num_winners, dim=1)
    winners = torch.zeros_like(winning_committee)
    W = torch.scatter(winners, 1, topk_indices, 1.0)

    all_c_terms = []
    for c in range(num_candidates):
        a = 1 - dgt(rank_counts_matrix[:, c, -1], n_voters // 2 + 1)
        c_h = torch.zeros((batch_size, num_candidates), dtype=torch.float32)
        c_h[:, c] = 1.0
        b = torch.sum(c_h * (1-W), dim=1)
        a = a.unsqueeze(1)
        b = b.unsqueeze(1)
        c_term = torch.max(a, b)
        all_c_terms.append(c_term)
    
    all_c_terms = torch.cat(all_c_terms, dim=1)
    loss = 1 - torch.min(all_c_terms, dim=1)[0]
    return loss.mean()


def _is_condorcet_loss(committee_tensor, candidate_pairs, n_voters, k=2):
    """
    Should have value close to 1 for each winning committee if it satisfies condorcet axiom. 0 otherwise.
    :return:
    """
    all_c_comparisons = []
    # Go through each candidate c in the committee
    # for c in torch.nonzero(committee_tensor).squeeze(1):
    c_scores, c_indices = torch.topk(committee_tensor, k, dim=1, largest=True)
    # for c, idx in zip(c_scores, c_indices):
    for c_idx in c_indices:
    # for c, indices in torch.topk(committee_tensor, k, dim=1, largest=True):
    #     c = top_scores[idx]
        all_d_comparisons = []

        # Go through each candidate d not in the committee
        # for d in torch.nonzero(1 - committee_tensor).squeeze(1):
        # for d in torch.topk(committee_tensor, k, dim=1, largest=False):
        d_scores, d_indices = torch.topk(committee_tensor, k, dim=1, largest=False)
        for d_idx in d_indices:
            if candidate_pairs.dim() == 2:
                candidate_pairs = candidate_pairs.unsqueeze(0)
            P_c_d = candidate_pairs[:, c_idx, d_idx]
            P_c_d = candidate_pairs[:, c_idx, d_idx]
            sigmoid_result = dgt(P_c_d, n_voters/2)
            # sigmoid_result = mu_P(c.item(), d.item(), candidate_pairs, n_voters)
            all_d_comparisons.append(sigmoid_result)

        # Find the minimum of all d comparisons
        min_d_comparison, _ = torch.min(torch.stack(all_d_comparisons), dim=0)
        all_c_comparisons.append(min_d_comparison)

    return torch.min(torch.stack(all_c_comparisons), dim=0)[0]

def condorcet_winner_loss(winning_committee, possible_committees, n_voters, num_winners, candidate_pairs):

    all_committee_condorcet_chances = []

    batch_size = winning_committee.shape[0]
    num_candidates = int(candidate_pairs.shape[1] ** 0.5)
    candidate_pairs = candidate_pairs.view(batch_size, num_candidates, num_candidates)
    candidate_pairs = candidate_pairs.clone().detach().requires_grad_(True)

    winning_committee = winning_committee.clone().detach().requires_grad_(True)

    # _, topk_indices = torch.topk(winning_committee, n_winners, dim=1)
    # winners = torch.zeros_like(winning_committee)
    # winning_committee = torch.scatter(winners, 1, topk_indices, 1.0)

    sample_committee_test = [0.8, 0.9, -0.4, 0.1, 0.4]
    pattern_tensor = torch.tensor(sample_committee_test, dtype=torch.float32, requires_grad=True)
    repeated_pattern = pattern_tensor.unsqueeze(0).repeat(batch_size, 1)
    repeated_pattern.requires_grad_(True)
    sample_test_loss = _is_condorcet_loss(repeated_pattern, candidate_pairs, n_voters)
    # _, topk_indices = torch.topk(winning_committee, n_winners, dim=1)
    # winners = torch.zeros_like(winning_committee)
    # winning_committee = torch.scatter(winners, 1, topk_indices, 1.0)

    sample_committee_test = [0.8, 0.9, -0.4, 0.1, 0.4]
    pattern_tensor = torch.tensor(sample_committee_test, dtype=torch.float32, requires_grad=True)
    repeated_pattern = pattern_tensor.unsqueeze(0).repeat(batch_size, 1)
    repeated_pattern.requires_grad_(True)
    sample_test_loss = _is_condorcet_loss(repeated_pattern, candidate_pairs, n_voters)

    # Iterate over all possible committees
    # This value will be the same for all input tensors so should
    for committee in possible_committees:
        committee_tensor = torch.tensor(committee, dtype=torch.float32, requires_grad=True)
        # all_c_comparisons = _is_condorcet_loss(committee_tensor, candidate_pairs, n_voters)
        #
        # # Go through each candidate c in the committee
        # for c in torch.nonzero(committee_tensor).squeeze(1):
        #     all_d_comparisons = []
        #
        #     # Go through each candidate d not in the committee
        #     for d in torch.nonzero(1 - committee_tensor).squeeze(1):
        #         sigmoid_result = mu_P(c.item(), d.item(), candidate_pairs, n_voters)
        #         all_d_comparisons.append(sigmoid_result)
        #
        #     # Find the minimum of all d comparisons
        #     min_d_comparison = torch.min(torch.stack(all_d_comparisons))
        #     all_c_comparisons.append(min_d_comparison)

        # Determine if the committee is a Condorcet committee
        # is_condorcet = torch.min(torch.stack(all_c_comparisons))
        is_condorcet = _is_condorcet_loss(committee_tensor, candidate_pairs, n_voters)
        all_committee_condorcet_chances.append(is_condorcet)

    # Get the maximum Condorcet chance for all possible committees (i.e. "is any committee Condorcet?")

    # Get the maximum Condorcet chance for all possible committees (i.e. "is any committee Condorcet?")
    max_condorcet_chance, _ = torch.max(torch.stack(all_committee_condorcet_chances), dim=0)
    #print(max_condorcet_chance)
    non_condorcet_chance = 1 - max_condorcet_chance
    # non_condorcet_chance = non_condorcet_chance.repeat(batch_size)

    # Check if the winning committee is a Condorcet committee
    winning_committee_condorcet_chances = []
    for idx, wc in enumerate(winning_committee):
        winning_committee_tensor = wc.clone().detach().requires_grad_(True)
        # all_c_comparisons = []
        #
        # for c in torch.nonzero(winning_committee_tensor).squeeze(1):
        #     all_d_comparisons = []
        #
        #     for d in torch.nonzero(1 - winning_committee_tensor).squeeze(1):
        #         sigmoid_result = mu_P(c.item(), d.item(), candidate_pairs, n_voters)
        #         all_d_comparisons.append(sigmoid_result)
        #
        #     min_d_comparison = torch.min(torch.stack(all_d_comparisons))
        #     all_c_comparisons.append(min_d_comparison)
        #
        # is_condorcet = torch.min(torch.stack(all_c_comparisons))
        cp = candidate_pairs[idx]
        is_condorcet = _is_condorcet_loss(winning_committee_tensor, cp, n_voters)
        #winning_committee_tensor.retain_grad()
        winning_committee_condorcet_chances.append(is_condorcet)

    # Get the maximum Condorcet chance for the winning committee
    winning_committee_condorcet_chances = torch.stack(winning_committee_condorcet_chances)
    #print(winning_committee_condorcet_chances)
    winning_committee_condorcet_chances = winning_committee_condorcet_chances.squeeze()
    # max_winning_committee_condorcet_chance = torch.max(torch.stack(winning_committee_condorcet_chances))

    sample_committee_test = [0.8, 0.9, -0.4, 0.1, 0.4]
    pattern_tensor = torch.tensor(sample_committee_test, dtype=torch.float32, requires_grad=True)
    repeated_pattern = pattern_tensor.unsqueeze(0).repeat(batch_size, 1)
    repeated_pattern.requires_grad_(True)
    sample_test_loss = _is_condorcet_loss(repeated_pattern, cp, n_voters)

    sample_committee_test = [0.8, 0.9, -0.4, 0.1, 0.4]
    pattern_tensor = torch.tensor(sample_committee_test, dtype=torch.float32, requires_grad=True)
    repeated_pattern = pattern_tensor.unsqueeze(0).repeat(batch_size, 1)
    repeated_pattern.requires_grad_(True)
    sample_test_loss = _is_condorcet_loss(repeated_pattern, cp, n_voters)


    # Return the final result
    loss = torch.max(non_condorcet_chance, winning_committee_condorcet_chances)
    loss = 1 - loss
    #loss.retain_grad()
    return loss.mean()


    # create list, all committee condorcet chances
    # iterate over all possible committees
    # determine whether or not a committee is a condorcet committee. To do this:
        # initialize array, all c comparisons
        # go through each c in the committee
            # initialize array, all d comparisons
            # go through each d not in the committee
                # calculate sigmoid(candidate_pairs[c][d] - (n_voters // 2 + 1))
                # add sigmoid result to all d comparisons
            # find the minimum of all d comparisons
            # append minimum of all d comparisons to all c comparisons
        # is_condorcet is now the minimum of all c comparisons
    # append the is_condorcet score to all committee condorcet chances
    # set is_condorcet_chance to the max of all committee condorcet chances, this will be variable a
    # now we need to find whether or not the winning_committee input is a condorcet committee, so repeat the above to find the loss of that
    # then return 1 - max(1-a, b)
"""
def condorcet_winner_loss(winning_committee, candidate_pairs, n_voters, n_winners):
    batch_size, num_candidates = winning_committee.shape
    
    def is_condorcet(committee):
        mask = committee.unsqueeze(1) - committee.unsqueeze(2)
        cp = candidate_pairs.view(batch_size, num_candidates, num_candidates)
        comparisons = torch.where(mask != 0, cp * mask.sign(), torch.zeros_like(cp))
        wins = (comparisons > n_voters / 2).float()
        condorcet_score = (wins.sum(dim=2) == (num_candidates - 1)).float()
        return condorcet_score.mean(dim=1)  # Average across batch

    # Compute Condorcet property for winning committee
    winning_condorcet = is_condorcet(winning_committee)

    # Compute Condorcet property for all possible committees
    all_committees = torch.combinations(torch.arange(num_candidates), r=n_winners)
    all_condorcet = torch.stack([is_condorcet(winning_committee.new_zeros(batch_size, num_candidates).index_fill_(1, comm, 1)) for comm in all_committees])

    # Compute loss
    loss = 1 - (winning_condorcet * (1 - all_condorcet.any(dim=0).float())).mean()
    
    # Ensure the loss requires gradients
    if not loss.requires_grad:
        loss = loss.clone().detach().requires_grad_(True)
    
    return loss
"""


def condorcet_winner_loss_josh(winning_committee, possible_committees, n_voters, num_winners, candidate_pairs):
    def mu_P(c, d, candidate_pairs_matrix, n):
        P_c_d = candidate_pairs_matrix[:, c, d]
        return torch.sigmoid(P_c_d - (n // 2 + 1))

    all_committee_condorcet_chances = []

    batch_size = winning_committee.shape[0]
    num_candidates = int(candidate_pairs.shape[1] ** 0.5)
    candidate_pairs = candidate_pairs.view(batch_size, num_candidates, num_candidates)

    _, topk_indices = torch.topk(winning_committee, num_winners, dim=1)
    winners = torch.zeros_like(winning_committee)
    winning_committee = torch.scatter(winners, 1, topk_indices, 1.0)

    # Iterate over all possible committees
    for committee in possible_committees:
        committee_tensor = torch.tensor(committee, dtype=torch.float32, requires_grad=True)
        all_c_comparisons = []

        # Go through each candidate c in the committee
        for c in torch.nonzero(committee_tensor).squeeze(1):
            all_d_comparisons = []

            # Go through each candidate d not in the committee
            for d in torch.nonzero(1 - committee_tensor).squeeze(1):
                sigmoid_result = mu_P(c.item(), d.item(), candidate_pairs, n_voters)
                all_d_comparisons.append(sigmoid_result)

            # Find the minimum of all d comparisons
            min_d_comparison = torch.min(torch.stack(all_d_comparisons))
            all_c_comparisons.append(min_d_comparison)

        # Determine if the committee is a Condorcet committee
        is_condorcet = torch.min(torch.stack(all_c_comparisons))
        all_committee_condorcet_chances.append(is_condorcet)

    # Get the maximum Condorcet chance for all possible committees
    max_condorcet_chance = torch.max(torch.stack(all_committee_condorcet_chances))

    # Check if the winning committee is a Condorcet committee
    winning_committee_condorcet_chances = []
    for wc in winning_committee:
        winning_committee_tensor = torch.tensor(wc, dtype=torch.float32, requires_grad=True)
        all_c_comparisons = []

        for c in torch.nonzero(winning_committee_tensor).squeeze(1):
            all_d_comparisons = []

            for d in torch.nonzero(1 - winning_committee_tensor).squeeze(1):
                sigmoid_result = mu_P(c.item(), d.item(), candidate_pairs, n_voters)
                all_d_comparisons.append(sigmoid_result)

            min_d_comparison = torch.min(torch.stack(all_d_comparisons))
            all_c_comparisons.append(min_d_comparison)

        is_condorcet = torch.min(torch.stack(all_c_comparisons))
        winning_committee_condorcet_chances.append(is_condorcet)

    # Get the maximum Condorcet chance for the winning committee
    max_winning_committee_condorcet_chance = torch.max(torch.stack(winning_committee_condorcet_chances))

    # Return the final result
    loss = 1 - torch.max(torch.tensor([1 - max_condorcet_chance, max_winning_committee_condorcet_chance], requires_grad=True))
    return loss



    """
    def is_condorcet_loss(w, candidate_pairs_matrix, n_voters):
        batch_size = w.shape[0]
        num_candidates = w.shape[1]
        all_c_comparisons = []

        for c in range(num_candidates):
            c_in_committee = w[:, c]
            if c_in_committee.sum() > 0:  # Check if c is in the committee in any batch
                all_d_comparisons = []
                for d in range(num_candidates):
                    d_in_committee = w[:, d]
                    if (d_in_committee.sum() == 0):  # Check if d is not in the committee in any batch
                        p = mu_P(c, d, candidate_pairs_matrix, n_voters)
                        all_d_comparisons.append(p)
                
                if all_d_comparisons:
                    min_d = torch.stack(all_d_comparisons, dim=1).mean(dim=1)  # Using mean as a smooth approximation
                    all_c_comparisons.append(min_d)
        
        if all_c_comparisons:
            return torch.stack(all_c_comparisons, dim=1).mean(dim=1)  # Using mean as a smooth approximation
        else:
            return torch.zeros(batch_size, device=w.device)  # Return 0 if no valid comparisons

    batch_size = winning_committee.shape[0]
    num_candidates = int(candidate_pairs.shape[1] ** 0.5)
    candidate_pairs_matrix = candidate_pairs.view(batch_size, num_candidates, num_candidates)
    
    _, topk_indices = torch.topk(winning_committee, n_winners, dim=1)
    winners = torch.zeros_like(winning_committee)
    W = torch.scatter(winners, 1, topk_indices, 1.0)
    
    all_possible_condorcet_chances = []

    for pc in possible_committees:
        pc_tensor = torch.tensor(pc, dtype=torch.float32, device=winning_committee.device).unsqueeze(0).expand(batch_size, -1)
        p = is_condorcet_loss(pc_tensor, candidate_pairs_matrix, n_voters)
        all_possible_condorcet_chances.append(p)
    
    is_condorcet = torch.stack(all_possible_condorcet_chances, dim=1).mean(dim=1)  # Using mean as a smooth approximation
    a = 1 - is_condorcet
    b = is_condorcet_loss(W, candidate_pairs_matrix, n_voters)

    return 1 - torch.max(a, b).mean()



    
    max_inner_sums = []
    
    for i in range(batch_size):
        w = W[i]
        inner_sums = []
        for c in range(num_candidates):
            if w[c] == 1:
                inner_sum = 0
                for d in range(num_candidates):
                    if w[d] == 0:  # d is not in the committee
                        inner_sum += torch.exp(-mu_P(c, d, n_voters)[i])
                inner_sums.append(torch.log(inner_sum))
        if inner_sums:
            inner_sums = torch.stack(inner_sums)
            max_inner_sums.append(torch.sum(torch.exp(inner_sums)))
    
    if max_inner_sums:
        max_inner_sums = torch.stack(max_inner_sums)
        max_outer_sum = torch.max(1 - torch.log(max_inner_sums), -torch.log(max_inner_sums))
        loss = 1 - max_outer_sum.mean()
    else:
        loss = torch.tensor(0.0, requires_grad=True)
    """


    """
    threshold = n_voters // 2
    condorcet_matrix = candidate_pairs_matrix.clone().detach()
    mask = torch.eye(num_candidates, device=candidate_pairs.device).bool()
    condorcet_matrix[:, mask] = float('inf')
    
    # Identify Condorcet winners
    condorcet_winners = (condorcet_matrix > threshold).all(dim=2).float()
    
    # Check if Condorcet winners are in the winning committee
    condorcet_in_winners = torch.sum(W * condorcet_winners, dim=1)
    
    # Penalize if any Condorcet winner is not in the winning committee
    missing_condorcet_winners = torch.sum((condorcet_winners.sum(dim=1) - condorcet_in_winners).clamp(min=0))
    
    # The penalty is scaled up to ensure it has a significant impact
    loss = missing_condorcet_winners
    """
    return loss


def condorcet_loser_loss(winning_committee, possible_committees, n_voters, num_winners, candidate_pairs):
    def mu_P(c, d, candidate_pairs_matrix, n):
        P_c_d = candidate_pairs_matrix[:, c, d]
        return torch.sigmoid(P_c_d - (n // 2 + 1))

    all_committee_condorcet_chances = []

    batch_size = winning_committee.shape[0]
    num_candidates = int(candidate_pairs.shape[1] ** 0.5)
    candidate_pairs = candidate_pairs.view(batch_size, num_candidates, num_candidates)
    candidate_pairs = candidate_pairs.clone().detach().requires_grad_(True)

    _, topk_indices = torch.topk(winning_committee, num_winners, dim=1)
    winners = torch.zeros_like(winning_committee)
    winning_committee = torch.scatter(winners, 1, topk_indices, 1.0)

    # Iterate over all possible committees
    # This value will be the same for all input tensors so should
    for committee in possible_committees:
        committee_tensor = torch.tensor(committee, dtype=torch.float32, requires_grad=True)
        # all_c_comparisons = _is_condorcet_loss(committee_tensor, candidate_pairs, n_voters)
        #
        # # Go through each candidate c in the committee
        # for c in torch.nonzero(committee_tensor).squeeze(1):
        #     all_d_comparisons = []
        #
        #     # Go through each candidate d not in the committee
        #     for d in torch.nonzero(1 - committee_tensor).squeeze(1):
        #         sigmoid_result = mu_P(c.item(), d.item(), candidate_pairs, n_voters)
        #         all_d_comparisons.append(sigmoid_result)
        #
        #     # Find the minimum of all d comparisons
        #     min_d_comparison = torch.min(torch.stack(all_d_comparisons))
        #     all_c_comparisons.append(min_d_comparison)

        # Determine if the committee is a Condorcet committee
        # is_condorcet = torch.min(torch.stack(all_c_comparisons))
        is_condorcet = _is_condorcet_loss(committee_tensor, candidate_pairs, n_voters)
        all_committee_condorcet_chances.append(is_condorcet)

    # Get the maximum Condorcet chance for all possible committees
    max_condorcet_chance = torch.max(torch.stack(all_committee_condorcet_chances))
    non_condorcet_chance = 1 - max_condorcet_chance
    non_condorcet_chance = non_condorcet_chance.repeat(batch_size)

    # Check if the winning committee is a Condorcet committee
    winning_committee_condorcet_chances = []
    for wc in winning_committee:
        winning_committee_tensor = torch.tensor(wc, requires_grad=True)
        # all_c_comparisons = []
        #
        # for c in torch.nonzero(winning_committee_tensor).squeeze(1):
        #     all_d_comparisons = []
        #
        #     for d in torch.nonzero(1 - winning_committee_tensor).squeeze(1):
        #         sigmoid_result = mu_P(c.item(), d.item(), candidate_pairs, n_voters)
        #         all_d_comparisons.append(sigmoid_result)
        #
        #     min_d_comparison = torch.min(torch.stack(all_d_comparisons))
        #     all_c_comparisons.append(min_d_comparison)
        #
        # is_condorcet = torch.min(torch.stack(all_c_comparisons))
        is_condorcet = _is_condorcet_loss(winning_committee_tensor, candidate_pairs, n_voters)
        winning_committee_condorcet_chances.append(is_condorcet)

    # Get the maximum Condorcet chance for the winning committee
    winning_committee_condorcet_chances = torch.stack(winning_committee_condorcet_chances)
    # max_winning_committee_condorcet_chance = torch.max(torch.stack(winning_committee_condorcet_chances))


    # Return the final result
    loss = torch.max(non_condorcet_chance, winning_committee_condorcet_chances)
    loss = 1 - loss
    return loss.sum()


def all_majority_committees(rank_counts, k, batch_size=64):

    m = int(math.sqrt(len(rank_counts)))
    # rank_counts = torch.detach().numpy().array(rank_counts)

    rank_counts = rank_counts.view(m, m)
    half_col_sums = rank_counts.sum(axis=0) / 2

    majority_winner = [i for i in range(m) if rank_counts[i][0] > half_col_sums[i]]

    if len(majority_winner) > 1:
        ValueError("Found more than 1 majority winner")
    if len(majority_winner) == 0:
        return None

    # majority_winner = majority_winner[0]

    possible_other_committee_members = list(set(range(m)) - set(majority_winner))
    num_required = k-1
    majority_winner = majority_winner[0]

    all_valid_committees = []
    for combo in itertools.combinations(possible_other_committee_members, num_required):
        committee = [0] * m
        committee[majority_winner] = 1
        for idx in combo:
            committee[idx] = 1
        all_valid_committees.append(committee)

    return torch.tensor(all_valid_committees, dtype=torch.float32)


def ben_loss_testing(outputs, targets, rank_counts, n_voters, k=2):
    m = len(outputs[0])
    rank_counts = rank_counts.view(m, m)
    half_col_sums = rank_counts.sum(axis=0) / 2

    majority_winner = [i for i in range(m) if rank_counts[i][0] > half_col_sums[i]]

    if len(majority_winner) > 1:
        ValueError("Found more than 1 majority winner")
    if len(majority_winner) == 0:
        return None

    # majority_winner = majority_winner[0]

    possible_other_committee_members = list(set(range(m)) - set(majority_winner))
    num_required = k-1
    majority_winner = majority_winner[0]

    all_valid_committees = []
    for combo in itertools.combinations(possible_other_committee_members, num_required):
        committee = [0] * m
        committee[majority_winner] = 1
        for idx in combo:
            committee[idx] = 1
        all_valid_committees.append(committee)

    return torch.tensor(all_valid_committees, dtype=torch.float32)


def ben_loss_testing(outputs, targets, rank_counts, n_voters, k=2):
    """
    Good function names are hard :/ Largely made by ChatGPT.
    Find the highest k indices in the outputs and the targets.
    Count how many people ranked that candidate first for both the output and the target.
    Return absolute difference of the two scores.
    Should be minimized when the candidates with most first place rankings are elected.

    Just something that gives a non-binary value that might actually teach a network.
    :param outputs:
    :param rank_counts:
    :return:
    """

    distances_across_batch = []
    for idx in range(len(outputs)):
        # generate list of all possible valid (majority-satisfying) committees for current index value of batch
        rc = rank_counts[idx]
        valid_committees = all_majority_committees(rc, k=k)
        if valid_committees is None:
            all_distances = torch.abs(outputs[idx] - outputs[idx])
            min_distance = torch.min(all_distances)
        else:
            all_distances = torch.abs(outputs[idx] - valid_committees)
            all_distances = torch.sum(all_distances, dim=1)
            min_distance = torch.min(all_distances)
    distances_across_batch = []
    for idx in range(len(outputs)):
        # generate list of all possible valid (majority-satisfying) committees for current index value of batch
        rc = rank_counts[idx]
        valid_committees = all_majority_committees(rc, k=k)
        if valid_committees is None:
            all_distances = torch.abs(outputs[idx] - outputs[idx])
            min_distance = torch.min(all_distances)
        else:
            all_distances = torch.abs(outputs[idx] - valid_committees)
            all_distances = torch.sum(all_distances, dim=1)
            min_distance = torch.min(all_distances)

        distances_across_batch.append(min_distance)
        distances_across_batch.append(min_distance)

    distances_across_batch = torch.stack(distances_across_batch)
    loss = torch.mean(distances_across_batch)
    return loss
    distances_across_batch = torch.stack(distances_across_batch)
    loss = torch.mean(distances_across_batch)
    return loss

    # list_tensor = torch.tensor([1, 1, 0, 0, 0], dtype=outputs.dtype)
    #
    #
    #
    # loss = torch.abs(outputs - list_tensor)
    # loss = torch.mean(loss)
    # return loss
    # list_tensor = torch.tensor([1, 1, 0, 0, 0], dtype=outputs.dtype)
    #
    #
    #
    # loss = torch.abs(outputs - list_tensor)
    # loss = torch.mean(loss)
    # return loss

    topk_values, topk_indices = torch.topk(outputs, k=2, dim=1)
    n_candidates = len(outputs[0])
    loser_values, loser_indices = torch.topk(outputs, n_candidates - k, largest=False)
    topk_values, topk_indices = torch.topk(outputs, k=2, dim=1)
    n_candidates = len(outputs[0])
    loser_values, loser_indices = torch.topk(outputs, n_candidates - k, largest=False)

    # # Step 2: Create a mask tensor initialized with True
    # mask = torch.ones_like(outputs, dtype=outputs.dtype, requires_grad=True)

    # Step 3: Binarize winner/loser indices
    # for i in range(outputs.size(0)):
    #     outputs[i, topk_indices[i]] = 1
    # for i in range(outputs.size(0)):
    #     outputs[i, loser_indices[i]] = 0

    loss = torch.abs(outputs - list_tensor)
    loss = torch.mean(loss)
    # # Step 2: Create a mask tensor initialized with True
    # mask = torch.ones_like(outputs, dtype=outputs.dtype, requires_grad=True)

    # Step 3: Binarize winner/loser indices
    # for i in range(outputs.size(0)):
    #     outputs[i, topk_indices[i]] = 1
    # for i in range(outputs.size(0)):
    #     outputs[i, loser_indices[i]] = 0

    loss = torch.abs(outputs - list_tensor)
    loss = torch.mean(loss)
    return loss

    def sigmoid_gt(a, b, alpha=10):
        return torch.sigmoid(alpha * (a - b))

    # Get indices of the winners in the predicted committees
    _, winner_indices = torch.topk(outputs, k)
    n_candidates = len(outputs[0])
    loser_indices = torch.topk(outputs, n_candidates-k, largest=False)

    sigmoid_results = []

    # Loop through the indices
    for i in range(winner_indices.size(0)):
        # Gather the element at the current index
        c = winner_indices[i]
        for d in loser_indices:
            gathered_element = outputs[0, c]

            # Apply the sigmoid function
            sigmoid_value = sigmoid_gt(gathered_element, 0)

            # Append the result to the list
            sigmoid_results.append(sigmoid_value)

    # Convert the list to a tensor
    sigmoid_results = torch.stack(sigmoid_results)

    # Find the minimum of the sigmoid results
    min_value, min_indices = torch.min(sigmoid_results, dim=1)

    # Calculate the mean of the minimum value
    average = torch.mean(min_value)

    # Calculate the absolute value of the mean
    absval = 10 * torch.abs(average)

    return absval
