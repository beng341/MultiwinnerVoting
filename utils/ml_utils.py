from sklearn.metrics import accuracy_score
from os.path import isfile, join
from MultiWinnerVotingRule import MultiWinnerVotingRule
import torch
import torch.nn as nn
from pref_voting.generate_profiles import generate_profile as gen_prof
from . import data_utils as du
import pandas as pd


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
        train_size_all = [2000]  # training size
        results.append(train_size_all)
    if num_winners:
        num_winners = [3]
        results.append(num_winners)
    if pref_dists:
        pref_dist_all = [
            # "stratification__args__weight=0.5",
            "URN-R",
            # "IC",
            # "IAC",
            "MALLOWS-RELPHI-R",
            # "single_peaked_conitzer",
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
            nn.L1Loss,
            nn.MSELoss,
            nn.CrossEntropyLoss,
            # nn.CTCLoss,
            # nn.NLLLoss,
            # nn.PoissonNLLLoss,
            # nn.GaussianNLLLoss,
            # nn.KLDivLoss,
            # nn.BCELoss,
            # nn.BCEWithLogitsLoss,
            # nn.MarginRankingLoss,
            # nn.HingeEmbeddingLoss,
            # nn.MultiLabelMarginLoss,
            nn.HuberLoss,
            # nn.SmoothL1Loss,
            # nn.SoftMarginLoss,
            # nn.MultiLabelSoftMarginLoss,
            # nn.CosineEmbeddingLoss,
            # nn.MultiMarginLoss,
            # nn.TripletMarginLoss,
            # nn.TripletMarginWithDistanceLoss
        ]
        results.append(losses_all)

    if networks_per_param:
        networks_per_param_set = 3  # How many networks to learn for each combination of parameters
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
    config = checkpoints['config']
    kwargs = checkpoints['kwargs']

    model = MultiWinnerVotingRule(num_candidates, config, **kwargs)
    model.load_state_dict(adjusted_state_dict)
    model.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    model.eval()

    return model


def saved_model_paths(n, m, pref_dist, features, num_models, loss):
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
            f"{base_model_path}/NN-num_voters={n}-m={m}-pref_dist={pref_dist}-features={features}-loss={str(loss)}-idx={idx}-.pt"
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
        "c": "candidate_pairs-normalized-no_diagonal",
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
        "c": "candidate_pairs-normalized-no_diagonal",
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
        'candidate_pairs-normalized-no_diagonal',
        'binary_pairs-no_diagonal',
        'rank_matrix-normalized'
    ])

    return df
