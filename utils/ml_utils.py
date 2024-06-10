from sklearn.metrics import accuracy_score
from os.path import isfile, join
from MultiWinnerVotingRule import SingleWinnerVotingRule
import torch


def load_model(model_path):
    """

    :param model_path:
    :return:
    """
    checkpoints = torch.load(model_path)

    adjusted_state_dict = {f"model.{k}": v for k, v in checkpoints['model_state_dict'].items()} # feels messy but seems to work

    num_candidates = checkpoints['num_candidates']
    config = checkpoints['config']
    kwargs = checkpoints['kwargs']

    model = SingleWinnerVotingRule(num_candidates, config, **kwargs)
    model.load_state_dict(adjusted_state_dict)
    model.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    model.eval()

    return model


def saved_model_paths(n, m, pref_dist, features, rule, num_models):
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
            f"{base_model_path}/NN-num_voters={n}-m={m}-pref_dist={pref_dist}-features={features}-rule={rule}-idx={idx}-.pt"
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


