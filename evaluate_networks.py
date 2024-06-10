import os
import pprint

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from itertools import product
from utils import ml_utils


def model_accuracies(test_df, rule, features, model_paths):
    """

    :param test_df:
    :param rule:
    :param feature_names:
    :param feature_values:
    :param model_paths:
    :return:
    """

    model_accs = dict()

    for model_path in model_paths:
        model = ml_utils.load_model(model_path)

        model.eval()

        x = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            y = model(x)

        committee_size = len(y[0])
        y_pred = [torch.argmax(y_i).item() for y_i in y]
        y_pred_committee = [np.argpartition(y_i, -3)[-3:].tolist() for y_i in y]
        y_pred_committees = [[0 if idx not in yc else 1 for idx in range(committee_size)] for yc in y_pred_committee]

        y_true = [eval(yt) for yt in test_df[f"{rule}-single_winner"].tolist()]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred_committees)

        model_accs[model_path] = acc

    return model_accs


def save_accuracies_of_all_network_types():
    """
    Loop over all parameter combinations and save the accuracy of each group of saved networks at predicting elections
    from the specified distribution.
    :return:
    """
    test_size = 2000
    m_all = [5]
    n_all = [20]
    # rules_all = ["Instant Runoff", "Plurality", "Borda", "Anti-Plurality", "Benham", "Coombs",
    #              "Baldwin", "Strict Nanson", "Weak Nanson", "Raynaud", "Tideman Alternative Top Cycle",
    #              "Tideman Alternative GOCHA", "Knockout Voting", "Banks", "Condorcet", "Copeland", "Llull",
    #              "Uncovered Set", "Slater", "Top Cycle", "GOCHA", "Bipartisan Set", "Minimax", "Split Cycle",
    #              "Beat Path", "Simple Stable Voting", "Stable Voting", "Loss-Trimmer Voting", "Daunou", "Blacks",
    #              "Condorcet Plurality", "Copeland-Local-Borda", "Copeland-Global-Borda", "Borda-Minimax Faceoff",
    #              "Bucklin", "Simplified Bucklin", "Weighted Bucklin", "Bracket Voting", "Superior Voting"]
    rules_all = ["Approval Voting (AV)", "Lexicographic Chamberlin-Courant (lex-CC)"]

    pref_dist_all = [
        "stratification__args__weight=0.5",
        "URN-R",
        "IC",
        "IAC",
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
    # features_all = ["b", "c", "r", "bc", "br", "cr", "bcr"]
    features_all = ["bcr"]
    num_winners = [3]

    base_data_folder = "data"
    num_trained_models_per_param_set = 2

    all_model_accs = dict()

    for m, n, train_size, pref_dist, features, winners_size, rule in product(m_all, n_all, [test_size],
                                                                             pref_dist_all,
                                                                             features_all, num_winners,
                                                                             rules_all):

        filename = f"n_profiles={test_size}-num_voters={n}-m={m}-committee_size={winners_size}-pref_dist={pref_dist}.csv"
        if not os.path.exists(f"{base_data_folder}/{filename}"):
            print(f"Tried loading path but it does not exist: {base_data_folder}/{filename}")
            continue

        # Load test data
        df = pd.read_csv(f"{base_data_folder}/{filename}")
        test_df = df.sample(n=test_size)
        feature_values = ml_utils.features_from_column_abbreviations(test_df, features)

        # Generate paths to all models
        model_paths = ml_utils.saved_model_paths(n, m, pref_dist, features, rule,
                                                 num_trained_models_per_param_set)

        # Compute accuracy of each model
        model_accs = model_accuracies(test_df,
                                      rule=rule,
                                      features=feature_values,
                                      model_paths=model_paths)

        # Update all accuracies with newly calculated ones
        # (make sure all rules have unique names or else they will override old results)
        all_model_accs = all_model_accs | model_accs

    pprint.pprint(all_model_accs)


if __name__ == "__main__":
    save_accuracies_of_all_network_types()
