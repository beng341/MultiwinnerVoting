import os
import random
from itertools import product
import numpy as np
import pandas as pd
import torch
from abcvoting import abcrules
from sklearn.metrics import accuracy_score
from utils import data_utils as du
from utils import ml_utils


def model_accuracies(test_df, features, model_paths, num_winners):
    """

    :param num_winners:
    :param features:
    :param test_df:
    :param model_paths:
    :return:
    """


    viols = dict()

    viols["Neural Network"] = dict()

    # calculate violations for each individual network
    for model_path in model_paths:
        model = ml_utils.load_model(model_path)
        model.eval()

        x = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            y = model(x)

        committee_size = len(y[0])
        y_pred = [torch.argmax(y_i).item() for y_i in y]
        y_pred_committee = [np.argpartition(y_i, -num_winners)[-num_winners:].tolist() for y_i in y]
        y_pred_committees = [[0 if idx not in yc else 1 for idx in range(committee_size)] for yc in y_pred_committee]
        y_true = [eval(yt) for yt in test_df[f"Winner"].tolist()]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred_committees)
        violations = du.eval_all_axioms(n_voters=len(eval(test_df["Profile"].iloc[0])),
                                        rank_choice=test_df["rank_matrix"],
                                        cand_pairs=test_df["candidate_pairs"],
                                        committees=y_pred_committees,
                                        n_winners=num_winners,
                                        profiles=test_df["Profile"])
        print(f"Finished calculating axiom violations for one model: {model_path}")

        for key, value in violations.items():
            if key in viols:
                viols["Neural Network"][key] += value
            else:
                viols["Neural Network"][key] = value

    viols["Neural Network"] = {k: v // len(model_paths) for k, v in viols["Neural Network"].items()}

    num_candidates = len(y_pred_committees[0])
    num_committees = len(y_pred_committees)

    # Calculate axiom violations for random committees
    y_random_committees = []

    for _ in range(num_committees):
        committee = [1] * num_winners + [0] * (num_candidates - num_winners)
        random.shuffle(committee)
        y_random_committees.append(committee)

    rand_viols = du.eval_all_axioms(len(eval(test_df["Profile"].iloc[0])), test_df["rank_matrix"],
                                    test_df["candidate_pairs"], y_random_committees, num_winners,
                                    test_df["Profile"])

    viols["Random Choice"] = rand_viols

    # Calculate axiom violations for each existing rule
    print("Counting violations for existing rules")
    voting_rules = du.load_mw_voting_rules()

    for rule in voting_rules:
        try:
            s = abcrules.get_rule(rule).longname
        except AttributeError:
            try:
                s = rule.name
            except AttributeError:
                print("Unknown rule")
                return

        if s not in viols:
            viols[s] = {}

        y_true_rule = [eval(yt) for yt in test_df[f"{s} Winner"]]
        violations_rule = du.eval_all_axioms(len(eval(test_df["Profile"].iloc[0])), test_df["rank_matrix"],
                                             test_df["candidate_pairs"], y_true_rule, num_winners,
                                             test_df["Profile"])

        viols[s] = violations_rule

    for model, sub_dicts in viols.items():
        for key, value in sub_dicts.items():
            if key == "total_violations":
                # Scale count of total violations by the higher bound that it has over individual axioms
                # assumes sub_dicts contains total_violations and count_violations
                value /= (len(sub_dicts.items()) - 2)
            if isinstance(value, (int, float)):
                viols[model][key] = value / (num_committees * len(model_paths))

    return viols


def save_accuracies_of_all_network_types(test_size, n, m, num_winners, pref_dist, axioms, base_data_folder="data", out_folder="results"):
    """
    Loop over all parameter combinations and save the accuracy of each group of saved networks at predicting elections
    from the specified distribution.
    :return:
    """

    _, _, _, _, features_all, losses_all, num_trained_models_per_param_set = ml_utils.get_default_parameter_value_sets(
        m=True, n=True, train_size=False, num_winners=True, pref_dists=True, features=True, losses=True,
        networks_per_param=True)

    all_viols = dict()

    for features, loss in product(features_all, losses_all):

        test_df = du.load_data(size=test_size,
                               n=n,
                               m=m,
                               num_winners=num_winners,
                               pref_dist=pref_dist,
                               axioms=axioms,
                               train=False,
                               base_data_folder=base_data_folder,
                               make_data_if_needed=False)
        if test_df is None:
            print("Could not find test file with the given parameters. Stopping testing.")
            break
        feature_values = ml_utils.features_from_column_abbreviations(test_df, features)

        # Generate paths to all models
        model_paths = ml_utils.saved_model_paths(n=n,
                                                 m=m,
                                                 num_winners=num_winners,
                                                 pref_dist=pref_dist,
                                                 axioms=axioms,
                                                 features=features,
                                                 num_models=num_trained_models_per_param_set,
                                                 loss=loss)

        # Compute accuracy and axiom violations of each model
        model_viols = model_accuracies(test_df,
                                       features=feature_values,
                                       model_paths=model_paths,
                                       num_winners=num_winners)

        # Update all accuracies with newly calculated ones
        # (make sure all rules have unique names or else they will override old results)

        all_viols = all_viols | model_viols

    df = pd.DataFrame.from_dict(all_viols, orient='index')
    if not os.path.exists(out_folder):
        print(f"{out_folder} does not exist; making it now")
        os.makedirs(out_folder)
    base_name = f"axiom_violation_results-n_profiles={test_size}-num_voters={n}-m={m}-k={num_winners}-pref_dist={pref_dist}-axioms={axioms}.csv"
    filename = os.path.join(out_folder, base_name)
    df.to_csv(filename)
    print(f"Saving results to: {filename}")



if __name__ == "__main__":
    pref_models = [
        # "URN-R",
        "IC",
        # "identity",
        # "MALLOWS-RELPHI-R",
        # "mixed"
    ]

    size = 1000
    num_voters = 50
    num_candidates = 5
    winners = 1
    axioms = "all"
    out_folder = "results/normalized_cp"
    for dist in pref_models:
        save_accuracies_of_all_network_types(test_size=size,
                                             n=num_voters,
                                             m=num_candidates,
                                             num_winners=winners,
                                             pref_dist=dist,
                                             axioms=axioms,
                                             out_folder=out_folder)
