import os
import pprint
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from itertools import product
from utils import ml_utils
from utils import data_utils as du
from utils import axiom_eval as ae
from pref_voting import profiles as pref_voting_profiles
from abcvoting.preferences import Profile
from abcvoting import abcrules
import random


def model_accuracies(test_df, features, model_paths, num_winners):
    """

    :param test_df:
    :param rule:
    :param feature_names:
    :param feature_values:
    :param model_paths:
    :return:
    """

    model_accs = dict()
    model_viols = dict()
    model_rule_viols = dict()

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
        model_viols[model_path] = violations

        model_accs[model_path] = acc

        voting_rules = du.load_mw_voting_rules()

        model_rule_viols[model_path] = dict()

        model_rule_viols[model_path]["Neural Network"] = violations

        num_candidates = len(y_pred_committees[0])
        num_committees = len(y_pred_committees)

        # Generate random committees for use by Random rule
        y_random_committees = []

        for _ in range(num_committees):
            committee = [1] * num_winners + [0] * (num_candidates - num_winners)
            random.shuffle(committee)
            y_random_committees.append(committee)

        # count Random rule axiom violations
        rand_viols = du.eval_all_axioms(len(eval(test_df["Profile"].iloc[0])), test_df["rank_matrix"],
                                        test_df["candidate_pairs"], y_random_committees, num_winners,
                                        test_df["Profile"])

        model_rule_viols[model_path]["Random Choice"] = rand_viols

        # count axiom violations for other pre-existing rules
        for rule in voting_rules:
            try:
                s = abcrules.get_rule(rule).longname
            except AttributeError:
                try:
                    s = rule.name
                except AttributeError:
                    print("Unknown rule")
                    return

            if s not in model_rule_viols[model_path]:
                model_rule_viols[model_path][s] = []

            if s == "Approval Voting (AV)":
                pass

            y_true_rule = [eval(yt) for yt in test_df[f"{s} Winner"]]
            violations_rule = du.eval_all_axioms(len(eval(test_df["Profile"].iloc[0])), test_df["rank_matrix"],
                                                 test_df["candidate_pairs"], y_true_rule, num_winners,
                                                 test_df["Profile"])
            model_rule_viols[model_path][s] = violations_rule

        model_rule_viols[model_path]["Neural Network"] = violations

        """
        for rule in voting_rules:
                try:
                    s = abcrules.get_rule(rule).longname
                    profiles = abc_profiles
                except AttributeError:
                    try:
                        s = rule.name
                        profiles = pref_voting_profiles
                    except AttributeError:
                        print("Unknown rule")
                        return
    
                print(f"Beginning to calculate winners & violations for {s} using {pref_model_shortname} preferences")
    
                try:
                    singlecomittee, tiedcomittees = du.generate_winners(rule, profiles, winners_size, m)
                    df[f"{s}-single_winner"] = singlecomittee
                    df[f"{s}-tied_winners"] = tiedcomittees
                    df = df.copy()
                except Exception as ex:
                    print(f"{s} broke everything")
                    print(f"{ex}")
                    return
        """

    return model_accs, model_viols, model_rule_viols


def save_accuracies_of_all_network_types(test_size, n, m, num_winners, pref_dist):
    """
    Loop over all parameter combinations and save the accuracy of each group of saved networks at predicting elections
    from the specified distribution.
    :return:
    """

    # test_size_all = [10000]
    _, _, _, _, features_all, losses_all, num_trained_models_per_param_set = ml_utils.get_default_parameter_value_sets(
        m=True, n=True, train_size=False, num_winners=True, pref_dists=True, features=True, losses=True,
        networks_per_param=True)

    base_cols = ["m", "n", "n_winners", "test_size", "dist", "features", "loss", "num_models"]

    violation_counts = dict()
    all_axioms = []

    all_model_accs = dict()
    all_model_viols = dict()
    all_model_rule_viols = dict()

    # dimensions = [(5957, 55, 6, 3), (8846, 37, 7, 4), (3792, 24, 9, 2), (3686, 39, 10, 3)]
    # ptr = 0

    for features, loss in product(features_all, losses_all):

        # test_size, n, m, winners_size = 5000, 100, 6, 3
        # ptr += 1

        test_df = du.load_data(size=test_size,
                          n=n,
                          m=m,
                          num_winners=num_winners,
                          pref_dist=pref_dist,
                          train=False,
                          make_data_if_needed=True)
        # test_df = df.sample(n=test_size)
        feature_values = ml_utils.features_from_column_abbreviations(test_df, features)

        # v_df = ml_utils.generate_viol_df(test_df["Profile"].tolist())

        # Generate paths to all models
        model_paths = ml_utils.saved_model_paths(n, m,
                                                 pref_dist,
                                                 features,
                                                 num_trained_models_per_param_set,
                                                 loss)

        # Compute accuracy and axiom violations of each model
        model_accs, model_viols, model_rule_viols = model_accuracies(test_df,
                                                                     features=feature_values,
                                                                     model_paths=model_paths,
                                                                     num_winners=num_winners)

        # Update all accuracies with newly calculated ones
        # (make sure all rules have unique names or else they will override old results)
        all_model_accs = all_model_accs | model_accs
        all_model_viols = all_model_viols | model_viols
        all_model_rule_viols = all_model_rule_viols | model_rule_viols

        # Get average number of violations for each axiom by this set of parameters
        vals = (m, n, num_winners, test_size, pref_dist, features, str(loss),
                num_trained_models_per_param_set)  # for readability
        for model, violations_dict in model_viols.items():
            violation_counts[vals] = []
            for ax, count in violations_dict.items():
                if ax not in all_axioms:
                    all_axioms.append(ax)

        # make list of axiom count_violations
        for ax in all_axioms:  # so we have a consistent ordering of axioms
            ax_violation_counts = []
            for model, violations_dict in model_viols.items():
                ax_violation_counts.append(violations_dict[ax])  # how many times each model violated axiom ax
            violation_counts[vals].append(
                (round(np.mean(ax_violation_counts), 4), round(np.std(ax_violation_counts), 4)))

        du.save_evaluation_results(base_cols, all_axioms, violation_counts, filename="results.csv")

    # pprint.pprint(all_model_accs)

    for network, violations in all_model_viols.items():
        if violations['condorcet_winner'] > 0:
            print(network, violations['condorcet_winner'])

    totals = {'condorcet_loser': 0, 'condorcet_winner': 0, 'majority': 0, 'majority_loser': 0, 'count_viols': 0}

    # Calculate totals
    for model in all_model_viols.values():
        for key in totals:
            totals[key] += model.get(key, 0)

    # header = base_cols + all_axioms + ["total_violation"]
    # rows = []
    # for base_vals, counts in violation_counts.items():
    #     mean_count = sum([pair[0] for pair in counts])
    #     row = list(base_vals) + counts + [mean_count]
    #     rows.append(row)
    #
    # df = pd.DataFrame(data=rows, columns=header, index=None)
    # df.to_csv("results.csv", index=False)

    for model, rule_viols in all_model_rule_viols.items():
        df = pd.DataFrame.from_dict(rule_viols, orient='index')
        filename = './results/' + model.replace('/', '_').replace('<', '').replace('>', '').replace(':', '').replace(
            '|', '').replace('?', '').replace('*', '').replace('"', '') + f'k={num_winners}' + '.csv'
        df.to_csv(filename)


if __name__ == "__main__":
    pref_models = [
        # "identity",
        "MALLOWS-RELPHI-R",
        "single_peaked_conitzer",
    ]

    size = 100
    num_voters = 10
    num_candidates = 5
    winners = 3
    for dist in pref_models:
        save_accuracies_of_all_network_types(test_size=size, n=num_voters, m=num_candidates, num_winners=winners, pref_dist=dist)
