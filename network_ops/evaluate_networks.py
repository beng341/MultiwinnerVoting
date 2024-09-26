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

    # calculate the model violations

    viols = dict()

    viols["Neural Network"] = dict()

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

    

    # then calculate random chance violations
    y_random_committees = []

    for _ in range(num_committees):
        committee = [1] * num_winners + [0] * (num_candidates - num_winners)
        random.shuffle(committee)
        y_random_committees.append(committee)

    rand_viols = du.eval_all_axioms(len(eval(test_df["Profile"].iloc[0])), test_df["rank_matrix"],
                                    test_df["candidate_pairs"], y_random_committees, num_winners,
                                    test_df["Profile"])

    viols["Random Choice"] = rand_viols

    # print(viols)

    # then calculate rule violations
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
            if isinstance(value, (int, float)):
                viols[model][key] = value / (num_committees * len(model_paths))

    return viols

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

            if s == "STV":
                pass

            y_true_rule = [eval(yt) for yt in test_df[f"{s} Winner"]]
            violations_rule = du.eval_all_axioms(len(eval(test_df["Profile"].iloc[0])), test_df["rank_matrix"],
                                                 test_df["candidate_pairs"], y_true_rule, num_winners,
                                                 test_df["Profile"])
            model_rule_viols[model_path][s] = violations_rule

        model_rule_viols[model_path]["Neural Network"] = violations

    return model_accs, model_viols, model_rule_viols
    """


def save_accuracies_of_all_network_types(test_size, n, m, num_winners, pref_dist, axioms, folder="results"):
    """
    Loop over all parameter combinations and save the accuracy of each group of saved networks at predicting elections
    from the specified distribution.
    :return:
    """

    _, _, _, _, features_all, losses_all, num_trained_models_per_param_set = ml_utils.get_default_parameter_value_sets(
        m=True, n=True, train_size=False, num_winners=True, pref_dists=True, features=True, losses=True,
        networks_per_param=True)

    base_cols = ["m", "n", "n_winners", "test_size", "dist", "features", "loss", "num_models"]

    all_viols = dict()

    for features, loss in product(features_all, losses_all):

        test_df = du.load_data(size=test_size,
                               n=n,
                               m=m,
                               num_winners=num_winners,
                               pref_dist=pref_dist,
                               axioms=axioms,
                               train=False,
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

        # all_model_accs = all_model_accs | model_accs
        all_viols = all_viols | model_viols
        # all_model_rule_viols = all_model_rule_viols | model_rule_viols

        ################################################################################
        ################################################################################
        #
        # TODO: Josh, each network should be saved after it is evaluated instead of: Evaluate all networks then save.
        # This is so we still get partial results in case the compute canada job doesn't finish in time.
        # See below for code that works outside of this context. I think you'll be faster than me at making it work here
        #
        ################################################################################
        ################################################################################

        # # save this model's results
        # df = pd.DataFrame.from_dict(rule_viols, orient='index')
        # base_name = f"single_network_axiom_violation_results-n_profiles={test_size}-num_voters={n}-m={m}-k={num_winners}-pref_dist={pref_dist}-network_idx={model_idx}.csv"
        # testname = os.path.join(job_file_folder, base_name)
        # df.to_csv(testname)
        # model_idx += 1

        # Get average number of violations for each axiom by this set of parameters
        """
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
        """

        # du.save_evaluation_results(base_cols, all_axioms, violation_counts, testname="results.csv")

    # pprint.pprint(all_model_accs)

    # for network, violations in all_model_viols.items():
    #     if violations['condorcet_winner'] > 0:
    #         print(network, violations['condorcet_winner'])

    # totals = {'condorcet_loser': 0, 'condorcet_winner': 0, 'majority': 0, 'majority_loser': 0, 'count_viols': 0}

    # # Calculate totals
    # for model in all_model_viols.values():
    #     for key in totals:
    #         totals[key] += model.get(key, 0)

    # header = base_cols + all_axioms + ["total_violation"]
    # rows = []
    # for base_vals, counts in violation_counts.items():
    #     mean_count = sum([pair[0] for pair in counts])
    #     row = list(base_vals) + counts + [mean_count]
    #     rows.append(row)
    #
    # df = pd.DataFrame(data=rows, columns=header, index=None)
    # df.to_csv("results.csv", index=False)

    df = pd.DataFrame.from_dict(all_viols, orient='index')
    if not os.path.exists(folder):
        print(f"{folder} does not exist; making it now")
        os.makedirs(folder)
    base_name = f"axiom_violation_results-n_profiles={test_size}-num_voters={n}-m={m}-k={num_winners}-pref_dist={pref_dist}-axioms={axioms}.csv"
    filename = os.path.join(folder, base_name)
    df.to_csv(filename)
    print(f"Saving results to: {filename}")

    # model_idx = 0
    # for model, rule_viols in all_model_rule_viols.items():
    #     df = pd.DataFrame.from_dict(rule_viols, orient='index')
    #     # testname = './results/' + model.replace('/', '_').replace('<', '').replace('>', '').replace(':', '').replace(
    #     #     '|', '').replace('?', '').replace('*', '').replace('"', '') + '.csv'
    #     if not os.path.exists(folder):
    #         print(f"{folder} does not exist; making it now")
    #         os.makedirs(folder)
    #     base_name = f"axiom_violation_results-n_profiles={test_size}-num_voters={n}-m={m}-k={num_winners}-pref_dist={pref_dist}-network_idx={model_idx}.csv"
    #     testname = os.path.join(folder, base_name)
    #     print(f"Saving results to: {testname}")
    #     df.to_csv(testname)
    #     model_idx += 1


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
                                             folder=out_folder)
