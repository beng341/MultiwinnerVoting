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
from pref_voting import profiles as pref_voting_profiles
from abcvoting.preferences import Profile
from abcvoting import abcrules


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
        violations = du.eval_all_axioms(len(test_df["Profile"].iloc[0]), test_df["rank_matrix"], test_df["candidate_pairs"], y_pred_committees, num_winners)
        model_viols[model_path] = violations

        model_accs[model_path] = acc

        voting_rules = du.load_mw_voting_rules()

        model_rule_viols[model_path] = dict()

        for index, row in test_df.iterrows():
            n_voters = len(row["Profile"])
            rank_choice = row["rank_matrix"]
            cand_pairs = row["candidate_pairs"]

            print(voting_rules)

            for rule in voting_rules:
                if rule not in model_rule_viols[model_path]:
                    model_rule_viols[model_path][rule] = []
                    committees = voting_rules[rule](rank_choice, cand_pairs)  # Assuming function returns committees
                    num_winners = len(committees)
                    violations = du.eval_all_axioms(n_voters, rank_choice, cand_pairs, committees, num_winners)
                    model_rule_viols[model_path][rule].append(violations)




            
    
        
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

    return model_accs, model_viols

def save_accuracies_of_all_network_types():
    """
    Loop over all parameter combinations and save the accuracy of each group of saved networks at predicting elections
    from the specified distribution.
    :return:
    """

    test_size_all = [2000]
    m_all, n_all, num_winners, pref_dist_all, features_all, losses_all, num_trained_models_per_param_set = ml_utils.get_default_parameter_value_sets(
        m=True, n=True, train_size=False, num_winners=True, pref_dists=True, features=True, losses=True,
        networks_per_param=True)

    base_cols = ["m", "n", "num_winners", "test_size", "dist", "features", "loss", "num_models"]

    violation_counts = dict()
    all_axioms = []

    all_model_accs = dict()
    all_model_viols = dict()

    for m, n, test_size, pref_dist, features, winners_size, loss in product(m_all, n_all, test_size_all,
                                                                             pref_dist_all,
                                                                             features_all, num_winners, losses_all):

        df = du.load_data(size=test_size,
                          n=n,
                          m=m,
                          num_winners=winners_size,
                          pref_dist=pref_dist,
                          train=False,
                          condorcet_only=False,
                          make_data_if_needed=True)
        test_df = df.sample(n=test_size)
        feature_values = ml_utils.features_from_column_abbreviations(test_df, features)

        #v_df = ml_utils.generate_viol_df(test_df["Profile"].tolist())

        # Generate paths to all models
        model_paths = ml_utils.saved_model_paths(n, m,
                                                 pref_dist,
                                                 features,
                                                 num_trained_models_per_param_set,
                                                 loss)
        # Compute accuracy of each model
        model_accs, model_viols = model_accuracies(test_df,
                                      features=feature_values,
                                      model_paths=model_paths,
                                      num_winners=winners_size)
        #model_viols = violations_count(v_df, model_paths=model_paths)
        #pprint.pprint(model_viols)

        # Update all accuracies with newly calculated ones
        # (make sure all rules have unique names or else they will override old results)
        all_model_accs = all_model_accs | model_accs
        all_model_viols = all_model_viols | model_viols

        # Get average number of violations for each axiom by this set of parameters
        vals = (m, n, winners_size, test_size, pref_dist, features, str(loss), num_trained_models_per_param_set)  # for readability
        for model, violations_dict in model_viols.items():
            violation_counts[vals] = []
            for ax, count in violations_dict.items():
                if ax not in all_axioms:
                    all_axioms.append(ax)

        # make list of axiom count_violations
        for ax in all_axioms:   # so we have a consistent ordering of axioms
            ax_violation_counts = []
            for model, violations_dict in model_viols.items():
                ax_violation_counts.append(violations_dict[ax])  # how many times each model violated axiom ax
            violation_counts[vals].append((round(np.mean(ax_violation_counts), 4), round(np.std(ax_violation_counts), 4)))

        du.save_evaluation_results(base_cols, all_axioms, violation_counts, filename="results.csv")

    pprint.pprint(all_model_accs)
    
    for network, violations in all_model_viols.items():
        if violations['condorcet_winner'] > 0:
            print(network, violations['condorcet_winner'])


    totals = {'condorcet_loser': 0, 'condorcet_winner': 0, 'majority': 0, 'majority_loser': 0, 'count_viols': 0}

    # Calculate totals
    for model in all_model_viols.values():
        for key in totals:
            totals[key] += model.get(key, 0)

    # Print the totals
    pprint.pprint(totals)

    # header = base_cols + all_axioms + ["total_violation"]
    # rows = []
    # for base_vals, counts in violation_counts.items():
    #     mean_count = sum([pair[0] for pair in counts])
    #     row = list(base_vals) + counts + [mean_count]
    #     rows.append(row)
    #
    # df = pd.DataFrame(data=rows, columns=header, index=None)
    # df.to_csv("results.csv", index=False)




if __name__ == "__main__":
    save_accuracies_of_all_network_types()
