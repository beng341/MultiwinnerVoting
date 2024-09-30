import itertools
import os.path
import pprint

import pandas as pd

import os
import sys

# Add the parent directory (Multiwinnervoting) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import ml_utils
import matplotlib.pyplot as plt
from utils import data_utils as du

all_pref_models = [
    "stratification__args__weight=0.5",
    "URN-R",
    "IC",
    "IAC",
    "identity",
    "MALLOWS-RELPHI-R",
    "single_peaked_conitzer",
    "single_peaked_walsh",
    "euclidean__args__dimensions=3_-_space=gaussian_ball",
    "euclidean__args__dimensions=10_-_space=gaussian_ball",
    "euclidean__args__dimensions=3_-_space=uniform_ball",
    "euclidean__args__dimensions=10_-_space=uniform_ball",
    "euclidean__args__dimensions=3_-_space=gaussian_cube",
    "euclidean__args__dimensions=10_-_space=gaussian_cube",
    "euclidean__args__dimensions=3_-_space=uniform_cube",
    "euclidean__args__dimensions=10_-_space=uniform_cube",
    "mixed"
]

all_axioms = [
        "dummett",
        "consensus",
        "fixed_majority",
        "majority_winner",
        "majority_loser",
        "condorcet_winner",
        "condorcet_loser",
        "solid_coalition",
        "strong_unanimity",
        "local_stability",
        "strong_pareto"
    ]


def generate_data():
    """
    Untested and for posterity. In principle, this should recreate the data used in this experiment.
    Safe to use only as a reference for writing/understanding what data exists.
    Better to create jobs to train individually on a cluster; see cc_jobs directory and cc_job_generator.py
    If those exist, this should correspond with cc_job_generator.py:make_data_generation_jobs() on 24/09/2024.
    :return:
    """

    if True:
        print("You didn't mean to run this method. You don't know what it does and didn't proofread before running it.")
        exit()

    n_profiles = 25000
    n_all = [50]
    m_all = [5, 6, 7]
    k_all = [1, 2, 3, 4, 5, 6]
    pref_dist_all = all_pref_models
    axiom = "all"

    for n, m, k, pref_dist in itertools.product(n_all, m_all, k_all, pref_dist_all):

        if k >= m:
            continue

        args = {
            "n_profiles": n_profiles,
            "prefs_per_profile": n,
            "m": m,
            "num_winners": k,
            "pref_model": pref_dist,
            "axioms": axiom,
            "output_folder": "results"
        }
        print("Making dataset with the following args:")
        pprint.pprint(args)

        output_frequency = 1000

        from network_ops.generate_data import make_one_multi_winner_dataset
        make_one_multi_winner_dataset(args=args,
                                      output_frequency=output_frequency
                                      )


def train_networks():
    """
    Train networks according to the below parameters.
    Unfortunately, things were built in a way that don't allow controlling all parameters during runtime.
    To accurately/completely run this experiment, you must:
        1.  Ensure that train_networks.py:train_networks() calls du.load_data with make_data_if_needed set to False.
            This is more of an efficiency thing than anything else. If the data isn't present, something else is wrong.
        2.  Ensure ml_utils.get_default_parameter_value_sets() uses a value of 20 networks_per_param_set.
            TODO: Based on external factors, this could be adjusted to allow faster results. But the value is likely
            set to something like 2 which is suitable for quick tests only, not a full experiment.
    :return:
    """
    n_profiles = 25000
    n_all = [50]
    m_all = [6, 7]
    k_all = [1, 2, 3, 4, 5, 6]
    pref_dist_all = all_pref_models
    axiom = "all"

    from network_ops.train_networks import train_networks
    for n, m, k, pref_dist in itertools.product(n_all, m_all, k_all, pref_dist_all[15:]):

        if k >= m:
            continue

        _, _, _, _, feature_set_all, losses_all, networks_per_param_set = ml_utils.get_default_parameter_value_sets(
            m=True, n=True, train_size=False, num_winners=True, pref_dists=True, features=True, losses=True,
            networks_per_param=True)
        if networks_per_param_set != 20:
            print("Make sure to read documentation on this method! Check BOTH make_data_if_needed and networks_per_param_set.")
            exit()
        
        if pref_dist == "mixed":
            du.generate_mixed_distribution(distributions=all_pref_models[:-1],
                                                       total_size=n_profiles,
                                                       n=n,
                                                       m=m,
                                                       num_winners=k,
                                                       axioms=axiom)

        train_networks(train_size=n_profiles,
                       n=n,
                       m=m,
                       num_winners=k,
                       pref_dist=pref_dist,
                       axioms=axiom,
                       base_data_folder="data"
                       )


def evaluate_networks():
    """
    Test networks according to the below parameters.
    Unfortunately, things were built in a way that don't allow controlling all parameters during runtime.
    To accurately/completely run this experiment, you must:
        1.  Ensure that evaluate_networks.py:save_accuracies_of_all_network_types() calls du.load_data with
            make_data_if_needed set to False. This is more of an efficiency thing than anything else.
            If the data isn't present, something else is wrong.
        2.  Ensure ml_utils.get_default_parameter_value_sets() uses a value of 20 networks_per_param_set.
            TODO: Based on external factors, this could be adjusted to allow faster results. But the value is likely
            set to something like 2 which is suitable for quick tests only, not a full experiment.
    :return:
    """
    n_profiles = 25000
    n_all = [50]
    m_all = [6]
    k_all = [1, 2, 3, 4, 5, 6]
    pref_dist_all = all_pref_models
    axiom = "all"

    from network_ops.evaluate_networks import save_accuracies_of_all_network_types
    for n, m, k, pref_dist in itertools.product(n_all, m_all, k_all, pref_dist_all):

        if k >= m:
            continue

        _, _, _, _, feature_set_all, losses_all, networks_per_param_set = ml_utils.get_default_parameter_value_sets(
            m=True, n=True, train_size=False, num_winners=True, pref_dists=True, features=True, losses=True,
            networks_per_param=True)
        if networks_per_param_set != 20:
            print(
                "Make sure to read documentation on this method! Check BOTH make_data_if_needed and networks_per_param_set.")
            exit()

        save_accuracies_of_all_network_types(test_size=n_profiles,
                                             n=n,
                                             m=m,
                                             num_winners=k,
                                             pref_dist=pref_dist,
                                             axioms=axiom,
                                             base_data_folder="data",
                                             out_folder="experiment_all_axioms"
                                             )


def plot_axioms_all_distributions_each_rule(m=5, rules="all", dist="mixed"):
    """
    Make a basic plot of results for a given number of voters.
    x-axis shows number of winners
    y-axis show the axiom violation rate; maximum value of 1 is achieved when every example violates every axiom.
    Each series corresponds to a single voting rule.
    :param m: Number of alternatives in the data on the plot. x-axis should have integer values from 1 to m-1
    :param rules: "all" or list of rules to include on the plot.
    TODO: Ben, get the rules parameter doing something. Need to figure out all rule names and if they have different presented/internal names.
    :param dist: the preference distribution which is being evaluated
    :return:
    """
    results_folder = "experiment_all_axioms"
    all_rule_violations = dict()

    all_k = []

    for k in range(1, m):

        filename = f"axiom_violation_results-n_profiles=25000-num_voters=50-m={m}-k={k}-pref_dist={dist}-axioms=all.csv"

        full_name = os.path.join(results_folder, filename)

        if not os.path.exists(full_name):
            print("File doesn't exist. Exiting.")
            exit()
        df = pd.read_csv(full_name)

        # From ChatGPT, this code suuuucks
        first_column = df.iloc[:, 0].tolist()  # rule_names
        second_column = df.iloc[:, 1].tolist()  # mean_violations

        rule_violations = dict(zip(first_column, second_column))

        for rule, violations in rule_violations.items():
            if rule not in all_rule_violations:
                all_rule_violations[rule] = []
            all_rule_violations[rule].append(violations)

        all_k.append(k)

    print(all_rule_violations)

    plt.figure(figsize=(14, 6), dpi=200)
    # cmap = plt.get_cmap('tab20', len(methods))

    for rule, violations in all_rule_violations.items():
        x_values = all_k
        y_values = violations
        if rule == "Neural Network":
            plt.plot(x_values, y_values, label=rule, linewidth=2, color="black", linestyle="--")
        else:
            plt.plot(x_values, y_values, label=rule)

    plt.xlabel("Number of Winners")
    plt.ylabel("Total Number of Violations")
    plt.title(f"# Violations for m={m} on {dist} preferences")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{results_folder}/rule_violations-m={m}-dist={dist}.png")

    # methods = summary_df['Method']
    # x_values = summary_df.columns[1:].astype(int)
    # y_values = summary_df.iloc[:, 1:]
    #
    # plt.figure(figsize=(14, 6), dpi=200)
    #
    # cmap = plt.get_cmap('tab20', len(methods))
    #
    # for i, method in enumerate(methods):
    #     plt.plot(x_values, y_values.iloc[i, :], marker='o', label=method, color=cmap(i))
    #
    # plt.xlabel("Number of Winners")
    # plt.ylabel("Number of Violations")
    # plt.title(f"Number of Violations for {dist} Distribution, {ax} Axiom, {num_candidates} Candidates")
    #
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    #
    # plt.tight_layout(rect=[0, 0, 0.85, 1])
    #
    # plot_filename = f"plots/violation_plot-dist={dist}-axiom={ax}-m={num_candidates}.png"
    # plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
    #
    # plt.close()



if __name__ == "__main__":
    train_networks()
    evaluate_networks()

    for dist in all_pref_models:
        plot_axioms_all_distributions_each_rule(m=6, dist=dist)

    # plot_axioms_all_distributions_each_rule(m=5, dist="IC")
    # plot_axioms_all_distributions_each_rule(m=5, dist="MALLOWS-RELPHI-R")
    # plot_axioms_all_distributions_each_rule(m=5, dist="single_peaked_conitzer")