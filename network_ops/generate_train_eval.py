import os
import sys

from random import randint
import train_networks as tn
import evaluate_networks as en
from utils import data_utils as du
import pandas as pd
import itertools
import matplotlib.pyplot as plt


def generate_train_eval():

    # Edit the distributions you could like to use, mixed is a mix of all of them
    distributions = [
        'stratification__args__weight=0.5',
        'URN-R',
        'IC',
        'MALLOWS-RELPHI-R',
        'single_peaked_conitzer',
        'IAC',
        'euclidean',
        'euclidean__args__dimensions=3_space=gaussian-ball',
        'euclidean__args__dimensions=3_space=uniform-sphere',
        'euclidean__args__dimensions=3_space=gaussian-cube',
        "mixed"
    ]

    # select which axioms you would like to use
    axioms = [
        "all",
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

    # Define the arguments you would like to use
    experiment_args = {
        "n_profiles": 100,
        "num_voters": 50,
        "num_candidates": range(7, 8),
        "num_winners": [2, 3, 4, 5, 6]
    }

    # BELOW HERE YOU SHOULDNT NEED TO TOUCH
    expected_columns = [
        'Method', 'total_violations', 'majority', 'majority_loser',
        'fixed_majority', 'condorcet_winner', 'condorcet_loser',
        'dummetts_condition', 'solid_coalitions', 'consensus_committee',
        'strong_unanimity', 'local_stability', 'strong_pareto_efficiency',
        'count_viols'
    ]

    # left is what the ax, right is what the column is called in the dataset
    axiom_map = {
        "all": "total_violations",
        "dummett": "dummetts_condition",
        "consensus": "consensus_committee",
        "fixed_majority": "fixed_majority",
        "majority_winner": "majority",
        "majority_loser": "majority_loser",
        "condorcet_winner": "condorcet_winner",
        "condorcet_loser": "condorcet_loser",
        "solid_coalition": "solid_coalitions",
        "strong_unanimity": "strong_unanimity",
        "local_stability": "local_stability",
        "strong_pareto": "strong_pareto_efficiency"
    }

    # generate single axiom, single distribution datasets
    # train_networks(train_size, n, m, num_winners, pref_dist, axioms)


    for dist in distributions:
        for ax in axioms:
            for num_candidates in experiment_args["num_candidates"]:
                datasets = {}
                for num_winners in experiment_args["num_winners"]:

                    if dist == "mixed":
                        # TODO: Unclear why this is done every time? Should probably just be run once then include mixed in the list of distributions
                        du.generate_mixed_distribution(distributions=distributions[:-1],
                                                       total_size=experiment_args["n_profiles"],
                                                       n=experiment_args["num_voters"],
                                                       m=num_candidates,
                                                       num_winners=num_winners,
                                                       axioms=ax)


                    print("Training)")
                    tn.train_networks(experiment_args["n_profiles"],
                                      experiment_args["num_voters"],
                                      num_candidates,
                                      num_winners,
                                      dist,
                                      ax
                    )

                    # train 20+ networks
                    print("Evaluating")

                    # evaluate them and average them together
                    #save_accuracies_of_all_network_types(test_size, n, m, num_winners, pref_dist, axioms, out_folder="results")
                    en.save_accuracies_of_all_network_types(experiment_args["n_profiles"],
                                                            experiment_args["num_voters"],
                                                            num_candidates,
                                                            num_winners,
                                                            dist,
                                                            ax
                    )


                    results_filename = (
                        f"results/axiom_violation_results-"
                        f"n_profiles={experiment_args['n_profiles']}-"
                        f"num_voters={experiment_args['num_voters']}-"
                        f"m={num_candidates}-"
                        f"k={num_winners}-"
                        f"pref_dist={dist}-"
                        f"axioms={ax}.csv"
                    )

                    if not os.path.exists(results_filename):
                        print(f"Warning: {results_filename} does not exist. Skipping.")
                        continue

                    # Read the results CSV
                    try:
                        result_dataset = pd.read_csv(results_filename)
                        result_dataset.rename(columns={result_dataset.columns[0]: "Method"}, inplace=True)
                    except Exception as e:
                        print(f"Error reading {results_filename}: {e}")
                        exit(1)
                        continue

                    # Ensure that 'Method' column is present
                    if 'Method' not in result_dataset.columns:
                        print(f"Error: 'Method' column not found in {results_filename}. Skipping.")
                        exit(1)
                        continue

                    # Store the DataFrame in the datasets dictionary
                    datasets[num_winners] = result_dataset
                
                # We want to generate a summary table that finds the number of violations for each setting
                    # so we have a given distribution, axiom, number of candidates, and we want a column in that table for each number of winners
                if not datasets:
                    print(f"No datasets found for Distribution: {dist}, Axiom: {ax}, Candidates: {num_candidates}. Skipping summary.")
                    exit(1)
                    continue

                summary_df = None
                sorted_num_winners = sorted(datasets.keys())

                violation_column = axiom_map.get(ax, "total_violations")
                if violation_column not in expected_columns:
                    print(f"Error: {violation_column} not found in expected columns. Skipping summary.")
                    exit(1)
                    continue
                
                for num_winners in sorted_num_winners:
                    df = datasets[num_winners][['Method', violation_column]].copy()
                    df.rename(columns={violation_column: str(num_winners)}, inplace=True)

                    if summary_df is None:
                        summary_df = df
                    else:
                        summary_df = summary_df.merge(df, on='Method', how='inner')

                summary_filename = (
                    f"summary_results/summary_violation_results-"
                    f"n_profiles={experiment_args['n_profiles']}-"
                    f"num_voters={experiment_args['num_voters']}-"
                    f"m={num_candidates}-"
                    f"pref_dist={dist}-"
                    f"axioms={ax}.csv"
                )

                try:
                    summary_df.to_csv(summary_filename, index=False)
                    print(f"Summary saved to {summary_filename}\n")
                except Exception as e:
                    print(f"Error saving summary to {summary_filename}: {e}")
                    exit(1)

                

                # making plots
                methods = summary_df['Method']
                x_values = summary_df.columns[1:].astype(int)
                y_values = summary_df.iloc[:, 1:]

                plt.figure(figsize=(14, 6))

                cmap = plt.get_cmap('tab20', len(methods))  

                for i, method in enumerate(methods):
                    plt.plot(x_values, y_values.iloc[i, :], marker='o', label=method, color=cmap(i))

                plt.xlabel("Number of Winners")
                plt.ylabel("Number of Violations")
                plt.title(f"Number of Violations for {dist} Distribution, {ax} Axiom, {num_candidates} Candidates")

                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

                plt.tight_layout(rect=[0, 0, 0.85, 1])

                plot_filename = f"plots/violation_plot-dist={dist}-axiom={ax}-m={num_candidates}.png"
                plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')

                plt.close()
                



                

            
                



            






"""

    for num_candidates, num_winners, dist, ax in itertools.product(experiment_args["num_candidates"], experiment_args["num_winners"], distributions, axioms):
        print("Training)")
        tn.train_networks(experiment_args["n_profiles"],
                          experiment_args["num_voters"],
                          num_candidates,
                          num_winners,
                          dist,
                          ax
        )

        # train 20+ networks
        print("Evaluating")

        # evaluate them and average them together
        #save_accuracies_of_all_network_types(test_size, n, m, num_winners, pref_dist, axioms, out_folder="results")
        en.save_accuracies_of_all_network_types(experiment_args["n_profiles"],
                                                experiment_args["num_voters"],
                                                num_candidates,
                                                num_winners,
                                                dist,
                                                ax
        )

    # generate single axiom, multiple distribution datasets

        # train 20+ networks

        # evaluate them and average them together

    # generate multiple axioms, single distribution datasets

        # train 20+ networks

        # evaluate them and average them together
    
    # generate multiple axioms, multiple distribution datasets

        # train 20+ networks

        # evaluate them and average them together


"""
    



"""
    
    # Define what we want to make 
    pref_models = [
        # ('stratification__args__weight=0.5', 3),
        # ('URN-R', 3),
        # ('IC', 3),
        # ('MALLOWS-RELPHI-R', 3),
        # ('single_peaked_conitzer', 3),
        # ('IAC', 3),
        # ('euclidean', 3),
        # ('euclidean__args__dimensions=3_space=gaussian-ball', 3),
        # ('euclidean__args__dimensions=3_space=uniform-sphere', 3),
        # ('euclidean__args__dimensions=3_space=gaussian-cube', 3),
    ]

    # Expand the pref_models list based on repetition counts
    pref_models = [model for model, count in pref_models for _ in range(count)]

    profile_counts = [randint(3000, 8000) for _ in range(len(pref_models))]
    prefs_per_profile = [randint(20, 100) for _ in range(len(pref_models))]

    candidate_sizes = []

    # winners and be anything, then winners = num_candidates - 1, then winners = 2

    for i in range(len(pref_models)):
        if i % 3 == 0:
            candidate_sizes.append(randint(5, 7))
        elif i % 3 == 1:
            candidate_sizes.append(randint(3, 7))
        else:
            candidate_sizes.append(randint(4, 7))

    winners_sizes = []

    for i in range(len(pref_models)):
        if i % 3 == 0:
            winners_sizes.append(randint(3, candidate_sizes[i] - 2))
        elif i % 3 == 1:
            winners_sizes.append(candidate_sizes[i] - 1)
        else:
            winners_sizes.append(2)

    print("Running experiments with the following parameters:")
    print("candidate_sizes: ", candidate_sizes)
    print("winners_sizes: ", winners_sizes)
    
    # We don't need to generate the data, as training the network with these params
    # will generate the data for us

    for i, (distribution, profile_count, num_voters, num_candidates, num_winners) in enumerate(
            zip(pref_models, profile_counts, prefs_per_profile, candidate_sizes, winners_sizes)):
        # Train the networks
        tn.train_networks(train_size=profile_count,
                          n=num_voters,
                          m=num_candidates,
                          num_winners=num_winners,
                          pref_dist=distribution)

        print("")
        print("FINISHED TRAINING")
        print("NOW EVALUATING")
        print("")

        # Evaluate the networks
        en.save_accuracies_of_all_network_types(test_size=profile_count,
                                                n=num_voters,
                                                m=num_candidates,
                                                num_winners=num_winners,
                                                pref_dist=distribution)

    dists = [
        'stratification__args__weight=0.5',
        'URN-R',
        'IC',
        'MALLOWS-RELPHI-R',
        'single_peaked_conitzer',
        'IAC',
        'euclidean',
        'euclidean__args__dimensions=3_space=gaussian-ball',
        'euclidean__args__dimensions=3_space=uniform-sphere',
        'euclidean__args__dimensions=3_space=gaussian-cube'
    ]

    num_voters = 50
    num_candidates = 7
    winners = list(range(4, 5))

    for num_winners in winners:
        # load data for each distribution
        train_dfs = []
        test_dfs = []

        print(f"Running experiments with {num_winners} winners")

        for dist in dists:
            print(f"Creating {dist} data")
            train_dfs.append(du.load_data(10000, num_voters, num_candidates, num_winners, dist, True))
            test_dfs.append(du.load_data(10000, num_voters, num_candidates, num_winners, dist, False))

        # combine the data
        mixed_train = pd.concat(train_dfs, axis=0)
        mixed_test = pd.concat(test_dfs, axis=0)

        mixed_train = pd.concat(train_dfs, axis=0).reset_index(drop=True)
        mixed_test = pd.concat(test_dfs, axis=0).reset_index(drop=True)

        # Shuffle the DataFrames
        shuffled_train = mixed_train.sample(frac=1).reset_index(drop=True)
        shuffled_test = mixed_test.sample(frac=1).reset_index(drop=True)

        train_file = f"n_profiles={10000 * len(dists)}-num_voters={num_voters}-m={num_candidates}-committee_size={num_winners}-pref_dist=mixed-TRAIN.csv"
        test_file = f"n_profiles={10000 * len(dists)}-num_voters={num_voters}-m={num_candidates}-committee_size={num_winners}-pref_dist=mixed-TEST.csv"

        filepath = os.path.join("data", train_file)
        shuffled_train.to_csv(filepath, index=False)

        filepath = os.path.join("data", test_file)
        shuffled_test.to_csv(filepath, index=False)

        # train
        tn.train_networks(train_size=10000 * len(dists),
                          n=num_voters,
                          m=num_candidates,
                          num_winners=num_winners,
                          pref_dist="mixed")

        # evaluate
        en.save_accuracies_of_all_network_types(test_size=10000 * len(dists),
                                                n=num_voters,
                                                m=num_candidates,
                                                num_winners=num_winners,
                                                pref_dist="mixed")
        """


if __name__ == "__main__":
    generate_train_eval()
