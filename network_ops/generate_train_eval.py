import os
import sys

from random import randint
import train_networks as tn
import evaluate_networks as en
from utils import data_utils as du
import pandas as pd


def generate_train_eval():
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
    winners = [6, 4, 2]

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


if __name__ == "__main__":
    generate_train_eval()
