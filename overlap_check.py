#!/usr/bin/env python3
import itertools

import pandas as pd

# read in the train and test set
# see how many overlaps there are

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
header = ["# Alternatives", "# Winners", "Distribution", "Overlap Count", "Overlap Percent"]
rows = []

for dist in all_pref_models:
    for m in [5, 6, 7]:
        for committee_size in range(1, m):
            base_path = "data/"
            filepath = f"n_profiles=25000-num_voters=50-m={m}-committee_size={committee_size}-pref_dist={dist}-axioms=all"
            test_suffix = "-TEST.csv"
            train_suffix = "-TRAIN.csv"

            test_path = base_path + filepath + test_suffix
            train_path = base_path + filepath + train_suffix

            try:
                test_df = pd.read_csv(test_path)
                train_df = pd.read_csv(train_path)

                # how many Profile entries in test_df are in train_df?
                train = train_df["Profile"].to_numpy()
                train_sets = []
                test = test_df["Profile"].to_numpy()
                test_sets = []
                overlap_count = 0
                for profile_idx in range(len(train)):
                    train_pr = set(train[profile_idx])
                    train_sets.append(train_pr)
                    test_pr = set(test[profile_idx])
                    test_sets.append(test_pr)
                # Check how many of the train sets appear in the test sets
                for train_set in train_sets:
                    if train_set in test_sets:
                        overlap_count += 1
                overlap_count = len(set(test_df['Profile']).intersection(set(train_df['Profile'])))
                print(f"m={m}, committee_size={committee_size}, dist={dist}")
                print(f"Total overlaps: {overlap_count}")
                print(f"Percentage overlap: {(overlap_count / len(test_df)) * 100:.2f}%\n")

                rows.append([m, committee_size, dist, overlap_count, round(overlap_count / len(test_df), 4)])
            except Exception as f:
                print(f"Exception: {f}")

    df = pd.DataFrame(data=rows, columns=header)
    df.to_csv("overlap_amounts_all_distributions.csv", index=False)
