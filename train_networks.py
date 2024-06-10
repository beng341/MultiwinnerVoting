import os
from itertools import product

import pandas as pd

from utils import ml_utils
import SingleWinnerVotingRule

# Define all the Networks that will be trained. Learn networks on all combinations of below parameters.
# NOTE: These parameters should match exactly the parameters used in generating the dataset. The information is encoded
# in the filenames and must match for data to load.
m_all = [5]  # all numbers of candidates
n_all = [20]  # all numbers of voters
train_size_all = [2000]  # training size
# rules_all = ["Plurality", "Borda", "Anti-Plurality", "Instant Runoff", "Benham", "Coombs", "Baldwin", "Strict Nanson", "Weak Nanson", "Raynaud", "Tideman Alternative Top Cycle", "Tideman Alternative GOCHA", "Knockout Voting", "Banks", "Condorcet", "Copeland", "Llull", "Uncovered Set", "Slater", "Top Cycle", "GOCHA", "Bipartisan Set", "Minimax", "Split Cycle", "Beat Path", "Simple Stable Voting", "Stable Voting", "Loss-Trimmer Voting", "Daunou", "Blacks", "Condorcet Plurality", "Copeland-Local-Borda", "Copeland-Global-Borda", "Borda-Minimax Faceoff", "Bucklin", "Simplified Bucklin", "Weighted Bucklin", "Bracket Voting", "Superior Voting"]
# rules_all = ["Plurality", "Borda", "Anti-Plurality", "Instant Runoff", "Banks", "Condorcet", "Copeland"]
rules_all = ["Approval Voting (AV)", "Lexicographic Chamberlin-Courant (lex-CC)"]  # List of rules to take as learning targets
# feature_set_all = ["b", "c", "r", "br", "bc",  "cr", "bcr"]
feature_set_all = ["bcr"]  # list of features to learn from (two letters means both features appended together)
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

networks_per_param_set = 2  # How many networks to learn for each combination of parameters

# create a config dict for each network that will get trained (several for each set of parameters)

base_data_folder = "data"
network_count = 0
for m, n, train_size, pref_dist, feature_set, rule in product(m_all, n_all, train_size_all, pref_dist_all,
                                                              feature_set_all, rules_all):

    filename = f"n_profiles={train_size}-num_voters={n}-m={m}-pref_dist={pref_dist}.csv"
    if not os.path.exists(f"{base_data_folder}/{filename}"):
        print(f"Tried loading path but it does not exist: {base_data_folder}/{filename}")
        continue

    df = pd.read_csv(f"{base_data_folder}/{filename}")
    for net_idx in range(networks_per_param_set):
        network_count += 1
        print(f"Network count: {network_count}")

        # sample training data, make fake config file to define this experiment
        train_sample = df.sample(n=train_size)
        features = ml_utils.features_from_column_abbreviations(train_sample, feature_set)

        name = f"num_voters={n}-m={m}-pref_dist={pref_dist}-features={feature_set}-rule={rule}-idx={net_idx}"
        config = {
            "experiment_name": name,
            "feature_column": ml_utils.feature_names_from_column_abbreviations(feature_set),
            "target_column": f"{rule}-single_winner",
            "tied_target_column": f"{rule}-tied_winners",
            "hidden_layers": 4,
            "hidden_nodes": 20,
            "output_folder": "./",
            "epochs": 200,
            "min_delta_loss": 0.001,
            "m": m,
            "num_features": len(features[0]),
            "experiment": name,  # I think this kwarg isn't used?
            "train_data": train_sample,
        }

        # training_sets[name] = config
        print(f"Finished making config for {name}")

        print(f"Beginning to train {name}")
        num_candidates = config["m"]
        num_features = config["num_features"]
        experiment = config["experiment"]
        train_df = config["train_data"]

        nn = SingleWinnerVotingRule.SingleWinnerVotingRule(num_candidates=num_candidates,
                                                           config=config,
                                                           experiment=experiment,
                                                           num_features=num_features)
        nn.train_df = train_df
        nn.trainmodel()

        nn.save_model()
