import os
from itertools import product
import torch.nn as nn
import pandas as pd

from utils import ml_utils
import MultiWinnerVotingRule

# Define all the Networks that will be trained. Learn networks on all combinations of below parameters.
# NOTE: These parameters should match exactly the parameters used in generating the dataset. The information is encoded
# in the filenames and must match for data to load.

train_size_all = [2000]
m_all, n_all, num_winners, pref_dist_all, feature_set_all, losses_all, networks_per_param_set = ml_utils.get_default_parameter_value_sets(
    m=True, n=True, train_size=False, num_winners=True, pref_dists=True, features=True, losses=True,
    networks_per_param=True)

# create a config dict for each network that will get trained (several for each set of parameters)

base_data_folder = "data"
network_count = 0
for m, n, train_size, pref_dist, feature_set, winners_size, loss in product(m_all, n_all, train_size_all, pref_dist_all,
                                                                            feature_set_all, num_winners, losses_all):

    filename = f"n_profiles={train_size}-num_voters={n}-m={m}-committee_size={winners_size}-pref_dist={pref_dist}.csv"
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

        name = f"num_voters={n}-m={m}-pref_dist={pref_dist}-features={feature_set}-loss={str(loss)}-idx={net_idx}"
        config = {
            "experiment_name": name,
            "feature_column": ml_utils.feature_names_from_column_abbreviations(feature_set),
            "target_column": f"Winner",
            "hidden_layers": 4,
            "hidden_nodes": 20,
            "output_folder": "./",
            "epochs": 200,
            "min_delta_loss": 0.001,
            "m": m,
            "n": n,
            "num_winners": num_winners,
            "num_features": len(features[0]),
            "experiment": name,  # I think this kwarg isn't used?
            "train_data": train_sample,
            "loss": loss
        }

        # training_sets[name] = config
        print(f"Finished making config for {name}")

        print(f"Beginning to train {name}")
        num_candidates = config["m"]
        num_voters = config["n"]
        num_features = config["num_features"]
        experiment = config["experiment"]
        train_df = config["train_data"]

        nn = MultiWinnerVotingRule.MultiWinnerVotingRule(num_candidates=num_candidates,
                                                         num_voters=num_voters,
                                                         num_winners=num_winners,
                                                         config=config,
                                                         experiment=experiment,
                                                         num_features=num_features)
        nn.train_df = train_df

        nn.trainmodel()

        nn.save_model()
