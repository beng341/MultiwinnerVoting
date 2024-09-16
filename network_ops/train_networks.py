import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import product
from utils import data_utils as du
from utils import ml_utils
import MultiWinnerVotingRule


# Define all the Networks that will be trained. Learn networks on all combinations of below parameters.
# NOTE: These parameters should match exactly the parameters used in generating the dataset. The information is encoded
# in the filenames and must match for data to load.

# train_size_all = [10000]
# m_all, n_all, n_winners, pref_dist_all, feature_set_all, losses_all, networks_per_param_set = ml_utils.get_default_parameter_value_sets(
#    m=True, n=True, train_size=False, n_winners=True, pref_dists=True, features=True, losses=True,
#    networks_per_param=True)

# m_all = [8]
# n_winners = [3]
# losses_all = [losses_all[0]]

# create a config dict for each network that will get trained (several for each set of parameters)

# base_data_folder = "data"
# network_count = 0

# dimensions = [(5957, 55, 6, 3), (8846, 37, 7, 4), (3792, 24, 9, 2), (3686, 39, 10, 3)]
# ptr = 0

# for pref_dist, feature_set, loss in product(pref_dist_all, feature_set_all, losses_all):

# train_size, n, m, num_winners = 5000, 100, 6, 3
# ptr += 1

def train_networks(train_size, n, m, num_winners, pref_dist):
    # feature_set, loss, networks_per_param_set

    _, _, _, _, feature_set_all, losses_all, networks_per_param_set = ml_utils.get_default_parameter_value_sets(
        m=True, n=True, train_size=False, num_winners=True, pref_dists=True, features=True, losses=True,
        networks_per_param=True)

    network_count = 0

    for feature_set, loss in product(feature_set_all, losses_all):

        df = du.load_data(size=train_size,
                          n=n,
                          m=m,
                          num_winners=num_winners,
                          pref_dist=pref_dist,
                          train=True,
                          make_data_if_needed=True)
        print("")
        print("DATA LOADED")
        print("")
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
                "hidden_layers": 3,
                "hidden_nodes": 128,
                "output_folder": "./",
                "epochs": 10,
                "min_delta_loss": 0.001,
                "m": m,
                "n": n,
                "n_winners": num_winners,
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


if __name__ == "__main__":
    pref_models = [
        "URN-R",
        "IC",
        "identity",
        "MALLOWS-RELPHI-R",
        # "mixed"
    ]

    size = 1000
    num_voters = 20
    num_candidates = 7
    winners = 3
    for dist in pref_models:
        train_networks(train_size=size, n=num_voters, m=num_candidates, num_winners=winners, pref_dist=dist)
