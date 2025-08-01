import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import product
from utils import data_utils as du
from utils import ml_utils
from network_ops.MultiWinnerVotingRule import MultiWinnerVotingRule
import pandas as pd


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

def train_networks(
        train_size,
        n_voters,
        varied_voters,
        voters_std_dev,
        m,
        num_winners,
        pref_dist,
        axioms,
        base_data_folder="data",
        network_folder="./"):
    # feature_set, loss, networks_per_param_set

    _, _, _, _, feature_set_all, losses_all, networks_per_param_set = ml_utils.get_default_parameter_value_sets(
        m=True, n=True, train_size=False, num_winners=True, pref_dists=True, features=True, losses=True,
        networks_per_param=True)

    network_count = 0

    for feature_set, loss in product(feature_set_all, losses_all):

        df = du.load_data(size=train_size,
                          n=n_voters,
                          varied_voters=varied_voters,
                          voters_std_dev=voters_std_dev,
                          m=m,
                          num_winners=num_winners,
                          pref_dist=pref_dist,
                          axioms=axioms,
                          train=True,
                          base_data_folder=base_data_folder,
                          make_data_if_needed=False)
        if df is None:
            print(
                f"Could not find training file with n={n_voters}, m={m}, k={num_winners}, pref_dist={pref_dist}, axioms={axioms} in directory {base_data_folder}. Stopping training.")
            break
        print("")
        print("DATA LOADED")
        print("")
        
        all_losses = []

        for net_idx in range(networks_per_param_set):
            network_count += 1
            print(f"Network count: {network_count}")

            # sample training data, make fake config file to define this experiment
            train_sample = df  # df.sample(n=train_size)
            features = ml_utils.features_from_column_abbreviations(train_sample, feature_set)

            name = f"num_voters={n_voters}-m={m}-num_winners={num_winners}-pref_dist={pref_dist}-axioms={axioms}-features={feature_set}-loss={str(loss)}-idx={net_idx}"

            config = {
                "experiment_name": name,
                "feature_column": ml_utils.feature_names_from_column_abbreviations(feature_set),
                # "target_column": f"Winner",
                "target_column": "min_violations-committee",
                "hidden_layers": 5,
                "hidden_nodes": 256,
                "output_folder": network_folder,
                "epochs": 50,
                # "min_delta_loss": 0.001, # AAMAS value; also had patience set to 20
                "min_delta_loss": 0.0005,  # thesis/IJCAI value with patience of 10
                "m": m,
                "n": n_voters,
                "varied_voters": varied_voters,
                "voters_std_dev": voters_std_dev,
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

            nn = MultiWinnerVotingRule(num_candidates=num_candidates,
                                       num_voters=num_voters,
                                       num_winners=num_winners,
                                       config=config,
                                       experiment=experiment,
                                       num_features=num_features)
            nn.train_df = train_df

            # torch.save(checkpoint, f"{path}/NN-{self.config['experiment_name']}-{suffix}.pt")

            out_folder = config["output_folder"]
            path = os.path.join(out_folder, f"trained_networks", f"NN-{name}-.pt")

            # if os.path.isfile(path) and os.path.getsize(path) > 10000:
            #     # file exists and is not empty so we will assume it is a trained network
            #     # Error modes - breaks if a poorly saved network is larger than 10000 bytes
            #     # Should be much faster than loading and doing eval() on each saved network
            #     print(f"Network {name} already trained. Skipping.")
            # else:
            #     print("Saved network not found. Beginning training.")
            #     nn.trainmodel()
            #     nn.save_model()

            try:
                out_folder = config["output_folder"]
                path = os.path.join(out_folder, f"trained_networks", f"NN-{name}-.pt")
                ml_utils.load_model(path)
                print(f"Network {name} already trained. Skipping.")
            except Exception as e:
                print("Saved network not found. Beginning training. Caught exception:", e)
                _, network_losses = nn.trainmodel()
                all_losses.append(network_losses)

                nn.save_model()
        
        # all_losses is a list of lists
        # make it into a df where each list is the column
        # and each column name is the NN-{index}
        if len(all_losses) > 0:
            # Ensure the folder exists
            folder_path = os.path.join(out_folder, "losses")
            os.makedirs(folder_path, exist_ok=True)
            
            # Create DataFrame and save to CSV
            all_losses_df = pd.DataFrame(all_losses).T
            all_losses_df.columns = [f"NN-{i}" for i in range(len(all_losses))]
            all_losses_df.to_csv(os.path.join(folder_path, f"losses-{name}.csv"), index=False)

        


def train_networks_from_cmd():
    args = dict()
    if len(sys.argv) > 1:
        kw = dict(arg.split('=', 1) for arg in sys.argv[1:])
        for k, v in kw.items():
            args[k] = eval(v)

    n_profiles = args["n_profiles"]
    n_voters = args["n_voters"]
    m = args["m"]
    num_winners = args["num_winners"]
    data_path = args["data_path"]
    varied_voters = args["varied_voters"]
    voters_std_dev = args["voters_std_dev"]

    output_folder = args["out_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    axioms = args.get("axioms", "all")

    if "pref_dist" in args:
        all_pref_models = [args["pref_dist"]]
    else:
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
            # "real_world"
        ]

    for dist in all_pref_models:
        train_networks(train_size=n_profiles,
                       n_voters=n_voters,
                       varied_voters=varied_voters,
                       voters_std_dev=voters_std_dev,
                       m=m,
                       num_winners=num_winners,
                       pref_dist=dist,
                       axioms=axioms,
                       base_data_folder=data_path,
                       network_folder=output_folder
                       )


if __name__ == "__main__":
    train_networks_from_cmd()
