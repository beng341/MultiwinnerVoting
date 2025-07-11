import os
import sys
import random
from itertools import product
import numpy as np
import pandas as pd
import torch
from abcvoting import abcrules
from sklearn.metrics import accuracy_score
from utils import data_utils as du
from utils import ml_utils
import pprint


def model_accuracies(
        test_df, 
        features, 
        model_paths, 
        num_winners, 
        n_voters, 
        varied_voters,
        voters_std_dev,
        m, 
        pref_dist, 
        out_folder, 
        include_best_and_worst=False
    ):
    """

    :param num_winners:
    :param features:
    :param test_df:
    :param model_paths:
    :return:
    """
    all_rule_predictions = dict()

    viols = dict()
    viols["Neural Network"] = dict()

    # find predictions for each individual network
    network_idx = 0
    for model_path in model_paths:
        try:
            model = ml_utils.load_model(model_path)
            model.eval()
        except Exception as e:
            print("Caught exception when loading networks. Skipping this results file. Exception is:")
            print(e)
            return None

        x = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            y = model(x)

        committee_size = len(y[0])
        # y_pred = [torch.argmax(y_i).item() for y_i in y]
        y_pred_committee = [np.argpartition(y_i, -num_winners)[-num_winners:].tolist() for y_i in y]
        y_pred_committees = [tuple([0 if idx not in yc else 1 for idx in range(committee_size)]) for yc in
                             y_pred_committee]
        all_rule_predictions[f"NN-{network_idx}"] = y_pred_committees
        network_idx += 1

    viols["Neural Network"] = {k: v // len(model_paths) for k, v in viols["Neural Network"].items()}

    num_candidates = len(y_pred_committees[0])
    num_committees = len(y_pred_committees)

    if include_best_and_worst:
        # include the best and worst committees identified
        best_committees = [eval(yt) for yt in test_df[f"min_violations-committee"]]
        all_rule_predictions["Min Violations Committee"] = best_committees

        worst_committees = [eval(yt) for yt in test_df[f"max_violations-committee"]]
        all_rule_predictions["Max Violations Committee"] = worst_committees

    # make predictions for random committees
    y_random_committees = []

    for _ in range(num_committees):
        committee = [1] * num_winners + [0] * (num_candidates - num_winners)
        random.shuffle(committee)
        y_random_committees.append(tuple(committee))
    all_rule_predictions["Random Choice"] = y_random_committees

    # Find outputs for each existing rule
    print("Counting violations for existing rules")
    voting_rules = du.load_mw_voting_rules(k=num_winners)

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

        all_rule_predictions[s] = y_true_rule

    make_rule_distances_table(
        all_rule_predictions, 
        n_voters, 
        varied_voters,
        voters_std_dev,
        m, 
        num_winners, 
        pref_dist, 
        out_folder=out_folder)

    profiles = test_df["Profile"]
    rank_matrix = test_df["rank_matrix"]
    candidate_pairs = test_df["candidate_pairs"]
    profile_info = {profiles[idx]: (rank_matrix[idx], candidate_pairs[idx]) for idx in range(len(profiles))}

    # all_predictions_per_profile = dict()  # maps each profile to set of all committees predicted for that profile
    unique_prediction_counts = []  # completely for Ben's own interest; should delete this

    # Make one set per unique profile to store all predicted committees for that profile
    all_predictions_per_profile = {profile: set() for profile in set(profiles)}
    for profile_idx in range(len(profiles)):
        # predicted_committees = set()
        # all_predictions_per_profile[profiles[profile_idx]] = set()
        # make set of predicted committees for each profile
        for rule, predictions in all_rule_predictions.items():
            current_pred = predictions[profile_idx]
            all_predictions_per_profile[profiles[profile_idx]].add(tuple(current_pred))

        unique_prediction_counts.append(len(all_predictions_per_profile[profiles[profile_idx]]))

    print(f"Average number of unique predicted committees is {np.mean(unique_prediction_counts)}")

    # For each test profile, find count of violations for each predicted committee
    # First, make pairs of all unique (profile, predicted committee) values
    unique_prediction_pairs = set()  # set instead of list because many profiles may be repeated at different indices
    for profile, committees in all_predictions_per_profile.items():
        for committee in committees:
            unique_prediction_pairs.add((profile, committee))
    unique_prediction_pairs = list(unique_prediction_pairs)
    unique_pred_pair_indices = {pp: idx for idx, pp in enumerate(unique_prediction_pairs)}

    # Count axiom violations for each unique profile/committee pair
    unique_profiles = []  # not actually all unique profiles; first component of all unique profile, committee pairs
    unique_committees = []
    for profile, committee in unique_prediction_pairs:
        unique_profiles.append(profile)
        unique_committees.append(committee)
    rank_matrices = [profile_info[profile][0] for profile in unique_profiles]
    cand_pairs = [profile_info[profile][1] for profile in unique_profiles]

    ax_violations = du.eval_all_axioms(rank_choice=rank_matrices,
                                       cand_pairs=cand_pairs,
                                       committees=unique_committees,
                                       n_winners=num_winners,
                                       profiles=unique_profiles)

    # Put everything back together to get count of violations for each individual rule

    axiom_names = sorted(list(ax_violations.keys()))
    all_rule_results = dict()
    for rule, predictions in all_rule_predictions.items():
        # Make list with one element per test example
        # Each element is a list of which axioms are violated by this rule by its prediction for that profile
        rule_ax_violations = []

        for idx, profile in enumerate(profiles):
            # get idx of profile in unique_profiles which corresponds to relevant idx in ax_violations
            p_idx = unique_pred_pair_indices[(profile, predictions[idx])]
            profile_ax_violations = [ax_violations[ax][p_idx] for ax in axiom_names]
            rule_ax_violations.append(profile_ax_violations)

        # Get mean/std result for each axiom
        # rule_ax_violations_sum = np.sum(rule_ax_violations, axis=0)
        rule_ax_violations_mean = np.mean(rule_ax_violations, axis=0)
        rule_ax_violations_std = np.std(rule_ax_violations, axis=0)

        merged_results = [rule]
        for idx in range(len(rule_ax_violations_mean)):
            merged_results.append(rule_ax_violations_mean[idx])
            # merged_results.append(rule_ax_violations_std[idx])
        all_rule_results[rule] = merged_results

        # if rule == "Greedy Monroe" and np.sum(rule_ax_violations_mean) > 0:
        #     print("Mean axiom violations for Greedy Monroe")
        #     print(rule_ax_violations_mean)
        #     axiom_idx = 8   # 8 = solid_coalitions
        #     # collect all row numbers where AV violates fixed majority
        #     violating_rows = [vidx for vidx in range(len(rule_ax_violations)) if rule_ax_violations[vidx][axiom_idx] > 0]
        #
        #     # collect violating profiles and Borda winners from corresponding rows in test_df
        #     violating_profiles = test_df.loc[violating_rows, "Profile"].tolist()
        #     violating_stv_winners = test_df.loc[violating_rows, "Greedy Monroe Winner"].tolist()
        #     for vidx in range(len(violating_rows)):
        #         print("Next violating profile:")
        #         pprint.pprint(eval(violating_profiles[vidx]))
        #
        #         print("Corresponding winner:")
        #         pprint.pprint(violating_stv_winners[vidx])
        #
        #         print("\n")
        #     exit()

    # Create Dataframe with all results (still need to merge individual network results)
    cols = ["Method"]
    for idx in range(len(axiom_names)):
        cols.append(f"{axiom_names[idx]}-mean")
        # cols.append(f"{axiom_names[idx]}-std")
    df = pd.DataFrame.from_dict(all_rule_results, columns=cols, orient="index")

    # # merge individual network rows into a single row
    # nn_rows = df[df['Method'].str.startswith('NN-')]
    # nn_mean = nn_rows.mean(numeric_only=True)
    # nn_mean_row = pd.DataFrame([['Neural Network'] + nn_mean.tolist()], columns=df.columns)
    # df = df[~df['Method'].str.startswith('NN-')]
    # df = pd.concat([df, nn_mean_row], ignore_index=True)

    # Add mean and std for total violation rate
    mean_columns = [col for col in df.columns if col.endswith('-mean')]
    # std_columns = [col for col in df.columns if col.endswith('-std')]
    df['violation_rate-mean'] = df[mean_columns].mean(axis=1)
    new_column_order = ['Method', 'violation_rate-mean'] + \
                       [col for col in df.columns if col not in ['violation_rate-mean', 'Method']]
    df = df[new_column_order]

    return df


def make_rule_distances_table(all_rule_predictions, n_voters, varied_voters, voters_std_dev, m, num_winners, pref_dist, out_folder):
    distances = {}

    nn_distances = {}

    # Find distance from each single NN to each rule
    for rule, predictions in all_rule_predictions.items():
        if not rule.startswith("NN-"):
            continue
        nn_distances[rule] = {}  # save it in a different dict for easy averaging
        for otherrule, otherpredictions in all_rule_predictions.items():
            if otherrule.startswith("NN-"):
                continue
            nn_distances[rule][otherrule] = np.mean(np.abs(np.array(predictions) - np.array(otherpredictions)))

    distances["NN"] = {}
    # Find average distance from NNs to each rule
    for nn_rule, nn_dists in nn_distances.items():
        # add marginal contribution of each network to the overall average distance between each rule
        for rule, dist in nn_dists.items():
            if rule not in distances:
                distances[rule] = {}
                distances["NN"][rule] = 0
                distances[rule]["NN"] = 0
            distances["NN"][rule] += dist / len(nn_distances.keys())
            distances[rule]["NN"] += dist / len(nn_distances.keys())

    # Calculate distance between every non-neural network rule
    for rule, predictions in all_rule_predictions.items():
        if rule.startswith("NN-"):
            # We've already dealt with all NNs, ignore them here
            continue
        distances[rule] = {}
        distances[rule]["NN"] = distances["NN"][rule]
        # go over each other rules
        for otherrule, otherpredictions in all_rule_predictions.items():
            if otherrule.startswith("NN-"):
                continue

            if otherrule in distances[rule]:
                # We have already calculated this from the other side of the matrix
                continue
            numpy_distance = np.mean(np.abs(np.array(predictions) - np.array(otherpredictions)))
            distances[rule][otherrule] = numpy_distance
            distances[otherrule][rule] = numpy_distance

    distances_df = pd.DataFrame(distances)

    # # Sort rows and columns to match order elsewhere (not vital here but might as well do it for readability)
    # col_row_order = ["NN", "Random Choice", "Borda ranking", "Plurality ranking", "STV", "Approval Voting (AV)", "Proportional Approval Voting (PAV)", "Approval Chamberlin-Courant (CC)", "Lexicographic Chamberlin-Courant (lex-CC)", "Sequential Approval Chamberlin-Courant (seq-CC)", "Monroe's Approval Rule (Monroe)", "Greedy Monroe", "Minimax Approval Voting (MAV)"]
    # distances_df = distances_df.reindex(columns=col_row_order).reindex(index=col_row_order)

    directory = f"{out_folder}/rule_distances"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = f"{directory}/num_voters={n_voters}-varied_voters={varied_voters}-voters_std_dev={voters_std_dev}-m={m}-k={num_winners}-pref_dist={pref_dist}-distances.csv"

    print(f"Saving rule distances to: {file_name}")

    distances_df.to_csv(file_name)


def save_accuracies_of_all_network_types(
        test_size, 
        n_voters, 
        varied_voters,
        voters_std_dev,
        m, 
        num_winners, 
        pref_dist, 
        axioms, 
        base_data_folder="data",
        out_folder="results", 
        base_model_folder="trained_networks",
        skip_if_result_file_exists=False
    ):
    """
    Loop over all parameter combinations and save the accuracy of each group of saved networks at predicting elections
    from the specified distribution.
    :return:
    """

    _, _, _, _, features_all, losses_all, num_trained_models_per_param_set = ml_utils.get_default_parameter_value_sets(
        m=True, n=True, train_size=False, num_winners=True, pref_dists=True, features=True, losses=True,
        networks_per_param=True)

    if len(losses_all) > 1 or len(features_all) > 1:
        print("For simplicity, evaluation was hobbled to allow only one loss/feature set at a time.")
        print("Adjust your parameters or adjust evaluate (should be straightforward to fix evaluation functionality)")
        exit()

    all_viols = dict()

    for features, loss in product(features_all, losses_all):

        base_name = f"axiom_violation_results-n_profiles={test_size}-num_voters={n_voters}-m={m}-k={num_winners}-pref_dist={pref_dist}-axioms={axioms}.csv"
        filename = os.path.join(out_folder, base_name)
        if os.path.isfile(path=filename) and skip_if_result_file_exists:
            print(f"Found existing results file: {filename}")
            print("Skipping generation of new results.")
            continue
        else:
            print(f"No existing results file found at: {filename}")

        if not os.path.exists(out_folder):
            print(f"{out_folder} does not exist; making it now")
            os.makedirs(out_folder)

        test_df = du.load_data(size=test_size,
                               n=n_voters,
                               m=m,
                               varied_voters=varied_voters,
                               voters_std_dev=voters_std_dev,
                               num_winners=num_winners,
                               pref_dist=pref_dist,
                               axioms=axioms,
                               train=False,
                               base_data_folder=base_data_folder,
                               make_data_if_needed=True)
        if test_df is None:
            print("Could not find test file with the given parameters. Stopping testing.")
            break

        feature_values = ml_utils.features_from_column_abbreviations(test_df, features)

        # Generate paths to all models
        model_paths = ml_utils.saved_model_paths(n=n_voters,
                                                 m=m,
                                                 num_winners=num_winners,
                                                 pref_dist=pref_dist,
                                                 axioms=axioms,
                                                 features=features,
                                                 num_models=num_trained_models_per_param_set,
                                                 loss=loss,
                                                 base_model_folder=base_model_folder)

        # Compute accuracy and axiom violations of each model
        # model_viols = model_accuracies(test_df,
        #                                features=feature_values,
        #                                model_paths=model_paths,
        #                                num_winners=num_winners)

        violation_results_df = model_accuracies(test_df,
                                                features=feature_values,
                                                model_paths=model_paths,
                                                num_winners=num_winners,
                                                n_voters=n_voters,
                                                varied_voters=varied_voters,
                                                voters_std_dev=voters_std_dev,
                                                m=m,
                                                pref_dist=pref_dist,
                                                out_folder=out_folder,
                                                include_best_and_worst=True)

        if violation_results_df is None:
            print(f"A network was not properly loaded. Skipping results for file: {base_name}")
            continue

        violation_results_df.to_csv(filename, index=False)
        print(f"Saving results to: {filename}")


def evaluate_networks_from_cmd():
    args = dict()
    if len(sys.argv) > 1:
        kw = dict(arg.split('=', 1) for arg in sys.argv[1:])
        for k, v in kw.items():
            args[k] = eval(v)

    n_profiles = args["n_profiles"]
    n_voters = args["n_voters"]
    varied_voters = args["varied_voters"]
    voters_std_dev = args["voters_std_dev"]
    m = args["m"]
    num_winners = args["num_winners"]
    data_path = args["data_path"]
    base_model_folder = args["network_path"]

    output_folder = args["out_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    axioms = args.get("axioms", "all")

    if "pref_dist" in args:
        all_pref_models = [args["pref_dist"]]
    else:
        all_pref_models = [
            "stratification__args__weight=0.5",         # Complete on all dists
            "URN-R",                                    # Complete on all dists
            "IC",                                       # Complete on all dists
            "IAC",                                      # Complete on all dists
            "identity",                                 # Complete on all dists
            "MALLOWS-RELPHI-R",
            "single_peaked_conitzer",
            "single_peaked_walsh",
            "euclidean__args__dimensions=3_-_space=gaussian_ball",      # m = 6 is done to here
            "euclidean__args__dimensions=10_-_space=gaussian_ball",
            "euclidean__args__dimensions=3_-_space=uniform_ball",       # m = 5 is done to here
            "euclidean__args__dimensions=10_-_space=uniform_ball",
            "euclidean__args__dimensions=3_-_space=gaussian_cube",
            "euclidean__args__dimensions=10_-_space=gaussian_cube",
            "euclidean__args__dimensions=3_-_space=uniform_cube",
            "euclidean__args__dimensions=10_-_space=uniform_cube",
            "mixed"
            # "real_world"
        ]

    for dist in all_pref_models:
        save_accuracies_of_all_network_types(
            test_size=n_profiles,
            n_voters=n_voters,
            varied_voters=varied_voters,
            voters_std_dev=voters_std_dev,
            m=m,
            num_winners=num_winners,
            pref_dist=dist,
            axioms=axioms,
            base_data_folder=data_path,
            out_folder=output_folder,
            base_model_folder=base_model_folder,
            skip_if_result_file_exists=True
        )


if __name__ == "__main__":
    evaluate_networks_from_cmd()
