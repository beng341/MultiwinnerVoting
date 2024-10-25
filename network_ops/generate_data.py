import os.path
import pprint
import sys
import pandas as pd
import pref_voting.profiles
import utils.data_utils
from pref_voting.generate_profiles import generate_profile as gen_prof
from utils import data_utils as du
from utils import voting_utils as vut
from abcvoting.preferences import Profile, Voter
from abcvoting import abcrules
import random


def create_profiles(args, **kwargs):
    """
    Given appropriate parameters create a dataframe containing one column with one preference profiles per row.
    Each preference profiles is saved as a string.
    :param args:
    :param kwargs:
    :return:
    """
    n_profiles = args["n_profiles"]
    prefs_per_profile = args["prefs_per_profile"]
    m = args["m"]
    pref_model = args["learned_pref_model"]
    num_winners = args["num_winners"]

    profiles = []
    abc_profile = []
    pref_voting_profiles = []

    # num_rejects = 0
    # for _ in range(n_profiles):
    while len(profiles) < n_profiles:
        profile = generate_profile(n=prefs_per_profile, m=m, model=pref_model, **kwargs)

        rankings = profile.rankings
        
        # randomly relabel alternatives
        # this step should ensure data is generated in a way that does not violate the neutrality principle

        labels = list(range(len(rankings[0])))
        relabeling = labels[:]
        random.shuffle(relabeling)
        relabel_dict = {labels[i]: relabeling[i] for i in range(len(rankings[0]))}
        rankings = [tuple(relabel_dict[x] for x in t) for t in rankings]

        profiles.append(rankings)

        abcvoting_profile = du.abc_profile_from_rankings(m=m, k=num_winners, rankings=rankings)
        # abcvoting_profile = Profile(num_cand=m)
        #
        # for rank in rankings:
        #     abcvoting_profile.add_voter(Voter(list(rank[:num_winners])))

        abc_profile.append(abcvoting_profile)
        pref_voting_profiles.append(pref_voting.profiles.Profile(rankings=rankings))
    # columns = ["Profile"]
    # profiles_df = pd.DataFrame(profiles, columns=columns)
    return profiles, abc_profile, pref_voting_profiles

    # abcvoting_profile = Profile(num_cand=m)

    # for rank in rankings:
    #    abcvoting_profile.add_voter(Voter(list(rank[:n_winners])))

    # profiles.append(f"{rankings}")
    # abc_profile.append(abcvoting_profile)
    # pref_voting_profiles.append(profiles)
    # columns = ["raw_profiles"]
    # profiles_df = pd.DataFrame(profiles, columns=columns)

    # return profiles_df, abc_profile, pref_voting_profiles


def generate_profile(n, m, model, **kwargs):
    """
    Generate a profiles of the given model and size using pref-voting library.
    :param n:
    :param m:
    :param model:
    :return:
    """
    return gen_prof(num_candidates=m, num_voters=n, probmodel=model, **kwargs)


def generate_computed_data(df):
    """
    Add columns for data computed from preference profiles like candidate pair matrices, binary candidate pairs, etc.
    These are what are actually used as input features.
    :param df:
    :return:
    """
    profiles = [elem for elem in df["Profile"].tolist()]

    # Initialize an empty list to collect all feature dictionaries
    all_features = []

    # Process each profiles individually
    for profile in profiles:
        features_dict = du.compute_features_from_profiles(profile)
        all_features.append(features_dict)

    # Convert the list of feature dictionaries into a DataFrame
    features_df = pd.DataFrame(all_features)

    # Concatenate the original DataFrame with the computed features
    df = pd.concat([df, features_df], axis=1)

    return df


def make_multi_winner_datasets(train=None):
    """
    Make datasets according to parameters set within this method. Save them to disk.
    This includes creation of voter preferences, winner computation, and transforming profiles to input features.
    :return:
    """

    # list of all preference models Ben used in some other project
    # follow the code to see how arguments are parsed from the string
    pref_models = [
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
    ]
    n_profiles = 88  # size of dataset generated
    n_voters = 20  # number of voters per profiles
    m = 5  # number of candidates in each profiles
    k = 3
    output_frequency = max(n_profiles // 20, 50)

    for pref_model in pref_models:
        args = {
            "n_profiles": n_profiles,
            "prefs_per_profile": n_voters,
            "m": m,
            "num_winners": k,
            "pref_model": pref_model,
            "axioms": "all",
            "out_folder": "results"
        }

        make_one_multi_winner_dataset(args=args,
                                      output_frequency=output_frequency
                                      )


def make_one_multi_winner_dataset(args, output_frequency=100, train=True, test=True):
    """
    Extracted from make_multi_winner_datasets() to allow calling it from elsewhere
    :param args
    :param output_frequency: Every time this many examples are generated the partial dataset is saved to file.
    :param train: True iff training data should be generated. Can generate train and test data at same time.
    :param test: True iff testing data should be generated. Can generate train and test data at same time.
    :return:
    """
    train_test_types = []
    if train:
        train_test_types.append(True)
    if test:
        train_test_types.append(False)
    for train in train_test_types:

        if train:
            type = "TRAIN"
        else:
            type = "TEST"

        # update args based on command line arguments
        print(sys.argv)
        if len(sys.argv) > 1:
            kw = dict(arg.split('=', 1) for arg in sys.argv[1:])
            for k, v in kw.items():
                args[k] = eval(v)

        n_profiles = args["n_profiles"]
        n_voters = args["prefs_per_profile"]
        m = args["m"]
        num_winners = args["num_winners"]
        # Can never remember whether this is supposed to be pref_model or learned_pref_model.
        # Leaving comment so if it breaks you can switch to to the other key :/
        pref_model = args["pref_model"]

        output_folder = args["out_folder"]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        axioms = args["axioms"]

        pref_model_shortname, kwargs = du.kwargs_from_pref_models(pref_model)
        args["learned_pref_model"] = pref_model_shortname

        print(
            f"Making a {type} dataset with {n_profiles} profiles, {n_voters} voters per profiles, {m} candidates, and {num_winners} winners, using a {pref_model} distribution and {axioms} axioms.")

        profiles, abc_profiles, pref_voting_profiles = create_profiles(args=args, **kwargs)

        profile_dict = {"Profile": [],
                        "min_violations-committee": [], "min_violations": [],
                        "max_violations-committee": [], "max_violations": [],}
        # For each profile, find committee with the least axiom violations
        for idx, profile in enumerate(profiles):

            # Track best/worst committees during training AND testing so we can see how close each rule is to each side
            minv_committee, minv, maxv_committee, maxv = du.find_winners(profile, num_winners,
                                                                         axioms_to_evaluate=[axioms])
            # # Don't bother finding min violations committee when making test data
            # if type == "TRAIN":
            #     minv_committee, minv, maxv_committee, maxv = du.find_winners(profile, num_winners, axioms_to_evaluate=[axioms])
            # elif type == "TEST":
            #     minv_committee, minv, maxv_committee, maxv = [[-1] * m], -1, [[-1] * m], -1
            # else:
            #     print(f"Somehow 'type' variable got bad value: {type}")
            #     exit()
            abc_profile = abc_profiles[idx]
            pref_voting_profile = pref_voting_profiles[idx]

            if len(minv_committee) > 1:
                # ensure lexicographic tie-breaking among tied winners
                # unclear if this is strictly better than random tie-breaking
                minv_committee.sort(key=lambda x: tuple(x))

            profile_dict["Profile"].append(profile)
            profile_dict["min_violations-committee"].append(tuple(minv_committee[0]))
            profile_dict["min_violations"].append(minv)
            profile_dict["max_violations-committee"].append(tuple(maxv_committee[0]))
            profile_dict["max_violations"].append(maxv)

            if type == "TEST":
                # Only calculate winners of each individual voting rule when making test data
                voting_rules = du.load_mw_voting_rules(k=num_winners)
                for rule in voting_rules:
                    if isinstance(rule, str):
                        # rule should be an abc rule
                        s = abcrules.get_rule(rule).longname
                        prof = abc_profile
                        abc_rule = True
                    else:
                        # should be from outside the abc library and we give the pref_voting profile
                        s = rule.name
                        prof = pref_voting_profile
                        abc_rule = False
                    try:
                        single_winner, _ = du.generate_winners(rule, [prof], num_winners, m, abc_rule=abc_rule)

                        if f"{s} Winner" not in profile_dict:
                            profile_dict[f"{s} Winner"] = []
                        profile_dict[f"{s} Winner"].append(single_winner[0])

                    except Exception as ex:
                        print(f"{s} broke everything")
                        print(f"{ex}")
                        print(f"Profile is:")
                        pprint.pprint(profile)
                        print("")
                        exit()

            # output the dataset once in a while in case execution is interrupted
            if idx % output_frequency == 0 and idx > 0:
                profiles_df = pd.DataFrame.from_dict(profile_dict)
                profiles_df = generate_computed_data(profiles_df)
                filename = f"n_profiles={args['n_profiles']}-num_voters={args['prefs_per_profile']}-m={args['m']}-committee_size={num_winners}-pref_dist={pref_model}-axioms={args['axioms']}-{type}.csv"
                filepath = os.path.join(output_folder, filename)
                profiles_df.to_csv(filepath, index=False)
                print(f"Saving partial dataset to: {filepath}")

        # Output the complete dataset for good measure, likely redundant
        profiles_df = pd.DataFrame.from_dict(profile_dict)
        profiles_df = generate_computed_data(profiles_df)
        filename = f"n_profiles={args['n_profiles']}-num_voters={args['prefs_per_profile']}-m={args['m']}-committee_size={num_winners}-pref_dist={pref_model}-axioms={args['axioms']}-{type}.csv"
        filepath = os.path.join(output_folder, filename)
        profiles_df.to_csv(filepath, index=False)
        print(f"Saving complete dataset to: {filepath}")


def make_dataset():
    """

    :return:
    """
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
    args = {
        "n_profiles": 100,
        "prefs_per_profile": 20,
        "m": 6,
        "num_winners": 3,
        "pref_model": all_pref_models[10],
        "axioms": "all",
        "out_folder": "testing_test_data"
    }

    output_folder = args["out_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pref_model_shortname, kwargs = du.kwargs_from_pref_models(all_pref_models[1])
    args["learned_pref_model"] = pref_model_shortname

    make_one_multi_winner_dataset(args, output_frequency=100, train=False, test=True)


def make_dataset_from_cmd():
    """
    Make a dataset given arguments from command line. Assumes that ALL possible arguments are given by cmd.
    :return:
    """

    output_frequency = 1000
    args = {
    }

    make_one_multi_winner_dataset(args=args,
                                  output_frequency=output_frequency,
                                  train=True,
                                  test=True
                                  )


if __name__ == "__main__":
    make_dataset_from_cmd()
    # make_multi_winner_datasets()
    # make_dataset()
