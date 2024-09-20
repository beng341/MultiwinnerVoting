import os.path
import sys
import pandas as pd
import pref_voting.profiles
from pref_voting.generate_profiles import generate_profile as gen_prof
from utils import data_utils as du
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

        abcvoting_profile = Profile(num_cand=m)

        for rank in rankings:
            abcvoting_profile.add_voter(Voter(list(rank[:num_winners])))

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
    n_profiles = 1000  # size of dataset generated
    n_voters = 20  # number of voters per profiles
    m = 7  # number of candidates in each profiles
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
            "output_folder": "results"
        }

        make_one_multi_winner_dataset(args=args,
                                      output_frequency=output_frequency
                                      )


def make_one_multi_winner_dataset(args, output_frequency=100):
    """
    Extracted from make_multi_winner_datasets() to allow calling it from elsewhere
    :param args
    :param output_frequency: Every time this many examples are generated the partial dataset is saved to file.
    :return:
    """

    for train in [True, False]:

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
        pref_model = args["pref_model"]

        output_folder = args["out_folder"]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        axioms = args["axioms"]

        pref_model_shortname, kwargs = du.kwargs_from_pref_models(pref_model)
        args["learned_pref_model"] = pref_model_shortname

        print(
            f"Making a {type} dataset with {n_profiles} profiles, {n_voters} voters per profiles, {m} candidates, and {num_winners} winners, using a {pref_model} distribution.")

        profiles, abc_profiles, pref_voting_profiles = create_profiles(args=args, **kwargs)

        profile_dict = {"Profile": [], "Winner": [], "Num_Violations": []}
        # For each profile, find committee with the least axiom violations
        for idx, profile in enumerate(profiles):
            winners, min_violations = du.find_winners(profile, num_winners, axioms_to_evaluate=axioms)
            abc_profile = abc_profiles[idx]
            pref_voting_profile = pref_voting_profiles[idx]

            if len(winners) > 1:
                # ensure lexicographic tie-breaking among tied winners
                # unclear if this is strictly better than random tie-breaking
                winners.sort(key=lambda x: tuple(x))

            profile_dict["Profile"].append(profile)
            profile_dict["Winner"].append(tuple(winners[0]))
            profile_dict["Num_Violations"].append(min_violations)

            voting_rules = du.load_mw_voting_rules()
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
                    return

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


def make_dataset_from_cmd():
    """
    Make a dataset given arguments from command line. Assumes that ALL possible arguments are given by cmd.
    :return:
    """

    output_frequency = 1000
    args = {
    }

    make_one_multi_winner_dataset(args=args,
                                  output_frequency=output_frequency
                                  )

if __name__ == "__main__":
    make_dataset_from_cmd()
    # make_multi_winner_datasets()
