import itertools
import os.path
import sys
import pandas as pd
from pref_voting.generate_profiles import generate_profile as gen_prof
from utils import data_utils as du
from utils import axiom_eval as ae
from abcvoting.preferences import Profile, Voter
from abcvoting import abcrules
import random

def create_profiles(args, num_winners, condorcet_only=False, **kwargs):
    """
    Given appropriate parameters create a dataframe containing one column with one preference profile per row.
    Each preference profile is saved as a string.
    :param args:
    :param kwargs:
    :return:
    """
    n_profiles = args["n_profiles"]
    prefs_per_profile = args["prefs_per_profile"]
    m = args["m"]
    pref_model = args["learned_pref_model"]

    profiles = []
    abc_profile = []
    pref_voting_profiles = []

    # num_rejects = 0
    # for _ in range(n_profiles):
    while len(profiles) < n_profiles:
        profile = generate_profile(n=prefs_per_profile, m=m, model=pref_model, **kwargs)
        rankings = profile.rankings

        exists_condorcet_winner = True
        if condorcet_only:
            cand_pairs = du.candidate_pairs_from_profiles(rankings)
            exists_condorcet_winner = ae.exists_condorcet_winner(du.generate_all_committees(len(rankings[0]), num_winners), cand_pairs)
        
        if exists_condorcet_winner:
            profiles.append(rankings)


        abcvoting_profile = Profile(num_cand=m)

        for rank in rankings:
            abcvoting_profile.add_voter(Voter(list(rank[:num_winners])))

        abc_profile.append(abcvoting_profile)
        pref_voting_profiles.append(profile)
    # columns = ["Profile"]
    # profiles_df = pd.DataFrame(profiles, columns=columns)
    return profiles, abc_profile, pref_voting_profiles

    # abcvoting_profile = Profile(num_cand=m)

    # for rank in rankings:
    #    abcvoting_profile.add_voter(Voter(list(rank[:num_winners])))

    # profiles.append(f"{rankings}")
    # abc_profile.append(abcvoting_profile)
    # pref_voting_profiles.append(profile)
    # columns = ["raw_profiles"]
    # profiles_df = pd.DataFrame(profiles, columns=columns)

    # return profiles_df, abc_profile, pref_voting_profiles


def generate_profile(n, m, model, **kwargs):
    """
    Generate a profile of the given model and size using pref-voting library.
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

    # Process each profile individually
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
        #"stratification__args__weight=0.5",
        "URN-R",
        "IC",
        #"IAC",
        "MALLOWS-RELPHI-R",
        "single_peaked_conitzer",
        # "single_peaked_walsh",
        # "euclidean__args__dimensions=2_space=uniform",
        # "euclidean__args__dimensions=3_space=uniform",
        # "euclidean__args__dimensions=2_space=ball",
        # "euclidean__args__dimensions=3_space=ball",
        # "euclidean__args__dimensions=2_space=gaussian",
        # "euclidean__args__dimensions=3_space=gaussian",
        # "euclidean__args__dimensions=2_space=sphere",
        # "euclidean__args__dimensions=3_space=sphere",
    ]
    profile_counts = [10000]  # size of dataset generated
    prefs_per_profile = [100]  # number of voters per profile
    candidate_sizes = [8]  # number of candidates in each profile
    num_winners = [3]

    for pref_model in pref_models:
        #make_one_multi_winner_dataset(random.randint(6, 10), random.randint(1000, 10000), random.randint(20, 100), pref_model, random.randint(2, 4), True, condorcet_only=False)
        make_one_multi_winner_dataset(6, 10000, 50, pref_model, 4, True, condorcet_only=False)

def make_one_multi_winner_dataset(m, n_profiles, ppp, pref_model, winners_size, train, condorcet_only=True, base_data_path="data"):
    """
    Extracted from make_multi_winner_datasets() to allow calling it from elsewhere
    :param m:
    :param n_profiles:
    :param ppp:
    :param pref_model:
    :param winners_size:
    :return:
    """

    for train in [True, False]:

        if train:
            type = "training"
        else:
            type = "testing"


        print(f"Making a {type} dataset with {n_profiles} profiles, {ppp} voters per profile, {m} candidates, and {winners_size} winners, using a {pref_model} distribution.")

        pref_model_shortname, kwargs = du.kwargs_from_pref_models(pref_model)
        args = {
            "n_profiles": n_profiles,
            "prefs_per_profile": ppp,
            "m": m,
            "learned_pref_model": pref_model_shortname,
        }
        print(sys.argv)
        if len(sys.argv) > 1:
            kw = dict(arg.split('=') for arg in sys.argv[1:])
            for k, v in kw.items():
                args[k] = eval(v)
        # profile_name = "impartial_culture"
        profiles, abc_profiles, pref_voting_profiles = create_profiles(args=args, num_winners=winners_size, condorcet_only=condorcet_only, **kwargs)
        # add various computed forms of profile data
        # df = generate_computed_data(df)

        profile_data = []
        # WE NEED TO PUT THE ABC PROFILES AND PREF VOTING PROFILES HERE AND FIND THEIR WINNERS
        for i, profile in enumerate(profiles):
            winners, min_violations = du.findWinners(profile, winners_size)
            abc_profile = abc_profiles[i]
            pref_voting_profile = pref_voting_profiles[i]

            rand_idx = random.randint(0, len(winners) - 1)

            toadd = {"Profile": profile,  "Winner": tuple(winners[rand_idx].tolist()), "Num_Violations": min_violations}
            profile_data.append(toadd)
        
        profiles_df = pd.DataFrame(profile_data)
        profiles_df = generate_computed_data(profiles_df)
        violations_count = profiles_df['Num_Violations'].sum()

        voting_rules = du.load_mw_voting_rules()
        for rule in voting_rules:
            print(rule)
            try:
                s = abcrules.get_rule(rule).longname
                prof = abc_profiles
            except AttributeError:
                try:
                    s = rule.name
                    prof = pref_voting_profiles
                except AttributeError:
                    print("Unknown rule")
                    return
            print(s)
            try:
                singlecomittee, _ = du.generate_winners(rule, prof, winners_size, m)
                profiles_df[f"{s} Winner"] = singlecomittee
            except Exception as ex:
                print(f"{s} broke everything")
                print(f"{ex}")
                return

        print("Total number of violations:", violations_count)
        print("Proportion of violations:", violations_count / n_profiles)

        """
                voting_rules = du.load_mw_voting_rules()
        
                # for name, rule in voting_rules:
                for rule in voting_rules:
                    try:
                        s = abcrules.get_rule(rule).longname
                        profiles = abc_profiles
                    except AttributeError:
                        try:
                            s = rule.name
                            profiles = pref_voting_profiles
                        except AttributeError:
                            print("Unknown rule")
                            return
        
                    print(f"Beginning to calculate winners & violations for {s} using {pref_model_shortname} preferences")
        
                    try:
                        singlecomittee, tiedcomittees = du.generate_winners(rule, profiles, winners_size, m)
                        df[f"{s}-single_winner"] = singlecomittee
                        df[f"{s}-tied_winners"] = tiedcomittees
                        df = df.copy()
                    except Exception as ex:
                        print(f"{s} broke everything")
                        print(f"{ex}")
                        return
                    
                    df[f"{s}-single_winner_majority_violation"] = df.apply(ae.eval_majority_axiom, axis=1, rule=s, tie=False)
                    df[f"{s}-tied_winners_majority_violations"] = df.apply(ae.eval_majority_axiom, axis=1, rule=s, tie=True)
                    df[f"{s}-single_winner_majority_loser_violations"] = df.apply(ae.eval_majority_loser_axiom, axis=1, rule=s, tie=False)
                    df[f"{s}-tied_winners_majority_loser_violations"] = df.apply(ae.eval_majority_loser_axiom, axis=1, rule=s, tie=True)
                    df[f"{s}-single_winner_condorcet_winner_violations"] = df.apply(ae.eval_condorcet_winner, axis=1, rule=s, tie=False)
                    df[f"{s}-tied_winners_condorcet_winner_violations"] = df.apply(ae.eval_condorcet_winner, axis=1, rule=s, tie=True)
                    df[f"{s}-single_winner_condorcet_loser_violations"] = df.apply(ae.eval_condorcet_loser, axis=1, rule=s, tie=False)
                    df[f"{s}-tied_winners_condorcet_loser_violations"] = df.apply(ae.eval_condorcet_loser, axis=1, rule=s, tie=True)
                
        
        
                print(f"Computed winners for {len(voting_rules)} voting rules.")
                """
        condorcet_tag = "condorcet_only-" if condorcet_only else ""
        if train:
            filename = (f"n_profiles={args['n_profiles']}-num_voters={args['prefs_per_profile']}"
                        f"-m={args['m']}-committee_size={winners_size}-pref_dist={pref_model}-{condorcet_tag}TRAIN.csv")
        else:
            filename = (f"n_profiles={args['n_profiles']}-num_voters={args['prefs_per_profile']}"
                        f"-m={args['m']}-committee_size={winners_size}-pref_dist={pref_model}-{condorcet_tag}TEST.csv")
        filepath = os.path.join(base_data_path, filename)
        profiles_df.to_csv(filepath, index=False)
        print(f"Saving to: {filepath}")


if __name__ == "__main__":
    make_multi_winner_datasets()
