import itertools
import sys
import pandas as pd
from pref_voting.generate_profiles import generate_profile as gen_prof
from pref_voting.c1_methods import condorcet
from utils import data_utils as du


def create_profiles(args, **kwargs):
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
    include_condorcet_profiles = args["include_condorcet_profiles"]

    profiles = []
    raw_profiles = []

    num_rejects = 0
    # for _ in range(n_profiles):
    while len(profiles) < n_profiles:
        profile = generate_profile(n=prefs_per_profile, m=m, model=pref_model, **kwargs)
        if not include_condorcet_profiles:
            # do not include profiles that have condorcet winners
            if len(condorcet(profile)) == 1:  # Only one condorcet winner so we should reject the profile
                num_rejects += 1
                if num_rejects % 100 == 0:
                    print(f"Num rejected profiles is {num_rejects}")
                continue
        rankings = profile.rankings
        profiles.append(f"{rankings}")
        raw_profiles.append(profile)
    columns = ["raw_profiles"]
    profiles_df = pd.DataFrame(profiles, columns=columns)

    return profiles_df, raw_profiles


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
    profiles = [eval(elem) for elem in df["raw_profiles"].tolist()]
    du.compute_features_from_profiles(profiles, df)

    return df


def kwargs_from_pref_models(pref_model):
    model_string = pref_model
    arg_dict = {}
    if "__args__" in pref_model:
        arg_dict = {}
        model_string = pref_model[:pref_model.index("__args__")]
        arg_string = pref_model[pref_model.index("__args__") + len("__args__"):]
        # assume args are split by a single underscore
        args = arg_string.split("_")
        for arg in args:
            pair = arg.split("=")
            key, value = pair[0], pair[1]
            try:
                arg_dict[key] = eval(value)
            except NameError:
                # simplest way to see if the argument should be a string or not
                arg_dict[key] = value
    return model_string, arg_dict


def make_single_winner_datasets():
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
        "MALLOWS-RELPHI-R",
        # "single_peaked_conitzer",
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
    profile_counts = [1000]  # size of dataset generated
    prefs_per_profile = [20]  # number of voters per profile
    candidate_sizes = [5]  # number of candidates in each profile
    num_winners = [3]

    for n_profiles, ppp, m, pref_model, winners_size in itertools.product(profile_counts, prefs_per_profile,
                                                                          candidate_sizes, pref_models, num_winners):

        pref_model_shortname, kwargs = kwargs_from_pref_models(pref_model)

        args = {
            "n_profiles": n_profiles,
            "prefs_per_profile": ppp,
            "m": m,
            "learned_pref_model": pref_model_shortname,
            "include_condorcet_profiles": True  # True iff profiles with condorcet winners should be included
        }
        print(sys.argv)
        if len(sys.argv) > 1:
            kw = dict(arg.split('=') for arg in sys.argv[1:])
            for k, v in kw.items():
                args[k] = eval(v)

        # profile_name = "impartial_culture"
        df, profiles = create_profiles(args=args, **kwargs)

        voting_rules = du.load_voting_rules()

        # for name, rule in voting_rules:
        for rule in voting_rules:
            s = rule.name
            print(f"Beginning to calculate winners for {s} using {pref_model_shortname} preferences")
            try:
                single_winners, tied_winners = du.generate_winners(rule, profiles)
                df[f"{s}-single_winner"] = single_winners
                df[f"{s}-tied_winners"] = tied_winners
                df = df.copy()
            except Exception as ex:
                print(f"{s} broke everything")
                print(f"{ex}")
        # print(f"Computed winners for {len(voting_rules)} voting rules.")

        # add various computed forms of profile data
        df = generate_computed_data(df)

        filename = (f"data/n_profiles={args['n_profiles']}-num_voters={args['prefs_per_profile']}"
                    f"-m={args['m']}-pref_dist={pref_model}-include_condorcet={args['include_condorcet_profiles']}.csv")
        df.to_csv(filename, index=False)
        print(f"Saving to: {filename}")


if __name__ == "__main__":
    make_single_winner_datasets()
