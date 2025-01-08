from preflibtools.instances import OrdinalInstance
import os
from pathlib import Path
import utils.data_utils as du
import pref_voting
from abcvoting import abcrules
import pprint
import pandas as pd
from network_ops.generate_data import generate_computed_data


def clean_election_profile(profile):
    profile = [[cand[0] for cand in order] for order in profile]

    # Find the minimum valued alternative across all order
    min_value = min(min(order) for order in profile)

    # Ensure orders all begin with alternative 0
    profile = [[cand - min_value for cand in order] for order in profile]

    return profile

def load_matching_profiles(folder, all_num_alternatives, max_num_voters=None, max_number=None):
    """
    Load all preflib instances in the given file and return the instances with the matching number of alternatives.
    Also filter on the number of winners k to make sure that 1<=k<m.
    :param folder:
    :param all_num_alternatives:
    :param max_num_voters: If set, limits the number of voters to reduce time needed for axiom evaluation.
    :param max_number: Load all matching instances if None or load at most the value given (set lower for quick testing)
    :return:
    """

    folder = Path(folder)

    instances = []
    profiles = []
    for file_path in folder.glob("*.soc"):
        if max_number and len(instances) >= max_number:
            break
        try:
            # Create new election instance for each file
            instance = OrdinalInstance()
            instance.parse_file(str(file_path))

            if max_num_voters is not None and max_num_voters < instance.num_voters:
                continue

            # Keep only the elections that have the right number of alternatives
            if instance.num_alternatives in all_num_alternatives:
                instances.append(instance)
                profiles.append(instance.full_profile())

                print(f"Adding election with {instance.num_voters} voters.")

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    return instances


def make_data_file_from_profiles(instances, output_folder, num_winners, num_alternatives, axioms="all"):
    """
    Make a data file suitable for training or evaluation, based on the given profiles.
    :param instances: PreflibInstance object
    :param output_folder
    :param num_winners
    :param num_alternatives
    :param axioms
    :return:
    """
    output_frequency = 20

    filename = f"preflib-n_profiles={len(instances)}-num_alternatives={num_alternatives}-num_winners={num_winners}-pref_dist=preflib-axioms={axioms}.csv"

    profile_dict = {"Profile": [], "n_voters": [],
                    "min_violations-committee": [], "min_viols-num_committees": [], "min_violations": [],
                    "max_violations-committee": [], "max_viols-num_committees": [], "max_violations": [], }

    for idx, instance in enumerate(instances):

        # num_alternatives = instance.num_alternatives
        profile = instance.full_profile()
        profile = clean_election_profile(profile)
        abc_profile = du.abc_profile_from_rankings(m=num_alternatives, k=num_winners, rankings=profile)
        pref_voting_profile = pref_voting.profiles.Profile(rankings=profile)

        # Track best/worst committees during training AND testing so we can see how close each rule is to each side
        minv_committee, minv, maxv_committee, maxv = du.find_winners(profile, num_winners,
                                                                     axioms_to_evaluate=[axioms])

        if len(minv_committee) > 1:
            # ensure lexicographic tie-breaking among tied winners
            # unclear if this is strictly better than random tie-breaking
            minv_committee.sort(key=lambda x: tuple(x))

        if len(maxv_committee) > 1:
            # ensure lexicographic tie-breaking among tied winners
            # unclear if this is strictly better than random tie-breaking
            maxv_committee.sort(key=lambda x: tuple(x))

        profile_dict["Profile"].append(profile)
        profile_dict["n_voters"].append(len(profile))
        profile_dict["min_violations-committee"].append(tuple(minv_committee[0]))
        profile_dict["min_viols-num_committees"].append(len(minv_committee))
        profile_dict["min_violations"].append(minv)
        profile_dict["max_violations-committee"].append(tuple(maxv_committee[0]))
        profile_dict["max_viols-num_committees"].append(len(maxv_committee))
        profile_dict["max_violations"].append(maxv)

        # if type == "TEST":
        #     # Only calculate winners of each individual voting rule when making test data
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
                single_winner, _ = du.generate_winners(rule, [prof], num_winners, num_alternatives, abc_rule=abc_rule)

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
            filepath = os.path.join(output_folder, filename)
            # profiles_df.to_csv(filepath, index=False)
            profiles_df.to_csv(filepath, index=False)
            print(f"Saving partial dataset to: {filepath}")

    # Output the complete dataset for good measure, likely redundant
    profiles_df = pd.DataFrame.from_dict(profile_dict)
    profiles_df = generate_computed_data(profiles_df)
    filepath = os.path.join(output_folder, filename)
    # profiles_df.to_csv(filepath, index=False)
    profiles_df.to_csv(filepath, index=False)
    print(f"Saving complete dataset to: {filepath}")


def make_data_file(num_alternatives=5, num_winners=2):
    instances = load_matching_profiles(
        folder="preflib/soc",
        all_num_alternatives=[num_alternatives],
        max_number=5
    )
    print(f"There are {len(instances)} elections.")

    make_data_file_from_profiles(instances=instances,
                                 output_folder="preflib",
                                 num_winners=num_winners,
                                 num_alternatives=num_alternatives,
                                 axioms="all")


if __name__ == "__main__":

    # alternatives = list(range(5, 8))
    for m in [5, 6, 7]:
        for k in range(3, m):
            make_data_file(num_alternatives=m, num_winners=k)
