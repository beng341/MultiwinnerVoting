import pprint

from preflibtools.instances import OrdinalInstance
import os
from collections import Counter


def load_all_instances_in_folder(folder):
    """

    :param folder:
    :return:
    """

    filenames = next(os.walk(folder), (None, None, []))[2]  # [] if no file

    alternative_counts = []
    voter_counts = []

    for fname in filenames:
        instance = OrdinalInstance()
        instance.parse_file(os.path.join(folder, fname))
        alternative_counts.append(instance.num_alternatives)

        if instance.num_alternatives in [7]:
            voter_counts.append(instance.num_voters)

    alternative_counter = Counter(alternative_counts)
    voter_counter = Counter(voter_counts)

    print(f"Number of elections with X alternatives is:")
    pprint.pprint(alternative_counter)

    print("\n")

    print(f"Number of elections with X voters is:")
    pprint.pprint(voter_counter)


load_all_instances_in_folder(folder="preflib/soc")