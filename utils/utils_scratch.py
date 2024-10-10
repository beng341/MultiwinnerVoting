import math
from collections import defaultdict
import itertools
from utils import data_utils as du
import pref_voting.profiles
import random


def calculate_borda_score(preference_orders):
    # Initialize a dictionary to store the scores
    borda_scores = defaultdict(int)

    # Number of alternatives
    num_alternatives = len(preference_orders[0])

    # Loop through each preference order
    for order in preference_orders:
        for rank, alternative in enumerate(order):
            # Borda score: num_alternatives - rank - 1
            borda_scores[alternative] += num_alternatives - rank - 1

    # Convert defaultdict to a list of tuples sorted by alternative
    sorted_scores = sorted(borda_scores.items())

    # Return just the scores, preserving the order of alternatives
    return [score for alt, score in sorted_scores]


def count_preferences_in_positions(preference_orders):
    # Initialize a dictionary where each key is a position and the value is another defaultdict counting occurrences
    position_counts = defaultdict(lambda: defaultdict(int))

    # Iterate through each preference order
    for order in preference_orders:
        # Iterate through the position and the alternative ranked in that position
        for position, alternative in enumerate(order):
            position_counts[position][alternative] += 1

    # Print the results
    for position, counts in position_counts.items():
        print(f"Position {position}:")
        for alternative, count in sorted(counts.items()):
            print(f"{alternative} was ranked in position {position} {count} times")
        print()  # For readability


def fixed_majority_required_winner(n_winners, n_alternatives, candidate_pairs, profile):
    """
    A committee satisfies fixed majority axiom if when there is some k-set of alternatives W such that there exists
    some set of voters V' with |V'| > |V|/2 where every member of V' ranks every member of W above every non-member
    of W. That is, some majority-size fixed set of voters ranks agrees that some k-set should be the committee.
    :param n_voters:
    :param n_winners:
    :param n_alternatives:
    :param candidate_pairs:
    :return:
    """
    # Check if any set exists where each member is ranked above the non-members by a majority of voters
    # That is, check all k-sets of candidates and see if any consistently beat all non-members.

    # Count the number of points for each alternative in top k positions
    # e.g. Get the k-approval score of each alternative
    kapproval_score = [0] * n_alternatives
    for order in profile:
        for k in range(n_winners):
            kapproval_score[order[k]] += 1

    kapproval_sorted = sorted(zip(range(len(kapproval_score)), kapproval_score), key=lambda x: x[1], reverse=True)
    fm_winner_might_exist = True
    for k in range(n_winners):
        if kapproval_sorted[k][1] <= len(profile) / 2:
            fm_winner_might_exist = False

    # If the k-approval winners have score above majority size, there might be an FM winner.
    # Check if a majority of voters put those k alternatives in first place

    fm_winner = None
    fm_count = 0
    if fm_winner_might_exist:
        fm_winner = [kapproval_sorted[idx][0] for idx in range(n_winners)]

        for prof in profile:
            if set(prof[:n_winners]) == set(fm_winner):
                fm_count += 1

    if fm_count > len(profile) / 2:
        pass
    else:
        fm_winner = None

    return fm_winner

    #
    #
    #
    # candidate_pairs = [candidate_pairs[i:i + n_alternatives] for i in range(0, len(candidate_pairs), n_alternatives)]
    #
    # all_candidates = set(range(n_alternatives))
    #
    # for W in itertools.combinations(range(n_alternatives), n_winners):
    #     W = list(W)
    #     losers = all_candidates - set(W)
    #     # check if all members of W are preferred by a majority to each non-member
    #     keep_searching_this_set = True
    #     for winner in W:
    #         for loser in losers:
    #             # if candidate_pairs[winner * n_alternatives + loser] <= candidate_pairs[loser * n_alternatives + winner]:
    #             if candidate_pairs[winner][loser] < candidate_pairs[loser][winner]:
    #                 keep_searching_this_set = False
    #                 break
    #         if not keep_searching_this_set:
    #             break
    #     if not keep_searching_this_set:
    #         continue
    #
    #     # if we reach this point, we have compared every winner to every loser and all winners have a majority win
    #     required_winning_committee = W
    #     break
    #
    # return required_winning_committee


def eval_fixed_majority_axiom(committee, required_winning_committee):
    """
    Return 1 if axiom is violated and 0 otherwise.
    :param committee:
    :param required_winning_committee:
    :return:
    """
    if required_winning_committee is None:
        return 0
    else:
        all_required_winners_are_winning = True
        for rw in required_winning_committee:
            if committee[rw] != 1:
                all_required_winners_are_winning = False
        return int(not all_required_winners_are_winning)


def find_candpairs(ballots, num_candidates):
    candidate_pairs = [0] * num_candidates ** 2
    for ballot in ballots:
        for i in range(num_candidates):
            for j in range(num_candidates):
                if i != j:
                    # Check if candidate i is ranked higher than candidate j
                    if ballot.index(i) < ballot.index(j):
                        candidate_pairs[i * num_candidates + j] += 1
    return candidate_pairs






def compare_stv():
    prefs = [(4, 3, 1, 0, 5, 2),
             (3, 1, 5, 4, 2, 0),
             (5, 0, 1, 2, 4, 3),
             (0, 5, 1, 2, 3, 4),
             (5, 0, 3, 4, 1, 2),
             (2, 3, 5, 0, 1, 4),
             (5, 0, 3, 4, 1, 2),
             (5, 0, 1, 2, 4, 3),
             (0, 3, 4, 2, 1, 5),
             (5, 0, 1, 2, 4, 3),
             (1, 3, 0, 2, 5, 4),
             (1, 4, 3, 0, 5, 2),
             (5, 1, 3, 4, 0, 2),
             (1, 3, 5, 4, 0, 2),
             (0, 3, 4, 2, 1, 5),
             (5, 1, 2, 0, 4, 3),
             (4, 3, 1, 0, 5, 2),
             (5, 1, 2, 0, 4, 3),
             (5, 0, 1, 2, 4, 3),
             (4, 5, 2, 1, 3, 0),
             (4, 3, 1, 0, 5, 2),
             (0, 4, 3, 5, 2, 1),
             (5, 1, 2, 0, 4, 3),
             (4, 5, 2, 1, 3, 0),
             (1, 5, 3, 0, 2, 4),
             (1, 4, 5, 3, 2, 0),
             (3, 0, 4, 1, 5, 2),
             (0, 5, 1, 2, 3, 4),
             (5, 0, 1, 2, 4, 3),
             (5, 0, 1, 2, 4, 3),
             (3, 2, 4, 5, 1, 0),
             (4, 3, 1, 5, 0, 2),
             (4, 3, 1, 0, 5, 2),
             (5, 0, 3, 4, 1, 2),
             (5, 0, 1, 2, 4, 3),
             (5, 0, 1, 2, 4, 3),
             (5, 0, 3, 4, 1, 2),
             (5, 1, 2, 3, 0, 4),
             (1, 4, 3, 0, 5, 2),
             (5, 0, 3, 4, 1, 2),
             (5, 0, 1, 2, 4, 3),
             (5, 0, 1, 2, 4, 3),
             (3, 0, 4, 1, 5, 2),
             (5, 0, 1, 2, 4, 3),
             (0, 5, 1, 2, 3, 4),
             (3, 0, 4, 1, 5, 2),
             (5, 0, 3, 4, 1, 2),
             (0, 4, 3, 5, 2, 1),
             (0, 5, 1, 2, 3, 4),
             (5, 0, 3, 4, 1, 2)
             ]

    generate_random_preferences = lambda l, m: [tuple(random.sample(range(m), m)) for _ in range(l)]

    for _ in range(100):
        prefs = generate_random_preferences(5, 5)
        num_winners = 2

        import utils.voting_utils as vut
        stv_broken = vut.single_transferable_vote
        stv_chatgpt = vut.stv

        pv_profile = pref_voting.profiles.Profile(rankings=prefs)

        stv_chatgpt_winner = stv_chatgpt(pv_profile, k=num_winners)
        print(f"Winning committee from chatgpt: {stv_chatgpt_winner}")

        stv_broken_winner = stv_broken(pv_profile, k=num_winners)
        print(f"Winning committee from pyrankvote: {stv_broken_winner}")

        if set(stv_broken_winner) != set(stv_chatgpt_winner):
            pass
            stv_chatgpt_winner = stv_chatgpt(pv_profile, k=num_winners)
            pass

        print("-------------------")


def make_complete_networks_csv():
    import csv
    import re
    from collections import defaultdict

    # Input and output file paths
    input_file = 'complete_networks.txt'
    output_csv = 'output.csv'
    output_txt = 'incomplete_network_sets.txt'

    # Regular expression to extract M, K, DIST, and IDX from filenames
    pattern = r'm=(\d+)-num_winners=(\d+)-pref_dist=(.+?)-axioms=.*-idx=(\d+)-.pt'

    # Dictionary to store sets of IDX for each (M, K, DIST) combination
    idx_dict = defaultdict(set)

    # Read the input file and process each line
    with open(input_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line.strip())
            if match:
                M, K, DIST, IDX = match.groups()
                key = (M, K, DIST)
                idx_dict[key].add(IDX)

    # Write the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['M', 'K', 'DIST', 'num_unique_IDX'])

        for (M, K, DIST), idx_set in idx_dict.items():
            writer.writerow([M, K, DIST, len(idx_set)])

    # Write the txt file for combinations with less than 20 unique IDX values
    with open(output_txt, 'w') as txtfile:
        for (M, K, DIST), idx_set in idx_dict.items():
            if len(idx_set) < 20:
                filename = f"sbatch cc_jobs/train_eval_jobs/cc_job_train_eval_n_profiles=25000_n_voters=50_n_alternative={M}_n_winners={K}_pref_dist={DIST}_axioms=all.sh"
                txtfile.write(filename + '\n')


if __name__ == "__main__":
    make_complete_networks_csv()
    exit()
    # compare_stv()
    # prefs = [(4, 3, 1, 0, 5, 2),
    #          (3, 1, 5, 4, 2, 0),
    #          (5, 0, 1, 2, 4, 3),
    #          (0, 5, 1, 2, 3, 4),
    #          (5, 0, 3, 4, 1, 2),
    #          (2, 3, 5, 0, 1, 4),
    #          (5, 0, 3, 4, 1, 2),
    #          (5, 0, 1, 2, 4, 3),
    #          (0, 3, 4, 2, 1, 5),
    #          (5, 0, 1, 2, 4, 3),
    #          (1, 3, 0, 2, 5, 4),
    #          (1, 4, 3, 0, 5, 2),
    #          (5, 1, 3, 4, 0, 2),
    #          (1, 3, 5, 4, 0, 2),
    #          (0, 3, 4, 2, 1, 5),
    #          (5, 1, 2, 0, 4, 3),
    #          (4, 3, 1, 0, 5, 2),
    #          (5, 1, 2, 0, 4, 3),
    #          (5, 0, 1, 2, 4, 3),
    #          (4, 5, 2, 1, 3, 0),
    #          (4, 3, 1, 0, 5, 2),
    #          (0, 4, 3, 5, 2, 1),
    #          (5, 1, 2, 0, 4, 3),
    #          (4, 5, 2, 1, 3, 0),
    #          (1, 5, 3, 0, 2, 4),
    #          (1, 4, 5, 3, 2, 0),
    #          (3, 0, 4, 1, 5, 2),
    #          (0, 5, 1, 2, 3, 4),
    #          (5, 0, 1, 2, 4, 3),
    #          (5, 0, 1, 2, 4, 3),
    #          (3, 2, 4, 5, 1, 0),
    #          (4, 3, 1, 5, 0, 2),
    #          (4, 3, 1, 0, 5, 2),
    #          (5, 0, 3, 4, 1, 2),
    #          (5, 0, 1, 2, 4, 3),
    #          (5, 0, 1, 2, 4, 3),
    #          (5, 0, 3, 4, 1, 2),
    #          (5, 1, 2, 3, 0, 4),
    #          (1, 4, 3, 0, 5, 2),
    #          (5, 0, 3, 4, 1, 2),
    #          (5, 0, 1, 2, 4, 3),
    #          (5, 0, 1, 2, 4, 3),
    #          (3, 0, 4, 1, 5, 2),
    #          (5, 0, 1, 2, 4, 3),
    #          (0, 5, 1, 2, 3, 4),
    #          (3, 0, 4, 1, 5, 2),
    #          (5, 0, 3, 4, 1, 2),
    #          (0, 4, 3, 5, 2, 1),
    #          (0, 5, 1, 2, 3, 4),
    #          (5, 0, 3, 4, 1, 2)]
    # candpairs = find_candpairs(prefs, 6)
    # req = fixed_majority_required_winner(3, 6, candpairs, prefs)
    # print(f"Required winners: {req}")
    # count_preferences_in_positions(preference_orders=prefs)
    #
    # score = calculate_borda_score(prefs)
    # print(score)
    #
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

    m_all = [5, 6, 7]
    k_all = [1, 2, 3, 4, 5, 6]
    for m, k in itertools.product(m_all, k_all):

        if k >= m:
            continue

        print(f"Making mixed distribution with m={m} and k={k}.")
        du.generate_mixed_distribution(distributions=all_pref_models[:-1],
                                       total_size=25000,
                                       n=50,
                                       m=m,
                                       num_winners=k,
                                       axioms="all",
                                       data_folder="/scratch/b8armstr/data")
