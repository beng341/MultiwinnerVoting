import math
from collections import defaultdict
import itertools


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
        if kapproval_sorted[k][1] <= len(profile)/2:
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

    if fm_count > len(profile)/2:
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


if __name__ == "__main__":
    prefs = [(3, 4, 0, 2, 1),
             (1, 3, 4, 0, 2),
             (4, 1, 0, 2, 3),
             (3, 4, 0, 2, 1),
             (3, 4, 0, 2, 1),
             (1, 2, 4, 0, 3),
             (1, 4, 3, 0, 2),
             (2, 0, 4, 1, 3),
             (1, 2, 3, 4, 0),
             (1, 4, 3, 2, 0),
             (1, 3, 4, 0, 2),
             (2, 0, 4, 1, 3),
             (0, 3, 2, 1, 4),
             (2, 4, 0, 3, 1),
             (2, 4, 3, 1, 0),
             (1, 4, 2, 3, 0),
             (4, 1, 2, 3, 0),
             (2, 4, 3, 1, 0),
             (1, 4, 3, 0, 2),
             (4, 3, 2, 0, 1),
             (1, 2, 3, 4, 0),
             (2, 4, 3, 1, 0),
             (4, 3, 2, 0, 1),
             (1, 2, 4, 0, 3),
             (3, 4, 0, 2, 1),
             (4, 3, 2, 0, 1),
             (4, 3, 2, 0, 1),
             (1, 2, 3, 4, 0),
             (1, 3, 4, 0, 2),
             (1, 2, 3, 4, 0),
             (1, 2, 3, 4, 0),
             (1, 3, 4, 0, 2),
             (0, 1, 4, 3, 2),
             (4, 3, 2, 0, 1),
             (3, 4, 0, 2, 1),
             (1, 4, 3, 2, 0),
             (4, 3, 2, 0, 1),
             (2, 0, 4, 1, 3),
             (2, 3, 0, 1, 4),
             (1, 3, 4, 0, 2),
             (2, 0, 4, 1, 3),
             (0, 2, 3, 1, 4),
             (2, 0, 4, 1, 3),
             (4, 0, 3, 2, 1),
             (1, 3, 4, 0, 2),
             (2, 4, 0, 3, 1),
             (0, 1, 4, 3, 2),
             (1, 3, 4, 0, 2),
             (3, 4, 0, 2, 1),
             (4, 1, 2, 3, 0)]
    # prefs = [(2, 0, 4, 3, 1),
    #          (4, 3, 0, 2, 1),
    #          (4, 3, 0, 1, 2),
    #          (4, 3, 0, 2, 1),
    #          (0, 2, 3, 1, 4),
    #          (3, 1, 0, 2, 4),
    #          (0, 2, 3, 1, 4),
    #          (1, 2, 4, 3, 0),
    #          (4, 3, 0, 2, 1),
    #          (1, 2, 4, 3, 0),
    #          (0, 2, 3, 1, 4),
    #          (3, 1, 2, 4, 0),
    #          (1, 2, 4, 3, 0),
    #          (1, 2, 4, 3, 0),
    #          (0, 2, 3, 1, 4),
    #          (0, 2, 3, 1, 4),
    #          (2, 0, 3, 4, 1),
    #          (4, 3, 0, 2, 1),
    #          (1, 2, 4, 3, 0),
    #          (4, 3, 0, 1, 2),
    #          (0, 2, 3, 1, 4),
    #          (3, 1, 0, 2, 4),
    #          (2, 0, 4, 3, 1),
    #          (3, 4, 2, 0, 1),
    #          (1, 2, 4, 3, 0),
    #          (0, 2, 3, 1, 4),
    #          (3, 1, 2, 4, 0),
    #          (1, 2, 4, 3, 0),
    #          (0, 2, 3, 1, 4),
    #          (1, 2, 4, 3, 0),
    #          (4, 3, 0, 1, 2),
    #          (2, 0, 3, 4, 1),
    #          (1, 2, 4, 3, 0),
    #          (3, 1, 2, 4, 0),
    #          (2, 0, 3, 4, 1),
    #          (1, 2, 4, 3, 0),
    #          (2, 0, 3, 4, 1),
    #          (0, 2, 3, 1, 4),
    #          (2, 0, 4, 3, 1),
    #          (2, 0, 3, 4, 1),
    #          (3, 4, 2, 0, 1),
    #          (0, 2, 3, 1, 4),
    #          (0, 2, 3, 1, 4),
    #          (1, 2, 4, 3, 0),
    #          (0, 2, 3, 1, 4),
    #          (0, 2, 3, 1, 4),
    #          (3, 1, 2, 4, 0),
    #          (2, 0, 3, 4, 1),
    #          (2, 0, 3, 4, 1),
    #          (2, 0, 3, 4, 1)]
    prefs = [
        (0, 1, 2, 3, 4),
        (0, 1, 2, 4, 3),
        (2, 1, 0, 3, 4),
        (3, 1, 2, 0, 4),
        (4, 1, 2, 3, 0),
    ]

    candpairs = find_candpairs(prefs, 5)
    req = fixed_majority_required_winner(2, 5, candpairs, prefs)
    print(f"Required winners: {req}")
    count_preferences_in_positions(preference_orders=prefs)

    # score = calculate_borda_score(prefs)
    # print(score)