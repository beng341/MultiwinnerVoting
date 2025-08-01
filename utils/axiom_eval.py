import itertools
import math
import utils.data_utils as du
import numpy as np

all_axioms = [
    "dummett",
    # "consensus",
    "fixed_majority",
    "majority_winner",
    "majority_loser",
    "condorcet_winner",
    "condorcet_loser",
    "solid_coalition",
    "strong_unanimity",
    "local_stability",
    "strong_pareto",
    "jr",
    "ejr",
    "core"
]

reduced_axioms = [
    'local_stability',
    'dummetts_condition',
    'condorcet_winner',
    'strong_pareto_efficiency',
    'core',
    'majority_loser'
]


def eval_majority_axiom(n_voters, committee, rank_choice):
    """
    Evaluate the majority axiom for a given committee and profiles.
    If the majority of agents put a candidate in the first position, then that 
    candidate should be in the winning committee
    :param profiles: Profile of voter preferences.
    :param committee: A committee to valuate.
    :param rank_choice: The rank choice matrix for the profiles.
    :return: 0 if the majority axiom is not violated and 1 if it is.
    """
    maj_threshold = n_voters // 2
    num_candidates = len(committee)
    top_rated = [rank_choice[i * num_candidates] for i in range(num_candidates)]

    for candidate in range(num_candidates):
        if top_rated[candidate] > maj_threshold and committee[candidate] == 0:
            return 1
    return 0


def eval_majority_loser_axiom(n_voters, committee, rank_choice):
    """
    Evaluate the majority loser axiom for a given committee and profiles.
    If the majority of agents put a candidate in the last position, then that 
    candidate should not be in the winning committee
    :param profiles: Profile of voter preferences.
    :param committee: A committee to valuate.
    :return: 0 if the majority loser axiom is not violated and 1 if it is.
    """
    maj_threshold = n_voters // 2
    num_candidates = len(committee)
    bottom_rated = [rank_choice[i * num_candidates + num_candidates - 1] for i in range(num_candidates)]

    for candidate in range(num_candidates):
        if bottom_rated[candidate] > maj_threshold and committee[candidate] == 1:
            return 1
    return 0


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


def fixed_majority_required_winner_old(n_winners, n_alternatives, candidate_pairs):
    """
    A committee satisfies fixed majority axiom if when there is some k-set W such that a majority rank every element
    of W above non-members then W is the unique winning set.
    :param n_voters:
    :param n_winners:
    :param n_alternatives:
    :param candidate_pairs:
    :return:
    """
    # Check if any set exists where each member is ranked above the non-members by a majority of voters
    # That is, check all k-sets of candidates and see if any consistently beat all non-members.

    all_candidates = set(range(n_alternatives))

    required_winning_committee = None
    for W in itertools.combinations(range(n_alternatives), n_winners):
        W = list(W)
        losers = all_candidates - set(W)
        # check if all members of W are preferred by a majority to each non-member
        keep_searching_this_set = True
        for winner in W:
            for loser in losers:
                if candidate_pairs[winner * n_alternatives + loser] <= candidate_pairs[loser * n_alternatives + winner]:
                    # if candidate_pairs[winner][loser] < candidate_pairs[loser][winner]:
                    keep_searching_this_set = False
                    break
            if not keep_searching_this_set:
                break
        if not keep_searching_this_set:
            continue

        # if we reach this point, we have compared every winner to every loser and all winners have a majority win
        required_winning_committee = W
        break

    return required_winning_committee


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


def exists_condorcet_winner(all_committees, cand_pairs):
    print("DEPRECATED. BETTER TO USE FAST VERSION.")
    for committee in all_committees:
        if eval_condorcet_winner(committee, cand_pairs) == 0:
            return True

    return False


def exists_condorcet_winner_fast(n_alternatives, n_winners, cand_pairs, n_voters):
    """
    Determine whether there exists a condorcet winner.
    Works by finding possible winners then testing all possible committees formed from those winners.

    :param n_alternatives:
    :param n_winners:
    :param cand_pairs:
    :param n_voters:
    :return:
    """

    # find candidates ranked above m-k others by at least half of voters
    # AKA - find candidates ranked in the top k by at least half of voters
    # TODO: Only pretty certain this is the correct precondition.
    # possible_condorcet_set_members = []
    cp = np.array(cand_pairs).reshape(n_alternatives, n_alternatives)
    # count how often each candidate is preferred by a majority of voters
    maj_win_counts = np.sum(cp >= n_voters / 2, axis=1)
    possible_condorcet_set_members = np.where(maj_win_counts >= (n_alternatives - n_winners))[0].tolist()

    if len(possible_condorcet_set_members) < n_winners:
        return False

    for committee in itertools.combinations(possible_condorcet_set_members, n_winners):
        khot = du.khot_committee_from_winners(committee, num_alternatives=n_alternatives)
        if eval_condorcet_winner(khot, cand_pairs) == 0:
            return True

    # for committee in all_committees:
    #     if eval_condorcet_winner(committee, cand_pairs) == 0:
    #         return True

    return False


def eval_condorcet_winner(committee, cand_pairs):
    """
    Evaluate the Condorcet winner axiom for a given committee and profiles.
    Condorcet winner in this instance is:
    A committee is a Condorcet committee if each candidate in it is preferred, 
    by a majority of voters, to each candidate outside it
    :param committee: A committee to valuate.
    :param cand_pairs: The candidate pairs matrix for the profiles.
    :return: 0 if the committee is a condorcet winner; return 1 if the committee is not a condorcet winner
    """

    in_committee = [i for i, x in enumerate(committee) if x == 1]
    not_in_committee = [i for i, x in enumerate(committee) if x == 0]
    num_candidates = len(committee)

    for c in in_committee:
        for d in not_in_committee:
            if cand_pairs[c * num_candidates + d] < cand_pairs[d * num_candidates + c]:
                # there is a candidate in committee that is not preferred by a majority to a candidate outside committee
                return 1
    return 0


def eval_condorcet_loser(committee, cand_pairs):
    """
    Evaluate the Condorcet loser axiom for a given committee and profiles.
    Condorcet loser in this instance is the opposite of the Condorcet winner above:
    A condorcet losing set is a set that, for each member c in the set, all non-members d
    are preferred to c by a majority
    :param committee: A committee to valuate.
    :param cand_pairs: The candidate pairs matrix for the profiles.
    :return: 0 if the axiom is not violated and 1 if it is.
    """

    in_committee = [i for i, x in enumerate(committee) if x == 1]
    not_in_committee = [i for i, x in enumerate(committee) if x == 0]
    num_candidates = len(committee)

    for c in in_committee:
        for d in not_in_committee:
            if cand_pairs[c * num_candidates + d] >= cand_pairs[d * num_candidates + c]:
                return 0
    return 1


def eval_dummetts_condition(committee, num_voters, num_winners, profile, required_winners):
    """
    Evaluate Dummett's condition for a given committee and profiles.
    Dummett's condition states that if for some l <= n_winners, there
    is a group of l * num_voters / n_winners that all rank the same l candidates
    on top, then those l candidates should be in the winning committee.
    This requirement tries to capture the idea of proportional representation,
    or proportionality for solid coalitions.
    :param committee: A committee to evaluate.
    :param num_voters: The number of voters in the profiles.
    :param num_winners: The number of winners in the committee.
    :param profile: Profile of voter preferences.
    """
    all_required_winners_are_winning = True
    for rw in required_winners:
        if committee[rw] != 1:
            all_required_winners_are_winning = False
    # return 1 iff required winners aren't winning (axiom is violated), 0 if required winners are winning (not violated)
    return int(not all_required_winners_are_winning)


def find_dummett_winners(num_voters, num_winners, profile):
    """
    Find the committees that are able to satisfy dummett's condition.
    Dummett's condition states that if for some l <= n_winners, there
    is a group of l * num_voters / n_winners that all rank the same l candidates
    on top, then those l candidates should be in the winning committee.
    This requirement tries to capture the idea of proportional representation, or proportionality for solid coalitions.
    :param num_voters:
    :param num_winners:
    :param profile:
    :return:
    """
    print("DEPRECATED. BETTER TO USE FAST VERSION.")
    required_winners = set()
    for l in range(1, num_winners + 1):
        # if this many voters rank the same l candidates in first l positions, those candidates must win
        # TODO: Should this be rounded up, rather than down?
        # e.g. 50 voters, 4 winners. Should threshold be 12 or 13? If 12, 4 voters could pass with l=1 and a 5th could pass with l=2 (right?)
        threshold = int(l * num_voters / num_winners)

        # look at all size l sets of candidates
        # check if at least threshold voters rank that set in the top
        # record all voters that appear in a winning set (can there be more than one winning set?)

        for candidates in itertools.combinations(range(len(profile[0])), l):
            cset = set(candidates)
            voter_count = 0
            for ballot in profile:
                if set(ballot[:l]) == cset:
                    # top l candidates in this ballot are same as the current set of candidates
                    voter_count += 1
                if voter_count >= threshold:
                    winners = ballot[:l]
                    required_winners |= set(winners)
                    break
    return required_winners


def find_dummett_winners_fast(num_voters, num_winners, num_alternatives, profile, rank_matrix):
    """
    Find the committees that are able to satisfy dummett's condition.
    Dummett's condition states that if for some l <= n_winners, there is a group of l * num_voters / n_winners
    that all rank the same l candidates on top, then those l candidates should be in the winning committee.
    NOTE: This method works faster by identifying the smaller set of candidates which *might* be required winners.
    A candidate might be a required winner if ln/k voters rank it in the top l positions
    :param num_voters:
    :param num_winners:
    :param profile:
    :return:
    """
    rm_square = np.array(rank_matrix).reshape(num_alternatives, num_alternatives)
    required_winners = set()
    for l in range(1, num_winners + 1):
        # if this many voters rank the same l candidates in first l positions, those candidates must win
        # TODO: Should this be rounded up, rather than down?
        # e.g. 50 voters, 4 winners. Should threshold be 12 or 13? If 12, 4 voters could pass with l=1 and a 5th could pass with l=2 (right?)
        threshold = int(l * num_voters / num_winners)

        rm = np.sum(rm_square[:, :l], axis=1)
        # ranked_in_top_l_count = np.sum(rm > threshold, axis=1)
        possible_winners = np.where(rm >= threshold)[0].tolist()

        # look at all size l sets of candidates
        # check if at least threshold voters rank that set in the top
        # record all voters that appear in a winning set (can there be more than one winning set?)

        for candidates in itertools.combinations(possible_winners, l):
            # for candidates in itertools.combinations(range(len(profile[0])), l):
            cset = set(candidates)
            voter_count = 0
            for ballot in profile:
                if set(ballot[:l]) == cset:
                    # top l candidates in this ballot are same as the current set of candidates
                    voter_count += 1
                if voter_count >= threshold:
                    winners = ballot[:l]
                    required_winners |= set(winners)
                    break
    return required_winners


def eval_solid_coalitions(committee, num_voters, num_winners, rank_choice):
    """
    Evaluate the solid coalitions axiom for a given committee and profiles.
    A solid coalition is if at least num_voters / n_winners voters
    rank some candidate c first, then c should be in the winning committee
    :param committee: A committee to evaluate.
    :param num_voters: The number of voters in the profiles.
    :param num_winners: The number of winners in the committee.
    :param rank_choice: The rank choice matrix for the profiles.
    """
    threshold = num_voters / num_winners

    for candidate in range(len(committee)):
        if rank_choice[candidate * len(committee)] >= threshold and committee[candidate] == 0:
            return 1

    return 0


def eval_consensus_committee(committee, num_voters, num_winners, profile, rank_choice, consensus_committees):
    """
    Evaluate the consensus committee axiom for a given committee and profiles.
    A consensus committee is for each k-element set W, where k = n_winners,
    such that each voter candidate_pairs some member of W first and each member of W is
    ranked first by either floor(num_voters / n_winners) or ceil(num_voters / n_winners)
    voters, then W should be the winning committee.
    :param committee: A committee to evaluate. A committee is a list of binary vectors where 1 indicates the candidate is in the committee.
    :param num_voters: The number of voters in the profiles.
    :param num_winners: The number of winners in the committee.
    :param profile: Profile of voters.
    :param rank_choice: The rank choice matrix for the profiles.
    """
    # lower_threshold = math.floor(num_voters / n_winners)
    # upper_threshold = math.ceil(num_voters / n_winners)

    satisfied = False
    if len(consensus_committees) == 0:
        # no consensus committee, impossible to violate
        satisfied = True
        # return True
    else:
        for cc in consensus_committees:
            # if committee[winner] == 1 for all winners in cc, return 0
            if all(committee[winner] == 1 for winner in cc):
                satisfied = True
                break

            # for winner in cc:
            #    if committee[winner] == 0:
            #        # a necessary winner is not winning :(
            #        satisfied = False
            #        break
    return int(not satisfied)

    # for W in itertools.combinations(range(len(committee)), n_winners):
    #     continue_flag = True
    #
    #     # need to check if each voter candidate_pairs some member of W first
    #     for voter in profiles:
    #         if not any(voter[0] == candidate for candidate in W):
    #             continue_flag = False
    #             break
    #
    #     if not continue_flag:
    #         continue
    #
    #     # need to check if each member of W is ranked first by either lower_threshold or upper_threshold voters
    #     for candidate in W:
    #         if not rank_choice[candidate * len(committee)] == lower_threshold and not rank_choice[candidate * len(
    #                 committee)] == upper_threshold:
    #             continue_flag = False
    #             break
    #
    #     if not continue_flag:
    #         continue
    #
    #     # if all conditions are met, check if W is the winning committee
    #     if all(committee[candidate] == 1 for candidate in W):
    #         return 0
    #     else:
    #         return 1
    #
    # return 0


def find_consensus_committees(num_voters, num_winners, profile):
    """
    Find all possible committees that would satisfy the consensus committee axiom. Defined as,
    For each k-element set W, where k = n_winners,
    such that each voter candidate_pairs some member of W first and each member of W is
    ranked first by either floor(num_voters / n_winners) or ceil(num_voters / n_winners)
    voters, then W should be the winning committee.
    :return:
    """
    lower_threshold = math.floor(num_voters / num_winners)
    upper_threshold = math.ceil(num_voters / num_winners)

    num_candidates = len(profile[0])

    consensus_committees = []

    # for each possible k-set of candidates
    for winner_set in itertools.combinations(range(num_candidates), num_winners):

        skip_to_next_winner_set = False
        # check if EVERY voter candidate_pairs some member of winner_set first
        for ballot in profile:
            if ballot[0] not in winner_set:
                # first choice is not in committee so this is not a consensus committee
                skip_to_next_winner_set = True
                break
        if skip_to_next_winner_set:
            continue

        # check if each member of winner_set is ranked first by either lower_threshold or upper_threshold voters
        for candidate in winner_set:
            first_count = 0
            for ballot in profile:
                if ballot[0] == candidate:
                    first_count += 1
            if first_count != lower_threshold and first_count != upper_threshold:
                # not ranked first by correct amount so not a consensus committee
                skip_to_next_winner_set = True
                break
        if skip_to_next_winner_set:
            continue

        # track consensus committees
        consensus_committees.append(winner_set)

    return consensus_committees


def eval_strong_unanimity(committee, num_winners, profile):
    """
    Evaluate the strong unanimity axiom for a given committee and profiles.
    Strong unanimity is satisfied if when each voter candidate_pairs the same n_winners candidates first,
    potentially in different order, then those candidates are the winning committee.
    :param committee: A committee to evaluate.
    :param num_winners: The number of winners in the committee.
    :param profile: Profile of voters.
    """

    unanimous_set = set(profile[0][:num_winners])

    for vote in profile:
        if set(vote[:num_winners]) != unanimous_set:
            return 0

    if all(committee[candidate] == 1 for candidate in unanimous_set):
        return 0

    return 1


def eval_local_stability(committee, profile, num_voters, quota):
    """
    Evaluate the local stability axiom for a given committee and profiles.
    A committee violate local stability if there is some subset of voters
    greater than the quota, and a candidate, c, not in the committee such that
    each voter from the subset prefers c to each member of the committee. 
    Otherwise, the committee provides local stability for the quota.
    :param committee: A committee to evaluate.
    :param profile: Profile of voters.
    :param quota: A quota to evaluate.
    """
    num_candidates = len(committee)
    not_in_committee = [i for i in range(num_candidates) if committee[i] == 0]

    if isinstance(profile[0], np.ndarray):
        using_numpy = True
    else:
        using_numpy = False

    for candidate in not_in_committee:

        preferred_by = 0

        for preferences in profile:
            if not using_numpy:
                if all(preferences.index(candidate) < preferences.index(member) for member in range(num_candidates) if
                       committee[member] == 1):
                    preferred_by += 1
            else:
                if all(np.where(preferences == candidate)[0][0] < np.where(preferences == member)[0][0]
                       for member in range(num_candidates) if committee[member] == 1):
                    preferred_by += 1

        # TODO: If we indent the below do we get an immediate (slight) speed boost?
        if preferred_by >= quota:
            return 1
    return 0


def eval_local_stability_fast(committee, profile, num_voters, rank_matrix, quota):
    """
    Evaluate the local stability axiom for a given committee and profiles.
    A committee violate local stability if there is some subset of voters
    greater than the quota, and a candidate, c, not in the committee such that
    each voter from the subset prefers c to each member of the committee.
    Otherwise, the committee provides local stability for the quota.
    :param committee: A committee to evaluate.
    :param profile: Profile of voters.
    :param quota: A quota to evaluate.
    """

    # Observation 1:
    # any candidate that might cause a violation of local stability is ranked in the top m-k more than quota times
    # Don't search for violations among other candidates

    # Observation 2:
    # we can search only committees that are a minimal size since larger violating committees also include these

    num_candidates = len(committee)
    num_winners = sum(committee)
    not_in_committee = [i for i in range(num_candidates) if committee[i] == 0]

    rm = np.array(rank_matrix).reshape(num_candidates, num_candidates)

    if isinstance(profile[0], np.ndarray):
        using_numpy = True
    else:
        using_numpy = False

    for candidate in not_in_committee:
        if np.sum(rm[candidate][:num_candidates - num_winners]) < quota:
            # no possible violation with this candidate
            continue

        preferred_by = 0

        for preferences in profile:
            if not using_numpy:
                if all(preferences.index(candidate) < preferences.index(member) for member in range(num_candidates) if
                       committee[member] == 1):
                    preferred_by += 1
            else:
                if all(np.where(preferences == candidate)[0][0] < np.where(preferences == member)[0][0]
                       for member in range(num_candidates) if committee[member] == 1):
                    preferred_by += 1

            # break early; quota has already been hit so no need to keep looking
            if preferred_by >= quota:
                return 1
    return 0


def eval_strong_pareto_efficiency(committee, profile):
    """
    Evaluate the strong Pareto efficiency axiom for a given committee and profile.
    A committee W1 dominates committee W2 if every voter has at least as many approved
    candidates in W1 as in W2 and there is one voter with strictly more approved candidates in W1.
    A committee is Pareto optimal if the output committee is not dominated by any other committee.
    Return 0 if the committee is Pareto optimal and 1 otherwise.
    :param committee: A committee to evaluate.
    :param profile: Profile of voters.
    """
    num_candidates = len(committee)
    num_winners = sum(committee)

    members = set(idx for idx, val in enumerate(committee) if val == 1)

    approval_sets = []
    for ranking in profile:
        approval_sets.append(set(ranking[:num_winners]))

    current_intersection = [len(approval & members) for approval in approval_sets]

    all_candidates = list(range(num_candidates))

    for W_prime in itertools.combinations(all_candidates, num_winners):
        W_prime_set = set(W_prime)
        if W_prime_set == members:
            continue

        W_prime_intersection = [len(approval & W_prime_set) for approval in approval_sets]
        dominates = True
        stricly_better = False

        for current, prime in zip(current_intersection, W_prime_intersection):
            if prime < current:
                dominates = False
                break
            if prime > current:
                stricly_better = True

        if dominates and stricly_better:
            return 1

    return 0


"""
def eval_majority_axiom(row, rule, tie):
    Evaluate the majority axiom for a given committee and profiles.
    If the majority of agents put a candidate in the first position, then that 
    candidate should be in the winning committee
    :param row: A row of a dataframe containing the preference profiles and committee.
    :param rule: The voting rule to evaluate.
    :param tie: Whether its considering ties or single results.
    :return: A float indicating the percentage of times the majority axiom is violated.
    rank_choice = eval(row["rank_matrix"])
    maj_threshold = len(eval(row["raw_profiles"])) // 2

    if tie:
        committee = row[f"{rule}-tied_winners"]
        num_candidates = len(committee[0])
    else:
        committee = row[f"{rule}-single_winner"]
        num_candidates = len(committee)

    top_rated = [rank_choice[i*num_candidates] for i in range(num_candidates)]
    # if if isnt a tie
    if not tie:
        # go through each candidate
        for candidate in range(num_candidates):
        # if candidate > 0.5, check if they are in the committee
            if top_rated[candidate] > maj_threshold:
                # if they are not in the committee, return 1
                if committee[candidate] == 0:
                    return 1
        return 0
    # if it is a tie
    else:
        # initialize a counter for number of violations, start at 0
        violations = 0
        # go through each winning committee
        for c in committee:
            # go through each candidate
            for candidate in range(num_candidates):
                # if candidate ranked #1 > 0.5, check if they are in the committee
                if top_rated[candidate] > maj_threshold:
                    # if they are not in the committee, increment the counter
                    if c[candidate] == 0:
                        violations += 1
        # return the counter divided by the number of winning committees
        return violations / len(committee)
    

def eval_majority_loser_axiom(row, rule, tie):
    Evaluate the majority loser axiom for a given committee and profiles.
    If the majority of agents put a candidate in the last position, then that 
    candidate should not be in the winning committee
    :param row: A row of a dataframe containing the preference profiles and committee.
    :param rule: The voting rule to evaluate.
    :param tie: Whether its considering ties or single results.
    :return: A float indicating the percentage of times the majority loser axiom is violated.
    rank_choice = eval(row["rank_matrix"])
    maj_threshold = len(eval(row["raw_profiles"])) // 2

    if tie:
        committee = row[f"{rule}-tied_winners"]
        num_candidates = len(committee[0])
    else:
        committee = row[f"{rule}-single_winner"]
        num_candidates = len(committee)

    bottom_rated = [rank_choice[i*num_candidates+num_candidates-1] for i in range(num_candidates)]

    # if if isnt a tie
    if not tie:
        # go through each candidate
        for candidate in range(num_candidates):
            # if bottom rated > 50% of the time, check if they are in the committee
            if bottom_rated[candidate] > maj_threshold:
                # if they are in the committee, return 1
                if committee[candidate] == 1:
                    return 1
        return 0
    # if it is a tie
    else:
        # initialize a counter for number of violations, start at 0
        violations = 0
        # go through each winning committee
        for c in committee:
            # go through each candidate
            for candidate in range(num_candidates):
                # if bottom rated > 50% of the time, check if they are in the committee
                if bottom_rated[candidate] > maj_threshold:
                    # if they are in the committee, increment the counter
                    if c[candidate] == 1:
                        violations += 1
        # return the counter divided by the number of winning committees
        return violations / len(committee)
"""

"""
def eval_condorcet_winner(row, rule, tie):
    
    Evaluate the Condorcet winner axiom for a given committee and profiles.
    Condorcet winner in this instance is the Elkind, Lang, and Saffidine variant:
    A condorcet winning set is a set that, for each member d not in the set, some member c in the set
    is preferred to d by a majority
    :param row: A row of a dataframe containing the preference profiles and committee.
    :param rule: The voting rule to evaluate.
    :param tie: Whether its considering ties or single results.
    :return: A float indicating the percentage of times the majority loser axiom is violated.
    

    def isCondorcetWinner(winning_committee):
        # find which members are in the committee and which arent
        in_committee = [i for i, x in enumerate(winning_committee) if x == 1]
        not_in_committee = [i for i, x in enumerate(winning_committee) if x == 0]

        # for each member not in the committee, d
        for d in not_in_committee:
            # have a "violation" flag, set to true
            violation = True
            # go through each member in the committee, c
            for c in in_committee:
                # if the candidate pairs matrix indicates that c > d
                if cand_pairs[c * num_candidates + d] > cand_pairs[d * num_candidates + c]:
                    # set violation flag to false
                    violation = False
                    break
            # if violation flag is true, return 1
            if violation:
                return 1
        return 0

    
    cand_pairs = eval(row["candidate_pairs"])

    if tie:
        committee = row[f"{rule}-tied_winners"]
        num_candidates = len(committee[0])
    else:
        committee = row[f"{rule}-single_winner"]
        num_candidates = len(committee)

    # if it isnt a tie
    if not tie:
        return isCondorcetWinner(committee)
    # if it is a tie
    else:
        violations = 0
        # for each committee
        for c in committee:
            # do same as above
            violations += isCondorcetWinner(c)
        return violations / len(committee)
"""
"""
def eval_condorcet_loser(row, rule, tie):
    
    Evaluate the Condorcet loser axiom for a given committee and profiles.
    Condorcet loser in this instance is the opposite of the Condorcet winner above:
    A condorcet losing set is a set that, for each member c in the set, there exists some member d
    not in the set that is preferred to c by a majority
    :param rank_counts: The number of times each candidate is ranked in each position.
    :param winning_committee: The committee to evaluate.
    :param num_candidates: The number of candidates.
    :return: A boolean indicating whether the Condorcet loser is the losing candidate.
    

    def isCondorcetLoser(winning_committee):
        # find which members are in the committee and which arent
        in_committee = [i for i, x in enumerate(winning_committee) if x == 1]
        not_in_committee = [i for i, x in enumerate(winning_committee) if x == 0]

        violation = 1
        # set violation flag to True
        # for each member in the committee, c
        for c in in_committee:
            # for each member not in the committee, d
            for d in not_in_committee:
                # if the binary matrix indicates that c > d
                if cand_pairs[c * num_candidates + d] >= cand_pairs[d * num_candidates + c]:
                    # set violation to false
                    violation = 0
        # return violation

        return violation

    
    cand_pairs = eval(row["candidate_pairs"])

    if tie:
        committee = row[f"{rule}-tied_winners"]
        num_candidates = len(committee[0])
    else:
        committee = row[f"{rule}-single_winner"]
        num_candidates = len(committee)

    # if it isnt a tie
    if not tie:
        return isCondorcetLoser(committee)
    # if it is a tie
    else:
        violations = 0
        # for each committee
        for c in committee:
            # do same as above
            violations += isCondorcetLoser(c)
        return violations / len(committee)
"""
