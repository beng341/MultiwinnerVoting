import itertools
import math


def eval_majority_axiom(n_voters, committee, rank_choice):
    """
    Evaluate the majority axiom for a given committee and profile.
    If the majority of agents put a candidate in the first position, then that 
    candidate should be in the winning committee
    :param profile: Profile of voter preferences.
    :param committee: A committee to valuate.
    :param rank_choice: The rank choice matrix for the profile.
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
    Evaluate the majority loser axiom for a given committee and profile.
    If the majority of agents put a candidate in the last position, then that 
    candidate should not be in the winning committee
    :param profile: Profile of voter preferences.
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


def exists_condorcet_winner(all_committees, cand_pairs):
    for committee in all_committees:
        if eval_condorcet_winner(committee, cand_pairs) == 0:
            return True

    return False


def eval_condorcet_winner(committee, cand_pairs):
    """
    Evaluate the Condorcet winner axiom for a given committee and profile.
    Condorcet winner in this instance is:
    A committee is a Condorcet committee if each candidate in it is preferred, 
    by a majority of voters, to each candidate outside it
    :param committee: A committee to valuate.
    :param cand_pairs: The candidate pairs matrix for the profile.
    :return: 0 if the axiom is not violated and 1 if it is.
    """

    in_committee = [i for i, x in enumerate(committee) if x == 1]
    not_in_committee = [i for i, x in enumerate(committee) if x == 0]
    num_candidates = len(committee)

    for c in in_committee:
        for d in not_in_committee:
            if cand_pairs[c * num_candidates + d] < cand_pairs[d * num_candidates + c]:
                return 1
    return 0


def eval_condorcet_loser(committee, cand_pairs):
    """
    Evaluate the Condorcet loser axiom for a given committee and profile.
    Condorcet loser in this instance is the opposite of the Condorcet winner above:
    A condorcet losing set is a set that, for each member c in the set, all non-members d
    are preferred to c by a majority
    :param committee: A committee to valuate.
    :param cand_pairs: The candidate pairs matrix for the profile.
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
    Evaluate Dummett's condition for a given committee and profile.
    Dummett's condition states that if for some l <= n_winners, there
    is a group of l * num_voters / n_winners that all rank the same l candidates
    on top, then those l candidates should be in the winning committee.
    This requirement tries to capture the idea of proportional representation,
    or proportionality for solid coalitions.
    :param committee: A committee to evaluate.
    :param num_voters: The number of voters in the profile.
    :param num_winners: The number of winners in the committee.
    :param profile: Profile of voter preferences.
    """
    all_required_winners_are_winning = True
    for rw in required_winners:
        if committee[rw] != 1:
            all_required_winners_are_winning = False
    # return 1 iff required winners aren't winning (axiom is violated), 0 if required winners are winning (not violated)
    return int(not all_required_winners_are_winning)

    # for l in range(1, n_winners + 1):
    #     threshold = int(l * num_voters / n_winners)
    #
    #     for candidates in itertools.combinations(range(len(profile[0])), l):
    #         count = sum(1 for ballot in profile if set(ballot[:l]) == set(candidates))
    #         if count >= threshold:
    #             if all(committee[candidate] == 1 for candidate in candidates):
    #                 continue
    #             else:
    #                 return 1
    # return 0


def find_dummett_winners(num_voters, num_winners, profile):
    """
    Find the committees that are able to satisfy dummett's condition.
    Dummett's condition states that if for some l <= n_winners, there
    is a group of l * num_voters / n_winners that all rank the same l candidates
    on top, then those l candidates should be in the winning committee.
    This requirement tries to capture the idea of proportional representation,
    or proportionality for solid coalitions.
    :param num_voters:
    :param num_winners:
    :param profile:
    :return:
    """
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


def eval_solid_coalitions(committee, num_voters, num_winners, rank_choice):
    """
    Evaluate the solid coalitions axiom for a given committee and profile.
    A solid coalition is if at least num_voters / n_winners voters
    rank some candidate c first, then c should be in the winning committee
    :param committee: A committee to evaluate.
    :param num_voters: The number of voters in the profile.
    :param num_winners: The number of winners in the committee.
    :param rank_choice: The rank choice matrix for the profile.
    """
    threshold = math.ceil(num_voters / num_winners)

    for candidate in range(len(committee)):
        if rank_choice[candidate * len(committee)] >= threshold and committee[candidate] == 0:
            return 1

    return 0


def eval_consensus_committee(committee, num_voters, num_winners, profile, rank_choice, consensus_committees):
    """
    Evaluate the consensus committee axiom for a given committee and profile.
    A consensus committee is for each k-element set W, where k = n_winners,
    such that each voter ranks some member of W first and each member of W is
    ranked first by either floor(num_voters / n_winners) or ceil(num_voters / n_winners)
    voters, then W should be the winning committee.
    :param committee: A committee to evaluate. A committee is a list of binary vectors where 1 indicates the candidate is in the committee.
    :param num_voters: The number of voters in the profile.
    :param num_winners: The number of winners in the committee.
    :param profile: Profile of voters.
    :param rank_choice: The rank choice matrix for the profile.
    """
    # lower_threshold = math.floor(num_voters / num_winners)
    # upper_threshold = math.ceil(num_voters / num_winners)

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

            #for winner in cc:
            #    if committee[winner] == 0:
            #        # a necessary winner is not winning :(
            #        satisfied = False
            #        break
    return int(not satisfied)

    # for W in itertools.combinations(range(len(committee)), num_winners):
    #     continue_flag = True
    #
    #     # need to check if each voter ranks some member of W first
    #     for voter in profile:
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
    such that each voter ranks some member of W first and each member of W is
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
        # check if EVERY voter ranks some member of winner_set first
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
    Evaluate the strong unanimity axiom for a given committee and profile.
    Strong unanimity is if each voter ranks the same n_winners candidates first,
    potentially in different order, then those candidates should be in the winning committee.
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
    Evaluate the local stability axiom for a given committee and profile.
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

    for candidate in not_in_committee:

        preferred_by = 0

        for preferences in profile:
            if all(preferences.index(candidate) < preferences.index(member) for member in range(num_candidates) if
                   committee[member] == 1):
                preferred_by += 1

        if preferred_by >= quota:
            return 1
    return 0


"""
def eval_majority_axiom(row, rule, tie):
    Evaluate the majority axiom for a given committee and profile.
    If the majority of agents put a candidate in the first position, then that 
    candidate should be in the winning committee
    :param row: A row of a dataframe containing the preference profile and committee.
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
    Evaluate the majority loser axiom for a given committee and profile.
    If the majority of agents put a candidate in the last position, then that 
    candidate should not be in the winning committee
    :param row: A row of a dataframe containing the preference profile and committee.
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
    
    Evaluate the Condorcet winner axiom for a given committee and profile.
    Condorcet winner in this instance is the Elkind, Lang, and Saffidine variant:
    A condorcet winning set is a set that, for each member d not in the set, some member c in the set
    is preferred to d by a majority
    :param row: A row of a dataframe containing the preference profile and committee.
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
    
    Evaluate the Condorcet loser axiom for a given committee and profile.
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
