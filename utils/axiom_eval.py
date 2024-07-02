import itertools

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
    top_rated = [rank_choice[i*num_candidates] for i in range(num_candidates)]
    
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
    bottom_rated = [rank_choice[i*num_candidates+num_candidates-1] for i in range(num_candidates)]

    for candidate in range(num_candidates):
        if  bottom_rated[candidate] > maj_threshold and committee[candidate] == 1:
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
    """

    in_committee = [i for i, x in enumerate(committee) if x == 1]
    not_in_committee = [i for i, x in enumerate(committee) if x == 0]
    num_candidates = len(committee)

    def is_condorcet_winner(i):
        for j in range(num_candidates):
            if i != j and cand_pairs[i*num_candidates + j] <= cand_pairs[j*num_candidates + i]:
                return False
        return True

    condorcet_winner = None

    for i in range(num_candidates):
        if is_condorcet_winner(i):
            condorcet_winner = i
            break
    
    if condorcet_winner is not None and committee[condorcet_winner] == 0:
        return 1
    return 0
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

