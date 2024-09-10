import math
import pprint
import numpy as np
import itertools


def generate_all_committees(n_candidates, n_winners):
    """
    Generate all possible winning committees with n_winners winners out of n_candidates candidates.
    """
    indices = list(itertools.combinations(range(n_candidates), n_winners))
    committees = np.zeros((len(indices), n_candidates), dtype=int)
    for i, idx in enumerate(indices):
        committees[i, idx] = 1
    committees = committees.tolist()
    return committees


def differentiable_greater_than(a, b, alpha=10):
    return 1 / (1 + math.exp(alpha * (b - a)))


def dgt(a, b):
    return differentiable_greater_than(a, b)


def first_count(c, ranks=None, candidate_pairs=None):
    """
    Return number of times c is ranked first in the given rank data.
    :param c:
    :param ranks:
    :param candidate_pairs:
    :return:
    """
    return ranks[c][0]


def prefer_count(a, b, ranks=None, candidate_pairs=None):
    """
    Return the number of voters that prefer a to b.
    :param a:
    :param b:
    :param ranks:
    :param candidate_pairs:
    :return:
    """
    return candidate_pairs[a][b]


def is_condorcet_committee(w, ranks=None, candidate_pairs=None):
    """
    Return 1 if w is a condorcet committee and 0 otherwise.
    :param w: binary list with one element per candidate. 1 if candidate is in committee, 0 otherwise.
    :param ranks:
    :param candidate_pairs:
    :return:
    """
    n = sum(ranks[0])
    is_condorcet = True

    for c, c_in_committee in enumerate(w):
        if not c_in_committee:
            continue
        for d, d_in_committee in enumerate(w):
            if d_in_committee:
                continue

            # if a majority prefer d to c then w is not a condorcet committee
            pc = prefer_count(d, c, candidate_pairs=candidate_pairs)
            if pc > n/2:
                is_condorcet = False
                break
        if not is_condorcet:
            break

    return 1 - int(is_condorcet)


def is_condorcet_loss(w, ranks=None, candidate_pairs=None):
    """
    Return 1 if w is a condorcet committee, and 0 otherwise. Using differentiable loss version.
    :param w: binary list with one element per candidate. 1 if candidate is in committee, 0 otherwise.
    :param candidate_pairs:
    :return:
    """
    n = candidate_pairs[0][1] + candidate_pairs[1][0]

    all_c_comparisons = []
    for c, c_in_committee in enumerate(w):
        if not c_in_committee:
            continue
        all_d_comparisons = []
        for d, d_in_committee in enumerate(w):
            if d_in_committee:
                continue

            p = prefer_count(c, d, candidate_pairs=candidate_pairs)
            p = dgt(p, n/2)
            all_d_comparisons.append(p)

        min_d = min(all_d_comparisons)
        all_c_comparisons.append(min_d)
    is_condorcet = min(all_c_comparisons)

    return is_condorcet


def condorcet_axiom_loss(w, k=2, ranks=None, candidate_pairs=None):
    """
    Return value close to 0 if condorcet axiom is satisfied by committee w and value close to 1 otherwise.
    :param w:
    :param k:
    :param ranks:
    :param candidate_pairs:
    :return:
    """
    # we have function to check if a committee is a condorcet committee
    # we need to check (1) if a condorcet committee exists, (2) if w is a condorcet committee
    # in future there may be some room for short-circuting by checking (2) first and returning it if it is close to 1
    # return max(1 - condorcet_exists, w_is_condorcet)
    m = len(w)

    # set a to 1 - (does a condorcet committee exist)
    # enumerate all possible committees of given size
    all_committee_condorcet_chances = []
    for possible_committee in generate_all_committees(m, k):
        p = is_condorcet_loss(possible_committee, ranks=ranks, candidate_pairs=candidate_pairs)
        all_committee_condorcet_chances.append(p)
    is_condorcet_chance = max(all_committee_condorcet_chances)
    a = 1 - is_condorcet_chance

    b = is_condorcet_loss(w, ranks=ranks, candidate_pairs=candidate_pairs)

    return 1 - max(a, b)


def majority_axiom(w, ranks=None, candidate_pairs=None):
    """
    Return 0 if majority axioms is satisfied by committee w.
    Return 1 otherwise.
    :return:
    """
    violated = False
    for c, in_committee in enumerate(w):
        if first_count(c, ranks, candidate_pairs) >= len(ranks) / 2:
            # c is ranked first by at least half so it should be in committee
            if not in_committee:
                violated = True
                break
    return int(violated)


def majority_axiom_loss(w, ranks=None, candidate_pairs=None):
    """
    Return low value (close to 0) if majority axioms is satisfied by committee w.
    Return high value (close to 1) otherwise.
    :param w:
    :param ranks:
    :param candidate_pairs:
    :return:
    """
    # return 1 - min_c(max(a, b)) for all candidates c
    # where a = 1 - dgt(first(c), n/2)
    # and b = one_hot_encoding(c) * w
    n = sum(ranks[0])
    m = len(w)

    # outer min term; collect max(a, b) over all values of c then do min of all maxes
    # checks that every candidate is either in the committee or not a majority winner
    all_c_terms = []
    for c, in_committee in enumerate(w):
        # is c a majority winner? (but negated; should be near 1 if c is not a majority winner)
        a = 1 - dgt(first_count(c, ranks), n / 2)

        # is c in the committee?
        one_hot_c = np.array([0] * m)
        one_hot_c[c] = 1
        b = np.dot(one_hot_c, np.array(w))
        all_c_terms.append(max(a, b))

    satisfied = min(all_c_terms)
    return 1 - satisfied


candidate_pairs = [
    [0, 5, 5, 4],
    [0, 0, 3, 3],
    [0, 2, 0, 1],
    [1, 2, 4, 0]
]
rank_matrix = [
    [4, 0, 0, 1],
    [1, 3, 0, 1],
    [0, 0, 3, 2],
    [0, 2, 2, 1]
]

committees = [
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
]

for committee in committees:
    actually_satisfied = majority_axiom(w=committee, ranks=rank_matrix)
    satisfied_loss = majority_axiom_loss(w=committee, ranks=rank_matrix)

    satisfies_condorcet = is_condorcet_committee(w=committee, ranks=rank_matrix, candidate_pairs=candidate_pairs)
    # satisfies_condorcet_loss = is_condorcet_loss(w=committee, candidate_pairs=rank_matrix, candidate_pairs=candidate_pairs)
    satisfies_condorcet_loss = condorcet_axiom_loss(w=committee, k=2, ranks=rank_matrix, candidate_pairs=candidate_pairs)
    print(f"committee: {committee}")
    print(f"Majority axiom satisfied? { actually_satisfied}")
    print(f"Majority axiom satisfied by loss? {satisfied_loss}")
    print(f"Is it Condorcet? {satisfies_condorcet}")
    print(f"Is it Condorcet by loss? {satisfies_condorcet_loss}")
    print()