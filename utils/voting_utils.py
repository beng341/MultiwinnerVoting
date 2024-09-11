import math
import random

from pref_voting.scoring_methods import scoring_rule
from pref_voting.voting_method import vm, _num_rank_first
from pref_voting.voting_method_properties import ElectionTypes
import numpy as np
import pyrankvote as prv
from pyrankvote import Ballot, Candidate


@vm(name="Two-Approval")
def two_approval(profile, curr_cands=None):
    """Returns the list of candidates with the largest two-approval score in the profiles restricted to curr_cands.
    """

    two_approval_score = lambda num_cands, rank: 1 if rank == 1 or rank == 2 else 0

    return scoring_rule(profile, curr_cands=curr_cands, score=two_approval_score)


@vm(name="Three-Approval")
def three_approval(profile, curr_cands=None):
    """Returns the list of candidates with the largest three-approval score in the profiles restricted to curr_cands.
    """

    three_approval_score = lambda num_cands, rank: 1 if rank == 1 or rank == 2 or rank == 3 else 0

    return scoring_rule(profile, curr_cands=curr_cands, score=three_approval_score)


@vm(name="STV", input_types=[ElectionTypes.PROFILE])
def single_transferable_vote(profile, k, curr_cands=None, tie_breaking=""):
    """
    Elect a committee of k candidates using the STV process in: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7089675/
    This acts as a wrapper for the pyrankvote library
    :param profile:
    :param curr_cands:
    :return: A sorted list of candidates
    """
    candidates = [Candidate(str(a)) for a in profile.rankings[0]]
    candidates_dict = {candidate.name: candidate for candidate in candidates}

    ballots = [Ballot(ranked_candidates=[candidates_dict[str(a)] for a in ranking]) for ranking in profile.rankings]

    election_result = prv.single_transferable_vote(
        candidates=candidates,
        ballots=ballots,
        number_of_seats=k
    )

    winners = [int(c.name) for c in election_result.get_winners()]
    return winners