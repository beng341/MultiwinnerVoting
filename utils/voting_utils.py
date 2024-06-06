from pref_voting.scoring_methods import scoring_rule
from pref_voting.voting_method import vm


@vm(name="Two-Approval")
def two_approval(profile, curr_cands=None):
    """Returns the list of candidates with the largest two-approval score in the profile restricted to curr_cands.
    """

    two_approval_score = lambda num_cands, rank: 1 if rank == 1 or rank == 2 else 0

    return scoring_rule(profile, curr_cands=curr_cands, score=two_approval_score)


@vm(name="Three-Approval")
def three_approval(profile, curr_cands=None):
    """Returns the list of candidates with the largest three-approval score in the profile restricted to curr_cands.
    """

    three_approval_score = lambda num_cands, rank: 1 if rank == 1 or rank == 2 or rank == 3 else 0

    return scoring_rule(profile, curr_cands=curr_cands, score=three_approval_score)
