from pref_voting.scoring_methods import scoring_rule
from pref_voting.voting_method import vm
from pref_voting.voting_method_properties import ElectionTypes
import pyrankvote as prv
from pyrankvote import Ballot, Candidate
from collections import Counter


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


@vm(name="STV_old", input_types=[ElectionTypes.PROFILE])
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
        number_of_seats=k,
        pick_random_if_blank=True
    )

    winners = [int(c.name) for c in election_result.get_winners()]
    return winners


@vm(name="STV", input_types=[ElectionTypes.PROFILE])
def stv(preferences, k, curr_cands=None, tie_breaking=""):
    """
    Runs STV with fractional weight transfers; all voters with an elected alternative have their weight reduced
    by an amount proportional to the number of excess votes.
    :param preferences:
    :param k:
    :param curr_cands:
    :param tie_breaking:
    :return:
    """
    # Initialize variables
    preferences = preferences.rankings
    num_voters = len(preferences)
    quota = num_voters // (k + 1) + 1  # Droop quota
    alternatives = set([alt for pref in preferences for alt in pref])
    winners = []
    remaining_candidates = alternatives.copy()

    # Track fractional transfer weights
    transfer_weights = [1] * num_voters

    def get_first_preferences(preferences, remaining_candidates, transfer_weights):
        """
        Count the first preference votes for remaining candidates, weighted by transfer_weights.
        """
        first_pref_count = Counter()
        for i, pref in enumerate(preferences):
            for alt in pref:
                if alt in remaining_candidates:
                    first_pref_count[alt] += transfer_weights[i]
                    break
        return first_pref_count

    def redistribute_votes(preferences, remaining_candidates, transfer_weights, excess, elected):
        """
        Redistribute excess votes of the elected candidate proportionally.
        """
        new_weights = transfer_weights[:]
        for i, pref in enumerate(preferences):
            if pref[0] == elected:
                # Transfer excess votes proportionally
                new_weights[i] *= excess
        return new_weights

    def eliminate_candidate(preferences, eliminated):
        """
        Remove the eliminated candidate from all preferences.
        """
        return [tuple(alt for alt in pref if alt != eliminated) for pref in preferences]

    # STV Process
    while len(winners) < k:
        # Get the first preference counts
        first_pref_count = get_first_preferences(preferences, remaining_candidates, transfer_weights)

        # Check if any candidate meets or exceeds the quota
        for candidate, votes in first_pref_count.items():
            if votes >= quota:
                winners.append(candidate)
                remaining_candidates.remove(candidate)

                # Calculate the excess votes
                excess = (votes - quota) / votes

                # Redistribute the excess votes of the elected candidate
                transfer_weights = redistribute_votes(preferences, remaining_candidates, transfer_weights, excess,
                                                      candidate)

                # Remove the elected candidate from the preferences
                preferences = eliminate_candidate(preferences, candidate)

                break
        else:
            # No candidate reached the quota, eliminate the candidate with the fewest votes
            
            min_votes = min(first_pref_count.values())
            eliminated = min(candidate for candidate, votes in first_pref_count.items() if votes == min_votes)

            # Eliminate the candidate and remove from preferences
            remaining_candidates.remove(eliminated)
            preferences = eliminate_candidate(preferences, eliminated)

    return winners

@vm(name="STV-OLD", input_types=[ElectionTypes.PROFILE])
def stv_old(preferences, k, curr_cands=None, tie_breaking=""):
    """
    ChatGPT code; Transfers weight from arbitrarily selected voters beyond the excess. Moved away to make
    space for a fractional transfer algorithm
    :param preferences:
    :param k:
    :param curr_cands:
    :param tie_breaking:
    :return:
    """
    preferences = preferences.rankings

    num_voters = len(preferences)
    quota = num_voters // (k + 1) + 1  # Droop quota
    alternatives = set([alt for pref in preferences for alt in pref])
    winners = []
    remaining_candidates = alternatives.copy()

    def get_first_preferences(preferences, remaining_candidates):
        """
        Count the first preference votes for remaining candidates.
        """
        first_pref_count = Counter()
        for pref in preferences:
            for alt in pref:
                if alt in remaining_candidates:
                    first_pref_count[alt] += 1
                    break
        return first_pref_count

    def transfer_excess_votes(preferences, candidate, excess_votes):
        """
        Transfer votes for the elected candidate to the next preferred remaining candidate.
        Randomly select excess_votes number of voters who voted for the elected candidate.
        """
        # Find all voters who voted for the elected candidate first
        voters_for_candidate = [i for i, pref in enumerate(preferences) if pref[0] == candidate]

        # select `excess_votes` number of these voters for transferring
        voters_to_transfer = voters_for_candidate[:excess_votes]

        # Create a new preferences list where selected voters' votes are transferred
        new_preferences = []
        for i, pref in enumerate(preferences):
            if i in voters_to_transfer:
                # Transfer this voter's vote to their next valid preference
                new_pref = tuple(alt for alt in pref[1:] if alt in remaining_candidates)
                if new_pref:
                    new_preferences.append(new_pref)
            else:
                # Keep the voter's preference list (but remove the elected candidate)
                new_pref = tuple(alt for alt in pref if alt != candidate)
                if new_pref:
                    new_preferences.append(new_pref)

        return new_preferences

    def eliminate_candidate(preferences, eliminated):
        """
        Remove the eliminated candidate from all preferences.
        """
        return [tuple(alt for alt in pref if alt != eliminated) for pref in preferences]

    # Start the STV process
    while len(winners) < k:
        # Get the first preference counts
        first_pref_count = get_first_preferences(preferences, remaining_candidates)

        # Check if any candidate meets or exceeds the quota
        for candidate, votes in first_pref_count.items():
            if votes >= quota:
                winners.append(candidate)
                remaining_candidates.remove(candidate)

                excess_votes = votes - quota
                preferences = transfer_excess_votes(preferences, candidate, excess_votes)
                preferences = eliminate_candidate(preferences, candidate)

                break
        else:
            # No candidate reached the quota, eliminate the candidate with the fewest votes
            min_votes = min(first_pref_count.values())
            eliminated = min(candidate for candidate, votes in first_pref_count.items() if votes == min_votes)

            # Eliminate the candidate and remove from preferences
            remaining_candidates.remove(eliminated)
            preferences = eliminate_candidate(preferences, eliminated)

    return winners


# def stv_fractional(preferences, k):
#     def droop_quota(n_voters):
#         """The `Droop quota`_ for Single Transferable Vote tabulation. A candidate
#         whose vote total meets this quota wins a seat.
#
#         .. _`Droop quota`: https://en.wikipedia.org/wiki/Droop_quota
#         """
#         return math.floor(n_voters / (k + 1)) + 1
#
#     def find_least(candidates):
#         iter_candidates = iter(candidates)
#         least = next(iter_candidates)
#         for candidate in candidates:
#             if candidate.total_votes < least.total_votes:
#                 least = candidate
#         return least
#
#     def declare_winner(winner):
#         ballots_to_transfer = transferable_votes(winner)
#         self.schedule.distribute_ballots(ballots_to_transfer)
#         self.elected.add(winner)
#         self.schedule.remove_candidate(winner)
#
#     def transferable_votes(candidate):
#         total = candidate.total_votes
#         # quota = self.quota
#
#         if total <= quota:
#             return BallotSet()
#
#         fraction = (total - quota) / total
#         transferable = BallotSet()
#         for vote, count in candidate.votes:
#             next_choice = vote.next_choice
#             if next_choice is not None:
#                 transferable.add(vote.eliminate(candidate), count * fraction)
#
#         return transferable
#
#     def find_winners(cands, quota):
#         return {candidate for candidate in cands if candidate.total_votes >= quota}
#
#
#     if len(preferences[0]) == k:
#         return preferences[0]
#
#     n_voters = len(preferences)
#     candidates = list(preferences[0])
#     preferences = copy.deepcopy(preferences)
#     elected = set()
#
#     quota = droop_quota(n_voters)
#
#     while elected < k and len(candidates) > 0:
#         if len(candidates) + len(elected) == k:
#             return {
#                 str(candidate) for candidate in elected.union(candidates)
#             }
#         winners = find_winners(candidates, droop_quota(n_voters=len(preferences)))
#         if len(winners) > 0:
#             for winner in winners:
#                 declare_winner(winner)
#         else:
#             least = find_least(candidates)
#             self.schedule.eliminate(least)
#     return {str(candidate) for candidate in self.elected}