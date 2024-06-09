import numpy as np
from sklearn.preprocessing import normalize
from pref_voting import profiles as pref_voting_profiles
from abcvoting import abcrules


def compute_features_from_profiles(profiles, df=None):
    """
    Make a dict containing each feature type for all of the given profiles.
    Feature type name points at feature values in the dict.
    :param profiles: A list of lists. Each sublist is an individual preference order.
    :param df: If given, add each computed feature as a column in this Dataframe.
    :return:
    """

    features_dict = dict()

    # add candidate pairs
    cps = candidate_pairs_from_profiles(profiles)
    pair_str = [str(w) for w in cps]
    features_dict[f"candidate_pairs"] = pair_str
    normalized = normalize_array(cps)
    pair_str = [str(w.tolist()) for w in normalized]
    features_dict[f"candidate_pairs-normalized"] = pair_str

    cps = candidate_pairs_from_profiles(profiles, remove_diagonal=True)
    pair_str = [str(w) for w in cps]
    features_dict[f"candidate_pairs-no_diagonal"] = pair_str
    normalized = normalize_array(cps)
    pair_str = [str(w.tolist()) for w in normalized]
    features_dict[f"candidate_pairs-normalized-no_diagonal"] = pair_str

    cps = candidate_pairs_from_profiles(profiles, upper_half_only=True)
    normalized = normalize_array(cps)
    pair_str = [str(w.tolist()) for w in normalized]
    features_dict[f"candidate_pairs-normalized-upper_half"] = pair_str

    # add binary candidate pairs
    bps = binary_candidate_pairs_from_profiles(profiles)
    pair_str = [str(w) for w in bps]
    features_dict[f"binary_pairs"] = pair_str

    bps = binary_candidate_pairs_from_profiles(profiles, remove_diagonal=True)
    pair_str = [str(w) for w in bps]
    features_dict[f"binary_pairs-no_diagonal"] = pair_str

    bps = binary_candidate_pairs_from_profiles(profiles, upper_half_only=True)
    pair_str = [str(w) for w in bps]
    features_dict[f"binary_pairs-upper_half"] = pair_str

    # add rank matrices
    ranks = rank_counts_from_profiles(profiles)
    pair_str = [str(w) for w in ranks]
    features_dict[f"rank_matrix"] = pair_str
    normalized = normalize_array(ranks)
    pair_str = [str(w.tolist()) for w in normalized]
    features_dict[f"rank_matrix-normalized"] = pair_str

    if df is not None:
        for key, val in features_dict.items():
            df[key] = val
    return features_dict


def candidate_pairs_from_profiles(profiles, remove_diagonal=False, upper_half_only=False):
    """
    Return a list where, for each profile, a flattened m*m list is returned. list[i][j] is the number of times
    alternative i is preferred to alternative j
    :param profiles: list of individual preferences orders/ballots
    :param upper_half_only:
    :param remove_diagonal:
    :return:
    """
    # raw_profiles = [[[0, 1, 2, 3], [1, 0, 2, 3], [0, 1, 3, 2], [3,2,1,0]]]
    m = len(profiles[0][0])  # length of first ballot in first profile

    features = []

    for profile in profiles:
        preferred_counts = np.zeros((m, m), dtype=np.int64)
        iterate_over = profile
        for ballot in iterate_over:
            # for ballot in profile:
            order = ballot
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    preferred_counts[order[i], order[j]] += 1

        if remove_diagonal:
            preferred_counts = preferred_counts[~np.eye(m, dtype=bool)].reshape(preferred_counts.shape[0], -1)
            # exit("Have not yet tested whether removing main diagonals works.")
        elif upper_half_only:
            preferred_counts = preferred_counts[np.triu_indices_from(preferred_counts, k=1)]

        features.append(preferred_counts.flatten().tolist())

    return features


def binary_candidate_pairs_from_profiles(profiles, remove_diagonal=False, upper_half_only=False):
    """
    Return a list where, for each profile, a flattened m*m list is returned. list[i][j] is 1 iff more voters prefer
    i to j and 0 otherwise. Note: Ties are represented as 0 values.
    :param profiles: list of individual preferences orders/ballots
    :param remove_diagonal:
    :param upper_half_only:
    :return:
    """

    features = []

    for profile in profiles:
        n = len(profile)  # equal to number of voters
        m = len(profile[0])  # should be equal to number of candidates
        preferred_counts = np.zeros((m, m), dtype=np.int64)
        iterate_over = profile
        for ballot in iterate_over:
            order = ballot
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    preferred_counts[order[i], order[j]] += 1

        # turn full data into binary majority matrix
        for (i, j), idx in np.ndenumerate(preferred_counts):
            if i >= j:
                continue

            # if more people prefer i to j
            if preferred_counts[i, j] > n // 2:
                preferred_counts[i, j] = 1
                preferred_counts[j, i] = 0
            elif preferred_counts[j, i] > n // 2:
                preferred_counts[j, i] = 1
                preferred_counts[i, j] = 0
            # elif preferred_counts[i, j] > preferred_counts[j, i]:
            else:
                preferred_counts[i, j] = 0
                preferred_counts[j, i] = 0

        if remove_diagonal:
            preferred_counts = preferred_counts[~np.eye(m, dtype=bool)].reshape(preferred_counts.shape[0], -1)
            # exit("Have not yet tested whether removing main diagonals works.")
        elif upper_half_only:
            preferred_counts = preferred_counts[np.triu_indices_from(preferred_counts, k=1)]

        features.append(preferred_counts.flatten().tolist())

    return features


def rank_counts_from_profiles(profiles):
    """
    Return a flattened m by m matrix R where R[c, p] is the number of voters who put candidate c in p-th position in
    their preference order.
    :param profiles:
    :return:
    """
    m = len(profiles[0][0])  # length of first ballot in first profile

    features = []

    for profile in profiles:
        rank_counts = np.zeros((m, m), dtype=np.int64)
        iterate_over = profile
        for ballot in iterate_over:
            # for ballot in profile:
            order = ballot
            for idx, c in enumerate(order):
                rank_counts[c, idx] += 1

        features.append(rank_counts.flatten().tolist())

    return features


def normalize_array(arr):
    """
    Given a list or numpy array (e.g. a neural net output) normalize it so that the sum is equal to 1.
    :param arr:
    :return:
    """
    x = normalize(arr, norm="l1")
    # tst = np.round(x, 5)
    return np.round(x, 5)

def load_mw_voting_rules():
    rules = [
        "av",
        "pav",
        "cc",
        "lexcc",
        "seqcc",
        "monroe",
        "greedy-monroe",
        "minimaxav",
    ]

    return rules

def load_voting_rules():
    """
    Return a list of all voting rules in pref-voting library that are (probably) suitable for use.
    :return:
    """
    import pref_voting.scoring_methods as sm
    import utils.voting_utils as vut
    # funcs = inspect.getmembers(sm)
    # score_rules = [f for f in funcs if isinstance(f[1], sm.VotingMethod)]

    scoring_vms = [
        sm.plurality,
        sm.borda,
        # borda_for_profile_with_ties,
        sm.anti_plurality,
        # sm.scoring_rule,
        vut.two_approval,
        vut.three_approval
    ]

    import pref_voting.iterative_methods as im
    # funcs = inspect.getmembers(im)
    # iterative_rules = [f for f in funcs if isinstance(f[1], im.VotingMethod)]

    iterated_vms = [
        im.instant_runoff,
        # instant_runoff_tb,
        # instant_runoff_put,
        # hare,
        # ranked_choice,
        im.bottom_two_runoff_instant_runoff,
        # bottom_two_runoff_instant_runoff_put,
        im.benham,
        # benham_put,
        # benham_tb,
        im.plurality_with_runoff_put,
        im.coombs,
        # coombs_tb,
        # coombs_put,
        im.baldwin,
        # baldwin_tb,
        # baldwin_put,
        im.strict_nanson,
        im.weak_nanson,
        im.iterated_removal_cl,
        im.raynaud,
        im.tideman_alternative_smith,
        # tideman_alternative_smith_put,
        # im.tideman_alternative_schwartz,
        # tideman_alternative_schwartz_put,
        im.knockout
    ]

    import pref_voting.c1_methods as c1
    # funcs = inspect.getmembers(c1)
    # c1_rules = [f for f in funcs if isinstance(f[1], c1.VotingMethod)]

    c1_vms = [
        c1.banks,
        c1.condorcet,
        c1.copeland,
        c1.llull,
        c1.uc_gill,
        c1.uc_fish,
        c1.uc_bordes,
        c1.uc_mckelvey,
        c1.slater,
        c1.top_cycle,
        c1.gocha,
        c1.bipartisan
    ]

    import pref_voting.margin_based_methods as mm
    # funcs = inspect.getmembers(mm)
    # margin_rules = [f for f in funcs if isinstance(f[1], mm.VotingMethod)]

    mg_vms = [
        mm.minimax,
        mm.split_cycle,
        # split_cycle_Floyd_Warshall,
        # beat_path,
        # mm.beat_path_Floyd_Warshall,
        # ranked_pairs,
        # ranked_pairs_with_test,
        mm.ranked_pairs_zt,
        mm.ranked_pairs_tb,
        # river,
        # river_with_test,
        mm.simple_stable_voting,
        # simple_stable_voting_faster,
        mm.stable_voting,
        # stable_voting_faster,
        mm.loss_trimmer
    ]

    import pref_voting.combined_methods as cm
    # funcs = inspect.getmembers(cm)
    # combined_rules = [f for f in funcs if isinstance(f[1], cm.VotingMethod)]

    combined_vms = [
        cm.daunou,
        cm.blacks,
        cm.condorcet_irv,
        cm.condorcet_irv_put,
        cm.smith_irv,
        # smith_irv_put,
        cm.smith_minimax,
        cm.condorcet_plurality,
        cm.copeland_local_borda,
        cm.copeland_global_borda,
        cm.borda_minimax_faceoff
    ]

    import pref_voting.other_methods as om
    # funcs = inspect.getmembers(cm)
    # other_rules = [f for f in funcs if isinstance(f[1], om.VotingMethod)]

    other_vms = [
        # om.kemeny_young,
        # om.majority,
        om.bucklin,
        om.simplified_bucklin,
        om.weighted_bucklin,
        om.bracket_voting,
        om.superior_voting
    ]

    # import pref_voting.grade_methods as gm
    # funcs = inspect.getmembers(gm)
    # grade_rules = [f for f in funcs if isinstance(f[1], gm.VotingMethod)]

    # grade_vms = [
    #     gm.score_voting,
    #     gm.approval,
    #     gm.star,
    #     gm.cumulative_voting,
    #     gm.majority_judgement
    # ]

    all_rules = scoring_vms + iterated_vms + c1_vms + mg_vms + combined_vms + other_vms

    # all_rules = score_rules + iterative_rules + c1_rules + margin_rules + combined_rules + other_rules + grade_rules
    # for r in all_rules:
    #     print(r[1])

    return all_rules


def generate_winners(rule, profiles, num_winners, num_candidates):
    """
    Determine the winning candidates for the given rule and profile.
    :param rule:
    :param profiles:
    :return:
    """
    #if isinstance(rule, str):
    #    rule = abcrules.get_rule(rule)
    #    if rule is None:
    #        return [], []

    if abcrules.get_rule(rule) is None:
        return [], []

    winners = []
    tied_winners = []
    for profile in profiles:
        if isinstance(profile, list) or isinstance(profile, np.ndarray):
            profile = pref_voting_profiles.Profile(profile)
        ws = abcrules.compute(rule, profile, committeesize=num_winners)

        winningcommittees = []

        for committee in ws:
            committee_array = np.zeros(num_candidates, dtype=int)
            for candidate in committee:
                committee_array[candidate-1] = 1
            winningcommittees.append(tuple(committee_array.tolist()))

        tied_winners.append(winningcommittees)
        winners.append(min(winningcommittees))
    return winners, tied_winners


def get_rule_by_name(rule_name):
    """

    :param rule_name:
    :return:
    """
    vr = load_voting_rules()
    for rule in vr:
        if rule.name.casefold() == rule_name.casefold():
            return rule
    return None
