import numpy as np
from sklearn.preprocessing import normalize
from pref_voting import profiles as pref_voting_profiles
from abcvoting import abcrules
import itertools
from . import axiom_eval as ae
import numpy as np
import os
import pandas as pd
import math


def load_data(size, n, m, num_winners, pref_dist, train, base_data_folder="data", make_data_if_needed=True):
    """

    :return:
    """
    if train:
        filename = f"n_profiles={size}-num_voters={n}-m={m}-committee_size={num_winners}-pref_dist={pref_dist}-TRAIN.csv"
    else:
        filename = f"n_profiles={size}-num_voters={n}-m={m}-committee_size={num_winners}-pref_dist={pref_dist}-TEST.csv"

    filepath = os.path.join(base_data_folder, filename)

    if not os.path.exists(filepath):
        if make_data_if_needed:
            print(f"Tried loading path but it does not exist: {filepath}")
            print("Creating data now.")
            from generate_data import make_one_multi_winner_dataset
            make_one_multi_winner_dataset(m, size, n, pref_dist, num_winners, train=train)
        else:
            print(f"Tried loading path but it does not exist: {filepath}")
            print("Model was told not to create the data if it did not exist.")

    df = pd.read_csv(filepath)
    return df


def save_evaluation_results(base_cols, all_axioms, violation_counts, filename):
    header = base_cols + all_axioms + ["total_violation"]
    rows = []
    for base_vals, counts in violation_counts.items():
        mean_count = sum([pair[0] for pair in counts])
        row = list(base_vals) + counts + [mean_count]
        rows.append(row)

    df = pd.DataFrame(data=rows, columns=header, index=None)
    df.to_csv(filename, index=False)


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
    # pair_str = [str(w) for w in cps]
    features_dict[f"candidate_pairs"] = cps
    normalized = normalize_array(cps)[0].tolist()
    # pair_str = [str(w) for w in normalized]
    features_dict[f"candidate_pairs-normalized"] = normalized

    cps = candidate_pairs_from_profiles(profiles, remove_diagonal=True)
    # pair_str = [str(w) for w in cps]
    features_dict[f"candidate_pairs-no_diagonal"] = cps
    normalized = normalize_array(cps)[0].tolist()
    # pair_str = [str(w) for w in normalized]
    features_dict[f"candidate_pairs-normalized-no_diagonal"] = normalized

    cps = candidate_pairs_from_profiles(profiles, upper_half_only=True)
    normalized = normalize_array(cps)[0]
    # pair_str = [str(w) for w in normalized]
    features_dict[f"candidate_pairs-normalized-upper_half"] = normalized

    # add binary candidate pairs
    bps = binary_candidate_pairs_from_profiles(profiles)
    # pair_str = [str(w) for w in bps]
    features_dict[f"binary_pairs"] = bps

    bps = binary_candidate_pairs_from_profiles(profiles, remove_diagonal=True)
    # pair_str = [str(w) for w in bps]
    features_dict[f"binary_pairs-no_diagonal"] = bps

    bps = binary_candidate_pairs_from_profiles(profiles, upper_half_only=True)
    # pair_str = [str(w) for w in bps]
    features_dict[f"binary_pairs-upper_half"] = bps

    # add rank matrices
    ranks = rank_counts_from_profiles(profiles)
    # pair_str = [str(w) for w in ranks]
    features_dict[f"rank_matrix"] = ranks
    normalized = normalize_array(ranks)[0].tolist()
    # pair_str = [str(w) for w in normalized]
    features_dict[f"rank_matrix-normalized"] = normalized

    if df is not None:
        for key, val in features_dict.items():
            df[key] = val
    return features_dict


def candidate_pairs_from_profiles(profile, remove_diagonal=False, upper_half_only=False):
    """
    Return a list where, for each profile, a flattened m*m list is returned. list[i][j] is the number of times
    alternative i is preferred to alternative j
    :param profiles: list of individual preferences orders/ballots
    :param upper_half_only:
    :param remove_diagonal:
    :return:
    """
    # print("Profile in candidate pairs from profiles:", profile)
    # raw_profiles = [[[0, 1, 2, 3], [1, 0, 2, 3], [0, 1, 3, 2], [3,2,1,0]]]
    m = len(profile[0])  # length of first ballot in first profile
    # features = []

    # for profile in profiles:
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

    return preferred_counts.flatten().tolist()

    # return features


def binary_candidate_pairs_from_profiles(profile, remove_diagonal=False, upper_half_only=False):
    """
    Return a list where, for each profile, a flattened m*m list is returned. list[i][j] is 1 iff more voters prefer
    i to j and 0 otherwise. Note: Ties are represented as 0 values.
    :param profiles: list of individual preferences orders/ballots
    :param remove_diagonal:
    :param upper_half_only:
    :return:
    """

    features = []

    # for profile in profiles:
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

    return preferred_counts.flatten().tolist()

    # return features


def rank_counts_from_profiles(profile):
    """
    Return a flattened m by m matrix R where R[c, p] is the number of voters who put candidate c in p-th position in
    their preference order.
    :param profiles:
    :return:
    """
    m = len(profile[0])  # length of first ballot in first profile

    features = []

    # for profile in profiles:
    rank_counts = np.zeros((m, m), dtype=np.int64)
    iterate_over = profile
    for ballot in iterate_over:
        # for ballot in profile:
        order = ballot
        for idx, c in enumerate(order):
            rank_counts[c, idx] += 1

    return rank_counts.flatten().tolist()

    # return features


def normalize_array(arr):
    """
    Given a list or numpy array (e.g. a neural net output) normalize it so that the sum is equal to 1.
    :param arr:
    :return:
    """
    arr = np.array(arr)
    arr = arr.reshape(1, -1)
    x = normalize(arr, norm="l1")
    # tst = np.round(x, 5)
    return np.round(x, 5)


def load_mw_voting_rules():
    abc_rules = [
        "av",
        "pav",
        "cc",
        "lexcc",
        "seqcc",
        "monroe",
        "greedy-monroe",
        "minimaxav",
    ]

    import pref_voting.scoring_methods as sm

    scoring_vms = [
        sm.borda_ranking,
        sm.plurality_ranking
    ]

    return scoring_vms + abc_rules


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

    if isinstance(rule, str):
        if abcrules.get_rule(rule) is None:
            return [], []

    winners = []
    tied_winners = []
    for profile in profiles:
        # if isinstance(profile, list) or isinstance(profile, np.ndarray):
        #    profile = pref_voting_profiles.Profile(profile)
        try:
            ws = abcrules.compute(rule, profile, committeesize=num_winners)
        except Exception as ex1:
            try:
                ws = rule(profile, tie_breaking="alphabetic")
                ws = np.array([ws])
            except Exception as ex2:
                print("Error computing rule")
                print(ex1)
                print(ex2)
                return [], []

        winningcommittees = []

        for committee in ws:

            committee_array = np.zeros(num_candidates, dtype=int)
            try:
                for i in range(num_winners):
                    candidate = committee[i]
                    committee_array[candidate] = 1
            except Exception as ex1:
                for candidate in committee:
                    committee_array[candidate] = 1
                winningcommittees.append(tuple(committee_array.tolist()))

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


def generate_all_committees(num_candidates, num_winners):
    """
    Find all possible winning committees
    :param num_candidates: Number of candidates
    :param committee_size: Number of winners in the winning committee
    :return: All possible combinations of winners
    """

    committees = list(itertools.combinations(range(num_candidates), num_winners))
    resulting_committees = []

    for committee in committees:
        binary_committee = np.zeros(num_candidates, dtype=int)

        for i in range(num_winners):
            binary_committee[committee[i]] = 1
        resulting_committees.append(binary_committee)

    return resulting_committees


def findWinners(profile, num_winners):
    """
    Find committees with the least amount of violations
    :param profile: A voting profile
    :param num_winners: The number of winners
    :return: The committees with the least number of violations
    """

    winning_committees = []
    all_committees = generate_all_committees(len(profile[0]), num_winners)
    rank_choice = rank_counts_from_profiles(profile)
    cand_pairs = candidate_pairs_from_profiles(profile)

    does_condorcet_exist = ae.exists_condorcet_winner(all_committees, cand_pairs)

    min_violations = float('inf')

    n_voters = len(profile)

    for committee in all_committees:
        violations = ae.eval_majority_axiom(n_voters, committee, rank_choice)
        violations += ae.eval_majority_loser_axiom(n_voters, committee, rank_choice)
        if does_condorcet_exist:
            violations += ae.eval_condorcet_winner(committee, cand_pairs)
        violations += ae.eval_condorcet_loser(committee, cand_pairs)
        violations += ae.eval_dummetts_condition(committee, n_voters, num_winners, profile)
        violations += ae.eval_solid_coalitions(committee, n_voters, num_winners, rank_choice)
        violations += ae.eval_consensus_committee(committee, n_voters, num_winners, profile, rank_choice)
        violations += ae.eval_strong_unanimity(committee, num_winners, profile)

        if violations < min_violations:
            min_violations = violations
            winning_committees = [committee]
        elif violations == min_violations:
            winning_committees.append(committee)
        

    return winning_committees, min_violations


def eval_all_axioms(n_voters, rank_choice, cand_pairs, committees, num_winners, profile):
    violations = {
        "total_violations": 0,
        "majority": 0,
        "majority_loser": 0,
        "condorcet_winner": 0,
        "condorcet_loser": 0,
        "dummetts_condition": 0,
        "solid_coalitions": 0,
        "consensus_committee": 0,
        "unanimity": 0,
        "local_stability": 0,
        "count_viols": 0,
    }

    for rank_choice_m, cand_pair, committee, prof in zip(rank_choice, cand_pairs, committees, profile):
        prof = eval(prof)
        does_condorcet_exist = ae.exists_condorcet_winner(
            generate_all_committees(len(committees[0]), sum(committees[0])), cand_pair)
        
        violations["majority"] += ae.eval_majority_axiom(n_voters, committee, eval(rank_choice_m))
        violations["majority_loser"] += ae.eval_majority_loser_axiom(n_voters, committee, eval(rank_choice_m))
        if does_condorcet_exist:
            violations["condorcet_winner"] += ae.eval_condorcet_winner(committee, eval(cand_pair))
        violations["condorcet_loser"] += ae.eval_condorcet_loser(committee, eval(cand_pair))   
        violations["dummetts_condition"] += ae.eval_dummetts_condition(committee, n_voters, num_winners, prof)
        violations["solid_coalitions"] += ae.eval_solid_coalitions(committee, n_voters, num_winners, eval(rank_choice_m))
        violations["consensus_committee"] += ae.eval_consensus_committee(committee, n_voters, num_winners, prof, eval(rank_choice_m))
        violations["unanimity"] += ae.eval_strong_unanimity(committee, num_winners, prof)
        violations["local_stability"] += ae.eval_local_stability(committee, prof, n_voters, math.ceil(n_voters / num_winners))

        if num_winners != sum(committee):
            violations["count_viols"] += 1
        
        total_sum = sum(value for key, value in violations.items() if key != "total_violations")
        violations["total_violations"] = total_sum


    return violations


def kwargs_from_pref_models(pref_model):
    model_string = pref_model
    arg_dict = {}
    if "__args__" in pref_model:
        arg_dict = {}
        model_string = pref_model[:pref_model.index("__args__")]
        arg_string = pref_model[pref_model.index("__args__") + len("__args__"):]
        # assume args are split by a single underscore
        args = arg_string.split("_")
        for arg in args:
            pair = arg.split("=")
            key, value = pair[0], pair[1]
            try:
                arg_dict[key] = eval(value.replace('-', '_'))
            except NameError:
                # simplest way to see if the argument should be a string or not
                arg_dict[key] = value.replace('-', '_')
    return model_string, arg_dict
