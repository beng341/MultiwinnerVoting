import numpy as np
import pandas.errors
from abcvoting.preferences import Profile
from pref_voting.voting_method import VotingMethod
from sklearn.preprocessing import normalize
from pref_voting import profiles as pref_voting_profiles
from abcvoting import abcrules
from abcvoting import properties as abc_prop
import utils.voting_utils as vut
import itertools
from . import axiom_eval as ae
# import axiom_eval as ae
import numpy as np
import os
import pandas as pd
import math


def load_data(size, n, m, num_winners, pref_dist, axioms, train, base_data_folder="data", make_data_if_needed=True):
    """

    :return:
    """
    if train:
        filename = f"n_profiles={size}-num_voters={n}-m={m}-committee_size={num_winners}-pref_dist={pref_dist}-axioms={axioms}-TRAIN.csv"
    else:
        filename = f"n_profiles={size}-num_voters={n}-m={m}-committee_size={num_winners}-pref_dist={pref_dist}-axioms={axioms}-TEST.csv"

    filepath = os.path.join(base_data_folder, filename)

    if not os.path.exists(filepath):
        if make_data_if_needed:
            print(f"Tried loading path but it does not exist: {filepath}")
            print("Creating data now.")
            from network_ops.generate_data import make_one_multi_winner_dataset

            args = {
                "n_profiles": size,
                "prefs_per_profile": n,
                "m": m,
                "num_winners": num_winners,
                "pref_model": pref_dist,
                "axioms": axioms,
                "out_folder": "data"
            }
            make_one_multi_winner_dataset(args)
        else:
            print(f"Tried loading path but it does not exist: {filepath}")
            print("Model was told not to create the data if it did not exist.")
    if os.path.exists(filepath):
        # If it was just created, this should now be true despite previously being false.
        df = pd.read_csv(filepath)
    else:
        df = None
    return df


def load_with_duplicate_lines(filepath):
    """
    Load the given csv file. If it throws an error on some line (as is currently happening) replace that row with
    a duplicate line from the row above it.
    :param filepath:
    :return:
    """
    try:
        # Attempt to load the file normally
        df = pd.read_csv(filepath)
        return df

    except pd.errors.ParserError as e:
        # Parse the error message to identify the faulty line
        error_message = str(e)
        error_line = int(error_message.split("line ")[-1].split()[0])

        # Load the file line by line
        with open(filepath, "r") as file:
            lines = file.readlines()

        # Replace the problematic line with the line above it
        if error_line > 1:  # Ensure there's a previous line to copy from
            lines[error_line - 1] = lines[error_line - 2]

        # Reload the modified data as a DataFrame
        from io import StringIO
        modified_data = StringIO("".join(lines))
        df = pd.read_csv(modified_data)

        return df


def generate_mixed_distribution(distributions, total_size, n, m, num_winners, axioms, dist_name="mixed",
                                data_folder="data"):
    """
    Combine train/test data from several distributions into a single mixed file with an equal amount of data from
    each individual distribution. In principle can be used to merge any given distributions but is likely to only
    be used to combine all distributions at once.
    :param distributions: list of strings containing name of each distribution to be joined
    :param total_size: total size of resulting dataset. Sum of all smaller dataset sizes must be larger than this.
    (however that requirement is not enforced logically)
    :param n:
    :param m:
    :param num_winners:
    :param axioms:
    :param dist_name:
    :param data_folder:
    :return:
    """
    train_dfs = []
    test_dfs = []

    # take slightly more data than needed so we have enough to remove some and end up with the correct amount
    size_per_dist = total_size  # math.ceil(total_size / len(distributions))

    train_file = f"n_profiles={total_size}-num_voters={n}-m={m}-committee_size={num_winners}-pref_dist={dist_name}-axioms={axioms}-TRAIN.csv"
    test_file = f"n_profiles={total_size}-num_voters={n}-m={m}-committee_size={num_winners}-pref_dist={dist_name}-axioms={axioms}-TEST.csv"

    # fn = os.path.join(data_folder, train_file)
    # if os.path.exists(fn):
    #     print(f"train file exists already:{fn}")
    #     return
    # fn = os.path.join(data_folder, test_file)
    # if os.path.exists(fn):
    #     print(f"test file exists already:{fn}")
    #     return

    # for subdist in distributions:
    for subdist in distributions:

        # try:
        print(f"Loading data from: {subdist} (train); m={m}, k={k}")
        train_dfs.append(load_data(size=size_per_dist,
                                   n=n,
                                   m=m,
                                   num_winners=num_winners,
                                   pref_dist=subdist,
                                   axioms=axioms,
                                   base_data_folder=data_folder,
                                   train=True)
                         )
        # except pandas.errors.ParserError as pe:
        #     print(f"Loading data from: {subdist} (train); m={m}, k={k}")
        #     print(f"Error is: {pe}")

        # try:
        print(f"Loading data from: {subdist} (test); m={m}, k={k}")
        test_dfs.append(load_data(size=size_per_dist,
                                  n=n,
                                  m=m,
                                  num_winners=num_winners,
                                  pref_dist=subdist,
                                  axioms=axioms,
                                  base_data_folder=data_folder,
                                  train=False)
                        )
        # except pandas.errors.ParserError as pe:
        #     print(f"Loading data from: {subdist} (test); m={m}, k={k}")
        #     print(f"Error is: {pe}")

    print("Loaded all data")
    mixed_train = pd.concat(train_dfs, axis=0).reset_index(drop=True)
    mixed_test = pd.concat(test_dfs, axis=0).reset_index(drop=True)

    shuffled_train = mixed_train.sample(n=total_size).reset_index(drop=True)
    shuffled_test = mixed_test.sample(n=total_size).reset_index(drop=True)

    filepath = os.path.join(data_folder, train_file)
    print(f"About to save train file to {filepath}")
    shuffled_train.to_csv(filepath, index=False)

    filepath = os.path.join(data_folder, test_file)
    print(f"About to save test file to {filepath}")
    shuffled_test.to_csv(filepath, index=False)


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

    # # add candidate pairs
    cps = candidate_pairs_from_profiles(profiles)
    # # pair_str = [str(w) for w in cps]
    features_dict[f"candidate_pairs"] = cps
    # normalized = normalize_array(cps)[0].tolist()
    # # pair_str = [str(w) for w in normalized]
    # features_dict[f"candidate_pairs-normalized"] = normalized

    cps = candidate_pairs_from_profiles(profiles, remove_diagonal=True)
    # # pair_str = [str(w) for w in cps]
    # # features_dict[f"candidate_pairs-no_diagonal"] = cps
    normalized = normalize_array(cps)[0].tolist()
    # # pair_str = [str(w) for w in normalized]
    features_dict[f"candidate_pairs-normalized-no_diagonal"] = normalized

    # cps = candidate_pairs_from_profiles(profiles, upper_half_only=True)
    # normalized = normalize_array(cps)[0].tolist()
    # # pair_str = [str(w) for w in normalized]
    # features_dict[f"candidate_pairs-normalized-upper_half"] = normalized

    # # add binary candidate pairs
    # bps = binary_candidate_pairs_from_profiles(profiles)
    # # pair_str = [str(w) for w in bps]
    # features_dict[f"binary_pairs"] = bps

    bps = binary_candidate_pairs_from_profiles(profiles, remove_diagonal=True)
    # pair_str = [str(w) for w in bps]
    features_dict[f"binary_pairs-no_diagonal"] = bps

    # bps = binary_candidate_pairs_from_profiles(profiles, upper_half_only=True)
    # # pair_str = [str(w) for w in bps]
    # features_dict[f"binary_pairs-upper_half"] = bps

    # add rank matrices
    ranks = rank_counts_from_profiles(profiles)
    # pair_str = [str(w) for w in candidate_pairs]
    features_dict[f"rank_matrix"] = ranks
    normalized = normalize_array(ranks)[0].tolist()
    # # pair_str = [str(w) for w in normalized]
    features_dict[f"rank_matrix-normalized"] = normalized

    if df is not None:
        for key, val in features_dict.items():
            df[key] = val
    return features_dict


def candidate_pairs_from_profiles(profile, remove_diagonal=False, upper_half_only=False):
    """
    Return a list where, for each profiles, a flattened m*m list is returned. list[i][j] is the number of times
    alternative i is preferred to alternative j
    :param profiles: list of individual preferences orders/ballots
    :param upper_half_only:
    :param remove_diagonal:
    :return:
    """
    # print("Profile in candidate pairs from profiles:", profiles)
    # raw_profiles = [[[0, 1, 2, 3], [1, 0, 2, 3], [0, 1, 3, 2], [3,2,1,0]]]
    m = len(profile[0])  # length of first ballot in first profiles
    # features = []

    # for profiles in profiles:
    preferred_counts = np.zeros((m, m), dtype=np.int64)
    iterate_over = profile
    for ballot in iterate_over:
        # for ballot in profiles:
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
    Return a list where, for each profiles, a flattened m*m list is returned. list[i][j] is 1 iff more voters prefer
    i to j and 0 otherwise. Note: Ties are represented as 0 values.
    :param profiles: list of individual preferences orders/ballots
    :param remove_diagonal:
    :param upper_half_only:
    :return:
    """

    features = []

    # for profiles in profiles:
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
    m = len(profile[0])  # length of first ballot in first profiles

    features = []

    # for profiles in profiles:
    rank_counts = np.zeros((m, m), dtype=np.int64)
    iterate_over = profile
    for ballot in iterate_over:
        # for ballot in profiles:
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


def load_mw_voting_rules(k=None):
    abc_rules = [
        "av",
        "pav",
        "cc",
        "lexcc",
        "seqcc",
        "monroe",
        "greedy-monroe",
        "minimaxav",
        "equal-shares",
        "eph",
        "rsd"
    ]

    import pref_voting.scoring_methods as sm

    vms = [
        sm.borda_ranking,
        sm.plurality_ranking,
        vut.stv
    ]
    rules = vms + abc_rules
    if k == 1:
        rules += load_sw_voting_rules()

    return rules


def load_sw_voting_rules():
    """
    Return a list of all voting rules in pref-voting library that are (probably) suitable for use.
    :return:
    """
    import pref_voting.scoring_methods as sm
    import utils.voting_utils as vut
    # funcs = inspect.getmembers(sm)
    # score_rules = [f for f in funcs if isinstance(f[1], sm.VotingMethod)]

    scoring_vms = [
        # sm.plurality,
        # sm.borda,
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


def generate_winners(rule, profiles, num_winners, num_candidates, abc_rule=True):
    """
    Determine the winning candidates for the given rule and profiles.
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
        # if isinstance(profiles, list) or isinstance(profiles, np.ndarray):
        #    profiles = pref_voting_profiles.Profile(profiles)
        if isinstance(rule, str):
            ws = abcrules.compute(rule, profile, committeesize=num_winners)
            if len(ws) > 1:
                pass
        elif rule.name == "Plurality ranking":
            scores = profile.plurality_scores()
            scores = [scores[i] for i in range(len(scores))]
            sorted_scores = -np.array(scores)
            sorted_scores = sorted_scores.argsort()
            ws = np.array([sorted_scores[:num_winners]])
        elif rule.name == "Borda ranking":
            scores = profile.borda_scores()
            scores = [scores[i] for i in range(len(scores))]
            sorted_scores = -np.array(scores)
            sorted_scores = sorted_scores.argsort()
            ws = np.array([sorted_scores[:num_winners]])
        elif rule.name == "STV":
            ws = np.array([rule(profile, k=num_winners)])
        elif isinstance(rule, VotingMethod) and num_winners == 1:
            ws = np.array([rule(profile)])
        else:
            ws = rule(profile, tie_breaking="alphabetic")
            ws = np.array([ws])
        # try:
        #     ws = abcrules.compute(rule, profile, committeesize=num_winners)
        # except Exception as ex1:
        #     try:
        #         print(f"Exception in generate_winners with {rule}")
        #         ws = rule(profile, tie_breaking="alphabetic")
        #         ws = np.array([ws])
        #     except Exception as ex2:
        #         print("Error computing rule")
        #         print(ex1)
        #         print(ex2)
        #         return [], []

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
        tied_winners.append(winningcommittees)
        winners.append(min(winningcommittees))
    return winners, tied_winners


def get_rule_by_name(rule_name):
    """

    :param rule_name:
    :return:
    """
    vr = load_sw_voting_rules()
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


def find_winners(profile, n_winners, axioms_to_evaluate="all"):
    """
    Find committees with the least amount of violations
    :param profile: A single profile containing rankings of each voter.
    :param n_winners: The number of winners
    :param axioms_to_evaluate: "all" or a list of individual axiom names which should be considered
    :return: The committees with the least number of violations
    """

    # all_axioms = [
    #     "dummett",
    #     "consensus",
    #     "fixed_majority",
    #     "majority_winner",
    #     "majority_loser",
    #     "condorcet_winner",
    #     "condorcet_loser",
    #     "solid_coalition",
    #     "strong_unanimity",
    #     "local_stability",
    #     "strong_pareto"
    # ]

    if axioms_to_evaluate == ["all"]:
        axioms_to_evaluate = ae.all_axioms

    m = len(profile[0])
    n_voters = len(profile)

    all_committees = generate_all_committees(m, n_winners)
    rank_choice = rank_counts_from_profiles(profile)
    cand_pairs = candidate_pairs_from_profiles(profile)
    abc_profile = abc_profile_from_rankings(m=m, k=n_winners, rankings=profile)

    does_condorcet_exist = ae.exists_condorcet_winner(all_committees, cand_pairs)

    min_violation_committees = []
    max_violation_committees = []
    min_violations = float('inf')
    max_violations = 0

    if "dummett" in axioms_to_evaluate:
        # print("Evaluating dummets")
        # Find committees able to satisfy Dummett's condition on this profiles
        dummett_winners = ae.find_dummett_winners(num_voters=n_voters, num_winners=n_winners, profile=profile)

    if "consensus" in axioms_to_evaluate:
        # print("Evaluating consensus")
        # Find committees able to satisfy the consensus axiom on this profiles
        consensus_committees = ae.find_consensus_committees(num_voters=n_voters, num_winners=n_winners, profile=profile)

    if "fixed_majority" in axioms_to_evaluate:
        # print("Evaluating fixed majority")
        fm_winner = ae.fixed_majority_required_winner(n_winners=n_winners,
                                                      n_alternatives=len(profile[0]),
                                                      candidate_pairs=cand_pairs,
                                                      profile=profile)

    for committee in all_committees:
        violations = 0

        if "majority_winner" in axioms_to_evaluate:
            # print("Evaluating maj win")
            violations += ae.eval_majority_axiom(n_voters, committee, rank_choice)
        if "majority_loser" in axioms_to_evaluate:
            # print("Evaluating maj los")
            violations += ae.eval_majority_loser_axiom(n_voters, committee, rank_choice)
        if "fixed_majority" in axioms_to_evaluate:
            # print("Evaluating fix maj")
            violations += ae.eval_fixed_majority_axiom(committee=committee,
                                                       required_winning_committee=fm_winner)
        if "condorcet_winner" in axioms_to_evaluate:
            # print("Evaluating cond win")
            if does_condorcet_exist:
                violations += ae.eval_condorcet_winner(committee, cand_pairs)
        if "condorcet_loser" in axioms_to_evaluate:
            # print("Evaluating cond los")
            violations += ae.eval_condorcet_loser(committee, cand_pairs)
        if "dummett" in axioms_to_evaluate:
            # print("Evaluating dummets")
            violations += ae.eval_dummetts_condition(committee, n_voters, n_winners, profile, dummett_winners)
        if "solid_coalition" in axioms_to_evaluate:
            # print("Evaluating solid")
            violations += ae.eval_solid_coalitions(committee, n_voters, n_winners, rank_choice)
        if "consensus" in axioms_to_evaluate:
            # print("Evaluating consen")
            violations += ae.eval_consensus_committee(committee, n_voters, n_winners, profile, rank_choice,
                                                      consensus_committees)
        if "strong_unanimity" in axioms_to_evaluate:
            # print("Evaluating str unan")
            violations += ae.eval_strong_unanimity(committee, n_winners, profile)

        if "strong_pareto" in axioms_to_evaluate:
            violations += ae.eval_strong_pareto_efficiency(committee, profile)

        # get winners in ABC friendly format
        winners = winner_set_from_khot_committee(committee)
        if "jr" in axioms_to_evaluate:
            violations += not abc_prop.check_JR(profile=abc_profile,
                                                committee=winners)
        if "ejr" in axioms_to_evaluate:
            violations += not abc_prop.check_EJR(profile=abc_profile,
                                                 committee=winners)
        if "ejr_plus" in axioms_to_evaluate:
            violations += not abc_prop.check_EJR_plus(profile=abc_profile,
                                                      committee=winners)
        if "pjr" in axioms_to_evaluate:
            violations += not abc_prop.check_PJR(profile=abc_profile,
                                                 committee=winners)
        if "fjr" in axioms_to_evaluate:
            violations += not abc_prop.check_FJR(profile=abc_profile,
                                                 committee=winners)
        if "core" in axioms_to_evaluate:
            violations += not abc_prop.check_core(profile=abc_profile,
                                                  committee=winners)

        if violations < min_violations:
            min_violations = violations
            min_violation_committees = [committee]
        elif violations == min_violations:
            min_violation_committees.append(committee)

        if violations > max_violations:
            max_violations = violations
            max_violation_committees = [committee]
        elif violations == max_violations:
            max_violation_committees.append(committee)

    return min_violation_committees, min_violations, max_violation_committees, max_violations


def eval_all_axioms(n_voters, rank_choice, cand_pairs, committees, n_winners, profiles):
    violations = {
        "majority": [],
        "majority_loser": [],
        "fixed_majority": [],
        "condorcet_winner": [],
        "condorcet_loser": [],
        "dummetts_condition": [],
        "solid_coalitions": [],
        # "consensus_committee": [],
        "strong_unanimity": [],
        "local_stability": [],
        "strong_pareto_efficiency": [],
        "jr": [],
        "ejr": [],
        "core": [],
    }

    for rank_choice_m, cand_pair, committee, prof in zip(rank_choice, cand_pairs, committees, profiles):
        prof = eval(prof)
        cand_pair = eval(cand_pair)
        rank_choice_m = eval(rank_choice_m)
        abc_profile = abc_profile_from_rankings(m=len(prof[0]), k=n_winners, rankings=prof)

        # Find committees able to satisfy Dummett's condition on this profile
        dummet_winners = ae.find_dummett_winners(num_voters=n_voters, num_winners=n_winners, profile=prof)

        # Find committees able to satisfy the consensus axiom on this profile
        consensus_committees = ae.find_consensus_committees(num_voters=n_voters, num_winners=n_winners, profile=prof)

        does_condorcet_exist = ae.exists_condorcet_winner(
            generate_all_committees(len(committees[0]), sum(committees[0])), cand_pair)

        fm_winner = ae.fixed_majority_required_winner(n_winners=n_winners,
                                                      n_alternatives=len(committee),
                                                      candidate_pairs=cand_pair,
                                                      profile=prof)
        fm_satisfied = ae.eval_fixed_majority_axiom(committee=committee,
                                                    required_winning_committee=fm_winner)
        violations["fixed_majority"].append(fm_satisfied)

        violations["majority"].append(ae.eval_majority_axiom(n_voters, committee, rank_choice_m))
        violations["majority_loser"].append(ae.eval_majority_loser_axiom(n_voters, committee, rank_choice_m))
        if does_condorcet_exist:
            violations["condorcet_winner"].append(ae.eval_condorcet_winner(committee, cand_pair))
        else:
            violations["condorcet_winner"].append(0)
        violations["condorcet_loser"].append(ae.eval_condorcet_loser(committee, cand_pair))
        violations["dummetts_condition"].append(ae.eval_dummetts_condition(committee,
                                                                           n_voters,
                                                                           n_winners,
                                                                           prof,
                                                                           required_winners=dummet_winners))
        violations["solid_coalitions"].append(ae.eval_solid_coalitions(committee, n_voters, n_winners,
                                                                       rank_choice_m))
        # violations["consensus_committee"].append(ae.eval_consensus_committee(committee,
        #                                                                      n_voters,
        #                                                                      n_winners,
        #                                                                      prof,
        #                                                                      rank_choice_m,
        #                                                                      consensus_committees=consensus_committees))
        violations["strong_unanimity"].append(ae.eval_strong_unanimity(committee, n_winners, prof))
        violations["local_stability"].append(ae.eval_local_stability(committee, prof, n_voters,
                                                                     math.ceil(n_voters / n_winners)))
        violations["strong_pareto_efficiency"].append(ae.eval_strong_pareto_efficiency(committee, prof))

        # get winners in ABC friendly format
        winners = winner_set_from_khot_committee(committee)
        # if "jr" in axioms_to_evaluate:
        violations["jr"].append(int(not abc_prop.check_JR(profile=abc_profile,
                                                          committee=winners)))
        # if "ejr" in axioms_to_evaluate:
        violations["ejr"].append(int(not abc_prop.check_EJR(profile=abc_profile,
                                                            committee=winners)))
        # if "core" in axioms_to_evaluate:
        violations["core"].append(int(not abc_prop.check_core(profile=abc_profile,
                                                              committee=winners)))
        # # if "ejr_plus" in axioms_to_evaluate:
        # violations["ejr_plus"].append(abc_prop.check_EJR_plus(profile=abc_profile,
        #                                                       committee=winners))
        # # if "pjr" in axioms_to_evaluate:
        # violations["pjr"].append(abc_prop.check_PJR(profile=abc_profile,
        #                                             committee=winners))
        # # if "fjr" in axioms_to_evaluate:
        # violations["fjr"].append(abc_prop.check_FJR(profile=abc_profile,
        #                                             committee=winners))

    return violations


def kwargs_from_pref_models(pref_model):
    model_string = pref_model
    arg_dict = {}
    if "__args__" in pref_model:
        arg_dict = {}
        model_string = pref_model[:pref_model.index("__args__")]
        arg_string = pref_model[pref_model.index("__args__") + len("__args__"):]
        # assume args are split by a single underscore
        args = arg_string.split("_-_")
        for arg in args:
            pair = arg.split("=")
            key, value = pair[0], pair[1]
            try:
                arg_dict[key] = eval(value.replace('-', '_'))
            except NameError:
                # simplest way to see if the argument should be a string or not
                arg_dict[key] = value.replace('-', '_')
    return model_string, arg_dict


def load_evaluation_results_df(path, metric="std", include_noise=True):
    """
    Load results from evaluation file stored at give path. The file should have one row for each neural network.
    Aggregate the neural network rows (rows starting with "NN-") and calculate the given metric to measure their
    statistical noisiness.
    :param path:
    :param metric:
    :return:
    """
    if not os.path.exists(path):
        print(f"File doesn't exist. Skipping these datapoints: {path}")
        return None
    df = pd.read_csv(path)

    # merge individual network rows into a single row
    nn_rows = df[df['Method'].str.startswith('NN-')]
    nn_rows = nn_rows.drop(columns='Method')
    nn_mean = nn_rows.mean(numeric_only=True)

    if metric.casefold() == "std":
        nn_noisiness = nn_rows.std()
    elif metric.casefold() == "iqr":
        # Can adjust the values here to include more networks if useful (maybe 80-20?)
        nn_noisiness = nn_rows.quantile(0.75) - nn_rows.quantile(0.25)
    else:
        print(f"Trying to aggregate data using unsupported metric: {metric}")
        exit()
    nn_mean_row = pd.DataFrame([['Neural Network'] + nn_mean.tolist()], columns=df.columns)
    nn_noise_row = pd.DataFrame([['Neural Network Noise'] + nn_noisiness.tolist()], columns=df.columns)
    df = df[~df['Method'].str.startswith('NN-')]
    if include_noise:
        df = pd.concat([nn_mean_row, nn_noise_row, df], ignore_index=True)
    else:
        df = pd.concat([nn_mean_row, df], ignore_index=True)

    return df


def abc_profile_from_rankings(m, k, rankings):
    abcvoting_profile = Profile(num_cand=m)
    all_voter_approvals = [rank[:k] for rank in rankings]
    abcvoting_profile.add_voters(all_voter_approvals)
    return abcvoting_profile


def winner_set_from_khot_committee(committee):
    return [i for i in range(len(committee)) if committee[i] == 1]


if __name__ == "__main__":
    print("Generating mixed data")
    all_dists = [
        "stratification__args__weight=0.5",
        "URN-R",
        "IC",
        "IAC",
        "identity",
        "MALLOWS-RELPHI-R",
        "single_peaked_conitzer",
        "single_peaked_walsh",
        "euclidean__args__dimensions=3_-_space=gaussian_ball",
        "euclidean__args__dimensions=10_-_space=gaussian_ball",
        "euclidean__args__dimensions=3_-_space=uniform_ball",
        "euclidean__args__dimensions=10_-_space=uniform_ball",
        "euclidean__args__dimensions=3_-_space=gaussian_cube",
        "euclidean__args__dimensions=10_-_space=gaussian_cube",
        "euclidean__args__dimensions=3_-_space=uniform_cube",
        "euclidean__args__dimensions=10_-_space=uniform_cube",
    ]
    # all_errors = []
    for m, k in itertools.product([5, 6, 7], [1, 2, 3, 4, 5, 6]):
        if k >= m:
            continue
        print(f"Generating mixed data for m={m}, k={k}")
        # try:
        generate_mixed_distribution(distributions=all_dists,
                                    total_size=25000,
                                    n=50,
                                    m=m,
                                    num_winners=k,
                                    axioms="all",
                                    # data_folder="/home/b8armstr/scratch/data"
                                    data_folder="data"
                                    )
        # except pandas.errors.ParserError as e:
        #     s = f"m={m}, k={k}\n"
        #     s += f"{e}"
        #     all_errors.append(s)
        #
        # for error in all_errors:
        #     print(error)
        #     print()
        # exit()
