import numpy as np
import pref_voting
import utils.axiom_eval as ae
import utils.data_utils as du
import math
from abcvoting import properties as abc_prop
import pandas as pd
from matplotlib import pyplot as plt
from optimal_voting.OptimizableRule import PositionalScoringRule
import sys

# all_axioms = [
#     "dummett",
#     # "fixed_majority",
#     # "majority_winner",
#     # "majority_loser",
#     "condorcet_winner",
#     # "condorcet_loser",
#     # "solid_coalition",
#     # "strong_unanimity",
#     # "local_stability",
#     # "strong_pareto",
#     # "jr",
#     # "ejr",
#     # "core"
# ]


def axiom_evaluation_function(idx, winners, profile, **kwargs):
    """

    :param idx:
    :param winners:
    :param profile: A pref_voting Profile
    :param kwargs:
    :return:
    """

    if "axioms" in kwargs:
        axioms = kwargs["axioms"]
    else:
        raise ValueError("Must include list of axioms to evaluate.")

    n_winners = len(winners)
    n_alternatives = profile.num_cands
    n_voters = profile.num_voters
    if "candidate_pairs" in kwargs:
        cand_pair = kwargs["candidate_pairs"][idx]
    else:
        raise ValueError("Must include candidate pairs when initializing Optimization Rule.")
    if "rank_matrix" in kwargs:
        rank_matrix = kwargs["rank_matrix"][idx]
    else:
        raise ValueError("Must include rank matrices when initializing Optimization Rule.")
    if "abc_profiles" in kwargs:
        abc_profile = kwargs["abc_profiles"][idx]
    else:
        raise ValueError("Must include abc_profiles when initializing Optimization Rule.")

    violations = 0
    khot_winners = du.khot_committee_from_winners(winners, num_alternatives=n_alternatives)

    if "fixed_majority" in axioms:
        fm_winner = ae.fixed_majority_required_winner(n_winners=n_winners,
                                                      n_alternatives=n_alternatives,
                                                      candidate_pairs=cand_pair,
                                                      profile=profile._rankings)
        fm_satisfied = ae.eval_fixed_majority_axiom(committee=khot_winners,
                                                    required_winning_committee=fm_winner)
        violations += fm_satisfied
    if "majority_winner" in axioms:
        violations += ae.eval_majority_axiom(n_voters, khot_winners, rank_matrix)
    if "majority_loser" in axioms:
        violations += ae.eval_majority_loser_axiom(n_voters, khot_winners, rank_matrix)
    # if "condorcet_winner" in axioms:
    #     # TODO: This could probably be made faster. Sort by margin of win and only make committees with possible winners
    #     does_condorcet_exist = ae.exists_condorcet_winner(
    #         du.generate_all_committees(num_candidates=n_alternatives, num_winners=n_winners), cand_pair)
    #     if does_condorcet_exist:
    #         violations += ae.eval_condorcet_winner(khot_winners, cand_pair)
    #     else:
    #         violations += 0  # unnecessary but just for completeness/clarity
    if "condorcet_winner" in axioms:
        # TODO: This could probably be made faster. Sort by margin of win and only make committees with possible winners
        # does_condorcet_exist = ae.exists_condorcet_winner(
        #     du.generate_all_committees(num_candidates=n_alternatives, num_winners=n_winners), cand_pair)
        does_condorcet_exist = ae.exists_condorcet_winner_fast(
            n_alternatives=n_alternatives,
            n_winners=n_winners,
            cand_pairs=cand_pair,
            n_voters=n_voters
        )
        if does_condorcet_exist:
            violations += ae.eval_condorcet_winner(khot_winners, cand_pair)
        else:
            violations += 0  # unnecessary but just for completeness/clarity
    if "condorcet_loser" in axioms:
        violations += ae.eval_condorcet_loser(khot_winners, cand_pair)

    if "dummett" in axioms:
        # Find committees able to satisfy Dummett's condition on this profile
        # dummett_winners = ae.find_dummett_winners(num_voters=n_voters, num_winners=n_winners, profile=profile._rankings)
        dummett_winners = ae.find_dummett_winners_fast(num_voters=n_voters,
                                                       num_winners=n_winners,
                                                       num_alternatives=n_alternatives,
                                                       rank_matrix=rank_matrix,
                                                       profile=profile._rankings
                                                       )
        violations += ae.eval_dummetts_condition(khot_winners,
                                                 n_voters,
                                                 n_winners,
                                                 profile._rankings,
                                                 required_winners=dummett_winners)

    if "solid_coalition" in axioms:
        violations += ae.eval_solid_coalitions(khot_winners, n_voters, n_winners,
                                               rank_matrix)
    if "strong_unanimity" in axioms:
        violations += ae.eval_strong_unanimity(khot_winners, n_winners, profile._rankings)

    # if "local_stability" in axioms:
    #     violations += ae.eval_local_stability(khot_winners, profile._rankings, n_voters,
    #                                           math.ceil(n_voters / n_winners))
    if "local_stability" in axioms:
        violations += ae.eval_local_stability_fast(khot_winners,
                                                   profile._rankings,
                                                   n_voters,
                                                   rank_matrix,
                                                   math.ceil(n_voters / n_winners))
    if "strong_pareto" in axioms:
        violations += ae.eval_strong_pareto_efficiency(khot_winners, profile._rankings)

    if "jr" in axioms:
        # winners = du.winner_set_from_khot_committee(winners)
        # if "jr" in axioms_to_evaluate:
        violations += int(not abc_prop.check_JR(profile=abc_profile,
                                                committee=winners))
    if "ejr" in axioms:
        violations += int(not abc_prop.check_EJR(profile=abc_profile,
                                                 committee=winners))
    if "core" in axioms:
        violations += int(not abc_prop.check_core(profile=abc_profile,
                                                  committee=winners))

    # example call of evaluation function. profile should be a pref_voting Profile and winners is a tuple.
    # self.evaluation_function(idx, winners, profile, **self.kwargs)

    return -(violations/len(axioms))


def score_of_vector_on_profiles(m, axioms_to_optimize, pv_profiles, abc_profiles, rank_matrix, candidate_pairs, vectors_to_test, all_num_winners):

    # profiles = df["Profile"]
    # pv_profiles = [pref_voting.profiles.Profile(eval(profile)) for profile in profiles]
    # abc_profiles = [du.abc_profile_from_rankings(m=m, k=all_num_winners[idx], rankings=profile._rankings) for
    #                 idx, profile in
    #                 enumerate(pv_profiles)]
    # rank_matrix = [eval(rm) for rm in df["rank_matrix"]]
    # candidate_pairs = [eval(cp) for cp in df["candidate_pairs"]]

    results = dict()
    for name, score_vector in vectors_to_test.items():
        rule = PositionalScoringRule(profiles=pv_profiles,
                                     eval_func=axiom_evaluation_function,
                                     m=m,
                                     initial_state=score_vector,
                                     num_winners=all_num_winners,
                                     rank_matrix=rank_matrix,
                                     candidate_pairs=candidate_pairs,
                                     abc_profiles=abc_profiles,
                                     axioms=axioms_to_optimize
                                     )

        score = rule.rule_score()
        score = round(score, 3)
        results[name] = score

        # print(f"Testing {name}: {score_vector} -- {score}")
        print("Testing {:<25}: {:<50} -- {:>16}".format(name, str(score_vector), score))
        # print(f"Violation rate is: {score}")

    # print(f"Testing Score Vector: {initial_state}")
    # print(f"Violation rate: {score}")
    return results


def get_axiom_list(name):
    if isinstance(name, list):
        return name
    if name == "all":
        axioms = ae.all_axioms
    elif name == "reduced":
        axioms = ae.reduced_axioms
    elif name == "custom":
        axioms = [
            "dummett",
            # "fixed_majority",
            # "majority_winner",
            # "majority_loser",
            # "condorcet_winner",
            # "condorcet_loser",
            # "solid_coalition",
            # "strong_unanimity",
            # "local_stability",
            # "strong_pareto",
            # "jr",
            # "ejr",
            # "core"
        ]
    else:
        raise ValueError(f"Unexpected axiom set name. Was given: {name}")
    return axioms


def optimize_scoring_rule(pref_dist, m, all_num_winners, axioms_to_optimize="all", num_profiles_to_sample=None,
                          n_annealing_steps=5000, plot_history=False):
    """

    :return:
    """

    # axioms_to_optimize = "custom"  # axioms that will actually be optimized for
    axiom_set_to_load = "all"  # just used for the filename. We only use the profiles, not any of the actual axiom data

    # Some assumed defaults for loading data. Update values/turn into function parameters as useful
    n_profiles = 25000
    n_voters = 50
    varied_voters = False
    voters_std_dev = 0

    # How many profiles to sample overall. Optimization targets this many profiles.
    # If running with multiple different numbers of winners, sample an even number of profiles from each distinct number
    # of winners.
    if num_profiles_to_sample is None:
        # Assume that we want to optimize for ALL existing data
        # this will be slow. You should probably set num_profiles_to_sample.
        num_profiles_to_sample = n_profiles * len(all_num_winners)
    num_samples = num_profiles_to_sample
    n_samples_per_winner = num_samples // len(all_num_winners)

    # places to store input "training" and "test" data. Optimization is done on data in the non-test df
    aggregate_df = None
    aggregate_num_winners = []
    aggregate_test_df = None
    aggregate_test_num_winners = []

    for num_winners in all_num_winners:
        df = du.load_data(size=n_profiles,
                          n=n_voters,
                          varied_voters=varied_voters,
                          voters_std_dev=voters_std_dev,
                          m=m,
                          num_winners=num_winners,
                          pref_dist=pref_dist,
                          axioms=axiom_set_to_load,
                          # just relevant to filename in loading data, doesn't affect optimization
                          train=True,
                          base_data_folder="data",
                          make_data_if_needed=False)
        test_df = du.load_data(size=n_profiles,
                               n=n_voters,
                               varied_voters=varied_voters,
                               voters_std_dev=voters_std_dev,
                               m=m,
                               num_winners=num_winners,
                               pref_dist=pref_dist,
                               axioms=axiom_set_to_load,
                               # just relevant to filename in loading data, doesn't affect optimization
                               train=False,
                               base_data_folder="data",
                               make_data_if_needed=False)

        aggregate_num_winners += [num_winners] * n_samples_per_winner
        aggregate_test_num_winners += [num_winners] * n_samples_per_winner

        # df = df[:n_samples_per_winner]
        # test_df = test_df[:n_samples_per_winner]
        df = df.sample(n_samples_per_winner)
        test_df = test_df.sample(n_samples_per_winner)
        aggregate_df = pd.concat([df, aggregate_df], ignore_index=True)
        aggregate_test_df = pd.concat([test_df, aggregate_test_df], ignore_index=True)

    profiles = aggregate_df["Profile"]
    pv_profiles = [pref_voting.profiles.Profile(eval(profile)) for profile in profiles]
    abc_profiles = [du.abc_profile_from_rankings(m=m, k=aggregate_num_winners[idx], rankings=profile._rankings) for idx, profile in
                    enumerate(pv_profiles)]
    rank_matrix = [eval(rm) for rm in aggregate_df["rank_matrix"]]
    candidate_pairs = [eval(cp) for cp in aggregate_df["candidate_pairs"]]

    test_profiles = aggregate_test_df["Profile"]
    test_pv_profiles = [pref_voting.profiles.Profile(eval(profile)) for profile in test_profiles]
    test_abc_profiles = [du.abc_profile_from_rankings(m=m, k=aggregate_test_num_winners[idx], rankings=profile._rankings) for idx, profile in enumerate(test_pv_profiles)]
    test_rank_matrix = [eval(rm) for rm in aggregate_test_df["rank_matrix"]]
    test_candidate_pairs = [eval(cp) for cp in aggregate_test_df["candidate_pairs"]]

    job_name = f"annealing-axioms={axioms_to_optimize}-steps={n_annealing_steps}-n_profiles={num_profiles_to_sample}-m={m}-k={all_num_winners}"

    axioms_to_optimize = get_axiom_list(axioms_to_optimize)

    if n_annealing_steps > 0:
        # Run optimization job
        rule = PositionalScoringRule(profiles=pv_profiles,
                                     eval_func=axiom_evaluation_function,
                                     m=m,
                                     num_winners=aggregate_num_winners,
                                     keep_history=True,
                                     history_path="annealing",
                                     job_name=job_name,
                                     rank_matrix=rank_matrix,
                                     candidate_pairs=candidate_pairs,
                                     abc_profiles=abc_profiles,
                                     axioms=axioms_to_optimize,
                                     verbose=True
                                     )

        print("Beginning annealing...")
        opt_dict = rule.optimize(n_steps=n_annealing_steps)
        vector = opt_dict["state"]
        print(f"Result of annealing: {vector}")

    borda = [round((m - i - 1) / (m - 1), 3) for i in range(m)]
    plurality = [1] + [0] * (m - 1)
    k_approval = [1] * num_winners + [0] * (m - num_winners)
    graded_approval = [1] + [0.5] * (num_winners - 1) + [0] * (m - num_winners)
    # half_approval_degrading = [1] + [0.95] * (num_winners-1) + [0] * (m - num_winners)
    half_approval_degrading_small = [1] + [0.95 for _ in range(m - (m // 2) - 2)] + [1 / (2 ** (idx + 1)) for idx in
                                                                                     range(m // 2 + 1)]
    half_approval_degrading_large = [1] + [0.95 for _ in range(m - (m // 2) - 1)] + [1 / (2 ** (idx + 1)) for idx in
                                                                                     range(m // 2)]
    annealing_k_all_all_axioms = [1, 0.6, 0.55, 0.46, 0.27, 0.18, 0]
    annealing_k_all_all_axioms = [1.0, 0.611, 0.5432, 0.463, 0.274, 0.176, 0.0]
    annealing_k1 = [1, 0.69, 0.51, 0.30, 0.08, 0.05, 0]
    annealing_k2 = [1, 0.82, 0.50, 0.45, 0.10, 0.01, 0]
    annealing_k3 = [1, 0.77, 0.71, 0.24, 0.16, 0.08, 0]
    annealing_k4 = [1, 0.74, 0.51, 0.46, 0.12, 0.09, 0]
    annealing_k5 = [1, 0.50, 0.44, 0.32, 0.25, 0.05, 0]
    annealing_k6 = [1, 0.54, 0.42, 0.42, 0.37, 0.30, 0]
    anneal_all_vecs = {
        1: annealing_k1,
        2: annealing_k2,
        3: annealing_k3,
        4: annealing_k4,
        5: annealing_k5,
        6: annealing_k6,
    }
    annealing_reduced_k1 = [1.0, 0.696, 0.422, 0.361, 0.193, 0.051, 0.0]
    annealing_reduced_k2 = [1.0, 0.865, 0.542, 0.410, 0.256, 0.098, 0.0]
    annealing_reduced_k3 = [1.0, 0.851, 0.695, 0.378, 0.129, 0.019, 0.0]
    annealing_reduced_k4 = [1.0, 0.628, 0.597, 0.512, 0.219, 0.045, 0.0]
    annealing_reduced_k5 = [1.0, 0.548, 0.471, 0.414, 0.382, 0.083, 0.0]
    annealing_reduced_k6 = [1.0, 0.708, 0.503, 0.476, 0.429, 0.306, 0.0]
    anneal_reduced_vecs = {
        1: annealing_reduced_k1,
        2: annealing_reduced_k2,
        3: annealing_reduced_k3,
        4: annealing_reduced_k4,
        5: annealing_reduced_k5,
        6: annealing_reduced_k6,
    }
    ben_choice_1 = [1, 0.75, 0.50, 0.25, 0, 0, 0]
    ben_choice_2 = [1, 0.75, 0.50, 0.00, 0, 0, 0]
    ben_choice_3 = [1, 0.80, 0.70, 0.40, 0, 0, 0]
    ben_choice_4 = [1, 0.80, 0.70, 0.30, 0, 0, 0]
    ben_choice_5 = [1, 0.80, 0.70, 0.00, 0, 0, 0]
    local_annealing = [1.0, 0.731482, 0.43044779, 0.40157911, 0.27055577, 0.04211509, 0.0]
    vectors_to_test = {
        # "annealed": vector,
        "borda": borda,
        # "plurality": plurality,
        # "k-approval": k_approval,
        # "graded_approval": graded_approval,
        # "half_degrading_small": half_approval_degrading_small,
        # "half_degrading_large": half_approval_degrading_large,
        # "annealing-all_axioms": annealing_k_all_all_axioms,
        f"annealing_k{num_winners}": anneal_all_vecs[num_winners],
        f"annealing_reduced_k{num_winners}": anneal_reduced_vecs[num_winners],
        # "annealing_k1": annealing_k1,
        # "annealing_k2": annealing_k2,
        # "annealing_k3": annealing_k3,
        # "annealing_k4": annealing_k4,
        # "annealing_k5": annealing_k5,
        # "annealing_k6": annealing_k6,
        # "annealing_reduced_k1": annealing_reduced_k1,
        # "annealing_reduced_k2": annealing_reduced_k2,
        # "annealing_reduced_k3": annealing_reduced_k3,
        # "annealing_reduced_k4": annealing_reduced_k4,
        # "annealing_reduced_k5": annealing_reduced_k5,
        # "annealing_reduced_k6": annealing_reduced_k6,
        # "ben_choice_1": ben_choice_1,
        # "ben_choice_2": ben_choice_2,
        # "ben_choice_3": ben_choice_3,
        # "ben_choice_4": ben_choice_4,
        # "ben_choice_5": ben_choice_5,
        # "local_annealing": local_annealing
    }
    results = score_of_vector_on_profiles(m=m,
                                          axioms_to_optimize=axioms_to_optimize,
                                          pv_profiles=test_pv_profiles,
                                          abc_profiles=test_abc_profiles,
                                          rank_matrix=test_rank_matrix,
                                          candidate_pairs=test_candidate_pairs,
                                          vectors_to_test=vectors_to_test,
                                          all_num_winners=aggregate_test_num_winners)

    # TODO: We very much should make sure to save the output of annealing to a file someplace so we can see what
    # TODO: the vector actually looks like. That's the most interesting bit for discussion.

    if plot_history and n_annealing_steps > 0:
        # Show plot of best energy over time
        history = opt_dict["history"]["best_energy"]
        plt.plot(history)
        plt.show()


if __name__ == "__main__":
    # Default values
    all_num_winners = [1, 2, 3, 4, 5, 6]
    axioms_to_optimize = "reduced"
    num_profiles_to_sample = 5000
    n_annealing_steps = 0

    # Parse command line arguments
    if len(sys.argv) > 1:
        args = dict(arg.split('=', 1) for arg in sys.argv[1:])
        for k, v in args.items():
            if k == "num_winners":
                all_num_winners = eval(v)  # This will convert string "[1]" to list [1]
            elif k == "axioms_to_optimize":
                axioms_to_optimize = eval(v)  # This will handle both string 'all' and lists
            elif k == "num_profiles_to_sample":
                num_profiles_to_sample = int(v)
            elif k == "n_annealing_steps":
                n_annealing_steps = int(v)

    for anw in all_num_winners:
        print(
            f"\n\n==============================\n\nNow printing results for num_winners={anw}, ax={axioms_to_optimize}\n")
        optimize_scoring_rule(pref_dist="mixed",
                              m=7,
                              all_num_winners=[anw],
                              axioms_to_optimize=axioms_to_optimize,
                              num_profiles_to_sample=num_profiles_to_sample,
                              n_annealing_steps=n_annealing_steps)
