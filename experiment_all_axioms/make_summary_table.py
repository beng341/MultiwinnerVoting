import itertools
import pandas as pd


n_profiles = [25000]
n_voters = [50]
m_set = [5, 6]
k_set = [1, 2, 3, 4, 5, 6]

all_pref_dists = [
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

all_axioms = [
    "all",
    "dummett",
    "consensus",
    "fixed_majority",
    "majority_winner",
    "majority_loser",
    "condorcet_winner",
    "condorcet_loser",
    "solid_coalition",
    "strong_unanimity",
    "local_stability",
    "strong_pareto"
]

def make_summary_table(n_profiles=[], num_voters=[], m_set=[], k_set=[], pref_dist=[], axioms=[]):
    """
    Make a summary table of all results.
    We want to read in all the results for a given set of parameters,
    and then average them all together for each rule and each axiom.
    The neural network one should be the average of all 20 networks.
    :param n_profiles: list of int
    :param num_voters: list of int
    :param m: list of int
    :param k: list of int
    :param pref_dist: list of str
    :return:
    """

    summ_stats = {}
    nn_stats = {}

    res_count = 0
    nn_count = 0
    result_folder = 'experiment_all_axioms/evaluation_results'

    for n, v, m, k, dist, ax in itertools.product(n_profiles, num_voters, m_set, k_set, pref_dist, axioms):
        if k >= m:
            continue

        try:
            df = pd.read_csv(f"{result_folder}/axiom_violation_results-n_profiles={n}-num_voters={v}-m={m}-k={k}-pref_dist={dist}-axioms={ax}.csv")
        except FileNotFoundError:
            print(f"File not found: {result_folder}/axiom_violation_results-n_profiles={n}-num_voters={v}-m={m}-k={k}-pref_dist={dist}-axioms={ax}.csv")
            continue

        res_count += 1
        for index, row in df.iterrows():
            method = row['Method']

            if method.startswith("NN"):
                for col in df.columns[1:]:
                    nn_count += 1
                    if col not in nn_stats:
                        nn_stats[col] = 0
                    nn_stats[col] += row[col]
            else:
                if method not in summ_stats:
                    summ_stats[method] = {col: 0 for col in df.columns[1:]}
                for col in df.columns[1:]:
                    summ_stats[method][col] += row[col]
    
    if nn_count > 0:
        nn_averages = {col: nn_stats[col] / nn_count for col in nn_stats}
        summ_stats['NN'] = nn_averages
    
    result_df = pd.DataFrame.from_dict(summ_stats, orient='index')

    result_df = result_df / res_count

    dists = ["all"] if len(pref_dist) > 1 else pref_dist

    result_df.to_csv(f"./experiment_all_axioms/summary_tables/summary_table-n_profiles={n_profiles}-num_voters={num_voters}-m={m}-k={k}-pref_dist={dists}-axioms={axioms}.csv")


if __name__ == "__main__":
    make_summary_table(n_profiles, n_voters, m_set, k_set, all_pref_dists, ["all"])

    for d in all_pref_dists:
        make_summary_table(n_profiles, n_voters, m_set, k_set, [d], ["all"])
            
            
            
        

    
