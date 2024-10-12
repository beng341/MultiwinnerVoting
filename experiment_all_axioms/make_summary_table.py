import itertools
import os
import sys

import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_utils as du

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

rule_shortnames = {
    "Neural Network": "NN",
    "Random Choice": "Random",
    "Borda ranking": "Borda",
    "Plurality ranking": "SNTV",     # "Plurality",
    # "STV": "STV",
    "Approval Voting (AV)": "Bloc",     # "AV",
    "Proportional Approval Voting (PAV)": "PAV",
    "Approval Chamberlin-Courant (CC)": "CC",
    "Lexicographic Chamberlin-Courant (lex-CC)": "lex-CC",
    "Sequential Approval Chamberlin-Courant (seq-CC)": "seq-CC",
    "Monroe's Approval Rule (Monroe)": "Monroe",
    "Greedy Monroe": "Greedy M.",
    "Minimax Approval Voting (MAV)": "MAV"
}

evaluation_column_shortnames = {
    "violation_rate-mean": "Mean",
    "dummetts_condition-mean": "Dummet",
    "consensus_committee-mean": "Cons.",
    "fixed_majority-mean": "F Maj",
    "majority-mean": "Maj W",
    "majority_loser-mean": "Maj L",
    "condorcet_winner-mean": "Cond W",
    "condorcet_loser-mean": "Cond L",
    "solid_coalitions-mean": "S. Coal.",
    "strong_unanimity-mean": "Unan.",
    "local_stability-mean": "Stab.",
    "strong_pareto_efficiency-mean": "Pareto"
}


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

    res_count = 0
    result_folder = 'experiment_all_axioms/evaluation_results'

    for n, v, m, k, dist, ax in itertools.product(n_profiles, num_voters, m_set, k_set, pref_dist, axioms):
        if k >= m:
            continue

        path = f"{result_folder}/axiom_violation_results-n_profiles={n}-num_voters={v}-m={m}-k={k}-pref_dist={dist}-axioms={ax}.csv"
        df = du.load_evaluation_results_df(path=path, metric="std", include_noise=False)
        if df is None:
            continue

        res_count += 1
        for index, row in df.iterrows():
            rule = row['Method']
            if rule == "STV":
                continue

            if rule_shortnames[rule] not in summ_stats:
                summ_stats[rule_shortnames[rule]] = {evaluation_column_shortnames[col]: 0 for col in df.columns[1:]}
            for col in df.columns[1:]:
                if rule == "Approval Voting (AV)" and col == "fixed_majority-mean":
                    consensus_violations = row[col]
                    if consensus_violations > 0:
                        print(path)

                summ_stats[rule_shortnames[rule]][evaluation_column_shortnames[col]] += row[col]

    result_df = pd.DataFrame.from_dict(summ_stats, orient='index')

    result_df = result_df / res_count


    # Add name to first column, useful when formatting
    result_df = result_df.reset_index().rename(columns={'index': 'Method'})

    # Sort existing rule columns by mean violation rate
    nn_random_rows = result_df.iloc[:2]
    try:
        others_rows_sorted = result_df.iloc[2:].sort_values(by='Mean', ascending=True)
    except KeyError:
        print(f"n_profiles = {n_profiles}, num_voters = {num_voters}, m = {m_set}, k = {k_set}, pref_dist = {pref_dist}, axioms = {axioms}")
        print(result_df)
        sys.exit(1)

    result_df = pd.concat([nn_random_rows, others_rows_sorted])

    dists = ["all"] if len(pref_dist) > 1 else pref_dist

    out_path = "experiment_all_axioms/summary_tables/"
    filename = f"summary_table-n_profiles={n_profiles}-num_voters={num_voters}-m={m_set}-k={k_set}-pref_dist={dists}-axioms={axioms}.csv"
    result_df.to_csv(os.path.join(out_path, filename), index=False)


def format_summary_table(n_profiles=[], num_voters=[], m_set=[], k_set=[], pref_dist=[], axioms=[]):

    # Rounding and custom formatting
    def format_value(row_value, col_name, value, min_val, values_to_underline):
        """Rounds the value, underlines it if it rounds to zero but is not zero, and bolds the minimum value."""
        rounded_value = round(value, 3)
        formatted_value = f"{rounded_value:.3f}"

        if value == 0:  # Bold if value is lowest in the column
            formatted_value = f"\\textbf{{{0}}}"

        if value == min_val and value > 0:  # Bold if equal to lowest in column
            formatted_value = f"\\textbf{{{rounded_value:.3f}}}"

        # Italicize if value rounds to 0 but wasn't 0 before rounding
        if rounded_value == 0 and value != 0:
            formatted_value = f"\\textit{{{rounded_value:.3f}}}"

        # Underline if corresponding rule satisfies this axiom
        if (row_value, col_name) in values_to_underline:
            formatted_value = f"\\cellcolor{{green!25}}{formatted_value}"
            # formatted_value = f"\\textcolor{{blue}}{{{formatted_value}}}"

        # Normal rounding
        return formatted_value

    known_past_results = [
        ("Bloc", "strong_pareto_efficiency-mean"),
        ("PAV", "strong_pareto_efficiency-mean"),
        ("Bloc", "fixed_majority-mean"),  # Contrary to our results; check tie-breaking
        ("Bloc", "strong_unanimity-mean"),
        ("SNTV", "solid_coalitions-mean"),
        ("SNTV", "consensus_committee-mean"),
        ("SNTV", "majority-mean"),
        ("Borda", "strong_unanimity-mean"),
        ("CC", "consensus_committee-mean"),  # Unclear how to compare between CC variants
        ("Monroe", "strong_unanimity-mean"),  # Unclear how to compare between Monroe variants
        ("Monroe", "consensus_committee-mean"),  # Unclear how to compare between Monroe variants
        ("Greedy M.", "solid_coalitions-mean"),
        ("Greedy M.", "consensus_committee-mean"),
        ("Greedy M.", "strong_unanimity-mean"),
    ]
    known_past_results = [(r, evaluation_column_shortnames[a]) for (r, a) in known_past_results]

    dists = ["all"] if len(pref_dist) > 1 else pref_dist
    out_path = "experiment_all_axioms/summary_tables/"
    filename = f"summary_table-n_profiles={n_profiles}-num_voters={num_voters}-m={m_set}-k={k_set}-pref_dist={dists}-axioms={axioms}.csv"
    df = pd.read_csv(os.path.join(out_path, filename))

    # Apply custom formatting for each column
    for col in df.columns[1:]:
        min_val = df[col].min()  # Find the minimum value in the column
        # df[col] = df[col].apply(lambda x: format_value(x, min_val))
        df[col] = df.apply(lambda row: format_value(row['Method'], col, row[col], min_val, known_past_results), axis=1)

    # Convert to LaTeX
    ncols = len(df.columns)
    col_alignment = "l" + "c"*(ncols-1)
    latex_table = df.to_latex(escape=False, index=False, column_format=col_alignment)  # Set escape=False to allow LaTeX formatting

    out_path = "./experiment_all_axioms/summary_tables"
    filename = f"formatted_table-n_profiles={n_profiles}-num_voters={num_voters}-m={m_set}-k={k_set}-pref_dist={dists}-axioms={axioms}.tex"
    file_path = os.path.join(out_path, filename)
    with open(file_path, 'w') as f:
        f.write(latex_table)


def make_tables_for_all_combinations():
    n_profiles = [25000]
    n_voters = [50]
    m_set = [5, 6, 7]
    k_set = [1, 2, 3, 4, 5, 6]

    for m, k, pref_dist in itertools.product(m_set, k_set, all_pref_dists):
        if k >= m:
            continue

        make_summary_table(n_profiles, n_voters, [m], [k], [pref_dist], ["all"])
        format_summary_table(n_profiles, n_voters, [m], [k], [pref_dist], ["all"])


def make_aggregated_table_single_m(m=5):

    n_profiles = [25000]
    n_voters = [50]
    k_set = [1, 2, 3, 4, 5, 6]

    make_summary_table(n_profiles, n_voters, [m], k_set, all_pref_dists, ["all"])
    format_summary_table(n_profiles, n_voters, [m], k_set, all_pref_dists, ["all"])


if __name__ == "__main__":
    make_aggregated_table_single_m(m=5)
    make_aggregated_table_single_m(m=6)
    make_aggregated_table_single_m(m=7)

    # make_tables_for_all_combinations()
