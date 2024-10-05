import itertools
import os

import pandas as pd
import utils.data_utils as du

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
    "Plurality ranking": "Plurality",
    "Approval Voting (AV)": "AV",
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
    others_rows_sorted = result_df.iloc[2:].sort_values(by='Mean', ascending=True)
    result_df = pd.concat([nn_random_rows, others_rows_sorted])

    dists = ["all"] if len(pref_dist) > 1 else pref_dist

    out_path = "experiment_all_axioms/summary_tables/"
    filename = f"summary_table-n_profiles={n_profiles}-num_voters={num_voters}-m={m_set}-k={k_set}-pref_dist={dists}-axioms={axioms}.csv"
    result_df.to_csv(os.path.join(out_path, filename), index=False)


def format_summary_table(n_profiles=[], num_voters=[], m_set=[], k_set=[], pref_dist=[], axioms=[]):

    # Rounding and custom formatting
    def format_value(row_value, col_name, value, min_val, values_to_underline):
        """Rounds the value, underlines it if it rounds to zero but is not zero, and bolds the minimum value."""
        rounded_value = round(value, 2)
        formatted_value = f"{rounded_value:.2f}"

        if value == 0:  # Bold if value is lowest in the column
            formatted_value = f"\\textbf{{{0}}}"

        if value == min_val and value > 0:  # Bold if equal to lowest in column
            formatted_value = f"\\textbf{{{rounded_value:.2f}}}"

        # Italicize if value rounds to 0 but wasn't 0 before rounding
        if rounded_value == 0 and value != 0:
            formatted_value = f"\\textit{{{rounded_value:.1f}}}"

        # Underline if corresponding rule satisfies this axiom
        if (row_value, col_name) in values_to_underline:
            formatted_value = f"\\underline{{{formatted_value}}}"
            # formatted_value = f"\\textcolor{{blue}}{{{formatted_value}}}"

        # Normal rounding
        return formatted_value

    known_past_results = [
        ("AV", "strong_pareto_efficiency-mean"),
        ("PAV", "strong_pareto_efficiency-mean"),
        ("AV", "fixed_majority-mean"),  # Contrary to our results; check tie-breaking
        ("Plurality", "solid_coalitions-mean"),
        ("Plurality", "consensus_committee-mean"),
        ("Borda", "strong_unanimity-mean"),
        ("CC", "consensus_committee-mean"),  # Unclear how to compare between CC variants
        ("Monroe", "strong_unanimity-mean"),  # Unclear how to compare between Monroe variants
        ("Monroe", "consensus_committee-mean"),  # Unclear how to compare between Monroe variants
        ("Greedy Monroe", "solid_coalitions-mean"),
        ("Greedy Monroe", "consensus_committee-mean"),
        ("Greedy Monroe", "strong_unanimity-mean"),
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
    latex_table = df.to_latex(escape=False, index=False)  # Set escape=False to allow LaTeX formatting

    # Save LaTeX table to file
    with open("./experiment_all_axioms/formatted_summary_table_new.tex", 'w') as f:
        f.write(latex_table)


if __name__ == "__main__":

    n_profiles = [25000]
    n_voters = [50]
    m_set = [5]
    k_set = [1, 2, 3, 4, 5, 6]

    make_summary_table(n_profiles, n_voters, m_set, k_set, all_pref_dists, ["all"])
    format_summary_table(n_profiles, n_voters, m_set, k_set, all_pref_dists, ["all"])

    # for d in all_pref_dists:
    #     make_summary_table(n_profiles, n_voters, m_set, k_set, [d], ["all"])
