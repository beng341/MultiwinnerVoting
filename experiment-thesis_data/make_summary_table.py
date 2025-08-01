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
    "mixed"
]

all_axioms = [
    "all",
    "dummett",
    # "consensus",
    "fixed_majority",
    "majority_winner",
    "majority_loser",
    "condorcet_winner",
    "condorcet_loser",
    "solid_coalition",
    "strong_unanimity",
    "local_stability",
    "strong_pareto",
    "jr",
    "ejr",
    "core"
]

rule_shortnames = {
    "Neural Network": "NN",
    "Random Choice": "Random",
    "Borda ranking": "Borda",
    "Plurality ranking": "SNTV",     # "Plurality",
    "STV": "STV",
    "Approval Voting (AV)": "Bloc",     # "AV",
    "Proportional Approval Voting (PAV)": "PAV",
    "Approval Chamberlin-Courant (CC)": "CC",
    "Lexicographic Chamberlin-Courant (lex-CC)": "lex-CC",
    "Sequential Approval Chamberlin-Courant (seq-CC)": "seq-CC",
    "Monroe's Approval Rule (Monroe)": "Monroe",
    "Greedy Monroe": "Greedy M.",
    "Minimax Approval Voting (MAV)": "MAV",
    "Method of Equal Shares (aka Rule X) with Phragmén phase": "MES",
    "E Pluribus Hugo (EPH)": "EPH",
    "Random Serial Dictator": "RSD",
    "Min Violations Committee": "Min",
    "Max Violations Committee": "Max",
}

evaluation_column_shortnames = {
    "violation_rate-mean": "Mean",
    "dummetts_condition-mean": "Dummett's",
    # "consensus_committee-mean": "Cons.",
    "fixed_majority-mean": "F Maj",
    "majority-mean": "Maj W",
    "majority_loser-mean": "Maj L",
    "condorcet_winner-mean": "Cond W",
    "condorcet_loser-mean": "Cond L",
    "solid_coalitions-mean": "S. Coalitions",
    "strong_unanimity-mean": "Unanimity",
    "local_stability-mean": "Stability",
    "strong_pareto_efficiency-mean": "Pareto",
    "jr-mean": "JR",
    "ejr-mean": "EJR",
    "core-mean": "Core",
}

pref_dist_map = {
    "all": "all",
    "stratification__args__weight=0.5": "Stratified",
    "URN-R": "Urn",
    "IC": "IC",
    "IAC": "IAC",
    "identity": "Identity",
    "MALLOWS-RELPHI-R": "Mallows",
    "single_peaked_conitzer": "SP Conitzer",
    "single_peaked_walsh": "SP Walsh",
    "euclidean__args__dimensions=3_-_space=gaussian_ball": "Gaussian Ball 3",
    "euclidean__args__dimensions=10_-_space=gaussian_ball": "Gaussian Ball 10",
    "euclidean__args__dimensions=3_-_space=uniform_ball": "Uniform Ball 3",
    "euclidean__args__dimensions=10_-_space=uniform_ball": "Uniform Ball 10",
    "euclidean__args__dimensions=3_-_space=gaussian_cube": "Gaussian Cube 3",
    "euclidean__args__dimensions=10_-_space=gaussian_cube": "Gaussian Cube 10",
    "euclidean__args__dimensions=3_-_space=uniform_cube": "Uniform Cube 3",
    "euclidean__args__dimensions=10_-_space=uniform_cube": "Uniform Cube 10",
    "mixed": "Mixed"
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
    # result_folder = 'experiment_all_axioms/evaluation_results'
    result_folder = 'evaluation_results_thesis'

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
            # if rule == "STV":
            #     continue

            if k == 1 and rule not in rule_shortnames:
                # skip all the single winner rules
                continue

            if rule_shortnames[rule] not in summ_stats:
                summ_stats[rule_shortnames[rule]] = {evaluation_column_shortnames[col]: 0 for col in df.columns[1:]}
            for col in df.columns[1:]:
                # if rule == "Approval Voting (AV)" and col == "fixed_majority-mean":
                #     consensus_violations = row[col]
                #     if consensus_violations > 0:
                #         print(path)

                summ_stats[rule_shortnames[rule]][evaluation_column_shortnames[col]] += row[col]

    # Sort rows by custom order -- aligned with increasing violation rate for m=7
    rule_sortorder = ["NN", "Min", "Max",
                      "Borda", "EPH", "SNTV", "Bloc",       # Individual excellence rules
                      "STV", "PAV", "MES", "CC", "seq-CC", "lex-CC", "Monroe", "Greedy M.",  # Proportional/diverse
                      "MAV", "RSD", "Random"    # Other rules
                      ]
    summ_stats = {k: summ_stats[k] for k in rule_sortorder}
    result_df = pd.DataFrame.from_dict(summ_stats, orient='index')

    result_df = result_df / res_count

    # Sort DF columns in custom order
    col_order = ["Mean",
                 "Maj W", "Maj L", "Cond W", "Cond L", "Pareto", "F Maj", "Unanimity",  # Individual excellence axioms
                 "Dummett's", "JR", "EJR", "Core", "S. Coalitions", "Stability",        # diversity/proportional axioms
                 ]
    result_df = result_df[col_order]

    # Add name to first column, useful when formatting
    result_df = result_df.reset_index().rename(columns={'index': 'Method'})

    # # Sort rows by custom order -- aligned with increasing violation rate for m=7
    # rule_sortorder = ["NN", "Random", "Borda", "SNTV", "Bloc", "PAV", "CC", "lex-CC", "seq-CC", "Monroe", "Greedy M.", "MAV"]
    # result_df.set_index("Method")
    # result_df = result_df.reindex(rule_sortorder)
    # result_df = result_df.reset_index()

    # # Sort existing rule columns by mean violation rate
    # nn_random_rows = result_df.iloc[:2]
    # try:
    #     others_rows_sorted = result_df.iloc[2:].sort_values(by='Mean', ascending=True)
    # except KeyError:
    #     print(f"n_profiles = {n_profiles}, num_voters = {num_voters}, m = {m_set}, k = {k_set}, pref_dist = {pref_dist}, axioms = {axioms}")
    #     print(result_df)
    #     sys.exit(1)
    # result_df = pd.concat([nn_random_rows, others_rows_sorted])

    dists = ["all"] if len(pref_dist) > 1 else pref_dist

    out_path = "experiment-thesis_data/summary_tables/"
    filename = f"summary_table-n_profiles={n_profiles}-num_voters={num_voters}-m={m_set}-pref_dist={dists}-axioms={axioms}.csv"
    result_df.to_csv(os.path.join(out_path, filename), index=False)


def format_summary_table(n_profiles=[], num_voters=[], m_set=[], k_set=[], pref_dist=[], axioms=[]):

    # Rounding and custom formatting
    def format_value(row_value, col_name, value, min_val, values_to_underline):
        """Rounds the value, underlines it if it rounds to zero but is not zero, and bolds the minimum value."""
        rounded_value = round(value, 3)
        formatted_value = f"{rounded_value:.3f}".lstrip('0') if rounded_value < 1 else f"{rounded_value:.3f}"

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

        formatted_value = f"{formatted_value}".replace('0.', '.')

        # Normal rounding
        return formatted_value

    known_past_results = [
        ("Bloc", "strong_pareto_efficiency-mean"),
        ("PAV", "strong_pareto_efficiency-mean"),
        ("Bloc", "fixed_majority-mean"),  # Contrary to our results; check tie-breaking
        ("Bloc", "strong_unanimity-mean"),
        ("SNTV", "solid_coalitions-mean"),
        # ("SNTV", "consensus_committee-mean"),
        ("SNTV", "majority-mean"),
        ("Borda", "strong_unanimity-mean"),
        # ("CC", "consensus_committee-mean"),  # Unclear how to compare between CC variants
        ("Monroe", "strong_unanimity-mean"),  # Unclear how to compare between Monroe variants
        # ("Monroe", "consensus_committee-mean"),  # Unclear how to compare between Monroe variants
        # ("Greedy M.", "solid_coalitions-mean"),     # Elkind 2017 shows this for ranked prefs but we use the Lackner/Skowron def'n of Greedy M which is an ABC rule; not comparable.
        # ("Greedy M.", "consensus_committee-mean"),
        ("Greedy M.", "strong_unanimity-mean"),
        ("EPH", "strong_pareto_efficiency-mean"),  # see Quinn and Schneier 2019,
        ("MES", "ejr-mean"),        # See Lackner etc. 2023
        ("MES", "jr-mean"),         # See Lackner etc. 2023
        ("CC", "jr-mean"),          # See Lackner etc. 2023
        ("seq-CC", "jr-mean"),      # See Lackner etc. 2023
        ("PAV", "jr-mean"),         # See Lackner etc. 2023
        ("PAV", "ejr-mean"),        # See Lackner etc. 2023
        ("Monroe", "jr-mean"),      # See Lackner etc. 2023
        ("Greedy M.", "jr-mean"),         # See Lackner etc. 2023
        ("STV", "solid_coalitions-mean"),         # See Tideman 1995
        # ("STV", "dummetts_condition-mean"),         # See Tideman 1995
        ("STV", "majority-mean"),         # See Tideman 1995

    ]
    known_past_results = [(r, evaluation_column_shortnames[a]) for (r, a) in known_past_results]

    dists = ["all"] if len(pref_dist) > 1 else pref_dist
    out_path = "experiment-thesis_data/summary_tables/"
    filename = f"summary_table-n_profiles={n_profiles}-num_voters={num_voters}-m={m_set}-pref_dist={dists}-axioms={axioms}.csv"
    df = pd.read_csv(os.path.join(out_path, filename))

    # Apply custom formatting for each column
    for col in df.columns[1:]:
        min_val = df[col].min()  # Find the minimum value in the column
        # df[col] = df[col].apply(lambda x: format_value(x, min_val))
        df[col] = df.apply(lambda row: format_value(row['Method'], col, row[col], min_val, known_past_results), axis=1)

    # Convert to LaTeX
    ncols = len(df.columns)
    col_alignment = "l" + "c"*(ncols-1)
    
    # Create the caption
    dist_text = "all" if len(pref_dist) > 1 else pref_dist_map[pref_dist[0]]
    # plural_s = "s" if len(pref_dist) > 1 else ""
    caption = f"Average Axiom Violation Rate for {m_set[0]} alternatives and $1 \\leq k < {m_set[0]}$ winners across {dist_text} preferences."

    df.columns = [df.columns[0]] + [f"\\rotatebox{{90}}{{{col}}}" for col in df.columns[1:]]
    # Generate the LaTeX table with the new formatting
    latex_table = df.to_latex(escape=False, index=False, column_format=col_alignment)
    
    # Modify the LaTeX table to span both columns and add the caption below the table
    latex_table = latex_table.replace(
        "\\begin{tabular}",
        f"\\begin{{table}}[h]\n\\label{{tab:summary_table-n_profiles={n_profiles}-num_voters={num_voters}-m={m_set}-pref_dist={dists}-axioms={axioms}}}\n\\centering\n\\fontsize{{7pt}}{{9pt}}\n\\selectfont\n\\setlength{{\\tabcolsep}}{{4.6pt}}\n\\renewcommand{{\\arraystretch}}{{1.05}}\\begin{{tabular}}"
    )
    latex_table = latex_table.replace(
        "\\end{tabular}",
        "\\end{tabular}\n\\caption{" + caption + "}\n\\end{table}"
    )

    out_path = "./experiment-thesis_data/summary_tables"
    filename = f"formatted_table-m={m_set}-pref_dist={dists}.tex"
    file_path = os.path.join(out_path, filename)
    with open(file_path, 'w') as f:
        f.write(latex_table)


def make_tables_for_all_combinations():
    n_profiles = [25000]
    n_voters = [50]
    m_set = [5, 6, 7]
    k_set = [1, 2, 3, 4, 5, 6]

    for m, pref_dist in itertools.product(m_set, all_pref_dists):

        make_summary_table(n_profiles, n_voters, [m], k_set, [pref_dist], ["all"])
        format_summary_table(n_profiles, n_voters, [m], k_set, [pref_dist], ["all"])


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

    make_tables_for_all_combinations()
