import itertools
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import data_utils as du
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

rule_shortnames = {
    # "Neural Network": "NN",
    "Random Choice": "Random",
    "Borda ranking": "Borda",
    "Plurality ranking": "SNTV",  # "Plurality",
    "STV": "STV",
    "Approval Voting (AV)": "Bloc",  # "AV",
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
    "NN-all": "NN-all",
    "NN-root": "NN-root",
}


def save_latex_table(df, m_set, pref_dist, folder='experiment-both_axiom_sets/arXiv/distance_tex_tables'):
    # Create the title based on m_set and pref_dist
    if len(m_set) == 1:
        caption = f"Difference between rules for {m_set[0]} alternatives with $1 \\leq k < {m_set[0]}$ "
    else:
        caption = f"Difference between rules for $m \\in \\{{{', '.join(map(str, sorted(m_set)))}\\}}$ alternatives with $1 \\leq k < {m_set[0]}$ "

    # Add the appropriate pref_dist description to the title
    if len(pref_dist) > 1:
        caption += "averaged over all preference distributions."
    else:
        caption += f"on {pref_dist_map.get(pref_dist[0], pref_dist[0])} preferences."

    # Caption for merge_root_and_all_axiom_results text figure:
    # caption += " Darker values correspond to larger distances. A distance of 0 between two rules indicates the rules always elect the same committee while a distance of 1 indicates that the rules' winning committees never have any overlap. Note that a distance of 1 is not possible when $k > \\frac{m}{2}$ as committees must then overlap on some alternatives."
    # # Caption for appendix figures:
    # caption += " Darker values correspond to larger distances. A distance of 0 between two rules indicates the rules always elect the same committee while a distance of 1 indicates that the rules' winning committees never have any overlap. Note that a distance of 1 is not possible when $k > \\frac{m}{2}$ as committees must then overlap on some alternatives."

    label = f"tab:rule_distance_heatmap-m={m_set}-pref_dist={pref_dist[0] if len(pref_dist) == 1 else 'all'}"

    # Round the DataFrame to 3 decimal places
    df = df.round(3).applymap(lambda x: f"{x:.3f}" if isinstance(x, (float, int)) else x)

    # Fill only below the diagonal (including the diagonal) and set values above to "--" or another placeholder
    for i in range(df.shape[0]):
        for j in range(i + 1, df.shape[1]):  # i+1 ensures only values above the diagonal are modified
            df.iloc[i, j] = "--"  # Replace values above the diagonal with "--"

    # Convert the DataFrame to LaTeX and ensure the index is not printed
    latex_table = df.to_latex(index=True, header=True, escape=False, column_format='l' + 'c' * (len(df.columns) - 1))

    # Replace NaNs with "0.000"
    latex_table = latex_table.replace("NaN", "0.000")

    # Combine the title and table
    latex_output = f"""
\\begin{{table*}}[h!]
\\label{{tab:distance_table-m={m_set}-pref_dist={pref_dist[0] if len(pref_dist) == 1 else 'all'}}}
\\centering
{latex_table}
\\caption{{{caption}}}
\\end{{table*}}
"""
    all = "all"
    # Define the output path
    output_path = f"{folder}/distance_table-m={m_set}-pref_dist={pref_dist[0] if len(pref_dist) == 1 else all}.tex"

    # Save the LaTeX table to the specified folder
    with open(output_path, 'w') as f:
        f.write(latex_output)

    # Create a heatmap from the DataFrame
    create_heatmap(df, caption, label=label, folder="experiment-both_axiom_sets/arXiv/distance_heatmaps", m_set=m_set, pref_dist=pref_dist)


def create_heatmap(df, title, label, folder, m_set, pref_dist):
    # df = df.set_index(df.columns[0])

    # Create the LaTeX table with color formatting
    latex_table = "\\begin{tabular}{@{}l" + "c" * (df.shape[1]) + "@{}}\n\\toprule\n"
    # Add column headers
    # latex_table += " & " + " & ".join(df.columns) + " \\\\\n\\midrule\n"
    latex_table += " & " + " & ".join([f"\\rotatebox{{90}}{{{col}}}" for col in df.columns]) + " \\\\\n\\midrule\n"
    # Add table rows with heatmap colors
    for idx, row in df.iterrows():
        # latex_table += idx + " & " + " & ".join([f"\\cellcolor{{blue!{int(float(val)*90)}}} {val}" if val != '--' else f"{val}" for val in row]) + " \\\\\n"
        latex_table += idx + " & " + " & ".join([
            f"\\cellcolor{{blue!{int(float(val) * 80)}}} {str(val)[1:]}" if val != '--' and str(val).startswith('0') else
            f"\\cellcolor{{blue!{int(float(val) * 80)}}} {val}" if val != '--' else
            f"{val}"
            for val in row
        ]) + " \\\\\n"
    latex_table += "\\bottomrule\n\\end{tabular}\n"

    # Final LaTeX output with table caption
    latex_output = f"""\\begin{{table}}[h]
\\centering
\\fontsize{{7pt}}{{9pt}}\selectfont
\\setlength{{\\tabcolsep}}{{4.6pt}}
\\renewcommand{{\\arraystretch}}{{1.05}}
{latex_table}
\\caption{{{title}}}
\\label{{{label}}}
\\end{{table}}"""

    # Save the output
    all = "all"
    output_path = f"{folder}/heatmap-m={m_set}-pref_dist={pref_dist[0] if len(pref_dist) == 1 else all}.tex"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(output_path, 'w') as f:
        f.write(latex_output)


def make_distance_table(n_profiles=[], num_voters=[], m_set=[], k_set=[], pref_dist=[]):
    distance_folder = 'evaluation_results-AAAI_combined/rule_distances'

    count_valid_files = 0  # To keep track of how many files were successfully read

    all_dfs = []

    # Loop through each combination of parameters
    for n, v, m, k, dist in itertools.product(n_profiles, num_voters, m_set, k_set, pref_dist):
        if k >= m:
            continue

        path = f"{distance_folder}/num_voters={v}-m={m}-k={k}-pref_dist={dist}-axioms=both-distances.csv"

        try:
            # Read the CSV file
            dists = pd.read_csv(path)
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue

        # Apply shortnames to the first column (row names) using the global rule_shortnames dictionary
        dists['Unnamed: 0'] = dists['Unnamed: 0'].map(rule_shortnames).fillna(dists['Unnamed: 0'])

        # Apply shortnames to the column headers (skipping the first column)
        dists.columns = [rule_shortnames.get(col, col) for col in dists.columns]

        # Make rule names column the index, not just a column
        dists = dists.set_index(dists.columns[0])

        # Filter to include only the rows and columns in the shortnames dict
        row_names = list(rule_shortnames.values())  # we don't want a NN row
        col_names = list(rule_shortnames.values())  # should include NN
        if "NN-root" not in dists.index:
            row_names.remove("NN-root")

        cols_to_remove = ["Min", "Max", "NN", "NN-all", "NN-root", "Random"]
        for bad_col in cols_to_remove:
            if bad_col in col_names:
                col_names.remove(bad_col)

        dists = dists.loc[row_names, col_names]
        dists = dists.dropna(how="all")
        dists = dists.dropna(axis=1, how="all")

        scaling_factor = m / (m - abs(m - (2 * k)))
        dists[dists.select_dtypes(include=['number']).columns] *= scaling_factor
        # clip dists at 1
        dists = dists.clip(upper=1.0)

        all_dfs.append(dists)

        # Increase the count of valid files processed
        count_valid_files += 1

    # If no valid files were found, return None
    if count_valid_files == 0:
        print("No valid files found for averaging.")
        return None

    average_df = pd.concat(all_dfs).groupby(level=0).mean()

    # Fix index and column names for final display
    average_df.index.name = None
    # average_df.columns = [""] + list(average_df.columns[1:])  # Remove the name of the first column

    rule_sortorder = []
    rule_sortorder += [
        "Borda", "EPH", "SNTV", "Bloc",  # Individual excellence rules
        "STV", "PAV", "MES", "CC", "seq-CC", "lex-CC", "Monroe", "Greedy M.",  # Proportional/diverse
        "MAV", "RSD", "Random",  # Other rules
        "Min", "Max",
    ]
    if "NN-all" in average_df.index:
        rule_sortorder.append("NN-all")
    if "NN-root" in average_df.index:
        rule_sortorder.append("NN-root")

    # Sort both rows and columns according to this ordering of rules
    col_order = [rule for rule in rule_sortorder if rule in average_df.columns]
    average_df = average_df.reindex(columns=col_order)
    average_df = average_df.reindex(rule_sortorder)

    # Save the table
    save_latex_table(average_df, m_set, pref_dist)

    return average_df


def make_aggregates_for_all_combos():
    n_profiles = [25000]
    num_voters = [50]
    m_set = [5, 6, 7]
    k_set = [1, 2, 3, 4, 5, 6]

    for m, pref_dist in itertools.product(m_set, all_pref_dists):
        make_distance_table(n_profiles, num_voters, [m], k_set, [pref_dist])

    for m in m_set:
        make_distance_table(n_profiles, num_voters, [m], k_set, all_pref_dists)

    make_distance_table(n_profiles, num_voters, m_set, k_set, all_pref_dists)


if __name__ == "__main__":

    make_aggregates_for_all_combos()
