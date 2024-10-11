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

pref_dist_map = {
    "stratification__args__weight=0.5": "Stratification",
    "URN-R": "Urn",
    "IC": "Impartial Culture",
    "IAC": "Impartial Anonymous Culture",
    "identity": "Identity",
    "MALLOWS-RELPHI-R": "Mallows",
    "single_peaked_conitzer": "Single-peaked (Conitzer)",
    "single_peaked_walsh": "Single-peaked (Walsh)",
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

def save_latex_table(df, m_set, pref_dist, folder='experiment_all_axioms/distance_tex_tables'):
    # Create the title based on m_set and pref_dist
    if len(m_set) == 1:
        title = f"Distance Between Rules for {m_set[0]} alternatives with $1 \\leq k < m$ on "
    else:
        title = f"Distance Between Rules for $m \\in \\{{{', '.join(map(str, sorted(m_set)))}\\}}$ alternatives with $1 \\leq k < m$ on "

        
    # Add the appropriate pref_dist description to the title
    if len(pref_dist) > 1:
        title += "all preference distributions."
    else:
        title += f"{pref_dist_map.get(pref_dist[0], pref_dist[0])} preference distribution."
    
    # Round the DataFrame to 3 decimal places
    df = df.round(3).applymap(lambda x: f"{x:.3f}" if isinstance(x, (float, int)) else x)

    # Fill only below the diagonal (including the diagonal) and set values above to "--" or another placeholder
    for i in range(df.shape[0]):
        for j in range(i+1, df.shape[1]):  # i+1 ensures only values above the diagonal are modified
            df.iloc[i, j] = "--"  # Replace values above the diagonal with "--"

    # Convert the DataFrame to LaTeX and ensure the index is not printed
    latex_table = df.to_latex(index=False, header=True, escape=False, column_format='l' + 'c' * (df.shape[1] - 1))

    # Replace NaNs with "0.000"
    latex_table = latex_table.replace("NaN", "0.000")
    
    # Combine the title and table
    latex_output = f"""
\\begin{{table*}}
\\centering
{latex_table}
\\caption{{{title}}}
\\end{{table*}}
"""
    all = "all"
    # Define the output path
    output_path = f"{folder}/distance_table-m={m_set}-pref_dist={pref_dist[0] if len(pref_dist) == 1 else all}.tex"

    # Save the LaTeX table to the specified folder
    with open(output_path, 'w') as f:
        f.write(latex_output)





def make_distance_table(n_profiles=[], num_voters=[], m_set=[], k_set=[], pref_dist=[]):
    dist_stats = {}
    res_count = 0
    distance_folder = 'experiment_all_axioms/rule_distances'

    # Initialize a DataFrame to accumulate the sum of all DataFrames
    cumulative_df = None
    count_valid_files = 0  # To keep track of how many files were successfully read

    # Loop through each combination of parameters
    for n, v, m, k, dist in itertools.product(n_profiles, num_voters, m_set, k_set, pref_dist):
        if k >= m:
            continue

        path = f"{distance_folder}/num_voters={v}-m={m}-k={k}-pref_dist={dist}-distances.csv"

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

        

        # Make sure the [NN, NN] value is explicitly set to 0.0 in case it's still NaN after the replacement
        if 'NN' in dists.columns and 'NN' in dists['Unnamed: 0'].values:
            dists.loc[dists['Unnamed: 0'] == 'NN', 'NN'] = 0.0
        
       

        # If it's the first valid file, initialize the cumulative DataFrame
        if cumulative_df is None:
            cumulative_df = dists.copy()  # Copy structure and values
            cumulative_df.iloc[:, 1:] = 0  # Set all numerical values to 0 (except first column if non-numeric)
        # Add the current DataFrame to the cumulative sum (ignoring the first column which is non-numeric)
        cumulative_df.iloc[:, 1:] += dists.iloc[:, 1:]
        # Increase the count of valid files processed
        count_valid_files += 1

    # If no valid files were found, return None
    if count_valid_files == 0:
        print("No valid files found for averaging.")
        return None

    
    # Compute the average by dividing the cumulative sum by the number of valid files
    average_df = cumulative_df.copy()
    average_df.iloc[:, 1:] = cumulative_df.iloc[:, 1:] / count_valid_files
    

    # Fix index and column names for final display
    average_df.index.name = None
    average_df.columns = [""] + list(average_df.columns[1:])  # Remove the name of the first column
    

    # Save the table
    save_latex_table(average_df, m_set, pref_dist)

    return average_df


def make_aggregates_for_all_combos():
    n_profiles = [25000]
    num_voters = [50]
    m_set = [5, 6, 7]
    k_set = [1, 2, 3, 4, 5, 6]
    

    for m, k, pref_dist in itertools.product(m_set, k_set, all_pref_dists):
        if k >= m:
            continue

        make_distance_table(n_profiles, num_voters, [m], [k], [pref_dist])
    
    for m in m_set:
        make_distance_table(n_profiles, num_voters, [m], k_set, all_pref_dists)
    
    make_distance_table(n_profiles, num_voters, m_set, k_set, all_pref_dists)

if __name__ == "__main__":
    make_aggregates_for_all_combos()
        
