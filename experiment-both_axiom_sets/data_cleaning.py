import itertools
import pandas as pd
import os
import re
from typing import Dict, List, Tuple, Optional


def parse_filename(filename: str, ax_type: str) -> Optional[Dict[str, str]]:
    """
    Parse filename to extract M, K, D, and AX parameters.
    Expected format: axiom_violation_results-n_profiles=25000-num_voters=50-m={M}-k={K}-pref_dist={D}-axioms={AX}.csv

    Args:
        filename: The CSV filename
        ax_type: Either "all" or "reduced" based on directory

    Returns:
        Dictionary with parsed parameters or None if parsing fails
    """
    # Remove .csv extension
    base_name = filename.replace('.csv', '')

    pattern = r'axiom_violation_results-n_profiles=\d+-num_voters=\d+-m=(\d+)-k=(\d+)-pref_dist=(.+?)-axioms=([^-]+)'

    match = re.search(pattern, base_name)
    if match:
        m_val, k_val, d_val, ax_val = match.groups()

        if ax_val != ax_type:
            raise ValueError(f"Warning: AX value '{ax_val}' in filename doesn't match expected value: '{ax_type}'")

        return {
            'M': int(m_val),
            'K': int(k_val),
            'D': d_val,
            'AX': ax_val,  # Use the actual value from filename rather than ax_type
            'filename': filename
        }

    print(f"Warning: Could not parse filename: {filename}")
    return None


def update_all_row_names(all_files: List[Tuple[pd.DataFrame, Dict]], s: str, return_updated_only: bool = False) \
        -> List[Tuple[pd.DataFrame, Dict]]:
    """
    Update row names in all loaded DataFrames.

    Args:
        all_files: List of (DataFrame, parameters_dict) tuples
        s: String parameter to insert in row names
        return_updated_only:

    Returns:
        List of (updated_DataFrame, parameters_dict) tuples
    """
    updated_files = []

    for df, params in all_files:
        updated_df = update_row_names(df, s, return_updated_only=return_updated_only)
        updated_files.append((updated_df, params))

    return updated_files


def update_row_names(df: pd.DataFrame, s: str, return_updated_only: bool = False) -> pd.DataFrame:
    """
    Update row names in the first column from "NN-idx" format to "NN-{s}-idx" format.

    Args:
        df: DataFrame with row names in the first column
        s: String parameter to insert between NN and idx
        return_updated_only: If True, return only rows that were updated (matched the pattern)

    Returns:
        DataFrame with updated row names in the first column, or only updated rows if return_updated_only=True
    """
    # Create a copy to avoid modifying the original
    df_updated = df.copy()

    # Get the first column name
    first_col = df_updated.columns[0]

    # NN can be letters/numbers, idx is an integer
    pattern = r'^(NN)-(\d+)'

    # Create boolean mask for rows that match the pattern
    def matches_pattern(name):
        if pd.isna(name):
            return False
        name_str = str(name)
        return bool(re.match(pattern, name_str))

    # Apply mask to identify matching rows
    matching_mask = df_updated[first_col].apply(matches_pattern)

    # Function to replace matching patterns
    def replace_pattern(name):
        if pd.isna(name):
            return name
        name_str = str(name)
        match = re.match(pattern, name_str)
        if match:
            nn_part, idx_part = match.groups()
            return f"{nn_part}-{s}-{idx_part}"
        else:
            return name_str

    # Apply the transformation to the first column
    df_updated[first_col] = df_updated[first_col].apply(replace_pattern)

    # Return only updated rows if requested
    if return_updated_only:
        return df_updated[matching_mask]
    else:
        return df_updated


def load_violation_rate_csv_files(directory: str, ax_type: str) -> List[Tuple[pd.DataFrame, Dict[str, str]]]:
    """
    Load all CSV files from a directory and extract parameters.

    Args:
        directory: Path to directory containing CSV files
        ax_type: Either "all" or "reduced"

    Returns:
        List of tuples containing (DataFrame, parameters_dict)
    """
    results = []

    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return results

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    for filename in csv_files:
        filepath = os.path.join(directory, filename)

        # Parse filename to extract parameters
        params = parse_filename(filename, ax_type)

        if params is None:
            continue

        # Validate M and K constraints
        if params['M'] not in [5, 6, 7]:
            print(f"Warning: M value {params['M']} not in expected range [5,6,7] for file {filename}")
            continue

        if not (1 <= params['K'] < params['M']):
            print(f"Warning: K value {params['K']} not in valid range [1, {params['M'] - 1}] for file {filename}")
            continue

        try:
            # Load the CSV file
            df = pd.read_csv(filepath)
            results.append((df, params))

        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")

    return results


def relabel_all_distance_dfs(all_files: List[Tuple[pd.DataFrame, Dict]], s: str) -> List[Tuple[pd.DataFrame, Dict]]:
    """
    Update NN name in all distance files. It appears as both a row and column label.

    Args:
        all_files: List of (DataFrame, parameters_dict) tuples
        s: String parameter to insert in row names

    Returns:
        List of (updated_DataFrame, parameters_dict) tuples
    """
    def relabel_distance_df(df_to_edit):
        df_to_edit.columns = df_to_edit.columns.str.replace('NN', s)
        df_to_edit = df_to_edit.replace('NN', s)
        return df_to_edit

    updated_files = []
    for df, params in all_files:
        updated_df = relabel_distance_df(df)
        updated_files.append((updated_df, params))

    return updated_files


def load_differences_csv_files(directory: str, ax_type: str, varied_voters=False) -> List[Tuple[pd.DataFrame, Dict[str, str]]]:
    """
    Load all CSV files from a directory and extract parameters.

    Args:
        directory: Path to directory containing CSV files
        ax_type: Either "all" or "reduced"

    Returns:
        List of tuples containing (DataFrame, parameters_dict)
    """

    def parse_differences_filename(filename: str) -> Optional[Dict[str, str]]:
        """
        Parse filename to extract M, K, D, and AX parameters.
        Expected format: axiom_violation_results-n_profiles=25000-num_voters=50-m={M}-k={K}-pref_dist={D}-axioms={AX}.csv

        Args:
            filename: The CSV filename
            ax_type: Either "all" or "reduced" based on directory

        Returns:
            Dictionary with parsed parameters or None if parsing fails
        """
        # Remove .csv extension
        base_name = filename.replace('.csv', '')

        if varied_voters:
            pattern = "num_voters=50-varied_voters=False-voters_std_dev=0-m=(\d+)-k=(\d+)-pref_dist=(.+?)-distances"
        else:
            pattern = "num_voters=50-m=(\d+)-k=(\d+)-pref_dist=(.+?)-distances"
        # pattern = r'axiom_violation_results-n_profiles=\d+-num_voters=\d+-m=(\d+)-k=(\d+)-pref_dist=(.+?)-axioms=([^-]+)'

        match = re.search(pattern, base_name)
        if match:
            m_val, k_val, d_val = match.groups()

            return {
                'M': int(m_val),
                'K': int(k_val),
                'D': d_val,
                'AX': ax_type,  # Use the actual value from filename rather than ax_type
                'filename': filename
            }

        print(f"Warning: Could not parse filename: {filename}")
        return None
    results = []

    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return results

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    for filename in csv_files:
        filepath = os.path.join(directory, filename)

        # Parse filename to extract parameters
        params = parse_differences_filename(filename)

        if params is None:
            continue

        # Validate M and K constraints
        if params['M'] not in [5, 6, 7]:
            print(f"Warning: M value {params['M']} not in expected range [5,6,7] for file {filename}")
            continue

        if not (1 <= params['K'] < params['M']):
            print(f"Warning: K value {params['K']} not in valid range [1, {params['M'] - 1}] for file {filename}")
            continue

        try:
            # Load the CSV file
            df = pd.read_csv(filepath)
            results.append((df, params))

        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")

    return results

def merge_root_and_all_axiom_results():
    """
    Main function to load all CSV files from both directories.
    """
    # Define directories
    thesis_dir = "evaluation_results_thesis"
    root_axioms_dir = "evaluation_results-root_axioms"

    out_dir = "evaluation_results-AAAI_combined"

    # Load files from both directories
    print("Loading files from evaluation_results_thesis directory (AX=all)...")
    thesis_files = load_violation_rate_csv_files(thesis_dir, "all")

    print(f"\nLoading files from evaluation_results-root_axioms directory (AX=reduced)...")
    root_axioms_files = load_violation_rate_csv_files(root_axioms_dir, "reduced")

    thesis_files = update_all_row_names(thesis_files, s="all")
    root_axioms_files = update_all_row_names(root_axioms_files, s="root", return_updated_only=True)

    all_m = set()
    all_k = set()
    all_dists = set()

    all_ax_index = dict()
    root_ax_index = dict()

    for df, params in thesis_files:
        all_m.add(params["M"])
        all_k.add(params["K"])
        all_dists.add(params["D"])
        if (params["M"], params["K"], params["D"]) in all_ax_index:
            raise ValueError("params already exist in all_ax")
        all_ax_index[(params["M"], params["K"], params["D"])] = df
    for df, params in root_axioms_files:
        if params["M"] not in all_m:
            raise ValueError("Unexpected.")
        if params["K"] not in all_k:
            raise ValueError("Unexpected.")
        if params["D"] not in all_dists:
            raise ValueError("Unexpected.")
        if (params["M"], params["K"], params["D"]) in root_ax_index:
            raise ValueError("params already exist in root_ax")
        root_ax_index[(params["M"], params["K"], params["D"])] = df

    for m, k, d in itertools.product(list(all_m), list(all_k), list(all_dists)):
        if k >= m:
            continue

        skip_merge = False
        if (m, k, d) not in all_ax_index:
            raise ValueError("params don't exist in all_ax")
        if (m, k, d) not in root_ax_index:
            skip_merge = True
            print(f"WARNING: Skipping params for root ax with dist={d}")

        all_ax_df = all_ax_index[(m, k, d)]

        # merge the two dfs
        if skip_merge:
            merged_df = all_ax_df
        else:
            root_ax_df = root_ax_index[(m, k, d)]
            merged_df = pd.concat([root_ax_df, all_ax_df], ignore_index=True)

        name = f"axiom_violation_results-n_profiles=25000-num_voters=50-m={m}-k={k}-pref_dist={d}-axioms=both"
        full_name = f"{out_dir}/{name}.csv"
        merged_df.to_csv(full_name, index=False)



def merge_difference_tables():
    """

    :return:
    """
    # Define directories
    thesis_dir = "evaluation_results_thesis/rule_distances"
    root_axioms_dir = "evaluation_results-root_axioms/rule_distances"

    out_dir = "evaluation_results-AAAI_combined/rule_distances"

    # Load files from both directories
    print("Loading files from evaluation_results_thesis directory (AX=all)...")
    thesis_files = load_differences_csv_files(thesis_dir, "all", varied_voters=False)

    print(f"\nLoading files from evaluation_results-root_axioms directory (AX=reduced)...")
    root_axioms_files = load_differences_csv_files(root_axioms_dir, "reduced", varied_voters=True)

    thesis_files = relabel_all_distance_dfs(thesis_files, s="NN-all")
    root_axioms_files = relabel_all_distance_dfs(root_axioms_files, s="NN-root")

    all_m = set()
    all_k = set()
    all_dists = set()

    all_ax_index = dict()
    root_ax_index = dict()

    for df, params in thesis_files:
        all_m.add(params["M"])
        all_k.add(params["K"])
        all_dists.add(params["D"])
        if (params["M"], params["K"], params["D"]) in all_ax_index:
            raise ValueError("params already exist in all_ax")
        all_ax_index[(params["M"], params["K"], params["D"])] = df
    for df, params in root_axioms_files:
        if params["M"] not in all_m:
            raise ValueError("Unexpected.")
        if params["K"] not in all_k:
            raise ValueError("Unexpected.")
        if params["D"] not in all_dists:
            raise ValueError("Unexpected.")
        if (params["M"], params["K"], params["D"]) in root_ax_index:
            raise ValueError("params already exist in root_ax")
        root_ax_index[(params["M"], params["K"], params["D"])] = df

    for m, k, d in itertools.product(list(all_m), list(all_k), list(all_dists)):
        if k >= m:
            continue

        skip_merge = False
        if (m, k, d) not in all_ax_index:
            raise ValueError("params don't exist in all_ax")
        if (m, k, d) not in root_ax_index:
            skip_merge = True
            print(f"WARNING: Skipping params for root ax with dist={d}")

        all_ax_df = all_ax_index[(m, k, d)]
        # drop first column; it only has redundant info and makes the format harder
        if len(all_ax_df.columns) > 1 and all_ax_df.columns[1] == "NN-all":
            all_ax_df = all_ax_df.drop(all_ax_df.columns[1], axis=1)

        # merge the two dfs
        if skip_merge:
            merged_df = all_ax_df
        else:
            root_ax_df = root_ax_index[(m, k, d)]
            # drop first column; it only has redundant info and makes the format harder
            if len(root_ax_df.columns) > 1 and root_ax_df.columns[1] == "NN-root":
                root_ax_df = root_ax_df.drop(root_ax_df.columns[1], axis=1)

            merged_df = all_ax_df.copy()

            # Check if final row of root_ax_df should be added
            if not root_ax_df.empty and root_ax_df.iloc[-1, 0] == "NN-root":
                # Add the final row of root_ax_df
                merged_df = pd.concat([merged_df, root_ax_df.iloc[[-1]]], ignore_index=True)

        # name = f"axiom_violation_results-n_profiles=25000-num_voters=50-m={m}-k={k}-pref_dist={d}-axioms=both"
        name = f"num_voters=50-m={m}-k={k}-pref_dist={d}-axioms=both-distances"
        full_name = f"{out_dir}/{name}.csv"
        merged_df.to_csv(full_name, index=False)


if __name__ == "__main__":
    # merge_root_and_all_axiom_results()
    merge_difference_tables()
