import os
from typing import Dict, List, Optional

# Parameter abbreviation mappings
PARAM_ABBREV = {
    "n_profiles": "n",
    "num_voters": "v", 
    "varied_voters": "vv",
    "voters_std_dev": "std",
    "m": "m",
    "committee_size": "c",
    "num_winners": "c",  # alias for committee_size
    "pref_dist": {
        "euclidean__args__dimensions": "euc_d",
        "_-_space=gaussian_ball": "_gb",
        "_-_space=gaussian_cube": "_gc",
        "_-_space=uniform_ball": "_ub",
        "_-_space=uniform_cube": "_uc",
        "stratification__args__weight": "strat",
        "URN-R": "urn",
        "IC": "ic",
        "IAC": "iac",
        "MALLOWS-RELPHI-R": "mall",
        "single_peaked_conitzer": "spc",
        "single_peaked_walsh": "spw",
        "mixed": "mix"
    }
}

# Axiom abbreviations 
AXIOM_ABBREV = {
    "local_stability": "ls",
    "dummetts_condition": "dc", 
    "condorcet_winner": "cw",
    "strong_pareto_efficiency": "spe",
    "core": "c",
    "majority_loser": "ml"
}

def abbreviate_param(param: str, value: str) -> str:
    """Convert a parameter value to its abbreviated form"""
    if param in PARAM_ABBREV:
        if isinstance(PARAM_ABBREV[param], dict):
            # Handle special cases like pref_dist
            result = value
            for full, abbrev in PARAM_ABBREV[param].items():
                result = result.replace(full, abbrev)
            # Clean up any remaining special characters
            result = result.replace("'", "").replace('"', "")
            return result
        else:
            # Handle simple parameter abbreviations
            return f"{PARAM_ABBREV[param]}{value}"
    return value

def abbreviate_axioms(axioms: List[str]) -> str:
    """Convert a list of axioms to abbreviated form"""
    if axioms == "all":
        return "all"
    if isinstance(axioms, str):
        # Handle string input
        if axioms.startswith("[") and axioms.endswith("]"):
            # Convert string representation of list to actual list
            axioms = eval(axioms)
        else:
            return axioms
    return "_".join(AXIOM_ABBREV.get(ax, ax) for ax in axioms)

def get_param_dir(params: Dict[str, any]) -> str:
    """Generate the directory path based on parameters"""
    n = abbreviate_param("n_profiles", str(params["n_profiles"]))
    v = abbreviate_param("num_voters", str(params["num_voters"]))
    m = abbreviate_param("m", str(params["m"]))
    c = abbreviate_param("num_winners", str(params["num_winners"]))
    
    # Handle preference distribution
    pref_dist = abbreviate_param("pref_dist", str(params["pref_dist"]))
    
    # Create directory structure
    return os.path.join(
        f"{n}_{v}",
        f"{m}_{c}",
        pref_dist
    )

def get_filename(params: Dict[str, any], prefix: str = "", suffix: str = "") -> str:
    """Generate abbreviated filename based on parameters"""
    parts = []
    
    # Add varied voters and std dev if present
    if "varied_voters" in params:
        parts.append(abbreviate_param("varied_voters", str(params["varied_voters"]).lower()[0]))
    if "voters_std_dev" in params:
        parts.append(abbreviate_param("voters_std_dev", str(params["voters_std_dev"])))
        
    # Add axioms if present
    if "axioms" in params:
        parts.append(abbreviate_axioms(params["axioms"]))
        
    # Add any additional identifiers
    if "features" in params:
        parts.append(f"feat_{params['features']}")
    if "loss" in params:
        parts.append(f"loss_{params['loss']}")
    if "idx" in params:
        parts.append(f"idx_{params['idx']}")
        
    # Add prefix/suffix
    if prefix:
        parts.insert(0, prefix)
    if suffix:
        parts.append(suffix)
        
    return "_".join(filter(None, parts)) + ".csv"

def get_full_path(base_dir: str, params: Dict[str, any], prefix: str = "", suffix: str = "", data_type: str = "results") -> str:
    """Generate complete file path with directory structure and abbreviated filename
    
    Args:
        base_dir: Base directory (e.g., 'aaai/results')
        params: Dictionary of parameters
        prefix: Optional prefix for filename
        suffix: Optional suffix for filename
        data_type: Type of data ('data', 'results', 'networks'). Defaults to 'results'
    """
    # Handle data type subdirectories
    if data_type == "data":
        # Data files go in data subdirectory
        base_dir = os.path.join(base_dir, "data")
    elif data_type == "results":
        # Results go in results subdirectory
        base_dir = os.path.join(base_dir, "results")
    elif data_type == "networks":
        # Networks go in trained_networks subdirectory
        base_dir = os.path.join(base_dir, "trained_networks")
    
    # Get parameter-based directory structure
    param_dir = get_param_dir(params)
    
    # Get filename
    filename = get_filename(params, prefix, suffix)
    
    # Combine everything
    full_path = os.path.join(base_dir, param_dir, filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    return full_path 