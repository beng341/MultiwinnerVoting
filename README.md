# Generating Data:

## network_ops.generate_data

The `network_ops.generate_data` module generates synthetic voting datasets for training and testing multi-winner voting rules. It creates preference profiles using various probability models and computes optimal/worst committees according to specified axioms.

### Usage

Run the script with command-line arguments in `key=value` format:

```bash
python -m network_ops.generate_data n_profiles=1000 prefs_per_profile=50 m=5 num_winners=3 pref_model="IC" axioms="all" out_folder="data"
```

### Required Arguments

- **`n_profiles`** (int): Number of preference profiles to generate
- **`prefs_per_profile`** (int): Number of voters per profile (mean if varied_voters=True)
- **`m`** (int): Number of candidates in each profile
- **`num_winners`** (int): Size of committees to elect
- **`pref_model`** (str): Preference generation model (see options below)
- **`axioms`** (str): Which axiom set to evaluate ("all", "root", "both")
- **`out_folder`** (str): Directory to save generated datasets

### Optional Arguments

- **`varied_voters`** (bool): If True, vary number of voters per profile. Default: False
- **`voters_std_dev`** (int): Standard deviation for voter count when varied_voters=True. Default: 0
- **`rwd_folder`** (str): Path to real-world data folder (required for pref_model="real_world")

### Preference Models

Available preference models include:

#### Standard Models
- **`"IC"`** - Impartial Culture (uniform random)
- **`"IAC"`** - Impartial Anonymous Culture  
- **`"URN-R"`** - Urn model
- **`"identity"`** - All voters have identical preferences
- **`"MALLOWS-RELPHI-R"`** - Mallows model

#### Spatial Models
- **`"euclidean__args__dimensions=3_-_space=gaussian_ball"`**
- **`"euclidean__args__dimensions=10_-_space=gaussian_ball"`** 
- **`"euclidean__args__dimensions=3_-_space=uniform_ball"`**
- **`"euclidean__args__dimensions=10_-_space=uniform_ball"`**
- **`"euclidean__args__dimensions=3_-_space=gaussian_cube"`**
- **`"euclidean__args__dimensions=10_-_space=gaussian_cube"`**
- **`"euclidean__args__dimensions=3_-_space=uniform_cube"`**
- **`"euclidean__args__dimensions=10_-_space=uniform_cube"`**

#### Structured Models  
- **`"single_peaked_conitzer"`** - Single-peaked preferences (Conitzer)
- **`"single_peaked_walsh"`** - Single-peaked preferences (Walsh)
- **`"stratification__args__weight=0.5"`** - Stratified model

#### Special Models
- **`"mixed"`** - Mixed distribution
- **`"real_world"`** - Generated from real-world voting data (requires rwd_folder)

### Output

The script generates CSV files with the following naming convention:
```
n_profiles={n}-num_voters={v}-varied_voters={vv}-voters_std_dev={std}-m={m}-committee_size={c}-pref_dist={model}-axioms={axioms}-{TYPE}.csv
```

Where `{TYPE}` is either `TRAIN` or `TEST`.

### Generated Data Columns

Each row represents one preference profile with columns including:
- **`Profile`**: The preference profile as rankings
- **`n_voters`**: Number of voters in this profile
- **`min_violations-committee`**: Committee that minimizes axiom violations
- **`min_violations`**: Number of violations for optimal committee
- **`max_violations-committee`**: Committee that maximizes axiom violations  
- **`max_violations`**: Number of violations for worst committee
- **Feature columns**: Candidate pair matrices, binary pairs, rank matrices (normalized)
- **Rule winner columns** (TEST data only): Winners for each voting rule

### Examples

Generate IC data with 1000 profiles:
```bash
python -m network_ops.generate_data n_profiles=1000 prefs_per_profile=50 m=5 num_winners=3 pref_model="IC" axioms="all" out_folder="results"
```

Generate varied voter data:
```bash
python -m network_ops.generate_data n_profiles=500 prefs_per_profile=50 varied_voters=True voters_std_dev=10 m=6 num_winners=2 pref_model="URN-R" axioms="both" out_folder="data"
```

Generate real-world based data:
```bash
python -m network_ops.generate_data n_profiles=100 prefs_per_profile=20 m=5 num_winners=2 pref_model="real_world" rwd_folder="data/real_world_data" axioms="all" out_folder="rwd_results"
```

### Notes

- The script automatically generates both TRAIN and TEST datasets
- Large datasets are saved incrementally to prevent data loss
- Real-world preference generation requires existing real-world voting data in the specified folder
- Preference models with `__args__` use the specified parameters (e.g., dimensions for euclidean models)

# Training Networks
## network_ops.train_networks

The `network_ops.train_networks` module trains neural networks to learn multi-winner voting rules from generated data. It trains multiple networks using different feature sets and loss functions for robust comparison.

### Usage

```bash
python -m network_ops.train_networks n_profiles=10000 n_voters=50 m=5 num_winners=3 data_path="data" varied_voters=True out_folder="trained_networks"
```

### Required Arguments

- **`n_profiles`** (int): Number of training profiles to use
- **`n_voters`** (int): Number of voters per profile (must match training data)
- **`m`** (int): Number of candidates (must match training data)
- **`num_winners`** (int): Committee size (must match training data)
- **`data_path`** (str): Directory containing training data (generated by generate_data)
- **`varied_voters`** (bool): Whether training data has varied voters (must match training data)
- **`out_folder`** (str): Directory to save trained networks

### Optional Arguments

- **`axioms`** (str): Axiom set used in training data ("all", "root", "both"). Default: "all"
- **`pref_dist`** (str): Specific preference distribution to train on. If not provided, trains on all available distributions
- **`voters_std_dev`** (int): Standard deviation for varied voters (must match training data if applicable)

### Training Process

The script automatically:
1. **Loads training data** from the specified data_path
2. **Trains multiple networks** per parameter set using different:
   - Feature sets (candidate pairs, binary pairs, rank matrices)
   - Loss functions (MSE, etc.)
3. **Saves trained models** with descriptive filenames
4. **Records training losses** for analysis

### Network Architecture

- **Hidden layers**: 5
- **Hidden nodes**: 256 per layer
- **Epochs**: 50
- **Early stopping**: Enabled with patience
- **Optimizer**: Adam

### Output

Trained networks are saved with naming convention:
```
NN-num_voters={n}-m={m}-num_winners={k}-pref_dist={dist}-axioms={axioms}-features={features}-loss={loss}-idx={idx}-.pt
```

### Examples

Train networks for IC distribution:
```bash
python -m network_ops.train_networks n_profiles=10000 n_voters=50 m=5 num_winners=3 data_path="data" varied_voters=False pref_dist="IC" out_folder="networks"
```

Train on all distributions with varied voters:
```bash
python -m network_ops.train_networks n_profiles=5000 n_voters=50 m=6 num_winners=2 data_path="data" varied_voters=True voters_std_dev=10 out_folder="trained_networks"
```

# Evaluating Networks and Rules
## network_ops.evaluate_networks

The `network_ops.evaluate_networks` module evaluates trained neural networks against test data and compares their performance with existing multi-winner voting rules.

### Usage

```bash
python -m network_ops.evaluate_networks n_profiles=1000 n_voters=50 varied_voters=True voters_std_dev=10 m=5 num_winners=3 data_path="data" network_path="trained_networks" out_folder="evaluation_results"
```

### Required Arguments

- **`n_profiles`** (int): Number of test profiles to evaluate on
- **`n_voters`** (int): Number of voters per profile (must match test data)
- **`varied_voters`** (bool): Whether test data has varied voters (must match test data)
- **`voters_std_dev`** (int): Standard deviation for varied voters (must match test data)
- **`m`** (int): Number of candidates (must match test data)
- **`num_winners`** (int): Committee size (must match test data)
- **`data_path`** (str): Directory containing test data
- **`network_path`** (str): Directory containing trained networks
- **`out_folder`** (str): Directory to save evaluation results

### Optional Arguments

- **`axioms`** (str): Axiom set to evaluate ("all", "root", "both"). Default: "all"
- **`pref_dist`** (str): Specific preference distribution to evaluate. If not provided, evaluates all available distributions

### Evaluation Process

The script:
1. **Loads test data** and trained networks
2. **Makes predictions** using neural networks
3. **Compares against existing rules**:
   - Borda, STV, PAV, Monroe, etc.
   - Random choice baseline
   - Min/Max violation committees
4. **Computes distances** between all rule pairs
5. **Counts axiom violations** for each rule
6. **Saves comprehensive results**

### Output Files

Generated files include:
- **Axiom violation results**: `axiom_violation_results-{params}.csv`
- **Rule distance matrices**: `rule_distances-{params}.csv`
- **Performance comparisons**: Accuracy and violation statistics

### Examples

Evaluate networks on IC test data:
```bash
python -m network_ops.evaluate_networks n_profiles=1000 n_voters=50 varied_voters=False voters_std_dev=0 m=5 num_winners=3 data_path="data" network_path="trained_networks" pref_dist="IC" out_folder="results"
```

Comprehensive evaluation across all distributions:
```bash
python -m network_ops.evaluate_networks n_profiles=5000 n_voters=50 varied_voters=True voters_std_dev=10 m=6 num_winners=2 data_path="data" network_path="trained_networks" out_folder="evaluation_results"
```

### Notes

- Test data parameters must exactly match training data parameters
- Networks must exist for the specified parameters
- Evaluation compares neural networks against 15+ established voting rules
- Results enable analysis of when neural networks perform well vs. poorly





