# Print start timestamp
date +%s
echo "About to load modules"

# Load Compute Canada modules
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack gurobi/11.0.1

# Create and activate virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

date +%s
echo "About to install requirements"

# Install Python packages from local wheels
pip install --no-index deprecated
pip install --no-deps cc_libs/llvmlite-*.whl

for f in cc_libs/*.whl; do
  [[ "$f" == *llvmlite* ]] && continue
  pip install --no-deps "$f"
done

pip install --no-index -U scikit_learn ortools torch

echo "About to start experiments"

# Run your experiment script
python -m network_ops.evaluate_networks \
  "n_profiles=250" \
  "n_voters=50" \
  "varied_voters=False" \
  "voters_std_dev=0" \
  "m=5" \
  "num_winners=1" \
  "data_path='aaai/results/data'" \
  "network_path='aaai/results/trained_networks/trained_networks'" \
  "out_folder='aaai/results/results'" \
  "pref_dist='euclidean__args__dimensions=3_-_space=gaussian_ball'" \
  "axioms=['local_stability','dummetts_condition','condorcet_winner','strong_pareto_efficiency','core','majority_loser']"
