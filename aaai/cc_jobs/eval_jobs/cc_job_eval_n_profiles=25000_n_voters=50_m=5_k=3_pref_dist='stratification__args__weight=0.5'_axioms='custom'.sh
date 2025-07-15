#!/bin/bash
#SBATCH --account=def-klarson
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --time=10:00:00
#SBATCH --mail-user=jcaiata@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_out/%j.out

date +%s
echo "About to load modules"

module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack gurobi/11.0.1

# Create virtual environment for python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

date +%s
echo "About to install requirements"

# install all requirements
pip install --no-index deprecated
pip install --no-deps -U cc_libs/*.whl
pip install --no-index -U scikit_learn llvmlite ortools
pip install --no-index torch

echo "About to start experiments"

python -m network_ops.evaluate_networks "n_profiles=25000" "n_voters=50" "varied_voters=False" "voters_std_dev=0" "m=5" "num_winners=3" "data_path='aaai/results/data'" "network_path='aaai/results/trained_networks/trained_networks'" "out_folder='aaai/results/results'" "pref_dist='stratification__args__weight=0.5'" "axioms=['local_stability', 'dummetts_condition', 'condorcet_winner', 'strong_pareto_efficiency', 'core', 'majority_loser']"
