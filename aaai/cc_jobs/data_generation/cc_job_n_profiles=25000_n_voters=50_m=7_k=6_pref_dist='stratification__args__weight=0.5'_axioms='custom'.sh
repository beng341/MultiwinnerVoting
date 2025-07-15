#!/bin/bash
#SBATCH --account=def-klarson
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=6:00:00
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

echo "About to start experiments"

python -m network_ops.generate_data "n_profiles=25000" "prefs_per_profile=50" "m=7" "num_winners=6" "learned_pref_model='stratification__args__weight=0.5'" "pref_model='stratification__args__weight=0.5'" "axioms=['local_stability', 'dummetts_condition', 'condorcet_winner', 'strong_pareto_efficiency', 'core', 'majority_loser']" "out_folder='aaai/results/data'" "varied_voters=False" "voters_std_dev=0" "rwd_folder=''" "train=True" "test=True"
