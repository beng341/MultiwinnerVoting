#!/bin/bash
#SBATCH --account=def-klarson
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --time=3:00:00
#SBATCH --mail-user=b8armstr@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_out/%j.out                   # Log will be written to job_name_job_id.out'



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

python -m network_ops.train_networks "m=6" "num_winners=4" "pref_dist='euclidean__args__dimensions=3_-_space=uniform_ball'" "data_path='/scratch/b8armstr/data'" "out_folder='/scratch/b8armstr'"
# python -m network_ops.evaluate_networks "m=6" "num_winners=4" "data_path='/scratch/b8armstr/data'" "out_folder='evaluation_results_fixed_fm'" "network_path='/scratch/b8armstr/trained_networks'"

