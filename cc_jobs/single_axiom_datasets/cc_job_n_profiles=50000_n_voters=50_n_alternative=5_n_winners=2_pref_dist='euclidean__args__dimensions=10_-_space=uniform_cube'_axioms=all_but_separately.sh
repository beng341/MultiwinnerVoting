#!/bin/bash
    #SBATCH --account=def-klarson
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=8G
    #SBATCH --time=14:00:00
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

    echo "About to start experiments"

    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['dummett']" "out_folder='data'"
    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['consensus']" "out_folder='data'"
    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['fixed_majority']" "out_folder='data'"
    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['majority_winner']" "out_folder='data'"
    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['majority_loser']" "out_folder='data'"
    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['condorcet_winner']" "out_folder='data'"
    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['condorcet_loser']" "out_folder='data'"
    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['solid_coalition']" "out_folder='data'"
    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['strong_unanimity']" "out_folder='data'"
    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['local_stability']" "out_folder='data'"
    python -m network_ops.generate_data "n_profiles=50000" "prefs_per_profile=50" "m=5" "num_winners=2" "learned_pref_model='euclidean__args__dimensions=10_-_space=uniform_cube'" "axioms=['strong_pareto']" "out_folder='data'"

    