import copy
import itertools
import os.path
from scipy.special import binom

generic_job = """#!/bin/bash
#SBATCH --account=def-klarson
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=$JOB_TIME
#SBATCH --mail-user=$EMAIL_TO_NOTIFY
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

python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "pref_model=$PREF_MODEL" "axioms=$AXIOMS" "out_folder=$OUT_FOLDER"

"""

train_job = """#!/bin/bash
#SBATCH --account=def-klarson
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --time=$JOB_TIME
#SBATCH --mail-user=$EMAIL_TO_NOTIFY
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

python -m network_ops.train_networks "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "pref_dist='$PREF_DIST'" "data_path='/scratch/b8armstr/data'" "out_folder='$NETWORK_FOLDER'"
# python -m network_ops.evaluate_networks "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "data_path='/scratch/b8armstr/data'" "out_folder='$OUT_FOLDER'" "network_path='$NETWORK_FOLDER/trained_networks'"

"""

eval_job = """#!/bin/bash
#SBATCH --account=def-klarson
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --time=$JOB_TIME
#SBATCH --mail-user=$EMAIL_TO_NOTIFY
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

python -m network_ops.evaluate_networks "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "data_path='/scratch/b8armstr/data'" "out_folder='$OUT_FOLDER'" "network_path='/scratch/b8armstr/thesis_results/trained_networks'"


"""


all_pref_models = [
        "stratification__args__weight=0.5",         # Complete on all dists
        "URN-R",                                    # Complete on all dists
        "IC",                                       # Complete on all dists
        "IAC",                                      # Complete on all dists
        "identity",                                 # Complete on all dists
        "MALLOWS-RELPHI-R",
        "single_peaked_conitzer",
        "single_peaked_walsh",
        "euclidean__args__dimensions=3_-_space=gaussian_ball",      # m = 6 is done to here
        "euclidean__args__dimensions=10_-_space=gaussian_ball",
        "euclidean__args__dimensions=3_-_space=uniform_ball",       # m = 5 is done to here
        "euclidean__args__dimensions=10_-_space=uniform_ball",
        "euclidean__args__dimensions=3_-_space=gaussian_cube",
        "euclidean__args__dimensions=10_-_space=gaussian_cube",
        "euclidean__args__dimensions=3_-_space=uniform_cube",
        "euclidean__args__dimensions=10_-_space=uniform_cube",
        "mixed"
    ]

all_axioms = [
        "dummett",
        "consensus",
        "fixed_majority",
        "majority_winner",
        "majority_loser",
        "condorcet_winner",
        "condorcet_loser",
        "solid_coalition",
        "strong_unanimity",
        "local_stability",
        "strong_pareto"
    ]


def generate_jobs(n_profiles, n_all, m_all, k_all, pref_dist_all, axioms="all", folder="cc_jobs/jobs", data_folder="data", job_time="48:00:00", email="b8armstr@uwaterloo.ca"):
    """

    :param n_profiles:
    :param n_all:
    :param m_all:
    :param k_all:
    :param pref_dist_all:
    :param axioms:
    :param folder: Where the job script itself is saved
    :param data_folder: Where the job will save the data it generates
    :param job_time: string formatted time requested for job
    :param email: email to notify when job is received, starts, and stops
    :return:
    """
    if not isinstance(axioms, list):
        axioms = [axioms]

    for n, m, k, pref_dist, axiom in itertools.product(n_all, m_all, k_all, pref_dist_all, axioms):

        if k >= m:
            continue

        hours = binom(m, k)
        rhours = round(hours)
        print(f"Giving (n={n}, m={m}, k={k}) time: {rhours}, from {hours}")
        job_time = f"{rhours}:00:00"

        generate_single_job_with_params(n_profiles, n, m, k, pref_dist, axiom,
                                        job_file_folder=folder,
                                        data_folder=data_folder,
                                        job_time=job_time,
                                        email=email)


def generate_single_job_with_params(n_profiles, n, m, k, pref_dist, axioms, job_file_folder, data_folder="data", job_time="48:00:00", email="b8armstr@uwaterloo.ca"):
    """

    :param n_profiles:
    :param n:
    :param m:
    :param k:
    :param pref_dist:
    :param axioms:
    :param job_file_folder:
    :param job_time:
    :param email:
    :return:
    """
    if not os.path.exists(job_file_folder):
        os.makedirs(job_file_folder)

    # string arguments need extra quotation marks
    pref_dist = f"'{pref_dist}'"
    data_folder = f"'{data_folder}'"
    if not isinstance(axioms, list):
        axioms = f"'{axioms}'"

    keys_to_replace = {
        "$JOB_TIME": job_time,
        "$EMAIL_TO_NOTIFY": email,
        "$N_PROFILES": f"{n_profiles}",
        "$N_VOTERS": f"{n}",
        "$N_ALTERNATIVES": f"{m}",
        "$N_WINNERS": f"{k}",
        "$PREF_MODEL": f"{pref_dist}",
        "$AXIOMS": f"{axioms}",
        "$OUT_FOLDER": data_folder,
    }

    new_job = copy.copy(generic_job)
    for key, value in keys_to_replace.items():
        new_job = new_job.replace(key, value)

    job_filename = f"cc_job_n_profiles={n_profiles}_n_voters={n}_n_alternative={m}_n_winners={k}_pref_dist={pref_dist}_axioms={axioms}.sh"
    job_filename = os.path.join(job_file_folder, job_filename)
    with open(job_filename, "w") as f:
        f.write(new_job)


def make_small_generation_jobs():
    n_profiles = 1000
    n_all = [50]
    m_all = [5]
    # m_all = [5]
    k_all = [1, 2, 3, 4, 5, 6]
    pref_dist_all = all_pref_models
    axioms = "all"

    job_file_location = "cc_jobs/small_jobs"
    # data_out_location = "data"
    data_out_location = "$HOME/scratch/data"
    email = "b8armstr@uwaterloo.ca"

    if not isinstance(axioms, list):
        axioms = [axioms]

    for n, m, k, pref_dist, axiom in itertools.product(n_all, m_all, k_all, pref_dist_all, axioms):

        if k >= m:
            continue

        job_time = f"1:00:00"
        generate_single_job_with_params(n_profiles, n, m, k, pref_dist, axiom,
                                        job_file_folder=job_file_location,
                                        data_folder=data_out_location,
                                        job_time=job_time,
                                        email=email)


def make_data_generation_jobs():
    n_profiles = 25000
    n_all = [50]
    m_all = [5, 6, 7]
    # m_all = [5]
    k_all = [1, 2, 3, 4, 5, 6]
    pref_dist_all = all_pref_models
    if "mixed" in pref_dist_all:
        pref_dist_all.remove("mixed")
    axioms = "all"

    job_file_location = "cc_jobs/data_generation"
    # data_out_location = "data"
    # data_out_location = "$HOME/scratch/data"
    data_out_location = "/scratch/b8armstr/data"
    email = "b8armstr@uwaterloo.ca"

    if not isinstance(axioms, list):
        axioms = [axioms]

    for n, m, k, pref_dist, axiom in itertools.product(n_all, m_all, k_all, pref_dist_all, axioms):

        if k >= m:
            continue

        hours = 1.5 * binom(m, k)
        hours = (k ** 0.6) * binom(m, k) + 1
        rhours = round(hours)
        print(f"Giving (n={n}, m={m}, k={k}) time: {rhours}, from {hours}")
        job_time = f"{rhours}:00:00"

        generate_single_job_with_params(n_profiles, n, m, k, pref_dist, axiom,
                                        job_file_folder=job_file_location,
                                        data_folder=data_out_location,
                                        job_time=job_time,
                                        email=email)


def make_single_axiom_dataset_jobs():
    axiom_job = """#!/bin/bash
    #SBATCH --account=def-klarson
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=8G
    #SBATCH --time=$JOB_TIME
    #SBATCH --mail-user=$EMAIL_TO_NOTIFY
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

    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['dummett']" "out_folder=$OUT_FOLDER"
    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['consensus']" "out_folder=$OUT_FOLDER"
    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['fixed_majority']" "out_folder=$OUT_FOLDER"
    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['majority_winner']" "out_folder=$OUT_FOLDER"
    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['majority_loser']" "out_folder=$OUT_FOLDER"
    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['condorcet_winner']" "out_folder=$OUT_FOLDER"
    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['condorcet_loser']" "out_folder=$OUT_FOLDER"
    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['solid_coalition']" "out_folder=$OUT_FOLDER"
    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['strong_unanimity']" "out_folder=$OUT_FOLDER"
    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['local_stability']" "out_folder=$OUT_FOLDER"
    python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=['strong_pareto']" "out_folder=$OUT_FOLDER"

    """

    n_profiles = 10000
    n_all = [50]
    m_all = [5, 6, 7]
    # m_all = [5]
    k_all = [1, 2, 3, 4, 5, 6]
    pref_dist_all = all_pref_models

    job_file_location = "cc_jobs/single_axiom_datasets"
    data_out_location = "$HOME/scratch/data"
    email = "b8armstr@uwaterloo.ca"

    if not os.path.exists(job_file_location):
        os.makedirs(job_file_location)

    for n, m, k, pref_dist in itertools.product(n_all, m_all, k_all, pref_dist_all):

        if k >= m:
            continue

        # should be much faster when considering only one axiom (at least, for most axioms)
        hours = 1.3 * (k ** 0.5) * binom(m, k)
        rhours = round(hours)
        print(f"Giving (n={n}, m={m}, k={k}) time: {rhours}, from {hours}")
        job_time = f"{rhours}:00:00"

        # string arguments need extra quotation marks
        pref_dist = f"'{pref_dist}'"
        data_folder = f"'{data_out_location}'"

        keys_to_replace = {
            "$JOB_TIME": job_time,
            "$EMAIL_TO_NOTIFY": email,
            "$N_PROFILES": f"{n_profiles}",
            "$N_VOTERS": f"{n}",
            "$N_ALTERNATIVES": f"{m}",
            "$N_WINNERS": f"{k}",
            "$PREF_MODEL": f"{pref_dist}",
            "$OUT_FOLDER": data_folder,
        }

        new_job = copy.copy(axiom_job)
        for key, value in keys_to_replace.items():
            new_job = new_job.replace(key, value)

        job_filename = f"cc_job_n_profiles={n_profiles}_n_voters={n}_n_alternative={m}_n_winners={k}_pref_dist={pref_dist}_axioms=all_but_separately.sh"
        job_filename = os.path.join(job_file_location, job_filename)
        with open(job_filename, "w") as f:
            f.write(new_job)


def make_training_jobs():

    job_file_location = "cc_jobs/train_eval_jobs"
    if not os.path.exists(job_file_location):
        os.makedirs(job_file_location)

    m_all = [5, 6, 7]
    k_all = [1, 2, 3, 4, 5, 6]
    for m, k, pref_dist in itertools.product(m_all, k_all, all_pref_models):

        if k >= m:
            continue

        rhours = m*3
        # if k == 1:
        #     rhours = m*6
        print(f"Giving (n=50, m={m}, k={k}) time: {rhours}")
        job_time = f"{rhours}:00:00"

        keys_to_replace = {
            "$JOB_TIME": job_time,
            "$EMAIL_TO_NOTIFY": "b8armstr@uwaterloo.ca",
            "$N_ALTERNATIVES": f"{m}",
            "$N_WINNERS": f"{k}",
            "$PREF_DIST": f"{pref_dist}",
            "$OUT_FOLDER": "evaluation_results_thesis",
            "$NETWORK_FOLDER": "/scratch/b8armstr/thesis_results"
        }

        new_job = copy.copy(train_job)
        for key, value in keys_to_replace.items():
            new_job = new_job.replace(key, value)

        job_filename = f"cc_job_train_eval_n_profiles=25000_n_voters=50_n_alternative={m}_n_winners={k}_pref_dist={pref_dist}_axioms=all.sh"
        job_filename = os.path.join(job_file_location, job_filename)
        with open(job_filename, "w") as f:
            f.write(new_job)


def make_evaluation_jobs():

    job_file_location = "cc_jobs/eval_jobs"
    if not os.path.exists(job_file_location):
        os.makedirs(job_file_location)

    m_all = [5, 6, 7]
    k_all = [1, 2, 3, 4, 5, 6]
    for m, k in itertools.product(m_all, k_all):

        if k >= m:
            continue

        rhours = int(m*2)
        # rhours = m
        print(f"Giving (n=50, m={m}, k={k}) time: {rhours}")
        job_time = f"{rhours}:00:00"

        keys_to_replace = {
            "$JOB_TIME": job_time,
            "$EMAIL_TO_NOTIFY": "b8armstr@uwaterloo.ca",
            "$N_ALTERNATIVES": f"{m}",
            "$N_WINNERS": f"{k}",
            "$OUT_FOLDER": "evaluation_results_thesis_with_min_and_max",
        }

        new_job = copy.copy(eval_job)
        for key, value in keys_to_replace.items():
            new_job = new_job.replace(key, value)

        job_filename = f"cc_job_eval_n_profiles=25000_n_voters=50_n_alternative={m}_n_winners={k}_pref_dist=all_axioms=all.sh"
        job_filename = os.path.join(job_file_location, job_filename)
        with open(job_filename, "w") as f:
            f.write(new_job)


if __name__ == "__main__":
    # make_single_axiom_dataset_jobs()
    # make_data_generation_jobs()
    make_evaluation_jobs()
    # make_small_generation_jobs()
    # make_training_jobs()
