import copy
import itertools
import os.path

generic_job = """
#!/bin/bash
#SBATCH --account=def-klarson
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
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

python -m network_ops.generate_data "n_profiles=$N_PROFILES" "prefs_per_profile=$N_VOTERS" "m=$N_ALTERNATIVES" "num_winners=$N_WINNERS" "learned_pref_model=$PREF_MODEL" "axioms=$AXIOMS" "out_folder=$OUT_FOLDER"

"""


all_pref_models = [
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

    for n, m, k, pref_dist in itertools.product(n_all, m_all, k_all, pref_dist_all):
        if k >= m:
            continue
        generate_single_job_with_params(n_profiles, n, m, k, pref_dist, axioms,
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


if __name__ == "__main__":
    n_profiles = 100
    n_all = [20]
    # m_all = [5, 6, 7]
    # k_all = [1, 2, 3, 4, 5, 6]
    m_all = [4, 7]
    k_all = [2, 6]
    # pref_dist_all = all_pref_models
    pref_dist_all = ["IC", "identity"]
    axioms = "all"
    # axioms = ['condorcet_winner', 'condorcet_loser']

    job_file_location = "cc_jobs/test_jobs"
    data_out_location = "data/test_job_results"

    generate_jobs(n_profiles=n_profiles,
                  n_all=n_all,
                  m_all=m_all,
                  k_all=k_all,
                  pref_dist_all=pref_dist_all,
                  axioms=axioms,
                  folder=job_file_location,
                  data_folder=data_out_location,
                  job_time="00:10:00",
                  email="b8armstr@uwaterloo.ca",)
