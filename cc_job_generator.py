import itertools
import os.path

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


def generate_jobs(n_profiles, n_all, m_all, k_all, pref_dist_all, axioms="all", folder="cc_jobs/jobs"):
    """

    :param n_profiles:
    :param n_all:
    :param m_all:
    :param k_all:
    :param pref_dist_all:
    :param axioms:
    :param folder:
    :return:
    """
    axioms = "all"

    for n, m, k, pref_dist in itertools.product(n_all, m_all, k_all, pref_dist_all):
        if k >= m:
            continue
        generate_single_job_with_params(n_profiles, n, m, k, pref_dist, axioms, folder)


def generate_single_job_with_params(n_profiles, n, m, k, pref_dist, axioms, folder):
    """

    :param n_profiles:
    :param n:
    :param m:
    :param k:
    :param pref_dist:
    :param axioms:
    :param folder:
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == "__main__":
    n_profiles = 100
    n_all = [50]
    m_all = [5, 6, 7]
    k_all = [1, 2, 3, 4, 5, 6]
    pref_dist_all = all_pref_models
    axioms = "all"

    out_folder = "cc_jobs/test_jobs"

    generate_jobs(n_profiles=n_profiles,
                  n_all=n_all,
                  m_all=m_all,
                  k_all=k_all,
                  pref_dist_all=pref_dist_all,
                  axioms=axioms,
                  folder=out_folder)
