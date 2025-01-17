import os
import pandas as pd
import ast
from sklearn.mixture import GaussianMixture
import numpy as np
import random
from itertools import permutations

def fit_gmm(rwd_folder, m):
    test_file = os.path.join(
        rwd_folder,
        f"n_profiles=25000-num_voters=50-varied_voters=True-voters_std_dev=10-m={m}-committee_size=1-pref_dist=mixed-axioms=all-TEST.csv",
    )

    test_data = pd.read_csv(test_file)
    test_data['Parsed_Profile'] = test_data['Profile'].apply(
        lambda x: [tuple(map(int, sublist)) for sublist in ast.literal_eval(x)]
    )

    flattened_preferences = []
    for profile in test_data['Parsed_Profile']:
        flattened_preferences.extend(profile)
    flattened_preferences = np.array(flattened_preferences)

    gmm = GaussianMixture(n_components=5, random_state=42)
    gmm.fit(flattened_preferences)

    return gmm, test_data

def generate_single_profile(gmm, all_permutations, test_profiles_set, num_voters):
    count = 0
    while count < 100:
        profile = []
        for _ in range(num_voters):
            all_permutations_array = np.array(all_permutations)
            likelihoods = gmm.score_samples(all_permutations_array)

            probabilities = np.exp(likelihoods - np.max(likelihoods))
            probabilities /= probabilities.sum()

            selected_ballot = random.choices(all_permutations, weights=probabilities)[0]
            profile.append(selected_ballot)

        if str(profile) not in set(test_profiles_set['Profile']):
            return profile

        count += 1

    raise Exception("Could not generate a unique profile within 100 attempts")

def generate_permuations(m):
    return list(permutations(range(m)))

