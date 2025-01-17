import os
import subprocess

# Fixed parameters
N_PROFILES = 25000
PREFS_PER_PROFILE = 50
VARIED_VOTERS = True
VOTERS_STD_DEV = 10
LEARNED_PREF_MODEL = 'real_world'
PREF_MODEL = 'real_world'
AXIOMS = 'all'
OUT_FOLDER_BASE = '/Users/joshuacaiata/Dev/MultiwinnerVoting/results/josh/results/test_generate'
RWD_FOLDER = '/Users/joshuacaiata/Dev/MultiwinnerVoting/results/josh/results/'

# Loop over values of m
for m in range(5, 8):
    # Loop over values of num_winners from 1 to m-1
    for num_winners in range(1, m):

        # Log the current configuration
        print(f"Running configuration: m={m}, num_winners={num_winners}")

        # Build the command
        command = [
            "python", "-m", "network_ops.generate_data",
            f"n_profiles={N_PROFILES}",
            f"prefs_per_profile={PREFS_PER_PROFILE}",
            f"m={m}",
            f"num_winners={num_winners}",
            f"varied_voters={VARIED_VOTERS}",
            f"voters_std_dev={VOTERS_STD_DEV}",
            f"learned_pref_model='{LEARNED_PREF_MODEL}'",
            f"pref_model='{PREF_MODEL}'",
            f"axioms='{AXIOMS}'",
            f"out_folder='{OUT_FOLDER_BASE}'",
            f"rwd_folder='{RWD_FOLDER}'"
        ]

        # Run the command
        subprocess.run(command)
