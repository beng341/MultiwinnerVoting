#!/usr/bin/env python3

import os
import itertools

# Parameters
M_VALUES = [5, 6, 7]
PREF_MODELS = [
    "stratification__args__weight=0.5",
    "URN-R",
    "IC",
    "IAC",
    "identity",
    "MALLOWS-RELPHI-R",
    "single_peaked_conitzer",
    "single_peaked_walsh",
    "euclidean__args__dimensions=3_-_space=gaussian_ball"
]
AXIOMS = ["reduced"]
N_PROFILES = 25000
N_VOTERS = 50

JOB_TEMPLATE = '''#!/bin/bash
#SBATCH --account=def-klarson
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --mail-user=jcaiata.slurm@gmail.com
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
pip install --no-deps cc_libs/llvmlite-*.whl
for f in cc_libs/*.whl; do
  [[ "$f" == *llvmlite* ]] && continue
  pip install --no-deps "$f"
done
pip install --no-index -U scikit_learn ortools

echo "About to start experiments"

python -m network_ops.generate_data "n_profiles={n_profiles}" "prefs_per_profile={n_voters}" "m={m}" "num_winners={num_winners}" "learned_pref_model='{pref_model}'" "pref_model='{pref_model}'" "axioms={axioms}" "out_folder='aaai/results/data'" "varied_voters=False" "voters_std_dev=0" "rwd_folder=''" "train=True" "test=True"
'''

def generate_jobs():
    os.makedirs("../cc_jobs/data_generation", exist_ok=True)
    
    for m, pref_model in itertools.product(M_VALUES, PREF_MODELS):
        # For each m, generate jobs for num_winners from 1 to m-1
        for num_winners in range(1, m):
            job_content = JOB_TEMPLATE.format(
                n_profiles=N_PROFILES,
                n_voters=N_VOTERS,
                m=m,
                num_winners=num_winners,
                pref_model=pref_model,
                axioms=AXIOMS
            )
            
            filename = f"cc_job_n_profiles={N_PROFILES}_n_voters={N_VOTERS}_m={m}_k={num_winners}_pref_dist='{pref_model}'_axioms='custom'.sh"
            filepath = os.path.join("../cc_jobs/data_generation", filename)
            
            with open(filepath, "w") as f:
                f.write(job_content)
            
            # Make the job file executable
            os.chmod(filepath, 0o755)

if __name__ == "__main__":
    generate_jobs() 