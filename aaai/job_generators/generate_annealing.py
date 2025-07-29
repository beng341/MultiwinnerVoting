import os

# All possible num_winners configurations
num_winners_configs = [[1], [2], [3], [4], [5], [6], [1, 2, 3, 4, 5, 6]]
axiom_sets = ["all", "reduced"]  # root is the reduced set
num_profiles = 2000
n_annealing_steps = 1000
job_folder = "../cc_jobs/annealing_jobs"

# Make sure output directory exists
os.makedirs("slurm_out/annealing", exist_ok=True)
os.makedirs(job_folder, exist_ok=True)

job_template = '''#!/bin/bash
#SBATCH --account=def-klarson
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --mail-user=jcaiata.slurm@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_out/annealing/{out_filename}.out

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
pip install simanneal
pip install -i https://test.pypi.org/simple/ optimal-voting==0.0.2
pip install --no-index torch

echo "About to start annealing optimization"

python optimize_interpretable_rules.py "num_winners={num_winners}" "axioms_to_optimize=\'{axiom_set}\'" "num_profiles_to_sample={num_profiles}" "n_annealing_steps={n_steps}"

date +%s
echo "Finished annealing optimization"
'''

# Generate job files
for num_winners in num_winners_configs:
    for axiom_set in axiom_sets:
        num_winners_str = f"[{','.join(map(str, num_winners))}]"

        job_filename = f"annealing_job_num_winners={num_winners_str}-axiom_set={axiom_set}.sh"

        job_content = job_template.format(
            num_winners=num_winners,
            axiom_set=axiom_set,
            num_profiles=num_profiles,
            n_steps=n_annealing_steps,
            out_filename=f"cc_job_annealing_num_winners={num_winners_str}_axiom_set={axiom_set}_num_profiles={num_profiles}_n_steps={n_annealing_steps}"
        )

        with open(os.path.join(job_folder, job_filename), 'w') as f:
            f.write(job_content)

        print(f"Created job file: {job_filename}")


print("\nAll job files have been generated. You can submit them using:")
print("for f in annealing_job_*.sh; do sbatch $f; done") 