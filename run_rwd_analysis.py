import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from experiment_both_axiom_sets.arXiv.make_summary_table import make_summary_table, make_aggregated_table_single_m
from experiment_both_axiom_sets.arXiv.make_distance_tables import make_distance_table
from experiment_both_axiom_sets.arXiv.plot_experiment_data import make_all_plots, plot_mixed_distribution_all_axioms_subplots_for_m, plot_each_axiom_specific_distribution, plot_each_rule_single_dist_axiom_series

# Parameters from your RWD files
n_profiles = [25000]
num_voters = [50]
m_set = [5, 6, 7]  # From your filenames
k_set = range(1, 7)  # From your filenames
pref_dist = ["mixed"]  # Will be changed to "rwd" in the code
axioms = ["all"]

# Make summary tables
print("Generating summary tables...")
make_summary_table(n_profiles=n_profiles, 
                  num_voters=num_voters,
                  m_set=m_set,
                  k_set=k_set,
                  pref_dist=pref_dist,
                  axioms=axioms)

for m in m_set:
    make_aggregated_table_single_m(m=m)

# Make distance tables
print("Generating distance tables...")
for m in m_set:
    make_distance_table(n_profiles=n_profiles,
                    num_voters=num_voters,
                    m_set=[m],
                    k_set=k_set,
                    pref_dist=pref_dist)

# Make plots
print("Generating plots...")
for m in m_set:
    make_all_plots(m=m)
#     #generate_plot_data_all_axioms_single_distribution(m=m, dist="rwd")
    plot_mixed_distribution_all_axioms_subplots_for_m(out_folder=f"experiment_both_axiom_sets/arXiv/plots")
    plot_each_axiom_specific_distribution(m=m, dist="mixed", out_folder=f"experiment_both_axiom_sets/arXiv/plots/m={m}")
    plot_each_rule_single_dist_axiom_series(m=m, dist="midex", out_folder=f"experiment_both_axiom_sets/arXiv/plots/m={m}")

print("Done!") 