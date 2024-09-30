import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NOTE: I've deliberately left mixed out of this list (and in shortnames) so we can easily make a 4x4 grid with
# these 16 distributions
all_pref_dists = [
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
pref_dist_shortnames = {
    "stratification__args__weight=0.5": "Stratified",
    "URN-R": "Urn",
    "IC": "IC",
    "IAC": "IAC",
    "identity": "Identity",
    "MALLOWS-RELPHI-R": "Mallows",
    "single_peaked_conitzer": "SP Conitzer",
    "single_peaked_walsh": "SP Walsh",
    "euclidean__args__dimensions=3_-_space=gaussian_ball": "Gaussian Ball 3",
    "euclidean__args__dimensions=10_-_space=gaussian_ball": "Gaussian Ball 10",
    "euclidean__args__dimensions=3_-_space=uniform_ball": "Uniform Ball 3",
    "euclidean__args__dimensions=10_-_space=uniform_ball": "Uniform Ball 10",
    "euclidean__args__dimensions=3_-_space=gaussian_cube": "Gaussian Cube 3",
    "euclidean__args__dimensions=10_-_space=gaussian_cube": "Gaussian Cube 10",
    "euclidean__args__dimensions=3_-_space=uniform_cube": "Uniform Cube 3",
    "euclidean__args__dimensions=10_-_space=uniform_cube": "Uniform Cube 10",
    "mixed": "Mixed"
}

rule_shortnames = {
    "Neural Network": "NN",
    "Random Choice": "Random",
    "Borda ranking": "Borda",
    "Plurality ranking": "Plurality",
    "Approval Voting (AV)": "AV",
    "Proportional Approval Voting (PAV)": "PAV",
    "Approval Chamberlin-Courant (CC)": "CC",
    "Lexicographic Chamberlin-Courant (lex-CC)": "lex-CC",
    "Sequential Approval Chamberlin-Courant (seq-CC)": "seq-CC",
    "Monroe's Approval Rule (Monroe)": "Monroe",
    "Greedy Monroe": "Greedy M.",
    "Minimax Approval Voting (MAV)": "MAV"
}

rule_colours = {
    "NN": "#9edae5",
    "Random": "#1f77b4",
    "Borda": "#aec7e8",
    "Plurality": "#ffbb78",
    "AV": "#98df8a",
    "PAV": "#ff9896",
    "CC": "#c5b0d5",
    "lex-CC": "#8c564b",
    "seq-CC": "#e377c2",
    "Monroe": "#7f7f7f",
    "Greedy M.": "#bcbd22",
    "MAV": "#17becf"
}

all_axioms = {
    "dummetts_condition": "Dummett's Condition",
    "consensus_committee": "Consensus",
    "fixed_majority": "Fixed Majority",
    "majority": "Majority",
    "majority_loser": "Majority Loser",
    "condorcet_winner": "Condorcet Winner",
    "condorcet_loser": "Condorcet Loser",
    "solid_coalitions": "Solid Coalitions",
    "strong_unanimity": "Strong Unanimity",
    "local_stability": "Local Stability",
    "strong_pareto_efficiency": "Strong Pareto Efficiency"
}


def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha])


def plot_data_on_axis(ax, data):
    """
    Meant to be useful as a part of making subplots. Plot given data on the given axis.
    See: https://stackoverflow.com/questions/50488894/plotly-py-change-line-opacity-leave-markers-opaque
    for details on line opacity
    :param ax:
    :param data:
    :return:
    """
    for rule_shortname, series_data in data["series"].items():
        x_values = series_data["x_values"]
        y_values = series_data["y_values"]
        colour = rule_colours[rule_shortname]
        line_colour = hex_to_rgba(h=colour, alpha=0.5)
        marker_colour = hex_to_rgba(h=colour, alpha=1)

        nn_line_colour = (0, 0, 0, 0.6)
        nn_marker_colour = (0, 0, 0, 1)

        if rule_shortname == "NN":
            # Use black and special line type for NN (maybe temporary?)
            # ax.plot(x_values, y_values, label=rule_shortname, linewidth=2, color="black", linestyle="--")
            ax.plot(x_values,
                    y_values,
                    label=rule_shortname,
                    marker='o',
                    markersize=5,
                    linestyle="--",
                    linewidth=2,
                    markerfacecolor=nn_marker_colour,
                    color=nn_line_colour,
                    zorder=20  # Puts this series on top of others
                    )
        else:
            # line_colour = my_colour + line_alpha
            # marker_colour = my_colour + marker_alpha
            ax.plot(x_values,
                    y_values,
                    label=rule_shortname,
                    marker='.',
                    markersize=5,
                    markerfacecolor=marker_colour,
                    color=line_colour
                    )

    # # Set labels and title from the dictionary
    # ax.set_xlabel(data["xlabel"])
    # ax.set_ylabel(data["ylabel"])
    # ax.set_title(data["title"])
    #
    # # Add a legend to the axis
    # ax.legend()


def generate_plot_data_all_axioms_single_distribution(m=5, dist="IC"):
    """
    Make a basic plot of results for a given number of voters.
    x-axis shows number of winners
    y-axis show the axiom violation rate; maximum value of 1 is achieved when every example violates every axiom.
    Each series corresponds to a single voting rule.
    :param m: Number of alternatives in the data on the plot. x-axis should have integer values from 1 to m-1
    :param dist: the preference distribution which is being evaluated
    :param out_folder: subfolder where axioms are saved
    :return:
    """
    experiment_folder = "experiment_all_axioms"
    all_rule_violations = dict()

    data_for_plot = {
        "series": dict(),  # map each series name to dict of data for that series
        "xlabel": "",
        "ylabel": "Violation Rate",
        "title": f"Axiom Violations for m={m}, dist={dist}, all aximos"
    }

    all_k = []

    for k in range(1, m):

        filename = f"axiom_violation_results-n_profiles=25000-num_voters=50-m={m}-k={k}-pref_dist={dist}-axioms=all.csv"

        full_name = os.path.join(experiment_folder, filename)

        if not os.path.exists(full_name):
            print(f"File doesn't exist. Skipping these datapoints: {full_name}")
            continue
        df = pd.read_csv(full_name)

        label_column = df.iloc[:, 0].tolist()  # rule names
        results_column = df.iloc[:, 1].tolist()  # mean violation rate over all axioms
        std_results_column = df.iloc[:, 2].tolist()  # std of mean violation rate over all axioms
        result_pairs = [(rc, st) for rc, st in zip(results_column, std_results_column)]

        rule_violations = dict(zip(label_column, result_pairs))

        for rule, violations in rule_violations.items():
            if rule not in all_rule_violations:
                all_rule_violations[rule] = []
            all_rule_violations[rule].append(violations)

        all_k.append(k)

    for rule, violations in all_rule_violations.items():
        rule_label = rule_shortnames[rule]
        x_values = all_k
        y_values = [v[0] for v in violations]
        std_values = [v[1] for v in violations]

        data_for_plot["series"][rule_label] = dict()
        data_for_plot["series"][rule_label]["x_values"] = x_values
        data_for_plot["series"][rule_label]["y_values"] = y_values
        if rule == "Neural Network":
            data_for_plot["series"][rule_label]["std_values"] = std_values

    return data_for_plot


def generate_plot_data_single_axioms_single_distribution(m=5, dist="IC", axioms=[]):
    column_names = [f"{ax}-mean" for ax in axioms]
    std_column_names = [f"{ax}-std" for ax in axioms]

    experiment_folder = "experiment_all_axioms"
    all_rule_violations = dict()

    data_for_plot = {
        "series": dict(),  # map each series name to dict of data for that series
        "xlabel": "",
        "ylabel": "Violation Rate",
        "title": f"Axiom Violations for m={m}, dist={dist}, axioms={axioms}"
    }

    all_k = []
    # Load data from each column to assemble all the rows
    for k in range(1, m):

        filename = f"axiom_violation_results-n_profiles=25000-num_voters=50-m={m}-k={k}-pref_dist={dist}-axioms=all.csv"
        full_name = os.path.join(experiment_folder, filename)
        if not os.path.exists(full_name):
            print(f"File doesn't exist. Skipping these datapoints: {full_name}")
            continue
        df = pd.read_csv(full_name)
        print(f"Loaded data from: {full_name}")

        df["mean_value"] = df[column_names].mean(axis=1)
        df["mean_std"] = df[std_column_names].mean(axis=1)

        label_column = df.iloc[:, 0].tolist()  # rule names
        results_column = df["mean_value"].tolist()  # mean violation rate for specified axioms
        std_results_column = df["mean_std"].tolist()  # std of mean violation rate for specified axioms
        result_pairs = [(rc, st) for rc, st in zip(results_column, std_results_column)]

        rule_violations = dict(zip(label_column, result_pairs))

        for rule, violations in rule_violations.items():
            if rule not in all_rule_violations:
                all_rule_violations[rule] = []
            all_rule_violations[rule].append(violations)

        all_k.append(k)

    for rule, violations in all_rule_violations.items():
        rule_label = rule_shortnames[rule]
        x_values = all_k
        y_values = [v[0] for v in violations]
        std_values = [v[1] for v in violations]

        data_for_plot["series"][rule_label] = dict()
        data_for_plot["series"][rule_label]["x_values"] = x_values
        data_for_plot["series"][rule_label]["y_values"] = y_values
        if rule == "Neural Network":
            data_for_plot["series"][rule_label]["std_values"] = std_values

    return data_for_plot


def plot_each_distribution_all_axioms(m, out_folder):
    """
    Make plot with 16 subplots, one for each distribution.
    :param m:
    :param out_folder:
    :return:
    """
    filename = f"all_distributions_all_axioms-by_distribution-m={m}.png"
    fig, axs = plt.subplots(figsize=(12, 8), nrows=4, ncols=4, sharey="row", sharex="col", constrained_layout=True)

    # Add all data
    for idx, dist in enumerate(all_pref_dists):
        ax = fig.axes[idx]
        single_dist_data = generate_plot_data_all_axioms_single_distribution(m=m, dist=dist)
        plot_data_on_axis(ax, single_dist_data)
        ax.set_title(pref_dist_shortnames[dist])

    for (_, _), ax in np.ndenumerate(axs):
        x_ticks = [i for i in range(1, m)]
        ax.set_xticks(x_ticks)

        ax.set_ylim((-0.05, 0.75))
        y_ticks = [0, 0.2, 0.4, 0.6]
        ax.set_yticks(y_ticks)

        ax.grid(alpha=0.5)

    # Make plot look better
    plt.suptitle(f"Axiom Violation Rates for {m} Alternatives over All Axioms", fontsize=16)

    fig.supxlabel('Number of Alternatives', fontsize=12, x=0.5, y=0.09)
    fig.supylabel('Axiom Violation Rate', fontsize=12, x=0.015, y=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=6, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.98, 0, 0])
    plt.subplots_adjust(bottom=0.15)

    if not os.path.exists(path=out_folder):
        os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, filename))


def plot_mixed_distribution_all_axioms(m, out_folder):
    """
    Make plot showing mixed distribution axiom violation rate
    :param m:
    :param out_folder:
    :return:
    """
    filename = f"mixed_distributions_all_axioms-by_distribution-m={m}.png"
    # fig, axs = plt.subplots(figsize=(12, 8), sharey="row", sharex="col", constrained_layout=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    # Add all data
    # for idx, dist in enumerate(all_pref_dists):
    # ax = fig.axes[idx]
    single_dist_data = generate_plot_data_all_axioms_single_distribution(m=m, dist="mixed")
    plot_data_on_axis(ax, single_dist_data)

    # for (_, _), ax in np.ndenumerate(axs):
    x_ticks = [i for i in range(1, m)]
    ax.set_xticks(x_ticks)

    ax.set_ylim((-0.05, 0.75))
    y_ticks = [0, 0.2, 0.4, 0.6]
    ax.set_yticks(y_ticks)

    ax.grid(alpha=0.5)

    # Make plot look better
    plt.suptitle(f"Mixed Preferences Axiom Violation Rates for {m} Alternatives", fontsize=14)

    ax.set_xlabel('Number of Alternatives', fontsize=12, x=0.5, y=0.09)
    ax.set_ylabel('Axiom Violation Rate', fontsize=12, x=0.015, y=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', ncol=4)

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.98, 0, 0])
    # plt.subplots_adjust(bottom=0.15)

    if not os.path.exists(path=out_folder):
        os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, filename))


def plot_each_axiom_specific_distribution(m, dist, out_folder):
    """

    :param m:
    :param dist:
    :param out_folder:
    :return:
    """
    filename = f"all_axioms-m={m}-dist={dist}-by_axiom.png"

    fig, axs = plt.subplots(figsize=(12, 8), nrows=4, ncols=3, sharey="row", sharex="col", constrained_layout=True)

    for (row, col), ax in np.ndenumerate(axs):
        ax.axis("off")

    # Add all data
    for idx, axiom in enumerate(all_axioms.keys()):
        ax = fig.axes[idx]
        ax.axis("on")
        single_ax_data = generate_plot_data_single_axioms_single_distribution(m=m,
                                                                              dist=dist,
                                                                              axioms=[axiom])
        plot_data_on_axis(ax, single_ax_data)
        ax.set_title(all_axioms[axiom])

        # Set up plot ticks, limits
        x_ticks = [i for i in range(1, m)]
        ax.set_xticks(x_ticks)

        # ax.set_ylim((-0.05, 0.75))
        ax.set_ylim((-0.05, 1.05))
        y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(y_ticks)

        ax.grid(alpha=0.5)

    # Make plot look better
    plt.suptitle(f"Axiom Violation Rates for {m} Alternatives on {pref_dist_shortnames[dist]} Preferences", fontsize=16)

    fig.supxlabel('Number of Alternatives', fontsize=12, x=0.5, y=0.09)
    fig.supylabel('Axiom Violation Rate', fontsize=12, x=0.015, y=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=6, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.98, 0, 0])
    plt.subplots_adjust(bottom=0.15)

    if not os.path.exists(path=out_folder):
        os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, filename))


if __name__ == "__main__":
    plot_mixed_distribution_all_axioms(m=6,
                                       out_folder="experiment_all_axioms/plots")
    plot_each_distribution_all_axioms(m=6,
                                      out_folder="experiment_all_axioms/plots")

    for dist in all_pref_dists + ["mixed"]:
        plot_each_axiom_specific_distribution(m=6,
                                              dist=dist,
                                              out_folder="experiment_all_axioms/plots")
