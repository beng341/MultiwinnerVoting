import os
import pprint
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_utils import load_evaluation_results_df

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
    "Plurality ranking": "SNTV",  # "Plurality",
    "STV": "STV",
    "Approval Voting (AV)": "Bloc",  # "AV",
    "Proportional Approval Voting (PAV)": "PAV",
    "Approval Chamberlin-Courant (CC)": "CC",
    "Lexicographic Chamberlin-Courant (lex-CC)": "lex-CC",
    "Sequential Approval Chamberlin-Courant (seq-CC)": "seq-CC",
    "Monroe's Approval Rule (Monroe)": "Monroe",
    "Greedy Monroe": "Greedy M.",
    "Minimax Approval Voting (MAV)": "MAV",
    "Method of Equal Shares (aka Rule X) with Phragmén phase": "MES",
    "E Pluribus Hugo (EPH)": "EPH",
    "Random Serial Dictator": "RSD",
    "Min Violations Committee": "Min",
    "Max Violations Committee": "Max",
}
rule_markers = {
    "Neural Network": "o",
    "Random Choice": ".",
    "Borda ranking": "d",
    "Plurality ranking": "p",  # "Plurality",
    "STV": "o",
    "Approval Voting (AV)": "x",  # "AV",
    "Proportional Approval Voting (PAV)": "+",
    "Approval Chamberlin-Courant (CC)": "^",
    "Lexicographic Chamberlin-Courant (lex-CC)": ">",
    "Sequential Approval Chamberlin-Courant (seq-CC)": "<",
    "Monroe's Approval Rule (Monroe)": "1",
    "Greedy Monroe": "3",
    "Minimax Approval Voting (MAV)": "s",
    "Method of Equal Shares (aka Rule X) with Phragmén phase": "v",
    "E Pluribus Hugo (EPH)": "X",
    "Random Serial Dictator": "4",
    "Min Violations Committee": ".",
    "Max Violations Committee": ".",
}

series_colours_aamas = {
    "NN": "#9edae5",
    "Random": "#1f77b4",
    "Borda": "#aec7e8",
    "SNTV": "#ffbb78",
    "STV": "#ff78bb",
    "Bloc": "#98df8a",
    "PAV": "#ff9896",
    "CC": "#c5b0d5",
    "lex-CC": "#8c564b",
    "seq-CC": "#e377c2",
    "Monroe": "#7f7f7f",
    "Greedy M.": "#bcbd22",
    "MAV": "#17becf",
    "MES": "#7f7f7f",
    "EPH": "#7f7f7f",
    "RSD": "#7f7f7f",
    # We also use axioms as series labels sometimes; use a different colour scheme for them:
    # "Dummett's Condition": '#000000',
    # 'Fixed Majority': '#0000dd',
    # 'Majority': '#0099dd',
    # 'Majority Loser': '#00ff00',
    # 'Condorcet Winner': '#00ba00',
    # 'Condorcet Loser': '#eded00',
    # 'Solid Coalitions': '#ff9900',
    # 'Strong Unanimity': '#ff00ae',
    # 'Local Stability': '#dd0000',
    # 'Strong Pareto Efficiency': '#cccccc',
    # "Justified Representation": "#870099",
    # "Extended JR": "#870099",
    # "Core": "#870099"

}

series_colours = {
    'Bloc': '#26ef00',
    'Borda': '#0062ff',
    'CC': '#b1ff33',
    'EPH': '#f1a9f3',
    'Greedy M.': '#ff0e00',
    'MAV': '#f106ff',
    'MES': '#bc50f7',
    'Monroe': '#ff7005',
    'NN': '#000080',
    'PAV': '#73e800',
    'RSD': '#229966',
    'Random': '#004348',
    'SNTV': '#00edff',
    'STV': '#00fa92',
    'lex-CC': '#fff700',
    'seq-CC': '#ffcd05',
    "Min": "#0000ff",
    "Max": "#ff0000",
    # And include colours for axioms:
    'Condorcet Loser': '#00a45a',
    'Condorcet Winner': '#00a4bb',
    'Core': '#cccccc',
    "Dummett's Condition": '#000000',
    'Extended JR': '#d70000',
    'Fixed Majority': '#820093',
    'Justified Representation': '#ff3300',
    'Local Stability': '#cbf900',
    'Majority': '#0000bb',
    'Majority Loser': '#0077dd',
    'Solid Coalitions': '#00ba00',
    'Strong Pareto Efficiency': '#ffcc00',
    'Strong Unanimity': '#00f300'
}
all_axioms = {
    # "consensus_committee": "Consensus",
    "dummetts_condition": "Dummett's Condition",
    "fixed_majority": "Fixed Majority",
    "majority": "Majority",
    "majority_loser": "Majority Loser",
    "condorcet_winner": "Condorcet Winner",
    "condorcet_loser": "Condorcet Loser",
    "solid_coalitions": "Solid Coalitions",
    "strong_unanimity": "Strong Unanimity",
    "local_stability": "Local Stability",
    "strong_pareto_efficiency": "Strong Pareto Efficiency",
    "jr": "Justified Representation",
    "ejr": "Extended JR",
    "core": "Core"
}

axiom_markers = {
    # "consensus_committee": ".",
    "dummetts_condition": "d",
    "fixed_majority": "*",
    "majority": "h",
    "majority_loser": "x",
    "condorcet_winner": "o",
    "condorcet_loser": "v",
    "solid_coalitions": "^",
    "strong_unanimity": "<",
    "local_stability": ">",
    "strong_pareto_efficiency": "p",
    "jr": "1",
    "ejr": "3",
    "core": "2"
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
    for series_label, series_data in data["series"].items():
        x_values = series_data["x_values"]
        y_values = series_data["y_values"]
        colour = series_colours[series_label]
        line_colour = hex_to_rgba(h=colour, alpha=0.5)
        marker_colour = hex_to_rgba(h=colour, alpha=1)
        if "marker" in series_data:
            marker = series_data["marker"]
        else:
            marker = "."

        nn_line_colour = (0, 0, 0, 0.6)
        nn_marker_colour = (0, 0, 0, 1)

        if series_label == "NN":
            # Use black and special line type for NN (maybe temporary?)
            # ax.plot(x_values, y_values, label=series_label, linewidth=2, color="black", linestyle="--")
            ax.plot(x_values,
                    y_values,
                    # label=series_label,
                    label=series_label,
                    marker='o',
                    markersize=5,
                    linestyle="--",
                    linewidth=2,
                    markerfacecolor=nn_marker_colour,
                    color=nn_line_colour,
                    zorder=20  # Puts this series on top of others
                    )
            if "noise" in series_data:
                y_values = np.array(y_values)
                noise = np.array(series_data["noise"])
                ax.fill_between(x_values,
                                y_values - noise,
                                y_values + noise,
                                # color="orange",
                                color=nn_line_colour,
                                # alpha=0.2
                                )
        elif series_label == "Min" or series_label == "Max":
            # Use black and special line type for NN (maybe temporary?)
            # ax.plot(x_values, y_values, label=series_label, linewidth=2, color="black", linestyle="--")
            ax.plot(x_values,
                    y_values,
                    # label=series_label,
                    label=series_label,
                    marker='.',
                    markersize=1,
                    linestyle=":",
                    linewidth=1,
                    markerfacecolor=hex_to_rgba(h=colour, alpha=1),
                    color=hex_to_rgba(h=colour, alpha=1),
                    zorder=19  # Puts this series on top of others, below NN
                    )
        else:
            # line_colour = my_colour + line_alpha
            # marker_colour = my_colour + marker_alpha
            ax.plot(x_values,
                    y_values,
                    label=series_label,
                    marker=marker,
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


def generate_plot_data_all_axioms_single_distribution(m=5, dist="IC", metric="std"):
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
    experiment_folder = "evaluation_results_thesis"
    all_rule_violations = dict()

    data_for_plot = {
        "series": dict(),  # map each series name to dict of data for that series
        "xlabel": "",
        "ylabel": "Violation Rate",
        "title": f"Axiom Violations for m={m}, dist={dist}, all axioms",
        "series_label": ""
    }

    all_k = []

    for k in range(1, m):
        # if k == 1:
        #     print("Skipping k=1 temporarily.")
        #     continue

        filename = f"axiom_violation_results-n_profiles=25000-num_voters=50-m={m}-k={k}-pref_dist={dist}-axioms=all.csv"
        full_name = os.path.join(experiment_folder, filename)
        df = load_evaluation_results_df(full_name, metric=metric)
        if df is None:
            print(f"File doesn't exist. Skipping these datapoints: {full_name}")
            continue
        print(f"Loaded data from: {full_name}")

        label_column = df.iloc[:, 0].tolist()  # rule names
        results_column = df.iloc[:, 1].tolist()  # mean violation rate over all axioms
        rule_violations = dict(zip(label_column, results_column))
        # std_results_column = df.iloc[:, 2].tolist()  # std of mean violation rate over all axioms
        # result_pairs = [(rc, st) for rc, st in zip(results_column, std_results_column)]

        # rule_violations = dict(zip(label_column, result_pairs))

        for rule, violations in rule_violations.items():
            if rule not in all_rule_violations:
                all_rule_violations[rule] = []
            all_rule_violations[rule].append(violations)

        all_k.append(k)

    for rule, violations in all_rule_violations.items():
        # if rule == "STV":
        #     continue
        x_values = all_k
        y_values = [v for v in violations]
        # std_values = [v[1] for v in violations]
        if rule == "Neural Network Noise":
            # should exclude this row from regular processing and add as special value for Neural Network
            data_for_plot["series"]["NN"]["noise"] = y_values
            continue

        if rule not in rule_shortnames:
            # When k = 1, we have many more rules than we want to plot here
            continue

        rule_label = rule_shortnames[rule]
        data_for_plot["series"][rule_label] = dict()
        data_for_plot["series"][rule_label]["x_values"] = x_values
        data_for_plot["series"][rule_label]["y_values"] = y_values
        data_for_plot["series"][rule_label]["series_label"] = rule_label
        data_for_plot["series"][rule_label]["marker"] = rule_markers[rule]
        # if rule == "Neural Network":
        #     data_for_plot["series"][rule_label]["std_values"] = std_values

    return data_for_plot


def generate_plot_data_specified_axioms_single_distribution(m=5, dist="IC", axioms=[], metric="std"):
    column_names = [f"{ax}-mean" for ax in axioms]
    # std_column_names = [f"{ax}-std" for ax in axioms]

    experiment_folder = "evaluation_results_thesis"
    all_rule_violations = dict()

    data_for_plot = {
        "series": dict(),  # map each series name to dict of data for that series
        "xlabel": "",
        "ylabel": "Violation Rate",
        "title": f"Axiom Violations for m={m}, dist={dist}, axioms={axioms}",
    }

    all_k = []
    # Load data from each column to assemble all the rows
    for k in range(1, m):
        # if k == 1:
        #     print("Skipping k=1 temporarily.")
        #     continue

        filename = f"axiom_violation_results-n_profiles=25000-num_voters=50-m={m}-k={k}-pref_dist={dist}-axioms=all.csv"
        full_name = os.path.join(experiment_folder, filename)
        df = load_evaluation_results_df(full_name, metric=metric)
        if df is None:
            print(f"File doesn't exist. Skipping these datapoints: {full_name}")
            continue
        print(f"Loaded data from: {full_name}")

        df["mean_value"] = df[column_names].mean(axis=1)
        # df["mean_std"] = df[std_column_names].mean(axis=1)

        label_column = df.iloc[:, 0].tolist()  # rule names
        results_column = df["mean_value"].tolist()  # mean violation rate for specified axioms
        rule_violations = dict(zip(label_column, results_column))

        # # std_results_column = df["mean_std"].tolist()  # std of mean violation rate for specified axioms
        # result_pairs = [(rc, st) for rc, st in zip(results_column, std_results_column)]
        #
        # rule_violations = dict(zip(label_column, result_pairs))

        for rule, violations in rule_violations.items():
            if rule not in all_rule_violations:
                all_rule_violations[rule] = []
            all_rule_violations[rule].append(violations)

        all_k.append(k)

    for rule, violations in all_rule_violations.items():
        # if rule == "STV":
        #     continue
        x_values = all_k
        y_values = [v for v in violations]
        # std_values = [v[1] for v in violations]
        if rule == "Neural Network Noise":
            # should exclude this row from regular processing and add as special value for Neural Network
            data_for_plot["series"]["NN"]["noise"] = y_values
            continue

        if rule not in rule_shortnames:
            # When k = 1, we have many more rules than we want to plot here
            continue

        rule_label = rule_shortnames[rule]
        data_for_plot["series"][rule_label] = dict()
        data_for_plot["series"][rule_label]["x_values"] = x_values
        data_for_plot["series"][rule_label]["y_values"] = y_values
        data_for_plot["series"][rule_label]["series_label"] = rule_label
        data_for_plot["series"][rule_label]["marker"] = rule_markers[rule]

    return data_for_plot


def generate_plot_data_each_rule_by_axiom(m=5, rule="Neural Network", dist="IC", metric="std"):
    """
    Make plot data for a specific rule where the axiom violation rates are the series.
    :param m:
    :param rule:
    :param dist:
    :param metric:
    :return:
    """
    # experiment_folder = "experiment_all_axioms/evaluation_results"
    experiment_folder = "evaluation_results_thesis"

    data_for_plot = {
        "series": dict(),  # map each series name to dict of data for that series
        "xlabel": "",
        "ylabel": "Violation Rate",
        "title": f"Axiom Violations for m={m}, dist={dist}, all axioms",
    }

    all_k = []
    ax_violations = dict()

    nn_noise_vals = dict()

    for k in range(1, m):
        filename = f"axiom_violation_results-n_profiles=25000-num_voters=50-m={m}-k={k}-pref_dist={dist}-axioms=all.csv"
        full_name = os.path.join(experiment_folder, filename)
        df = load_evaluation_results_df(full_name, metric=metric)
        if df is None:
            print(f"File doesn't exist. Skipping these datapoints: {full_name}")
            continue
        print(f"Loaded data from: {full_name}")

        all_k.append(k)  # track x values

        for ax in all_axioms.keys():

            col_name = f"{ax}-mean"
            ax_violation_rate = df.loc[df['Method'] == rule, col_name].values[0]
            if ax not in ax_violations:
                ax_violations[ax] = []
            ax_violations[ax].append(ax_violation_rate)

            if rule == "Neural Network":
                ax_violation_rate = df.loc[df['Method'] == "Neural Network Noise", col_name].values[0]
                if ax not in nn_noise_vals:
                    nn_noise_vals[ax] = []
                nn_noise_vals[ax].append(ax_violation_rate)

            # label_column = df.iloc[:, 0].tolist()  # rule names
            # results_column = df.iloc[:, 1].tolist()  # mean violation rate over all axioms
            # rule_violations = dict(zip(label_column, results_column))
            # # std_results_column = df.iloc[:, 2].tolist()  # std of mean violation rate over all axioms
            # # result_pairs = [(rc, st) for rc, st in zip(results_column, std_results_column)]
            #
            # # rule_violations = dict(zip(label_column, result_pairs))
            #
            # for rule, violations in rule_violations.items():
            #     if rule not in all_rule_violations:
            #         all_rule_violations[rule] = []
            #     all_rule_violations[rule].append(violations)
            #
    for ax, violations in ax_violations.items():
        x_values = all_k
        y_values = [v for v in violations]
        if rule == "Neural Network Noise":
            # should exclude this row from regular processing and add as special value for Neural Network
            data_for_plot["series"]["Neural Network"]["noise"] = nn_noise_vals[ax]
            continue

        # rule_label = rule_shortnames[rule]
        ax_label = all_axioms[ax]
        data_for_plot["series"][ax_label] = dict()
        data_for_plot["series"][ax_label]["x_values"] = x_values
        data_for_plot["series"][ax_label]["y_values"] = y_values
        data_for_plot["series"][ax_label]["series_label"] = all_axioms[ax]
        data_for_plot["series"][ax_label]["marker"] = axiom_markers[ax]

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

        ax.set_ylim((-0.05, 1.05))
        y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_yticks(y_ticks)

        ax.grid(alpha=0.5)

    # Make plot look better
    plt.suptitle(f"Axiom Violation Rates for {m} Alternatives over All Axioms", fontsize=16)

    fig.supxlabel('Number of Winners', fontsize=12, x=0.5, y=0.09)
    fig.supylabel('Axiom Violation Rate', fontsize=12, x=0.015, y=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.98, 0, 0])
    plt.subplots_adjust(bottom=0.15)

    # plt.show()

    if not os.path.exists(path=out_folder):
        os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, filename), dpi=100)


def plot_mixed_distribution_all_axioms(m, out_folder):
    """
    Make plot showing mixed distribution axiom violation rate
    :param m:
    :param out_folder:
    :return:
    """
    filename = f"mixed_distributions_all_axioms-by_distribution-m={m}.png"
    # fig, axs = plt.subplots(figsize=(12, 8), sharey="row", sharex="col", constrained_layout=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

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

    ax.set_xlabel('Number of Winners', fontsize=12, x=0.5, y=0.09)
    ax.set_ylabel('Axiom Violation Rate', fontsize=12, x=0.015, y=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', ncol=4)

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.98, 0, 0])
    # plt.subplots_adjust(bottom=0.15)

    if not os.path.exists(path=out_folder):
        os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, filename), dpi=100)


def plot_each_axiom_specific_distribution(m, dist, out_folder):
    """

    :param m:
    :param dist:
    :param out_folder:
    :return:
    """
    filename = f"by_axiom-all_axioms-m={m}-dist={dist}.png"

    fig, axs = plt.subplots(figsize=(12, 8), nrows=5, ncols=3, sharey="row", sharex="col", constrained_layout=True)

    for (row, col), ax in np.ndenumerate(axs):
        ax.axis("off")

    # Add all data
    for idx, axiom in enumerate(all_axioms.keys()):
        # if idx >= 12:
        #     continue
        ax = fig.axes[idx]
        ax.axis("on")
        single_ax_data = generate_plot_data_specified_axioms_single_distribution(m=m,
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

    fig.supxlabel('Number of Winners', fontsize=12, x=0.212, y=0.02)
    # fig.supxlabel('Number of Winners', fontsize=12, x=0.5, y=0.09)
    fig.supylabel('Axiom Violation Rate', fontsize=12, x=0.015, y=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=6, bbox_to_anchor=(0.68, 0.11))

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.98, 0, 0])
    plt.subplots_adjust(bottom=0.08)

    if not os.path.exists(path=out_folder):
        os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, filename), dpi=100)


def plot_each_rule_single_dist_axiom_series(m, dist, out_folder):
    """

    :param m:
    :param dist:
    :param out_folder:
    :return:
    """
    filename = f"by_rule-all_axioms-m={m}-dist={dist}.png"

    fig, axs = plt.subplots(figsize=(12, 9), nrows=6, ncols=3, sharey="row", sharex="col", constrained_layout=True)

    for (row, col), ax in np.ndenumerate(axs):
        ax.axis("off")

    # Add all data
    for idx, rule in enumerate(rule_shortnames.keys()):
        if rule == "Min Violations Committee" or rule == "Max Violations Committee":
            continue
        ax = fig.axes[idx]
        ax.axis("on")
        single_ax_data = generate_plot_data_each_rule_by_axiom(m=m,
                                                               rule=rule,
                                                               dist=dist,
                                                               metric="std")
        # single_ax_data = generate_plot_data_specified_axioms_single_distribution(m=m,
        #                                                                          dist=dist,
        #                                                                          axioms=[axiom])
        plot_data_on_axis(ax, single_ax_data)
        ax.set_title(rule_shortnames[rule])

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

    # fig.supxlabel('Number of Winners', fontsize=12, x=0.5, y=0.09)
    fig.supxlabel('Number of Winners', fontsize=12, x=0.212, y=0.02)
    fig.supylabel('Axiom Violation Rate', fontsize=12, x=0.015, y=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, bbox_to_anchor=(0.68, 0.054), fontsize=10)

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.98, 0, 0])
    plt.subplots_adjust(bottom=0.08)

    if not os.path.exists(path=out_folder):
        os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, filename), dpi=100)


def plot_mixed_distribution_all_axioms_subplots_for_m(out_folder):
    """
    Make figure with 3 subplots. Each one shows average violation rate across all k for a specific m.
    :return:
    """
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    all_m = [5, 6, 7]
    for idx, ax in enumerate(axs):
        single_dist_data = generate_plot_data_all_axioms_single_distribution(m=all_m[idx], dist="mixed")
        plot_data_on_axis(ax, single_dist_data)

        x_ticks = [i for i in range(1, all_m[idx])]
        ax.set_xticks(x_ticks)

        ax.set_ylim((-0.05, 0.6))
        ax.set_title(f"{all_m[idx]} Alternatives")

        ax.grid(alpha=0.5)

    fig.supxlabel("Number of Winners", fontsize=14, x=0.5, y=0.17)
    fig.supylabel("Violation Rate", fontsize=14, x=0.005, y=0.55)
    fig.suptitle("Mixed Preference Axiom Violation Rates on All Axioms", fontsize=15)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=9)

    plt.tight_layout(rect=(-0.015, 0, 1, 1))
    fig.subplots_adjust(bottom=0.3)

    # plt.show()

    plt.savefig(os.path.join(out_folder, "axiom_violations_all_m.png"), dpi=100)


def make_all_plots(m=5):
    out_folder = f"experiment-thesis_data/plots/m={m}"

    plot_mixed_distribution_all_axioms(m=m,
                                       out_folder=out_folder)
    plot_each_distribution_all_axioms(m=m,
                                      out_folder=out_folder)

    for dist in all_pref_dists + ["mixed"]:
        plot_each_axiom_specific_distribution(m=m,
                                              dist=dist,
                                              out_folder=out_folder)

        plot_each_rule_single_dist_axiom_series(m=m,
                                                dist=dist,
                                                out_folder=out_folder)


def get_colormap_colors(cmap_name, num_colors):
    # Get the colormap
    cmap = plt.get_cmap(cmap_name, num_colors)

    colours = []
    # Print colors in order (as RGB values or hex)
    for i in range(num_colors):
        color = cmap(i / (num_colors - 1))  # Normalize i to get colors in order
        # Convert to hex format
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
        )
        # print(hex_color)
        colours.append(hex_color)
    return colours


def print_colormap_with_dict_values(dic, cmap_name):
    colours = get_colormap_colors(cmap_name, len(dic))

    colour_map = dict(zip(dic.values(), colours))
    pprint.pprint(colour_map)


#
#
# # # Example usage
# get_colormap_colors('hsv', 12)

if __name__ == "__main__":
    # # print_colormap_with_dict_values(dic=rule_shortnames, cmap_name="gist_ncar")
    #
    # # print_colormap_with_dict_values(dic=all_axioms, cmap_name="nipy_spectral")
    # # exit()

    # plt.rcParams['font.size'] = 12  # Default text size
    plt.rcParams['axes.titlesize'] = 14  # Title font size
    plt.rcParams['axes.labelsize'] = 12  # Axis label font size
    plt.rcParams['xtick.labelsize'] = 11  # X-axis tick label font size
    plt.rcParams['ytick.labelsize'] = 11  # Y-axis tick label font size
    plt.rcParams['legend.fontsize'] = 12  # Legend font size

    m = 7
    make_all_plots(m)

    m = 6
    make_all_plots(m)

    m = 5
    make_all_plots(m)

    plot_mixed_distribution_all_axioms_subplots_for_m(out_folder=f"experiment-thesis_data/plots")
