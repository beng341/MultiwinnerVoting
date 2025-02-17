import os

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
    "mixed"
]

pref_dist_map = {
    "all": "all",
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


def make_appendix(m_set, all_pref_dists):
    heatmaps_folder = "experiment-thesis_data/distance_heatmaps"
    summary_folder = "experiment-thesis_data/summary_tables"

    index_items = []

    current_latex = ""

    for m in m_set:
        current_latex += f"\\subsection{{{m} Alternatives -- All preferences}}\n"
        current_latex += f"\\label{{sec:{m}_alternatives-all_preferences}}\n"
        index_items.append(f"sec:{m}_alternatives-all_preferences")
        # latex_code_2 = f"\\FloatBarrier\n"
        plot_folder = f"mw_plots/m={m}"

        mid_point = len(all_pref_dists) // 2
        # current_latex = latex_code_1

        for i, pref_dist in enumerate(all_pref_dists):
            dist_name = pref_dist_map[pref_dist]
            # current_latex = latex_code_1 if i < mid_point else latex_code_2
            # current_latex = latex_code_1

            if pref_dist != "all":
                current_latex += f"\\subsection{{{m} Alternatives, {dist_name}}}\n"
                current_latex += f"\\label{{sec:{m}_alternatives-{dist_name}_preferences}}\n"
                index_items.append(f"sec:{m}_alternatives-{dist_name}_preferences")

            # Add table summarizing AVR
            summary = f"formatted_table-m=[{m}]-pref_dist=['{pref_dist}'].tex"
            if os.path.exists(f"{summary_folder}/{summary}"):
                with open(f"{summary_folder}/{summary}", "r") as f:
                    current_latex += f.read() + "\n"
            else:
                print(f"File not found: {summary_folder}/{summary}")

            # Add table with heatmap of distances
            heatmap = f"heatmap-m=[{m}]-pref_dist={pref_dist}.tex"
            if os.path.exists(f"{heatmaps_folder}/{heatmap}"):
                with open(f"{heatmaps_folder}/{heatmap}", "r") as f:
                    current_latex += f.read() + "\n"
            else:
                print(f"File not found: {heatmaps_folder}/{heatmap}")

            if pref_dist == "all":
                avr_plot = f"all_distributions_all_axioms-by_distribution-m={m}.png"
                current_latex += "\\begin{figure}[htbp!]\n"
                current_latex += f"\\includegraphics[width=\\textwidth]{{{plot_folder}/{avr_plot}}}\n"
                current_latex += f"\\caption{{Axiom violation rates for each rule under each preference distribution for {m} alternatives}}\n"
                # current_latex += f"\\Description{{A graphical representation of axiom violation rates for various distributions and axioms with {m} alternatives.}}\n"
                current_latex += "\\end{figure}\n\n"
            
            if pref_dist != "all":
                # current_latex += f"\\subsection*{{{m} Alternatives, {dist_name}}}\n"

                avr_by_axiom_plot = f"by_axiom-all_axioms-m={m}-dist={pref_dist}.png"
                avr_by_rule_plot = f"by_rule-all_axioms-m={m}-dist={pref_dist}.png"

                current_latex += "\\begin{figure}[htbp!]\n"
                current_latex += f"\\includegraphics[width=0.9\\textwidth]{{{plot_folder}/{avr_by_axiom_plot}}}\n"
                current_latex += f"\\caption{{Axiom violation rate for each axiom on {pref_dist_map[pref_dist]} preferences with {m} alternatives.}}\n"
                # current_latex += f"\\Description{{A graphical representation of the Axiom Violation Rates for various axioms with {m} alternatives.}}\n"
                current_latex += "\\end{figure}\n\n"

                current_latex += "\\begin{figure}[htbp!]\n"
                current_latex += f"\\includegraphics[width=0.9\\textwidth]{{{plot_folder}/{avr_by_rule_plot}}}\n"
                current_latex += f"\\caption{{Axiom violation rate for each rule on {pref_dist_map[pref_dist]} preferences with {m} alternatives.}}\n"
                # current_latex += f"\\Description{{A graphical representation of the Axiom Violation Rates for each rule with {m} alternatives.}}\n"
                current_latex += "\\end{figure}\n\n"

            current_latex += "\\newpage\n"
            current_latex += "\\clearpage\n\n"

    index = "\\begin{itemize}\n"
    for idx_item in index_items:
        index += f"\\item \\nameref{{{idx_item}}}\n"
    index += "\\end{itemize}\n\n"

    # current_latex = index + "\\onecolumn\n\n \\newgeometry{top=0.8in, bottom=0.8in, left=1in, right=1in}\n\n" + current_latex
    current_latex = index + "\n\n" + current_latex

    with open("combined_appendix.tex", "w") as latex_file:
        latex_file.write(current_latex)


if __name__ == "__main__":
    m_set = [5, 6, 7]

    make_appendix(m_set, ["all"] + all_pref_dists)