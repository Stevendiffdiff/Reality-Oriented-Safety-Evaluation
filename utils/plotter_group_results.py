import os
import pprint

import autoroot
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from utils.common import LINESTYLE_DICT, PALETTE, find_all_files


def two_decimal(x, pos):
    "The two args are the value and tick position"
    return "%.2f" % (x)


formatter = FuncFormatter(two_decimal)
pp = pprint.PrettyPrinter()

# configure matplotlib
mpl.use("AGG")
plt.style.use("seaborn-v0_8-whitegrid")
mpl.rc(["pdf", "ps"], fonttype=42)
mpl.rc("font", size=7)
mpl.rc("axes", titlesize=8, labelpad=2, titlepad=4)
mpl.rc("figure.subplot", wspace=0.3, hspace=0.4)

AX_INDEX_TO_DATA_INDEX = {
    0: 5,
    1: 1,
    2: 2,
    3: 0,
    4: 3,
    5: 4,
}


def plot_group_results(
    # figure params
    fig_name=None,
    fig_width=60 / 25.4,
    fig_height=None,
    row_n=2,
    col_n=3,
    suptitle=None,
    # data params
    root_dir=autoroot.root / "logs/gpt2-imdb-toxicity/",
    fig_folder="logs/figs",
    file_pattern=r"results_\d+seeds.csv",
    # axes params
    titles=[  # noqa: B006
        "Stealthiness of Adv. Prompts",
        "Lexical Diversity of Adv. Prompts",
        "Semantic Diversity of Adv. Prompts",
        "Toxicity of Responses",
        "Lexical Diversity of Responses",
        "Semantic Diversity of Responses",
    ],
    ylabels=[  # noqa: B006
        r"$1 + R_{S}$",
        r"$1 + R_{D}^{L-\pi}$",
        r"$1 + R_{D}^{S-\pi}$",
        r"\% of Toxic Adv. Prompts",
        r"$1 + R_{D}^{L-f}$",
        r"$1 + R_{D}^{S-f}$",
    ],
    xlabels=["Toxicity Threshold"] * 6,
    # legend params
    legend=None,
    legend_order=None,
    # style params
    color_style="tab10",
    color_ids=None,
    linestyles=None,
):
    fig_name = fig_name or file_pattern.split("/")[0].replace("/", "_").replace(".", "_")
    fig_height = fig_width / 6.4 * 4.8
    palette = PALETTE[color_style]
    os.chdir(root_dir)

    # prepare data
    file_list = find_all_files(root_dir, file_pattern)
    file_list = [os.path.relpath(f, root_dir) for f in file_list]
    file_list.sort()
    legend = [k.split("/")[0] for k in file_list] if legend is None else legend
    legend_dict = {k: legend[i] for i, k in enumerate(file_list)}
    legend_order = list(range(len(legend_dict))) if legend_order is None else legend_order
    legend_ranking = {legend[i]: k for i, k in enumerate(legend_order)}
    file_list = sorted(file_list, key=lambda x: legend_ranking[legend_dict[x]])
    pp.pprint(file_list)

    # plot
    fig, axes = plt.subplots(row_n, col_n, figsize=(fig_width * col_n, fig_height * row_n))
    axes = axes.flatten()
    for file_index, file in enumerate(file_list):
        results = np.loadtxt(file, delimiter=",")
        for ax_index, ax in enumerate(axes):
            data_index = AX_INDEX_TO_DATA_INDEX[ax_index]
            x, y, y_std = results[:, 0], results[:, data_index * 2 + 1], results[:, data_index * 2 + 2]
            if y[0] < 0:
                y += 1
            color_id = file_index % len(palette) if color_ids is None else color_ids[file_index]
            linestyle = LINESTYLE_DICT["n_solid" if linestyles is None else linestyles[file_index]]
            ax.plot(x, y, color=palette[color_id], label=legend_dict[file], linestyle=linestyle, marker=".")
            ax.fill_between(x, y - y_std, y + y_std, color=palette[color_id], alpha=0.2)
            title = titles[ax_index]
            ax.set_title(title)
            ax.set_xlabel(xlabels[ax_index])
            ax.set_ylabel(ylabels[ax_index])
            ax.grid(True)
            ax.yaxis.set_major_formatter(formatter)

    # add legend
    fig.legend(
        *axes[-1].get_legend_handles_labels(),
        loc="center",
        bbox_to_anchor=[0.5, -0.01],
        ncol=len(file_list),
        frameon=True,
    )
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14, y=1.2)

    # save
    path = autoroot.root / fig_folder
    os.makedirs(path, exist_ok=True)
    fig.savefig(path / (fig_name + ".jpg"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"save {fig_name} to {path}")


if __name__ == "__main__":
    plot_group_results()
