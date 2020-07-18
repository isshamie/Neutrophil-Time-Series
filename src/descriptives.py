import seaborn as sns
import pandas as pd
from src.fig_utils import helper_save
from src.config import FIGURES_DIR, DATA_DIR
from os.path import join, exists
import matplotlib.pyplot as plt
import os


def make_dirs(fig_dir):
    curr_dir = join(fig_dir, "descriptives")
    if not exists(curr_dir):
        os.mkdir(curr_dir)
    return curr_dir


def cells_distribution(meta, fig_dir):
    cells_per = meta.groupby(["Stimuli", "Sample", "Timepoint"]).size()
    cells_per.name = "cells"
    cells_per = cells_per.reset_index()

    plt.figure()
    sns.violinplot(
        meta.groupby(["Stimuli", "Sample", "Timepoint"]).size())
    plt.title("Number of cells per experiment timepoint")
    helper_save(join(fig_dir, "cells_per_experiment_timepoint"))


    plt.figure()
    sns.violinplot(meta.groupby(["Stimuli", "Sample"]).size())
    plt.title("Number of cells captured per experiment")
    helper_save(join(fig_dir, "cells_per_experiment"))


    g = sns.catplot(x="Stimuli", y="cells", row="Timepoint",
                    data=cells_per, kind="violin", height=10,
                    aspect=0.7)
    g.savefig(os.path.join(fig_dir, "cell_distribution.png"))
    g.savefig(os.path.join(fig_dir, "cell_distribution.pdf"))

    g = sns.catplot(x="Stimuli", y="cells", row="Timepoint",
                    data=cells_per, kind="violin", height=10,
                    aspect=0.7, hue="Genotype", split=True)

    g.savefig(os.path.join(fig_dir, "cell_distribution_geno.png"))
    g.savefig(os.path.join(fig_dir, "cell_distribution_geno.pdf"))



    sns.violinplot(meta.groupby(["Timepoint", "Stimuli",
                               "Genotype"]).size())
    plt.title("Number of cells per experiment")


def main():
    meta_df = pd.read_csv(join(DATA_DIR, "meta.tsv"), sep="\t",
                          index_col=0)
    curr_dir = make_dirs(FIGURES_DIR)
    cells_distribution(meta_df, curr_dir)

    return


if __name__ == "__main__":
    main()