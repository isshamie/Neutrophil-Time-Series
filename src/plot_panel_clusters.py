import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import os
from src.params import PARAM_DIR, RESULTS, FIGURES_DIR
from src.utils.config import read_config_file, write_config_file
from src.fig_utils import helper_save


def calc_percentages(data,cols=("Timepoint", "Stimuli Names",
                                "Genotype",
                                "Sample")):

    return data.groupby(cols).apply(lambda group: (group[
        "Cluster"].value_counts(
        )/len(group)).rename_axis('Label')).reset_index()


def plot_panel(data, f_save=None):
    """

    :param labels: 1d array corresponding to cluster labels
    :param index: 1d array corresponding to the actual raw data indices
    :param data: dataframe, which has the timepoint, stimuli,
    gene KO, and replicate
    :return:
    """

    g = sns.FacetGrid(data, row="Stimuli Names", col="Label", height=4,
                      hue="Genotype")
    g = g.map(sns.lineplot, "Timepoint", "Cluster",
              marker=".").add_legend()
    g.add_legend(label_order=g.hue_names)
    helper_save(f_save)
    return g


def create_panel(data_f, label_f, embedding_f, f_save=None):
    data = pd.read_csv(data_f, sep="\t", index_col=0)
    [_, indices] = pickle.load(open(embedding_f,"rb"))
    data = data.loc[indices]

    labels = pickle.load(open(label_f.replace(".p","")+".p",'rb'))
    data["Cluster"] = labels

    perc = calc_percentages(data, cols=("Timepoint", "Stimuli Names",
                                   "Genotype", "Sample"))

    plot_panel(perc, f_save=f_save)
    return


def all_filenames(p):
    for stage in p["stages"]:
        p = create_filenames(p, stage)
    return p


def create_filenames(p, stage):
    stage_p = p[stage]

    if stage == "data" or stage == "umap":
        return p

    p[stage]["data_folder"] = os.path.join(RESULTS, stage,
                                           stage_p["folder"])
    p[stage]["figure_folder"] = os.path.join(FIGURES_DIR, stage,
                                           stage_p["folder"])

    for f in p[stage]["filenames"]:
        p[stage]["filenames"][f] = os.path.join(p[stage][
                                                    "data_folder"],
                                                p[stage][
                                                    "filenames"][f])

    for f in p[stage]["figures"]:
        p[stage]["figures"][f] = os.path.join(p[stage][
                                                    "figure_folder"],
                                                p[stage][
                                                    "figures"][f])

    if not os.path.exists(p[stage]["data_folder"]):
        os.makedirs(p[stage]["data_folder"])
    if not os.path.exists(p[stage]["figure_folder"]):
        os.makedirs(p[stage]["figure_folder"])

    return p


def main(config):
    p = read_config_file(config)
    p = all_filenames(p)
    create_panel(data_f=p["data"]["filenames"]["meta"],
                 label_f=p["cluster"]["filenames"]["results"],
                 embedding_f = p["umap"]["filenames"]["embedding"],
                 f_save=p["plot_panels"]["figures"]["panel"])

    write_config_file(
        os.path.join(p["plot_panels"]["data_folder"], "input.yaml"), p)
    return


if __name__ == "__main__":
    config = os.path.join(PARAM_DIR, "params_sample_1.yaml")
    # config = os.path.join(PARAM_DIR,
    #                       "plot_cluster_params_sample_2.yaml")
    main(config)


#
#
# import seaborn as sns; sns.set()
# import matplotlib.pyplot as plt
# fmri = sns.load_dataset("fmri")
# ax = sns.lineplot(x="timepoint", y="signal", data=fmri)