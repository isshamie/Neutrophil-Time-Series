import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
from src.params import FIGURES_DIR, RESULTS, PROCESSED, DATA_DIR
import click
import numpy as np
from mplh.fig_utils import num_rows_cols, helper_save


def load_data(data_f, meta_f):
    meta = pd.read_csv(meta_f, sep="\t", index_col=0)
    data = pickle.load(open(data_f, "rb"))
    return data, meta


def plot_embedding_features_loop(data, meta, umap_dir, fig_dir ):
    for curr_param in glob.glob(f"{umap_dir}/*p"):
        print(curr_param)
        neigh = curr_param.strip(".p").split("_")[-2]
        dist = curr_param.strip(".p").split("_")[-1]
        plot_embedding_features(data, meta, curr_param, fig_dir, min_neighbor=neigh, min_distance=dist)
    return


def plot_embedding_features(data, meta, umap_f, fig_dir, min_neighbor=None, min_distance=None, labels_f=None, labels_to_keep=None, geno_col="Sample", max_samp=5000):

    "results/{norm}/{dim}_out/{min_neighbor}_{min_distance}/embedding_{sim}.p"
    #for curr_param in glob.glob(f"{umap_dir}/*p"):
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    inp = pickle.load(open(umap_f, "rb"))

    if not len(inp) == 2:
        return
    curr_df = subset_on_clusters(data, meta, umap_f, labels_f=labels_f, labels_to_keep=labels_to_keep)
    # Add the well information
    curr_df["Well"] = curr_df.apply(
        lambda x: str(x["Stimuli"]) + "_" + str(x["Sample"]), axis=1)

    # Run on full data hex density of timepoints
    run_plot_hex_tp(curr_df, col="Well", vals=None, outdir=fig_dir, show_cbar=True)
    for i in data.columns.values:
        f = plt.figure()

        hb = plt.hexbin(x=curr_df["embedding_1"].values, y=curr_df["embedding_2"].values,
                                       cmap=cmap,
                                       C=curr_df[i].values.astype(float),
                                       reduce_C_function=np.median)
        cb = f.colorbar(hb, ax=plt.gca())
        #plt.cm.YlOrRd_r
        helper_save(os.path.join(fig_dir, i + "_umap_hex_wells"),
                    to_svg=False)
        plt.close()

    if min_neighbor is None or min_distance is None:
        neigh, dist = os.path.basename(fig_dir).split("_")
    else:
        neigh = min_neighbor
        dist = min_distance


    # Create figures
    color_labels = curr_df["Stimuli"].unique()
    rgb_values = (sns.color_palette("Set2", len(color_labels)))
    #color_map = dict(zip(color_labels, rgb_values))

    if len(curr_df[geno_col].unique()) > 6:  # Too many style markers to plot
        print("No style marker will be used")
        geno_col = None

    if curr_df.shape[0] > max_samp:
        curr_df = curr_df.sample(max_samp)

    # Create separate panel for each well
    run_plot_hex(curr_df, col="Well", vals=None, norm_color=False,
                 show_cbar=False, outdir=fig_dir)
    run_plot(curr_df, col="Well", vals=None, outdir=fig_dir)

    for i in data.columns.values:
        plt.figure(figsize=(15, 15))
        cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
        sns.scatterplot(data=curr_df, x="embedding_1",
                        y="embedding_2", hue=i, palette=cmap) #, style=geno_col)#,
                        #size=0.5)
        # curr_df.plot.scatter("embedding_1","embedding_2", hue=curr_df['Stimuli'])#.map(color_map),s=0.5)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(fig_dir,
                                 f"{i.replace(' ','')}.png"))
        plt.savefig(os.path.join(fig_dir,
                                 f"{i.replace(' ','')}.pdf"))
        plt.close()


    pallete = sns.color_palette("bright", len(color_labels))
    plt.figure(figsize=(15, 15))
    sns.scatterplot(data=curr_df, x="embedding_1", y="embedding_2",
                    palette=pallete, hue='Stimuli',
                    style=geno_col) #, size=0.5)
    plt.legend(loc='upper right')
    title = f"Neighbors={neigh}\nMinimum Distance={dist}\nN Samples = {len(curr_df)}"

    plt.title(title)
    plt.savefig(
        os.path.join(fig_dir, f"umap_stimuli.png"))
    plt.savefig(
        os.path.join(fig_dir, f"umap_stimuli.pdf"))
    plt.close()

    plt.figure(figsize=(15, 15))
    sns.scatterplot(data=curr_df, x="embedding_1", y="embedding_2", hue='Timepoint',
                    style=geno_col, size=0.5)
    plt.legend(loc='upper right')
    title = f"Neighbors={neigh}\nMinimum Distance={dist}\nN Samples = {len(curr_df)}"
    plt.title(title)
    plt.savefig(
        os.path.join(fig_dir, f"umap_stimuli_tp.png"))
    plt.savefig(
        os.path.join(fig_dir, f"umap_stimuli_tp.pdf"))
    plt.close()
    return


def subset_on_clusters(data, meta, umap_f, labels_f=None, labels_to_keep=None):
    inp = pickle.load(open(umap_f, "rb"))

    if not len(inp) == 2:
        return
    embedding,  samples_ind = inp[0], inp[1]
    curr_df = meta.loc[samples_ind]
    curr_df["embedding_1"] = embedding[:, 0]
    curr_df["embedding_2"] = embedding[:, 1]
    curr_df = pd.merge(curr_df, data, how='inner', left_index=True,
                       right_index=True)
    #print(curr_df.head())
    if labels_f is not None:
        labels = pickle.load(open(labels_f, "rb"))
        curr_df["Cluster"] = labels
    else:
        return curr_df
    if labels_to_keep is not None:
        curr_df = curr_df[curr_df["Cluster"].astype(str) == str(labels_to_keep)]
    return curr_df.drop("Cluster", axis=1)


def run_plot_hex(curr_df, col="Well", vals=None,
                 norm_color=False, show_cbar=False, outdir=None, share=False):
    #cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    if vals is None:
        nrows, ncols = num_rows_cols(len(curr_df[col].unique()),
                                     max_cols=4)
    else:
        nrows, ncols = num_rows_cols(len(vals), max_cols=4)
        curr_df = curr_df[curr_df[col].isin(vals)]

    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15),
                         squeeze=False, sharex=share, sharey=share)
    count = 0
    print(nrows, ncols)
    for ind, val in curr_df.groupby(col):
        curr_r, curr_c = int(np.floor(count / ncols)), count % ncols
        hb = ax[curr_r, curr_c].hexbin(x=val["embedding_1"].values,
                                       y=val["embedding_2"].values,
                                       gridsize=(50, 50),
                                       cmap='viridis')
        # ax[curr_r, curr_c].axis('off')
        ax[curr_r, curr_c].set_title(ind)
        ax[curr_r, curr_c].set_xlabel("embedding_1")
        ax[curr_r, curr_c].set_ylabel("embedding_2")
        count += 1

        if not norm_color:
            if show_cbar:
                cb = f.colorbar(hb, ax=ax[curr_r, curr_c])
                cb.set_label(ind)  # Normalize based on max size
        else:
            hb = ax[curr_r, curr_c].hexbin(x=val["embedding_1"].values,
                                           y=val["embedding_2"].values,
                                           C=np.ones_like(val[
                                                              "embedding_2"].values,
                                                          dtype=np.float) / hb.get_array().max(),
                                           gridsize=(50, 50),
                                           cmap='viridis',
                                           reduce_C_function=np.sum)
            if show_cbar:
                cb = f.colorbar(hb, ax=ax[curr_r, curr_c])
                cb.set_label(ind)  # hb = plt.hexbin(x,y, cmap=plt.cm.YlOrRd_r)  # plt.axis([xmin, xmax, ymin, ymax])
    f.subplots_adjust(hspace=.7)

    if outdir is not None:
        helper_save(os.path.join(outdir, "umap_hex_wells"))
    return


def run_plot(curr_df, col="Well", vals=None, outdir=None, hue="Timepoint",
             share=False, out_f=""):
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    # cmap = sns.color_palette('Blues')
    if vals is not None:
        curr = curr_df[curr_df[col].isin(vals)]
        print(curr.shape)
        g = sns.lmplot(x="embedding_1", y="embedding_2", col=col,
                       col_wrap=4, data=curr, aspect=.8,
                       hue=hue, palette="Blues",
                       fit_reg=False, sharey=share,
                       sharex=share)  # kind="reg")
    else:
        g = sns.lmplot(x="embedding_1", y="embedding_2", col=col,
                       col_wrap=4, data=curr_df, aspect=.8,
                       hue=hue, palette="Blues",
                       fit_reg=False, sharey=share,
                       sharex=share)  # ,kind="reg")

    counts = curr_df[col].value_counts()
    for ax in g.axes:
        ax.set_title(str(ax.get_title()) + " N=" + str(
            counts[ax.get_title().split("=")[-1].strip()]))
    plt.xticks(rotation='vertical')

    if outdir is not None:
        if out_f == "":
            helper_save(os.path.join(outdir, "_umap_hex_wells"),to_svg=False)
        else:
            helper_save(
                os.path.join(outdir, out_f),
                to_svg=False)
    return g


def run_plot_hex_tp(curr_df, col="Well", vals=None, show_cbar=False, outdir=None,
                    color_col="Timepoint", share=False, out_f=""):
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    if vals is None:
        nrows, ncols = num_rows_cols(len(curr_df[col].unique()),
                                     max_cols=4)
    else:
        nrows, ncols = num_rows_cols(len(vals), max_cols=4)
        curr_df = curr_df[curr_df[col].isin(vals)]

    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15),
                         squeeze=False, sharey=share, sharex=share)
    count = 0
    print(nrows, ncols)
    for ind, val in curr_df.groupby(col):
        curr_r, curr_c = int(np.floor(count / ncols)), count % ncols

        hb = ax[curr_r, curr_c].hexbin(x=val["embedding_1"].values,
                                       y=val["embedding_2"].values,
                                       C=val[color_col].values.astype(
                                           float), cmap='viridis',
                                       reduce_C_function=np.median)

        # ax[curr_r, curr_c].axis('off')
        ax[curr_r, curr_c].set_title(ind)
        ax[curr_r, curr_c].set_xlabel("embedding_1")
        ax[curr_r, curr_c].set_ylabel("embedding_2")
        count += 1
        plt.suptitle("Color based on " + color_col)
        f.subplots_adjust(hspace=.7)
        if show_cbar:
            cb = f.colorbar(hb, ax=ax[curr_r, curr_c])
            cb.set_label(
                ind)  # hb = plt.hexbin(x,y, cmap=plt.cm.YlOrRd_r)  # plt.axis([xmin, xmax, ymin, ymax])

    if outdir is not None:
        if out_f == "":
            helper_save(os.path.join(outdir, color_col+"_umap_hex_wells"),to_svg=False)
        else:
            helper_save(
                os.path.join(outdir, color_col + "_" + out_f),
                to_svg=False)
    return


def subsamp(data, meta=None, n_subsample=-1):
    if n_subsample > 0 and n_subsample <= 1:  # Fraction
        print(f"{n_subsample} fraction")
        if meta is not None:
            samples = meta.groupby(
                ["Stimuli", "Sample", "Timepoint"]).apply(
                lambda x: x.sample(frac=n_subsample).reset_index())
        else:
            samples = data.groupby(
                ["Stimuli", "Sample", "Timepoint"]).apply(
                lambda x: x.sample(frac=n_subsample).reset_index())
    elif n_subsample==-1:
        return data
    else:
        if meta is not None:
            samples = meta.groupby(
                ["Stimuli", "Sample", "Timepoint"]).apply(
                lambda x: x.sample(n=int(n_subsample)).reset_index())
        else:
            samples = data.groupby(
                ["Stimuli", "Sample", "Timepoint"]).apply(
                lambda x: x.sample(n=int(n_subsample)).reset_index())

    samples = samples.set_index("index")
    samples = data.loc[samples.index]
    return samples



def hex_with_two_groups(g1,g2):
    ##TODO
    return

# Example to adjust colorbar
# n = 1000
# x = np.random.standard_normal(n)
# y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
#
# fig, axes = plt.subplots(nrows=2, ncols=2)
# for ax in axes.flat:
#     im = ax.hexbin(x,y)
#
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im, cax=cbar_ax)
#

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('data_f',type=click.Path(exists=True))
@click.argument('meta_f',type=click.Path(exists=True))
@click.argument('umap_f',type=click.Path(exists=True))
@click.argument('fig_dir',type=click.Path())
@click.argument('min_neighbor',type=click.INT)
@click.argument('min_distance',type=click.FLOAT)
@click.option('--cluster_f',type=click.Path(exists=True), default=None)
@click.option('--labels_to_keep',type=click.STRING, default=None)
def main_command_line(data_f, meta_f, umap_f, fig_dir, min_neighbor, min_distance, cluster_f, labels_to_keep):
    print("plotting umap results")
    data, meta = load_data(data_f, meta_f)
    plot_embedding_features(data, meta, umap_f, fig_dir,
                            min_neighbor=min_neighbor, min_distance=min_distance, labels_f=cluster_f, labels_to_keep=labels_to_keep)
    return


if __name__ == "__main__":
    main_command_line()
