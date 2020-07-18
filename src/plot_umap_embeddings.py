import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
from src.params import FIGURES_DIR, RESULTS, PROCESSED, DATA_DIR
import click


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


def plot_embedding_features(data, meta, umap_f, fig_dir, min_neighbor=None, min_distance=None, labels_f=None, labels_to_keep=None):

    "results/{norm}/{dim}_out/{min_neighbor}_{min_distance}/embedding_{sim}.p"
    #for curr_param in glob.glob(f"{umap_dir}/*p"):
    inp = pickle.load(open(umap_f, "rb"))

    if not len(inp) == 2:
        return
    curr_df = subset_on_clusters(data, meta, umap_f, labels_f=labels_f, labels_to_keep=labels_to_keep)

    if min_neighbor is None or min_distance is None:
        neigh, dist = os.path.basename(fig_dir).split("_")
    else:
        neigh = min_neighbor
        dist = min_distance


    # Create figures
    color_labels = curr_df["Stimuli Names"].unique()
    rgb_values = (sns.color_palette("Set2", len(color_labels)))
    #color_map = dict(zip(color_labels, rgb_values))

    for i in data.columns.values:
        plt.figure(figsize=(15, 15))
        pallete = sns.color_palette("bright", len(color_labels))
        sns.scatterplot(data=curr_df, x="embedding_1",
                        y="embedding_2", hue=i, style="Genotype",
                        size=0.5)
        # curr_df.plot.scatter("embedding_1","embedding_2", hue=curr_df['Stimuli'])#.map(color_map),s=0.5)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(fig_dir,
                                 f"{i.replace(' ','')}.png"))
        plt.savefig(os.path.join(fig_dir,
                                 f"{i.replace(' ','')}.pdf"))
        plt.close()

    plt.figure(figsize=(15, 15))
    pallete = sns.color_palette("bright", len(color_labels))
    sns.scatterplot(data=curr_df, x="embedding_1", y="embedding_2",
                    palette=pallete, hue='Stimuli Names',
                    style="Genotype", size=0.5)

    plt.legend(loc='upper right')

    title = f"Neighbors={neigh}\nMinimum Distance={dist}\nN Samples = {len(curr_df)}"

    plt.title(title)
    plt.savefig(
        os.path.join(fig_dir, f"umap_stimuli.png"))
    plt.savefig(
        os.path.join(fig_dir, f"umap_stimuli.pdf"))
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

    if labels_f is not None:
        labels = pickle.load(open(labels_f, "rb"))
        curr_df["Cluster"] = labels
    else:
        return curr_df
    if labels_to_keep is not None:
        curr_df = curr_df[curr_df["Cluster"].astype(str) == str(labels_to_keep)]
    return curr_df.drop("Cluster", axis=1)



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
