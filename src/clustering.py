from src.utils.config import read_config_file, write_config_file
from src.params import PARAM_DIR, NUM_CORES, ROOT_DIR, RESULTS, FIGURES_DIR
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np

# Dimension reduction and clustering libraries
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from src.fig_utils import helper_save


sns.set(style='white', rc={'figure.figsize':(10,8)})


def plot_hdb(data, labels, f_save_fig=None, title=""):
    clustered = (labels >= 0)
    plt.scatter(data[~clustered, 0],
                data[~clustered, 1], c=(0.5, 0.5, 0.5),
                s=0.1, alpha=0.5)
    plt.scatter(data[clustered, 0],
                data[clustered, 1], c=labels[clustered],
                s=0.1, cmap='Spectral')
    plt.title(title + f"\nFraction unclustered: {(labels == -1).sum()/len(labels)}")
    plt.legend()
    helper_save(f_save_fig)
    plt.close()
    return


def hdb_cluster(data, min_s_ratio, min_clust_ratio, f_save=None,
                num_cores=None, min_s_num=None, min_clust_num=None,
                clust_select='eom'):
    n_obs = data.shape[0]
    min_samples = int(n_obs/min_s_ratio)
    min_cluster_size = int(n_obs/min_clust_ratio)

    if min_s_num is not None:
        min_samples = min_s_num
    if min_clust_num is not None:
        min_cluster_size = min_clust_num

    print(clust_select)
    # Assume needs to have
    if num_cores is not None:
        labels = hdbscan.HDBSCAN(min_samples=min_samples,
                                 min_cluster_size=min_cluster_size,
                                 core_dist_n_jobs=num_cores,cluster_selection_method=clust_select
                                 ).fit_predict(data)
    else:
        labels = hdbscan.HDBSCAN(min_samples=min_samples,
                                 min_cluster_size=min_cluster_size,cluster_selection_method=clust_select
                                 ).fit_predict(data)
    if f_save is not None:
        f_save = f_save.replace(".p", "") + ".p"
        pickle.dump(labels, open(f_save, "wb"))
    return labels


def compute_cluster_purity(target, predict):
    (adjusted_rand_score(target, predict),
     adjusted_mutual_info_score(target, predict))
    return


def compute_pca(data, n_components=2):
    lowd_pca = PCA(n_components=n_components).fit_transform(data)
    return lowd_pca


def run(p, test=None):
    embedding_f = p["umap"]["filenames"]["embedding"]
    f_save = p["cluster"]["filenames"]["cluster_results"]
    f_save_fig = p["cluster"]["filenames"]["cluster_label_figure"]

    [data, index] = pickle.load(open(embedding_f,"rb"))

    if test is not None:
        data = data[np.random.choice(data.shape[0], test,
                                     replace=False), :]
    cluster_type = p["cluster"]["params"]["method"]
    min_sample = p["cluster"]["params"]["min_sample"]
    min_cluster_size = p["cluster"]["params"]["min_cluster_size"]

    min_s_num = p["cluster"]["params"]["minimum_sample_number"]
    min_clust_num= p["cluster"]["params"]["minimum_cluster_size"]

    if p["cluster"]["params"]["clust_select"] is None:
        p["cluster"]["params"]["clust_select"] = 'eom'
    if cluster_type == "hdb":
        labels = hdb_cluster(data, min_sample, min_cluster_size,
                             f_save=f_save, min_s_num=min_s_num,
                             min_clust_num=min_clust_num, clust_select=p["cluster"]["params"]["clust_select"]) #
        if min_s_num is not None and min_clust_num is not None:
            title = f"s={min_s_num} c={min_clust_num}"
        else:
            title = f"min_sample={min_sample} clust size={min_cluster_size}"
        plot_hdb(data, labels, f_save_fig=f_save_fig, title=title)


    write_config_file(os.path.join(p["cluster"]["data_folder"],"input.yaml"), p)
    return


def main(config):
    p = read_config_file(config)

    stage_p = p["cluster"]

    p["cluster"]["data_folder"] = os.path.join(RESULTS, "cluster",stage_p["folder"])
    p["cluster"]["figure_folder"] = os.path.join(FIGURES_DIR, "cluster", stage_p["folder"])


    p["cluster"]["filenames"]["cluster_results"] = os.path.join(
        stage_p["data_folder"], stage_p["filenames"]["results"])

    p["cluster"]["filenames"]["cluster_label_figure"] = os.path.join(
        stage_p["figure_folder"], stage_p["filenames"]["results"])


    if not os.path.exists(os.path.join(RESULTS,"cluster")):
        os.mkdir(os.path.join(RESULTS,"cluster"))
    if not os.path.exists(os.path.join(FIGURES_DIR,"cluster")):
        os.mkdir(os.path.join(FIGURES_DIR,"cluster"))

    if not os.path.exists(p["cluster"]["data_folder"]):
        os.mkdir(p["cluster"]["data_folder"])
    if not os.path.exists(p["cluster"]["figure_folder"]):
        os.mkdir(p["cluster"]["figure_folder"])
    run(p)
    return


if __name__ == '__main__':
    config = os.path.join(PARAM_DIR, "params_sample_11.yaml")
    main(config)
    # config = os.path.join(PARAM_DIR, "params_sample_10.yaml")
    # main(config)
    # config = os.path.join(PARAM_DIR, "params_sample_9.yaml")
    # main(config)
    # config = os.path.join(PARAM_DIR, "params_sample_5.yaml")
    # main(config)
    # config = os.path.join(PARAM_DIR, "params_sample_6.yaml")
    # main(config)
    # config = os.path.join(PARAM_DIR, "params_sample_7.yaml")
    # main(config)
    # config = os.path.join(PARAM_DIR, "params_sample_8.yaml")
    # main(config)
    # #
    # config = os.path.join(PARAM_DIR, "params_sample_3.yaml")
    # main(config)
    #
    # config = os.path.join(PARAM_DIR, "params_sample_1.yaml")
    # main(config)
    #
    # config = os.path.join(PARAM_DIR, "params_sample_2.yaml")
    # main(config)

    # config = os.path.join(PARAM_DIR, "params_sample_4.yaml")
    # main(config)



