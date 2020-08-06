from src.params import PROCESSED, DATA_DIR, RESULTS
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pickle
import time

def run_full():
    dir_save = os.path.join(RESULTS,"tsne_out", "0")
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    f_save = os.path.join(dir_save, "embedding.p")

    ## Load in z-scored data

    df = pickle.load(open(os.path.join(PROCESSED,"data_df_log10.p"),"rb"))
    print(df.shape)
    df.head()

    ## Run umap
    embedding = TSNE(n_components=2,perplexity=50).fit_transform(
        df.sample(1000))
    pickle.dump(obj=embedding, file=open(f_save, "wb"))

    plt.scatter(embedding[:,0], embedding[:,1])


def run_tsne(data, f_save=None, n_neighbors=100, min_distance=0, attrs=None):
    ## Run umap
    embedding = TSNE(n_components=2, perplexity=50).fit_transform(
        data.sample(1000))
    pickle.dump(obj=embedding, file=open(f_save, "wb"))
    return


def run_tsne_transform(data, f_save=None, n_comp=2, perp=50, attrs=None):
    """ Runs tsne with specified parameters and saves the output transformed data along with actual embeddings and
        the length of time it took. This will take longer than run_tsne because it also receives the embedding."""
    t1 = time.time()
    trans = TSNE(n_components=int(n_comp), perplexity=int(perp)).fit_transform(data)
    if attrs is not None:
        data = data[attrs]
    t2 = time.time()
    print(f"Time in seconds:{(t2-t1)/60}")
    if f_save is not None:
        f_save = f_save.replace(".p","") + ".p"
        pickle.dump([trans, data.index], open(f_save, 'wb'))
        with open(f_save.replace(".p", "") + "_time_took.txt", "w") as f:
            f.write(str(t2-t1))
    return trans


def subsample_and_run_tsne(data, meta, n_iter, n_components, perplexity, savedir, n_subsample, attrs=None, overwrite=True, has_geno=False):
    """
    Function that
     subsamples the data and runs umap a number of iterations and saving the output.
    :param data: Data containing meta-data and full data. Output from prepare_data
    :param n_iter: Number of simulations
    :param n_neighbors_l: List of UMAP minimum_neighbor parameter values
    :param min_distance_l: List of UMAP minimum_distance parameter values
    :param savedir: Directory to save
    :param n_subsample: Number of samples for each iteration.
    :param attrs: Features of the data to keep. If None, will keep all.
    :return:
    """
    print("savedir",savedir)
    for i in range(n_iter):
        print('i', i)
        for n_comp in n_components:
            print('number of components', n_comp)
            for perp in perplexity:
                print('perplexity', perp)
                # File name
                curr_f_save = f"{savedir}/{n_comp}_{perp}/embedding_{i}.p"
                if os.path.exists(curr_f_save) and not overwrite:
                    print(f"Already ran {curr_f_save}")
                else:
                    print("Running")
                    # Collect samples
                    if n_subsample != 0:
                        if has_geno:
                            if n_subsample>0 and n_subsample<=1: #Fraction
                                print(f"{n_subsample} fraction")
                                samples = meta.groupby(
                                    ["Stimuli", "Genotype", "Timepoint"]).apply(
                                    lambda x: x.sample(frac=n_subsample).reset_index())
                                samples = samples.set_index("index")
                            else:
                                samples = meta.groupby(
                                    ["Stimuli", "Genotype", "Timepoint"]).apply(
                                    lambda x: x.sample(n=int(n_subsample)).reset_index())
                                samples = samples.set_index("index")
                        else:
                            if n_subsample>0 and n_subsample<=1: #Fraction
                                print(f"{n_subsample} fraction")
                                samples = meta.groupby(
                                    ["Stimuli", "Sample", "Timepoint"]).apply(
                                    lambda x: x.sample(frac=n_subsample).reset_index())
                            else:
                                samples = meta.groupby(
                                    ["Stimuli", "Sample", "Timepoint"]).apply(
                                    lambda x: x.sample(n=int(n_subsample)).reset_index())
                            samples = samples.set_index("index")
                        print(samples.head())
                    else:
                        samples=data
                        print(samples.head())

                    samples = data.loc[samples.index]
                    run_tsne_transform(samples, curr_f_save, n_comp, perp, attrs)
    return
