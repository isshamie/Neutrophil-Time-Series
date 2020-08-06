import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import time
import os
import sys
from scipy.stats import zscore
import pickle
import threading
import click
import yaml
from src.params import NUM_CORES
from src.utils.config import write_config_file
from MulticoreTSNE import MulticoreTSNE as TSNE
from src.tsne_full import subsample_and_run_tsne

# df = pickle.load(open("fc_z.p","rb"))
# t1 = time.time()
# reducer = umap.UMAP()
# embedding = reducer.fit_transform(data_df.iloc[:,:-1])
# embedding.shape
# t2 = time.time()
# print(f"Time in seconds:{(t2-t1)/60}")

# pickle.dump(embedding,open('umap_results.p','wb'))
# with open("time_took.txt","w") as f:
#     f.write(str(t2-t1))


def prepare_data(data_f, meta_f, features=None):
    """ Loads the sample annotation data and the processed data to use
    features: If None, use all features. Otherwise, should be a list of columns variables to keep in data"""
    df = pickle.load(open(data_f, "rb"))
    meta = pd.read_csv(meta_f, sep="\t", index_col=0)
    df = pd.concat((df, meta), axis=1)
    if features is not None:
        return df.loc[:, features]
    else:
        return df


def subsample_and_run(data, meta, n_iter, n_neighbors_l, min_distance_l, savedir, n_subsample, attrs=None, overwrite=True, has_geno=False):
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
    # First make sure data and meta have same indices
    data = data.loc[data.index.isin(meta.index)]
    meta = meta.loc[meta.index.isin(data.index)]

    for i in range(n_iter):
        print('i', i)
        for neigh in n_neighbors_l:
            print('number of neighbors', neigh)
            for dist in min_distance_l:
                print('minimum distance', dist)
                # File name
                curr_f_save = f"{savedir}/{neigh}_{dist}/embedding_{i}.p"
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
                    run_umap_transform(samples, curr_f_save, neigh, dist, attrs)
    return


def run_umap(data, f_save=None, n_neighbors=100, min_distance=0,
             attrs=None):
    """ Runs UMAP with specified parameters and saves the output along with the length of time it took."""
    t1 = time.time()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_distance)
    if attrs is not None:
        data = data[attrs]
    embedding = reducer.fit_transform(data)
    t2 = time.time()
    print(f"Time in minutes:{(t2-t1)/60}")
    if f_save is not None:
        f_save = f_save.replace(".p", "") + ".p"
        pickle.dump([embedding,data.index],open(f_save,'wb'))
        with open(f_save.replace(".p","") + "_time_minutes.txt","w") as f:
            f.write(str((t2-t1)/60))
    return


def run_umap_transform(data, f_save=None, n_neighbors=100, min_distance=0, attrs=None):
    """ Runs UMAP with specified parameters and saves the output transformed data along with actual embeddings and
        the length of time it took. This will take longer than run_umap because it also receives the embedding."""
    t1 = time.time()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_distance)
    if attrs is not None:
        data = data[attrs]
    trans = reducer.fit(data)
    t2 = time.time()
    print(f"Time in seconds:{(t2-t1)/60}")
    if f_save is not None:
        f_save = f_save.replace(".p","") + ".p"
        pickle.dump(trans,open(f_save + "_fit",'wb'))
        pickle.dump([trans.transform(data), data.index], open(f_save, 'wb'))
        with open(f_save.replace(".p", "") + "_time_took.txt", "w") as f:
            f.write(str(t2-t1))
    return trans


maximumNumberOfThreads = NUM_CORES
threadLimiter = threading.BoundedSemaphore(maximumNumberOfThreads)


class EncodeThread(threading.Thread):
    def run_umap(self, data, f_save):
        threadLimiter.acquire()
        try:
            run_umap(data=data, f_save=f_save)
        finally:
            threadLimiter.release()

#
# CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
# @click.command(context_settings=CONTEXT_SETTINGS)
# @click.argument('parameter')#, type=click.Path(exists=True))
def main(data_f, meta_f, outdir, min_neighbor, min_distance, n_iter, n_subsample, features):
    #params = yaml.load(parameter)
    print('params',params)
    print('type n_neighbors', type(n_neighbors_l))
    data = prepare_data(data_f, meta_f)
    subsample_and_run(data, int(n_iter), [int(min_neighbor)], [float(min_distance_l)], outdir, int(n_subsample), attrs=None)
    write_config_file(outdir, params)
    return



# CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
# @click.command(context_settings=CONTEXT_SETTINGS)
# @click.argument('parameter')#, type=click.Path(exists=True))
# def main_yaml(parameter):
# 	""" TO DO"""
# 	params = yaml.load(parameter)
# 	return


# params:
# neighbors = min_neighbors,
# distances = min_distances,
# outdir = lambda wildcards, output: os.path.dirname(output),
# n_iter = n_iters,
# n_subsample = n_subsample
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('data_f',type=click.Path(exists=True))
@click.argument('meta_f',type=click.Path(exists=True))
@click.argument('outdir',type=click.Path())
@click.argument('min_neighbor',type=click.INT)
@click.argument('min_distance',type=click.FLOAT)
@click.argument('n_subsample',type=click.FLOAT)
@click.option('--n_iter',default=3, type=click.INT)
@click.option('--features',default='all', type=click.STRING)
@click.option('--embed',default='umap', type=click.STRING)
def main_command_line(data_f, meta_f, outdir, min_neighbor, min_distance, n_subsample, n_iter, features, embed):
    """ """

    #Generate random seed and store later
    seed = np.random.randint(2**31 - 1)
    np.random.seed(seed)

    print("running umap")
    #data = prepare_data(data_f, meta_f)
    data = pickle.load(open(data_f, "rb"))
    if features == "intensity":
        attrs =  ['Cell Tracker Intensity', 'PI Intensity','AnexinV Intensity']
    else:
        attrs = None
    meta = pd.read_csv(meta_f,sep="\t", index_col=0)
    #data = pd.concat((data, meta),axis=0)
    #print(data.head())

    if embed == "umap":
        subsample_and_run(data, meta, int(n_iter), [int(min_neighbor)], [float(min_distance)], outdir, float(n_subsample), attrs=attrs)
    else:
        print("Running Tsne")
        subsample_and_run_tsne(data, meta, int(n_iter), [int(min_neighbor)],
                          [float(min_distance)], outdir,
                          float(n_subsample), attrs=attrs)


    cmd = f"echo {seed} > {os.path.join(outdir,'random_seed.txt')}"
    print(cmd)
    os.system(cmd)
    #write_config_file(outdir, params)
    return


if __name__ == "__main__":
    main_command_line()

# data_f = sys.argv[1]
# meta_f = sys.argv[2]
# min_neighbor = sys.argv[3]
# outdir = sys.argv[4]
# n_iter = sys.argv[5]
# n_subsample = sys.argv[6]
# features = sys.argv[7]
# #params = sys.argv[3]
#
# output = "results/{dim}_out/{min_neighbor}_{min_distance}/embedding_{sim}.p"
# main(data_f, meta_f, min_neighbor, outdir, n_iter, n_subsample, features)

#main(["parameters/1.yaml"])
