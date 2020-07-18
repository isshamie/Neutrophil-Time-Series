from src.params import PROCESSED

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import time
import os
from os.path import join
from scipy.stats import zscore
import pickle
from pandarallel import pandarallel
import sys

pandarallel.initialize(nb_workers=36)



def group_by_replicate(var, df):
    # Create a dataframe for each variable, where each column is
    # Need to remember the indices to unstack by.

    groupings = df.groupby(["Stimuli", "Sample"])
    max_val = groupings.size().max()
    new_df = pd.DataFrame(index=range(max_val), columns=(
        groupings.groups.keys()), dtype=float)
    new_df_inds = pd.DataFrame(index=range(max_val), columns=(
        groupings.groups.keys()), dtype=str)

    for ind,val in groupings:
        new_df.loc[:len(val)-1,ind] = val[var].values
        new_df_inds.loc[:len(val) - 1, ind] = val.index
    return new_df, new_df_inds


# #Normalize by sample
# def quantile_norm(df, perc=None):
#     if perc is None:
#         rank_mean = df.stack().groupby(
#             df.rank(method='first').stack().astype(int)).mean()
#         df = df.rank(method='min').stack().astype(int).map(
#             rank_mean).unstack()
#     else:
#         # Get the perc value for each column and divide each column
#         # by it
#         df.divide(df.quantile(0.75, axis=0), axis=1)
#     return df


def create_quantile_df(df, vars, perc=0.75):
    groupings = df.groupby(["Stimuli", "Sample"])
    quantile_df = pd.DataFrame(index=groupings.groups.keys(),
                               columns=vars, dtype=str)
    for ind,val in groupings:
        quantile_df.loc[ind,:] = val[vars].quantile(perc, axis=0)
        # for v in vars:
        #     quantile_df.loc[ind,v] = val[v].quantile(perc)
    return quantile_df

#data_df.groupby(["Sample","Stimuli"]).apply()

def divide_quantile(s, quantile_matrix):
    vars = quantile_matrix.columns.values
    for v in vars:
        s[v] = s[v]/quantile_matrix.loc[(s["Stimuli"],s["Sample"]), v]

    return s


# Steps:
# 1) create group-by-variable dataframe where each element is the 75%
# value
# 2) divide each value in data by their respective location in the
# dataframe
# Algorithm:
# 1) A) create dataframe where index is TS-Stimuli indices, column is
# variables.
# B) groupby and then apply the percentile function to each group
# -variable.
# C) fill in dataframe with those values
#2) apply row-wise, the division of each variable. This can be done
# with pandarallel and then a loop over the variables
def wrap_quantile_norm(data, vars, out_dir=None, perc=0.75):
    quantile_df = create_quantile_df(data, vars)
    # data = data.apply(divide_quantile, args=(quantile_df,),
    #                         axis=1)
    print('here')
    data = data.parallel_apply(divide_quantile, args=(quantile_df,),
                             axis=1)

    if out_dir is not None:
        pickle.dump(obj=data, file=open(join(out_dir,
                                             f"data_quantNorm_{perc*100}.p"), "wb"))
    return data


def log_and_z(data_df, outdir, overwrite=True):
    if not overwrite:
        if os.path.exists(join(outdir, "data_df_log10_z.p")) and (os.path.exists(join(outdir, "data_df_log10.p"))
                and os.path.exists(join(outdir, "data_df_z.p"))):
            return
    ## Transformations
    data_df_log = np.log10(data_df)
    pickle.dump(data_df_log,
                open(join(outdir, "data_df_log10.p"), "wb"))

    data_df_z = data_df.apply(zscore, axis=0)
    pickle.dump(data_df_z, open(join(outdir, "data_df_z.p"), "wb"))

    data_df_log_z = data_df_log.apply(zscore, axis=0)
    pickle.dump(data_df_log_z,
                open(join(outdir, "data_df_log10_z.p"), "wb"))
    return


def main(data, outdir, transform, overwrite=True):
    # data_dir = PROCESSED
    # vars = ["Cell Size", "Cell Circularity", "Cell Aspect Ratio",
    #         "Cell Tracker Intensity","PI Intensity", "AnexinV Intensity"]

    ## Transformations
    data_df = pd.read_csv(data, sep="\t", index_col=0)
    cols = ["Sample", "Timepoint", "Stimuli"]
    cols = [c for c in cols if c in data_df.columns.values]

    if (transform == "log10" or transform == "log10_z") or transform == "z":
        log_and_z(data_df.drop(cols,
            axis=1), outdir, overwrite=overwrite)
        print("logging the data and z-scoring")
    elif transform == "quantile":
        vars = ["Cell Size", "Cell Circularity", "Cell Aspect Ratio",
                "Cell Tracker Intensity","PI Intensity", "AnexinV Intensity"]
        wrap_quantile_norm(data_df, out_dir=outdir,
                           vars=vars)
    return

#python src/preprocess.py data/processed/fc.tsv data/processed/transform log10

if __name__ == "__main__":
    data = sys.argv[1]
    outdir = sys.argv[2]
    transform = sys.argv[3]
    print("outdir", outdir)
    main(data, outdir, transform)

