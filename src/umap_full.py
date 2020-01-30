from src.run_umap import run_umap
from src.params import PROCESSED, DATA_DIR, RESULTS

import pymc3
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import time
import os
from scipy.stats import zscore
import threading
import pickle

dir_save = os.path.join(RESULTS,"umap_out", "0")
if not os.path.exists(dir_save):
    os.mkdir(dir_save)
f_save = os.path.join(dir_save, "embedding.p")

## Load in z-scored data

df = pickle.load(open(os.path.join(PROCESSED,"data_df_log10.p"),"rb"))
print(df.shape)
df.head()

## Run umap

run_umap(df, f_save=f_save, n_neighbors=500, min_distance=0, attrs=None)
