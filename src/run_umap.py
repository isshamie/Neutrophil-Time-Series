import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import time
import os
from scipy.stats import zscore
import pickle
import threading
import click
import yaml


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


def run_umap(data, f_save=None, n_neighbors=100, min_distance=0,
			 attrs=None):
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


def run_umap_transform(data, f_save=None, n_neighbors=100,
					   min_distance=0, attrs=None):
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

maximumNumberOfThreads = 4
threadLimiter = threading.BoundedSemaphore(maximumNumberOfThreads)


class EncodeThread(threading.Thread):
	def run_umap(self, data, f_save):
		threadLimiter.acquire()
		try:
			run_umap(data=data, f_save=f_save)
		finally:
			threadLimiter.release()


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('parameter')#, type=click.Path(exists=True))
def main(parameter):
	yaml.load(parameter)
	return


if __name__ == "__main__":
	main(["parameters/1.yaml"])

