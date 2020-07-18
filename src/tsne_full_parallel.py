from src.params import PROCESSED, DATA_DIR, RESULTS
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import os
import pickle

dir_save = os.path.join(RESULTS,"tsne_out", "0")
if not os.path.exists(dir_save):
    os.makedirs(dir_save)

f_save = os.path.join(dir_save, "embedding_parallel.p")

## Load in z-scored data

df = pickle.load(open(os.path.join(PROCESSED,"data_df_log10.p"),"rb"))
print(df.shape)
df.head()

## Run umap

embedding = TSNE(n_jobs=16,n_components=2,perplexity=50).fit_transform(df)
pickle.dump(obj=embedding, file=open(f_save, "wb"))

#plt.scatter(embedding[:,0], embedding[:,1])
