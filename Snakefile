import os
import pandas as pd
from snakemake.utils import validate
import numpy as np
from src import run_umap
configfile: "parameters/parameters_snakemake_001.yaml"
#from src.utils import logs

# samples = pd.read_table(config["samples"], dtype=str).set_index(["sample"], drop=False)
# RAW_SAMPLES = samples.apply(lambda x: os.path.join(x["raw"], x["bam"]), axis=1)
#print('index',samples.index)


normalizations = config["normalizations"]
reduct_technique = config["reduction"]
n_iters = np.arange(config["n_iters"])
n_subsample = config["n_subsample"]
min_neighbors= config["min_neighbors"]
min_neighbors = list(map(lambda x: int(x), min_neighbors))
min_distances = config["min_distances"]
min_distances = list(map(lambda x: float(x), min_distances))

features = config["features"]
print(features)
cluster = config["cluster_technique"]
min_cluster_size = config["min_cluster_size"]
min_sample = config["min_sample"]

#report: "report/workflow.rst"

rule all:
    input:
        expand("figures/panels/{norm}/{dim}_out/{min_neighbor}_{min_distance}/{cluster}_out/{min_sample}_{min_cluster_size}/clusters_over_time.png", dim=reduct_technique, sim=n_iters,norm=normalizations,
                                                                                                                              min_neighbor=min_neighbors, min_distance=min_distances,
                                                                                                                              features=features, cluster=cluster, min_sample=min_sample, min_cluster_size=min_cluster_size),
        # "data/processed/fc.tsv",
        # expand("data/processed/transform/data_df_{norm}.p", norm=normalizations),
        # expand("results/{norm}/{dim}_out/{min_neighbor}_{min_distance}/embedding_{sim}.p", dim=reduct_technique, sim=n_iters, min_neighbor=min_neighbors, min_distance=min_distances, norm=normalizations),
        expand("figures/{norm}/{dim}_out/{min_neighbor}_{min_distance}/sim{sim}/umap_stimuli.png", dim=reduct_technique, sim=n_iters, min_neighbor=min_neighbors, min_distance=min_distances, norm=normalizations),
        # expand("results/{norm}/{dim}_out/{min_neighbor}_{min_distance}/{cluster}_out/{min_sample}_{min_cluster_size}.p",dim=reduct_technique, sim=n_iters, min_neighbor=min_neighbors, min_distance=min_distances, norm=normalizations,
        #                                                                                                                         cluster=cluster, min_sample=min_sample, min_cluster_size=min_cluster_size),
        # expand("figures/panels/{min_sample}_{min_cluster_size}/{cluster}_out/{min_neighbor}_{min_distance}/{dim}_out/{norm}/cluster_over_time.png", dim=reduct_technique, sim=n_iters,norm=normalizations,
        #                                                                                                                       min_neighbor=min_neighbors, min_distance=min_distances,
        #                                                                                                                       features=features, cluster=cluster, min_sample=min_sample, min_cluster_size=min_cluster_size)


#sample=config["samples"]),
#expand("{raw_f}.bam.bai", raw_f=RAW_SAMPLES),
#
#
# def get_raw_bai(wildcards):
#     print(wildcards.sample)
#     bam = os.path.join(samples.loc[wildcards.sample, "raw"],samples.loc[wildcards.sample, "bam"])
#     return bam + ".bam.bai"
#


#make_data -> preprocess -> run_umap -> run_cluster -> plot_panels
rule tidy_data:
    """Go from raw data folder where each snapshot is its own csv file, to one big file"""
    input: "data/raw/fcs output"
    output:
        "data/processed/fc.tsv",
        "data/processed/meta.tsv"
    shell: "python src/make_data.py {input:q} data/processed/"


rule normalize_data:
    """Normalize the data according to norm"""
    input:
        data_f = "data/processed/fc.tsv",
        meta_f = "data/processed/meta.tsv"
    output: "data/processed/transform/data_df_{norm}.p"
    params:
        outdir = lambda wildcards, output: os.path.dirname(str(output)),
    shell: "python src/preprocess.py {input.data_f} {params.outdir} {wildcards.norm}"


rule run_umap:
    #"""Run umap"""
    input:
         data_f = "data/processed/transform/data_df_{norm}.p",
         meta_f = "data/processed/meta.tsv"
    output:
        "results/{norm}/{dim}_out/{min_neighbor}_{min_distance}/embedding_{sim}.p"
    params:
        outdir = lambda wildcards:  f"results/{wildcards.norm}/{wildcards.dim}_out", # output: os.path.dirname(str(output)),
        n_iter = config["n_iters"],
        n_subsample = n_subsample,
        features = "intensity"
    shell: "python src/run_umap.py {input.data_f} {input.meta_f} {params.outdir} {wildcards.min_neighbor} {wildcards.min_distance} {params.n_subsample} --n_iter {params.n_iter} --features {params.features}"


rule plot_umap:
    input:
        data_f = "data/processed/transform/data_df_{norm}.p",
        meta_f = "data/processed/meta.tsv",
        umap_f = "results/{norm}/{dim}_out/{min_neighbor}_{min_distance}/embedding_{sim}.p"
    output:
        stim_fig = "figures/{norm}/{dim}_out/{min_neighbor}_{min_distance}/sim{sim}/umap_stimuli.png",
    params:
        fig_dir = lambda wildcards: f"figures/{wildcards.norm}/{wildcards.dim}_out/{wildcards.min_neighbor}_{wildcards.min_distance}/sim{wildcards.sim}"
    shell: "python src/plot_umap_embeddings.py {input.data_f} {input.meta_f} {input.umap_f} {params.fig_dir} {wildcards.min_neighbor} {wildcards.min_distance}"


rule run_cluster:
    input:
        umap_f = "results/{norm}/{dim}_out/{min_neighbor}_{min_distance}/embedding_0.p"
    output:
        cluster_f = "results/{norm}/{dim}_out/{min_neighbor}_{min_distance}/{cluster}_out/{min_sample}_{min_cluster_size}.p",
        f_save_fig = "figures/{norm,^[^/]+$}/{dim,^[^/]+$}_out/{min_neighbor}_{min_distance}/{cluster}_out/{min_sample}_{min_cluster_size}.png"
    shell: "python src/clustering.py {input.umap_f} {output.cluster_f} {output.f_save_fig} {wildcards.cluster} {wildcards.min_sample} {wildcards.min_cluster_size}"


rule plot_panels:
    input:
        meta_f = "data/processed/meta.tsv",
        cluster_f = "results/{norm}/{dim,^[^/]+$}_out/{min_neighbor}_{min_distance}/{cluster}_out/{min_sample}_{min_cluster_size}.p",
        embedding_f = "results/{norm}/{dim}_out/{min_neighbor}_{min_distance}/embedding_0.p"
    output:
        f_save = "figures/panels/{norm}/{dim}_out/{min_neighbor}_{min_distance}/{cluster}_out/{min_sample}_{min_cluster_size}/clusters_over_time.png"
    run:
        shell("python src/plot_panel_clusters.py {input.meta_f} {input.cluster_f} {input.embedding_f} {output.f_save}")
        shell("echo input:{input:q} output:{output} wildcards:{wildcards} > {output.f_save}.parameters")


# rule plot_umap_cluster:
#     input:
#         data_f = "data/processed/transform/data_df_{norm}.p",
#         meta_f = "data/processed/meta.tsv",
#         umap_f = "results/{norm}/{dim}_out/{min_neighbor}_{min_distance}/embedding_{sim}.p",
#         cluster_f = "results/{norm}/{dim,^[^/]+$}_out/{min_neighbor}_{min_distance}/{cluster}_out/{min_sample}_{min_cluster_size}.p",
#         labels_to_keep = "0"
#     output:
#         stim_fig = "figures/{norm}/{dim}_out/{min_neighbor}_{min_distance}/sim{sim}/umap_stimuli.png"
#     params:
#         fig_dir = lambda wildcards: f"figures/{wildcards.norm}/{wildcards.dim}_out/{wildcards.min_neighbor}_{wildcards.min_distance}/sim{wildcards.sim}"
#     run:
#         shell("python src/plot_umap_embeddings.py {input.data_f} {input.meta_f} {input.umap_f} {params.fig_dir} {wildcards.min_neighbor} {wildcards.min_distance} {input.cluster_f} {input.labels_to_keep}")

#
# rule test:
#     input:
#          "tests/test.txt"
#     output:
#           touch("tests/{x}_out_test.txt")
#     run:
#         logs.save_parameters(output)
#
