import os
import pandas as pd
from snakemake.utils import validate

#configfile: "parameters/Zhu_Single_Cell_10X_genomics_Human2_002.yaml"

samples = config["samples"]
raw = config["raw"]
bam = config["bam"]
num_reads_filter = config["num_reads_filter"]
maxBP = config["maxBP"]
ref_fa = config["ref_fa"]
samples = pd.read_table(config["samples"], dtype=str).set_index(["sample"], drop=False)

#samples.index = samples.index.set_levels([i.astype(str) for i in samples.index.levels])  # enforce str in index


#print(raw["A"])

# raw folders, bam files with different name, and sample name (the new name)
#Problem: Indexing the file, and then wanting to switch the name
# Could do:
# A) copy the initial folder by using the input as the raw bam and then the output is the new name. Then index
# B) conf

#

RAW_SAMPLES = samples.apply(lambda x: os.path.join(x["raw"], x["bam"]), axis=1)

print('index',samples.index)

rule all:
    input:
        expand("figures/{sample}/{sample}_CB_coverage_hist.png",sample=samples["sample"].values),
        expand("data/processed/{sample}/{sample}_scPileup_{num_read}",sample=samples["sample"].values, num_read=config["num_reads_filter"]),
        expand("data/processed/{sample}/scPileup_concat_{num_read}/{sample}_{num_read}_all.coverage.txt.gz",sample=samples["sample"].values, num_read=config["num_reads_filter"]),
        expand("figures/{sample}/{sample}_{num_read}_MT_position.png", sample=samples["sample"].values, num_read=config["num_reads_filter"]),
        expand("figures/{sample}/{sample}_{num_read}_MT_position_coverage.png", sample=samples["sample"].values, num_read=config["num_reads_filter"])

#sample=config["samples"]),
#expand("{raw_f}.bam.bai", raw_f=RAW_SAMPLES),


def get_raw_bai(wildcards):
    print(wildcards.sample)
    bam = os.path.join(samples.loc[wildcards.sample, "raw"],samples.loc[wildcards.sample, "bam"])
    return bam + ".bam.bai"


# rule orig_index:
#     input:  get_raw_bam #"{raw_f}"  #lambda wildcards: f"{config['bam'][wildcards.sample]}"
#     output: "{bam_f}.bam.bai" #"{raw}/{bam}.bai"
#     shell: "samtools index {input}"



#make_data -> preprocess -> run_umap -> run_cluster -> plot_panels


rule process_data:
    """ """
    input: get_sample_bam
    output: "data/processed/{sample}/{sample}.bam"
    shell: "cp {input} {output}"

rule normalize_data:
    """ """
    input: get_sample_bam
    output: "data/processed/{sample}/{sample}.bam"
    shell: "cp {input} {output}"

rule run_umap:
    """ """
    input: "data/processed/{sample}/{sample}.bam"
    output: "data/processed/{sample}/{sample}.bam.bai"
    shell: "samtools index {input}"

rule run_cluster:
    """ """
    input:
        bam = "data/processed/{sample}/{sample}.bam",
        bai = "data/processed/{sample}/{sample}.bam.bai"
    output:
        mt_bam="data/processed/{sample}/{sample}.MT.bam",
        mt_bai="data/processed/{sample}/{sample}.MT.bam.bai"
    run:
        #shell("samtools {input.bam}")
        shell("samtools view -b {input.bam} MT > {output.mt_bam}")
        shell("samtools index {output.mt_bam}")


rule plot_panels:
    """ """
    input:
        bam = "data/processed/{sample}/{sample}.bam",
        bai = "data/processed/{sample}/{sample}.bam.bai"
    output:
        mt_bam="data/processed/{sample}/{sample}.MT.bam",
        mt_bai="data/processed/{sample}/{sample}.MT.bam.bai"
    run:
        #shell("samtools {input.bam}")
        shell("samtools view -b {input.bam} MT > {output.mt_bam}")
        shell("samtools index {output.mt_bam}")

