from src.params import RAW_DIR, PROCESSED, Stimuli_Names
import pandas as pd
from glob import glob
from os.path import join, basename, exists
import sys
import os

def parse_file(f ):
    """Parsing file based on how its named. Use different function if parsed differently to get annotation data."""
    vals = basename(f).replace(".csv", "").replace(" ","").split("-")
    d = dict()
    d["Timepoint"] = int(vals[-1].replace("Timepoint", ""))
    d["Stimuli"] = vals[1][0]
    d["Sample"] = int(vals[1][1:])
    return d


def create_dataframe(raw_dir, outdir):
    columns = ["Timepoint", "Stimuli", "Sample", "Cell Size",
               "Cell Circularity", "Cell Aspect Ratio",
               "Cell Tracker Intensity", "PI Intensity",
               "AnexinV Intensity"]
    data_df = pd.DataFrame(columns=columns)

    if len(glob(join(raw_dir, "*csv"))) == 0:
        print(f"No data in folder {raw_dir}")
        return
    for i in glob(join(raw_dir, "*csv")):
        print(i)
        d = parse_file(i)
        curr_df = pd.read_csv(i)
        curr_df["Timepoint"] = d["Timepoint"]
        curr_df["Stimuli"] = d["Stimuli"]
        curr_df["Sample"] = d["Sample"]
        curr_df.index = curr_df.apply(
            lambda x: str(x.name) + "_" + str(x["Sample"]) + "_" + x[
                "Stimuli"] + "_" + str(x["Timepoint"]), axis=1)
        data_df = pd.concat((data_df, curr_df), sort=False)
    if len(data_df) == 0:
        print("No files here! Not saving")
    metadata = data_df[["Sample", "Timepoint", "Stimuli"]]
    data_df = data_df.drop(["Sample", "Timepoint", "Stimuli"], axis=1)
    data_df.to_csv(join(outdir, "fc.tsv"), sep="\t")
    metadata["Stimuli Names"] = metadata["Stimuli"].map(Stimuli_Names)
    metadata["Genotype"] = "WT"
    metadata.loc[metadata["Sample"].astype(int) >= 6, "Genotype"] = "Pad4KO"
    metadata.to_csv(join(outdir, "meta.tsv"), sep="\t")
    return


### With the new kawasaki Test Run format:
def parse_file_kawasaki(f, params):
    """Parsing file based on how its named. Use different function if parsed differently to get annotation data."""
    vals = basename(f).replace(params["start"], "").replace(params["end"],"")
    d = dict()
    d["Stimuli"] = vals[0]
    d["Sample"] = int(vals[1:])
    return d


def create_dataframe_kawaski(raw_dir, outdir, params_name, wells=None, overwrite=False):
    if exists(outdir):
        if overwrite:
            print("Overwriting the directory")
        else:
            print("Directory already exists, please remove and rerun")
            return
    else:
        os.mkdir(outdir)
    columns = ["Timepoint", "Stimuli", "Sample", "Field Number", "Cell Size",
               "Cell Circularity", "Cell Aspect Ratio",
               "Cell Tracker Intensity", "PI Intensity",
               "AnexinV Intensity"]
    data_df = pd.DataFrame(columns=columns)

    if len(glob(join(raw_dir, "*csv"))) == 0:
        print(f"No data in folder {raw_dir}")
        return

    if wells is None:
        files = glob(join(raw_dir, params_name["start"]+"*"+params_name["end"]))

        #wells = list(map(lambda x: x.replace(params_name["start"],"").replace(params_name["end"],"")))
        print("Using all files")
    else:
        files = list(map(lambda x: join(raw_dir, params_name["start"]+x+params_name["end"]), wells))
    for i in files: # glob(join(raw_dir, "*"+params["end"])):
        print(i)
        d = parse_file_kawasaki(i, params=params_name)
        curr_df = pd.read_csv(i)
        curr_df = curr_df.rename({"Time Point": "Timepoint"}, axis=1)
        curr_df["Stimuli"] = d["Stimuli"]
        curr_df["Sample"] = d["Sample"]
        curr_df.index = curr_df.apply(
            lambda x: str(x.name) + "_" + str(x["Sample"]) + "_" + x[
                "Stimuli"] + "_" + str(x["Timepoint"]), axis=1)
        data_df = pd.concat((data_df, curr_df), sort=False)
    if len(data_df) == 0:
        print("No files here! Not saving")
    metadata = data_df[["Sample", "Timepoint", "Stimuli","Field Number"]]
    data_df = data_df.drop(["Sample", "Timepoint", "Stimuli","Field Number"], axis=1)
    data_df.to_csv(join(outdir, "fc.tsv"), sep="\t")
    #metadata["Stimuli Names"] = metadata["Stimuli"].map(Stimuli_Names)
    #metadata["Genotype"] = "WT"
    #metadata.loc[metadata["Sample"].astype(int) >= 6, "Genotype"] = "Pad4KO"
    metadata.to_csv(join(outdir, "meta.tsv"), sep="\t")
    return


def main(indir, outdir, is_kawa=False, params_name=None, wells=None):
    #create_dataframe(join(RAW_DIR,"fcs_output"), PROCESSED)
    print(indir)
    if not is_kawa:
        create_dataframe(indir,outdir)
    else:
        create_dataframe_kawaski(indir, outdir, params_name, wells=wells)
    return


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
