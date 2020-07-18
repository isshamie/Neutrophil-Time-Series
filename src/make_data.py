from src.params import RAW_DIR, PROCESSED, Stimuli_Names
import pandas as pd
import glob
from os.path import join, basename
import sys


def parse_file(f):
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

    if len(glob.glob(join(raw_dir, "*csv"))) == 0:
        print(f"No data in folder {raw_dir}")
        return
    for i in glob.glob(join(raw_dir, "*csv")):
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


def main(indir, outdir):
    #create_dataframe(join(RAW_DIR,"fcs_output"), PROCESSED)
    print(indir)
    create_dataframe(indir,outdir)
    return


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
