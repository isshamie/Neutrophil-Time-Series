import pathlib
import os
# from dotenv import load_dotenv
# from dotenv.main import dotenv_values


## Get the project directory
path = os.path.abspath(__file__)
dir_path = pathlib.Path(os.path.dirname(path))
ROOT_DIR = dir_path.parents[0]
print(f"Project Directory: {ROOT_DIR}")


############
# Head directories to be used
############
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS = os.path.join(DATA_DIR, "processed")
#RESULTS_DIR = os.path.join(ROOT_DIR, "models")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
FIG_NET_DIR = os.path.join(FIGURES_DIR,"networks")
PIPELINE_F = os.path.join(ROOT_DIR, "reports/pipeline/Pipeline.xlsx")
DATA_FIG_DIR = os.path.join(DATA_DIR, "processed/figures")
PARAMS_DIR = os.path.join(ROOT_DIR, "parameters")
TEST_DIR = os.path.join(ROOT_DIR, "tests")
NUM_CORES = 32







