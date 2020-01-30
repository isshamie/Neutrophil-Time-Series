import pathlib
import os
from dotenv import load_dotenv
from dotenv.main import dotenv_values


## Get the project directory
path = os.path.abspath(__file__)
dir_path = pathlib.Path(os.path.dirname(path))
ROOT_DIR = dir_path.parents[0]
print(f"Project Directory: {ROOT_DIR}")


############
# Head directories to be used
############
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED = os.path.join(DATA_DIR, "processed")
PARAM_DIR = os.path.join(ROOT_DIR, "parameters")
RESULTS = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
NUM_CORES = 8

FEATURES = ['Cell Size', 'Cell Circularity','Cell Aspect Ratio', 'Cell Tracker Intensity', 'PI Intensity', 'AnexinV Intensity']
Stimuli_Names = {"A":"DMSO","B":"zVD", "C":"BPT", "D":"BPT_zVD",
                 "E":"BPT_Nec1s", "F":"BPT_zVD_Nec1s","G": "Nec1s","H":"PIK75"}


print("\nConfig paths:")
for name, value in globals().copy().items():
    if type(value) == str:
        print(name, value)

##################################
# .env variables
##################################
# Look in directories above for .env file
# ROOT_DIR = os.getenv("ROOT_DIR")
load_dotenv()
EMAIL = os.getenv("EMAIL")


PATH_DICT = dotenv_values()
if len(PATH_DICT) > 0:
    print("Hidden parameters to load:")
    for d in PATH_DICT:
        print(d)


