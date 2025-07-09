import os
import sys
import json

current_file = os.path.realpath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
sys.path.append(project_root)

from src.plotting_validation import plot_validation_results, plot_extension_over_time

base_dir = "/gpfs/home4/nroebersen/espresso_project/data/validation_coil/"
LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

def load_config(path):
    with open(path) as f:
        return json.load(f)

config_path = "/gpfs/home4/nroebersen/espresso_project/configs/default.json"
config = load_config(config_path)

plot_extension_over_time(base_dir, config)