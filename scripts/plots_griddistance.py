import os
import sys
import json

current_file = os.path.realpath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
sys.path.append(project_root)

from src.plotting_griddistances_avg import plot_grid_results_avg

def load_config(path):
    with open(path) as f:
        return json.load(f)

config_path = "/gpfs/home4/nroebersen/espresso_project/configs/default.json"
config = load_config(config_path)
base_dir = "/home/nroebersen/espresso_project/validation_data/test_grids_1K"
plot_grid_results_avg(config, base_dir, plot_mode="mean_std")
base_dir = "/home/nroebersen/espresso_project/validation_data/test_grids_100K"
plot_grid_results_avg(config, base_dir, plot_mode="mean_std")
base_dir = "/home/nroebersen/espresso_project/validation_data/test_grids_10000K"
plot_grid_results_avg(config, base_dir, plot_mode="mean_std")