import os
import sys
import json

current_file = os.path.realpath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
sys.path.append(project_root)

from src.plotting_validation import plot_validation_results, plot_extension_over_time
from src.plotting_griddistances import plot_grid_results
from src.plotting_time_series import plot_simulation_results

def load_config(path):
    with open(path) as f:
        return json.load(f)

config_path = "/gpfs/home4/nroebersen/espresso_project/configs/default.json"
config = load_config(config_path)

base_dir_validation_coil = "/gpfs/home4/nroebersen/espresso_project/data/validation_coil_old/"
# LJ_folders = [f for f in os.listdir(base_dir_validation_coil) if os.path.isdir(os.path.join(base_dir_validation_coil, f))]

# plot_extension_over_time(base_dir_validation_coil, config)


base_dir_grids = "/gpfs/home4/nroebersen/espresso_project/data/test_grids/"
plot_grid_results(config, base_dir_grids)

base_dir_validation_uncoil = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil/"
# plot_simulation_results(config, base_dir_validation_uncoil, plot_type="extension")
# plot_simulation_results(config, base_dir_validation_uncoil, plot_type="force")