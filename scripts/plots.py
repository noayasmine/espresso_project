import os
import sys
import json

current_file = os.path.realpath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
sys.path.append(project_root)

from src.plotting_validation import plot_validation_results, plot_extension_over_time
from src.plotting_griddistances import plot_grid_results
from src.plotting_time_series import plot_simulation_results
from src.plotting_griddistances_avg import plot_grid_results_avg

def load_config(path):
    with open(path) as f:
        return json.load(f)

config_path = "/gpfs/home4/nroebersen/espresso_project/configs/default.json"
config = load_config(config_path)

# base_dir_validation_coil = "/gpfs/home4/nroebersen/espresso_project/data/validation_coil_old/"
# # LJ_folders = [f for f in os.listdir(base_dir_validation_coil) if os.path.isdir(os.path.join(base_dir_validation_coil, f))]

# # plot_extension_over_time(base_dir_validation_coil, config)

# base_dir_grids = "/gpfs/home4/nroebersen/espresso_project/data/test_grids_single_runs/"
# # plot_grid_results(config, base_dir_grids)

# base_dir_validation_uncoil = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil/"
# # plot_simulation_results(config, base_dir_validation_uncoil, plot_type="extension")
# # plot_simulation_results(config, base_dir_validation_uncoil, plot_type="force")

# base_dir_grids_avg = "/gpfs/home4/nroebersen/espresso_project/data/test_grids_stdev010/"
# # plot_grid_results_avg(config, base_dir_grids_avg)

# base_dir_grids_avg = "/gpfs/home4/nroebersen/espresso_project/data/test_grids_stdev001/"
# # plot_grid_results_avg(config, base_dir_grids_avg)

# # plotting of the uncoiling dynamics of a chosen LJ strength for multiple values of gamma
# base_dir_uncoiling_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil/LJ_00100/"
# plot_validation_results(config, base_dir_uncoiling_single)
# base_dir_uncoiling_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil/LJ_00500/"
# plot_validation_results(config, base_dir_uncoiling_single)
# base_dir_uncoiling_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil/LJ_01000/"
# plot_validation_results(config, base_dir_uncoiling_single)
# base_dir_uncoiling_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil_10K/LJ_00100/"
# plot_validation_results(config, base_dir_uncoiling_single)
# base_dir_uncoiling_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil_10K/LJ_00500/"
# plot_validation_results(config, base_dir_uncoiling_single)
# base_dir_uncoiling_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil_100K/LJ_00100/"
# plot_validation_results(config, base_dir_uncoiling_single)
# base_dir_uncoiling_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil_100K/LJ_00500/"
# plot_validation_results(config, base_dir_uncoiling_single)


# base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil/LJ_00100/gamma_00050/"
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)

# base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil/LJ_00500/gamma_00050/"
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)

# base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil_10K/LJ_00100/gamma_00050/"
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)


# base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil_10K/LJ_00500/gamma_00050/"
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)

# base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil_10K/LJ_00700/gamma_00050/"
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)

# base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/validation_uncoil_100K/LJ_00100/gamma_00050/"
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
# plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)

base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/LJ_00500/gamma_00050/"
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)
base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/LJ_00500/gamma_00100/"
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)
base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/LJ_00500/gamma_00025/"
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)

base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/LJ_00250/gamma_00050/"
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)
base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/LJ_00250/gamma_00100/"
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)
base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/LJ_00250/gamma_00025/"
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)

base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/LJ_01000/gamma_00050/"
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)
base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/LJ_01000/gamma_00100/"
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)
base_dir_uncoiling_time_series_single = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/LJ_01000/gamma_00025/"
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="extension")
plot_simulation_results(config, base_dir_uncoiling_time_series_single, plot_type="force", target_shear=11000)