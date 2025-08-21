import os
import sys
import json

current_file = os.path.realpath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
sys.path.append(project_root)

# from src.plotting_validation import plot_validation_results, plot_extension_over_time
# from src.plotting_validation_free_flow import plot_validation_results_free, plot_simulation_results_free
# from src.plotting_griddistances import plot_grid_results
# from src.plotting_time_series import plot_simulation_results
# from src.plotting_griddistances_avg import plot_grid_results_avg

from src.all_plotting import plot_time_series, plot_validation_results

def load_config(path):
    with open(path) as f:
        return json.load(f)

config_path = "/gpfs/home4/nroebersen/espresso_project/configs/default.json"
config = load_config(config_path)

base_dir = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/"
LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
for LJ_folder in LJ_folders:
    base_dir_LJ = os.path.join(base_dir, LJ_folder)
    plot_validation_results(config, base_dir_LJ, validation_type="tethered")
    gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

    for gamma_folder in gamma_folders:
        base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
        plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="tethered")
        plot_time_series(config, base_dir_gamma, plot_type="force", validation_type="tethered", target_shear=11636)

# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_free_100K/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_validation_results(config, base_dir_LJ, validation_type="free")
#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="free")
#         plot_time_series(config, base_dir_gamma, plot_type="force", validation_type="free", target_shear=11636)

# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_free_100K_Schneider/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_validation_results(config, base_dir_LJ, validation_type="free", Box_Lx=100)
#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="free", Box_Lx=100)
#         plot_time_series(config, base_dir_gamma, plot_type="force", validation_type="free", Box_Lx=100, target_shear=11636)


# base_dir_grids = "/gpfs/home4/nroebersen/espresso_project/data/test_grids_single_runs/"
# plot_grid_results(config, base_dir_grids)

# base_dir_validation_coil = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_coil_100K/"
# plot_extension_over_time(base_dir_validation_coil, config)

# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_coil_100K/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_extension_over_time(base_dir_LJ, config)


# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_free_100K/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_validation_results_free(config, base_dir_LJ)
#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         print(f"running for {base_dir_gamma}")
#         plot_simulation_results_free(config, base_dir_gamma, plot_type="extension")
#         # plot_simulation_results(config, base_dir_gamma, plot_type="force", target_shear=11636)

