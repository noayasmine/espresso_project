import os
import sys
import json

current_file = os.path.realpath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
sys.path.append(project_root)

# from src.all_plotting import plot_time_series, plot_validation_results, plot_tethered_across_LJ_for_gamma, \
# plot_cumulative_force_across_gammas, plot_recoiling_across_LJ_for_gamma, plot_free_across_LJ_for_gamma_max_one
# from src.plotting_griddistances_avg import plot_grid_results_avg
from scripts.plot_validation import plot_validation_results, plot_time_series, plot_tethered_across_LJ_for_gamma, plot_cumulative_force_across_gammas, \
    plot_recoiling_across_LJ_for_gamma, plot_cumulative_force_across_gammas_aggregate

def load_config(path):
    with open(path) as f:
        return json.load(f)

config_path = "/gpfs/home4/nroebersen/espresso_project/configs/default.json"
config = load_config(config_path)

# # make plot for gamma validation
# base_dir = "/home/nroebersen/espresso_project/validation/single_runs/gamma_validation/timestepping_001/gamma_validation_k500/LJ_01800"
# plot_cumulative_force_across_gammas(config, base_dir, target_shear=11636, avg_window=20,
#     use_last_fraction=0.15, start_from_right=50, skip=[0.006, 0.007, 0.008, 0.0115, 0.015])

# # make plot for gamma validation
# base_dir = "/home/nroebersen/espresso_project/validation/single_runs/gamma_validation_N100/timestepping_001/gamma_validation_k500/LJ_01800"
# plot_cumulative_force_across_gammas(config, base_dir, target_shear=11636, avg_window=20,
#     use_last_fraction=0.15, start_from_right=100)

# base_dir = "/home/nroebersen/espresso_project/validation/gamma_validation/N100/LJ_01800"
# plot_cumulative_force_across_gammas_aggregate(config, base_dir, target_shear=11636, avg_window=20, time_avg_steps=250, start_from_right=100, monomer_index=70,skip=[0.010, 0.012, 0.014, 0.015])

# make plot for tethered uncoiling data
# base_dir = "/home/nroebersen/espresso_project/validation/single_runs/gamma_validation/timestepping_001/gamma_validation_k500"
base_dir = "/home/nroebersen/espresso_project/validation/single_runs/visualization_timeseries_uncoiling_k50"
LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
for LJ_folder in LJ_folders:
    base_dir_LJ = os.path.join(base_dir, LJ_folder)
    # plot_validation_results(config, base_dir_LJ, validation_type="tethered")

    gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

    for gamma_folder in gamma_folders:
        base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
        plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="tethered")
        plot_time_series(config, base_dir_gamma, plot_type="force", validation_type="tethered", time_step=0.001, target_shear=11636, monomer_index=30, force_window=20, last_fraction=0.65)
        # plot_time_series(config, base_dir_gamma, plot_type="force", validation_type="tethered", time_step=0.001, target_shear=11636, force_window=1)

# base_dir = "/home/nroebersen/espresso_project/validation/N50/validation_uncoil_k500"
# target_gamma = "gamma_00110"
# plot_tethered_across_LJ_for_gamma(config, base_dir, target_gamma, aggregate_runs=True, N_mono=50)

# base_dir = "/home/nroebersen/espresso_project/validation/N100/validation_uncoil_k500"
# target_gamma = "gamma_00110"
# plot_tethered_across_LJ_for_gamma(config, base_dir, target_gamma, aggregate_runs=True, N_mono=100)

# base_dir = "/home/nroebersen/espresso_project/validation/N50/validation_coil_k500"
# target_gamma = "gamma_00110"
# plot_recoiling_across_LJ_for_gamma(config, base_dir, target_gamma, N_mono=50, skip=[16, 14, 12, 10])
# #plot_recoiling_across_LJ_for_gamma(config, base_dir, target_gamma, N_mono=50)
# base_dir = "/home/nroebersen/espresso_project/validation/N100/validation_coil_k500"
# target_gamma = "gamma_00110"
# plot_recoiling_across_LJ_for_gamma(config, base_dir, target_gamma, N_mono=100, skip=[22, 20, 18, 16, 14])
# #plot_recoiling_across_LJ_for_gamma(config, base_dir, target_gamma, N_mono=100)
