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

from src.all_plotting import plot_time_series, plot_validation_results, plot_tethered_across_LJ_for_gamma, \
plot_cumulative_force_across_gammas, plot_recoiling_across_LJ_for_gamma, plot_free_across_LJ_for_gamma

from src.plotting_griddistances_avg import plot_grid_results_avg

def load_config(path):
    with open(path) as f:
        return json.load(f)

config_path = "/gpfs/home4/nroebersen/espresso_project/configs/default.json"
config = load_config(config_path)

# plot the grid distances plots
# base_dir = "/home/nroebersen/espresso_project/validation_data/grid_test/test_grids_k1"
# plot_grid_results_avg(config, base_dir, plot_mode="mean_std")
# base_dir = "/home/nroebersen/espresso_project/validation_data/grid_test/test_grids_k10"
# plot_grid_results_avg(config, base_dir, plot_mode="mean_std")
# base_dir = "/home/nroebersen/espresso_project/validation_data/grid_test/test_grids_k100"
# plot_grid_results_avg(config, base_dir, plot_mode="mean_std")

# base_dir = "/home/nroebersen/espresso_project/validation_data/grid_test_gamma00180/test_grids_k1"
# plot_grid_results_avg(config, base_dir, plot_mode="mean_std")
# base_dir = "/home/nroebersen/espresso_project/validation_data/grid_test_gamma00180/test_grids_k10"
# plot_grid_results_avg(config, base_dir, plot_mode="mean_std")
# base_dir = "/home/nroebersen/espresso_project/validation_data/grid_test_gamma00180/test_grids_k100"
# plot_grid_results_avg(config, base_dir, plot_mode="mean_std")

# make plot for tensile force validation
# base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/single_runs/N50/validation_uncoil_k10/LJ_01400"
# plot_cumulative_force_across_gammas(config, base_dir, target_shear=11636, avg_window=1,
#     use_last_fraction=0.30, start_from_right=50, skip=[0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015])

# make plot for tethered uncoiling data
# base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/single_runs/N50/validation_uncoil_k10/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_validation_results(config, base_dir_LJ, validation_type="tethered")

#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="tethered", k_fene=10, a=0.03)
#         plot_time_series(config, base_dir_gamma, plot_type="force", validation_type="tethered", target_shear=11636, monomer_index=10, a=0.03)

base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/N50/validation_uncoil_k10/"
target_gamma = "gamma_00180"
plot_tethered_across_LJ_for_gamma(
    config,
    base_dir,
    target_gamma,
    aggregate_runs=True,
    #skip = {"LJ_00200", "LJ_00250", "LJ_00300", "LJ_00350"},
    N_mono=50,
    a=0.03
)

base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/N50/validation_coil_k10/"
plot_recoiling_across_LJ_for_gamma(config, base_dir, gamma_value=0.018, time_step=None, add_expected=True, N_mono=50, extension_metric="x")

base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/N100/validation_uncoil_k10/"
target_gamma = "gamma_00180"
plot_tethered_across_LJ_for_gamma(
    config,
    base_dir,
    target_gamma,
    aggregate_runs=True,
    #skip = {"LJ_00200", "LJ_00250", "LJ_00300", "LJ_00350"},
    N_mono=50,
    a=0.03
)

base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/N100/validation_coil_k10/"
plot_recoiling_across_LJ_for_gamma(config, base_dir, gamma_value=0.018, time_step=None, add_expected=True, N_mono=50, extension_metric="x")



######################################################################################################

######################################################################################################


# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/timeseries/validation_uncoil_100K/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_validation_results(config, base_dir_LJ, validation_type="tethered", target_gamma=0.0055)
#     plot_cumulative_force_across_gammas(config, base_dir_LJ, target_shear=11636, avg_window=1,
#     use_last_fraction=0.25)

#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="tethered", k_fene=6.5)
        
# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/timeseries/validation_uncoil_1K/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_validation_results(config, base_dir_LJ, validation_type="tethered", target_gamma=0.0055)
#     plot_cumulative_force_across_gammas(config, base_dir_LJ, target_shear=11636, avg_window=1,
#     use_last_fraction=0.25)

#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="tethered", k_fene=0.065)
 


# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/timeseries/validation_uncoil_free_10000K_04s_Schneider"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     # plot_validation_results(config, base_dir_LJ, validation_type="free", target_gamma=0.0055, Box_Lx=64)
#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="free", Box_Lx=64, time_step=0.00025, k_fene=650)
#         # plot_time_series(config, base_dir_gamma, plot_type="force", validation_type="free", target_shear=11636)
#         # plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="free", k_fene=650)
        

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

# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_free_100K/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_validation_results(config, base_dir_LJ, validation_type="free")
#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="free")
#         # plot_time_series(config, base_dir_gamma, plot_type="force", validation_type="free", target_shear=11636)

# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_free_100K/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_validation_results(config, base_dir_LJ, validation_type="free", free_metric="end_to_end", target_gamma=0.0055)
#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="free", free_metric="end_to_end")


# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/test_vtk/validation_uncoil_free_100K_comfixed/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_validation_results(config, base_dir_LJ, validation_type="free")
#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="free")
#         # plot_time_series(config, base_dir_gamma, plot_type="force", validation_type="free", target_shear=11636)


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

# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/"
# target_gamma = "gamma_00055"
# plot_tethered_across_LJ_for_gamma(
#     config,
#     base_dir,
#     target_gamma
# )

# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_10000K/"
# target_gamma = "gamma_00055"
# plot_tethered_across_LJ_for_gamma(
#     config,
#     base_dir,
#     target_gamma
# )

# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/longrun/validation_uncoil_100K/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# for LJ_folder in LJ_folders:
#     base_dir_LJ = os.path.join(base_dir, LJ_folder)
#     plot_validation_results(config, base_dir_LJ, validation_type="tethered", target_gamma=0.0055)
#     # plot_cumulative_force_across_gammas(config, base_dir_LJ, target_shear=11636, avg_window=1,
#     # start_from_right=30, use_last_fraction=0.25)

#     gamma_folders = [f for f in os.listdir(base_dir_LJ) if os.path.isdir(os.path.join(base_dir_LJ, f))]

#     for gamma_folder in gamma_folders:
#         base_dir_gamma = os.path.join(base_dir_LJ, gamma_folder)
#         plot_time_series(config, base_dir_gamma, plot_type="extension", validation_type="tethered", k_fene=6.5)
  

# base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/validation_coil_10000K/"
# plot_recoiling_across_LJ_for_gamma(config, base_dir, gamma_value=0.0055, time_step=None, add_expected=True, N_mono=30)
# base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/validation_coil_10000K_N60/"
# LJ_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
# plot_recoiling_across_LJ_for_gamma(config, base_dir, gamma_value=0.0055, time_step=None, add_expected=True, N_mono=60)

# base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/validation_uncoil_10000K_multiple/"
# target_gamma = "gamma_00055"
# plot_tethered_across_LJ_for_gamma(
#     config,
#     base_dir,
#     target_gamma,
#     aggregate_runs=True,
#     N_mono=30
# )

# base_dir = "/gpfs/home4/nroebersen/espresso_project/data/corrected_a/N50/validation_uncoil_k1000/"
# target_gamma = "gamma_00090"
# plot_tethered_across_LJ_for_gamma(
#     config,
#     base_dir,
#     target_gamma,
#     aggregate_runs=True,
#     N_mono=50,
#     a=0.03,
#     skip = {"LJ_00005", "LJ_00050", "LJ_00100", "LJ_00200"},
# )

# base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/N50/validation_uncoil_k1000/"
# target_gamma = "gamma_00090"
# plot_tethered_across_LJ_for_gamma(
#     config,
#     base_dir,
#     target_gamma,
#     aggregate_runs=True,
#     N_mono=50,
#     a=0.03
#     #skip = {"LJ_00300", "LJ_00600", "LJ_01000", "LJ_02400"}
# )

# base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/validation_uncoil_10000K_multiple_N60/"
# target_gamma = "gamma_00055"
# plot_tethered_across_LJ_for_gamma(
#     config,
#     base_dir,
#     target_gamma,
#     aggregate_runs=True,
#     skip = {"LJ_00200", "LJ_00250", "LJ_00300", "LJ_00350"},
#     N_mono=60
# )

# base_dir = "/gpfs/home4/nroebersen/espresso_project/validation_data/validation_uncoil_free_10000K_multiple/"
# plot_free_across_LJ_for_gamma(
#     config,
#     base_dir,
#     target_gamma="gamma_00055",                 # float like 0.0055 or string "gamma_00055"
#     free_metric="end_to_end",     # "end_to_end" (default) or "max_pairwise"
#     aggregate_runs=True,          # if True: mean±std over run_*; if False: show each run
#     include_expected=True,
#     Box_Lx=64
# ) 
