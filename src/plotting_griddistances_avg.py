import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

def compute_avg_polymer_distance(npz_path, n_polymers=2, monomers_per_polymer=30):
    try:
        data = np.load(npz_path)
        if "positions" not in data:
            print(f"Warning: 'positions' key not found in {npz_path}")
            return None

        positions = data["positions"]
        ntimesteps, total_monomers, _ = positions.shape
        expected = n_polymers * monomers_per_polymer
        if total_monomers != expected:
            print(f"Warning: Unexpected number of monomers in {npz_path}")
            return None

        reshaped = positions.reshape((ntimesteps, n_polymers, monomers_per_polymer, 3))
        polymer1 = reshaped[:, 0]
        polymer2 = reshaped[:, 1]

        distances = np.linalg.norm(polymer1 - polymer2, axis=-1)
        avg_distance = np.mean(distances, axis=1)

        return avg_distance
    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        return None

def plot_grid_results_avg(config, base_dir, plot_mode="mean_std"):
    assert plot_mode in {"mean_std", "individual"}, "Invalid plot_mode. Choose 'mean_std' or 'individual'."

    grid_pattern = re.compile(r"grid_(\d+)")
    shear_pattern = re.compile(r"shear_(\d+)")
    run_pattern = re.compile(r"run_\d+")

    STEPS = config["simulation"]["STEPS"]
    shear_data = {}

    for grid_folder in sorted(os.listdir(base_dir)):
        grid_match = grid_pattern.match(grid_folder)
        if not grid_match:
            continue

        grid_value = int(grid_match.group(1)) / 100.0
        grid_path = os.path.join(base_dir, grid_folder)

        # Define TIMESTEP based on grid
        if grid_value == 1:
            TIMESTEP = 0.001
        elif grid_value == 0.5:
            TIMESTEP = 0.000125
        elif grid_value == 0.25:
            TIMESTEP = 0.000015625
        else:
            continue

        for shear_folder in sorted(os.listdir(grid_path)):
            shear_match = shear_pattern.match(shear_folder)
            if not shear_match:
                continue

            shear_value = int(shear_match.group(1))
            shear_path = os.path.join(grid_path, shear_folder)

            run_curves = []
            for run_folder in sorted(os.listdir(shear_path)):
                if not run_pattern.match(run_folder):
                    continue

                run_path = os.path.join(shear_path, run_folder, "positions.npz")
                if os.path.isfile(run_path):
                    avg_distances = compute_avg_polymer_distance(run_path)
                    if avg_distances is not None:
                        dt = 1e-4 * TIMESTEP * STEPS
                        time_s = np.arange(len(avg_distances)) * dt
                        run_curves.append(avg_distances)

            if run_curves:
                try:
                    run_array = np.stack(run_curves)
                    time_s = np.arange(run_array.shape[1]) * dt

                    if plot_mode == "mean_std":
                        mean = np.mean(run_array, axis=0)
                        std = np.std(run_array, axis=0)
                        shear_data.setdefault(shear_value, []).append((grid_value, time_s, mean, std))
                    else:  # individual mode
                        for i, curve in enumerate(run_array):
                            shear_data.setdefault(shear_value, []).append((grid_value, time_s, curve, i))

                except Exception as e:
                    print(f"Error stacking data for shear={shear_value}, grid={grid_value}: {e}")

    # Create subplots
    n_subplots = len(shear_data)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 4 * n_subplots), sharex=True)

    if n_subplots == 1:
        axes = [axes]

    for ax, (shear_value, curves) in zip(axes, sorted(shear_data.items())):
        # Sort curves safely using a custom key
        if plot_mode == "mean_std":
            curves.sort(key=lambda x: x[0])  # sort by grid_value
        else:
            curves.sort(key=lambda x: (x[0], x[3]))  # sort by grid_value, run_idx

        for item in curves:
            grid_value = item[0]
            time_s = item[1]

            if plot_mode == "mean_std":
                mean = item[2]
                std = item[3]
                label = f"grid={grid_value}"
                ax.plot(time_s, mean, label=label)
                ax.fill_between(time_s, mean - std, mean + std, alpha=0.3)
            else:
                run_curve = item[2]
                run_idx = item[3]
                label = f"grid={grid_value} run={run_idx}"
                ax.plot(time_s, run_curve, label=label, alpha=0.6)

        ax.set_title(f"Shear = {shear_value}")
        ax.set_ylabel("Avg. distance")
        ax.legend(fontsize='small')
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    title_mode = "Mean ± Std" if plot_mode == "mean_std" else "Individual Runs"
    fig.suptitle(f"Polymer separation over time ({title_mode})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_suffix = "mean_std" if plot_mode == "mean_std" else "individual"
    save_path = os.path.join(base_dir, f"polymer_distances_{save_suffix}.png")
    plt.savefig(save_path)
