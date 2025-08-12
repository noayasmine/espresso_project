import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json


def plot_validation_results(config, base_dir):

    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("tab10").colors

    variable = "shear"
    max_extension = config["physical"]["N_MONOMERS"] * 2 * config["physical"]["a"] 
    dyn_visc_SI = config["physical"]["VISC_dyn"] / 10

    # Extract LJ strength from base_dir
    lj_match = re.search(r"LJ_(\d{5})", base_dir)
    if lj_match:
        lj_strength = int(lj_match.group(1)) / 1e2
        lj_title = f"Lennard Jones potential = {lj_strength:.2f} × 4kT"
    else:
        lj_title = "Lennard Jones potential = unknown"

    # Match folders like gamma_00050, gamma_00100, etc.
    gamma_pattern = re.compile(r"gamma_(\d{5})")
    gamma_folders = sorted([
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and gamma_pattern.match(f)
    ])

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, gamma_folder_name in enumerate(gamma_folders):
        match = gamma_pattern.match(gamma_folder_name)
        if not match:
            continue

        gamma_val = int(match.group(1)) / 1e4  # e.g. 00050 → 0.0050
        gamma_folder = os.path.join(base_dir, gamma_folder_name)

        dirs = [d for d in os.listdir(gamma_folder) if re.match(rf"{variable}_\d+", d)]
        shears = []
        means = []
        stds = []

        for folder in dirs:
            shear_dir = os.path.join(gamma_folder, folder)
            match = re.match(r"shear_(\d+)$", folder)
            if not match:
                continue

            shear_label = int(match.group(1))
            run_dirs = [d for d in os.listdir(shear_dir) if re.match(r"run_\d+", d)]
            extensions = []

            for run in run_dirs if run_dirs else [""]:
                data_dir = os.path.join(shear_dir, run) if run else shear_dir
                file_path = os.path.join(data_dir, "positions.npz")
                if not os.path.isfile(file_path):
                    print(f"Missing file: {file_path}")
                    continue

                try:
                    npzfile = np.load(file_path)
                    positions = npzfile["positions"]
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

                extension = np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=1)
                extension = extension / max_extension
                extensions.append(extension[-1])

            if not extensions:
                continue

            shear_stress = float(shear_label) * dyn_visc_SI * 10
            shears.append(shear_stress)

            if len(extensions) == 1:
                means.append(extensions[0])
                stds.append(0)
            else:
                means.append(np.mean(extensions))
                stds.append(np.std(extensions))

        if shears:
            shears = np.array(shears)
            means = np.array(means)
            stds = np.array(stds)

            # Sort by shear stress
            sort_idx = np.argsort(shears)
            shears = shears[sort_idx]
            means = means[sort_idx]
            stds = stds[sort_idx]

            color = color_cycle[idx % len(color_cycle)]
            label = fr"$\gamma$ = {gamma_val:.4f}"

            if np.all(stds == 0):
                ax.plot(shears, means, label=label, color=color, marker='o',
                        markersize=4, linewidth=1.5)
            else:
                ax.plot(shears, means, label=label, color=color, marker='o',
                        markersize=4, linewidth=1.5)
                ax.fill_between(shears, means - stds, means + stds, color=color, alpha=0.25)

    # ax.set_title("Final Polymer Extension vs Shear Rate", fontsize=16)
    ax.set_title(f"Final Polymer Extension vs Shear Rate\n{lj_title}", fontsize=16)

    ax.set_xlabel("Wall shear stress [dyn/cm²]", fontsize=13)
    ax.set_ylabel("Extension (normalized)", fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True)

    plt.tight_layout()
    output_path = os.path.join(base_dir, "extension_vs_shearstress.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def plot_extension_over_time(base_dir, config):
    plt.figure(figsize=(12, 6))

    max_extension = config["physical"]["N_MONOMERS"] * 2 * config["physical"]["a"]

    LJ_dirs = [d for d in os.listdir(base_dir) if d.startswith("LJ_")]
    LJ_dirs.sort()

    for lj_dir in LJ_dirs:
        lj_path = os.path.join(base_dir, lj_dir)
        gamma_dirs = [d for d in os.listdir(lj_path) if d.startswith("gamma_") and os.path.isdir(os.path.join(lj_path, d))]

        for gamma_dir in gamma_dirs:
            gamma_match = re.match(r"gamma_(\d+)", gamma_dir)
            if not gamma_match:
                continue

            gamma_value = int(gamma_match.group(1)) / 1e4
            gamma_path = os.path.join(lj_path, gamma_dir)
            positions_path = os.path.join(gamma_path, "positions.npz")

            if not os.path.exists(positions_path):
                print(f"Missing positions file in {positions_path}")
                continue

            try:
                npzfile = np.load(positions_path)
                positions = npzfile["positions"]  # shape: (timesteps, monomers, 3)
                extension = np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=1)
                normalized_extension = extension / max_extension

                time = np.arange(len(extension)) * config["simulation"]["TIME_STEP"] * 1e-4
                label = f"{lj_dir}, γ={gamma_value:.4f}"
                plt.plot(time, normalized_extension, label=label)
            except Exception as e:
                print(f"Error reading {positions_path}: {e}")
                continue

    plt.title("Polymer Extension Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Extension (normalized)")
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig("extension_over_time.png", dpi=300)
