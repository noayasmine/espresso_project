import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

def plot_validation_results(config, base_dir):
    plt.figure(figsize=(12, 6))
    variable = "shear"
    max_extension = config["physical"]["N_MONOMERS"] * 2 * config["physical"]["a"] 
    dyn_visc = config["physical"]["VISC_dyn"]  # dyn/cm²
    AGRID = config["geometry"]["AGRID"]
    TIMESTEP = config["simulation"]["TIME_STEP"]

    gamma_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    gammas = []
    for folder in gamma_folders:
        match = re.match(r"gamma_(\d+)", folder)
        if match:
            gamma_value = int(match.group(1)) / 1e4  # convert back to float
            gammas.append(gamma_value)

    gammas.sort()

    for gamma in gammas:
        gamma_folder = f"{base_dir}gamma_{int(gamma * 1e4)}"

        if not os.path.exists(gamma_folder):
            print("Folder does not exist")
            continue

        dirs = [d for d in os.listdir(gamma_folder) if re.match(rf"{variable}_\d+", d)]

        shears = []
        extensions = []
        
        for folder in dirs:
            data_dir = os.path.join(gamma_folder, folder)

            match = re.match(r"shear_(\d+)$", folder)
        
            if not match:
                continue

            shear_label = int(match.group(1))

            try:
                npzfile = np.load(f"{data_dir}/positions.npz")
                positions = npzfile["positions"]  # Access the array by its key
                print("Loaded positions shape:", positions.shape)

            except FileNotFoundError:
                print(f"Skipping {folder}: missing data files.")
                continue

            extension = np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=1)
            extension = extension / max_extension  # normalized to max extension (Nmonomers x (a + r_0) = Nmonomers x 2a = 5)
            shear_stress = float(shear_label) * dyn_visc * 10 # shear stress in dyn/cm^2
            shears.append(shear_stress)
            extensions.append(extension[-1])
        
        plt.scatter(shears, extensions, label=r"$\gamma$=" + f"{gamma}", marker='o')

    plt.title("Final Polymer Extension vs Shear Rate")
    plt.xlabel("Wall shear stress (dyn/cm²)")
    plt.ylabel("Extension (normalized)")
    plt.legend()
    plt.savefig("extension_vs_shearstress.png", dpi=300, bbox_inches='tight')
            

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

                time = np.arange(len(extension)) * config["simulation"]["TIME_STEP"] * config["simulation"]["STEPS"]
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
