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
            gamma_value = int(match.group(1)) / 10000  # convert back to float
            gammas.append(gamma_value)

    gammas.sort()

    for gamma in gammas:
        gamma_folder = f"{base_dir}gamma_{int(gamma * 10000)}"

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
    # plt.show()
    plt.savefig("extension_vs_shearstress.png", dpi=300, bbox_inches='tight')
            

# plot_validation_results([0.0005,0.005,0.05],config_path="configs/default.json", base_dir="data/validation/")  # Example gamma values