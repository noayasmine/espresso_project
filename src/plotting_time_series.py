import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

def plot_simulation_results(config, base_dir, plot_type):
    print(f"Plotting simulation results from directory: {base_dir}")

    variable = "shear"
    max_extension = config["physical"]["N_MONOMERS"] * 2 * config["physical"]["a"] 
    AGRID = config["geometry"]["AGRID"]
    TIMESTEP = config["simulation"]["TIME_STEP"]

    dirs = [d for d in os.listdir(base_dir) if re.match(rf"{variable}_\d+.\d+", d)]

    if plot_type=="velocity_profile":
        plt.figure(figsize=(12, 6))
        for folder in dirs:
            data_dir = os.path.join(base_dir, folder)

            # Parse folder name
            match = re.match(r"shear_(\d+).(\d+)", folder)
            if not match:
                continue
            velocity_label = f"{int(match.group(1))}.{match.group(2)}"
            
            # Load data
            try:
                vprofile = np.load(f"{data_dir}/velocity_profile.npy")
                z_pos = np.load(f"{data_dir}/z_positions.npy")


            except FileNotFoundError:
                print(f"Skipping {folder}: missing data files.")
                continue

            calculated_shear = (vprofile[int(len(vprofile)-1)] - vprofile[0])/(len(z_pos)-(AGRID/0.1 + 1)) * 1e5
            label_shear = f"shear {round(calculated_shear)} 1/s" 

            plt.plot(z_pos, vprofile, label=f"{label_shear}")

        plt.title("Velocity profile")
        plt.xlabel(r"z $\mu m$")
        # 10^{-2} m/s
        plt.ylabel(r"$v_z [cm/s]$")
        plt.legend()
        plt.show()

    if plot_type=="extension":
        plt.figure(figsize=(12, 6))
        for folder in dirs:
            data_dir = os.path.join(base_dir, folder)

            match = re.match(r"shear_(\d+)", folder)
            if not match:
                continue
            shear_label = f"{int(match.group(1))} 1/s"

            try:
                positions = np.load(f"{data_dir}/positions.npy")

            except FileNotFoundError:
                print(f"Skipping {folder}: missing data files.")
                continue

            extension = np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=1)
            extension = extension / max_extension  # normalized to max extension (50 x (a + r_0) = 50 x 2a = 5)

            dt = 1e-4 * TIMESTEP  # seconds per timestep
            print(len(extension))
            time_s = np.arange(len(extension)) * dt

            plt.plot(time_s, extension, label=f"{shear_label}")

        plt.title("Polymer extension over time")
        plt.xlabel("Time [s]")
        plt.ylabel("Normalized extension")
        plt.legend()
        plt.show()

    if plot_type=="force":
        plt.figure(figsize=(12, 6))
        for folder in dirs:
            data_dir = os.path.join(base_dir, folder)

            match = re.match(r"shear_(\d+)", folder)
            if not match:
                continue
            shear_label = f"{int(match.group(1))} 1/s"
            
            # Load data
            try:
                forces = np.load(f"{data_dir}/forces.npy")

            except FileNotFoundError:
                print(f"Skipping {folder}: missing data files.")
                continue

            force_mag = np.linalg.norm(forces[:, 1, :], axis=1)  # Magnitude of the force on the first monomer

            dt = 1e-4 * TIMESTEP  # seconds per timestep
            time_s = np.arange(len(force_mag)) * dt

            plt.plot(time_s, force_mag * 1000, label=f"{shear_label}")  # Convert to pN

        plt.title("Tension on first monomer")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [pN]")
        plt.legend()
        plt.show()

# plot_simulation_results("./data/validation/gamma_500/")  # Adjust the path as needed