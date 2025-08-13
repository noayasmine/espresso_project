import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

def moving_average(data, N):
    """Compute simple moving average over axis=0."""
    if N <= 1:
        return data
    kernel = np.ones(N) / N
    if data.ndim == 1:
        return np.convolve(data, kernel, mode='valid')
    else:
        return np.array([np.convolve(col, kernel, mode='valid') for col in data.T]).T

def plot_simulation_results(config, base_dir, plot_type, target_shear=None, avg_window=50):
    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("tab10").colors

    print(f"Plotting simulation results from directory: {base_dir}")

    variable = "shear"
    max_extension = config["physical"]["N_MONOMERS"] * 2 * config["physical"]["a"]
    TIMESTEP = config["simulation"]["TIME_STEP"]
    STEPS = config["simulation"]["STEPS"]
    N_MONOMERS = config["physical"]["N_MONOMERS"]

    lj_match = re.search(r"LJ_(\d{5})", base_dir)
    gamma_match = re.search(r"gamma_(\d{5})", base_dir)

    lj_strength = int(lj_match.group(1)) / 1e2 if lj_match else None
    gamma_val = int(gamma_match.group(1)) / 1e4 if gamma_match else None

    lj_text = f"Lennard Jones potential = {lj_strength:.2f} × 4kT" if lj_strength is not None else ""
    gamma_text = f"$\\gamma$ = {gamma_val:.4f}" if gamma_val is not None else ""
    subtitle = ", ".join(filter(None, [gamma_text, lj_text]))

    dirs = [d for d in os.listdir(base_dir) if re.match(rf"{variable}_\d+", d)]

    if target_shear is not None:
        dirs = [d for d in dirs if d == f"{variable}_{target_shear}"]
        if not dirs:
            print(f"No data found for shear = {target_shear}")
            return

    if plot_type not in ["extension", "force"]:
        print(f"Unsupported plot_type: {plot_type}")
        return

    if plot_type == "extension":
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, folder in enumerate(dirs):
            data_dir = os.path.join(base_dir, folder)
            shear_val = int(re.match(r"shear_(\d+)", folder).group(1))
            shear_label = f"{shear_val} 1/s"

            file_path = os.path.join(data_dir, "positions.npz")
            if not os.path.isfile(file_path):
                continue

            try:
                npzfile = np.load(file_path)
                data = npzfile["positions"]
                values = np.linalg.norm(data[:, -1, :] - data[:, 0, :], axis=1)
                values /= max_extension
                values = moving_average(values, avg_window)
            except Exception:
                continue

            dt = 1e-4 * TIMESTEP * STEPS
            time_s = np.arange(len(values)) * dt
            color = color_cycle[idx % len(color_cycle)]

            ax.plot(time_s, values, label=shear_label, color=color, linewidth=1.5)

        ax.set_title(f"Polymer extension over time\n{subtitle}", fontsize=16)
        ax.set_ylabel("Normalized extension", fontsize=13)
        ax.set_xlabel("Time [s]", fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        if ax.has_data():
            ax.legend(fontsize=10, frameon=False)
        ax.grid(True)
        plt.tight_layout()

        output_path = os.path.join(base_dir, "extension.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved extension plot to {output_path}")

    elif plot_type == "force":
        fig, axs = plt.subplots(N_MONOMERS, 1, figsize=(10, 2 * N_MONOMERS), sharex=True)
        if N_MONOMERS == 1:
            axs = [axs]

        for idx, folder in enumerate(dirs):
            data_dir = os.path.join(base_dir, folder)
            shear_val = int(re.match(r"shear_(\d+)", folder).group(1))
            shear_label = f"{shear_val} 1/s"

            file_path = os.path.join(data_dir, "drag_forces.npz")
            if not os.path.isfile(file_path):
                continue

            try:
                npzfile = np.load(file_path)
                data = npzfile["forces"]
            except Exception:
                continue

            dt = 1e-4 * TIMESTEP * STEPS
            time_s = np.arange(data.shape[0]) * dt

            # Precompute average once for all monomers
            data_avg = moving_average(np.linalg.norm(data, axis=2) * 1000, avg_window)
            time_s = np.arange(len(data_avg)) * dt  # Adjust time after averaging
            N=int(len(data_avg)/10)
            
            color = color_cycle[idx % len(color_cycle)]
            for m in range(N_MONOMERS):
                axs[m].plot(time_s[-N:], data_avg[:, m][-N:], label=shear_label, color=color, linewidth=1.2)
                axs[m].set_ylabel(f"M{m} [pN]", fontsize=9)
                axs[m].tick_params(axis='y', labelsize=8)
                axs[m].grid(True)

        for ax in axs:
            if ax.has_data():
                ax.legend(fontsize=8, frameon=False)
        axs[-1].set_xlabel("Time [s]", fontsize=12)
        plt.suptitle(f"Force on each monomer over time\n{subtitle}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        output_path = os.path.join(base_dir, "force_per_monomer.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved force subplot per monomer to {output_path}")


def plot_velocity_profile(config, base_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import re

    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("tab10").colors

    print(f"Plotting velocity profiles from: {base_dir}")
    variable = "shear"
    AGRID = config["geometry"]["AGRID"]

    dirs = [d for d in os.listdir(base_dir) if re.match(rf"{variable}_\d+.\d+", d)]
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, folder in enumerate(dirs):
        data_dir = os.path.join(base_dir, folder)
        match = re.match(r"shear_(\d+).(\d+)", folder)
        if not match:
            continue

        try:
            vprofile = np.load(os.path.join(data_dir, "velocity_profile.npy"))
            z_pos = np.load(os.path.join(data_dir, "z_positions.npy"))
        except FileNotFoundError:
            print(f"Missing velocity data in {folder}, skipping.")
            continue

        shear_val = (vprofile[-1] - vprofile[0]) / (len(z_pos) - (AGRID / 0.1 + 1)) * 1e5
        label = f"shear {round(shear_val)} 1/s"
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(z_pos, vprofile, label=label, color=color, linewidth=1.5)

    ax.set_title("Velocity profile", fontsize=16)
    ax.set_xlabel(r"z [$\mu$m]", fontsize=13)
    ax.set_ylabel(r"$v_z$ [cm/s]", fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True)
    plt.tight_layout()
    output_path = os.path.join(base_dir, "velocity_profile.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved velocity profile plot to {output_path}")
