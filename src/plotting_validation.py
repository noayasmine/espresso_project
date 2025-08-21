import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

def plot_validation_results(config, base_dir):
    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("tab10").colors

    # Expected results (given list)
    expected_x = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    expected_y = [0.00, 0.35, 0.55, 0.63, 0.70, 0.75, 0.81, 0.85, 0.90, 0.92, 0.94, 0.96, 0.96]
    expected_std = [0.1, 0.2, 0.2, 0.25, 0.3, 0.35, 0.35,0.3, 0.3, 0.3,0.3,0.3,0.3]

    variable = "shear"
    max_extension = config["physical"]["N_MONOMERS"] * 2 * config["physical"]["a"]  # raw length scale
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

                # --- RAW extension (no normalization here) ---
                extension = np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=1)
                extensions.append(extension[-1])

            if not extensions:
                continue

            # Convert shear label to wall shear stress [dyn/cm²]
            shear_stress = float(shear_label) * dyn_visc_SI * 10
            shears.append(shear_stress)

            if len(extensions) == 1:
                means.append(extensions[0])
                stds.append(0.0)
            else:
                means.append(np.mean(extensions))
                stds.append(np.std(extensions))

        if shears:
            shears = np.array(shears, dtype=float)
            means = np.array(means, dtype=float)   # RAW mean extensions
            stds  = np.array(stds, dtype=float)    # RAW std of extensions

            # Sort by shear stress
            sort_idx = np.argsort(shears)
            shears = shears[sort_idx]
            means  = means[sort_idx]
            stds   = stds[sort_idx]

            # --- Normalize so curve starts at zero per your formula ---
            # ext0: extension at shear == 0 (if present), else use the smallest shear as reference
            zero_mask = (shears == 0)
            if np.any(zero_mask):
                ext0 = means[zero_mask][0]
            else:
                ext0 = means[0]  # nearest available (smallest shear)

            denom = (max_extension - ext0)
            if denom <= 0:
                print(f"[WARN] Normalization skipped for {gamma_folder_name}: "
                      f"max_extension ({max_extension:.6g}) <= ext0 ({ext0:.6g}).")
                norm_means = means.copy()
                norm_stds  = stds.copy()
            else:
                norm_means = (means - ext0) / denom
                norm_stds  = stds / denom  # subtraction doesn't affect std; only scale

            # ----------------------------------------------------------

            color = color_cycle[idx % len(color_cycle)]
            label = fr"$\gamma$ = {gamma_val:.4f}"

            if np.all(norm_stds == 0):
                ax.plot(shears, norm_means, label=label, color=color, marker='o',
                        markersize=4, linewidth=1.5)
            else:
                ax.plot(shears, norm_means, label=label, color=color, marker='o',
                        markersize=4, linewidth=1.5)
                ax.fill_between(shears, norm_means - norm_stds, norm_means + norm_stds,
                                color=color, alpha=0.25)

    # Add expected results curve (already 0..1 normalized)
    ax.plot(expected_x, expected_y, label="Expected", color="black",
            linestyle="--", linewidth=2, marker='s', markersize=5)

    ax.set_title(f"Final Polymer Extension vs Shear Rate\n{lj_title}", fontsize=16)
    ax.set_xlabel("Wall shear stress [dyn/cm²]", fontsize=13)
    ax.set_ylabel("Total extension (normalized)", fontsize=13)
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
    TIMESTEP = config["simulation"]["TIME_STEP"]
    STEPS = config["simulation"]["STEPS"]

    gamma_dirs = [d for d in os.listdir(base_dir) if d.startswith("gamma_") and os.path.isdir(os.path.join(base_dir, d))]

    for gamma_dir in gamma_dirs:
        gamma_match = re.match(r"gamma_(\d+)", gamma_dir)
        if not gamma_match:
            continue

        gamma_value = int(gamma_match.group(1)) / 1e4
        gamma_path = os.path.join(base_dir, gamma_dir)
        positions_path = os.path.join(gamma_path, "positions.npz")

        if not os.path.exists(positions_path):
            print(f"Missing positions file in {positions_path}")
            continue

        try:
            npzfile = np.load(positions_path)
            positions = npzfile["positions"]  # shape: (timesteps, monomers, 3)
            extension = np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=1)

            # Normalize by the initial extension so that value at t=0 is 1.0
            initial_extension = extension[0]
            if initial_extension <= 0:
                print(f"Initial extension is zero or negative in {positions_path}; skipping normalization.")
                continue
            normalized_extension = extension / initial_extension

            dt = 1e-4 * TIMESTEP * STEPS
            time = np.arange(len(extension)) * dt
            label = f"γ={gamma_value:.4f}"
            plt.plot(time, normalized_extension, label=label)
        except Exception as e:
            print(f"Error reading {positions_path}: {e}")
            continue

    # Add expected curve
    expected_x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    expected_y = [1, 0.4, 0.2, 0.1, 0.08, 0.08, 0.07, 0.07]
    plt.plot(expected_x, expected_y, 'k--', label="Expected", linewidth=2)

    plt.title("Polymer Extension Over Time (normalized to t=0)")
    plt.xlabel("Time [s]")
    plt.ylabel("Extension / initial extension")
    plt.legend(fontsize='small')
    plt.tight_layout()
    output_path = os.path.join(base_dir, "extension_over_time.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def _minimum_image_xy(d, Lx, Ly):
    """
    Apply minimum image only in x and y; z is untouched.
    d: (..., 3) displacement vector(s)
    """
    if Lx is not None and np.isfinite(Lx):
        d[..., 0] = d[..., 0] - Lx * np.round(d[..., 0] / Lx)
    if Ly is not None and np.isfinite(Ly):
        d[..., 1] = d[..., 1] - Ly * np.round(d[..., 1] / Ly)
    # z is non-periodic → no wrapping
    return d

def unwrap_chain_once_xy(frame_positions, Lx, Ly):
    """
    Unwrap one frame by walking along bonded neighbors;
    only x,y use minimum image; z is left continuous as given.
    frame_positions: (N,3) wrapped coords
    Lx, Ly: box lengths (set to None if not periodic)
    returns: (N,3) unwrapped (up to translation)
    """
    r = np.asarray(frame_positions, dtype=np.float64)
    N = r.shape[0]
    out = np.empty_like(r)
    out[0] = r[0]
    for i in range(1, N):
        d = r[i] - r[i-1]
        d = _minimum_image_xy(d, Lx, Ly)
        out[i] = out[i-1] + d
    # optional: remove COM (all axes) – doesn’t affect distances
    out -= out.mean(axis=0, keepdims=True)
    return out

def end_to_end_series_xy(positions, Lx, Ly):
    """
    positions: (T,N,3) wrapped
    returns: (T,) end-to-end using xy-periodic unwrapping
    """
    T = positions.shape[0]
    e2e = np.empty(T, dtype=np.float64)
    for t in range(T):
        ru = unwrap_chain_once_xy(positions[t], Lx, Ly)
        e2e[t] = np.linalg.norm(ru[-1] - ru[0])
    return e2e

def plot_extension_over_time_free(base_dir, config):
    """
    Same plot as your tethered case, but with xy-only PBC handling.
    Lx, Ly: box lengths along x,y (numbers or None if not periodic)
    """
    
    plt.figure(figsize=(12, 6))
    TIMESTEP = config["simulation"]["TIME_STEP"]
    STEPS = config["simulation"]["STEPS"]
    Lx = config["geometry"]["BOX_Lx"]
    Ly = config["geometry"]["BOX_Lx"]
    gamma_dirs = [d for d in os.listdir(base_dir)
                  if d.startswith("gamma_") and os.path.isdir(os.path.join(base_dir, d))]

    for gamma_dir in gamma_dirs:
        m = re.match(r"gamma_(\d+)", gamma_dir)
        if not m:
            continue
        gamma_value = int(m.group(1)) / 1e4
        positions_path = os.path.join(base_dir, gamma_dir, "positions.npz")
        if not os.path.exists(positions_path):
            print(f"Missing positions file in {positions_path}")
            continue

        try:
            data = np.load(positions_path)
            positions = data["positions"]  # (T,N,3), wrapped in xy, free in z
            e2e = end_to_end_series_xy(positions, Lx, Ly)

            if e2e[0] <= 0:
                print(f"Initial end-to-end non-positive in {positions_path}; skipping normalization.")
                continue

            time = np.arange(len(e2e)) * (1e-4 * TIMESTEP * STEPS)
            plt.plot(time, e2e / e2e[0], label=f"γ={gamma_value:.4f}")
        except Exception as e:
            print(f"Error reading {positions_path}: {e}")
            continue
        print(f'Running for {gamma_value}')
    plt.title("Polymer End-to-End Distance Over Time (normalized, xy-periodic)")
    plt.xlabel("Time [s]")
    plt.ylabel("E2E / initial E2E")
    plt.legend(fontsize='small')
    plt.tight_layout()
    output_path = os.path.join(base_dir, "extension_over_time_free.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


