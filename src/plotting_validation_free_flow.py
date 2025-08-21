import numpy as np
import os, re
import matplotlib.pyplot as plt

def minimum_image_1d(dr, boxlen):
    """Apply the minimum-image convention to a 1D displacement."""
    if boxlen is None or boxlen <= 0:
        return dr
    # shift by nearest integer number of box lengths
    return dr - np.round(dr / boxlen) * boxlen

def unwrap_chain_frame(pos_frame, Lx=None, Ly=None, Lz=None):
    """
    Unwrap a single frame of a polymer (shape: [N, D]) using minimum-image,
    building positions from bond vectors to remove PBC breaks.
    Works for D=2 or D=3. Only axes with a valid box length are treated as periodic.
    """
    N, D = pos_frame.shape
    unwrapped = np.empty_like(pos_frame)
    unwrapped[0] = pos_frame[0]

    for i in range(1, N):
        dr = pos_frame[i] - pos_frame[i-1]

        # X is usually periodic in your setup
        dr_x = minimum_image_1d(dr[0], Lx)
        if D >= 2:
            dr_y = minimum_image_1d(dr[1], Ly)
        if D == 3:
            dr_z = minimum_image_1d(dr[2], Lz)

        if D == 1:
            dr_corr = np.array([dr_x])
        elif D == 2:
            dr_corr = np.array([dr_x, dr_y])
        else:  # D == 3
            dr_corr = np.array([dr_x, dr_y, dr_z])

        unwrapped[i] = unwrapped[i-1] + dr_corr

    return unwrapped

def end_to_end_with_pbc(positions, Lx=None, Ly=None, Lz=None):
    """
    positions: array shaped [T, N, D]
    Returns the end-to-end distance per frame after unwrapping (shape [T]).
    """
    T, N, D = positions.shape
    e2e = np.empty(T, dtype=float)

    for t in range(T):
        unwrapped = unwrap_chain_frame(positions[t], Lx=Lx, Ly=Ly, Lz=Lz)
        # rotation/translation invariant end-to-end distance
        e2e[t] = np.linalg.norm(unwrapped[-1] - unwrapped[0])

    return e2e

def plot_validation_results_free(config, base_dir):
    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("tab10").colors

    variable = "shear"
    max_extension = config["physical"]["N_MONOMERS"] * 2 * config["physical"]["a"]
    dyn_visc_SI = config["physical"]["VISC_dyn"] / 10

    # --- Box sizes / periodicity ---
    # X is periodic per your description; include Y/Z if present.
    Lx = config["geometry"].get("BOX_Lx", None)
    Ly = config["geometry"].get("BOX_Ly", None)  # if your config uses BOX_Ly
    Lz = config["geometry"].get("BOX_Lz", None)  # optional 3D

    # Extract LJ strength from base_dir
    lj_match = re.search(r"LJ_(\d{5})", base_dir)
    if lj_match:
        lj_strength = int(lj_match.group(1)) / 1e2
        lj_title = f"Lennard Jones potential = {lj_strength:.2f} × 4kT"
    else:
        lj_title = "Lennard Jones potential = unknown"

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

        gamma_val = int(match.group(1)) / 1e4
        gamma_folder = os.path.join(base_dir, gamma_folder_name)

        dirs = [d for d in os.listdir(gamma_folder) if re.match(rf"{variable}_\d+", d)]
        shears, means, stds = [], [], []

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
                    positions = npzfile["positions"]   # shape [T, N, D], D=2 or 3
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

                # --- NEW: end-to-end with PBC unwrapping ---
                e2e = end_to_end_with_pbc(positions, Lx=Lx, Ly=Ly, Lz=Lz)

                # Use the last frame (to match your original intent)
                extensions.append(e2e[-1])

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

        if not shears:
            continue

        shears = np.array(shears, dtype=float)
        means = np.array(means, dtype=float)
        stds  = np.array(stds, dtype=float)

        # Sort by shear stress
        sort_idx = np.argsort(shears)
        shears, means, stds = shears[sort_idx], means[sort_idx], stds[sort_idx]

        # Normalize to start at zero
        zero_mask = (shears == 0)
        ext0 = means[zero_mask][0] if np.any(zero_mask) else means[0]
        denom = (max_extension - ext0)
        if denom <= 0:
            print(f"[WARN] Normalization skipped for {gamma_folder_name}: "
                  f"max_extension ({max_extension:.6g}) <= ext0 ({ext0:.6g}).")
            norm_means, norm_stds = means.copy(), stds.copy()
        else:
            norm_means = (means - ext0) / denom
            norm_stds  = stds / denom

        color = color_cycle[idx % len(color_cycle)]
        label = fr"$\gamma$ = {gamma_val:.4f}"

        ax.plot(shears, norm_means, label=label, color=color, marker='o',
                markersize=4, linewidth=1.5)
        if not np.all(norm_stds == 0):
            ax.fill_between(shears, norm_means - norm_stds, norm_means + norm_stds,
                            color=color, alpha=0.25)

    # Expected curve (already normalized 0..1)
    #ax.plot(expected_x, expected_y, label="Expected", color="black",
            #linestyle="--", linewidth=2, marker='s', markersize=5)

    ax.set_title(f"Final Polymer Extension vs Shear Rate\n{lj_title}", fontsize=16)
    ax.set_xlabel("Wall shear stress [dyn/cm²]", fontsize=13)
    ax.set_ylabel("Total extension (normalized)", fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True)

    plt.tight_layout()
    output_path = os.path.join(base_dir, "extension_vs_shearstress_freeflow.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def com_with_pbc(positions, Lx=None, Ly=None, Lz=None):
    """
    positions: (T, N, D). Returns unwrapped center-of-mass trajectory (T, D).
    """
    T, N, D = positions.shape
    com = np.empty((T, D), dtype=float)
    for t in range(T):
        u = unwrap_chain_frame(positions[t], Lx=Lx, Ly=Ly, Lz=Lz)
        com[t] = u.mean(axis=0)
    return com

# ---------- Your moving average (unchanged) ----------
def moving_average(data, N):
    """Compute simple moving average over axis=0."""
    if N <= 1:
        return data
    kernel = np.ones(N) / N
    if data.ndim == 1:
        return np.convolve(data, kernel, mode='valid')
    else:
        return np.array([np.convolve(col, kernel, mode='valid') for col in data.T]).T

# ---------- Updated plotting with PBC-aware extension & new position plot ----------
import numpy as np
import os, re
import matplotlib.pyplot as plt

# ---------- PBC helpers ----------
def minimum_image_1d(dr, boxlen):
    if boxlen is None or boxlen <= 0:
        return dr
    return dr - np.round(dr / boxlen) * boxlen

def unwrap_chain_frame(pos_frame, Lx=None, Ly=None, Lz=None):
    N, D = pos_frame.shape
    u = np.empty_like(pos_frame)
    u[0] = pos_frame[0]
    for i in range(1, N):
        dr = pos_frame[i] - pos_frame[i-1]
        dx = minimum_image_1d(dr[0], Lx)
        if D >= 2:
            dy = minimum_image_1d(dr[1], Ly)
        if D == 3:
            dz = minimum_image_1d(dr[2], Lz)
        if D == 1:
            drc = np.array([dx])
        elif D == 2:
            drc = np.array([dx, dy])
        else:
            drc = np.array([dx, dy, dz])
        u[i] = u[i-1] + drc
    return u

def end_to_end_with_pbc(positions, Lx=None, Ly=None, Lz=None):
    T, N, D = positions.shape
    e2e = np.empty(T, dtype=float)
    for t in range(T):
        u = unwrap_chain_frame(positions[t], Lx=Lx, Ly=Ly, Lz=Lz)
        e2e[t] = np.linalg.norm(u[-1] - u[0])
    return e2e

def extension_time_series_with_pbc(positions, Lx=None, Ly=None, Lz=None,
                                   mode="max_pairwise", span_axis=None):
    """
    positions: (T, N, D)
    mode:
      - "end_to_end": ||u[-1]-u[0]||
      - "max_pairwise": max_{i,j} ||u[j]-u[i]||
      - "span": max(u[:,axis]) - min(u[:,axis])  (axis in {"x","y","z"} or 0/1/2)
    Returns: (T,)
    """
    T, N, D = positions.shape
    out = np.empty(T, dtype=float)

    # map span_axis
    if isinstance(span_axis, str):
        ax_map = {"x": 0, "y": 1, "z": 2}
        ax = ax_map.get(span_axis.lower(), 0)
    elif isinstance(span_axis, int):
        ax = span_axis
    else:
        ax = 0

    for t in range(T):
        u = unwrap_chain_frame(positions[t], Lx=Lx, Ly=Ly, Lz=Lz)

        if mode == "end_to_end":
            out[t] = np.linalg.norm(u[-1] - u[0])

        elif mode == "max_pairwise":
            # O(N^2) pairwise distances (fine for typical N<=O(10^3))
            diff = u[None, :, :] - u[:, None, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            out[t] = np.sqrt(np.max(d2))

        elif mode == "span":
            ax_eff = int(np.clip(ax, 0, u.shape[1]-1))
            out[t] = float(np.max(u[:, ax_eff]) - np.min(u[:, ax_eff]))

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return out

# ---------- Your moving average ----------
def moving_average(data, N):
    if N <= 1: return data
    kernel = np.ones(N) / N
    if data.ndim == 1:
        return np.convolve(data, kernel, mode='valid')
    else:
        return np.array([np.convolve(col, kernel, mode='valid') for col in data.T]).T

# ---------- Updated plotting ----------
def plot_simulation_results_free(config, base_dir, plot_type, target_shear=None,
                            avg_window=100, extension_mode="max_pairwise", span_axis=None):
    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("tab10").colors

    print(f"Plotting simulation results from directory: {base_dir}")

    variable = "shear"
    max_extension = config["physical"]["N_MONOMERS"] * 2 * config["physical"]["a"]
    TIMESTEP = config["simulation"]["TIME_STEP"]
    STEPS = config["simulation"]["STEPS"]
    N_MONOMERS = config["physical"]["N_MONOMERS"]

    # Box lengths
    geom = config.get("geometry", {})
    Lx = geom.get("BOX_Lx", None)
    Ly = geom.get("BOX_Ly", None)  # optional
    Lz = geom.get("BOX_Lz", None)  # optional

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

    if plot_type not in ["extension", "force", "position"]:
        print(f"Unsupported plot_type: {plot_type}")
        return

    dt = 1e-4 * TIMESTEP * STEPS

    # -------------------- EXTENSION (choose definition) --------------------
    if plot_type == "extension":
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, folder in enumerate(dirs):
            data_dir = os.path.join(base_dir, folder)
            m = re.match(r"shear_(\d+)", folder)
            if not m: continue
            shear_val = int(m.group(1))
            shear_label = f"{shear_val} 1/s"

            file_path = os.path.join(data_dir, "positions.npz")
            if not os.path.isfile(file_path):
                continue

            try:
                npzfile = np.load(file_path)
                pos = npzfile["positions"]  # (T, N, D)
                # ---- NEW: selectable extension definition (PBC-safe) ----
                ext_ts = extension_time_series_with_pbc(
                    pos, Lx=Lx, Ly=Ly, Lz=Lz,
                    mode=extension_mode, span_axis=span_axis
                )
                values = moving_average(ext_ts / max_extension, avg_window)
            except Exception:
                continue

            time_s = np.arange(len(values)) * dt
            color = color_cycle[idx % len(color_cycle)]
            ax.plot(time_s, values, label=shear_label, color=color, linewidth=1.5)

        title_mode = {
            "end_to_end": "End-to-end",
            "max_pairwise": "Max pairwise distance",
            "span": f"Span along {span_axis if span_axis is not None else 'x'}"
        }.get(extension_mode, extension_mode)

        ax.set_title(f"{title_mode} extension over time (PBC-corrected)\n{subtitle}", fontsize=16)
        ax.set_ylabel("Normalized extension", fontsize=13)
        ax.set_xlabel("Time [s]", fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        if ax.has_data():
            ax.legend(fontsize=10, frameon=False)
        ax.grid(True)
        plt.tight_layout()

        output_path = os.path.join(base_dir, f"extension_{extension_mode}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved extension plot to {output_path}")

    # -------------------- FORCE (unchanged math; time base fixed for MA length) --------------------
    elif plot_type == "force":
        fig, axs = plt.subplots(N_MONOMERS, 1, figsize=(10, 2 * N_MONOMERS), sharex=True)
        if N_MONOMERS == 1:
            axs = [axs]

        lastN_forces_by_monomer = {}  # shear_label -> (N_last, N_MONOMERS)

        for idx, folder in enumerate(dirs):
            data_dir = os.path.join(base_dir, folder)
            shear_val = int(re.match(r"shear_(\d+)", folder).group(1))
            shear_label = f"{shear_val} 1/s"

            file_path = os.path.join(data_dir, "drag_forces.npz")
            if not os.path.isfile(file_path):
                continue

            try:
                npzfile = np.load(file_path)
                data = npzfile["forces"]  # (T, N_MONOMERS, 3)
            except Exception:
                continue

            # moving-averaged magnitude in pN -> (T', N_MONOMERS)
            data_avg = moving_average(np.linalg.norm(data, axis=2) * 1000, avg_window)
            time_s = np.arange(len(data_avg)) * dt
            N_last = int(len(data_avg) / 4) if len(data_avg) > 0 else 0  # last quarter

            color = color_cycle[idx % len(color_cycle)]
            for m in range(N_MONOMERS):
                if N_last > 0:
                    axs[m].plot(time_s[-N_last:], data_avg[:, m][-N_last:], label=shear_label,
                                color=color, linewidth=1.2)
                axs[m].set_ylabel(f"M{m} [pN]", fontsize=9)
                axs[m].tick_params(axis='y', labelsize=8)
                axs[m].grid(True)

            if N_last > 0:
                lastN_forces_by_monomer[shear_label] = data_avg[-N_last:, :]

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

        # -------- cumulative average + std from monomer 30 back to 1 (as before) --------
        if lastN_forces_by_monomer:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            monomer_idx = np.arange(1, N_MONOMERS + 1)
            requested_start = 30
            start_monomer = min(requested_start, N_MONOMERS)

            for idx, (shear_label, forces_lastN) in enumerate(
                sorted(lastN_forces_by_monomer.items(), key=lambda x: int(x[0].split()[0]))
            ):
                color = color_cycle[idx % len(color_cycle)]
                subset_time = forces_lastN[:, :start_monomer]      # (Nt, start)
                subset_time_rev = subset_time[:, ::-1]
                cum_time_rev = np.cumsum(subset_time_rev, axis=1)
                cum_time = cum_time_rev[:, ::-1]
                cum_mean = np.mean(cum_time, axis=0)
                cum_std = np.std(cum_time, axis=0, ddof=0)

                cumulative_arr = np.full(N_MONOMERS, np.nan, dtype=float)
                cumulative_err = np.full(N_MONOMERS, np.nan, dtype=float)
                cumulative_arr[:start_monomer] = cum_mean
                cumulative_err[:start_monomer] = cum_std

                ax2.errorbar(
                    monomer_idx, cumulative_arr, yerr=cumulative_err,
                    marker='o', linewidth=1.5, elinewidth=1, capsize=3,
                    label=shear_label, color=color
                )

            ref_line = np.full(N_MONOMERS, np.nan, dtype=float)
            ref_line[:start_monomer] = 0.5 * (start_monomer - np.arange(1, start_monomer + 1) + 1)
            ax2.plot(monomer_idx, ref_line, linestyle='--', linewidth=1.2,
                     label=f"Target: 0.5 pN/step from M{start_monomer}")

            ax2.set_title(
                f"Cumulative average force vs monomer index\n(starting from monomer {start_monomer} back to 1)\n{subtitle}",
                fontsize=16
            )
            ax2.set_xlabel("Monomer index", fontsize=13)
            ax2.set_ylabel("Cumulative force [pN]", fontsize=13)
            ax2.set_xticks(monomer_idx)
            ax2.tick_params(axis='both', labelsize=11)
            if ax2.has_data():
                ax2.legend(title="Shear", fontsize=10, title_fontsize=10, frameon=False)
            ax2.grid(True)
            plt.tight_layout()

            output_path2 = os.path.join(base_dir, "force_cumulative_vs_monomer.png")
            plt.savefig(output_path2, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"Saved cumulative average force vs monomer index to {output_path2}")

    # -------------------- POSITION (new): PBC-unwrapped COM trajectory --------------------
    elif plot_type == "position":
        # One figure per shear: COM x(t), y(t) (if D>=2). Starts at zero by subtracting initial COM.
        for idx, folder in enumerate(dirs):
            data_dir = os.path.join(base_dir, folder)
            shear_val = int(re.match(r"shear_(\d+)", folder).group(1))
            shear_label = f"{shear_val} 1/s"

            file_path = os.path.join(data_dir, "positions.npz")
            if not os.path.isfile(file_path):
                continue

            try:
                npzfile = np.load(file_path)
                pos = npzfile["positions"]  # (T, N, D)
                com = com_with_pbc(pos, Lx=Lx, Ly=Ly, Lz=Lz)  # (T, D)
                com -= com[0]  # start at origin for visual clarity
                # moving average each component (keeps shape (T', D))
                com_sm = moving_average(com, avg_window)
            except Exception:
                continue

            Tprime = com_sm.shape[0]
            time_s = np.arange(Tprime) * dt
            D = com_sm.shape[1]

            # Make a clean per-shear plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_s, com_sm[:, 0], label="COM x (unwrapped)", linewidth=1.5)
            if D >= 2:
                ax.plot(time_s, com_sm[:, 1], label="COM y (unwrapped)", linewidth=1.2)
            if D == 3:
                ax.plot(time_s, com_sm[:, 2], label="COM z (unwrapped)", linewidth=1.2)

            ax.set_title(f"Polymer COM position over time (PBC-corrected)\n{shear_label}, {subtitle}",
                         fontsize=15)
            ax.set_xlabel("Time [s]", fontsize=13)
            ax.set_ylabel("Displacement [simulation units]", fontsize=13)
            ax.grid(True)
            ax.legend(fontsize=10, frameon=False)
            plt.tight_layout()

            out = os.path.join(data_dir, "position_com_time.png")
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved COM position time series to {out}")

