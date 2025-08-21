import os, re
import numpy as np
import matplotlib.pyplot as plt

# ---------- PBC helpers ----------
def min_image_1d(dr, L):
    return dr if (L is None or L <= 0) else dr - np.round(dr / L) * L

def unwrap_frame(pos, Lx=None, Ly=None, Lz=None):
    """Build a continuous chain in one frame using minimum-image bond vectors."""
    N, D = pos.shape
    u = np.empty_like(pos); u[0] = pos[0]
    for i in range(1, N):
        dr = pos[i] - pos[i-1]
        dx = min_image_1d(dr[0], Lx)
        if D >= 2: dy = min_image_1d(dr[1], Ly)
        if D == 3: dz = min_image_1d(dr[2], Lz)
        u[i] = u[i-1] + (np.array([dx]) if D==1 else
                         np.array([dx, dy]) if D==2 else
                         np.array([dx, dy, dz]))
    return u

# ---------- Metrics ----------
def max_pairwise_extension_ts_pbc(positions, Lx=None, Ly=None, Lz=None):
    """(T,N,D) -> per-frame max distance (diameter), PBC-safe via unwrapping."""
    T = positions.shape[0]
    out = np.empty(T, float)
    for t in range(T):
        u = unwrap_frame(positions[t], Lx, Ly, Lz)   # (N,D)
        diff = u[None,:,:] - u[:,None,:]            # (N,N,D)
        d2 = np.einsum('ijk,ijk->ij', diff, diff)   # (N,N)
        out[t] = np.sqrt(np.max(d2))
    return out

def tethered_max_extension_ts(positions, anchor=0):
    """(T,N,D) -> per-frame max distance from anchor bead (no PBC assumed)."""
    T = positions.shape[0]
    out = np.empty(T, float)
    for t in range(T):
        p = positions[t]
        d = np.linalg.norm(p - p[anchor], axis=1)
        out[t] = np.max(d)
    return out

def com_ts(positions):
    """(T,N,D) -> COM(t), shape (T,D)."""
    return positions.mean(axis=1)

# ---------- Forces-only smoothing ----------
def moving_average_centered(x, N):
    """Centered uniform MA, preserves length (pads by reflection)."""
    x = np.asarray(x)
    if N <= 1:
        return x.copy()
    k = np.ones(N, dtype=float) / N
    pad_left = (N - 1) // 2
    pad_right = pad_left + (N % 2 == 0)
    if x.ndim == 1:
        xp = np.pad(x, (pad_left, pad_right), mode='reflect')
        return np.convolve(xp, k, mode='valid')
    pad = ((pad_left, pad_right),) + ((0,0),) * (x.ndim - 1)
    xp = np.pad(x, pad, mode='reflect')
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode='valid'), 0, xp)

# ---------- Validation plot (dynamic N + optional Box_Lx override) ----------
def plot_validation_results(config, base_dir, validation_type="free", Box_Lx=None):
    """
    Aggregate extension per shear and plot normalized vs wall shear stress.
    Normalization: (ext - ext0) / (max_extension - ext0),
    with ext0 from shear==0 (or smallest shear).
    Uses dynamic N from the first positions.npz found per gamma (max_extension = N*2*a).

    Box_Lx: optional override for box length along x used in PBC handling for the 'free' metric.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("tab10").colors

    a = config["physical"]["a"]
    dyn_visc_SI = config["physical"]["VISC_dyn"] / 10

    # box lengths (Ly falls back to effective Lx if missing)
    geom = config.get("geometry", {})
    Lx_eff = Box_Lx if (Box_Lx is not None) else geom.get("BOX_Lx", None)
    Ly_eff = geom.get("BOX_Ly", None)
    if Ly_eff is None:
        Ly_eff = Lx_eff
    Lz = geom.get("BOX_Lz", None)

    # title bits
    lj_match = re.search(r"LJ_(\d{5})", base_dir)
    lj_title = (f"Lennard Jones potential = {int(lj_match.group(1))/1e2:.2f} × 4kT"
                if lj_match else "Lennard Jones potential = unknown")

    gamma_pattern = re.compile(r"gamma_(\d{5})")
    gamma_dirs = sorted([g for g in os.listdir(base_dir)
                         if os.path.isdir(os.path.join(base_dir, g)) and gamma_pattern.match(g)])

    fig, ax = plt.subplots(figsize=(10,6))
    for gi, gname in enumerate(gamma_dirs):
        gamma_val = int(gamma_pattern.match(gname).group(1)) / 1e4
        gpath = os.path.join(base_dir, gname)

        # dynamic max_extension per gamma
        max_extension_gamma = None

        shears, means, stds = [], [], []
        for sd in sorted([d for d in os.listdir(gpath) if re.match(r"shear_\d+$", d)],
                         key=lambda s: int(s.split('_')[1])):
            shear_label = int(sd.split('_')[1])
            sdir = os.path.join(gpath, sd)
            runs = [d for d in os.listdir(sdir) if re.match(r"run_\d+$", d)] or [""]

            finals = []
            for r in runs:
                ddir = os.path.join(sdir, r) if r else sdir
                f = os.path.join(ddir, "positions.npz")
                if not os.path.isfile(f):
                    continue
                try:
                    pos = np.load(f)["positions"]  # (T,N,D)
                    if max_extension_gamma is None:
                        N = pos.shape[1]
                        max_extension_gamma = N * 2 * a
                    if str(validation_type).lower() == "tethered":
                        finals.append(tethered_max_extension_ts(pos)[-1])
                    else:
                        finals.append(max_pairwise_extension_ts_pbc(pos, Lx_eff, Ly_eff, Lz)[-1])
                except Exception as e:
                    print(f"Skip {f}: {e}")

            if not finals:
                continue
            shears.append(shear_label * dyn_visc_SI * 10)     # -> dyn/cm^2
            means.append(np.mean(finals)); stds.append(np.std(finals))

        if not shears:
            continue
        shears, means, stds = map(np.array, (shears, means, stds))
        idx = np.argsort(shears); shears, means, stds = shears[idx], means[idx], stds[idx]

        if max_extension_gamma is None:
            print(f"[WARN] No positions.npz in {gpath}; cannot normalize.")
            continue

        ext0 = means[shears==0][0] if np.any(shears==0) else means[0]
        denom = max_extension_gamma - ext0
        if denom <= 0:
            norm_m, norm_s = means, stds
            print(f"[WARN] skip normalization for {gname}: denom<=0")
        else:
            norm_m, norm_s = (means - ext0)/denom, stds/denom

        c = color_cycle[gi % len(color_cycle)]
        ax.plot(shears, norm_m, marker='o', lw=1.5, color=c, label=fr"$\gamma$={gamma_val:.4f}")
        if not np.all(norm_s == 0):
            ax.fill_between(shears, norm_m - norm_s, norm_m + norm_s, color=c, alpha=0.25)

    # Expected overlays
    if str(validation_type).lower() == "tethered":
        expected_x = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200], dtype=float)
        expected_y = np.array([0.00, 0.35, 0.55, 0.63, 0.70, 0.75, 0.81, 0.85, 0.90, 0.92, 0.94, 0.96, 0.96], dtype=float)
        expected_std = np.array([0.1, 0.2, 0.2, 0.25, 0.3, 0.35, 0.35, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=float)
        ax.plot(expected_x, expected_y, linestyle='--', color='black', linewidth=2,
                marker='s', markersize=5, label="Expected (tethered)")
        ax.fill_between(expected_x, expected_y - expected_std, expected_y + expected_std,
                        color='black', alpha=0.12, linewidth=0)

    if str(validation_type).lower() == "free":
        expected_x = np.array([1,10,100,400,1100,5000,10000], dtype=float) * dyn_visc_SI * 10
        expected_y = np.array([0,0,0,0,0,0.75,1], dtype=float)
        expected_std = np.array([0.1,0.1,0.1,0.1,0.1,0.2,0.2], dtype=float)
        ax.plot(expected_x, expected_y, linestyle='--', color='black', linewidth=2,
                marker='s', markersize=5, label="Expected (free)")
        ax.fill_between(expected_x, expected_y - expected_std, expected_y + expected_std,
                        color='black', alpha=0.12, linewidth=0)

    ax.set_title(f"Final Polymer Extension vs Shear Rate\n{lj_title}", fontsize=16)
    ax.set_xlabel("Wall shear stress [dyn/cm²]"); ax.set_ylabel("Total extension (normalized)")
    ax.legend(fontsize=10, frameon=False); ax.grid(True); plt.tight_layout()
    out = os.path.join(base_dir, "extension_vs_shearstress_test.png")
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Plot saved to {out}")

# ---------- Time series (dynamic N + optional Box_Lx override) ----------
def plot_time_series(config, base_dir, plot_type, target_shear=None, avg_window=50, validation_type="free", Box_Lx=None):
    """
    plot_type = 'extension' (normalized extension over time; no smoothing),
                'force'     (per-monomer + cumulative summary; smoothed)
    validation_type: "free" -> max pairwise (PBC-safe); "tethered" -> anchor (farthest);
                     "recoiling" -> anchor (farthest), normalized to t=0
    Uses dynamic N from each file (no N_MONOMERS from config).

    Box_Lx: optional override for x-box length used in PBC handling (free case).
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("tab10").colors

    a = config["physical"]["a"]
    TIMESTEP = config["simulation"]["TIME_STEP"]; STEPS = config["simulation"]["STEPS"]
    dt = 1e-4 * TIMESTEP * STEPS

    # box lengths for PBC (free case)
    geom = config.get("geometry", {})
    Lx_eff = Box_Lx if (Box_Lx is not None) else geom.get("BOX_Lx", None)
    Ly_eff = geom.get("BOX_Ly", None)
    if Ly_eff is None:
        Ly_eff = Lx_eff
    Lz = geom.get("BOX_Lz", None)

    # subtitle
    lj_match = re.search(r"LJ_(\d{5})", base_dir)
    gamma_match = re.search(r"gamma_(\d{5})", base_dir)
    lj_text = f"Lennard Jones potential = {int(lj_match.group(1))/1e2:.2f} × 4kT" if lj_match else ""
    gamma_text = f"$\\gamma$ = {int(gamma_match.group(1))/1e4:.4f}" if gamma_match else ""
    subtitle = ", ".join([t for t in (gamma_text, lj_text) if t])

    shear_dirs_all = [d for d in os.listdir(base_dir) if re.match(r"shear_\d+$", d)]
    shear_dirs_all.sort(key=lambda s: int(s.split('_')[1]))

    # ---------- Recoiling (tethered metric, no PBC, normalized to t=0) ----------
    if plot_type == "extension" and str(validation_type).lower() == "recoiling":
        fig, ax = plt.subplots(figsize=(12,6))
        gamma_dirs = [d for d in os.listdir(base_dir)
                      if d.startswith("gamma_") and os.path.isdir(os.path.join(base_dir, d))]
        gamma_dirs.sort(key=lambda g: int(re.match(r"gamma_(\d+)", g).group(1)) if re.match(r"gamma_(\d+)", g) else 0)

        for i, gdir in enumerate(gamma_dirs):
            gmatch = re.match(r"gamma_(\d+)", gdir)
            if not gmatch:
                continue
            gamma_val = int(gmatch.group(1)) / 1e4
            f = os.path.join(base_dir, gdir, "positions.npz")
            if not os.path.isfile(f):
                print(f"Missing positions file in {f}")
                continue
            try:
                pos = np.load(f)["positions"]              # (T, N, D)
                y = tethered_max_extension_ts(pos, anchor=0)   # NO PBC
                init = y[0]
                if init <= 0:
                    print(f"Initial extension <= 0 in {f}; skipping normalization.")
                    continue
                y = y / init
                t = np.arange(len(y)) * dt
                ax.plot(t, y, label=f"γ={gamma_val:.4f}",
                        color=plt.get_cmap('tab10').colors[i % 10], lw=1.5)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        # expected recoiling curve (time vs normalized extension)
        expected_x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        expected_y = [1, 0.4, 0.2, 0.1, 0.08, 0.08, 0.07, 0.07]
        ax.plot(expected_x, expected_y, 'k--', label="Expected", linewidth=2)

        ax.set_title("Polymer Extension Over Time (tethered metric, normalized to t=0; no PBC)", fontsize=16)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Extension / initial extension")
        ax.legend(fontsize='small'); ax.grid(True); plt.tight_layout()
        out = os.path.join(base_dir, "recoiling_extension_over_time.png")
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
        print(f"Saved {out}")
        return

    # ---------- Extension (always plot all shears; dynamic N per shear) ----------
    if plot_type == "extension":
        fig, ax = plt.subplots(figsize=(10,6))
        for i, sd in enumerate(shear_dirs_all):
            f = os.path.join(base_dir, sd, "positions.npz")
            if not os.path.isfile(f):
                continue
            try:
                pos = np.load(f)["positions"]  # (T,N,D)
                N_here = pos.shape[1]
                max_extension_shear = N_here * 2 * a

                if str(validation_type).lower() == "tethered":
                    y = tethered_max_extension_ts(pos, anchor=0)  # no PBC
                else:
                    y = max_pairwise_extension_ts_pbc(pos, Lx_eff, Ly_eff, Lz)  # free (PBC-safe)

                y = y / max_extension_shear
                t = np.arange(len(y)) * dt
                ax.plot(t, y, label=f"{sd.split('_')[1]} 1/s", color=color_cycle[i % 10], lw=1.5)
            except Exception as e:
                print(f"Skip {f}: {e}")
        ax.set_title(f"Extension over time\n{subtitle}", fontsize=16)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Normalized extension")
        if ax.has_data(): ax.legend(frameon=False, fontsize=10)
        ax.grid(True); plt.tight_layout()
        out = os.path.join(base_dir, "extension_time_test.png")
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")
        return

    # ---------- Force (dynamic N from first forces file among selected) ----------
    if plot_type == "force":
        # Select shears (respect target_shear if provided)
        if target_shear is not None:
            target_dir = f"shear_{target_shear}"
            shear_dirs_force = [target_dir] if target_dir in shear_dirs_all else []
            if not shear_dirs_force:
                print(f"No data for shear = {target_shear}")
                return
        else:
            shear_dirs_force = shear_dirs_all

        # detect N from first available forces file
        N_detect = None
        for sd in shear_dirs_force:
            candidate = os.path.join(base_dir, sd, "drag_forces.npz")
            if os.path.isfile(candidate):
                try:
                    Fsample = np.load(candidate)["forces"]  # (T, N, 3)
                    N_detect = Fsample.shape[1]
                    break
                except Exception:
                    continue
        if N_detect is None:
            print("No drag_forces.npz found in selected shears; cannot plot forces.")
            return

        fig, axs = plt.subplots(N_detect, 1, figsize=(10, 2*N_detect), sharex=True)
        if N_detect == 1: axs = [axs]
        lastN = {}

        for i, sd in enumerate(shear_dirs_force):
            f = os.path.join(base_dir, sd, "drag_forces.npz")
            if not os.path.isfile(f): 
                continue
            try:
                F = np.load(f)["forces"]                           # (T, N, 3)
                N_here = F.shape[1]
                N_use = min(N_detect, N_here)                      # trim if needed
                Fmag = np.linalg.norm(F[:, :N_use, :], axis=2) * 1000   # pN
                Fm = moving_average_centered(Fmag, avg_window)          # centered, same length
                t = np.arange(len(Fm)) * dt
                nL = int(len(Fm)/4) if len(Fm)>0 else 0                 # last quarter
                c = color_cycle[i % 10]
                for m in range(N_use):
                    if nL>0:
                        axs[m].plot(t[-nL:], Fm[:,m][-nL:], lw=1.2, color=c, label=f"{sd.split('_')[1]} 1/s")
                    axs[m].set_ylabel(f"M{m} [pN]", fontsize=9); axs[m].grid(True)
                if nL>0: lastN[sd] = Fm[-nL:,:]
            except Exception as e:
                print(f"Skip {f}: {e}")

        for ax in axs:
            if ax.has_data(): ax.legend(fontsize=8, frameon=False)
        axs[-1].set_xlabel("Time [s]"); plt.suptitle(f"Force per monomer over time\n{subtitle}", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.97])
        out = os.path.join(base_dir, "force_per_monomer_test.png")
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")

        if lastN:
            fig, ax = plt.subplots(figsize=(10,6))
            idx = np.arange(1, N_detect+1); start = min(30, N_detect)
            for i, (sd, block) in enumerate(sorted(lastN.items(), key=lambda x: int(x[0].split('_')[1]))):
                c = color_cycle[i % 10]
                sub = block[:, :start][:, ::-1]
                cum = np.cumsum(sub, axis=1)[:, ::-1]
                mu, sig = cum.mean(0), cum.std(0)
                y = np.full(N_detect, np.nan); e = np.full(N_detect, np.nan)
                y[:start], e[:start] = mu, sig
                ax.errorbar(idx, y, yerr=e, marker='o', lw=1.5, elinewidth=1, capsize=3,
                            color=c, label=f"{sd.split('_')[1]} 1/s")
            ref = np.full(N_detect, np.nan); ref[:start] = 0.5 * (start - np.arange(1,start+1) + 1)
            ax.plot(idx, ref, '--', lw=1.2, label=f"Target: 0.5 pN/step from M{start}")
            ax.set_title(f"Cumulative force vs monomer index\n{subtitle}", fontsize=16)
            ax.set_xlabel("Monomer index"); ax.set_ylabel("Cumulative force [pN]")
            ax.set_xticks(idx); ax.grid(True); ax.legend(frameon=False, fontsize=10)
            plt.tight_layout()
            out2 = os.path.join(base_dir, "force_cumulative_vs_monomer_test.png")
            plt.savefig(out2, dpi=300, bbox_inches='tight'); plt.close(fig); print(f"Saved {out2}")
        return

    print(f"Unsupported plot_type: {plot_type}")
