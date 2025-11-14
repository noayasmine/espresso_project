import os, re
import numpy as np
import csv
import matplotlib.pyplot as plt

# ---------- PBC helpers ----------
def min_image_1d(dr, L):
    """Minimum-image in 1D."""
    return dr if (L is None or L <= 0) else dr - np.round(dr / L) * L

def unwrap_frame(pos, Lx=None, Ly=None, Lz=None):
    """Unwrap a single frame (N,D) by building it from minimum-image bond vectors."""
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
    """(T,N,D) -> per-frame max pairwise distance (diameter), PBC-safe via unwrapping each frame."""
    T = positions.shape[0]
    out = np.empty(T, float)
    for t in range(T):
        u = unwrap_frame(positions[t], Lx, Ly, Lz)      # (N,D)
        diff = u[None,:,:] - u[:,None,:]                # (N,N,D)
        d2 = np.einsum('ijk,ijk->ij', diff, diff)       # (N,N)
        out[t] = np.sqrt(np.max(d2))
    return out

def end_to_end_extension_ts_pbc(positions, Lx=None, Ly=None, Lz=None):
    """(T,N,D) -> per-frame end-to-end distance ||u[-1]-u[0]||, PBC-safe via unwrapping each frame."""
    T = positions.shape[0]
    out = np.empty(T, float)
    for t in range(T):
        u = unwrap_frame(positions[t], Lx, Ly, Lz)
        out[t] = np.linalg.norm(u[-1] - u[0])
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

# ---------- Validation plot (dynamic N + optional Box_Lx override + free_metric) ----------
def plot_validation_results(config, base_dir, validation_type="free",
                            Box_Lx=None, free_metric="end_to_end",
                            target_gamma=None, fine_model=False, a=0.05):
    """
    Aggregate extension per shear and plot normalized vs wall shear stress.
    Normalization per gamma: (E_final - E0) / (Lmax - E0), with E0 at shear==0 (or smallest).
    Lmax = N * 2 * a using dynamic N from the first positions.npz per gamma.

    validation_type: "free" (uses PBC) or "tethered" (anchor→farthest, no PBC).
    free_metric: "max_pairwise" (default) or "end_to_end".
    Box_Lx: optional override for x-box length in PBC handling (free case).
    target_gamma: optional; filter to a single gamma. Accepts e.g. 0.0050 or "gamma_00050".
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # color_cycle = plt.get_cmap("tab20").colors
    color_cycle = plt.get_cmap("Dark2").colors

    dyn_visc_SI = config["physical"]["VISC_dyn"] / 10

    # box lengths (Ly falls back to effective Lx if missing)
    geom = config.get("geometry", {})
    Lx_eff = Box_Lx if (Box_Lx is not None) else geom.get("BOX_Lx", None)
    Ly_eff = geom.get("BOX_Ly", None) if geom.get("BOX_Ly", None) is not None else Lx_eff
    Lz = geom.get("BOX_Lz", None)

    # title bits
    lj_match = re.search(r"LJ_(\d{5})", base_dir)
    lj_title = (f"Lennard Jones potential = {int(lj_match.group(1))/1e2:.2f} × 4kT"
                if lj_match else "Lennard Jones potential = unknown")

    gamma_pattern = re.compile(r"gamma_(\d{5})")
    gamma_dirs_all = sorted([g for g in os.listdir(base_dir)
                             if os.path.isdir(os.path.join(base_dir, g)) and gamma_pattern.match(g)])

    # ---- Optional filter to a single gamma ----
    gamma_dirs = gamma_dirs_all
    if target_gamma is not None:
        selected = []
        if isinstance(target_gamma, str) and target_gamma.startswith("gamma_"):
            if target_gamma in gamma_dirs_all:
                selected = [target_gamma]
        else:
            try:
                gval = float(target_gamma)
                code = int(round(gval * 1e4))
                folder = f"gamma_{code:05d}"
                if folder in gamma_dirs_all:
                    selected = [folder]
                else:
                    # tolerant match if naming differs slightly
                    tol = 5e-6
                    for g in gamma_dirs_all:
                        gv = int(gamma_pattern.match(g).group(1)) / 1e4
                        if abs(gv - gval) <= tol:
                            selected.append(g)
            except Exception:
                pass
        if selected:
            gamma_dirs = sorted(selected)
        else:
            print(f"[INFO] target_gamma={target_gamma} not found; plotting all gammas.")

    fig, ax = plt.subplots(figsize=(6,4))
    for gi, gname in enumerate(gamma_dirs):
        gamma_val = int(gamma_pattern.match(gname).group(1)) / 1e4
        gpath = os.path.join(base_dir, gname)

        Lmax_gamma = None
        shears, means, stds = [], [], []

        shear_dirs = [d for d in os.listdir(gpath) if re.match(r"shear_\d+$", d)]
        shear_dirs.sort(key=lambda s: int(s.split('_')[1]))

        for sd in shear_dirs:
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
                    print(f"N monomers = {pos.shape[1]}")
                    # if Lmax_gamma is None:
                    #     N = pos.shape[1]
                    #     Lmax_gamma = N * 2 * a

                    if str(validation_type).lower() == "tethered":
                        finals.append(tethered_max_extension_ts(pos)[-1])
                    else:
                        if free_metric == "end_to_end":
                            finals.append(end_to_end_extension_ts_pbc(pos, Lx_eff, Ly_eff, Lz)[-1])
                        else:  # "max_pairwise"
                            finals.append(max_pairwise_extension_ts_pbc(pos, Lx_eff, Ly_eff, Lz)[-1])
                except Exception as e:
                    print(f"Skip {f}: {e}")

            if not finals:
                continue

            # ADD THIS right after `if not finals: continue`
            Lmax_gamma = np.max(finals) if Lmax_gamma is None else max(Lmax_gamma, np.max(finals))

            shears.append(shear_label * dyn_visc_SI * 10)     # -> dyn/cm^2
            means.append(np.mean(finals)); stds.append(np.std(finals))

        if not shears or Lmax_gamma is None:
            continue
        shears, means, stds = map(np.array, (shears, means, stds))
        idx = np.argsort(shears); shears, means, stds = shears[idx], means[idx], stds[idx]

        E0 = means[shears==0][0] if np.any(shears==0) else means[0]
        denom = Lmax_gamma - E0
        if denom <= 0:
            norm_m, norm_s = means, stds
            print(f"[WARN] skip normalization for {gname}: denom<=0")
        else:
            norm_m, norm_s = (means - E0)/denom, stds/denom

        c = color_cycle[gi % len(color_cycle)]
        label = fr"$\gamma$={gamma_val:.4f}"
        if str(validation_type).lower() == "free":
            label += f" ({'end-to-end' if free_metric=='end_to_end' else 'max-pairwise'})"
        ax.plot(shears, norm_m, marker='o', lw=1.5, color=c, label=label)

        if not np.all(norm_s == 0):
            ax.fill_between(shears, norm_m - norm_s, norm_m + norm_s, color=c, alpha=0.25)

    # Expected overlays
    if str(validation_type).lower() == "tethered":
        expected_x = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1280], dtype=float)
        expected_y = np.array([0.00, 0.35, 0.55, 0.63, 0.70, 0.75, 0.81, 0.85, 0.90, 0.92, 0.94, 0.96, 0.96, 0.96], dtype=float)
        expected_std = np.array([0.1, 0.2, 0.2, 0.25, 0.3, 0.35, 0.35, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=float)
        ax.plot(expected_x, expected_y, linestyle='--', color='black', linewidth=2,
                marker='s', markersize=5, label="Expected")
        ax.fill_between(expected_x, expected_y - expected_std, expected_y + expected_std,
                        color='black', alpha=0.12, linewidth=0)

    if str(validation_type).lower() == "free":
        expected_x = np.array([1,10,100,400,1100,5000,10000], dtype=float) * dyn_visc_SI * 10
        expected_y = np.array([0,0,0,0,0,0.75,1], dtype=float)
        expected_std = np.array([0.1,0.1,0.1,0.1,0.1,0.2,0.2], dtype=float)
        ax.plot(expected_x, expected_y, linestyle='--', color='black', linewidth=2,
                marker='s', markersize=5, label="Schneider et al.")
        ax.fill_between(expected_x, expected_y - expected_std, expected_y + expected_std,
                        color='black', alpha=0.12, linewidth=0)

    title_suffix = ""
    # if str(validation_type).lower() == "free":
    #     title_suffix = f" (free metric: {'end-to-end' if free_metric=='end_to_end' else 'max-pairwise'})"
    # ax.set_title(f"Final Polymer Extension vs Shear Rate{title_suffix}\n{lj_title}", fontsize=25)
    if str(validation_type).lower() == "free":
        ax.set_xlabel("Wall shear stress [dyn/cm²]",fontsize=15)
        ax.set_ylabel(r"$\hat{E}^{final}_{free}$", fontsize=15)
        ax.set_xscale("log")
        ax.set_xlim(0,2000)
    else:
        ax.set_xlabel("Wall shear stress [dyn/cm²]",fontsize=15)
        ax.set_ylabel(r"$\hat{E}^{final}_{tethered}$", fontsize=15)
    ax.legend(fontsize=10, frameon=False); ax.grid(False); plt.tight_layout()
    out = os.path.join(base_dir, "extension_vs_shearstress_test.png")
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Plot saved to {out}")


# ---------- Time series (dynamic N + optional Box_Lx override + free_metric) ----------
def plot_time_series(config, base_dir, plot_type, target_shear=None, avg_window=1,
                     validation_type="free", Box_Lx=None, free_metric="end_to_end", time_step=None, k_fene=10, 
                     monomer_index=None, fine_model=False, a=0.03):
    """
    plot_type = 'extension' (normalized extension over time),
                'force'     (per-monomer + cumulative summary; smoothed)
    validation_type:
        - "free":     PBC-safe (choose metric via free_metric), normalize (E(t)-E(0))/(Lmax - E(0))
        - "tethered": anchor→farthest, normalize E(t)/Lmax
        - "recoiling": anchor→farthest, normalize E(t)/E(0) (no PBC)
    free_metric: "max_pairwise" (default) or "end_to_end"
    Box_Lx: optional override for x-box length in PBC handling (free case).
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("Dark2").colors

    TIMESTEP = config["simulation"]["TIME_STEP"]; STEPS = config["simulation"]["STEPS"]
    if time_step != None:
        TIMESTEP = time_step

    dt = 1e-4 * TIMESTEP * STEPS

    # box lengths for PBC (free case)
    geom = config.get("geometry", {})
    Lx_eff = Box_Lx if (Box_Lx is not None) else geom.get("BOX_Lx", None)
    Ly_eff = geom.get("BOX_Ly", None) if geom.get("BOX_Ly", None) is not None else Lx_eff
    Lz = geom.get("BOX_Lz", None)

    # subtitle
    lj_match = re.search(r"LJ_(\d{5})", base_dir)
    gamma_match = re.search(r"gamma_(\d{5})", base_dir)
    lj_text = f"Lennard Jones potential = {int(lj_match.group(1))/1e2:.2f} × 4kT" if lj_match else ""
    gamma_text = f"$\\gamma$ = {int(gamma_match.group(1))/1e4:.4f}" if gamma_match else ""
    subtitle = ", ".join([t for t in (gamma_text, lj_text) if t])

    shear_dirs_all = [d for d in os.listdir(base_dir) if re.match(r"shear_\d+$", d)]
    shear_dirs_all.sort(key=lambda s: int(s.split('_')[1]))

    # ----- Recoiling (tethered metric, normalized to t=0; no PBC) -----
    if plot_type == "extension" and str(validation_type).lower() == "recoiling":
        fig, ax = plt.subplots(figsize=(6,4))
        gamma_dirs = [d for d in os.listdir(base_dir)
                      if d.startswith("gamma_") and os.path.isdir(os.path.join(base_dir, d))]
        gamma_dirs.sort(key=lambda g: int(re.match(r"gamma_(\d+)", g).group(1)) if re.match(r"gamma_(\d+)", g) else 0)

        for i, gdir in enumerate(gamma_dirs):
            m = re.match(r"gamma_(\d+)", gdir)
            if not m: continue
            gamma_val = int(m.group(1)) / 1e4
            f = os.path.join(base_dir, gdir, "positions.npz")
            if not os.path.isfile(f):
                print(f"Missing positions file in {f}"); continue
            try:
                pos = np.load(f)["positions"]          # (T, N, D)
                E = tethered_max_extension_ts(pos, anchor=0)
                E0 = E[0]
                if E0 <= 0:
                    print(f"Initial extension <= 0 in {f}; skipping normalization.")
                    continue
                y = E / E0
                t = np.arange(len(y)) * dt
                ax.plot(t, y, label=f"γ={gamma_val:.4f}",
                        color=plt.get_cmap('tab10').colors[i % 20], lw=1.5)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        expected_x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        expected_y = [1, 0.4, 0.2, 0.1, 0.08, 0.08, 0.07, 0.07]
        ax.plot(expected_x, expected_y, 'k--', label="Expected", linewidth=2)

        ax.set_title("Polymer Extension Over Time (tethered metric, normalized to t=0)", fontsize=16)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Extension / initial extension")
        ax.legend(fontsize='small'); ax.grid(False); plt.tight_layout()
        out = os.path.join(base_dir, "recoiling_extension_over_time.png")
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
        print(f"Saved {out}")
        return

    # ----- Extension (plot all shears; dynamic N per shear) -----
    if plot_type == "extension":
        color_cycle = plt.get_cmap("tab20").colors
        fig, ax = plt.subplots(figsize=(6,4))
        for i, sd in enumerate(shear_dirs_all):
            f = os.path.join(base_dir, sd, "positions.npz")
            if not os.path.isfile(f): continue
            try:
                pos = np.load(f)["positions"]  # (T,N,D)
                N_here = pos.shape[1]
                print(f"N monomers = {N_here}")
                Lmax = N_here * 2 * a
                if fine_model:
                    Lmax = 3

                vtype = str(validation_type).lower()
                # if vtype == "tethered":
                #     E = tethered_max_extension_ts(pos, anchor=0)      # tethered metric (no PBC), normalize E/Lmax
                #     y = E / Lmax
                if vtype == "tethered":
                    E = tethered_max_extension_ts(pos, anchor=0)
                    E0 = E[0]
                    den = Lmax - E0
                    if den <= 0:
                        den = np.finfo(float).eps
                    y = (E - E0) / den

                else:
                    # FREE: choose metric, PBC-safe; normalize as (E - E0)/(Lmax - E0)
                    if free_metric == "end_to_end":
                        E = end_to_end_extension_ts_pbc(pos, Lx_eff, Ly_eff, Lz)
                    else:
                        E = max_pairwise_extension_ts_pbc(pos, Lx_eff, Ly_eff, Lz)
                    E0 = E[0]
                    den = Lmax - E0
                    if den <= 0: den = np.finfo(float).eps
                    y = (E - E0) / den

                t = np.arange(len(y)) * dt
                shear_int = int(sd.split('_')[1])

                #lbl = r"$\dot \gamma=$" + f" {shear_int} " + r"$s^{-1}$"
                shear_stress_int = round(int(shear_int * 0.11),-1)
                lbl = r"$\tau=$" + f" {shear_stress_int} " + r"dyn cm$^{-1}$"
                if vtype == "free":
                    lbl += f" ({'end-to-end' if free_metric=='end_to_end' else 'max-pairwise'})"
                ax.plot(t, y, label=lbl, color=color_cycle[i % 20], lw=1.5)
            except Exception as e:
                print(f"Skip {f}: {e}")
        # ax.set_title(f"Extension over time\n{subtitle}", fontsize=16)
        # ax.set_title(r"$k_{FENE}$=" + f"{k_fene}" + "mN/m", fontsize=16)
        ax.set_xlabel("Time [s]", fontsize=15); ax.set_ylabel(r"$\hat{E}_{teth}$", fontsize=15)
        # ax.set_ylim(-0.05,1.15)
        #ax.set_ylim(-0.05,2.8)
        #ax.set_xlim(0,0.4)
        ax.axhline(y=1, color='k', linestyle='--')
        if ax.has_data(): ax.legend(frameon=True, fontsize=10, ncol=2)
        ax.grid(False); plt.tight_layout()
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        out = os.path.join(base_dir, "extension_time.png")
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")
        return

    # ----- Force (unchanged) -----
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

        # If a specific monomer is requested, validate index and do single-axis plot
        if monomer_index is not None:
            if not (0 <= int(monomer_index) < N_detect):
                print(f"Requested monomer_index={monomer_index} is out of range [0, {N_detect-1}]")
                return

            fig, ax = plt.subplots(figsize=(6, 4))
            for i, sd in enumerate(shear_dirs_force):
                f = os.path.join(base_dir, sd, "drag_forces.npz")
                if not os.path.isfile(f):
                    continue
                try:
                    F = np.load(f)["forces"]                    # (T, N, 3)
                    if monomer_index >= F.shape[1]:
                        print(f"{sd}: monomer_index {monomer_index} >= N={F.shape[1]} (skipping)")
                        continue
                    Fmag = np.linalg.norm(F[:, monomer_index, :], axis=1) * 1000  # pN
                    Fm = moving_average_centered(Fmag, avg_window)                # (T,)
                    t = np.arange(len(Fm)) * dt

                    # nL = int(len(Fm)/4) if len(Fm) > 0 else 0          # last quarter (your comment)
                    nL = int(len(Fm))                                    # currently full length
                    if nL > 0:
                        ax.plot(t[-nL:], Fm[-nL:], lw=1.5,
                                # color=color_cycle[i % 20],
                                color="grey",
                                label=f"Monomer index={monomer_index}")
                                
                except Exception as e:
                    print(f"Skip {f}: {e}")

            ax.set_xlabel("Time [s]", fontsize=15)
            ax.set_ylabel(f"Drag force [pN]", fontsize=15)
            # ax.set_title(f"Force vs time (monomer {monomer_index})\n{subtitle}", fontsize=14)
            # if ax.has_data(): ax.legend(frameon=False, fontsize=10, ncol=2)
            ax.set_xlim(0.15,0.4)
            ax.set_ylim(0.50,0.53)
            # ax.set_xlim(0.3,0.4)
            ax.grid(False); plt.tight_layout()
            out = os.path.join(base_dir, f"force_monomer_{monomer_index}_vs_time.png")
            plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
            print(f"Saved {out}")
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
                N_use = min(N_detect, N_here)
                Fmag = np.linalg.norm(F[:, :N_use, :], axis=2) * 1000   # pN
                Fm = moving_average_centered(Fmag, avg_window)          # centered, same length
                t = np.arange(len(Fm)) * dt

                # Take the last quarter of temporal data (i.e. when the data has equilibrated)
                nL = int(len(Fm)/4) if len(Fm)>0 else 0
                # nL = int(len(Fm))
                print(nL)
                c = color_cycle[i % 8]
                for m in range(N_use):
                    if nL>0:
                        axs[m].plot(t[-nL:], Fm[:,m][-nL:], lw=1.2, color="grey", label=f"{sd.split('_')[1]} 1/s")
                    axs[m].set_ylabel(f"M{m} [pN]", fontsize=9); axs[m].grid(False)
                if nL>0: lastN[sd] = Fm[-nL:,:]
            except Exception as e:
                print(f"Skip {f}: {e}")

        for ax in axs:
            if ax.has_data(): ax.legend(fontsize=8, frameon=False)
        axs[-1].set_xlabel("Time [s]"); plt.suptitle(f"Force per monomer over time\n{subtitle}", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.97])
        out = os.path.join(base_dir, "force_per_monomer.png")
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")

        if lastN:
            fig, ax = plt.subplots(figsize=(6,4))
            idx = np.arange(0, N_detect); start = min(30, N_detect)
            for i, (sd, block) in enumerate(sorted(lastN.items(), key=lambda x: int(x[0].split('_')[1]))):
                c = color_cycle[i % 8]
                sub = block[:, :start][:, ::-1]
                cum = np.cumsum(sub, axis=1)[:, ::-1]
                mu, sig = cum.mean(0), cum.std(0)
                y = np.full(N_detect, np.nan); e = np.full(N_detect, np.nan)
                y[:start], e[:start] = mu, sig
                # ax.errorbar(idx, y, yerr=e, marker='o', lw=1.5, elinewidth=1, capsize=3,
                #             color=c, label=f"{sd.split('_')[1]} 1/s")
                ax.errorbar(idx, y, yerr=e, marker='o', lw=1.5, elinewidth=1, capsize=3,
                             color="black", markersize=4, label=r"1280 $dyn$ $cm^{-2}$")
            # ref = np.full(N_detect, np.nan); ref[:start] = 0.5 * (start - np.arange(1,start+1) + 1)
            ref = np.full(N_detect, np.nan)
            # steps from the right: start, start-1, ..., 1 (for the first `start` monomers)
            steps_from_right = (start - np.arange(1, start+1) + 1)  # length = start
            ref[:start] = 0.5 * steps_from_right                     # 0.5 pN per monomer cumulatively

            # cumulative std: 0.1 pN per monomer => grows linearly with steps
            sigma = np.full(N_detect, np.nan)
            sigma[:start] = 0.1 * steps_from_right


            ax.plot(idx, ref, '--', lw=1.2, color="grey", label=r"Fu et al. $0.5 \pm 0.1$ pN/monomer")
            mask = ~np.isnan(ref)
            ax.fill_between(idx[mask], ref[mask] - sigma[mask], ref[mask] + sigma[mask], color='grey', alpha=0.15)

            # ax.fill_between(idx, ref - 0.1, ref + 0.1, color='black', alpha=0.15)
            # ax.set_title(f"Cumulative force vs monomer index\n{subtitle}", fontsize=16)
            ax.set_xlabel("Monomer index", fontsize=15); ax.set_ylabel("Tensile force [pN]", fontsize=15)
            #ax.set_xticks(idx)
            ax.set_xticks(np.arange(0, N_detect+1, 5))
            ax.set_xlim(-1,30)
            ax.grid(False); ax.legend(frameon=False, fontsize=9)
            plt.tight_layout()
            out2 = os.path.join(base_dir, "force_cumulative_vs_monomer.png")
            plt.savefig(out2, dpi=300, bbox_inches='tight'); plt.close(fig); print(f"Saved {out2}")
        return

    print(f"Unsupported plot_type: {plot_type}")


def plot_tethered_across_LJ_for_gamma(
    config,
    base_dir,
    target_gamma,                 # float like 0.0055 or string "gamma_00055"
    anchor=0,                     # tethered bead index
    include_expected=True,
    outname=None,
    save_errors=True,
    errors_name=None,
    aggregate_runs=True,
    skip = {},         # if True, use mean±std across run_* per shear; if False, plot each run
    N_mono=50,
    a=0.03
):

    # ---------- helpers ----------
    def tethered_max_extension_last_frame(positions, anchor_idx=0):
        p = positions[-1]                             # last frame (N,D)
        d = np.linalg.norm(p - p[anchor_idx], axis=1)
        return float(np.max(d))

    def fmt_kT(val):
        r = round(val)
        if abs(val - r) < 1e-9:
            return f"{int(r)} kT"
        s = f"{val:.2f}".rstrip('0').rstrip('.')
        return f"{s} kT"

    # ---------- styling / constants ----------
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("Dark2").colors
    
    dyn_visc_SI = config["physical"]["VISC_dyn"] / 10.0   # -> dyn·s/cm²
    print(N_mono)
    # expected tethered curve
    exp_x = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1280], float)
    exp_y = np.array([0.00, 0.35, 0.55, 0.63, 0.70, 0.75, 0.81, 0.85, 0.90, 0.92, 0.94, 0.96, 0.96, 0.96], float)
    if N_mono==30:
        exp_s = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], float)
    elif N_mono==50 and a==0.03:
        exp_s = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], float)
    elif N_mono==60:
        exp_s = np.array([0.07, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], float)
    elif N_mono==100 and a==0.03:
        exp_s = np.array([0.07, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], float)

    # parse/format gamma
    gpat = re.compile(r"gamma_(\d{5})")
    if isinstance(target_gamma, str) and gpat.match(target_gamma):
        gamma_folder = target_gamma
        gamma_val = int(gpat.match(target_gamma).group(1)) / 1e4
    else:
        g = float(target_gamma)
        gamma_folder = f"gamma_{int(round(g*1e4)):05d}"
        gamma_val = g

    # find LJ folders
    lj_pat = re.compile(r"LJ_(\d{5})")
    entries = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    lj_dirs = [d for d in entries if lj_pat.match(d)] or entries
    def _lj_key(d):
        m = lj_pat.match(d)
        return int(m.group(1)) if m else d
    lj_dirs.sort(key=_lj_key)

    lj_dirs = [d for d in lj_dirs if d not in skip]

    fig, ax = plt.subplots(figsize=(8, 4))
    any_curve = False
    error_rows = []

    for i, lj in enumerate(lj_dirs):
        lj_path = os.path.join(base_dir, lj)
        gpath = os.path.join(lj_path, gamma_folder)
        if not os.path.isdir(gpath):
            continue

        m = lj_pat.match(lj)
        if m:
            lj_factor_4kT = int(m.group(1)) / 1e2
            if a==0.05:
                kT_value = lj_factor_4kT * 4.0
            if a==0.03:
                kT_value = lj_factor_4kT
            lj_label_base = fmt_kT(kT_value)
        else:
            lj_label_base = lj

        shear_dirs = [d for d in os.listdir(gpath) if re.match(r"shear_\d+$", d)]
        if not shear_dirs:
            continue
        shear_dirs.sort(key=lambda s: int(s.split('_')[1]))

        Lmax = None
        xs, ys_mean, ys_std = [], [], []
        run_curves = {}

        for sd in shear_dirs:
            shear_int = int(sd.split('_')[1])
            sdir = os.path.join(gpath, sd)
            run_dirs = [d for d in os.listdir(sdir) if re.match(r"run_\d+$", d)] or [""]

            E_finals, per_run_Efinal, per_run_E0 = [], {}, {}
            for r in run_dirs:
                ddir = os.path.join(sdir, r) if r else sdir
                f = os.path.join(ddir, "positions.npz")
                if not os.path.isfile(f):
                    continue
                try:
                    pos = np.load(f)["positions"]
                    if Lmax is None:
                        N = pos.shape[1]
                        Lmax = N * 2.0 * a
                    Efin = tethered_max_extension_last_frame(pos, anchor)
                    #print(f"[DATA] {f} -> extension={Efin:.6f}")

                    run_id = r if r else "run_00000"
                    E_finals.append(Efin)
                    per_run_Efinal[run_id] = Efin
                    if shear_int == 0:
                        per_run_E0[run_id] = Efin
                except Exception as e:
                    print(f"Skip {f}: {e}")

            if not E_finals:
                continue

            xs.append(shear_int * dyn_visc_SI * 10.0)
            ys_mean.append(np.mean(E_finals))
            ys_std.append(np.std(E_finals))

            for run_id, Efin in per_run_Efinal.items():
                if run_id not in run_curves:
                    run_curves[run_id] = {"x": [], "Efin": [], "E0": None}
                run_curves[run_id]["x"].append(xs[-1])
                run_curves[run_id]["Efin"].append(Efin)
                if run_id in per_run_E0:
                    run_curves[run_id]["E0"] = per_run_E0[run_id]
                    
        if not xs or Lmax is None:
            continue

        xs = np.array(xs, float)
        order = np.argsort(xs)
        xs, ys_mean, ys_std = xs[order], np.array(ys_mean)[order], np.array(ys_std)[order]

        c = colors[i % len(colors)]

        if aggregate_runs:
            E0_mean = ys_mean[xs == 0][0] if np.any(xs == 0) else ys_mean[0]
            # print(E0_mean)
            denom = Lmax - E0_mean
            if denom <= 0 or not np.isfinite(denom):
                print(f"[WARN] Skipping {lj_label_base}")
                continue
            y_mean_norm = (ys_mean - E0_mean) / denom
            y_std_norm  = ys_std / denom

            legend_label = lj_label_base
            mask = (xs >= np.min(exp_x)) & (xs <= np.max(exp_x))
            if np.any(mask):
                y_exp = np.interp(xs[mask], exp_x, exp_y)
                s_exp = np.interp(xs[mask], exp_x, exp_s)
                s_exp = np.where(s_exp <= 0, 1e-12, s_exp)

                resid = y_mean_norm[mask] - y_exp
                s_sim = np.where(y_std_norm[mask] <= 0, 1e-12, y_std_norm[mask])

                denom_err = s_exp**2 + s_sim**2
                sse_scaled = float(np.sum((resid**2) / denom_err))
                npt = int(np.sum(mask))
                mse_scaled = sse_scaled / npt
                rmse_scaled = float(np.sqrt(mse_scaled))

                legend_label = r"$\epsilon_{LJ}=$" + f"{lj_label_base} " + r"($t^{2}_{14}$="+f"{mse_scaled:.3f})"
                if save_errors:
                    error_rows.append(["mean", lj_label_base, lj_label_base, npt,
                                       sse_scaled, mse_scaled, rmse_scaled, float('nan'), float('nan')])

            # ---- use errorbar instead of fill ----
            ax.errorbar(xs, y_mean_norm, yerr=y_std_norm,
                        fmt='o-', lw=1.6, markersize=4, color=c, capsize=3, label=legend_label)

        else:
            for rj, dct in sorted(run_curves.items()):
                x_run = np.array(dct["x"], float)
                Efin  = np.array(dct["Efin"], float)
                if x_run.size < 2:
                    continue
                if dct["E0"] is not None:
                    E0_run = float(dct["E0"])
                else:
                    E0_run = float(Efin[np.argmin(x_run)])
                den_run = Lmax - E0_run
                if den_run <= 0 or not np.isfinite(den_run):
                    continue
                y_run_norm = (Efin - E0_run) / den_run
                ord_run = np.argsort(x_run)
                x_run, y_run_norm = x_run[ord_run], y_run_norm[ord_run]

                ax.errorbar(x_run, y_run_norm, yerr=None,  # no std per single run
                            fmt='o-', lw=1.0, ms=3, color=c, alpha=0.65,
                            label=f"{lj_label_base}:{rj}")

        any_curve = True

    if not any_curve:
        print(f"[INFO] No usable data for {gamma_folder} under {base_dir}.")
        return

    # expected tethered curve (keep shaded band)
    if include_expected:
        ax.plot(exp_x, exp_y, '--', color='grey', linewidth=1,
        label="Fu et al. (tethered)")
        ax.fill_between(exp_x, exp_y - exp_s, exp_y + exp_s, color='grey', alpha=0.2, linewidth=0)
    ax.set_ylim(0,1.2)
    ax.set_xlim(0,1280)
    ax.axhline(y=1, color='k', linestyle='--')
    ax.set_xlabel(r"Wall shear stress [dyn cm$^{-2}$]", fontsize=15)
    ax.set_ylabel(r"$\hat{E}^{final}_{uncoiling}$", fontsize=15)
    ax.legend(fontsize=9, frameon=True, ncols=1 if aggregate_runs else 2,loc='center left', bbox_to_anchor=(1, 0.5))
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(False)
    plt.tight_layout()

    if outname is None:
        outname = f"tethered_extension_vs_shear_across_LJ_gamma_{int(round(gamma_val*1e4)):05d}.png"
        if not aggregate_runs:
            outname = outname.replace(".png", "_allruns.png")
    out_png = os.path.join(base_dir, outname)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_png}")

    if save_errors:
        if errors_name is None:
            errors_name = outname.replace(".png", "_errors.csv")
        csv_path = os.path.join(base_dir, errors_name)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["level","lj_label","identifier","n_points","sse","mse","rmse","chi2","chi2_per_point"])
            for row in error_rows:
                w.writerow(row)
        print(f"Saved {csv_path}")

def plot_cumulative_force_across_gammas(
    config,
    lj_dir,
    target_shear,
    avg_window=100,
    start_from_right=30,   # how many monomers (from the chain end) to include in the cumulative
    use_last_fraction=0.25, # take last 25% of time series for averaging; set None for full length
    fine_model=False,
    skip=[]
):
    """
    Plot cumulative tensile force vs monomer index for all gamma_* in lj_dir at a given shear,
    and compute a reduced chi2-like metric = mean((diff/sigma_ref)^2).

    If fine_model is True:
      - validation reference uses 0.5/6 pN per monomer (with 0.1/6 pN uncertainty per monomer),
      - number of monomers is taken from the data (e.g., 150 for fine model).
    """
    import os, re
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("Dark2").colors

    # --- time scaling (unchanged) ---
    TIMESTEP = config["simulation"]["TIME_STEP"]
    STEPS    = config["simulation"]["STEPS"]
    dt = 1e-4 * TIMESTEP * STEPS

    lj_match = re.search(r"LJ_(\d{5})", lj_dir)
    lj_text = f"Lennard Jones potential = {int(lj_match.group(1))/1e2:.2f} × 4kT" if lj_match else ""

    gamma_dirs = [d for d in os.listdir(lj_dir) if os.path.isdir(os.path.join(lj_dir, d)) and d.startswith("gamma_")]
    def _gkey(g):
        m = re.match(r"gamma_(\d+)", g)
        return int(m.group(1)) if m else 0
    gamma_dirs.sort(key=_gkey)

    def _gamma_float(name):
        m = re.match(r"gamma_(\d+)", name)
        return (int(m.group(1)) / 1e4) if m else None

    def _load_forces_npz(path):
        try:
            return np.load(path)["forces"]  # (T, N, 3)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            return None

    N_detect = None
    blocks = {}

    # ---- collect blocks per gamma ----
    for gdir in gamma_dirs:
        gval = _gamma_float(gdir)
        if gval is None:
            continue

        if gval in skip:
            print(f"[skip] {gval}")
            continue

        shear_dir = os.path.join(lj_dir, gdir, f"shear_{target_shear}")
        fpath = os.path.join(shear_dir, "drag_forces.npz")
        if not os.path.isfile(fpath):
            print(f"[skip] {fpath} not found")
            continue

        F = _load_forces_npz(fpath)
        if F is None:
            continue

        if N_detect is None:
            N_detect = F.shape[1]
        N_use = min(N_detect, F.shape[1])

        # pN conversion (assuming original units → nN? keep as in your code)
        Fmag = np.linalg.norm(F[:, :N_use, :], axis=2) * 1000.0

        # moving_average_centered is assumed available in your environment
        Fm   = moving_average_centered(Fmag, avg_window)

        if use_last_fraction is not None and 0 < use_last_fraction <= 1.0:
            nL = max(int(len(Fm) * use_last_fraction), 1)
        else:
            nL = len(Fm)
        tail = Fm[-nL:, :] if nL > 0 else Fm

        blocks[gval] = tail

    if not blocks or N_detect is None:
        print("No usable force data found for this LJ folder and shear.")
        return {}

    # ---- reference per-monomer (standard vs fine model) ----
    ref_per_mono   = 0.5     # pN per monomer
    sigma_per_mono = 0.1     # pN per monomer (keeps same relative band)

    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(0, N_detect)
    start = min(start_from_right, N_detect)

    # Reference line (per-monomer cumulatively from the right) and uncertainty
    steps_from_right = (start - np.arange(1, start+1) + 1)  # 1,2,3,...,start
    ref = np.full(N_detect, np.nan)
    ref[:start] = ref_per_mono * steps_from_right
    sigma_ref = np.full(N_detect, np.nan)
    sigma_ref[:start] = sigma_per_mono * steps_from_right   # grows linearly with steps

    mse_by_gamma = {}

    for i, (gval, block) in enumerate(sorted(blocks.items(), key=lambda kv: kv[0])):
        # take last 'start' monomers from the chain end: reverse index
        sub = block[:, :start][:, ::-1]
        cum = np.cumsum(sub, axis=1)[:, ::-1]
        mu  = cum.mean(axis=0)
        sig = cum.std(axis=0)

        # --- Normalized MSE: divide by sigma_ref^2 ---
        diff = mu - ref[:start]
        norm_mse = float(np.mean((diff / sigma_ref[:start])**2))
        mse_by_gamma[gval] = norm_mse

        y = np.full(N_detect, np.nan); e = np.full(N_detect, np.nan)
        y[:start], e[:start] = mu, sig

        ax.errorbar(
            idx, y, yerr=e,
            marker='o', lw=1.2, elinewidth=1, capsize=3, markersize=3,
            color=color_cycle[i % len(color_cycle)],
            label=fr"$\gamma$={gval:.4f}" + r" ($t^{2}_{50}$="+ f"{round(norm_mse,3)})"
        )

    # Reference + band
    mask = ~np.isnan(ref)
    ref_label = (r"Fu et al. $0.5 \pm 0.1$ pN/mono")
    ax.plot(idx, ref, '--', lw=1.2, color='grey', label=ref_label)
    ax.fill_between(idx[mask], ref[mask] - sigma_ref[mask], ref[mask] + sigma_ref[mask], alpha=0.25, color="grey")
    ax.set_xlim(0,0.1)
    ax.set_xlabel("Monomer index", fontsize=15)
    ax.set_ylabel("Tensile force [pN]", fontsize=15)

    # Adaptive x ticks for long chains
    if N_detect <= 60:
        xtick_step = 5
    elif N_detect <= 150:
        xtick_step = 10
    else:
        xtick_step = max(1, N_detect // 15)
    ax.set_xticks(np.arange(0, N_detect+1, xtick_step))
    ax.set_xlim(-1, max(start, N_detect))

    ax.grid(False)
    ax.legend(frameon=False, fontsize=9, ncol=1)
    title_top = f"Cumulative force vs monomer index (shear={target_shear})"
    if lj_text:
        title_top += f"\n{lj_text}"
    if fine_model:
        title_top += " [fine model]"
    # ax.set_title(title_top, fontsize=14)
    plt.tight_layout()

    suffix = "_fine" if fine_model else ""
    out = os.path.join(lj_dir, f"force_cumulative_vs_monomer_all_gamma_shear_{target_shear}{suffix}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    print("== Normalized MSE per gamma ==")
    for gval in sorted(mse_by_gamma):
        print(f"  gamma={gval:.4f}: " + r"$\chi^{2}_{14}$" + f" = {mse_by_gamma[gval]:.6g}")

    return mse_by_gamma


def plot_recoiling_across_LJ_for_gamma(
    config,
    root_dir,
    gamma_value,                 # float like 0.0055 or string like "gamma_00055"
    N_mono,
    time_step=None,
    add_expected=True,           # overlay expected reference if desired
    aggregate_runs=True,         # if True: mean±std across run_*; if False: plot each run
    show_individual_when_agg=False,  # if True, also show faint individual runs behind the mean
    runs_glob="run_*",           # pattern to discover run folders
    extension_metric="x",        # "euclidean" or "x"
    skip=[]
):
    """
    Plot recoiling dynamics (tethered metric, normalized to t=0) for ONE gamma across ALL LJ_* folders.
    Computes reduced chi-squared (χ²/N) vs an expected curve by interpolating simulation to the reference x-points.
    The per-point denominator uses combined uncertainty: (sigma_exp + sigma_sim)^2, where sigma_sim is
    the std across runs (interpolated to the expected x).
    """
    import os, re, glob
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    # --- helpers ---
    def tethered_extension_ts(positions, metric, anchor=0):
        """
        Per-frame 'tethered' extension relative to anchor:
          - euclidean: max || r_i - r_anchor || over monomers i
          - x        : max | x_i - x_anchor | over monomers i
        Returns shape (T,)
        """
        p0 = positions[:, anchor, :]  # (T, D)
        if metric == "euclidean":
            d = np.linalg.norm(positions - p0[:, None, :], axis=2)  # (T, N)
        elif metric == "x":
            dx = positions[:, :, 0] - p0[:, None, 0]                # (T, N)
            d = dx  # use np.abs(dx) if you want absolute separation
        else:
            raise ValueError(f"Unknown extension_metric='{metric}'. Use 'euclidean' or 'x'.")
        return d.max(axis=1)  # (T,)

    def _gamma_folder_from_value(gv):
        if isinstance(gv, str):
            m = re.match(r"gamma_(\d{5})$", gv)
            if m:
                return gv, int(m.group(1)) / 1e4
            gv_float = float(gv)
        else:
            gv_float = float(gv)
        code = f"{int(round(gv_float * 1e4)):05d}"
        return f"gamma_{code}", gv_float

    # --- reduced chi-squared (χ²/N) with combined error (σ_exp + σ_sim) ---
    def curve_chi2_reduced_with_simerr(t_dense, y_mean, y_std, x_ref, y_ref, err_ref):
        """
        Interpolate y_mean and y_std at x_ref (within sim coverage),
        then compute reduced chi² using σ_total = σ_exp + σ_sim:
            chi2_red = (1/N) * sum_i [ ((y_i - y_ref_i) / (σ_exp_i + σ_sim_i))^2 ].
        """
        if (t_dense is None) or (len(t_dense) == 0) or (x_ref is None) or (err_ref is None):
            return np.inf, 0
        mask = (x_ref >= t_dense[0]) & (x_ref <= t_dense[-1])
        N = int(mask.sum())
        if N == 0:
            return np.inf, 0

        y_interp = np.interp(x_ref[mask], t_dense, y_mean)
        ystd_interp = np.interp(x_ref[mask], t_dense, y_std)
        sigma_total = np.maximum(err_ref[mask] + ystd_interp, 1e-12)  # avoid divide-by-zero
        chi2 = np.sum(((y_interp - y_ref[mask]) / sigma_total) ** 2)
        chi2_red = chi2 / N
        return float(chi2_red), N

    # --- time step setup ---
    TIMESTEP = config["simulation"]["TIME_STEP"]
    STEPS    = config["simulation"]["STEPS"]
    if time_step is not None:
        TIMESTEP = time_step
    dt = 1e-4 * TIMESTEP * STEPS

    # colors
    cmap = plt.get_cmap("Dark2")
    color_cycle = list(cmap.colors)

    gamma_folder, gamma_val = _gamma_folder_from_value(gamma_value)

    # collect LJ_* folders
    lj_dirs = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d)) and re.match(r"LJ_\d{5}$", d)]
    lj_dirs.sort(key=lambda d: int(re.search(r"LJ_(\d{5})", d).group(1)))

    fig, ax = plt.subplots(figsize=(6, 4))
    nplotted = 0

    # expected/reference curve + errors (plotting AND metrics)
    if N_mono == 50:
        expected_x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        expected_y = np.array([1, 0.4, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        expected_err = np.array([0.1, 0.3, 0.2, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    elif N_mono == 100:
        expected_x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        expected_y = np.array([1, 0.55, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1])
        expected_err = np.array([0.1, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1])
    else:
        expected_x = expected_y = expected_err = None

    # store reduced chi2 per LJ
    chi2_by_lj = {}

    for i, lj in enumerate(lj_dirs):
        lj_path = os.path.join(root_dir, lj)
        gamma_path = os.path.join(lj_path, gamma_folder)

        # Case A: single positions.npz directly in gamma folder
        single_positions = os.path.join(gamma_path, "positions.npz")

        # Case B: multiple runs inside gamma folder
        run_positions = sorted(glob.glob(os.path.join(gamma_path, runs_glob, "positions.npz")))

        series_list = []  # list of y(t) arrays for this LJ

        if os.path.isfile(single_positions):
            try:
                pos = np.load(single_positions)["positions"]     # (T,N,D)
                E = tethered_extension_ts(pos, anchor=0, metric=extension_metric)
                if len(E) > 0 and E[0] > 0:
                    series_list.append(E / E[0])
                else:
                    print(f"[skip] Bad initial extension in {single_positions}")
            except Exception as e:
                print(f"[skip] Error reading {single_positions}: {e}")

        if run_positions:
            for fpos in run_positions:
                try:
                    pos = np.load(fpos)["positions"]
                    E = tethered_extension_ts(pos, anchor=0, metric=extension_metric)
                    # normalization using a theoretical resting extension (if desired)
                    if N_mono == 50:
                        theoretical_R = 0.3
                    if N_mono == 100:
                        theoretical_R = 0.4 #+ 0.5
                    if len(E) > 0 and E[0] > 0:
                        series_list.append((E - theoretical_R) / (E[0] - theoretical_R))
                    else:
                        print(f"[skip] Bad initial extension in {fpos}")
                except Exception as e:
                    print(f"[skip] Error reading {fpos}: {e}")

        if not series_list:
            print(f"[skip] No usable positions for {lj}/{gamma_folder}")
            continue

        # align runs by truncating to min length
        min_len = min(len(s) for s in series_list)
        Y = np.stack([s[:min_len] for s in series_list], axis=0)   # (R,T)
        t = np.arange(min_len) * dt
        print(len(t))
        # label base & skip logic
        mlj = re.search(r"LJ_(\d{5})", lj)
        if int(int(mlj.group(1))/1e2) in skip:
            continue
        lj_base = f"{int(int(mlj.group(1))/1e2)}" if mlj else lj  # numeric LJ in kT for the label
        color = color_cycle[nplotted % len(color_cycle)]

        # --- plotting and chi² computation, with label including χ² ---
        if aggregate_runs and Y.shape[0] > 1:
            y_mean = Y.mean(axis=0)
            y_std  = Y.std(axis=0, ddof=1)
            if show_individual_when_agg:
                for r in range(Y.shape[0]):
                    ax.plot(t, Y[r], lw=0.8, alpha=0.25, color=color)

            if expected_x is not None:
                chi2_red, n_used = curve_chi2_reduced_with_simerr(
                    t, y_mean, y_std, expected_x, expected_y, expected_err
                )
                chi2_by_lj[lj_base] = (chi2_red, n_used)
                lj_text = fr"$\epsilon_{{\mathrm{{LJ}}}} = {lj_base}\,\mathrm{{k_bT}}$ $(t^2_{{\mathrm{{8}}}} = {chi2_red:.3f})$"
            else:
                lj_text = fr"$\epsilon_{{\mathrm{{LJ}}}} = {lj_base}\,\mathrm{{kT}}$"

            ax.plot(t, y_mean, lw=1.8, color=color, label=lj_text)
            ax.fill_between(t, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)

        else:
            # per-run plotting; average reduced chi2 across runs (simple mean)
            chi2_accum = 0.0
            n_counted = 0
            for r in range(Y.shape[0]):
                ax.plot(t, Y[r], lw=1.2, color=color, alpha=0.9 if Y.shape[0]==1 else 0.8,
                        label=None if Y.shape[0] > 1 else fr"$\epsilon_{{\mathrm{{LJ}}}} = {lj_base}\,\mathrm{{kT}}$")

                if expected_x is not None:
                    # no std available per single trace → 0
                    chi2_r, n_used = curve_chi2_reduced_with_simerr(
                        t, Y[r], np.zeros_like(Y[r]), expected_x, expected_y, expected_err
                    )
                    if np.isfinite(chi2_r):
                        chi2_accum += chi2_r
                        n_counted += 1

            if expected_x is not None and n_counted > 0:
                chi2_red = chi2_accum / n_counted
                chi2_by_lj[lj_base] = (chi2_red, n_used)
                lj_text = fr"$\epsilon_{{\mathrm{{LJ}}}} = {lj_base}\,\mathrm{{kT}},\ \chi^2_{{\mathrm{{red}}}} = {chi2_red:.3f}$"
                # add a representative line to the legend (use the mean across runs just for legend clarity)
                ax.plot([], [], lw=1.8, color=color, label=lj_text)  # invisible handle for legend
            elif expected_x is None:
                lj_text = fr"$\epsilon_{{\mathrm{{LJ}}}} = {lj_base}\,\mathrm{{kT}}$"
                ax.plot([], [], lw=1.8, color=color, label=lj_text)

        nplotted += 1

    if nplotted == 0:
        print("No recoiling data found for the requested gamma across LJ folders.")
        plt.close(fig)
        return {}

    # optional expected reference (for plot only)
    if add_expected and (expected_x is not None):
        ax.plot(expected_x, expected_y, '--', lw=1.2, color='grey', label=r"Fu et al.")
        ax.fill_between(expected_x, expected_y - expected_err, expected_y + expected_err, alpha=0.25, color="grey")
    if N_mono==100:
        ax.set_ylim(-0.05,1)
    if N_mono==50:
        ax.set_ylim(-0.1,1)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Time [s]", fontsize=13)
    ax.set_ylabel(r"$\hat{E}_{\mathrm{recoiling}}$", fontsize=13)
    #ax.legend(fontsize=9)
    ax.legend(fontsize=9, frameon=True, loc='upper right')
    ax.grid(False)
    plt.tight_layout()

    out = os.path.join(root_dir, f"recoiling_all_LJ_gamma_{gamma_folder.split('_')[1]}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # pretty-print and return metrics, sorted best-to-worst
    if chi2_by_lj:
        items = sorted(chi2_by_lj.items(), key=lambda kv: kv[1][0])
        print("\nReduced χ² (σ_total = σ_exp + σ_sim):")
        for lj_base, (chi2_red, n_used) in items:
            print(f"  {lj_base:>6} kT: χ²/N = {chi2_red:.5f}  (N_ref_used={n_used})")
        return {k: v[0] for k, v in items}
    else:
        print("No χ² computed (no expected curve or no overlap).")
        return {}


def plot_free_across_LJ_for_gamma(
    config,
    base_dir,
    target_gamma,                 
    free_metric="end_to_end",     # "end_to_end" (default) or "max_pairwise"
    aggregate_runs=True,
    include_expected=True,
    Box_Lx=64,
    Box_Ly=32,
    Box_Lz=32,
    outname=None
):
    """
    Plot normalized final extension vs wall shear stress for the FREE case,
    aggregating over run_* per shear and drawing multiple LJ_* in one figure.

    Directory structure assumed:
        base_dir/
          LJ_XXXXX/
            gamma_YYYYY/
              shear_SSS/
                run_RRR/
                  positions.npz

    Normalization per LJ & gamma:
        y = (E_final - E0) / (Lmax - E0), with E0 at shear==0 (or smallest).
        Lmax = N * 2 * a (dynamic N from the first loaded positions file).
    """

    # ---------- small helpers ----------
    def fmt_kT(val):
        r = round(val)
        if abs(val - r) < 1e-9:
            return f"{int(r)} kT"
        s = f"{val:.2f}".rstrip('0').rstrip('.')
        return f"{s} kT"

    def _compute_final_extension(positions, Lx, Ly, Lz, metric):
        if metric == "end_to_end":
            return float(end_to_end_extension_ts_pbc(positions, Lx, Ly, Lz)[-1])
        else:
            return float(max_pairwise_extension_ts_pbc(positions, Lx, Ly, Lz)[-1])


    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("Dark2").colors

    a = config["physical"]["a"]
    dyn_visc_SI = config["physical"]["VISC_dyn"] / 10.0

    Lx_eff = Box_Lx 
    Ly_eff = Box_Ly
    Lz_eff = Box_Lz

    # expected (free) "Schneider et al."
    exp_x = np.array([1, 10, 100, 400, 1100, 5000, 10000], dtype=float)
    exp_y     = np.array([0.07,  0.07,   0.07,  0.07,    0.07,   0.75,   1], dtype=float)
    exp_s     = np.array([0.07, 0.07, 0.07, 0.07,  0.07,  0.3,   0.3], dtype=float)

    # parse/format gamma
    gpat = re.compile(r"gamma_(\d{5})")
    if isinstance(target_gamma, str) and gpat.match(target_gamma):
        gamma_folder = target_gamma
        gamma_val = int(gpat.match(target_gamma).group(1)) / 1e4
    else:
        g = float(target_gamma)
        gamma_folder = f"gamma_{int(round(g*1e4)):05d}"
        gamma_val = g

    # collect LJ folders within base_dir
    lj_pat = re.compile(r"LJ_(\d{5})")
    entries = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    lj_dirs = [d for d in entries if lj_pat.match(d)] or entries

    def _lj_key(d):
        m = lj_pat.match(d)
        return int(m.group(1)) if m else d
    lj_dirs.sort(key=_lj_key)

    fig, ax = plt.subplots(figsize=(6, 4))
    any_curve = False

    for i, lj in enumerate(lj_dirs):
        lj_path = os.path.join(base_dir, lj)
        gpath = os.path.join(lj_path, gamma_folder)
        if not os.path.isdir(gpath):
            continue

        # Format LJ label as kT value like your tethered helper
        m = lj_pat.match(lj)
        if m:
            lj_factor_kT = int(m.group(1)) / 1e2     # “XX.XX × 4kT”
            kT_value = lj_factor_kT 
            lj_label_base = fmt_kT(kT_value)
        else:
            lj_label_base = lj

        # Discover shear_* directories
        shear_dirs = [d for d in os.listdir(gpath) if re.match(r"shear_\d+$", d)]
        if not shear_dirs:
            continue
        shear_dirs.sort(key=lambda s: int(s.split('_')[1]))

        # Dynamic Lmax from first file we encounter for this LJ/gamma
        Lmax = None

        xs, ys_mean, ys_std = [], [], []
        run_points = {}  # if aggregate_runs=False, collect per-run points

        for sd in shear_dirs:
            shear_int = int(sd.split('_')[1])
            sdir = os.path.join(gpath, sd)
            run_dirs = [d for d in os.listdir(sdir) if re.match(r"run_\d+$", d)] or [""]

            Efinals = []
            per_run_val = {}

            for r in run_dirs:
                ddir = os.path.join(sdir, r) if r else sdir
                f = os.path.join(ddir, "positions.npz")
                if not os.path.isfile(f):
                    continue
                try:
                    pos = np.load(f)["positions"]  # (T,N,D)
                    if Lmax is None:
                        N = pos.shape[1]
                        Lmax = N * 2.0 * a
                    Efin = _compute_final_extension(pos, Lx_eff, Ly_eff, Lz_eff, free_metric)
                    Efinals.append(Efin)
                    run_id = r if r else "run_00000"
                    per_run_val[run_id] = Efin
                except Exception as e:
                    print(f"Skip {f}: {e}")

            # test this
            Lmax = np.max(Efinals) if Lmax is None else max(Lmax, np.max(Efinals))

            if not Efinals:
                continue

            x_val = shear_int # shear rate in 1/s
            xs.append(x_val)
            ys_mean.append(np.mean(Efinals))
            ys_std.append(np.std(Efinals))

            if not aggregate_runs:
                for run_id, Efin in per_run_val.items():
                    if run_id not in run_points:
                        run_points[run_id] = {"x": [], "Efin": []}
                    run_points[run_id]["x"].append(x_val)
                    run_points[run_id]["Efin"].append(Efin)

        if not xs or Lmax is None:
            continue

        xs = np.array(xs, float)
        order = np.argsort(xs)
        xs, ys_mean, ys_std = xs[order], np.array(ys_mean)[order], np.array(ys_std)[order]

        # normalization E/Lmax 
        # normalization (E - E0)/(Lmax - E0), E0 at shear==0 if present, else smallest shear
        if np.any(xs == 0):
            E0_mean = ys_mean[xs == 0][0]
        else:
            E0_mean = ys_mean[0]

        # denom = Lmax - E0_mean
        # normalization E/Lmax
        denom = Lmax
        if denom <= 0 or not np.isfinite(denom):
            print(f"[WARN] Skipping {lj_label_base}: invalid normalization (denom={denom})")
            continue

        # y_mean_norm = (ys_mean - E0_mean) / denom
        # normalization E/Lmax
        y_mean_norm = ys_mean / denom
        y_std_norm  = ys_std / denom

        c = colors[i % len(colors)]
        label = f"{lj_label_base} ({'e2e' if free_metric=='end_to_end' else 'max-diam'})"

        if aggregate_runs:
            ax.errorbar(xs, y_mean_norm, yerr=y_std_norm,
                        fmt='o-', lw=1.6, markersize=4, capsize=3, color=c, label=label)
        else:
            # plot each run as a light curve normalized by its own E0
            # (E0 per run is the run's value at shear==0 if present, else its smallest shear)
            for rj, dct in sorted(run_points.items()):
                x_run = np.array(dct["x"], float)
                Efin  = np.array(dct["Efin"], float)
                if x_run.size < 1:
                    continue
                ord_run = np.argsort(x_run)
                x_run, Efin = x_run[ord_run], Efin[ord_run]
                # find E0 per run
                if np.any(x_run == 0):
                    E0_r = float(Efin[x_run == 0][0])
                else:
                    E0_r = float(Efin[0])
                den_r = Lmax - E0_r
                if den_r <= 0 or not np.isfinite(den_r):
                    continue
                y_run = (Efin - E0_r) / den_r
                ax.plot(x_run, y_run, 'o-', lw=1.0, ms=3, alpha=0.65, color=c,
                        label=f"{label}:{rj}")

        any_curve = True

    if not any_curve:
        print(f"[INFO] No usable data for gamma={gamma_folder} under {base_dir}.")
        return

    # expected overlay for free case
    if include_expected:
        ax.plot(exp_x, exp_y, linestyle='--', color='black', linewidth=1.6,
                marker='s', markersize=4, label="Schneider et al. (free)")
        ax.fill_between(exp_x, exp_y - exp_s, exp_y + exp_s,
                        color='black', alpha=0.12, linewidth=0)

    ax.set_xlabel(r"Shear rate [s$^{-1}$]", fontsize=15)
    ax.set_ylabel(r"$\hat{E}^{final}_{free}$", fontsize=15)
    ax.set_xscale("log")
    # choose a sensible x-limit automatically (covers both your data and expected curve)
    xmin = max(1e-1, min(ax.get_lines()[0].get_xdata())) if ax.get_lines() else 1e-1
    xmax = max([np.max(ln.get_xdata()) for ln in ax.get_lines() if len(ln.get_xdata()) > 0] + [2000.0])
    ax.set_xlim(xmin, xmax * 1.05)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9, frameon=False, ncols=1 if aggregate_runs else 2)
    ax.grid(False)
    plt.tight_layout()

    if outname is None:
        outname = f"free_extension_vs_shear_across_LJ_gamma_{int(round(gamma_val*1e4)):05d}"
        if not aggregate_runs:
            outname += "_allruns"
        outname += f"_{'e2e' if free_metric=='end_to_end' else 'maxpair'}"
        outname += ".png"

    out_png = os.path.join(base_dir, outname)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_png}")