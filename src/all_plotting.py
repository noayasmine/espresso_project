import os, re
import numpy as np
import matplotlib.pyplot as plt

# ---------- PBC helpers ----------
def min_image_1d(dr, L):
    """Minimum-image in 1D."""
    return dr if (L is None or L <= 0) else dr - np.round(dr / L) * L

def _normalize_to_Lmax(E, Lmax):
    E0 = E[0]
    den = Lmax - E0
    if den <= 0 or not np.isfinite(den):
        den = np.finfo(float).eps
    return (E - E0) / den

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
                            target_gamma=None):
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


    color_cycle = plt.get_cmap("tab20").colors

    a = config["physical"]["a"]
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

    fig, ax = plt.subplots(figsize=(8,4))
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
    else:
        ax.set_xlabel("Wall shear stress [dyn/cm²]",fontsize=15)
        ax.set_ylabel(r"$\hat{E}^{final}_{tethered}$", fontsize=15)
    ax.legend(fontsize=10, frameon=False); ax.grid(True); plt.tight_layout()
    out = os.path.join(base_dir, "extension_vs_shearstress_test.png")
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Plot saved to {out}")



# ---------- Time series (dynamic N + optional Box_Lx override + free_metric) ----------
def plot_time_series(config, base_dir, plot_type, target_shear=None, avg_window=1,
                     validation_type="free", Box_Lx=None, free_metric="end_to_end", time_step=None, k_fene=6.5):
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
    color_cycle = plt.get_cmap("tab20").colors

    a = config["physical"]["a"]
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
        fig, ax = plt.subplots(figsize=(12,6))
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
        ax.legend(fontsize='small'); ax.grid(True); plt.tight_layout()
        out = os.path.join(base_dir, "recoiling_extension_over_time.png")
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
        print(f"Saved {out}")
        return

    # ----- Extension (plot all shears; dynamic N per shear) -----
    if plot_type == "extension":
        fig, ax = plt.subplots(figsize=(6,4))
        for i, sd in enumerate(shear_dirs_all):
            f = os.path.join(base_dir, sd, "positions.npz")
            if not os.path.isfile(f): continue
            try:
                pos = np.load(f)["positions"]  # (T,N,D)
                N_here = pos.shape[1]
                print(f"N monomers = {N_here}")
                Lmax = N_here * 2 * a

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

                lbl = f"{shear_int} " + r"$s^{-1}$"
                if vtype == "free":
                    lbl += f" ({'end-to-end' if free_metric=='end_to_end' else 'max-pairwise'})"
                ax.plot(t, y, label=lbl, color=color_cycle[i % 20], lw=1.5)
            except Exception as e:
                print(f"Skip {f}: {e}")
        # ax.set_title(f"Extension over time\n{subtitle}", fontsize=16)
        ax.set_title(r"$k_{FENE}$=" + f"{k_fene}" + "mN/m", fontsize=16)
        ax.set_xlabel("Time [s]", fontsize=15); ax.set_ylabel(r"$\hat{E}_{teth}$", fontsize=15)
        
        if ax.has_data(): ax.legend(frameon=False, fontsize=10, ncol=2)
        ax.grid(True); plt.tight_layout()
        out = os.path.join(base_dir, "extension_time_test.png")
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
                
                c = color_cycle[i % 20]
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
            fig, ax = plt.subplots(figsize=(6,4))
            idx = np.arange(0, N_detect); start = min(30, N_detect)
            for i, (sd, block) in enumerate(sorted(lastN.items(), key=lambda x: int(x[0].split('_')[1]))):
                c = color_cycle[i % 20]
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


            ax.plot(idx, ref, '--', lw=1.2, color="pink", label=r"Fu et al. $0.5 \pm 0.1$ pN/monomer")
            mask = ~np.isnan(ref)
            ax.fill_between(idx[mask], ref[mask] - sigma[mask], ref[mask] + sigma[mask], color='pink', alpha=0.5)

            # ax.fill_between(idx, ref - 0.1, ref + 0.1, color='black', alpha=0.15)
            # ax.set_title(f"Cumulative force vs monomer index\n{subtitle}", fontsize=16)
            ax.set_xlabel("Monomer index", fontsize=15); ax.set_ylabel("Tensile force [pN]", fontsize=15)
            #ax.set_xticks(idx)
            ax.set_xticks(np.arange(0, N_detect+1, 5))
            ax.set_xlim(0,31)
            ax.grid(True); ax.legend(frameon=False, fontsize=10)
            plt.tight_layout()
            out2 = os.path.join(base_dir, "force_cumulative_vs_monomer_test.png")
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
    errors_name=None
):
    """
    One plot: normalized final extension vs wall shear stress for ONE gamma,
    aggregating across ALL LJ_* folders in `base_dir`. One curve per LJ.

    Metric (tethered): E_final = max_i || r_i - r_anchor || at the last frame.
    Normalization (per curve): (E_final - E0) / (Lmax - E0),
        where E0 is from shear==0 (or the smallest available shear).
    Lmax = N * 2 * a with N read dynamically from data (positions shape).

    Also computes squared error vs the expected tethered curve using
    linear interpolation of expected_y and expected_std at the curve’s x-points:
        SSE = sum( (y - y_exp)^2 )
        chi2 = sum( ((y - y_exp)/sigma)^2 )
    Reported for:
        - the LJ mean curve (aggregated across runs per shear) — shown in legend as χ²/pt
        - each run id curve (if the same run id appears across multiple shears)

    Saves a CSV with columns:
      level, lj_label, identifier, n_points, sse, mse, rmse, chi2, chi2_per_point
    """
    import os, re, csv
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------- helpers ----------
    def tethered_max_extension_last_frame(positions, anchor_idx=0):
        p = positions[-1]                             # last frame (N,D)
        d = np.linalg.norm(p - p[anchor_idx], axis=1)
        return float(np.max(d))

    def tethered_time_series(positions, anchor_idx=0):
        T = positions.shape[0]
        out = np.empty(T, float)
        for t in range(T):
            p = positions[t]
            d = np.linalg.norm(p - p[anchor_idx], axis=1)
            out[t] = np.max(d)
        return out

    def fmt_kT(val):
        r = round(val)
        if abs(val - r) < 1e-9:
            return f"{int(r)} kT"
        s = f"{val:.2f}".rstrip('0').rstrip('.')
        return f"{s} kT"

    # ---------- styling / constants ----------
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("tab20").colors
    
    a = config["physical"]["a"]
    dyn_visc_SI = config["physical"]["VISC_dyn"] / 10.0   # -> dyn·s/cm²

    # expected tethered curve
    exp_x = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1280], float)
    exp_y = np.array([0.00, 0.35, 0.55, 0.63, 0.70, 0.75, 0.81, 0.85, 0.90, 0.92, 0.94, 0.96, 0.96, 0.96], float)
    exp_s = np.array([0.1, 0.2, 0.2, 0.25, 0.3, 0.35, 0.35, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], float)

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

    fig, ax = plt.subplots(figsize=(6, 4))
    any_curve = False
    error_rows = []

    for i, lj in enumerate(lj_dirs):
        lj_path = os.path.join(base_dir, lj)
        gpath = os.path.join(lj_path, gamma_folder)
        if not os.path.isdir(gpath):
            continue

        m = lj_pat.match(lj)
        if m:
            # folder encodes factor × (4kT); multiply by 4 to display in kT
            lj_factor_4kT = int(m.group(1)) / 1e2
            kT_value = lj_factor_4kT * 4.0
            lj_label_base = fmt_kT(kT_value)
        else:
            lj_label_base = lj

        shear_dirs = [d for d in os.listdir(gpath) if re.match(r"shear_\d+$", d)]
        if not shear_dirs:
            continue
        shear_dirs.sort(key=lambda s: int(s.split('_')[1]))

        Lmax = None
        xs, ys_mean, ys_std = [], [], []
        run_curves = {}   # run_id -> dict(x=[], Efin=[], E0=None)

        for sd in shear_dirs:
            shear_val = int(sd.split('_')[1])
            sdir = os.path.join(gpath, sd)
            run_dirs = [d for d in os.listdir(sdir) if re.match(r"run_\d+$", d)] or [""]

            E_finals = []
            per_run_Efinal = {}
            per_run_E0_candidates = {}

            for r in run_dirs:
                ddir = os.path.join(sdir, r) if r else sdir
                f = os.path.join(ddir, "positions.npz")
                if not os.path.isfile(f):
                    continue
                try:
                    pos = np.load(f)["positions"]  # (T, N, D)
                    if Lmax is None:
                        N = pos.shape[1]
                        Lmax = N * 2.0 * a
                    Efin = tethered_max_extension_last_frame(pos, anchor)
                    E_finals.append(Efin)
                    run_id = r if r else "run_00000"
                    per_run_Efinal[run_id] = Efin
                    if shear_val == 0:
                        E0_run = tethered_max_extension_last_frame(pos, anchor)
                        per_run_E0_candidates[run_id] = ("exact", E0_run)
                except Exception as e:
                    print(f"Skip {f}: {e}")

            if not E_finals:
                continue

            xs.append(shear_val * dyn_visc_SI * 10.0)           # dyn/cm^2
            ys_mean.append(np.mean(E_finals))
            ys_std.append(np.std(E_finals))
            for run_id, Efin in per_run_Efinal.items():
                if run_id not in run_curves:
                    run_curves[run_id] = {"x": [], "Efin": [], "E0": None}
                run_curves[run_id]["x"].append(xs[-1])
                run_curves[run_id]["Efin"].append(Efin)
                if run_id in per_run_E0_candidates and per_run_E0_candidates[run_id][0] == "exact":
                    run_curves[run_id]["E0"] = per_run_E0_candidates[run_id][1]

        if not xs or Lmax is None:
            continue

        xs = np.array(xs, float)
        ys_mean = np.array(ys_mean, float)
        ys_std = np.array(ys_std, float)

        order = np.argsort(xs)
        xs, ys_mean, ys_std = xs[order], ys_mean[order], ys_std[order]
        E0_mean = ys_mean[xs == 0][0] if np.any(xs == 0) else ys_mean[0]
        denom = Lmax - E0_mean
        if denom <= 0 or not np.isfinite(denom):
            print(f"[WARN] Skipping {lj_label_base}: Lmax - E0_mean <= 0 (Lmax={Lmax:.6g}, E0={E0_mean:.6g}).")
            continue
        y_mean_norm = (ys_mean - E0_mean) / denom
        y_std_norm  = ys_std / denom

        # ---- error metrics (mean curve) BEFORE plotting so we can show χ² in legend ----
        legend_label = lj_label_base
        mask = (xs >= np.min(exp_x)) & (xs <= np.max(exp_x))
        if np.any(mask):
            y_exp = np.interp(xs[mask], exp_x, exp_y)
            s_exp = np.interp(xs[mask], exp_x, exp_s)
            s_exp = np.where(s_exp <= 0, 1e-12, s_exp)
            resid = y_mean_norm[mask] - y_exp
            sse   = float(np.sum(resid**2))
            npt   = int(np.sum(mask))
            mse   = sse / npt
            rmse  = np.sqrt(mse)
            chi2  = float(np.sum((resid / s_exp)**2))
            chi2pp= chi2 / npt
            legend_label = fr"{lj_label_base} ($\chi^2$/pt={chi2pp:.2f})"
            if save_errors:
                error_rows.append(["mean", lj_label_base, lj_label_base, npt, sse, mse, rmse, chi2, chi2pp])

        # ---- plot mean curve with χ² in legend ----
        c = colors[i % len(colors)]
        ax.plot(xs, y_mean_norm, marker='o', lw=1.6, color=c, label=legend_label)
        if not np.all(y_std_norm == 0):
            ax.fill_between(xs, y_mean_norm - y_std_norm, y_mean_norm + y_std_norm, color=c, alpha=0.25)
        any_curve = True

        # ---- per-run χ² (kept for CSV only; not in legend) ----
        for run_id, dct in run_curves.items():
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
            maskr = (x_run >= np.min(exp_x)) & (x_run <= np.max(exp_x))
            if not np.any(maskr):
                continue
            y_exp_r = np.interp(x_run[maskr], exp_x, exp_y)
            s_exp_r = np.interp(x_run[maskr], exp_x, exp_s)
            s_exp_r = np.where(s_exp_r <= 0, 1e-12, s_exp_r)
            resid_r = y_run_norm[maskr] - y_exp_r
            sse_r   = float(np.sum(resid_r**2))
            npt_r   = int(np.sum(maskr))
            mse_r   = sse_r / npt_r
            rmse_r  = np.sqrt(mse_r)
            chi2_r  = float(np.sum((resid_r / s_exp_r)**2))
            chi2pp_r= chi2_r / npt_r
            if save_errors:
                error_rows.append(["run", lj_label_base, f"{lj_label_base}:{run_id}", npt_r, sse_r, mse_r, rmse_r, chi2_r, chi2pp_r])

    if not any_curve:
        print(f"[INFO] No usable data for {gamma_folder} under {base_dir}.")
        return

    # overlay expected tethered curve
    if include_expected:
        ax.plot(exp_x, exp_y, 'k--', linewidth=2, marker='s', markersize=5, label="Fu et al. (tethered)")
        ax.fill_between(exp_x, exp_y - exp_s, exp_y + exp_s, color='black', alpha=0.12, linewidth=0)

    # ax.set_title(f"Tethered: Extension vs Shear Stress at γ={gamma_val:.4f}", fontsize=16)
    ax.set_xlabel("Wall shear stress [dyn/cm²]", fontsize=15)
    ax.set_ylabel(r"$\hat{E}^{final}_{tethered}$", fontsize=15)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True)
    plt.tight_layout()

    if outname is None:
        outname = f"tethered_extension_vs_shear_across_LJ_gamma_{int(round(gamma_val*1e4)):05d}.png"
    out_png = os.path.join(base_dir, outname)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_png}")

