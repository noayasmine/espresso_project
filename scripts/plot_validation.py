import os, re
import numpy as np
import matplotlib.pyplot as plt
import glob

def unwrap_chain_frame(frame, Lx, Ly, Lz):
    "helper function for free vwf"
    u = np.array(frame, dtype=float, copy=True)
    if Lx is None or Ly is None or Lz is None:
        return u
    box = np.array([Lx, Ly, Lz], dtype=float)
    for i in range(1, u.shape[0]):
        dr = u[i] - u[i - 1]
        dr = dr - box * np.round(dr / box)
        u[i] = u[i - 1] + dr
    return u

def tethered_max_extension_ts(positions, anchor=0):
    "get the extension for tethered vwf (max value from monomer 0 to any monomer)"
    T = positions.shape[0]
    out = np.empty(T, float)
    for t in range(T):
        p = positions[t]
        out[t] = np.linalg.norm(p - p[anchor], axis=1).max()
    return out

def free_max_pairwise_extension_ts(positions, Lx, Ly, Lz):
    "get the max extension for free vwf"
    T = positions.shape[0]
    out = np.empty(T, float)
    for t in range(T):
        u = unwrap_chain_frame(positions[t], Lx, Ly, Lz)
        dif = u[:, None, :] - u[None, :, :]
        out[t] = np.linalg.norm(dif, axis=2).max()
    return out

def moving_average_trailing_matrix(x, window):
    "calculate moving avg"
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    for i in range(x.shape[0]):
        start = max(0, i - window + 1)
        out[i] = x[start:i + 1].mean(axis=0)
    return out

def tethered_tail_mean_max_extension(positions, anchor=0, tail=20):
    "get the raw extension averaged over the last 20 steps"
    T = positions.shape[0]
    frames = positions[T - tail:T]
    ref = frames[:, anchor:anchor + 1, :]
    d = np.linalg.norm(frames - ref, axis=2)
    return d.max(axis=1).mean()

def free_tail_mean_max_pairwise(positions, Lx, Ly, Lz, tail=20):
    T = positions.shape[0]
    # tail = min(tail, T)
    frames = positions[T - tail:T]
    out = np.empty(tail, float)
    for k in range(tail):
        u = unwrap_chain_frame(frames[k], Lx, Ly, Lz)
        dif = u[:, None, :] - u[None, :, :]
        dist = np.linalg.norm(dif, axis=2)
        out[k] = dist.max()
    return out.mean()

def plot_validation_results(config, base_dir, validation_type="free", Box_Lx=None, target_gamma=None, tail=20, anchor=0, outname="extension_vs_shearstress.png"):
    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.get_cmap("Dark2").colors

    dyn_visc_SI = config["physical"]["VISC_dyn"] / 10

    geom = config.get("geometry", {})
    Lx_eff = Box_Lx if Box_Lx is not None else geom.get("BOX_Lx", None)
    Ly_eff = geom.get("BOX_Ly", None) if geom.get("BOX_Ly", None) is not None else Lx_eff
    Lz = geom.get("BOX_Lz", None)

    gamma_re = re.compile(r"^gamma_(\d{5})$")
    shear_re = re.compile(r"^shear_(\d+)$")
    run_re = re.compile(r"^run_(\d+)$")

    all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    bad = [d for d in all_dirs if d.startswith("gamma_") and not gamma_re.match(d)]
    if bad:
        raise ValueError("Found folders starting with 'gamma_' that are not 'gamma_#####': " + ", ".join(sorted(bad)))

    gamma_dirs = sorted([d for d in all_dirs if gamma_re.match(d)])

    if target_gamma is not None:
        if isinstance(target_gamma, str) and gamma_re.match(target_gamma) and target_gamma in gamma_dirs:
            gamma_dirs = [target_gamma]
        else:
            raise ValueError("target_gamma must be like 'gamma_#####' and exist in base_dir.")

    fig, ax = plt.subplots(figsize=(6, 4))

    for gi, gname in enumerate(gamma_dirs):
        gamma_val = int(gamma_re.match(gname).group(1)) / 1e4
        gpath = os.path.join(base_dir, gname)

        shear_dirs = sorted([d for d in os.listdir(gpath) if os.path.isdir(os.path.join(gpath, d)) and shear_re.match(d)], key=lambda s: int(shear_re.match(s).group(1)))

        shears = []
        means = []
        stds = []
        Lmax_gamma = None

        for sd in shear_dirs:
            sdir = os.path.join(gpath, sd)
            shear_rate = int(shear_re.match(sd).group(1))

            run_dirs = sorted([d for d in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, d)) and run_re.match(d)], key=lambda r: int(run_re.match(r).group(1)))
            run_paths = [os.path.join(sdir, r) for r in run_dirs] if run_dirs else [sdir]

            finals = []
            for rp in run_paths:
                pos = np.load(os.path.join(rp, "positions.npz"))["positions"]
                if str(validation_type).lower() == "tethered":
                    finals.append(tethered_tail_mean_max_extension(pos, anchor=anchor, tail=tail))
                else:
                    finals.append(free_tail_mean_max_pairwise(pos, Lx_eff, Ly_eff, Lz, tail=tail))

            finals = np.asarray(finals, float)
            Lmax_gamma = finals.max() if Lmax_gamma is None else max(Lmax_gamma, finals.max())

            tau_wall = shear_rate * dyn_visc_SI * 10
            shears.append(tau_wall)
            means.append(finals.mean())
            stds.append(finals.std())

        shears = np.asarray(shears, float)
        means = np.asarray(means, float)
        stds = np.asarray(stds, float)

        idx = np.argsort(shears)
        shears = shears[idx]
        means = means[idx]
        stds = stds[idx]

        E0 = means[shears == 0][0] if np.any(shears == 0) else means[0]
        denom = Lmax_gamma - E0
        norm_m = (means - E0) / denom if denom > 0 else means
        norm_s = stds / denom if denom > 0 else stds

        c = color_cycle[gi % len(color_cycle)]
        label = fr"$\gamma$={gamma_val:.4f}"
        label = label + (" (max-pairwise)" if str(validation_type).lower() == "free" else " (tethered)")

        ax.plot(shears, norm_m, marker="o", lw=1.5, color=c, label=label)
        ax.fill_between(shears, norm_m - norm_s, norm_m + norm_s, color=c, alpha=0.25)

    if str(validation_type).lower() == "tethered":
        expected_x = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1280], float)
        expected_y = np.array([0.00, 0.35, 0.55, 0.63, 0.70, 0.75, 0.81, 0.85, 0.90, 0.92, 0.94, 0.96, 0.96, 0.96], float)
        expected_std = np.array([0.1, 0.2, 0.2, 0.25, 0.3, 0.35, 0.35, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], float)
        ax.plot(expected_x, expected_y, linestyle="--", color="black", linewidth=2, marker="s", markersize=5, label="Expected")
        ax.fill_between(expected_x, expected_y - expected_std, expected_y + expected_std, color="black", alpha=0.12, linewidth=0)

    if str(validation_type).lower() == "free":
        expected_x = np.array([1, 10, 100, 400, 1100, 5000, 10000], float) * dyn_visc_SI * 10
        expected_y = np.array([0, 0, 0, 0, 0, 0.75, 1], float)
        expected_std = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2], float)
        ax.plot(expected_x, expected_y, linestyle="--", color="black", linewidth=2, marker="s", markersize=5, label="Schneider et al.")
        ax.fill_between(expected_x, expected_y - expected_std, expected_y + expected_std, color="black", alpha=0.12, linewidth=0)

    ax.set_xlabel("Wall shear stress [dyn/cm²]", fontsize=15)
    ax.set_ylabel(r"$\hat{E}^{final}$", fontsize=15)

    if str(validation_type).lower() == "free":
        ax.set_xscale("log")
        ax.set_xlim(0, 2000)

    ax.legend(fontsize=10, frameon=False)
    ax.grid(False)
    plt.tight_layout()

    out = os.path.join(base_dir, outname)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out}")

def plot_time_series(config, base_dir, plot_type, validation_type="free", Box_Lx=None, target_shear=None, time_step=None, a=0.03, anchor=0, force_window=1, monomer_index=None, last_fraction=0.6):
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("tab20").colors

    TIMESTEP = config["simulation"]["TIME_STEP"]
    STEPS = config["simulation"]["STEPS"]
    if time_step is not None:
        TIMESTEP = time_step
    dt = 1e-4 * TIMESTEP * STEPS

    geom = config.get("geometry", {})
    Lx = Box_Lx if Box_Lx is not None else geom.get("BOX_Lx", None)
    Ly = geom.get("BOX_Ly", None) if geom.get("BOX_Ly", None) is not None else Lx
    Lz = geom.get("BOX_Lz", None)

    shear_dirs = sorted([d for d in os.listdir(base_dir) if re.match(r"^shear_\d+$", d)], key=lambda s: int(s.split("_")[1]))
    if target_shear is not None:
        shear_dirs = [f"shear_{target_shear}"]

    vtype = str(validation_type).lower()

    if plot_type == "extension":
        fig, ax = plt.subplots(figsize=(6, 4))

        for i, sd in enumerate(shear_dirs):
            pos = np.load(os.path.join(base_dir, sd, "positions.npz"))["positions"]
            N = pos.shape[1]
            Lmax = N * 2 * a

            if vtype == "tethered":
                E = tethered_max_extension_ts(pos, anchor)
            elif vtype == "free":
                E = free_max_pairwise_extension_ts(pos, Lx, Ly, Lz)
            else:
                raise ValueError("validation_type must be 'free' or 'tethered'")

            # normalization
            E0 = E[0]
            den = Lmax - E0
            if den <= 0:
                print("den <= 0")
                den = np.finfo(float).eps
            y = (E - E0) / den

            t = np.arange(len(y)) * dt
            shear_rate = int(sd.split("_")[1])
            shear_stress_int = round(int(shear_rate * 0.11), -1)
            lbl = r"$\tau=$" + f" {shear_stress_int} " + r"dyn cm$^{-2}$"
            ax.plot(t, y, lw=1.5, color=colors[i % len(colors)], label=lbl)

        ax.set_xlabel("Time [s]", fontsize=15)
        ax.set_ylabel(r"$\hat{E}(t)$", fontsize=15)
        ax.axhline(1, color="k", linestyle="--")
        ax.legend(frameon=False, fontsize=10, ncol=2)
        ax.grid(False)
        plt.tight_layout()
        out = os.path.join(base_dir, "extension_time.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")
        return

    if plot_type != "force":
        raise ValueError("plot_type must be 'extension' or 'force'")

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, sd in enumerate(shear_dirs):
        F = np.load(os.path.join(base_dir, sd, "drag_forces.npz"))["forces"]
        Fmag = np.linalg.norm(F, axis=2) * 1000
        Fmag = moving_average_trailing_matrix(Fmag, force_window)
        t = np.arange(Fmag.shape[0]) * dt

        if monomer_index is not None:
            ax.plot(t, Fmag[:, int(monomer_index)], lw=1.5, color="grey", label=sd.split("_")[1])
        else:
            for m in range(Fmag.shape[1]):
                ax.plot(t, Fmag[:, m], lw=1.0, color="grey")

    ax.set_xlim(0,0.4)
    ax.set_xlabel("Time [s]", fontsize=15)
    ax.set_ylabel("Drag force [pN]", fontsize=15)
    ax.grid(False)
    plt.tight_layout()
    out = os.path.join(base_dir, "force_vs_time.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, sd in enumerate(shear_dirs):
        F = np.load(os.path.join(base_dir, sd, "drag_forces.npz"))["forces"]
        Fmag = np.linalg.norm(F, axis=2) * 1000
        Fmag = moving_average_trailing_matrix(Fmag, force_window)
        t = np.arange(Fmag.shape[0]) * dt

        n_last = int(np.ceil(last_fraction * Fmag.shape[0]))
        if n_last < 1:
            n_last = 1

        if monomer_index is not None:
            ax.plot(t[-n_last:], Fmag[-n_last:, int(monomer_index)], lw=1.5, color="grey", label=sd.split("_")[1])
        else:
            for m in range(Fmag.shape[1]):
                ax.plot(t[-n_last:], Fmag[-n_last:, m], lw=1.0, color="grey")

    ax.set_xlabel("Time [s]", fontsize=15)
    ax.set_ylabel("Drag force [pN]", fontsize=15)
    ax.grid(False)
    ax.set_xlim(0.15,0.4)
    # ax.set_title(f"Last {last_fraction:.2f} of trajectory", fontsize=13)
    plt.tight_layout()
    out_last = os.path.join(base_dir, f"force_vs_time_last_{last_fraction:.2f}.png")
    plt.savefig(out_last, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_last}")

def plot_tethered_across_LJ_for_gamma(config, base_dir, target_gamma, anchor=0, include_expected=True, outname=None, aggregate_runs=True, skip={}, N_mono=50, a=0.03, tail=20):
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("Dark2").colors

    dyn_visc_SI = config["physical"]["VISC_dyn"] / 10.0

    exp_x = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1280], float)
    exp_y = np.array([0.00, 0.35, 0.55, 0.63, 0.70, 0.75, 0.81, 0.85, 0.90, 0.92, 0.94, 0.96, 0.96, 0.96], float)

    if N_mono == 50:
        exp_s = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], float)
    elif N_mono == 100:
        exp_s = np.array([0.07, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], float)
    else:
        raise ValueError("N_mono must be 50 or 100")

    gpat = re.compile(r"^gamma_(\d{5})$")
    if isinstance(target_gamma, str) and gpat.match(target_gamma):
        gamma_folder = target_gamma
        gamma_val = int(gpat.match(target_gamma).group(1)) / 1e4
    else:
        gamma_val = float(target_gamma)
        gamma_folder = f"gamma_{int(round(gamma_val * 1e4)):05d}"

    lj_pat = re.compile(r"^LJ_(\d{5})$")

    skip_set = set(skip) if isinstance(skip, (set, list, tuple)) else set(skip.keys())
    lj_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and lj_pat.match(d) and d not in skip_set]
    lj_dirs.sort(key=lambda d: int(lj_pat.match(d).group(1)))

    def lj_label(lj):
        val = int(int(lj_pat.match(lj).group(1)) / 1e2)
        return f"{val} kT".rstrip("0").rstrip(".")

    fig, ax = plt.subplots(figsize=(8, 4))
    any_curve = False

    print(f"[INFO] Produce epsilon LJ validation plot for gamma {target_gamma}")
    for i, lj in enumerate(lj_dirs):
        gpath = os.path.join(base_dir, lj, gamma_folder)
        if not os.path.isdir(gpath):
            continue

        shear_dirs = sorted([d for d in os.listdir(gpath) if re.match(r"^shear_\d+$", d)], key=lambda s: int(s.split("_")[1]))
        if not shear_dirs:
            continue

        xs, ys_mean, ys_std = [], [], []
        run_counts = {}
        Lmax = None

        for sd in shear_dirs:
            shear_int = int(sd.split("_")[1])
            tau = shear_int * dyn_visc_SI * 10.0
            sdir = os.path.join(gpath, sd)

            run_dirs = sorted([d for d in os.listdir(sdir) if re.match(r"^run_\d+$", d)])
            run_paths = [os.path.join(sdir, r) for r in run_dirs] if run_dirs else [sdir]
            run_counts[tau] = len(run_paths)

            Evals = []
            for rp in run_paths:
                try:
                    pos = np.load(os.path.join(rp, "positions.npz"))["positions"]
                except:
                    print(f"No positions file for {rp}")
                    continue
                if Lmax is None:
                    N_here = pos.shape[1]
                    # max extension N_monomers * 2a with a monomer radius
                    Lmax = N_here * 2.0 * a
                # get the avg max extension at the tail of the timeseries per run
                Evals.append(tethered_tail_mean_max_extension(pos, anchor=anchor, tail=tail))
                # if shear_int==0:
                #     print(f"Polymer extension for shear=0 = {np.mean(Evals)}")

            xs.append(tau)
            ys_mean.append(np.mean(Evals))
            ys_std.append(np.std(Evals))

        xs = np.asarray(xs, float)
        ys_mean = np.asarray(ys_mean, float)
        ys_std = np.asarray(ys_std, float)

        order = np.argsort(xs)
        xs, ys_mean, ys_std = xs[order], ys_mean[order], ys_std[order]

        print(f"[INFO] {lj}/{gamma_folder}")
        for x in xs:
            print(f"    τ = {int(round(x))} dyn/cm^2 : {run_counts.get(x, 0)} runs")

        if Lmax is None:
            continue

        c = colors[i % len(colors)]
        lj_label_base = lj_label(lj)

        if aggregate_runs:
            E0_mean = ys_mean[xs == 0][0] if np.any(xs == 0) else ys_mean[0]
            print(f"Average coiled up extension is max_i(abs(E(m0) - E(mi))) = {round(E0_mean,2)}")
            denom = Lmax - E0_mean
            if denom <= 0 or not np.isfinite(denom):
                print(f"[WARN] Skipping {lj_label_base}")
                continue

            # normalization accounting for the finite size while coiled up (E0_mean)
            y_mean_norm = (ys_mean - E0_mean) / denom
            y_std_norm = ys_std / denom

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
                legend_label = r"$\epsilon_{LJ}=$" + f"{lj_label_base} " + r"($t^{2}_{14}$=" + f"{mse_scaled:.3f})"

            ax.errorbar(xs, y_mean_norm, yerr=y_std_norm, fmt="o-", lw=1.6, markersize=4, color=c, capsize=3, label=legend_label)
            any_curve = True
        else:
            E0_mean = ys_mean[xs == 0][0] if np.any(xs == 0) else ys_mean[0]
            denom = Lmax - E0_mean
            if denom <= 0 or not np.isfinite(denom):
                print(f"[WARN] Skipping {lj_label_base}")
                continue
            y_mean_norm = (ys_mean - E0_mean) / denom
            ax.plot(xs, y_mean_norm, "o-", lw=1.2, color=c, label=lj_label_base)
            any_curve = True

    if not any_curve:
        print(f"[INFO] No usable data for {gamma_folder}")
        return

    if include_expected:
        ax.plot(exp_x, exp_y, "--", color="grey", lw=1, label="Fu et al.")
        ax.fill_between(exp_x, exp_y - exp_s, exp_y + exp_s, color="grey", alpha=0.2)

    ax.axhline(1, color="k", linestyle="--")
    #ax.set_xlim(0, 1280)
    #ax.set_ylim(0, 1.2)
    ax.set_xlabel(r"Wall shear stress [dyn cm$^{-2}$]", fontsize=15)
    ax.set_ylabel(r"$\hat{E}^{final}_{uncoiling}$", fontsize=15)
    ax.legend(frameon=True, fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(False)
    plt.tight_layout()

    if outname is None:
        outname = f"tethered_extension_vs_shear_across_LJ_gamma_{int(round(gamma_val * 1e4)):05d}.png"

    outpath = os.path.join(base_dir, outname)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")

def plot_cumulative_force_across_gammas(config, lj_dir, target_shear, avg_window=100, start_from_right=30, use_last_fraction=0.25, skip=[]):
    """
    Plot cumulative tensile force vs monomer index for all gamma_* in lj_dir at a given shear,
    and compute a reduced chi2-like metric = mean((diff/sigma_ref)^2).
    """

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("Dark2").colors

    lj_match = re.search(r"LJ_(\d{5})", lj_dir)
    lj_text = f"Lennard Jones potential = {int(lj_match.group(1)) / 1e2:.2f} kT" if lj_match else ""

    gamma_dirs = sorted(
        [d for d in os.listdir(lj_dir) if os.path.isdir(os.path.join(lj_dir, d)) and re.match(r"^gamma_\d+$", d)],
        key=lambda g: int(g.split("_")[1]))

    blocks = {}
    N_detect = None
    skip_set = set(skip)

    for gdir in gamma_dirs:
        gval = int(gdir.split("_")[1]) / 1e4
        if gval in skip_set:
            print(f"[skip] {gval}")
            continue

        fpath = os.path.join(lj_dir, gdir, f"shear_{target_shear}", "drag_forces.npz")
        if not os.path.isfile(fpath):
            print(f"[skip] {fpath} not found")
            continue

        F = np.load(fpath)["forces"]
        if N_detect is None:
            N_detect = F.shape[1]
        N_use = min(N_detect, F.shape[1])

        Fmag = np.linalg.norm(F[:, :N_use, :], axis=2) * 1000.0
        Fm = moving_average_trailing_matrix(Fmag, avg_window)

        nL = max(int(len(Fm) * use_last_fraction), 1) if use_last_fraction is not None and 0 < use_last_fraction <= 1.0 else len(Fm)
        blocks[gval] = Fm[-nL:, :] if nL > 0 else Fm

    if not blocks or N_detect is None:
        print("No usable force data found for this LJ folder and shear.")
        return {}

    ref_per_mono = 0.5
    sigma_per_mono = 0.1

    idx = np.arange(N_detect)
    start = min(start_from_right, N_detect)

    steps_from_right = start - np.arange(1, start + 1) + 1
    ref = np.full(N_detect, np.nan)
    sigma_ref = np.full(N_detect, np.nan)
    ref[:start] = ref_per_mono * steps_from_right
    sigma_ref[:start] = sigma_per_mono * steps_from_right

    fig, ax = plt.subplots(figsize=(6, 4))
    mse_by_gamma = {}

    for i, (gval, block) in enumerate(sorted(blocks.items(), key=lambda kv: kv[0])):
        sub = block[:, :start][:, ::-1]
        cum = np.cumsum(sub, axis=1)[:, ::-1]
        mu = cum.mean(axis=0)
        sig = cum.std(axis=0)

        diff = mu - ref[:start]
        norm_mse = float(np.mean((diff / sigma_ref[:start]) ** 2))
        mse_by_gamma[gval] = norm_mse

        y = np.full(N_detect, np.nan)
        e = np.full(N_detect, np.nan)
        y[:start] = mu
        e[:start] = sig

        ax.errorbar(
            idx, y, yerr=e,
            marker="o", lw=1.2, elinewidth=1, capsize=3, markersize=3,
            color=colors[i % len(colors)],
            label=fr"$\gamma$={gval:.4f}" + r" ($t^{2}_{50}$=" + f"{round(norm_mse, 3)})")

    mask = ~np.isnan(ref)
    ax.plot(idx, ref, "--", lw=1.2, color="grey", label=r"Fu et al. $0.5 \pm 0.1$ pN/mono")
    ax.fill_between(idx[mask], ref[mask] - sigma_ref[mask], ref[mask] + sigma_ref[mask], alpha=0.25, color="grey")

    ax.set_xlabel("Monomer index", fontsize=15)
    ax.set_ylabel("Tensile force [pN]", fontsize=15)

    xtick_step = 5 if N_detect <= 60 else 10 if N_detect <= 150 else max(1, N_detect // 15)
    ax.set_xticks(np.arange(0, N_detect + 1, xtick_step))
    ax.set_xlim(-1, max(start, N_detect))

    ax.grid(False)
    ax.legend(frameon=False, fontsize=9, ncol=1)
    plt.tight_layout()

    out = os.path.join(lj_dir, f"force_cumulative_vs_monomer_all_gamma_shear_{target_shear}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    print("== Normalized MSE per gamma ==")
    for gval in sorted(mse_by_gamma):
        print(f"  gamma={gval:.4f}: " + r"$\chi^{2}_{14}$" + f" = {mse_by_gamma[gval]:.6g}")

    return mse_by_gamma

def plot_cumulative_force_across_gammas_aggregate(config, lj_dir, target_shear,
                                                  avg_window=100, time_avg_steps=100,
                                                  start_from_right=30,
                                                  monomer_index=50,
                                                  plot_timeseries=True,
                                                  dt_ms=0.1,
                                                  skip=[]):
    """
    Plot cumulative tensile force vs monomer index for all gamma_* in lj_dir at a given shear.

    Behavior:
      - gamma_*/shear_{target_shear}/run_*/drag_forces.npz
      - per run:
          * force magnitude
          * trailing moving average (avg_window)
          * time average over last time_avg_steps saved points
      - error bars = std across runs
      - additionally:
          * per-gamma overlay plot of full timeseries (all runs) with x-axis in time (ms)
          * per-gamma overlay plot of last time_avg_steps only with x-axis in time (ms)
    """

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("Dark2").colors

    gamma_dirs = sorted([d for d in os.listdir(lj_dir)
                          if os.path.isdir(os.path.join(lj_dir, d)) and re.match(r"^gamma_\d+$", d)],
                         key=lambda g: int(g.split("_")[1]))

    skip_set = set(skip)
    runs_timeavg_by_gamma = {}
    N_detect = None

    for gdir in gamma_dirs:
        gval = int(gdir.split("_")[1]) / 1e4
        if gval in skip_set:
            print(f"[skip] {gval}")
            continue

        shear_dir = os.path.join(lj_dir, gdir, f"shear_{target_shear}")
        if not os.path.isdir(shear_dir):
            print(f"[skip] {shear_dir} not found")
            continue

        run_dirs = sorted([d for d in os.listdir(shear_dir)
                            if os.path.isdir(os.path.join(shear_dir, d)) and re.match(r"^run_\d+$", d)])

        if not run_dirs:
            print(f"[skip] no run_* folders in {shear_dir}")
            continue

        per_run_values = []
        ts_full = []
        ts_last = []

        for rdir in run_dirs:
            fpath = os.path.join(shear_dir, rdir, "drag_forces.npz")
            if not os.path.isfile(fpath):
                print(f"[skip] {fpath} not found")
                continue

            F = np.load(fpath)["forces"]
            if F.ndim != 3 or F.shape[0] < 1:
                print(f"[skip] unexpected forces shape {F.shape} in {fpath}")
                continue

            if N_detect is None:
                N_detect = F.shape[1]
                print(f"Running for N={N_detect} monomers")

            N_use = min(N_detect, F.shape[1])
            if monomer_index >= N_use:
                print(f"[skip] monomer_index={monomer_index} out of range for {fpath}")
                continue

            Fmag = np.linalg.norm(F[:, :N_use, :], axis=2) * 1000.0
            Fm = moving_average_trailing_matrix(Fmag, avg_window)

            nT = Fm.shape[0]
            n_avg = min(time_avg_steps, nT)

            if plot_timeseries:
                ts_full.append(Fm[:, monomer_index].copy())
                ts_last.append(Fm[-n_avg:, monomer_index].copy())

            per_run = Fm[-n_avg:, :].mean(axis=0)
            per_run_values.append(per_run)

        if len(per_run_values) == 0:
            print(f"[skip] no usable runs for gamma {gval:.4f}")
            continue

        runs_timeavg_by_gamma[gval] = np.vstack(per_run_values)

        if plot_timeseries and len(ts_full) > 0:
            fig_ts, ax_ts = plt.subplots(figsize=(6, 4))

            for ts in ts_full:
                t = np.arange(len(ts)) * dt_ms
                ax_ts.plot(t, ts, lw=1.0, alpha=0.35, color="purple")

            minT = min(len(ts) for ts in ts_full)
            if minT > 1:
                ts_stack = np.vstack([ts[:minT] for ts in ts_full])
                t_mean = np.arange(minT) * dt_ms
                ax_ts.plot(t_mean, ts_stack.mean(axis=0), lw=2.0, color="black", label="mean")

            ax_ts.set_xlabel("Time [ms]", fontsize=13)
            ax_ts.set_xlim(0,200)
            ax_ts.set_ylabel(r"$|F^{\mathrm{drag}}|$ [pN]", fontsize=13)
            #ax_ts.set_title(fr"$\gamma={gval:.4f}$, monomer {monomer_index}", fontsize=11)
            ax_ts.grid(False)
            ax_ts.legend(frameon=False, fontsize=9)

            plt.tight_layout()
            out_ts = os.path.join(shear_dir, f"drag_force_timeseries_all_runs_m{monomer_index}.png")
            plt.savefig(out_ts, dpi=300, bbox_inches="tight")
            plt.close(fig_ts)
            print(f"Saved {out_ts}")

            fig_ts2, ax_ts2 = plt.subplots(figsize=(6, 4))

            # determine absolute start index from full timeseries length
            T_full = max(len(ts) for ts in ts_full)

            for ts in ts_last:
                n = len(ts)
                t = (np.arange(n) + T_full - n) * dt_ms
                ax_ts2.plot(t, ts, lw=1.0, alpha=0.35, color="purple")

            minT2 = min(len(ts) for ts in ts_last)
            if minT2 > 1:
                ts_stack2 = np.vstack([ts[-minT2:] for ts in ts_last])
                t_mean2 = (np.arange(minT2) + T_full - minT2) * dt_ms
                ax_ts2.plot(t_mean2, ts_stack2.mean(axis=0), lw=2.0, color="black", label="mean")
            ax_ts2.set_xlim(np.min(t_mean2), 200)
            ax_ts2.set_xlabel("Time [ms]", fontsize=13)
            ax_ts2.set_ylabel(r"$|F^{\mathrm{drag}}|$ [pN]", fontsize=13)
            #ax_ts2.set_title(fr"$\gamma={gval:.4f}$, monomer {monomer_index} (last window)", fontsize=11)
            ax_ts2.grid(False)
            ax_ts2.legend(frameon=False, fontsize=9)

            plt.tight_layout()
            out_ts2 = os.path.join(shear_dir, f"drag_force_timeseries_last{time_avg_steps}_m{monomer_index}.png")
            plt.savefig(out_ts2, dpi=300, bbox_inches="tight")
            plt.close(fig_ts2)
            print(f"Saved {out_ts2}")


    if not runs_timeavg_by_gamma or N_detect is None:
        print("No usable force data found for this LJ folder and shear.")
        return {}

    ref_per_mono = 0.5
    sigma_per_mono = 0.1

    idx = np.arange(N_detect)
    start = min(start_from_right, N_detect)

    steps_from_right = start - np.arange(1, start + 1) + 1
    ref = np.full(N_detect, np.nan)
    sigma_ref = np.full(N_detect, np.nan)
    ref[:start] = ref_per_mono * steps_from_right
    sigma_ref[:start] = sigma_per_mono * steps_from_right

    fig, ax = plt.subplots(figsize=(6, 4))
    mse_by_gamma = {}

    for i, (gval, run_mat) in enumerate(sorted(runs_timeavg_by_gamma.items(), key=lambda kv: kv[0])):
        sub = run_mat[:, :start][:, ::-1]
        cum = np.cumsum(sub, axis=1)[:, ::-1]

        mu = cum.mean(axis=0)
        sig = cum.std(axis=0, ddof=1) if cum.shape[0] > 1 else np.zeros_like(mu)

        diff = mu - ref[:start]
        norm_mse = float(np.mean((diff / sigma_ref[:start]) ** 2))
        mse_by_gamma[gval] = norm_mse

        y = np.full(N_detect, np.nan)
        e = np.full(N_detect, np.nan)
        y[:start] = mu
        e[:start] = sig

        ax.errorbar(idx, y, yerr=e,
                    marker="o", lw=1.2, elinewidth=0.5, capsize=2, markersize=2,
                    color=colors[i % len(colors)],
                    label=fr"$\gamma$={gval:.4f} ($t^2_{{100}}$={round(norm_mse, 3)})")

    mask = ~np.isnan(ref)
    ax.plot(idx, ref, "--", lw=1.2, color="grey", label=r"Fu et al. $0.5 \pm 0.1$ pN/mono")
    ax.fill_between(idx[mask], ref[mask] - sigma_ref[mask], ref[mask] + sigma_ref[mask], alpha=0.25, color="grey")

    ax.set_xlabel("Monomer index", fontsize=15)
    ax.set_ylabel("Tensile force [pN]", fontsize=15)

    xtick_step = 5 if N_detect <= 60 else 10 if N_detect <= 150 else max(1, N_detect // 15)
    ax.set_xticks(np.arange(0, N_detect + 1, xtick_step))
    ax.set_xlim(-1, max(start, N_detect))

    ax.grid(False)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    out = os.path.join(lj_dir, f"force_cumulative_vs_monomer_all_gamma_shear_{target_shear}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    print("== Normalized MSE per gamma ==")
    for gval in sorted(mse_by_gamma):
        print(f"  gamma={gval:.4f}: chi^2 = {mse_by_gamma[gval]:.6g}")

    return mse_by_gamma

def plot_recoiling_across_LJ_for_gamma(config, root_dir, target_gamma, N_mono, time_step=None, add_expected=True, aggregate_runs=True, show_individual_when_agg=False, skip=None, t_fit_max=0.4, t_fit_min=None):
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("Dark2").colors
    skip_set = set(skip or [])

    TIMESTEP = config["simulation"]["TIME_STEP"]
    STEPS = config["simulation"]["STEPS"]

    if time_step is not None:
        TIMESTEP = time_step

    dt = 1e-4 * TIMESTEP * STEPS

    if not isinstance(target_gamma, str) or not re.match(r"^gamma_\d{5}$", target_gamma):
        raise ValueError("gamma_value must be a string like 'gamma_00110'")

    gamma_folder = target_gamma

    if N_mono == 50:
        expected_x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        expected_y = np.array([1, 0.4, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        expected_err = np.array([0.1, 0.3, 0.2, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
        theoretical_R = 0.0
        ylim = (-0.1, 1.0)
    elif N_mono == 100:
        expected_x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        expected_y = np.array([1, 0.55, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1])
        expected_err = np.array([0.1, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1])
        theoretical_R = 0.0
        ylim = (-0.05, 1.0)
    else:
        raise ValueError("N_mono must be 50 or 100")

    lj_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and re.match(r"^LJ_\d{5}$", d)]
    lj_dirs.sort(key=lambda d: int(d.split("_")[1]))

    def reduced_t2(t, y_mean, y_std, x_ref, y_ref, err_ref, t_fit_max=0.4, t_fit_min=None):
        mask = t <= t_fit_max
        if t_fit_min is not None:
            mask &= t >= t_fit_min
        mask &= (t >= x_ref[0]) & (t <= x_ref[-1])
        if not np.any(mask):
            return np.inf, 0
        t_fit = t[mask]
        y_fit = y_mean[mask]
        ystd_fit = y_std[mask]
        yref_i = np.interp(t_fit, x_ref, y_ref)
        errref_i = np.interp(t_fit, x_ref, err_ref)
        sig = np.maximum(errref_i + ystd_fit, 1e-12)
        return float(np.mean(((y_fit - yref_i) / sig) ** 2)), int(t_fit.size)

    fig, ax = plt.subplots(figsize=(6, 4))
    t2_by_lj = {}
    plotted = 0

    for lj in lj_dirs:
        lj_kT = int(lj.split("_")[1]) / 1e2
        if int(lj_kT) in skip_set:
            continue

        gamma_path = os.path.join(root_dir, lj, gamma_folder)
        if not os.path.isdir(gamma_path):
            print(f"[INFO] {lj}/{gamma_folder} : no gamma folder")
            continue

        single_file = os.path.join(gamma_path, "positions.npz")
        run_dirs = sorted([d for d in os.listdir(gamma_path) if re.match(r"^run_\d+$", d)])

        files = []
        if os.path.isfile(single_file):
            files.append(single_file)
        files += [os.path.join(gamma_path, r, "positions.npz") for r in run_dirs if os.path.isfile(os.path.join(gamma_path, r, "positions.npz"))]

        if not files:
            print(f"[INFO] {lj}/{gamma_folder} : no positions.npz file")
            continue

        series = []
        for fpos in files:
            pos = np.load(fpos)["positions"]
            E = tethered_max_extension_ts(pos, anchor=0)
            if E[0] <= 0:
                continue
            series.append((E - theoretical_R) / (E[0] - theoretical_R) if "/run_" in fpos else E / E[0])

        if not series:
            continue

        min_len = min(len(s) for s in series)
        Y = np.stack([s[:min_len] for s in series])
        t = np.arange(min_len) * dt

        color = colors[plotted % len(colors)]
        lj_label = f"{int(lj_kT)}"
        plotted += 1

        if aggregate_runs and Y.shape[0] > 1:
            y_mean = Y.mean(axis=0)
            y_std = Y.std(axis=0, ddof=1)
            if show_individual_when_agg:
                for r in range(Y.shape[0]):
                    ax.plot(t, Y[r], lw=0.8, alpha=0.25, color=color)
            t2, n_used = reduced_t2(t, y_mean, y_std, expected_x, expected_y, expected_err, t_fit_max=t_fit_max, t_fit_min=t_fit_min)
            t2_by_lj[lj_label] = (t2, n_used)
            ax.plot(t, y_mean, lw=1.8, color=color, label=fr"$\epsilon_{{LJ}}={lj_label}$ $(t^2={t2:.3f},\,n={n_used})$")
            ax.fill_between(t, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)
        else:
            t2_vals = []
            n_used_any = 0
            for r in range(Y.shape[0]):
                ax.plot(t, Y[r], lw=1.2, color=color, alpha=0.9)
                t2_r, n_used = reduced_t2(t, Y[r], np.zeros_like(Y[r]), expected_x, expected_y, expected_err, t_fit_max=t_fit_max, t_fit_min=t_fit_min)
                if np.isfinite(t2_r):
                    t2_vals.append(t2_r)
                    n_used_any = max(n_used_any, n_used)
            if t2_vals:
                t2 = float(np.mean(t2_vals))
                t2_by_lj[lj_label] = (t2, n_used_any)
                ax.plot([], [], lw=1.8, color=color, label=fr"$\epsilon_{{LJ}}={lj_label}$ $(t^2={t2:.3f},\,n={n_used_any})$")

    if add_expected:
        ax.plot(expected_x, expected_y, "--", lw=1.2, color="grey", label="Fu et al.")
        ax.fill_between(expected_x, expected_y - expected_err, expected_y + expected_err, alpha=0.25, color="grey")

    if t_fit_max is not None:
        ax.axvline(t_fit_max, ls=":", lw=1.0, color="k", alpha=0.5)

    ax.set_xlim(0, 0.8)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Time [s]", fontsize=13)
    ax.set_ylabel(r"$\hat{E}_{\mathrm{recoiling}}$", fontsize=13)
    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.grid(False)
    plt.tight_layout()

    out = os.path.join(root_dir, f"recoiling_all_LJ_gamma_{gamma_folder.split('_')[1]}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    return {k: v[0] for k, v in sorted(t2_by_lj.items(), key=lambda kv: kv[1][0])}
