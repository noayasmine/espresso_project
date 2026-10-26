import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# I/O helpers
# ============================================================

def load_npz_first(path):
    d = np.load(path)
    return d[d.files[0]]


# ============================================================
# Settings
# ============================================================

# shear = 1454
# shear = 3636
shear = 5455
# shear = 2727

RUNS     = [0, 1, 2, 3, 4]
RUN_BASE = "/home/nroebersen/espresso_project/experiments/density_experiment"

AREA                   = 72.0
USE_DENSITY_X          = True
N_LAST                 = 200
N_MOV                  = 1
INCLUDE_NS             = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
N_MONOMERS_PER_POLYMER = 50.0

# Timeseries window: None = full series; integers = timestep indices
T1 = None
T2 = None
DT = 1e-4       # timestep → seconds

THRESHOLD       = 20
THRESHOLDS_2COL = [20.0, 27.5]

SHEARS_TO_INCLUDE = [1454, 1818, 2727, 3636, 5455]

SHEAR_TO_STRESS = {
    1454: 160,
    2091: 230,
    1818: 200,
    2727: 300,
    3636: 400,
    5455: 600,
}

# ============================================================
# Style
# ============================================================

COL_A = "purple"      # selfassociation
COL_B = "cadetblue"   # no selfassociation
ALPHA = 0.7

COND_STYLE = [
    ("selfassociation",    COL_A, "o", "selfassociation"),
    ("no_selfassociation", COL_B, "s", "no selfassociation"),
]

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, length=4)

TAB10 = plt.get_cmap("tab10").colors


# ============================================================
# Path helpers
# ============================================================

def run_base_dir(run, shear_val, condition):
    return os.path.join(RUN_BASE, f"run_{run}", str(shear_val), condition)

def run_folder(run, shear_val, condition, Npol):
    return os.path.join(run_base_dir(run, shear_val, condition), f"N_polymers_{Npol}")

def all_Npols_for(shear_val, condition):
    """Union of N_polymers values found across all runs for a shear/condition."""
    Npols = set()
    for run in RUNS:
        bd = run_base_dir(run, shear_val, condition)
        if not os.path.isdir(bd):
            continue
        for name in os.listdir(bd):
            if not name.startswith("N_polymers_"):
                continue
            Npol = int(name.split("_")[-1])
            if INCLUDE_NS is not None and Npol not in INCLUDE_NS:
                continue
            Npols.add(Npol)
    return sorted(Npols)

def intersect_Ns(A, B):
    """N_polymers values present in both condition dicts."""
    return sorted(set(A["tensile_mu_ma"]) & set(B["tensile_mu_ma"]))


# ============================================================
# Core computation helpers
# ============================================================

def trailing_moving_avg_axis0(arr_TN, window):
    arr = np.asarray(arr_TN, dtype=float)
    T, N = arr.shape
    out  = np.empty_like(arr)
    for t in range(T):
        s = max(0, t - window + 1)
        out[t] = np.nanmean(arr[s:t+1], axis=0)
    return out

def slice_ts(series, t1, t2):
    """Slice a 1-D array to [t1:t2] (None = no limit). Returns (sliced, i1, i2)."""
    i1 = 0              if t1 is None else t1
    i2 = len(series)    if t2 is None else min(t2, len(series))
    return series[i1:i2], i1, i2

def _stack_mean_std(series_list):
    """Truncate to shortest, return (mean, std) across runs."""
    if not series_list:
        return None, None
    L   = min(len(s) for s in series_list)
    arr = np.array([s[:L] for s in series_list], dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


# ============================================================
# Per-run scalar collectors
# ============================================================

def _assoc_scalar_per_run(shear_val, condition, Npol):
    vals = []
    for run in RUNS:
        path = os.path.join(run_folder(run, shear_val, condition, Npol),
                            "added_bonds.npz")
        if not os.path.exists(path):
            continue
        ts = load_npz_first(path).astype(float)
        vals.append(np.nansum(ts) / Npol)
    return vals

def _assoc_frac_per_run(shear_val, condition, Npol):
    return [v / N_MONOMERS_PER_POLYMER
            for v in _assoc_scalar_per_run(shear_val, condition, Npol)]

def _tensile_mean_per_run(shear_val, condition, Npol):
    vals = []
    for run in RUNS:
        path = os.path.join(run_folder(run, shear_val, condition, Npol),
                            "tensile_forces.npz")
        if not os.path.exists(path):
            continue
        tens = load_npz_first(path).astype(float)
        last = min(N_LAST, tens.shape[0])
        vals.append(np.mean(tens[-last:]) * 1000.0)   # pN
    return vals

def _frac_above_per_run(shear_val, condition, Npol, threshold):
    vals = []
    for run in RUNS:
        path = os.path.join(run_folder(run, shear_val, condition, Npol),
                            "tensile_forces.npz")
        if not os.path.exists(path):
            continue
        tens  = load_npz_first(path).astype(float) * 1000.0   # pN
        ma    = trailing_moving_avg_axis0(tens, N_MOV)
        last  = min(N_LAST, ma.shape[0])
        ma_mn = np.nanmean(ma[-last:], axis=0)
        vals.append(np.sum(ma_mn > threshold) / tens.shape[1])
    return vals


# ============================================================
# Density-sweep aggregator
# ============================================================

def _density_sweep(shear_val, condition, scalar_fn, **kwargs):
    """
    For each Npol collect per-run scalars → (densities, means, stds).
    std = population std of the data across runs.
    """
    Npols = all_Npols_for(shear_val, condition)
    dens, means, stds = [], [], []
    for Npol in Npols:
        vals = scalar_fn(shear_val, condition, Npol, **kwargs)
        if not vals:
            continue
        dens.append(Npol / AREA)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    order = np.argsort(dens)
    return (np.array(dens)[order],
            np.array(means)[order],
            np.array(stds)[order])

def collect_assoc_density_sweep(shear_val, condition):
    return _density_sweep(shear_val, condition, _assoc_scalar_per_run)

def collect_assoc_frac_density_sweep(shear_val, condition):
    return _density_sweep(shear_val, condition, _assoc_frac_per_run)

def collect_tensile_density_sweep(shear_val, condition):
    return _density_sweep(shear_val, condition, _tensile_mean_per_run)

def collect_frac_above_density_sweep(shear_val, condition, threshold=THRESHOLD):
    return _density_sweep(shear_val, condition, _frac_above_per_run,
                          threshold=threshold)


# ============================================================
# Timeseries aggregators
# ============================================================

def added_bonds_timeseries(shear_val, condition, Npol):
    """Mean & std of added_bonds (T,) across runs."""
    series = []
    for run in RUNS:
        path = os.path.join(run_folder(run, shear_val, condition, Npol),
                            "added_bonds.npz")
        if os.path.exists(path):
            series.append(load_npz_first(path).astype(float))
    return _stack_mean_std(series)

def tensile_ma_timeseries(shear_val, condition, Npol):
    """Mean & std of moving-average tensile force timeseries across runs."""
    series = []
    for run in RUNS:
        path = os.path.join(run_folder(run, shear_val, condition, Npol),
                            "tensile_forces.npz")
        if not os.path.exists(path):
            continue
        tens  = load_npz_first(path).astype(float)
        avg_t = np.nanmean(tens, axis=1) * 1000.0   # pN, shape (T,)
        ma    = np.array([avg_t[max(0, i-N_MOV+1):i+1].mean()
                          for i in range(len(avg_t))])
        series.append(ma)
    return _stack_mean_std(series)


# ============================================================
# collect_one_condition  –  builds unified data dict for one
#   shear / condition, used by all single-shear plots.
# ============================================================

def collect_one_condition(shear_val, condition, label):
    Npols = all_Npols_for(shear_val, condition)

    dens_a, assoc_mean, assoc_std = collect_assoc_density_sweep(shear_val, condition)
    dens_t, t_mean,    t_std     = collect_tensile_density_sweep(shear_val, condition)

    x = dens_t if USE_DENSITY_X else dens_t * AREA

    tensile_mu_ma, tensile_sd_ma = {}, {}
    added_mu,      added_sd      = {}, {}
    for Npol in Npols:
        mu, sd = tensile_ma_timeseries(shear_val, condition, Npol)
        if mu is not None:
            tensile_mu_ma[Npol] = mu
            tensile_sd_ma[Npol] = sd
        mu2, sd2 = added_bonds_timeseries(shear_val, condition, Npol)
        if mu2 is not None:
            added_mu[Npol] = mu2
            added_sd[Npol] = sd2

    return {
        "label":         label,
        "condition":     condition,
        "x":             x,
        "dens_assoc":    dens_a,
        "assoc_mean":    assoc_mean,
        "assoc_std":     assoc_std,
        "t_mean":        t_mean,
        "t_std":         t_std,
        "tensile_mu_ma": tensile_mu_ma,
        "tensile_sd_ma": tensile_sd_ma,
        "added_mu":      added_mu,
        "added_sd":      added_sd,
    }


# ============================================================
# ============================================================
#  SINGLE-SHEAR PLOTS  (plots 1-4)
# ============================================================
# ============================================================

out_single = os.path.join(RUN_BASE, str(shear))
os.makedirs(out_single, exist_ok=True)

xlabel = (r"Polymer density [1/$\mu$m$^2$]" if USE_DENSITY_X
          else r"Number of polymers")

# Collect both conditions for this shear rate
A = collect_one_condition(shear, "selfassociation",    "selfassociation")
B = collect_one_condition(shear, "no_selfassociation", "no selfassociation")

# ----------------------------------------------------------
# Plot 1: self-associations per polymer vs density
#         selfassociation only – no_selfassociation has no added_bonds
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 3))
ax.errorbar(A["dens_assoc"], A["assoc_mean"], yerr=A["assoc_std"],
            fmt="o-", lw=2.0, ms=5, capsize=3, color=COL_A, alpha=ALPHA,
            label=A["label"])
ax.axhline(50, lw=1.5, ls="--", color=COL_B, alpha=ALPHA,
           label="max associations")
ax.set_xlabel(xlabel, size=10)
ax.set_ylabel(f"Avg self-associations per polymer\n(last {N_LAST} timesteps)", size=10)
ax.set_xlim(0, 3.0)
ax.legend(frameon=False)
style_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(out_single, "self_associations_total_vs_x.png"), dpi=300)
plt.close(fig)

# ----------------------------------------------------------
# Plot 2: tensile force vs density  – BOTH conditions overlay
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 3))
ax.errorbar(A["x"], A["t_mean"], yerr=A["t_std"],
            fmt="o-", lw=2.0, ms=5, capsize=3, color=COL_A, alpha=ALPHA,
            label=A["label"])
ax.errorbar(B["x"], B["t_mean"], yerr=B["t_std"],
            fmt="s-", lw=2.0, ms=5, capsize=3, color=COL_B, alpha=ALPHA,
            label=B["label"])
ax.set_xlabel(xlabel, size=10)
ax.set_ylabel("Avg tensile force [pN]", size=10)
ax.set_xlim(0, 3.0)
ax.legend(frameon=False)
style_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(out_single, "tensile_force_stats_vs_x.png"), dpi=300)
plt.close(fig)

# ----------------------------------------------------------
# Plot 3: added bonds per timestep – mean ± std shaded
#         selfassociation only (no_selfassociation has no added_bonds)
# ----------------------------------------------------------
Npols_A = sorted(A["added_mu"])
fig, ax  = plt.subplots(figsize=(5, 3))
for i, Npol in enumerate(Npols_A):
    mu_sl, i1, i2 = slice_ts(A["added_mu"][Npol], T1, T2)
    sd_sl, _,  _  = slice_ts(A["added_sd"][Npol], T1, T2)
    t = np.arange(i1, i2) * DT
    color = TAB10[i % len(TAB10)]
    ax.plot(t, mu_sl, lw=1.5, alpha=ALPHA, color=color, label=f"N={Npol}")
    ax.fill_between(t, mu_sl - sd_sl, mu_sl + sd_sl, alpha=0.15, color=color)
ax.set_xlabel("Time (s)", size=10)
ax.set_ylabel("Added self-associations per timestep\n(mean ± std across runs)", size=10)
ax.legend(frameon=False, fontsize=8, ncol=2)
style_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(out_single,
                         "self_associations_per_timestep_timeseries.png"), dpi=300)
plt.close(fig)

# ----------------------------------------------------------
# Plot 4: tensile MA over time – stacked per Npol,
#         BOTH conditions overlaid, mean ± std shaded
# ----------------------------------------------------------
Ns_common = intersect_Ns(A, B)
if Ns_common:
    fig, axes = plt.subplots(len(Ns_common), 1,
                             figsize=(7.5, 1.8 * len(Ns_common)),
                             sharex=True)
    if len(Ns_common) == 1:
        axes = [axes]

    for ax, Npol in zip(axes, Ns_common):
        muA_sl, i1A, i2A = slice_ts(A["tensile_mu_ma"][Npol], T1, T2)
        sdA_sl, _,   _   = slice_ts(A["tensile_sd_ma"][Npol], T1, T2)
        muB_sl, i1B, i2B = slice_ts(B["tensile_mu_ma"][Npol], T1, T2)
        sdB_sl, _,   _   = slice_ts(B["tensile_sd_ma"][Npol], T1, T2)

        tA = np.arange(i1A, i2A) * DT
        tB = np.arange(i1B, i2B) * DT

        ax.plot(tA, muA_sl, color=COL_A, lw=2.0, alpha=ALPHA, label=A["label"])
        ax.fill_between(tA, muA_sl - sdA_sl, muA_sl + sdA_sl,
                        color=COL_A, alpha=0.20, linewidth=0)
        ax.plot(tB, muB_sl, color=COL_B, lw=2.0, alpha=ALPHA, label=B["label"])
        ax.fill_between(tB, muB_sl - sdB_sl, muB_sl + sdB_sl,
                        color=COL_B, alpha=0.20, linewidth=0)

        dens = Npol / AREA
        ax.set_ylabel(f"N={Npol}\nρ={dens:.3f} " + r"($1/\mu$m$^2$)", size=9)
        style_axes(ax)

    axes[-1].set_xlabel("Time (s)", size=10)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_single,
                             "tensile_force_avg_over_time_subplots.png"), dpi=300)
    plt.close(fig)


# ============================================================
# ============================================================
#  MULTI-SHEAR PLOTS  (plots 5-8)
# ============================================================
# ============================================================

out_global = RUN_BASE
os.makedirs(out_global, exist_ok=True)

markers    = ["o", "s", "^", "D", "v", "P", "X"]
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]

# ----------------------------------------------------------
# Plot 5: cumulative assoc fraction vs density (all shears)
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for i, sh in enumerate(SHEARS_TO_INCLUDE):
    dens, means, stds = collect_assoc_frac_density_sweep(sh, "selfassociation")
    if len(dens) == 0:
        continue
    stress_lbl = SHEAR_TO_STRESS.get(sh, sh)
    ax.errorbar(dens, means, yerr=stds,
                fmt=markers[i % len(markers)],
                ls=linestyles[i % len(linestyles)],
                lw=2.0, ms=5, capsize=3,
                color=TAB10[i % len(TAB10)], alpha=ALPHA,
                label=rf"$\tau= {stress_lbl}\,\mathrm{{dyn\,cm^{{-2}}}}$")
ax.set_xlabel(r"Polymer density [1/$\mu$m$^2$]", size=10)
ax.set_ylabel("Fraction of self-associated monomers", size=10)
ax.set_xlim(0, 3)
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.15))
style_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(out_global,
                         "assoc_fraction_vs_density_all_shears.png"), dpi=300)
plt.close(fig)


# ============================================================
# F > threshold figures  (plots 6-8)
# ============================================================

SHEARS_2x2 = [1818, 2727, 3636, 5455]

def _both_conditions_exist(sh):
    for run in RUNS:
        if (os.path.isdir(run_base_dir(run, sh, "selfassociation")) and
                os.path.isdir(run_base_dir(run, sh, "no_selfassociation"))):
            return True
    return False

shears_ok = [sh for sh in SHEARS_2x2 if _both_conditions_exist(sh)][:4]

# ----------------------------------------------------------
# Plot 6: 2×2 – fraction of monomers with F > THRESHOLD
# ----------------------------------------------------------
# ----------------------------------------------------------
# Plot 6: 2×2 – fraction of monomers with F > THRESHOLD
# ----------------------------------------------------------
if shears_ok:
    fig, axes = plt.subplots(2, 2, figsize=(6, 4),
                             sharex=True, sharey=False, dpi=300)
    axes = axes.ravel()
    for j, ax in enumerate(axes):
        if j >= len(shears_ok):
            ax.axis("off")
            continue
        sh = shears_ok[j]
        for condition, color, marker, label in COND_STYLE:
            dens, means, stds = collect_frac_above_density_sweep(sh, condition, THRESHOLD)
            if len(dens) == 0:
                continue
            ax.errorbar(dens, means, yerr=stds,
                        fmt=f"{marker}-", lw=2.0, ms=5, capsize=3,
                        color=color, alpha=ALPHA, label=label)

        # ── right y-axis label instead of in-figure text ──────────────
        stress_lbl = SHEAR_TO_STRESS.get(sh, sh)
        ax_r = ax.twinx()
        ax_r.set_yticks([])                          # no tick marks
        ax_r.set_ylabel(
            rf"$\tau = {stress_lbl}\,\mathrm{{dyn~cm^{{-2}}}}$",
            fontsize=11, labelpad=8
        )
        # ───────────────────────────────────────────────────────────────

        ax.set_xlim(0, 3)
        ax.set_ylim(0, None)
        style_axes(ax)

    fig.supxlabel(r"Polymer density [1/$\mu$m$^2$]")
    fig.supylabel(fr"Monomer fraction with $F > {THRESHOLD}$ pN")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_global,
                f"frac_monomers_F_gt_{int(THRESHOLD)}pN_all_shears_2x2.png"),
                dpi=300)
    plt.close(fig)

# ----------------------------------------------------------
# Plot 7: stacked subplots – fraction above THRESHOLD
# ----------------------------------------------------------
if shears_ok:
    fig, axes = plt.subplots(len(shears_ok), 1,
                             figsize=(5.5/1.5, 1.8*len(shears_ok)/1.5),
                             sharex=True, sharey=False)
    if len(shears_ok) == 1:
        axes = [axes]

    for ax, sh in zip(axes, shears_ok):
        for condition, color, marker, label in COND_STYLE:
            dens, means, stds = collect_frac_above_density_sweep(sh, condition, THRESHOLD)
            if len(dens) == 0:
                continue
            ax.errorbar(dens, means, yerr=stds,
                        fmt=f"{marker}-", lw=2.0, ms=5, capsize=3,
                        color=color, alpha=ALPHA, label=label)
        stress_lbl = SHEAR_TO_STRESS.get(sh, sh)
        ax.text(1.08, 0.5,
                rf"$\tau = {stress_lbl}$" "\n" r"$\mathrm{dyn\,cm^{-2}}$",
                transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=11)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, None)
        style_axes(ax)

    axes[-1].set_xlabel(r"Polymer density [1/$\mu$m$^2$]", size=10)
    fig.supylabel(fr"Monomer fraction with $F > {int(THRESHOLD)}$ pN")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_global,
                             f"frac_monomers_F_gt_{int(THRESHOLD)}pN_all_shears_stacked.png"),
                dpi=300)
    plt.close(fig)

# ----------------------------------------------------------
# Plot 8: 2-threshold × N-shears grid
# ----------------------------------------------------------
if shears_ok:
    nrows = len(shears_ok)
    ncols = len(THRESHOLDS_2COL)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5/1.5, 1.8*nrows/1.5),
                             sharex=True, sharey=False)
    if nrows == 1:
        axes = np.array([axes])

    for i in range(nrows):
        axes[i, 1].sharey(axes[i, 0])

    for i, sh in enumerate(shears_ok):
        for j, thr in enumerate(THRESHOLDS_2COL):
            ax = axes[i, j]
            for condition, color, marker, label in COND_STYLE:
                dens, means, stds = collect_frac_above_density_sweep(sh, condition, thr)
                if len(dens) == 0:
                    continue
                ax.errorbar(dens, means, yerr=stds,
                            fmt=f"{marker}-", lw=2.0, ms=5, capsize=3,
                            color=color, alpha=ALPHA, label=label)
            ax.set_xlim(0, 3)
            ax.set_ylim(0, None)
            style_axes(ax)
            if i == 0:
                ax.set_title(fr"$F > {thr}$ pN", fontsize=11)

        stress_lbl = SHEAR_TO_STRESS.get(sh, sh)
        axes[i, -1].text(1.08, 0.5,
                         rf"$\tau = {stress_lbl}$" "\n" r"$\mathrm{dyn\,cm^{-2}}$",
                         transform=axes[i, -1].transAxes, rotation=90,
                         ha="center", va="center", fontsize=11)

    for j in range(ncols):
        axes[-1, j].set_xlabel(r"Polymer density [1/$\mu$m$^2$]", size=10)

    fig.supylabel(r"Monomer fraction with $F > \mathrm{threshold}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_global,
                             "frac_monomers_F_gt_20_and_27p5pN_all_shears_2xN.png"),
                dpi=300)
    plt.close(fig)

print("Done. All figures saved.")