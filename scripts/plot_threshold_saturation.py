import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Settings
# ============================================================

RUNS     = [0, 1, 2, 3, 4]
RUN_BASE = "/home/nroebersen/espresso_project/experiments/density_experiment"

AREA       = 72.0
N_LAST     = 200
N_MOV      = 1
INCLUDE_NS = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

SHEARS     = [1818, 2727, 3636, 5455]
THRESHOLDS = [20.0, 30.0]

SHEAR_TO_STRESS = {1818: 200, 2727: 300, 3636: 400, 5455: 600}

COL_A = "purple"      # selfassociation
COL_B = "cadetblue"   # no selfassociation
ALPHA = 0.7

COND_STYLE = [
    ("selfassociation",    COL_A, "o", "selfassociation"),
    ("no_selfassociation", COL_B, "s", "no selfassociation"),
]
# threshold -> linestyle
THR_STYLE = {20.0: "-", 30.0: ":"}

XLABEL_SIGMA = r"Polymer surface density $\Sigma$ [$\mu$m$^{-2}$]"

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, length=4)

# ============================================================
# Data
# ============================================================

def load_npz_first(path):
    d = np.load(path)
    return d[d.files[0]]

def run_base_dir(run, shear_val, condition):
    return os.path.join(RUN_BASE, f"run_{run}", str(shear_val), condition)

def trailing_moving_avg_axis0(arr_TN, window):
    arr = np.asarray(arr_TN, dtype=float)
    out = np.empty_like(arr)
    for t in range(arr.shape[0]):
        s = max(0, t - window + 1)
        out[t] = np.nanmean(arr[s:t+1], axis=0)
    return out

def frac_above_sweep(shear_val, condition):
    """
    Returns {threshold: (densities, means, stds)}, std across runs.
    Each tensile_forces.npz is loaded once for all thresholds.
    """
    per_N = {}  # Npol -> {thr: [frac per run]}
    for run in RUNS:
        bd = run_base_dir(run, shear_val, condition)
        if not os.path.isdir(bd):
            continue
        for Npol in INCLUDE_NS:
            path = os.path.join(bd, f"N_polymers_{Npol}", "tensile_forces.npz")
            if not os.path.exists(path):
                continue
            tens  = load_npz_first(path).astype(float) * 1000.0   # pN
            ma    = trailing_moving_avg_axis0(tens, N_MOV)
            last  = min(N_LAST, ma.shape[0])
            ma_mn = np.nanmean(ma[-last:], axis=0)
            for thr in THRESHOLDS:
                per_N.setdefault(Npol, {}).setdefault(thr, []).append(
                    np.sum(ma_mn > thr) / tens.shape[1])

    Ns = sorted(per_N)
    dens = np.array(Ns) / AREA
    return {thr: (dens,
                  np.array([np.mean(per_N[N][thr]) for N in Ns]),
                  np.array([np.std(per_N[N][thr]) for N in Ns]))
            for thr in THRESHOLDS}

# ============================================================
# Plot: 2×2 – fraction above 20/30 pN (saturation)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=False, dpi=300)
axes = axes.ravel()
for ax, sh in zip(axes, SHEARS):
    for condition, color, marker, label in COND_STYLE:
        sweep = frac_above_sweep(sh, condition)
        for thr in THRESHOLDS:
            dens, means, stds = sweep[thr]
            if len(dens) == 0:
                continue
            ax.errorbar(dens, means, yerr=stds, fmt=marker, ls=THR_STYLE[thr], lw=2.0, ms=5,
                        capsize=3, color=color, alpha=ALPHA)

    ax_r = ax.twinx()
    ax_r.set_yticks([])
    ax_r.set_ylabel(rf"$\tau = {SHEAR_TO_STRESS[sh]}\,\mathrm{{dyn~cm^{{-2}}}}$", fontsize=11, labelpad=8)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, None)
    style_axes(ax)

fig.supxlabel(XLABEL_SIGMA)
fig.supylabel(r"Monomer fraction with $F > F_\mathrm{thr}$")
cond_handles = [Line2D([], [], color=c, marker=m, lw=2, label=l) for _, c, m, l in COND_STYLE]
thr_handles = [Line2D([], [], color="gray", ls=THR_STYLE[t], lw=2,
                      label=fr"$F_\mathrm{{thr}} = {int(t)}$ pN") for t in THRESHOLDS]
fig.legend(handles=cond_handles, frameon=False, ncol=2,
           loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=10)
fig.legend(handles=thr_handles, frameon=False, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.88])
out = os.path.join(RUN_BASE, "frac_monomers_F_gt_20_30pN_all_shears_2x2.png")
fig.savefig(out, dpi=300)
plt.close(fig)
print(f"Saved {out}")
