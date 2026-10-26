

import os
import numpy as np
import matplotlib.pyplot as plt

def load_npz_first(path):
    d = np.load(path)
    return d[d.files[0]]

# ----------------------------
# settings
# ----------------------------
# shear = 1454
# shear = 3636
shear = 5455
# shear = 2727

base_dir = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{shear}/selfassociation"

AREA = 72.0
USE_DENSITY_X = True
N_LAST = 20
N_MOV = 20
INCLUDE_NS = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

# normalization
N_MONOMERS_PER_POLYMER = 50.0

# for F>threshold plots
# THRESHOLD = 20.0  # pN
THRESHOLD = 20

# choose which shear rates to include (edit freely)
SHEARS_TO_INCLUDE = [1454, 1818, 2727, 3636, 5455]  # 2x2 by default; add more and the code will still work (it will use first 4)

# ----------------------------
# shear → wall shear stress mapping (dyn/cm^2)
# ----------------------------
SHEAR_TO_STRESS = {
    1454: 160,
    2091: 230,
    1818: 200,
    2727: 300,
    3636: 400,
    5455: 600,
}

# ----------------------------
# colors / style
# ----------------------------
COL_PURPLE = "purple"
COL_CADET  = "cadetblue"
ALPHA = 0.7

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, length=4)

TAB10 = plt.get_cmap("tab10").colors

# ----------------------------
# collect (single shear: selfassociation only)
# ----------------------------
Ns, densities = [], []
assoc_mean, assoc_std = [], []
t_mean, t_std = [], []
added_timeseries = {}
tensile_avg_ts = {}
tensile_avg_ma = {}

for name in sorted(os.listdir(base_dir)):
    if not name.startswith("N_polymers_"):
        continue

    Npol = int(name.split("_")[-1])
    if INCLUDE_NS is not None and Npol not in INCLUDE_NS:
        continue

    folder = os.path.join(base_dir, name)
    Ns.append(Npol)
    densities.append(Npol / AREA)

    added_ts = load_npz_first(os.path.join(folder, "added_bonds.npz")).astype(float)
    added_timeseries[Npol] = added_ts

    assoc_total_per_polymer = np.nansum(added_ts) / Npol
    assoc_mean.append(assoc_total_per_polymer)
    assoc_std.append(0.0)

    tens = load_npz_first(os.path.join(folder, "tensile_forces.npz"))
    avg_t = np.nanmean(tens, axis=1) * 1000.0
    tensile_avg_ts[Npol] = avg_t

    ma = np.empty_like(avg_t, dtype=float)
    for i in range(len(avg_t)):
        start = max(0, i - N_MOV + 1)
        ma[i] = avg_t[start:i+1].mean()
    tensile_avg_ma[Npol] = ma

    last = min(N_LAST, tens.shape[0])
    tens_last = tens[-last:]
    mean_per_monomer = np.mean(tens_last, axis=0)

    t_mean.append(np.mean(mean_per_monomer) * 1000.0)
    t_std.append(np.std(mean_per_monomer) * 1000.0)

# ----------------------------
# sort
# ----------------------------
order = np.argsort(Ns)
Ns = np.array(Ns)[order]
densities = np.array(densities)[order]
assoc_mean = np.array(assoc_mean)[order]
assoc_std = np.array(assoc_std)[order]
t_mean = np.array(t_mean)[order]
t_std = np.array(t_std)[order]

x = densities if USE_DENSITY_X else Ns
xlabel = r"Polymer density [1/$\mu$m$^2$]" if USE_DENSITY_X else r"Number of polymers"

# ----------------------------
# plot 1: self associations vs x
# ----------------------------
fig, ax = plt.subplots(figsize=(4, 3))
ax.errorbar(x, assoc_mean, yerr=assoc_std, fmt="o-", lw=2.0, ms=5, capsize=3, color=COL_PURPLE, alpha=ALPHA)
ax.axhline(50, lw=1.5, ls="--", color=COL_CADET, alpha=ALPHA, label="max associations")

ax.set_xlabel(xlabel)
ax.set_ylabel(f"Avg self-associations per polymer\n(last {N_LAST} timesteps)")
ax.legend(frameon=False)
style_axes(ax)

fig.tight_layout()
fig.savefig(os.path.join(base_dir, "self_associations_total_vs_x.png"), dpi=300)
plt.close(fig)

# ----------------------------
# plot 2: tensile stats vs x
# ----------------------------
fig, ax = plt.subplots(figsize=(4, 3))
ax.errorbar(x, t_mean, yerr=t_std, fmt="o-", lw=2.0, ms=5, capsize=3, color=COL_PURPLE, alpha=ALPHA)

ax.set_xlabel(xlabel)
ax.set_ylabel(f"Avg tensile force (pN)\n(MA window {N_MOV}, last {N_LAST} timesteps)")
style_axes(ax)

fig.tight_layout()
fig.savefig(os.path.join(base_dir, "tensile_force_stats_vs_x.png"), dpi=300)
plt.close(fig)

# ----------------------------
# plot 3: added bonds per timestep (tab10)
# ----------------------------
fig, ax = plt.subplots(figsize=(5, 3))
for i, Npol in enumerate(sorted(added_timeseries)):
    ts = added_timeseries[Npol]
    ax.plot(ts, lw=1.5, alpha=ALPHA, color=TAB10[i % len(TAB10)], label=f"N={Npol}")

ax.set_xlabel("Timestep")
ax.set_ylabel("Added self-associations per timestep (total)")
ax.legend(frameon=False, fontsize=8, ncol=2)
style_axes(ax)

fig.tight_layout()
fig.savefig(os.path.join(base_dir, "self_associations_per_timestep_timeseries.png"), dpi=300)
plt.close(fig)

# ----------------------------
# plot 4: avg tensile over time (stacked)
# ----------------------------
Ns_sorted = sorted(tensile_avg_ma)
fig, axes = plt.subplots(len(Ns_sorted), 1, figsize=(6, 1.6 * len(Ns_sorted)), sharex=True)
if len(Ns_sorted) == 1:
    axes = [axes]

for ax, Npol in zip(axes, Ns_sorted):
    ax.plot(tensile_avg_ma[Npol], lw=2.0, color=COL_PURPLE, alpha=ALPHA)
    ax.set_ylabel(f"N={Npol}")
    style_axes(ax)

axes[-1].set_xlabel("Timestep")
fig.tight_layout()
fig.savefig(os.path.join(base_dir, "tensile_force_avg_over_time_subplots.png"), dpi=300)
plt.close(fig)

# ----------------------------
# plot 5: ALL shears – cumulative fraction vs density (tab10, wall shear stress labels)
# ----------------------------
def collect_assoc_vs_density_for_shear(shear_val):
    bd = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{shear_val}/selfassociation"
    if not os.path.isdir(bd):
        return None

    Ns_loc, dens_loc, assoc_loc = [], [], []
    for name in sorted(os.listdir(bd)):
        if not name.startswith("N_polymers_"):
            continue

        Npol = int(name.split("_")[-1])
        if INCLUDE_NS is not None and Npol not in INCLUDE_NS:
            continue

        folder = os.path.join(bd, name)
        added_path = os.path.join(folder, "added_bonds.npz")
        if not os.path.exists(added_path):
            continue

        added_ts = load_npz_first(added_path).astype(float)
        assoc_total = np.nansum(added_ts) / Npol
        assoc_frac = assoc_total / N_MONOMERS_PER_POLYMER

        Ns_loc.append(Npol)
        dens_loc.append(Npol / AREA)
        assoc_loc.append(assoc_frac)

    if len(Ns_loc) == 0:
        return None

    order = np.argsort(Ns_loc)
    return np.array(dens_loc)[order], np.array(assoc_loc)[order]

fig, ax = plt.subplots(figsize=(4.6, 3.4))
markers = ["o", "s", "^", "D", "v", "P", "X"]
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]

for i, sh in enumerate(SHEARS_TO_INCLUDE):
    out = collect_assoc_vs_density_for_shear(sh)
    if out is None:
        continue
    dens_loc, assoc_loc = out
    stress = SHEAR_TO_STRESS.get(sh, sh)

    ax.plot(dens_loc, assoc_loc, lw=2.0, ms=5,
            marker=markers[i % len(markers)], ls=linestyles[i % len(linestyles)],
            color=TAB10[i % len(TAB10)], alpha=ALPHA,
            label=rf"$\tau= {stress}\,\mathrm{{dyn\,cm^{{-2}}}}$")

ax.set_xlabel(r"Polymer density [1/$\mu$m$^2$]")
ax.set_ylabel("Fraction of self-associated monomers")
ax.set_xlim(0, 3)
ax.set_ylim(0, 1)
ax.legend(
    frameon=False,
    fontsize=9,
    loc="lower right",
    bbox_to_anchor=(1.0, 0.15))

style_axes(ax)

fig.tight_layout()
fig.savefig(os.path.join("/home/nroebersen/espresso_project/experiments/density_experiment/run_1", "assoc_fraction_vs_density_all_shears.png"), dpi=300)
plt.close(fig)

# ======================================================================================
# NEW: 2x2 subplot figure: monomer fraction with F > THRESHOLD pN vs density
#      (purple=selfassociation, cadetblue=no selfassociation), one panel per shear
#      NOTE: no shared y-axis; each panel uses its own ylim(0, ...)
# ======================================================================================

SHEARS_TO_INCLUDE = [1818, 2727, 3636, 5455]

def trailing_moving_avg_axis0(arr_TN, window):
    arr = np.asarray(arr_TN, dtype=float)
    T, N = arr.shape
    out = np.empty_like(arr, dtype=float)
    for t in range(T):
        s = max(0, t - window + 1)
        out[t] = np.nanmean(arr[s:t+1], axis=0)
    return out

def collect_frac_above_threshold(base_dir_condition, threshold=THRESHOLD):
    Ns_loc, dens_loc, frac_loc = [], [], []
    THRESHOLD = threshold
    for name in sorted(os.listdir(base_dir_condition)):
        if not name.startswith("N_polymers_"):
            continue

        Npol = int(name.split("_")[-1])
        if INCLUDE_NS is not None and Npol not in INCLUDE_NS:
            continue

        folder = os.path.join(base_dir_condition, name)
        tens_path = os.path.join(folder, "tensile_forces.npz")
        if not os.path.exists(tens_path):
            continue

        tens = load_npz_first(tens_path).astype(float)        # (T, N_total), MD units
        tens_pN = tens * 1000.0                               # pN
        ma_monomer = trailing_moving_avg_axis0(tens_pN, N_MOV)

        last = min(N_LAST, ma_monomer.shape[0])
        ma_last = ma_monomer[-last:]
        ma_last_mean_per_monomer = np.nanmean(ma_last, axis=0)

        N_total = tens.shape[1]
        frac_above = np.sum(ma_last_mean_per_monomer > THRESHOLD) / N_total

        Ns_loc.append(Npol)
        dens_loc.append(Npol / AREA)
        frac_loc.append(frac_above)

    if len(Ns_loc) == 0:
        return None

    order = np.argsort(Ns_loc)
    return np.array(dens_loc)[order], np.array(frac_loc)[order]

# keep only shears that exist for both conditions
shears_ok = []
for sh in SHEARS_TO_INCLUDE:
    bd_self  = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{sh}/selfassociation"
    bd_noself = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{sh}/no_selfassociation"
    if os.path.isdir(bd_self) and os.path.isdir(bd_noself):
        shears_ok.append(sh)

# 2x2 layout; if more than 4 are provided, we plot the first 4
shears_ok = shears_ok[:4]

if len(shears_ok) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=False, dpi=300)
    axes = axes.ravel()
    print("Tensile force above tension plot")

    for j, ax in enumerate(axes):
        if j >= len(shears_ok):
            ax.axis("off")
            continue

        sh = shears_ok[j]
        base_self  = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{sh}/selfassociation"
        base_noself = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{sh}/no_selfassociation"

        outA = collect_frac_above_threshold(base_self)
        outB = collect_frac_above_threshold(base_noself)

        if outA is not None:
            xA, yA = outA
            ax.plot(xA, yA, "o-", lw=2.0, ms=5, color=COL_PURPLE, alpha=ALPHA, label="selfassociation")
        if outB is not None:
            xB, yB = outB
            ax.plot(xB, yB, "s-", lw=2.0, ms=5, color=COL_CADET, alpha=ALPHA, label="no selfassociation")

        stress = SHEAR_TO_STRESS.get(sh, sh)
        stress = SHEAR_TO_STRESS.get(sh, sh)
        if j == 0:
            ax.text(0.96, 0.2, rf"$\tau = {stress}\,\mathrm{{dyn~cm^{{-2}}}}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
        else:
            ax.text(0.96, 0.1, rf"$\tau = {stress}\,\mathrm{{dyn~cm^{{-2}}}}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11)


        ax.set_xlim(0, 3)
        ax.set_ylim(0, None)  # <-- individual y-scale per panel
        style_axes(ax)

    fig.supxlabel(r"Polymer density [1/$\mu$m$^2$]")
    fig.supylabel(fr"Monomer fraction with $F > {THRESHOLD}$ pN")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        os.path.join(
            "/home/nroebersen/espresso_project/experiments/density_experiment/run_1",
            f"frac_monomers_F_gt_{int(THRESHOLD)}pN_all_shears_2x2.png"
        ),
        dpi=300
    )
    print("Tensile force above tension plot")
    plt.close(fig)

# ======================================================================================
# NEW: stacked subplot figure: monomer fraction with F > THRESHOLD pN vs density
#      one panel per shear (ALL available shears), independent y-axes
# ======================================================================================

if len(shears_ok) > 0:
    fig, axes = plt.subplots(len(shears_ok), 1,
                             figsize=(5.5/1.5, 1.8 * len(shears_ok)/1.5),
                             sharex=True, sharey=False)

    if len(shears_ok) == 1:
        axes = [axes]

    for ax, sh in zip(axes, shears_ok):
        base_self  = f"/home/nroebersen/espresso_project/experiments/density_experiment/{sh}/selfassociation"
        base_noself = f"/home/nroebersen/espresso_project/experiments/density_experiment/{sh}/no_selfassociation"

        outA = collect_frac_above_threshold(base_self)
        outB = collect_frac_above_threshold(base_noself)

        if outA is not None:
            xA, yA = outA
            ax.plot(xA, yA, "o-", lw=2.0, ms=5,
                    color=COL_PURPLE, alpha=ALPHA, label="selfassociation")

        if outB is not None:
            xB, yB = outB
            ax.plot(xB, yB, "s-", lw=2.0, ms=5,
                    color=COL_CADET, alpha=ALPHA, label="no selfassociation")

        stress = SHEAR_TO_STRESS.get(sh, sh)
        # ax.text(0.96, 0.06,
        #         rf"$\tau = {stress}\,\mathrm{{dyn~cm^{{-2}}}}$",
        #         transform=ax.transAxes,
        #         ha="right", va="bottom", fontsize=11)

        # ax.text(
        # 0.04, 0.96,
        # rf"$\tau = {stress}\,\mathrm{{dyn~cm^{{-2}}}}$",
        # transform=ax.transAxes,
        # ha="left", va="top",
        # fontsize=11)
        ax.text(
            1.08, 0.5,
            rf"$\tau = {stress}$" "\n" r"$\mathrm{dyn\,cm^{-2}}$",
            transform=ax.transAxes,
            rotation=90,
            ha="center", va="center",
            fontsize=11)


        ax.set_xlim(0, 3)
        ax.set_ylim(0, None)
        style_axes(ax)

    axes[-1].set_xlabel(r"Polymer density [1/$\mu$m$^2$]")
    fig.supylabel(fr"Monomer fraction with $F > {int(THRESHOLD)}$ pN")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        os.path.join(
            "/home/nroebersen/espresso_project/experiments/density_experiment",
            f"frac_monomers_F_gt_{int(THRESHOLD)}pN_all_shears_stacked.png"
        ),
        dpi=300
    )
    plt.close(fig)


thresholds = [20, 27.5]

if len(shears_ok) > 0:
    nrows = len(shears_ok)
    ncols = 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2 * (5.5/1.5), 1.8 * nrows/1.5),
        sharex=True, sharey=False
    )
    for i in range(nrows):
        axes[i, 1].sharey(axes[i, 0])


    if nrows == 1:
        axes = np.array([axes])  # shape (1, 2)

    for i, sh in enumerate(shears_ok):
        base_self = f"/home/nroebersen/espresso_project/experiments/density_experiment/{sh}/selfassociation"
        base_noself = f"/home/nroebersen/espresso_project/experiments/density_experiment/{sh}/no_selfassociation"

        for j, thr in enumerate(thresholds):
            ax = axes[i, j]

            outA = collect_frac_above_threshold(base_self, threshold=thr)
            outB = collect_frac_above_threshold(base_noself, threshold=thr)

            if outA is not None:
                xA, yA = outA
                ax.plot(xA, yA, "o-", lw=2.0, ms=5, color=COL_PURPLE, alpha=ALPHA, label="selfassociation")

            if outB is not None:
                xB, yB = outB
                ax.plot(xB, yB, "s-", lw=2.0, ms=5, color=COL_CADET, alpha=ALPHA, label="no selfassociation")

            ax.set_xlim(0, 3)
            ax.set_ylim(0, None)
            style_axes(ax)

            if i == 0:
                ax.set_title(fr"$F > {thr}$ pN", fontsize=11)

        # Stress label only once: on the RIGHT of the RIGHT column axes
        axR = axes[i, 1]
        stress = SHEAR_TO_STRESS.get(sh, sh)
        axR.text(
            1.08, 0.5,
            rf"$\tau = {stress}$" "\n" r"$\mathrm{dyn\,cm^{-2}}$",
            transform=axR.transAxes,
            rotation=90,
            ha="center", va="center",
            fontsize=11
        )

    for j in range(ncols):
        axes[-1, j].set_xlabel(r"Polymer density [1/$\mu$m$^2$]")

    fig.supylabel(fr"Monomer fraction with $F > \mathrm{{threshold}}$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        os.path.join(
            "/home/nroebersen/espresso_project/experiments/density_experiment",
            "frac_monomers_F_gt_20_and_27p5pN_all_shears_2xN.png"
        ),
        dpi=300
    )
    plt.close(fig)

