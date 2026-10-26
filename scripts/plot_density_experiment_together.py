import os
import numpy as np
import matplotlib.pyplot as plt

def load_npz_first(path):
    d = np.load(path)
    return d[d.files[0]]

# ----------------------------
# settings
# ----------------------------

#shear=3636
shear=5455
#shear=1454
#shear=2727

base_self  = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{shear}/selfassociation"
base_noself = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{shear}/no_selfassociation"

AREA = 72.0
USE_DENSITY_X = True
N_LAST = 20
N_MOV = 1  # moving average window for tensile timeseries
INCLUDE_NS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
THRESHOLD = 20.0  # pN

save_thr  = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{shear}/num_monomers_above_threshold_vs_density_overlay.png"
save_stats = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{shear}/tensile_force_stats_vs_density_overlay.png"
save_ts    = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{shear}/tensile_force_timeseries_subplots_overlay.png"

# choose plotting window for timeseries (indices, inclusive/exclusive like Python slices)
T1 = 0
T2 = 1500  # e.g. 5000 or None

# histogram: choose ONE N only
HIST_N = 200
dens_hist = round(HIST_N/AREA,1)

save_hist  = f"/home/nroebersen/espresso_project/experiments/density_experiment/run_1/{shear}/tensile_force_equilibrium_histograms_overlay_{dens_hist}.png"

# histogram axis and bins
XMIN, XMAX = -50, 80
NBINS = 2*26

# ----------------------------
# consistent paper-style colors
# ----------------------------
COL_A = "purple"
COL_B = "cadetblue"
ALPHA = 0.7

# ----------------------------
# helpers
# ----------------------------
def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, length=4)

def trailing_moving_avg(x, window):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.nanmean(x[start:i+1])
    return out

def slice_ts(arr, t1, t2):
    arr = np.asarray(arr)
    n = len(arr)
    if t1 is None:
        t1 = 0
    if t2 is None:
        t2 = n
    t1 = max(0, min(int(t1), n))
    t2 = max(0, min(int(t2), n))
    if t2 < t1:
        t1, t2 = t2, t1
    return arr[t1:t2], t1, t2

def trailing_moving_avg_axis0(arr_TN, window):
    """
    Trailing moving average along axis=0 (time) for a (T,N) array.
    Returns same shape (T,N).
    """
    arr = np.asarray(arr_TN, dtype=float)
    T, N = arr.shape
    out = np.empty_like(arr, dtype=float)
    for t in range(T):
        s = max(0, t - window + 1)
        out[t] = np.nanmean(arr[s:t+1], axis=0)
    return out

def collect_one_condition(base_dir, label):
    Ns, densities = [], []
    t_mean, t_std = [], []
    n_above = []   # fraction above threshold

    tensile_mu_ts = {}     # (T,) pN
    tensile_sd_ts = {}     # (T,) pN
    tensile_mu_ma = {}     # (T,) pN
    tensile_sd_ma = {}     # (T,) pN

    eq_mean_per_monomer = {}  # Npol -> (N_total,) in pN

    for name in sorted(os.listdir(base_dir)):
        if not name.startswith("N_polymers_"):
            continue

        Npol = int(name.split("_")[-1])
        if INCLUDE_NS is not None and Npol not in INCLUDE_NS:
            continue

        folder = os.path.join(base_dir, name)
        tens_path = os.path.join(folder, "tensile_forces.npz")
        if not os.path.exists(tens_path):
            continue

        Ns.append(Npol)
        densities.append(Npol / AREA)  # area density

        tens = load_npz_first(tens_path).astype(float)   # (T, N_total)
        tens_pN = tens * 1000.0  # (T, N_total) in pN

        # moving avg per monomer (NOT the mean over monomers)
        ma_monomer = trailing_moving_avg_axis0(tens_pN, N_MOV)  # (T, N_total)

        # last N_LAST timesteps: average the moving-avg signal in that window
        last = min(N_LAST, ma_monomer.shape[0])
        ma_last = ma_monomer[-last:]                      # (last, N_total)
        ma_last_mean_per_monomer = np.nanmean(ma_last, axis=0)  # (N_total,)

        # fraction monomers above threshold
        N_total = tens.shape[1]
        frac_above = np.sum(ma_last_mean_per_monomer > THRESHOLD) / N_total
        n_above.append(frac_above)

        # per-timestep mean and std across ALL monomers -> pN
        mu_t = np.nanmean(tens, axis=1) * 1000.0
        sd_t = np.nanstd(tens, axis=1) * 1000.0

        tensile_mu_ts[Npol] = mu_t
        tensile_sd_ts[Npol] = sd_t

        tensile_mu_ma[Npol] = trailing_moving_avg(mu_t, N_MOV)
        tensile_sd_ma[Npol] = trailing_moving_avg(sd_t, N_MOV)

        # equilibrium window: last N_LAST timesteps
        last = min(N_LAST, tens.shape[0])
        tens_last = tens[-last:]                          # (last, N_total)

        # mean tensile force per monomer (time-averaged) -> pN
        mean_per_monomer = np.nanmean(tens_last, axis=0) * 1000.0  # (N_total,)
        eq_mean_per_monomer[Npol] = mean_per_monomer

        # stats of that distribution
        t_mean.append(float(np.nanmean(mean_per_monomer)))
        t_std.append(float(np.nanstd(mean_per_monomer)))

    order = np.argsort(Ns)
    Ns = np.array(Ns)[order]
    densities = np.array(densities)[order]
    n_above = np.array(n_above)[order]
    t_mean = np.array(t_mean)[order]
    t_std = np.array(t_std)[order]

    x = densities if USE_DENSITY_X else Ns
    return {
        "label": label,
        "Ns": Ns,
        "densities": densities,
        "x": x,
        "t_mean": t_mean,
        "t_std": t_std,
        "tensile_mu_ts": tensile_mu_ts,
        "tensile_sd_ts": tensile_sd_ts,
        "tensile_mu_ma": tensile_mu_ma,
        "tensile_sd_ma": tensile_sd_ma,
        "eq_mean_per_monomer": eq_mean_per_monomer,
        "n_above": n_above
    }

def intersect_Ns(A, B):
    return sorted(set(A["Ns"].tolist()).intersection(set(B["Ns"].tolist())))

# ----------------------------
# collect both conditions
# ----------------------------
A = collect_one_condition(base_self,  "selfassociation")
B = collect_one_condition(base_noself, "no selfassociation")

xlabel = r"Polymer density [1/$\mu$m$^2$]" if USE_DENSITY_X else r"Number of polymers"

# ----------------------------
# plot 1: tensile stats vs x (OVERLAY)
# ----------------------------
fig, ax = plt.subplots(figsize=(4, 3))
ax.errorbar(A["x"], A["t_mean"], yerr=A["t_std"], fmt="o-", capsize=3, lw=2.0, ms=5,
            label=A["label"], color=COL_A, alpha=ALPHA)
ax.errorbar(B["x"], B["t_mean"], yerr=B["t_std"], fmt="s-", capsize=3, lw=2.0, ms=5,
            label=B["label"], color=COL_B, alpha=ALPHA)

ax.set_xlabel(xlabel, size=10)
ax.set_ylabel("Avg tensile force [pN]", size=10)
ax.set_xlim(0, 3.0)
ax.legend(frameon=False)
style_axes(ax)
fig.tight_layout()

os.makedirs(os.path.dirname(save_stats), exist_ok=True)
fig.savefig(save_stats, dpi=300)
plt.show()

# ----------------------------
# plot 2: timeseries subplots per Npol, overlay both conditions
#         show mean +/- std with selectable window [T1:T2]
# ----------------------------
Ns_common = intersect_Ns(A, B)
if len(Ns_common) == 0:
    raise RuntimeError("No overlapping N_polymers_* folders between selfassociation and no_selfassociation.")

fig, axes = plt.subplots(len(Ns_common), 1, figsize=(7.5, 1.8 * len(Ns_common)), sharex=True)
if len(Ns_common) == 1:
    axes = [axes]

for ax, Npol in zip(axes, Ns_common):
    muA_full = A["tensile_mu_ma"][Npol]
    sdA_full = A["tensile_sd_ma"][Npol]
    muB_full = B["tensile_mu_ma"][Npol]
    sdB_full = B["tensile_sd_ma"][Npol]

    muA, i1A, i2A = slice_ts(muA_full, T1, T2)
    sdA, _, _     = slice_ts(sdA_full, T1, T2)
    muB, i1B, i2B = slice_ts(muB_full, T1, T2)
    sdB, _, _     = slice_ts(sdB_full, T1, T2)

    tA = np.arange(i1A, i2A) * 1e-4
    tB = np.arange(i1B, i2B) * 1e-4

    ax.plot(tA, muA, color=COL_A, lw=2.0, alpha=ALPHA, label="selfassociation")
    ax.fill_between(tA, muA - sdA, muA + sdA, color=COL_A, alpha=0.25, linewidth=0)

    ax.plot(tB, muB, color=COL_B, lw=2.0, alpha=ALPHA, label="no selfassociation")
    ax.fill_between(tB, muB - sdB, muB + sdB, color=COL_B, alpha=0.25, linewidth=0)

    dens = Npol / AREA
    ax.set_ylabel(f"N={Npol}\nρ={dens:.3f} " + r"($1/\mu m^2$)")
    style_axes(ax)

axes[-1].set_xlabel("Time (s)")
axes[0].legend(frameon=False, ncol=2)

fig.tight_layout()
os.makedirs(os.path.dirname(save_ts), exist_ok=True)
fig.savefig(save_ts, dpi=300)
plt.show()

# ----------------------------
# plot 3: equilibrium histogram for ONE chosen Npol (overlay both conditions)
# ----------------------------
if HIST_N not in Ns_common:
    raise RuntimeError(f"HIST_N={HIST_N} not available in both conditions. Overlap: {Ns_common}")

a = A["eq_mean_per_monomer"][HIST_N]
b = B["eq_mean_per_monomer"][HIST_N]

bins = np.linspace(XMIN, XMAX, NBINS + 1)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(a[np.isfinite(a)], bins=bins, density=True, alpha=ALPHA, label="selfassociation", color=COL_A)
ax.hist(b[np.isfinite(b)], bins=bins, density=True, alpha=ALPHA, label="no selfassociation", color=COL_B)

ax.set_xlim(XMIN, XMAX)
ax.set_xlabel("Equilibrium tensile per monomer [pN]",size=10)
ax.set_ylabel("Probability density",size=10)
#ax.set_xticks(np.arange(XMIN, XMAX + 1, 10))

dens = HIST_N / AREA
#ax.set_title(f"N={HIST_N}, ρ={dens:.3f} " + r"($1/\mu m^2$)")
ax.legend(frameon=False, fontsize=12)
style_axes(ax)

fig.tight_layout()
os.makedirs(os.path.dirname(save_hist), exist_ok=True)
fig.savefig(save_hist, dpi=300)
plt.show()

# ----------------------------
# plot 4: fraction of monomers above threshold (moving-avg over last N_LAST)
# ----------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(A["x"], A["n_above"], "o-", lw=2.0, ms=5, label=A["label"], color=COL_A, alpha=ALPHA)
ax.plot(B["x"], B["n_above"], "s-", lw=2.0, ms=5, label=B["label"], color=COL_B, alpha=ALPHA)

ax.set_xlabel(xlabel)
ax.set_ylabel(f"Fraction monomers with F > {THRESHOLD:.0f} pN")
ax.legend(frameon=False)
ax.set_xlim(0,3)
ax.set_ylim(0,)
style_axes(ax)

fig.tight_layout()
os.makedirs(os.path.dirname(save_thr), exist_ok=True)
fig.savefig(save_thr, dpi=300)
plt.show()
