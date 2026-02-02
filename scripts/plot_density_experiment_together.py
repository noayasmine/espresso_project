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
#shear=5455
shear=1454

base_self  = f"/home/nroebersen/espresso_project/experiments/density_experiment/{shear}/selfassociation"
base_noself = f"/home/nroebersen/espresso_project/experiments/density_experiment/{shear}/no_selfassociation"

AREA = 72.0
USE_DENSITY_X = True
N_LAST = 20
N_MOV = 20  # moving average window for tensile timeseries
INCLUDE_NS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
THRESHOLD = 20.0  # pN
save_thr  = f"/home/nroebersen/espresso_project/experiments/density_experiment/{shear}/num_monomers_above_threshold_vs_density_overlay.png"

# choose plotting window for timeseries (indices, inclusive/exclusive like Python slices)
T1 = 0
T2 = 1500  # e.g. 5000 or None

save_stats = f"/home/nroebersen/espresso_project/experiments/density_experiment/{shear}/tensile_force_stats_vs_density_overlay.png"
save_ts    = f"/home/nroebersen/espresso_project/experiments/density_experiment/{shear}/tensile_force_timeseries_subplots_overlay.png"
save_hist  = f"/home/nroebersen/espresso_project/experiments/density_experiment/{shear}/tensile_force_equilibrium_histograms_overlay.png"

# ----------------------------
# helpers
# ----------------------------
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
    n_above = []   # count of monomers above threshold per Npol

    tensile_mu_ts = {}     # (T,) pN
    tensile_sd_ts = {}     # (T,) pN
    tensile_mu_ma = {}     # (T,) pN
    tensile_sd_ma = {}     # (T,) pN

    # equilibrium per-monomer means (distribution across monomers)
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

        # count monomers above threshold
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

        # --- equilibrium window: last N_LAST timesteps
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

xlabel = r"Polymer density (1/$\mu$m$^2$)" if USE_DENSITY_X else r"Number of polymers"

# ----------------------------
# plot 1: tensile stats vs x (OVERLAY)
# ----------------------------
plt.figure(figsize=(4.6, 3.4))
plt.errorbar(A["x"], A["t_mean"], yerr=A["t_std"], fmt="o-", capsize=3, label=A["label"])
plt.errorbar(B["x"], B["t_mean"], yerr=B["t_std"], fmt="s-", capsize=3, label=B["label"])
plt.xlabel(xlabel)
plt.ylabel(f"Avg tensile force per monomer (pN)")
plt.legend(frameon=False)
plt.tight_layout()
os.makedirs(os.path.dirname(save_stats), exist_ok=True)
plt.savefig(save_stats, dpi=300)
plt.show()

# ----------------------------
# plot 2: timeseries subplots per Npol, overlay both conditions
#         show mean +/- std (alpha=0.3)
#         with selectable window [T1:T2]
# ----------------------------
Ns_common = intersect_Ns(A, B)
if len(Ns_common) == 0:
    raise RuntimeError("No overlapping N_polymers_* folders between selfassociation and no_selfassociation.")

fig, axes = plt.subplots(len(Ns_common), 1,
                         figsize=(7.5, 1.8 * len(Ns_common)),
                         sharex=True)

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

    tA = np.arange(i1A, i2A) * 10**-4
    tB = np.arange(i1B, i2B) * 10**-4

    ax.plot(tA, muA, label="selfassociation")
    ax.fill_between(tA, muA - sdA, muA + sdA, alpha=0.3)

    ax.plot(tB, muB, label="no selfassociation")
    ax.fill_between(tB, muB - sdB, muB + sdB, alpha=0.3)

    dens = Npol / AREA
    ax.set_ylabel(f"N={Npol}\nρ={dens:.3f}" + r"$(\mu m^2)$")

axes[-1].set_xlabel("Time (s)")
axes[0].legend(frameon=False, ncol=2)

plt.tight_layout()
os.makedirs(os.path.dirname(save_ts), exist_ok=True)
plt.savefig(save_ts, dpi=300)
plt.show()

# ----------------------------
# plot 3: equilibrium histograms (mean_per_monomer), one subplot per Npol,
#         overlay selfassociation and no selfassociation
# ----------------------------
# Use common bin edges per Npol (computed from both datasets) so the overlay is fair.
fig, axes = plt.subplots(len(Ns_common), 1,
                         figsize=(7.5, 1.8 * len(Ns_common)),
                         sharex=False)

if len(Ns_common) == 1:
    axes = [axes]

XMIN, XMAX = -50.0, 75.0
NBINS = 40

for ax, Npol in zip(axes, Ns_common):
    a = A["eq_mean_per_monomer"][Npol]  # (N_total,) pN
    b = B["eq_mean_per_monomer"][Npol]  # (N_total,) pN

    # shared bins for fair comparison
    allv = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if allv.size == 0:
        continue
    bins = np.histogram_bin_edges(allv, bins=40)
     # shared bins with fixed range
    bins = np.linspace(XMIN, XMAX, NBINS + 1)

    ax.hist(a, bins=bins, density=True, alpha=0.5, label="selfassociation")
    ax.hist(b, bins=bins, density=True, alpha=0.5, label="no selfassociation")
    ax.set_xlim(XMIN, XMAX)

    dens = Npol / AREA
    ax.set_ylabel(f"N={Npol}\nρ={dens:.3f} " + r"($1 / \mu m^2$)")

axes[-1].set_xlabel(f"Equilibrium tensile per monomer (pN)")
axes[0].legend(frameon=False, ncol=2)

plt.tight_layout()
os.makedirs(os.path.dirname(save_hist), exist_ok=True)
plt.savefig(save_hist, dpi=300)

# ----------------------------
# plot 4: number of monomers above threshold (moving-avg over last N_LAST)
# ----------------------------
plt.figure(figsize=(4.6, 3.4))
plt.plot(A["x"], A["n_above"], "o-", label=A["label"])
plt.plot(B["x"], B["n_above"], "s-", label=B["label"])
plt.xlabel(xlabel)
plt.ylabel(f" Fraction of monomers with F > {THRESHOLD:.0f} pN")
plt.legend(frameon=False)
plt.tight_layout()
os.makedirs(os.path.dirname(save_thr), exist_ok=True)
plt.savefig(save_thr, dpi=300)
plt.show()
