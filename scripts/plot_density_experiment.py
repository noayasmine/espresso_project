import os
import numpy as np
import matplotlib.pyplot as plt

def load_npz_first(path):
    d = np.load(path)
    return d[d.files[0]]

# ----------------------------
# settings
# ----------------------------
base_dir = "/home/nroebersen/espresso_project/experiments/density_experiment/selfassociation"
AREA = 72.0
USE_DENSITY_X = True
N_LAST = 20
N_MOV = 20  # moving average window for tensile timeseries
INCLUDE_NS = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

# ----------------------------
# collect
# ----------------------------
Ns, densities = [], []
assoc_mean, assoc_std = [], []
t_mean, t_std = [], []
added_timeseries = {}
tensile_avg_ts = {}      # raw avg over monomers vs time
tensile_avg_ma = {}      # moving-avg version

for name in sorted(os.listdir(base_dir)):
    if not name.startswith("N_polymers_"):
        continue

    Npol = int(name.split("_")[-1])
    if INCLUDE_NS is not None and Npol not in INCLUDE_NS:
        continue

    folder = os.path.join(base_dir, name)
    Ns.append(Npol)
    densities.append(Npol / AREA)

    added_ts = load_npz_first(os.path.join(folder, "added_bonds.npz")).astype(float)  # (T,)
    added_timeseries[Npol] = added_ts

    added_ts = load_npz_first(os.path.join(folder, "added_bonds.npz")).astype(float)  # (T,)
    added_timeseries[Npol] = added_ts

    assoc_total_per_polymer = np.nansum(added_ts) / Npol
    assoc_mean.append(assoc_total_per_polymer)
    # cannot get std because i do not track the self associations per polymer, but per timestep
    assoc_std.append(0)

    # --- tensile: scalar net (T,N), KEEP SIGN
    tens = load_npz_first(os.path.join(folder, "tensile_forces.npz"))  # (T,N)
    avg_t = np.nanmean(tens, axis=1) * 1000.0                          # (T,) pN
    tensile_avg_ts[Npol] = avg_t

    # moving avg (simple trailing window like before)
    ma = np.empty_like(avg_t, dtype=float)
    for i in range(len(avg_t)):
        start = max(0, i - N_MOV + 1)
        ma[i] = avg_t[start:i+1].mean()
    tensile_avg_ma[Npol] = ma

    # tensile forces: scalar net (T, N_total), signed (where N_total is all monomers)
    tens = load_npz_first(os.path.join(folder, "tensile_forces.npz"))  # (T, N_total)

    # last N timesteps
    last = min(20, tens.shape[0])
    tens_last = tens[-last:]                       # (50, N_total)

    # mean tensile force per monomer (time-averaged)
    mean_per_monomer = np.mean(tens_last, axis=0)   # (N_total,)

    # average over ALL monomers
    t_mean.append(np.mean(mean_per_monomer) * 1000.0)  # pN

    # std of the mean per monomer (SEM over monomers)
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
xlabel = r"Polymer density (1/$\mu$m$^2$)" if USE_DENSITY_X else r"Number of polymers"

# ----------------------------
# plot 1: self associations vs x
# ----------------------------
plt.figure(figsize=(4, 3))
plt.errorbar(x, assoc_mean, yerr=assoc_std, fmt="o-", capsize=3)
plt.axhline(50, color="r", label="max associations")
plt.xlabel(xlabel)
plt.ylabel(f"Avg self-associations per polymer\n(last {N_LAST} timesteps)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "self_associations_total_vs_x.png"), dpi=300)
plt.close()

# ----------------------------
# plot 2: tensile stats vs x
# ----------------------------
plt.figure(figsize=(4, 3))
plt.errorbar(x, t_mean, yerr=t_std, fmt="o-", capsize=3)
plt.xlabel(xlabel)
plt.ylabel(f"Avg tensile force (pN)\n(MA window {N_MOV}, last {N_LAST} timesteps)")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "tensile_force_stats_vs_x.png"), dpi=300)
plt.close()

# ----------------------------
# plot 3: added bonds per timestep
# ----------------------------
plt.figure(figsize=(5, 3))
for Npol in sorted(added_timeseries):
    ts = added_timeseries[Npol]
    plt.plot(np.arange(len(ts)), ts, label=f"N={Npol}")
plt.xlabel("Timestep")
plt.ylabel("Added self-associations per timestep (total)")
plt.legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "self_associations_per_timestep_timeseries.png"), dpi=300)
plt.close()

# ----------------------------
# plot 4: avg tensile over time (MOVING AVG), stacked
# ----------------------------
Ns_sorted = sorted(tensile_avg_ma)
fig, axes = plt.subplots(len(Ns_sorted), 1, figsize=(6, 1.6 * len(Ns_sorted)), sharex=True)
if len(Ns_sorted) == 1:
    axes = [axes]

for ax, Npol in zip(axes, Ns_sorted):
    ts = tensile_avg_ma[Npol]
    ax.plot(np.arange(len(ts)), ts)
    ax.set_ylabel(f"N={Npol}")

axes[-1].set_xlabel("Timestep")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "tensile_force_avg_over_time_subplots.png"), dpi=300)
plt.close()
