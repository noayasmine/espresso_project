import numpy as np
import matplotlib.pyplot as plt
import os

def load_npz_first(npz_path):
    data = np.load(npz_path)
    return data[list(data.keys())[0]]

# ----------------------------
# user settings
# ----------------------------
base_dir = "/home/nroebersen/espresso_project/experiments/density_experiment"
AREA = 72.0
USE_DENSITY_X = True
N_LAST = 20

INCLUDE_NS = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

# ----------------------------
# collect data
# ----------------------------
Ns = []
densities = []

# self-association equilibrium stats (last N_LAST timesteps)
assoc_mean = []
assoc_std = []

# tensile equilibrium stats (last N_LAST timesteps)
t_mean = []
t_std = []

added_timeseries = {}     # total added per timestep (T,)
tensile_avg_ts = {}       # avg tensile over monomers vs time (T,) in pN

for name in sorted(os.listdir(base_dir)):
    if not name.startswith("N_polymers_"):
        continue

    Npol = int(name.split("_")[-1])
    if INCLUDE_NS is not None and Npol not in INCLUDE_NS:
        continue

    folder = os.path.join(base_dir, name)

    Ns.append(Npol)
    densities.append(Npol / AREA)

    # ---- added bonds -> timeseries total per timestep
    added = load_npz_first(os.path.join(folder, "added_bonds.npz"))
    if added.ndim == 1:
        added_ts = added.astype(np.float32)  # (T,)
    else:
        added_ts = np.nansum(added, axis=tuple(range(1, added.ndim))).astype(np.float32)  # (T,)

    added_timeseries[Npol] = added_ts

    # equilibrium mean/std from LAST N_LAST timesteps (normalize per polymer)
    last = min(N_LAST, added_ts.shape[0])
    assoc_eq = added_ts[-last:]
    assoc_mean.append(np.nanmean(assoc_eq) / Npol)
    assoc_std.append(np.nanstd(assoc_eq) / Npol)

    # ---- tensile forces: load raw -> magnitude -> avg over monomers vs time
    tens_path = os.path.join(folder, "tensile_forces.npz")
    tens_arr = load_npz_first(tens_path)  # (T,N) or (T,N,3)

    if tens_arr.ndim == 3:
        mags = np.linalg.norm(tens_arr, axis=-1)  # (T,N)
    elif tens_arr.ndim == 2:
        mags = np.abs(tens_arr)                   # (T,N)
    else:
        raise ValueError("Unexpected tensile_forces shape", tens_arr.shape)

    avg_over_monomers_time = np.nanmean(mags, axis=1).astype(np.float32) * 1000.0  # (T,) pN
    tensile_avg_ts[Npol] = avg_over_monomers_time

    # equilibrium mean/std from LAST N_LAST timesteps
    last = min(N_LAST, avg_over_monomers_time.shape[0])
    tens_eq = avg_over_monomers_time[-last:]
    t_mean.append(np.nanmean(tens_eq))
    t_std.append(np.nanstd(tens_eq))


# ----------------------------
# sort by N
# ----------------------------
order = np.argsort(Ns)
Ns = np.array(Ns)[order]
densities = np.array(densities)[order]

assoc_mean = np.array(assoc_mean)[order]
assoc_std = np.array(assoc_std)[order]

t_mean = np.array(t_mean)[order]
t_std = np.array(t_std)[order]

x = densities if USE_DENSITY_X else Ns
xlabel = r"Polymer density (N/$\mu$m$^2$)" if USE_DENSITY_X else r"Number of polymers"


# ----------------------------
# plot 1: self associations vs x (mean ± std over last N_LAST)
# ----------------------------
plt.figure(figsize=(4, 3))
plt.errorbar(x, assoc_mean, yerr=assoc_std, fmt="o-", capsize=3)
plt.axhline(50, color='r', label="max associations")
plt.xlabel(xlabel)
plt.ylabel(f"Avg self-associations per polymer\n(last {N_LAST} timesteps)")
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(base_dir, "self_associations_total_vs_x.png"), dpi=300)
plt.close()


# ----------------------------
# plot 2: tensile force vs x (mean ± std over last N_LAST)
# ----------------------------
plt.figure(figsize=(4, 3))
plt.errorbar(x, t_mean, yerr=t_std, fmt="o-", capsize=3, label="mean ± std")
plt.xlabel(xlabel)
plt.ylabel(f"Avg tensile force (pN)\n(last {N_LAST} timesteps)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "tensile_force_stats_vs_x.png"), dpi=300)
plt.close()


# ----------------------------
# plot 3: added bonds per timestep (NO averaging)
# ----------------------------
plt.figure(figsize=(5, 3))
for Npol in sorted(added_timeseries.keys()):
    ts = added_timeseries[Npol]
    t = np.arange(len(ts))
    plt.plot(t, ts, label=f"N={Npol}")

plt.xlabel("Timestep")
plt.ylabel("Added self-associations per timestep")
plt.legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "self_associations_per_timestep_timeseries.png"), dpi=300)
plt.close()


# ----------------------------
# plot 4: avg tensile force over time (NO averaging), stacked subplots
# ----------------------------
Ns_sorted = sorted(tensile_avg_ts.keys())
nrows = len(Ns_sorted)

fig, axes = plt.subplots(nrows, 1, figsize=(6, 1.6 * nrows), sharex=True)
if nrows == 1:
    axes = [axes]

for ax, Npol in zip(axes, Ns_sorted):
    ts = tensile_avg_ts[Npol]
    t = np.arange(len(ts))
    ax.plot(t, ts)
    ax.set_ylabel(f"N={Npol}")

axes[-1].set_xlabel("Timestep")
fig.suptitle("Average tensile force over time (avg over monomers, no moving avg)", y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "tensile_force_avg_over_time_subplots.png"), dpi=300)
plt.close()
