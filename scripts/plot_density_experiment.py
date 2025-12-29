import numpy as np
import matplotlib.pyplot as plt
import os

def load_avg_last_timesteps(npz_path, N=20):
    data = np.load(npz_path)
    arr = data[list(data.keys())[0]]

    T = arr.shape[0]
    N = min(N, T)
    arr_last = arr[-N:, ...]

    if arr_last.ndim == 3:
        magnitudes = np.linalg.norm(arr_last, axis=-1)
    elif arr_last.ndim == 2:
        magnitudes = np.abs(arr_last)
    else:
        raise ValueError("Unexpected array shape", arr.shape)

    return magnitudes.mean(axis=0) * 1000.0  # pN


def load_npz_first(npz_path):
    data = np.load(npz_path)
    return data[list(data.keys())[0]]


# ----------------------------
# user settings
# ----------------------------
base_dir = "/home/nroebersen/espresso_project/experiments/density_experiment"   # <-- change
AREA = 112.0
USE_DENSITY_X = True
N_LAST = 20

# INCLUDE_NS = None
INCLUDE_NS = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
# INCLUDE_NS = [10, 40, 80]


# ----------------------------
# collect data
# ----------------------------
Ns = []
densities = []

tot_assocs = []
t_mean = []
t_min = []
t_max = []

added_timeseries = {}   # NEW: store full timeseries per N

for name in sorted(os.listdir(base_dir)):
    if not name.startswith("N_polymers_"):
        continue

    Npol = int(name.split("_")[-1])
    if INCLUDE_NS is not None and Npol not in INCLUDE_NS:
        continue

    folder = os.path.join(base_dir, name)

    Ns.append(Npol)
    densities.append(Npol / AREA)

    # ---- added bonds (NO AVERAGING)
    added = load_npz_first(os.path.join(folder, "added_bonds.npz"))
    tot_assocs.append(np.nansum(added))

    # store raw timeseries (collapse extra dims if needed)
    # If added is (T,), this does nothing. If (T,...) it sums per timestep.
    if added.ndim == 1:
        added_ts = added
    else:
        added_ts = np.nansum(added, axis=tuple(range(1, added.ndim)))  # -> (T,)
    added_timeseries[Npol] = added_ts

    # ---- tensile forces
    per_particle_eq = load_avg_last_timesteps(
        os.path.join(folder, "tensile_forces.npz"),
        N_LAST
    )
    t_mean.append(np.nanmean(per_particle_eq))
    t_min.append(np.nanmin(per_particle_eq))
    t_max.append(np.nanmax(per_particle_eq))


# ----------------------------
# sort by N
# ----------------------------
order = np.argsort(Ns)
Ns = np.array(Ns)[order]
densities = np.array(densities)[order]
tot_assocs = np.array(tot_assocs)[order]
norm_tot_assocs = tot_assocs/Ns
t_mean = np.array(t_mean)[order]
t_min = np.array(t_min)[order]
t_max = np.array(t_max)[order]

x = densities if USE_DENSITY_X else Ns
xlabel = r"Polymer density (N/$\mu$m$^2$)" if USE_DENSITY_X else r"Number of polymers"


# ----------------------------
# plot 1: total self associations vs x
# ----------------------------
plt.figure(figsize=(4, 3))
plt.plot(x, norm_tot_assocs, "o-")
plt.axhline(50, color='r',label="max associations")
plt.xlabel(xlabel)
plt.ylabel("Avg number self-associations")
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(base_dir, "self_associations_total_vs_x.png"), dpi=300)
plt.close()

# ----------------------------
# plot 2: tensile force stats vs x
# ----------------------------
plt.figure(figsize=(4, 3))
plt.plot(x, t_mean, "o-", label="mean")
#plt.plot(x, t_min, "--", label="min")
#plt.plot(x, t_max, "--", label="max")
plt.xlabel(xlabel)
plt.ylabel(f"Avg tensile force per monomer (pN)\n(avg last {N_LAST} timesteps)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "tensile_force_stats_vs_x.png"), dpi=300)
plt.close()


# ----------------------------
# plot 3: added bonds per timestep (NO averaging)  <-- NEW
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

folder = "/home/nroebersen/espresso_project/experiments/density_experiment/N_polymers_10"   # <-- change to ONE folder
f = os.path.join(folder, "added_bonds.npz")

data = np.load(f)
key = list(data.keys())[0]
arr = data[key]

print("File:", f)
print("Key:", key)
print("Array shape:", arr.shape)
print("Array dtype:", arr.dtype)
print()

# --- global info
print("Global min:", np.nanmin(arr))
print("Global max:", np.nanmax(arr))
print("Global sum:", np.nansum(arr))
print()

# --- first few timesteps (RAW)
print("First 10 timesteps (raw):")
print(arr[:10])
print()

# --- last few timesteps (RAW)
print("Last 10 timesteps (raw):")
print(arr[-10:])
print()

# --- per-timestep sums (ONLY if multidimensional)
if arr.ndim > 1:
    per_timestep_sum = np.nansum(arr, axis=tuple(range(1, arr.ndim)))
    print("Per-timestep sum:")
    print("  min:", per_timestep_sum.min())
    print("  max:", per_timestep_sum.max())
    print("  first 10:", per_timestep_sum[:10])
