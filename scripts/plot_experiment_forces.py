import numpy as np
import matplotlib.pyplot as plt
import os

def load_avg_last_timesteps(npz_path, N=10, key="forces"):
    """
    Load the average of the last N timesteps of per-particle forces.
    Assumes shape (T, Nparticles, 1).
    Returns array (Nparticles,).
    """
    data = np.load(npz_path)
    arr = data[key]                # shape (T, N, 1)
    T = arr.shape[0]
    N = min(N, T)                  # safety clamp

    print(f"Loaded: {npz_path}, shape={arr.shape}, averaging last {N} steps")

    # avg over last N steps, keep shape (N,1), multiply by 1000
    avg_last = arr[-N:, :].mean(axis=0) * 1000.0
    return avg_last


def load_timeseries(npz_path, key="forces"):
    """
    Load full time series (T, Nparticles).
    """
    data = np.load(npz_path)
    arr = data[key]                # shape (T, N, 1)

    # keep shape (T, N, 1), multiply by 1000
    return arr[:, :] * 1000.0


# ----------------------------------------------------------------------
# User parameters
file_A = "/home/nroebersen/espresso_project/experiments/try_exp/Nmono_50_Npol1_FENE_tensile_abs_1280dyncm2/selfassociation/tensile_forces.npz"
file_B = "/home/nroebersen/espresso_project/experiments/try_exp/Nmono_50_Npol1_FENE_tensile_abs_eps0_1280dyncm2/selfassociation/tensile_forces.npz"

key = "forces"
particle_id = 10           # which particle to track over time
N_AVG = 50                 # average over last N timesteps
N_MOV = 20                 # MOVING AVERAGE WINDOW
outdir = "/home/nroebersen/espresso_project/experiments/try_exp/"
# ----------------------------------------------------------------------

# Load averaged last N timesteps
forces_A = load_avg_last_timesteps(file_A, N_AVG, key)
forces_B = load_avg_last_timesteps(file_B, N_AVG, key)

# Load full timeseries (A only needed for subplot)
ts_A = load_timeseries(file_A, key)   # shape (T, N, 1)
# ts_B not needed for timeseries plot

# ----------------------------------------------------------------------
# Plot 1: Averaged tensile forces per particle over last N timesteps
plt.figure(figsize=(10, 5))
plt.plot(forces_A, label=r"$\epsilon_{LJ}=18$" + f" (avg last {N_AVG})", linewidth=1.5)
plt.plot(forces_B, label=r"$\epsilon_{LJ}=0$" + f" (avg last {N_AVG})", linewidth=1.5)

plt.title(f"Tensile Forces per Particle (Averaged over Last {N_AVG} Timesteps) at "
          + r"1280 dyn $\mathrm{cm^{-2}}$")
plt.xlabel("Particle ID")
plt.ylabel("Tensile Force pN")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
out1 = os.path.join(outdir, f"forces_avg_last_{N_AVG}.png")
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()
print("Saved:", out1)

# ----------------------------------------------------------------------
# Plot 2: Raw timeseries of A and its moving average (subplots)
signal = ts_A[:, particle_id]  # shape (T,)

# Moving average
kernel = np.ones(N_MOV) / N_MOV
mov_avg = np.convolve(signal, kernel, mode="valid")

timesteps = np.arange(signal.shape[0])
timesteps_ma = np.arange(N_MOV - 1, signal.shape[0])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top: raw signal
ax1.plot(timesteps, signal, linewidth=1.5)
ax1.set_title(f"Tensile Force Over Time – Particle {particle_id} (Dataset A)")
ax1.set_ylabel("Tensile Force pN")
ax1.grid(alpha=0.3)

# Bottom: moving average
ax2.plot(timesteps_ma, mov_avg, linewidth=2.0)
ax2.set_title(f"Moving Average (window = {N_MOV}) – Particle {particle_id}")
ax2.set_xlabel("Timestep")
ax2.set_ylabel("Tensile Force pN")
ax2.grid(alpha=0.3)

plt.tight_layout()
out2 = os.path.join(outdir, f"forces_particle_{particle_id}_movingavg.png")
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.close()
print("Saved:", out2)
