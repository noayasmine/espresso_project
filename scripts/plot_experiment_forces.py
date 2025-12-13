import numpy as np
import matplotlib.pyplot as plt
import os

# ================================================================
# FORCE LOADER UTILITIES  (key removed + auto-detect npz content)
# ================================================================

def load_avg_last_timesteps(npz_path, N=10):
    """
    Load the average of the last N timesteps from a .npz file
    containing a single array shaped (T,N), (T,N,1) or (T,N,3).
    Returns (Nparticles,) in pN.
    """
    data = np.load(npz_path)
    arr = data[list(data.keys())[0]]      # auto-detect stored key

    T = arr.shape[0]
    N = min(N, T)

    arr_last = arr[-N:, ...]

    if arr_last.ndim == 3:
        magnitudes = np.linalg.norm(arr_last, axis=-1)
    elif arr_last.ndim == 2:
        magnitudes = np.abs(arr_last)
    else:
        raise ValueError("Unexpected array shape", arr.shape)

    return magnitudes.mean(axis=0) * 1000.0


def load_timeseries(npz_path):
    data = np.load(npz_path)
    arr = data[list(data.keys())[0]]

    if arr.ndim == 3:
        magnitudes = np.linalg.norm(arr, axis=-1)
    elif arr.ndim == 2:
        magnitudes = np.abs(arr)
    else:
        raise ValueError("Unexpected array shape", arr.shape)

    return magnitudes * 1000.0


def cumulative_from_last(vec):
    """ cumulative sum from the last monomer toward index 0 """
    return np.flip(np.cumsum(np.flip(vec)))


# ================================================================
# PART 1 — ORIGINAL LJ vs NO-LJ FORCE PLOTS
# ================================================================

print("\n=== Running OLD LJ-vs-noLJ plot generation ===")

dir_A = "/home/nroebersen/espresso_project/experiments/try_exp/Nmono_50_Npol1_FENE_tensile_abs_1280dyncm2/selfassociation/"
dir_B = "/home/nroebersen/espresso_project/experiments/try_exp/Nmono_50_Npol1_FENE_tensile_abs_eps0_1280dyncm2/selfassociation/"

file_A_tensile = os.path.join(dir_A, "tensile_forces.npz")
file_A_drag    = os.path.join(dir_A, "drag_forces.npz")
file_A_total   = os.path.join(dir_A, "forces.npz")

file_B_tensile = os.path.join(dir_B, "tensile_forces.npz")
file_B_drag    = os.path.join(dir_B, "drag_forces.npz")
file_B_total   = os.path.join(dir_B, "forces.npz")

particle_id = 5
N_AVG = 20
N_MOV = 20

old_outdir = "/home/nroebersen/espresso_project/experiments/try_exp/LJ_vs_noLJ/"
os.makedirs(old_outdir, exist_ok=True)

# Load averaged forces
tensile_A = load_avg_last_timesteps(file_A_tensile, N_AVG)
drag_A    = load_avg_last_timesteps(file_A_drag,    N_AVG)
total_A   = load_avg_last_timesteps(file_A_total,   N_AVG)

tensile_B = load_avg_last_timesteps(file_B_tensile, N_AVG)
drag_B    = load_avg_last_timesteps(file_B_drag,    N_AVG)
total_B   = load_avg_last_timesteps(file_B_total,   N_AVG)

drag_A_cum = cumulative_from_last(drag_A)
drag_B_cum = cumulative_from_last(drag_B)

ts_A = load_timeseries(file_A_tensile)

# ----------------------------------------------------------------------
# LJ vs noLJ averaged forces
plt.figure(figsize=(6, 4))
plt.plot(tensile_A, label=r"$ε_{LJ}$=18 tensile")
plt.plot(drag_A_cum, label=r"$ε_{LJ}$=18 cumulative drag")
plt.plot(total_A, label=r"$ε_{LJ}$=18 total")

plt.plot(tensile_B, "--", label=r"$ε_{LJ}$=0 tensile")
plt.plot(drag_B_cum, "--", label=r"$ε_{LJ}$=0 cumulative drag")
plt.plot(total_B, "--", label=r"$ε_{LJ}$=0 total")

# plt.title("Forces per Particle (avg last steps)")
plt.xlabel("Monomer index")
plt.ylabel("Force (pN)")
plt.legend()
plt.xlim(-1,50)
plt.ylim(0,28)
# plt.grid(alpha=0.3)
plt.tight_layout()
old_plot1 = os.path.join(old_outdir, f"forces_avg_last_{N_AVG}.png")
plt.savefig(old_plot1, dpi=300, bbox_inches='tight')
plt.close()
print("Saved old plot:", old_plot1)

# ----------------------------------------------------------------------

signal = ts_A[:, particle_id]
mov_avg = np.convolve(signal, np.ones(N_MOV)/N_MOV, mode="valid")
timesteps = np.arange(len(signal))
timesteps_ma = np.arange(N_MOV - 1, len(signal))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(timesteps, signal)
# ax1.set_title(f"Tensile force over time (monomer {particle_id})")

ax2.plot(timesteps_ma, mov_avg)
# ax2.set_title("Moving average")
ax2.set_xlabel("Timestep")

old_plot2 = os.path.join(old_outdir, f"forces_particle_{particle_id}_movingavg.png")
plt.savefig(old_plot2, dpi=300, bbox_inches='tight')
plt.close()
print("Saved old plot:", old_plot2)


# ================================================================
# PART 2 — NEW SCRIPT: N=250 polymer, multiple k-values
# ================================================================

print("\n=== Running NEW k-dependent plots for N=250 ===")

base_dir_250 = "/home/nroebersen/espresso_project/experiments/try_exp/Nmono_250_Npol1_FENE_tensile_abs_1280dyncm2"

k_dirs = [
    "k_10",
    "k_100",
    "k_1000",
    "k_10000",
    "k_100000",
    "k_1000000"
]

particle_id_k = 5     # which monomer to inspect
N_AVG_K = 50
N_MOV_K = 50

new_outdir = os.path.join(base_dir_250, "k_plots")
os.makedirs(new_outdir, exist_ok=True)
print("Saving new k-plots to:", new_outdir)

for k_dir in k_dirs:
    sim_dir = os.path.join(base_dir_250, k_dir)

    f_tens = os.path.join(sim_dir, "tensile_forces.npz")
    f_drag = os.path.join(sim_dir, "drag_forces.npz")
    f_total = os.path.join(sim_dir, "forces.npz")

    if not all(os.path.isfile(f) for f in (f_tens, f_drag, f_total)):
        print("Skipping missing:", k_dir)
        continue

    print("\nProcessing:", k_dir)

    tensile = load_avg_last_timesteps(f_tens, N_AVG_K)
    drag    = load_avg_last_timesteps(f_drag, N_AVG_K)
    total   = load_avg_last_timesteps(f_total, N_AVG_K)

    drag_cum = cumulative_from_last(drag)

    # --- Plot 1 (forces for this k)
    plt.figure(figsize=(4,3))
    x = np.arange(0,250)
    x2 = np.arange(0,250)
    y =  0.5 * 250 - 0.5 * x2
    plt.plot(tensile, label="tensile")
    plt.plot(drag_cum, label="cumulative drag")
    plt.plot(total, label="total")
    plt.plot(x,y, label="0.5 pN/Mono")
    # plt.title(f"Forces per particle — {k_dir}")
    plt.xlabel("Monomer index")
    plt.ylabel("Force (pN)")
    plt.legend()
    # plt.grid(alpha=0.3)

    p1 = os.path.join(new_outdir, f"forces_{k_dir}.png")
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", p1)

    # --- Plot 2 (timeseries)
    ts = load_timeseries(f_tens)

    if particle_id_k >= ts.shape[1]:
        print("Skipping timeseries for", k_dir)
        continue

    sig = ts[:, particle_id_k]
    mov = np.convolve(sig, np.ones(N_MOV_K)/N_MOV_K, mode="valid")
    t = np.arange(len(sig))
    t_ma = np.arange(N_MOV_K - 1, len(sig))

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)

    ax1.plot(t, sig)#; ax1.set_title(f"Tensile force over time — {k_dir}")
    ax2.plot(t_ma, mov)#; ax2.set_title("Moving average")
    ax2.set_xlabel("Timestep")

    p2 = os.path.join(new_outdir, f"timeseries_{k_dir}.png")
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", p2)

print("\n=== All plots generated successfully ===")
