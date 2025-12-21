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
# NEW: TIMESERIES PLOT FUNCTION FOR A GIVEN FOLDER
# ================================================================

def plot_tensile_timeseries_in_folder(
    folder,
    particle_id=0,
    N_MOV=50,
    tensile_filename="tensile_forces.npz",
    outname=None,
    dpi=300,
    outname_ma=None,
    fraction=0.5,          # NEW: plot only this fraction of timesteps (0 < fraction <= 1)
    which_part="first",    # NEW: "first" or "last"
):
    """
    Load tensile_forces timeseries from `folder/tensile_forces.npz`,
    plot raw signal + moving average for `particle_id`,
    and save the plot(s) inside `folder`.

    fraction:
      - 1.0 plots all timesteps
      - 0.5 plots half the timesteps (first/last half depending on which_part)

    which_part:
      - "first": take the first fraction of the series
      - "last":  take the last fraction of the series

    Produces:
      1) timeseries_particle_<id>_mov<N_MOV>.png           (unless outname is given)
      2) timeseries_particle_<id>_mov<N_MOV>_onlyMA.png    (unless outname_ma is given)
    """
    folder = os.path.abspath(folder)
    f_tens = os.path.join(folder, tensile_filename)

    if not os.path.isfile(f_tens):
        print(f"[timeseries] Missing file: {f_tens}")
        return None

    ts = load_timeseries(f_tens)  # (T, Nmono) in pN

    if ts.ndim != 2:
        raise ValueError(f"[timeseries] Expected (T,N) after load_timeseries, got {ts.shape}")

    if particle_id < 0 or particle_id >= ts.shape[1]:
        print(f"[timeseries] particle_id={particle_id} out of range (0..{ts.shape[1]-1}) for {folder}")
        return None

    # --- select signal
    sig_full = ts[:, particle_id]

    # --- slice timesteps
    if fraction is None:
        fraction = 1.0
    if not (0 < float(fraction) <= 1.0):
        raise ValueError(f"[timeseries] fraction must be in (0,1], got {fraction}")

    T_full = len(sig_full)
    T_use = max(1, int(np.floor(T_full * float(fraction))))

    which_part = str(which_part).lower().strip()
    if which_part not in ("first", "last"):
        raise ValueError(f"[timeseries] which_part must be 'first' or 'last', got {which_part}")

    if T_use == T_full:
        sig = sig_full
        offset_idx = 0
    else:
        if which_part == "first":
            start, end = 0, T_use
        else:  # "last"
            start, end = T_full - T_use, T_full
        sig = sig_full[start:end]
        offset_idx = start  # used only if you ever want original-index time; here we just re-zero time

    # time axis (you currently label ms, but your scaling is 0.1 s per step; keep as you wrote)
    # If you truly want ms, change 0.1 -> 100 (ms) or set dt_ms explicitly.
    t = np.arange(len(sig)) * 0.1  # seconds

    # --- moving average on the sliced signal
    if N_MOV is None or N_MOV <= 1:
        mov = None
        t_ma = None
    else:
        if N_MOV > len(sig):
            print(f"[timeseries] N_MOV={N_MOV} > T_use={len(sig)} in {folder}; reducing N_MOV to {len(sig)}")
            N_MOV = len(sig)
        mov = np.convolve(sig, np.ones(N_MOV) / N_MOV, mode="valid")
        t_ma = np.arange(N_MOV - 1, len(sig)) * 0.1  # seconds

    # -------------------------
    # Figure 1: 2-panel plot
    # -------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    ax1.plot(t, sig)
    ax1.set_ylabel("Tensile force (pN)")

    if mov is not None:
        ax2.plot(t_ma, mov)
        ax2.set_ylabel("Tensile force (pN)")
    else:
        ax2.plot(t, sig)
        ax2.set_ylabel("Tensile force (pN)")

    ax2.set_xlabel("Time (ms)")  # your label; adjust if desired

    if outname is None:
        tag = "" if fraction == 1.0 else f"_{which_part}{fraction:g}"
        outname = f"timeseries_particle_{particle_id}_mov{N_MOV}{tag}.png"

    outpath = os.path.join(folder, outname)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("[timeseries] Saved:", outpath)

    # -------------------------
    # Figure 2: moving-average-only plot
    # -------------------------
    outpath_ma = None
    if mov is not None:
        fig_ma, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.plot(t_ma, mov)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Tensile force (pN)")

        if outname_ma is None:
            tag = "" if fraction == 1.0 else f"_{which_part}{fraction:g}"
            outname_ma = f"timeseries_particle_{particle_id}_mov{N_MOV}{tag}_onlyMA.png"

        outpath_ma = os.path.join(folder, outname_ma)
        plt.tight_layout()
        plt.savefig(outpath_ma, dpi=dpi, bbox_inches="tight")
        plt.close()
        print("[timeseries] Saved (MA only):", outpath_ma)
    else:
        print("[timeseries] No moving average computed (N_MOV <= 1); skipping MA-only plot.")

    return outpath, outpath_ma

# ================================================================
# PART 1 — ORIGINAL LJ vs NO-LJ FORCE PLOTS
# ================================================================

print("\n=== Running OLD LJ-vs-noLJ plot generation ===")

dir_A = "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_50_Npol1_FENE_tensile_abs_1280dyncm2/selfassociation/"
dir_B = "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_50_Npol1_FENE_tensile_abs_eps0_1280dyncm2/selfassociation/"

file_A_tensile = os.path.join(dir_A, "tensile_forces.npz")
file_A_drag    = os.path.join(dir_A, "drag_forces.npz")
file_A_total   = os.path.join(dir_A, "forces.npz")

file_B_tensile = os.path.join(dir_B, "tensile_forces.npz")
file_B_drag    = os.path.join(dir_B, "drag_forces.npz")
file_B_total   = os.path.join(dir_B, "forces.npz")

particle_id = 5
N_AVG = 20
N_MOV = 20

old_outdir = "/home/nroebersen/espresso_project/experiments/forces_exp/LJ_vs_noLJ/"
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

plt.xlabel("Monomer index")
plt.ylabel("Force (pN)")
plt.legend()
plt.xlim(-1, 50)
plt.ylim(0, 28)
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

ax2.plot(timesteps_ma, mov_avg)
ax2.set_xlabel("Timestep")

old_plot2 = os.path.join(old_outdir, f"forces_particle_{particle_id}_movingavg.png")
plt.savefig(old_plot2, dpi=300, bbox_inches='tight')
plt.close()
print("Saved old plot:", old_plot2)


# ================================================================
# PART 2 — N=250 polymer, multiple k-values
# ================================================================

print("\n=== Running NEW k-dependent plots for N=250 ===")

base_dir_250 = "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_250_Npol1_FENE_tensile_abs_1280dyncm2"

k_dirs = [
    "k_10",
    "k_100",
    "k_1000",
    "k_10000",
    "k_100000",
    "k_1000000"
]

particle_id_k = 0     # which monomer to inspect
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
    plt.figure(figsize=(4, 3))
    x = np.arange(0, 250)
    x2 = np.arange(0, 250)
    y = 0.5 * 250 - 0.5 * x2
    plt.plot(tensile, label="tensile")
    plt.plot(drag_cum, label="cumulative drag")
    plt.plot(total, label="total")
    plt.plot(x, y, label="0.5 pN/Mono")
    plt.xlabel("Monomer index")
    plt.ylabel("Force (pN)")
    plt.legend()

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t, sig)
    ax2.plot(t_ma, mov)
    ax2.set_xlabel("Timestep")

    p2 = os.path.join(new_outdir, f"timeseries_{k_dir}.png")
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", p2)

print("\n=== All plots generated successfully ===")


# ================================================================
# EXTRA: GENERATE TIMESERIES PLOTS DIRECTLY INSIDE YOUR TWO FOLDERS
# ================================================================

print("\n=== Running folder-based timeseries plots ===")

folders_to_plot = [
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_50_Npol1_FENE_tensile_abs_600dyncm2/selfassociation",
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_50_Npol1_FENE_tensile_abs_1280dyncm2/selfassociation",
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_100_Npol1_FENE_tensile_abs_600dyncm2/selfassociation",
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_100_Npol1_FENE_tensile_abs_1280dyncm2/selfassociation",
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_250_Npol1_FENE_tensile_abs_600dyncm2/selfassociation",
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_250_Npol1_FENE_tensile_abs_1280dyncm2/k_1000"
]

# pick what you want here:
particle_id_folder = 5
N_MOV_FOLDER = 20

for folder in folders_to_plot:
    plot_tensile_timeseries_in_folder(
        folder=folder,
        particle_id=particle_id_folder,
        N_MOV=N_MOV_FOLDER,
        tensile_filename="tensile_forces.npz",
        outname=None
    )
