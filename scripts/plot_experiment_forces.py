import os
import numpy as np
import matplotlib.pyplot as plt

# =====================
# SETTINGS
# =====================
folders_to_plot = [
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_50_Npol1_FENE_tensile_abs_600dyncm2/selfassociation",
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_50_Npol1_FENE_tensile_abs_1280dyncm2/selfassociation",
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_100_Npol1_FENE_tensile_abs_600dyncm2/selfassociation",
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_100_Npol1_FENE_tensile_abs_1280dyncm2/selfassociation",
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_250_Npol1_FENE_tensile_abs_600dyncm2/selfassociation",
    "/home/nroebersen/espresso_project/experiments/forces_exp/Nmono_250_Npol1_FENE_tensile_abs_1280dyncm2/k_1000",
]

particle_id = 5
N_AVG = 20          # avg over last N timesteps for profile plot
N_MOV = 20          # moving avg window for timeseries
dpi = 300

tensile_filename = "tensile_forces.npz"
drag_filename    = "drag_forces.npz"
total_filename   = "forces.npz"


def load_npz_first(path):
    d = np.load(path)
    return d[d.files[0]]


def to_timeseries(arr):
    """Return (T,N) timeseries in 'native' units (no abs for scalars)."""
    if arr.ndim == 3:
        return np.linalg.norm(arr, axis=-1)  # (T,N) magnitude
    return arr                               # (T,N) signed


def avg_last(arr_TN, N):
    return arr_TN[-min(N, arr_TN.shape[0]):].mean(axis=0)


def cumulative_from_last(vec):
    return np.flip(np.cumsum(np.flip(vec)))


def plot_timeseries(folder):
    fpath = os.path.join(folder, tensile_filename)
    if not os.path.isfile(fpath):
        print("missing:", fpath)
        return

    arr = load_npz_first(fpath)
    ts = to_timeseries(arr) * 1000.0  # pN

    sig = ts[:, particle_id]

    # simple trailing window moving average (same style as your sim)
    mov = np.empty_like(sig, dtype=float)
    for i in range(len(sig)):
        start = max(0, i - N_MOV + 1)
        mov[i] = sig[start:i+1].mean()

    t = np.arange(len(sig))

    plt.figure(figsize=(6, 4))
    plt.plot(t, sig, label="raw")
    plt.plot(t, mov, label=f"movavg {N_MOV}")
    plt.xlabel("timestep")
    plt.ylabel("tensile force (pN)")
    plt.title(f"{os.path.basename(folder)} | monomer {particle_id}")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(folder, f"timeseries_mon{particle_id}_mov{N_MOV}.png")
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("saved:", out)


def plot_profiles(folder):
    ft = os.path.join(folder, tensile_filename)
    fd = os.path.join(folder, drag_filename)
    ff = os.path.join(folder, total_filename)

    if not (os.path.isfile(ft) and os.path.isfile(fd) and os.path.isfile(ff)):
        print("missing one of:", folder)
        return

    tensile = avg_last(to_timeseries(load_npz_first(ft)), N_AVG) * 1000.0   # signed if scalar
    drag    = avg_last(to_timeseries(load_npz_first(fd)), N_AVG) * 1000.0
    total   = avg_last(to_timeseries(load_npz_first(ff)), N_AVG) * 1000.0

    drag_cum = cumulative_from_last(drag)

    plt.figure(figsize=(6, 4))
    plt.plot(tensile, label="tensile (signed if scalar)")
    plt.plot(drag_cum, label="cumulative drag")
    plt.plot(total, label="total")
    plt.xlabel("monomer index")
    plt.ylabel("force (pN)")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(folder, f"forces_profile_last{N_AVG}.png")
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("saved:", out)


# =====================
# RUN
# =====================
for folder in folders_to_plot:
    plot_profiles(folder)
    plot_timeseries(folder)
