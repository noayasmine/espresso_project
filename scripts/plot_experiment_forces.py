import os
import numpy as np
import matplotlib.pyplot as plt

# =====================
# SETTINGS
# =====================
folders_to_plot = [
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N100/10/11636",
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N100/100/11636",
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N100/500/11636",
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N100/1000/11636",
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N50/500/11636",
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N250/500/11636",
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N50/500/5455",
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N100/500/5455",
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N250/500/5455"
]
# folders_to_plot = [
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N50/500/11636",
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N50/500/5455"

# ]

particle_id = 1
N_AVG = 20
N_MOV = 20
dpi = 300

tensile_filename = "tensile_forces.npz"
drag_filename = "drag_forces.npz"
total_filename = "forces.npz"


def load_npz_first(path):
    d = np.load(path)
    return d[d.files[0]]


def to_timeseries(arr):
    if arr.ndim == 3:
        return np.linalg.norm(arr, axis=-1)
    return arr


def moving_average_trailing_matrix(x, window):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    for i in range(x.shape[0]):
        start = max(0, i - window + 1)
        out[i] = x[start:i + 1].mean(axis=0)
    return out


def read_timeseries_TN(path, mov_window=None):
    arr = load_npz_first(path)
    ts = to_timeseries(arr)
    if mov_window is not None and mov_window > 1:
        ts = moving_average_trailing_matrix(ts, mov_window)
    return ts


def avg_last(arr_TN, N):
    last_forces = arr_TN[-min(N, arr_TN.shape[0]):]
    return last_forces.mean(axis=0), last_forces.std(axis=0)


def cumulative_from_last(vec):
    return np.flip(np.cumsum(np.flip(vec)))


def plot_timeseries(folder):
    fpath = os.path.join(folder, tensile_filename)
    if not os.path.isfile(fpath):
        print("missing:", fpath)
        return

    ts_raw = read_timeseries_TN(fpath, mov_window=None) * 1000.0 
    ts_mov = read_timeseries_TN(fpath, mov_window=N_MOV) * 1000.0

    sig_raw = ts_raw[:, particle_id]
    sig_mov = ts_mov[:, particle_id]
    t = np.arange(len(sig_raw)) * 0.1

    plt.figure(figsize=(6, 4))
    plt.plot(t, sig_mov, label=f"movavg {N_MOV}", color="cadetblue")
    plt.xlim(0,200)
    plt.xlabel("Time [ms]", fontsize=15)
    plt.ylabel("Tensile force [pN]", fontsize=15)
    plt.tight_layout()

    out = os.path.join(folder, f"timeseries_mon{particle_id}_mov{N_MOV}.png")
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("saved:", out)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    ax1.plot(t, sig_raw, color="cadetblue")
    ax1.set_ylabel("Tensile force [pN]", fontsize=13)
    ax1.grid(False)
    ax1.set_xlim(0,140)


    ax2.plot(t, sig_mov, color="cadetblue", lw=2)
    ax2.set_xlabel("Time [ms]", fontsize=13)
    ax2.set_ylabel("Tensile force [pN]", fontsize=13)
    ax2.grid(False)
    ax2.set_xlim(0,140)
    #ax2.set_ylim(-50,250)
    plt.tight_layout()

    out2 = os.path.join(folder, f"timeseries_mon{particle_id}_raw_vs_mov{N_MOV}.png")
    plt.savefig(out2, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out2)


def plot_profiles(folder):
    ft = os.path.join(folder, tensile_filename)
    fd = os.path.join(folder, drag_filename)
    ff = os.path.join(folder, total_filename)

    if not (os.path.isfile(ft) and os.path.isfile(fd) and os.path.isfile(ff)):
        print("missing one of:", folder)
        return

    tensile_ts = read_timeseries_TN(ft, mov_window=N_MOV)
    drag_ts = read_timeseries_TN(fd, mov_window=N_MOV)
    total_ts = read_timeseries_TN(ff, mov_window=N_MOV)

    tensile, tens_stdev = avg_last(tensile_ts, N_AVG)
    drag, drag_stdev = avg_last(drag_ts, N_AVG)
    total, total_stdev = avg_last(total_ts, N_AVG)

    tensile, tens_stdev = tensile * 1000.0, tens_stdev * 1000.0
    drag, drag_stdev = drag * 1000.0, drag_stdev * 1000.0
    total, total_stdev = total * 1000.0, total_stdev * 1000.0

    drag_cum = cumulative_from_last(drag)
    drag_stdev_cum = cumulative_from_last(drag_stdev)

    idx = np.arange(len(tensile))

    plt.figure(figsize=(6, 4))

    plt.plot(idx, tensile, label="tensile", color="C0")
    plt.fill_between(idx, tensile - tens_stdev, tensile + tens_stdev, color="C0", alpha=0.25)

    plt.plot(idx, drag_cum, label="cumulative drag", color="C1")
    plt.fill_between(idx, drag_cum - drag_stdev_cum, drag_cum + drag_stdev_cum, color="C1", alpha=0.25)

    plt.plot(idx, total, label="total", color="C2")
    plt.fill_between(idx, total - total_stdev, total + total_stdev, color="C2", alpha=0.25)

    N = len(tensile)
    ref_steps = np.arange(1, N + 1)
    ref = 0.5 * ref_steps[::-1]
    ref_std = 0.1 * ref_steps[::-1]

    plt.plot(idx, ref, "--", color="grey", lw=1.2, label="0.5 ± 0.1 pN / monomer")
    plt.fill_between(idx, ref - ref_std, ref + ref_std, color="grey", alpha=0.25)

    plt.xlim(0, tensile.shape[0])
    plt.ylim(0,)
    plt.xlabel("Monomer index")
    plt.ylabel("Force (pN)")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(folder, f"forces_profile_last{N_AVG}_mov{N_MOV}.png")
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("saved:", out)


for folder in folders_to_plot:
    plot_profiles(folder)
    plot_timeseries(folder)
