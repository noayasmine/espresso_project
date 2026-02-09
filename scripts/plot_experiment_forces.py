import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# ks = [10, 100, 500, 1000, 5000, 10000]
# Ns = [100]
# folders_to_plot = []
# for N in Ns:
#     for k in ks:
#         folders_to_plot.append(f"/home/nroebersen/espresso_project/validation/forces_gamma_00130/experiment_forces_N{N}/{k}/11636")

# folders_to_plot = [
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N100/10/11636",
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N100/100/11636",
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N100/500/11636",
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N100/1000/11636",
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N50/500/11636",
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N250/500/11636",
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N50/500/5455",
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N100/500/5455",
# "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N250/500/5455"
# ]
folders_to_plot = [
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N50/500/11636",
"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N50/500/5455"
]

# ks = [10, 100, 500, 1000, 5000, 10000]
# Ns = [100, 250]
# folders_to_plot = []
# for N in Ns:
#     for k in ks:
#         folders_to_plot.append(f"/home/nroebersen/espresso_project/validation/forces/experiment_forces_N{N}/{k}/11636")

# particle_id = 50
# N_AVG = 100
# N_MOV = 20
# dpi = 300

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

    plt.plot(idx, tensile, label="tensile", color="cadetblue")
    plt.fill_between(idx, tensile - tens_stdev, tensile + tens_stdev, color="cadetblue", alpha=0.25)

    plt.plot(idx, drag_cum, label="cumulative drag", color="purple")
    plt.fill_between(idx, drag_cum - drag_stdev_cum, drag_cum + drag_stdev_cum, color="purple", alpha=0.25)

    # plt.plot(idx, total, label="total", color="C2")
    # plt.fill_between(idx, total - total_stdev, total + total_stdev, color="C2", alpha=0.25)

    N = len(tensile)
    ref_steps = np.arange(1, N + 1)
    ref = 0.5 * ref_steps[::-1]
    ref_std = 0.1 * ref_steps[::-1]

    #plt.plot(idx, ref, "--", color="grey", lw=1.2, label="0.5 ± 0.1 pN / monomer")
    plt.fill_between(idx, ref - ref_std, ref + ref_std, color="grey", alpha=0.2)

    plt.xlim(0, tensile.shape[0])
    plt.ylim(0,)
    plt.xlabel("Monomer index",size=12)
    plt.ylabel("Force [pN]",size=12)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(folder, f"forces_profile_last{N_AVG}_mov{N_MOV}.png")
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("saved:", out)

particle_id = 5
N_AVG = 100
N_MOV = 20
dpi = 300

for folder in folders_to_plot:
    plot_profiles(folder)
    plot_timeseries(folder)


ks = [500, 1000, 5000, 10000]
Ns = [100, 250]

run_id = "11636"
base_dir = "/home/nroebersen/espresso_project/validation/forces"

particle_id = 50
N_AVG = 100
N_MOV = 1
dpi = 300

tensile_filename = "tensile_forces.npz"
drag_filename = "drag_forces.npz"
total_filename = "forces.npz"


def plot_tensile_profiles_onefig_per_N(Ns, ks, base_dir, run_id="11636", outdir=None):
    if outdir is None:
        outdir = base_dir

    for N in Ns:
        fig, axes = plt.subplots(
            len(ks), 1,
            figsize=(4, 1.5 * len(ks)),
            sharex=True)

        if len(ks) == 1:
            axes = [axes]

        global_ymax = 0.0
        profiles = []

        # ------------------------------------------------
        # first pass: load all data (for shared y-limits)
        # ------------------------------------------------
        for k in ks:
            folder = os.path.join(base_dir, f"experiment_forces_N{N}", str(k), run_id)
            ft = os.path.join(folder, tensile_filename)

            if not os.path.isfile(ft):
                profiles.append(None)
                continue

            tensile_ts = read_timeseries_TN(ft, mov_window=N_MOV)
            mean, std = avg_last(tensile_ts, N_AVG)

            mean *= 1000.0  # pN
            std  *= 1000.0

            global_ymax = max(global_ymax, np.max(mean + std))
            profiles.append((mean, std))

        # ------------------------------------------------
        # plot
        # ------------------------------------------------
        for ax, k, prof in zip(axes, ks, profiles):
            if prof is None:
                ax.text(0.5, 0.5, f"missing k={k}",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            mean, std = prof
            idx = np.arange(len(mean))

            ax.plot(idx, mean, lw=2, color="purple", label=r"$\kappa$"+f"={k}")
            ax.fill_between(
                idx,
                mean - std,
                mean + std,
                color="purple",
                alpha=0.18)

            # reference line
            ref_steps = np.arange(1, len(mean) + 1)
            ref = 0.5 * ref_steps[::-1]
            ref_std = 0.1 * ref_steps[::-1]

            ax.plot(idx, ref, "--", color="grey", lw=1.2, label="Reference")
            ax.fill_between(
                idx,
                ref - ref_std,
                ref + ref_std,
                color="grey",
                alpha=0.15)

            ax.set_ylabel("Force [pN]", fontsize=11)
            ax.set_ylim(0, 1.05 * global_ymax)
            ax.grid(False)
            ax.legend()

        axes[-1].set_xlabel("Monomer index", fontsize=13)
        axes[-1].set_xlim(0, int(N))

        fig.tight_layout(h_pad=0.15)

        outname = os.path.join(
            outdir,
            f"tensile_profile_stackk_N{N}_last{N_AVG}_mov{N_MOV}_with_stdev.png")
        fig.savefig(outname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print("saved:", outname)


# plot_tensile_profiles_onefig_per_N(Ns, ks, base_dir, run_id="11636")




def plot_profiles_multi(folders, labels, outpath, N_AVG=100, N_MOV=20, dpi=300, show_ref=True):
    """
    Plot tensile + cumulative drag + total vs monomer index for multiple folders on ONE figure.

    Encoding:
      - color      → folder (model: kBT=18 vs NoLJ)
      - linestyle  → force type
          tensile         : dashed
          cumulative drag : solid
          total           : dotted
    """
    if len(folders) != len(labels):
        raise ValueError("folders and labels must have same length")

    fig, ax = plt.subplots(figsize=(6, 4))

    # fixed colors per folder
    base_colors = ["purple", "cadetblue"]
    if len(folders) > len(base_colors):
        base_colors = (base_colors * ((len(folders) + 1) // len(base_colors)))[:len(folders)]

    for folder, lab, col in zip(folders, labels, base_colors):
        ft = os.path.join(folder, tensile_filename)
        fd = os.path.join(folder, drag_filename)
        ff = os.path.join(folder, total_filename)

        if not (os.path.isfile(ft) and os.path.isfile(fd) and os.path.isfile(ff)):
            print("missing one of:", folder)
            continue

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

        # tensile — dashed
        ax.plot(idx, tensile, color=col, lw=2.0, ls="-", label=f"tensile {lab}")
        ax.fill_between(idx, tensile - tens_stdev, tensile + tens_stdev, color=col, alpha=0.18)

        # # cumulative drag — solid
        # ax.plot(idx, drag_cum, color=col, lw=2.0, ls="-", label=f"cumulative drag {lab}")
        # ax.fill_between(idx, drag_cum - drag_stdev_cum, drag_cum + drag_stdev_cum, color=col, alpha=0.18)

        # total — dotted
        ax.plot(idx, total, color=col, lw=2.0, ls="--", label=f"total {lab}")
        ax.fill_between(idx, total - total_stdev, total + total_stdev, color=col, alpha=0.14)

    if show_ref:
        Nref = None
        for folder in folders:
            ft = os.path.join(folder, tensile_filename)
            if os.path.isfile(ft):
                tmp = read_timeseries_TN(ft, mov_window=None)
                Nref = tmp.shape[1]
                break
        if Nref is not None:
            ref_steps = np.arange(1, Nref + 1)
            ref = 0.5 * ref_steps[::-1]
            ref_std = 0.1 * ref_steps[::-1]
            idx_ref = np.arange(Nref)
            ax.plot(idx_ref, ref, "--", color="grey", lw=1.2, label="0.5 ± 0.1 pN / monomer")
            ax.fill_between(idx_ref, ref - ref_std, ref + ref_std, color="grey", alpha=0.20)

    ax.set_xlim(0,50)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Monomer index",size=12)
    ax.set_ylabel("Force [pN]",size=12)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved:", outpath)

# ----------------------------
# settings + run
# ----------------------------
N_AVG = 100
N_MOV = 20
dpi = 300

folders = [
    "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N50/500/11636",
    "/home/nroebersen/espresso_project/validation/forces/experiment_forces_N50_NoLJ/500/11636",
]
labels = [
    r"$\epsilon_{LJ}=18$",
    r"$\epsilon_{LJ}=0$",
]

outpath = "/home/nroebersen/espresso_project/validation/forces/forces_profile_compare_kBT18_vs_kBT0.png"
plot_profiles_multi(folders, labels, outpath, N_AVG=N_AVG, N_MOV=N_MOV, dpi=dpi, show_ref=False)
