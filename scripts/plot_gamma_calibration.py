import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# settings
# ------------------------------------------------
# gamma sweep of the tethered N=100 calibration (Fig. 5 setup) at 1280 dyn/cm2 = 128,000 1/s in 1 cP buffer
lj_dir = "/home/nroebersen/espresso_project/experiments/gamma_calibration/LJ_01800"
target_shear = 128000
avg_window = 20       # trailing moving average over saved points
time_avg_steps = 100  # average over the last saved points
ref_per_mono, sigma_per_mono = 0.5, 0.1  # Fu et al. [pN per monomer]
unstable_drag = 100.0  # [pN] runs with a drag spike above this are reported and left out of the average


def moving_average_trailing_matrix(x, window):
    out = np.empty_like(x, dtype=float)
    for i in range(x.shape[0]):
        start = max(0, i - window + 1)
        out[i] = x[start:i + 1].mean(axis=0)
    return out


def tensile_profiles(shear_dir):
    """Cumulative drag force from the free end, one row per stable run [pN]."""
    runs = sorted(d for d in os.listdir(shear_dir) if re.match(r"^run_\d+$", d))
    profiles = []
    for r in runs:
        path = os.path.join(shear_dir, r, "drag_forces.npz")
        if not os.path.exists(path):
            print(f"  missing: {path}")
            continue
        Fmag = np.linalg.norm(np.load(path)["forces"], axis=2) * 1000.0
        per_run = moving_average_trailing_matrix(Fmag, avg_window)[-time_avg_steps:].mean(axis=0)
        cum = np.cumsum(per_run[::-1])[::-1]
        if Fmag.max() > unstable_drag:
            print(f"  unstable: {r} (max |F_drag| = {Fmag.max():.0f} pN, tether force {cum[0]:.1f} pN)")
            continue
        profiles.append(cum)
    return np.vstack(profiles) if profiles else None


gamma_dirs = sorted((d for d in os.listdir(lj_dir) if re.match(r"^gamma_\d+$", d)), key=lambda g: int(g.split("_")[1]))
colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, max(len(gamma_dirs), 1)))

plt.figure(figsize=(6, 4))
for gdir, color in zip(gamma_dirs, colors):
    gamma = int(gdir.split("_")[1]) / 1e4
    shear_dir = os.path.join(lj_dir, gdir, f"shear_{target_shear}")
    if not os.path.isdir(shear_dir):
        continue
    print(f"gamma = {gamma:.4f}")
    cum = tensile_profiles(shear_dir)
    if cum is None:
        print("  no stable runs")
        continue
    N = cum.shape[1]
    steps = N - np.arange(N)
    ref, sigma_ref = ref_per_mono * steps, sigma_per_mono * steps
    mu = cum.mean(axis=0)
    sig = cum.std(axis=0, ddof=1) if cum.shape[0] > 1 else np.zeros(N)
    t2 = np.mean(((mu - ref) / (sig + sigma_ref)) ** 2)
    print(f"  stable runs={cum.shape[0]}, tether force={mu[0]:.1f} +- {sig[0]:.1f} pN (ref {ref[0]:.0f}), "
          f"drag per monomer={mu[0] / N:.3f} pN, t2={t2:.3f}")
    plt.errorbar(np.arange(N), mu, yerr=sig, color=color, lw=1.2, elinewidth=0.5, capsize=2,
                 label=fr"$\gamma$={gamma:.4f} ($t^2_{{{N}}}$={t2:.3f})")

plt.plot(np.arange(N), ref, "--", color="grey", label=r"Fu et al. $0.5 \pm 0.1$ pN/mono")
plt.fill_between(np.arange(N), ref - sigma_ref, ref + sigma_ref, color="grey", alpha=0.2)
plt.xlabel("Monomer index")
plt.ylabel("Tensile force [pN]")
plt.legend(fontsize=7)
plt.tight_layout()
out = os.path.join(lj_dir, f"gamma_calibration_shear_{target_shear}.png")
plt.savefig(out, dpi=300)
print(f"Saved: {out}")
