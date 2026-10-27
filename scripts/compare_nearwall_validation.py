import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# settings
# ------------------------------------------------
# tethered N=100 calibration runs (Fig. 5 setup): stock build vs patched build + NEARWALL_FIX
base = "/home/nroebersen/espresso_project/experiments/nearwall_test"
case_dirs = {
    "stock 4.2.2": os.path.join(base, "validation_stock", "LJ_01800", "gamma_00110", "shear_11636"),
    "nearwall-fix": os.path.join(base, "validation_fix", "LJ_01800", "gamma_00110", "shear_11636"),
}
avg_window = 20       # trailing moving average over saved points
time_avg_steps = 100  # average over the last saved points
ref_per_mono, sigma_per_mono = 0.5, 0.1  # Fu et al. [pN per monomer]


def moving_average_trailing_matrix(x, window):
    out = np.empty_like(x, dtype=float)
    for i in range(x.shape[0]):
        start = max(0, i - window + 1)
        out[i] = x[start:i + 1].mean(axis=0)
    return out


def tensile_profiles(shear_dir):
    """Cumulative drag force from the free end, one row per run [pN]."""
    runs = sorted(d for d in os.listdir(shear_dir) if re.match(r"^run_\d+$", d))
    profiles = []
    for r in runs:
        F = np.load(os.path.join(shear_dir, r, "drag_forces.npz"))["forces"]
        Fmag = np.linalg.norm(F, axis=2) * 1000.0
        per_run = moving_average_trailing_matrix(Fmag, avg_window)[-time_avg_steps:].mean(axis=0)
        profiles.append(np.cumsum(per_run[::-1])[::-1])
    return np.vstack(profiles)


plt.figure(figsize=(6, 4))
for label, shear_dir in case_dirs.items():
    if not os.path.isdir(shear_dir):
        print(f"Missing: {shear_dir}")
        continue
    cum = tensile_profiles(shear_dir)
    N = cum.shape[1]
    steps = N - np.arange(N)
    ref, sigma_ref = ref_per_mono * steps, sigma_per_mono * steps
    mu = cum.mean(axis=0)
    sig = cum.std(axis=0, ddof=1) if cum.shape[0] > 1 else np.zeros(N)
    t2 = np.mean(((mu - ref) / (sig + sigma_ref)) ** 2)
    print(f"{label}: runs={cum.shape[0]}, tether force={mu[0]:.1f} +- {sig[0]:.1f} pN "
          f"(ref {ref[0]:.0f}), mean ratio to ref={np.mean(mu / ref):.3f}, t2={t2:.3f}")
    plt.errorbar(np.arange(N), mu, yerr=sig, lw=1.2, elinewidth=0.5, capsize=2, label=f"{label} ($t^2$={t2:.3f})")

plt.plot(np.arange(N), ref, "--", color="grey", label=r"Fu et al. $0.5 \pm 0.1$ pN/mono")
plt.fill_between(np.arange(N), ref - sigma_ref, ref + sigma_ref, color="grey", alpha=0.2)
plt.xlabel("Monomer index")
plt.ylabel("Tensile force [pN]")
plt.legend()
plt.tight_layout()
out = os.path.join(base, "validation_tensile_comparison.png")
plt.savefig(out, dpi=300)
print(f"Saved: {out}")
