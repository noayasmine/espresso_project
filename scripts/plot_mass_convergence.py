import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# settings
# ------------------------------------------------
# monomer mass convergence at the gamma calibration point (N=100, LJ 18, gamma 0.001, 128,000 1/s in 1 cP)
base = "/home/nroebersen/espresso_project/experiments"
sub = os.path.join("LJ_01800", "gamma_00010", "shear_128000")
mass_dirs = {
    1.0: os.path.join(base, "gamma_calibration", sub),   # ESPResSo default, from the first gamma sweep
    0.1: os.path.join(base, "mass_test", "m_0.1", sub),
    0.01: os.path.join(base, "mass_test", "m_0.01", sub),
    0.001: os.path.join(base, "mass_test", "m_0.001", sub),
}
wall_z = 1.0          # LB wall with NEARWALL_FIX (AGRID = 1)
dt_ms = 0.1           # time between saved frames [ms]
avg_window = 20       # trailing moving average over saved points
time_avg_steps = 100  # average over the last saved points
ref_per_mono = 0.5    # Fu et al. [pN per monomer]


def moving_average_trailing_matrix(x, window):
    out = np.empty_like(x, dtype=float)
    for i in range(x.shape[0]):
        start = max(0, i - window + 1)
        out[i] = x[start:i + 1].mean(axis=0)
    return out


fig, (ax_t, ax_e, ax_h) = plt.subplots(1, 3, figsize=(14, 4))
colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(mass_dirs)))

for (mass, shear_dir), color in zip(mass_dirs.items(), colors):
    if not os.path.isdir(shear_dir):
        print(f"m = {mass}: missing {shear_dir}")
        continue
    runs = sorted(d for d in os.listdir(shear_dir) if re.match(r"^run_\d+$", d))
    profiles, ext_late, h_late = [], [], []
    for k, r in enumerate(runs):
        F = np.linalg.norm(np.load(os.path.join(shear_dir, r, "drag_forces.npz"))["forces"], axis=2) * 1000.0
        P = np.load(os.path.join(shear_dir, r, "positions.npz"))["positions"]
        per_run = moving_average_trailing_matrix(F, avg_window)[-time_avg_steps:].mean(axis=0)
        profiles.append(np.cumsum(per_run[::-1])[::-1])
        ext = np.linalg.norm(P[:, -1] - P[:, 0], axis=1)
        height = P[:, :, 2].mean(axis=1) - wall_z
        half = len(ext) // 2
        ext_late.append((ext[half:].mean(), ext[half:].std()))
        h_late.append(height[half:].mean())
        t = np.arange(len(ext)) * dt_ms
        ax_e.plot(t, ext, color=color, lw=0.8, alpha=0.8, label=f"m={mass}" if k == 0 else None)
        ax_h.plot(t, height, color=color, lw=0.8, alpha=0.8, label=f"m={mass}" if k == 0 else None)
    cum = np.vstack(profiles)
    N = cum.shape[1]
    mu = cum.mean(axis=0)
    sig = cum.std(axis=0, ddof=1) if cum.shape[0] > 1 else np.zeros(N)
    e_mean = np.mean([e for e, _ in ext_late]); e_fluct = np.mean([s for _, s in ext_late])
    print(f"m = {mass}: runs={cum.shape[0]}, tether force={mu[0]:.1f} +- {sig[0]:.1f} pN (ref {ref_per_mono * N:.0f}), "
          f"end-to-end (2nd half)={e_mean:.2f} um, fluctuation={e_fluct:.2f} um, mean height (2nd half)={np.mean(h_late):.3f} um")
    ax_t.errorbar(np.arange(N), mu, yerr=sig, color=color, lw=1.2, elinewidth=0.5, capsize=2, label=f"m={mass}")

steps = N - np.arange(N)
ax_t.plot(np.arange(N), ref_per_mono * steps, "--", color="grey", label="Fu et al.")
ax_t.set_xlabel("Monomer index"); ax_t.set_ylabel("Tensile force [pN]")
ax_e.set_xlabel("Time [ms]"); ax_e.set_ylabel("End-to-end distance [um]")
ax_h.set_xlabel("Time [ms]"); ax_h.set_ylabel("Mean height above wall [um]")
for ax in (ax_t, ax_e, ax_h):
    ax.legend(fontsize=7)
plt.tight_layout()
out = os.path.join(base, "mass_test", "mass_convergence.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=300)
print(f"Saved: {out}")
