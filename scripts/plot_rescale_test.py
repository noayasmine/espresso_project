import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# settings
# ------------------------------------------------
# Viscosity rescaling test of the tethered N=100 gamma calibration (Fig. 5 setup, 1280 dyn/cm2, LJ 18):
#   A: old run (11 cP, gamma 0.011, m 1, 11636 1/s, dt = LB dt = 0.001)
#   B: same run with the time unit divided by 11 (identical LB numbers, fluid density / 121)
#   C: physical 1 cP buffer (water density, LB dt = 11 MD steps)
# B and C: gamma 0.001, m 1/121, 128000 1/s, dt 1e-3/11. All save every 1000 steps (same strain per frame).
experiments = "/home/nroebersen/espresso_project/experiments"
cases = {  # label: (shear dir, shear rate [1/s], MD time step [MD], color)
    r"A: old, 11 cP, $\gamma$=0.011, m=1": (
        os.path.join(experiments, "nearwall_test", "validation_fix", "LJ_01800", "gamma_00110", "shear_11636"),
        11636, 1e-3, "tab:blue"),
    r"B: rescaled, $\gamma$=0.001, m=1/121, $\rho$/121": (
        os.path.join(experiments, "rescale_test", "B_exact", "LJ_01800", "gamma_00010", "shear_128000"),
        128000, 1e-3 / 11, "tab:orange"),
    r"C: rescaled, 1 cP water, $\gamma$=0.001, m=1/121": (
        os.path.join(experiments, "rescale_test", "C_water", "LJ_01800", "gamma_00010", "shear_128000"),
        128000, 1e-3 / 11, "tab:green"),
}
out = os.path.join(experiments, "gamma_calibration", "rescale_test_LJ_01800_N100.png")
steps_per_frame = 1000
time_unit = 1e-4      # s per MD time unit
wall_z = 1.0          # particle wall with NEARWALL_FIX
avg_window = 20       # trailing moving average over saved points
time_avg_steps = 100  # average over the last saved points
ref_per_mono, sigma_per_mono = 0.5, 0.1  # Fu et al. [pN per monomer]
unstable_drag = 100.0  # [pN] runs with a drag spike above this are left out (as in plot_gamma_calibration.py)


def moving_average_trailing(x, window):
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    out = np.empty_like(x, dtype=float)
    for i in range(x.shape[0]):
        start = max(0, i - window + 1)
        out[i] = (c[i + 1] - c[start]) / (i + 1 - start)
    return out


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
N = None
for label, (shear_dir, shear, dt, color) in cases.items():
    profiles, tether_ts, ext_ts = [], [], []
    for r in sorted(d for d in os.listdir(shear_dir) if re.match(r"^run_\d+$", d)):
        F = np.linalg.norm(np.load(os.path.join(shear_dir, r, "drag_forces.npz"))["forces"], axis=2) * 1000.0
        pos_file = np.load(os.path.join(shear_dir, r, "positions.npz"))
        pos = pos_file[list(pos_file.keys())[0]]
        if F.max() > unstable_drag:
            print(f"{label}: {r} unstable (max |F_drag| = {F.max():.0f} pN), left out")
            continue
        per = moving_average_trailing(F, avg_window)[-time_avg_steps:].mean(axis=0)
        profiles.append(np.cumsum(per[::-1])[::-1])
        tether_ts.append(F.sum(axis=1))
        ext_ts.append(pos[:, -1, 0] - pos[:, 0, 0])
        h = (pos[-time_avg_steps:, :, 2] - wall_z).mean()
        print(f"{label}: {r} tether {profiles[-1][0]:.1f} pN, mean height {h:.3f} um")
    prof = np.array(profiles)
    N = prof.shape[1]
    steps = N - np.arange(N)
    ref, sigma_ref = ref_per_mono * steps, sigma_per_mono * steps
    mu, sig = prof.mean(axis=0), prof.std(axis=0, ddof=1)
    t2 = np.mean(((mu - ref) / (sig + sigma_ref)) ** 2)
    print(f"{label}: {len(prof)} stable runs, tether {mu[0]:.1f} +- {sig[0]:.1f} pN, t2 = {t2:.3f}")
    ax1.errorbar(np.arange(N), mu, yerr=sig, color=color, lw=1.2, elinewidth=0.5, capsize=2,
                 label=fr"{label} ($t^2_{{{N}}}$={t2:.3f}, n={len(prof)})")

    n = min(len(t) for t in tether_ts)
    strain = (np.arange(n) + 1) * steps_per_frame * dt * time_unit * shear
    ax2.plot(strain, np.mean([e[:n] for e in ext_ts], axis=0) / (N - 1) / 0.06, color=color, lw=1.2, label=label)
    ax3.plot(strain, moving_average_trailing(np.mean([t[:n] for t in tether_ts], axis=0), avg_window),
             color=color, lw=1.2, label=label)

ax1.plot(np.arange(N), ref, "--", color="grey", label=r"Fu et al. $0.5 \pm 0.1$ pN/mono")
ax1.fill_between(np.arange(N), ref - sigma_ref, ref + sigma_ref, color="grey", alpha=0.2)
ax1.set_xlabel("Monomer index")
ax1.set_ylabel("Tensile force [pN]")
ax1.set_title("Steady-state tension profile (last 100 frames)", fontsize=9)
ax1.legend(fontsize=6.5)

ax2.set_xlabel(r"Strain $\dot\gamma t$")
ax2.set_ylabel("Extension along flow / contour length")
ax2.set_title("Uncoiling", fontsize=9)
ax2.legend(fontsize=6.5)

ax3.axhspan((ref_per_mono - sigma_per_mono) * N, (ref_per_mono + sigma_per_mono) * N, color="grey", alpha=0.2)
ax3.set_yscale("log")
ax3.set_xlabel(r"Strain $\dot\gamma t$")
ax3.set_ylabel("Tensile force at tether [pN] (moving avg 20)")
ax3.set_title("Approach to steady state", fontsize=9)
ax3.legend(fontsize=6.5)

fig.suptitle(r"Viscosity rescaling test: N=100, $\epsilon_{LJ}$=18 $k_BT$, 1280 dyn/cm$^2$", fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=300)
plt.close(fig)
print(f"Saved: {out}")
