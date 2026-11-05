import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# settings
# ------------------------------------------------
# Tether force and chain height vs gamma for one monomer mass and LJ strength of the tethered N=100
# calibration (1280 dyn/cm2 = 128,000 1/s in 1 cP), all gammas (no tether-force cut).
# Heights are bead centres above the particle wall (z = 1 with NEARWALL_FIX).
experiments = "/home/nroebersen/espresso_project/experiments"
mass = 0.01
lj = "LJ_00400"
base = os.path.join(experiments, "gamma_mass", f"m_{mass}")
out_dir = os.path.join(experiments, "gamma_calibration")
target_shear = 128000
wall_z = 1.0
time_avg_steps = 100  # average over the last saved points
ref_per_mono, sigma_per_mono = 0.5, 0.1  # Fu et al. [pN per monomer]
fu_height = (0.010, 0.020)  # Fu et al. estimated height of VWF above the surface [um]


def gamma_of(folder):
    return float(re.match(r"g_([0-9.eE+-]+)$", os.path.basename(folder)).group(1))


rows = []  # gamma, tether mean, tether std, mean h, h tether end, h free end
for gdir in sorted(glob.glob(os.path.join(base, "g_*")), key=gamma_of):
    runs = sorted(glob.glob(os.path.join(gdir, lj, "gamma_*", f"shear_{target_shear}", "run_*")))
    tethers, heights = [], []
    for r in runs:
        try:
            drag = np.load(os.path.join(r, "drag_forces.npz"))["forces"][-time_avg_steps:]
            pos_file = np.load(os.path.join(r, "positions.npz"))
            pos = pos_file[list(pos_file.keys())[0]][-time_avg_steps:]
        except (FileNotFoundError, KeyError, OSError):
            continue
        tethers.append((np.linalg.norm(drag, axis=2) * 1000.0).mean(axis=0).sum())
        heights.append((pos[:, :, 2] - wall_z).mean(axis=0))
    if not tethers:
        print(f"{os.path.basename(gdir)}: no finished runs")
        continue
    h = np.mean(heights, axis=0)
    g = gamma_of(gdir)
    rows.append((g, np.mean(tethers), np.std(tethers), h.mean(), h[:10].mean(), h[-10:].mean()))
    print(f"gamma = {g:.6g}: runs={len(tethers)}, tether = {np.mean(tethers):.1f} +- {np.std(tethers):.1f} pN, "
          f"mean h = {h.mean():.3f} um (tether end {h[:10].mean():.3f}, free end {h[-10:].mean():.3f})")

g, f, s, h_mean, h_tether, h_end = np.array(rows).T
N = 100
eps_label = int(lj.split("_")[1]) / 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.errorbar(g, f, yerr=s, marker="o", ms=4, capsize=2, color="tab:blue", label="simulation")
ax1.axhline(ref_per_mono * N, color="grey", ls="--", label=r"Fu et al. $0.5 \pm 0.1$ pN/mono")
ax1.axhspan((ref_per_mono - sigma_per_mono) * N, (ref_per_mono + sigma_per_mono) * N, color="grey", alpha=0.2)
ax1.set_xscale("log")
ax1.set_xlabel(r"$\gamma$ [MD]")
ax1.set_ylabel("Tensile force at tether [pN]")
ax1.legend(fontsize=8)

ax2.plot(g, h_mean * 1000, marker="o", ms=4, color="tab:blue", label="mean over chain")
ax2.plot(g, h_tether * 1000, marker="s", ms=3, ls=":", color="tab:green", label="monomers 0-9 (tether)")
ax2.plot(g, h_end * 1000, marker="^", ms=3, ls=":", color="tab:red", label="monomers 90-99 (free end)")
ax2.axhspan(fu_height[0] * 1000, fu_height[1] * 1000, color="grey", alpha=0.3, label="Fu et al. estimate (10-20 nm)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$\gamma$ [MD]")
ax2.set_ylabel("Bead height above wall [nm]")
ax2.legend(fontsize=8)

fig.suptitle(fr"monomer mass {mass}, $\epsilon_{{LJ}}$ = {eps_label:g} $k_BT$, 128,000 s$^{{-1}}$", fontsize=10)
fig.tight_layout()
out = os.path.join(out_dir, f"gamma_tether_height_{lj}_m{mass}_shear_{target_shear}.png")
fig.savefig(out, dpi=300)
plt.close(fig)
print(f"Saved: {out}")
