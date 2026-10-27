import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------
# settings
# ------------------------------------------------
base = "/home/nroebersen/espresso_project/experiments/nearwall_test"
folders = {
    "stock 4.2.2": os.path.join(base, "stock", "11636"),
    "nearwall-fix": os.path.join(base, "nearwall_fix", "11636"),
}
step = 0.1  # grid spacing in z

profiles = {}
for label, folder in folders.items():
    # ------------------------------------------------
    # load data
    # ------------------------------------------------
    npz_file = os.path.join(folder, "velocity_profile.npz")
    if not os.path.exists(npz_file):
        print(f"Missing: {npz_file}")
        continue
    z_avg_vx = np.load(npz_file)["z_avg_vx"]  # (Nz,)

    # reconstruct z axis
    z = np.arange(len(z_avg_vx)) * step

    # ------------------------------------------------
    # compute gradient dv/dz
    # ------------------------------------------------
    dv_dz = np.gradient(z_avg_vx, step)
    profiles[label] = (z, z_avg_vx, dv_dz)

# ------------------------------------------------
# plot velocity profile
# ------------------------------------------------
plt.figure(figsize=(5, 4))
for label, (z, z_avg_vx, _) in profiles.items():
    plt.plot(z, z_avg_vx, lw=2, label=label)
plt.xlabel("z")
plt.ylabel(r"$\langle v_x \rangle_{x,y}$")
plt.legend()
plt.tight_layout()

outpath = os.path.join(base, "velocity_profile.png")
plt.savefig(outpath, dpi=300)
plt.close()
print(f"Saved: {outpath}")

# ------------------------------------------------
# plot velocity gradient
# ------------------------------------------------
plt.figure(figsize=(5, 4))
for label, (z, _, dv_dz) in profiles.items():
    plt.plot(z, dv_dz * 10 ** 4, lw=2, label=label)
plt.axhline(11636, color='red', label="Expected shear")
plt.axvline(0.5+0.03, color='green', label="Polymer position")
plt.xlabel("z")
plt.ylabel(r"$\mathrm{d}\langle v_x \rangle_{x,y} / \mathrm{d}z$")
plt.legend()
plt.tight_layout()

outpath_grad = os.path.join(base, "velocity_gradient_dvdz.png")
plt.savefig(outpath_grad, dpi=300)
plt.close()
print(f"Saved: {outpath_grad}")

