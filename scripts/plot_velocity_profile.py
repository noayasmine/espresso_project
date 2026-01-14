import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------
# settings
# ------------------------------------------------
folder = "/home/nroebersen/espresso_project/experiments/test_visualization_flowfield_grid_z4_highres/selfassociation"
npz_file = os.path.join(folder, "velocity_profile.npz")
step = 0.1  # grid spacing in z

# ------------------------------------------------
# load data
# ------------------------------------------------
data = np.load(npz_file)
z_avg_vx = data["z_avg_vx"]  # (Nz,)

# reconstruct z axis
z = np.arange(len(z_avg_vx)) * step

# ------------------------------------------------
# compute gradient dv/dz
# ------------------------------------------------
dv_dz = np.gradient(z_avg_vx, step)

# ------------------------------------------------
# plot velocity profile
# ------------------------------------------------
plt.figure(figsize=(5, 4))
plt.plot(z, z_avg_vx, lw=2)
plt.xlabel("z")
plt.ylabel(r"$\langle v_x \rangle_{x,y}$")
plt.tight_layout()

outpath = os.path.join(folder, "velocity_profile.png")
plt.savefig(outpath, dpi=300)
plt.close()
print(f"Saved: {outpath}")

# ------------------------------------------------
# plot velocity gradient
# ------------------------------------------------
plt.figure(figsize=(5, 4))
plt.plot(z, dv_dz * 10 ** 4, lw=2)
plt.axhline(11636, color='red')
plt.axvline(0.5+0.03, color='green')
plt.xlabel("z")
plt.ylabel(r"$\mathrm{d}\langle v_x \rangle_{x,y} / \mathrm{d}z$")
plt.tight_layout()

outpath_grad = os.path.join(folder, "velocity_gradient_dvdz.png")
plt.savefig(outpath_grad, dpi=300)
plt.close()
print(f"Saved: {outpath_grad}")

