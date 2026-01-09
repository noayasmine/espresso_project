import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------
# settings
# ------------------------------------------------
folder = "/home/nroebersen/espresso_project/experiments/test_visualization_flowfield_npz/selfassociation"
npz_file = os.path.join(folder, "velocity_profile.npz")
step = 0.1  # must match the grid spacing used when generating the file

# ------------------------------------------------
# load data
# ------------------------------------------------
data = np.load(npz_file)
z_avg_vx = data["z_avg_vx"]  # (Nz,)

# reconstruct z axis
z = np.arange(len(z_avg_vx)) * step

# ------------------------------------------------
# plot
# ------------------------------------------------
plt.figure(figsize=(5, 4))
plt.plot(z, z_avg_vx, lw=2)
plt.xlabel("z")
plt.ylabel(r"$\langle v_x \rangle_{x,y}$")
plt.tight_layout()

# ------------------------------------------------
# save
# ------------------------------------------------
outpath = os.path.join(folder, "velocity_profile.png")
plt.savefig(outpath, dpi=300)
plt.close()

print(f"Saved: {outpath}")
