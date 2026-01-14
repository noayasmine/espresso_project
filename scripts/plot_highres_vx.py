import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# load data
# ----------------------------
folder = "/home/nroebersen/espresso_project/experiments/test_visualization_flowfield_grid_z4_highres/selfassociation"
npz_file = os.path.join(folder, "velocity_plane_y4_step0p01.npz")

data = np.load(npz_file)
vx_plane = data["vx_plane"]  # (Nx, Nz)

# average over x → function of z only
vx_z = vx_plane.mean(axis=0)

# reconstruct z axis (step = 0.01)
step = 0.01
z = np.arange(vx_z.size) * step

# ----------------------------
# plot vx(z)
# ----------------------------
plt.figure(figsize=(5, 4))
plt.plot(z, vx_z, lw=2)
plt.xlabel("z")
plt.ylabel(r"$v_x$")
plt.tight_layout()
outpath_v = os.path.join(folder, "velocity_profile_planey4.png")
plt.savefig(outpath_v, dpi=300)
plt.close()
print(f"Saved: {outpath_v}")

# ----------------------------
# gradient dvx/dz
# ----------------------------
dvdz = np.gradient(vx_z, z)  # uses nonuniform-safe spacing; here z is uniform anyway

plt.figure(figsize=(5, 4))
plt.plot(z, dvdz, lw=2)
plt.xlabel("z")
plt.ylabel(r"$dv_x/dz$")
plt.axvline(0.5+0.03, color='green', label="Polymer position")
plt.axhline(1.16, color='red', label="Expected shear")
plt.tight_layout()
plt.legend()
outpath_grad = os.path.join(folder, "velocity_gradient_dvdz_planey4.png")
plt.savefig(outpath_grad, dpi=300)
plt.close()
print(f"Saved: {outpath_grad}")

# ----------------------------
# print velocity at z = 0.53
# ----------------------------
z_target = 0.53
idx = int(round(z_target / step))
print(f"v_x at z = {z_target:.2f} = {vx_z[idx]:.6e}")
print(f"dvdz= {vx_z[idx]/(z_target-0.5) * 10**4}")

z_target = 1
idx = int(round(z_target / step))
print(f"v_x at z = {z_target:.2f} = {vx_z[idx]:.6e}")
print(f"dvdz= {vx_z[idx]/(z_target-0.5) * 10**4}")

z_target = 1.3
idx = int(round(z_target / step))
print(f"v_x at z = {z_target:.2f} = {vx_z[idx]:.6e}")
print(f"dvdz= {vx_z[idx]/(z_target-0.5) * 10**4}")

z_target = 2.8
idx = int(round(z_target / step))
print(f"v_x at z = {z_target:.2f} = {vx_z[idx]:.6e}")
print(f"dvdz= {vx_z[idx]/(z_target-0.5) * 10**4}")
