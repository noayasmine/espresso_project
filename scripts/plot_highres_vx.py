import numpy as np
import matplotlib.pyplot as plt
import os
import re

def analyze_velocity_plane(folder, npz_name="velocity_plane_y4_step0p01.npz",step=0.01,x_target=7,z_targets=(0.53, 1.0, 1.3, 2.8),expected_shear=1.16):
    npz_file = os.path.join(folder, npz_name)
    data = np.load(npz_file)
    wall_z = 0.5
    a = 0.03
    vx_plane = data["vx_plane"]  # (Nx, Nz)
    print(vx_plane.shape)
    x_size, z_size = (vx_plane.shape[0]-1)*step, (vx_plane.shape[1]-1)*step
    print(x_size,z_size)

    x_target = x_target  # physical x position
    x_index = int(round(x_target / step))
    vx_z = vx_plane[x_index, :]



    # reconstruct z axis
    z = np.arange(vx_z.size) * step

    # plot vx(z)
    plt.figure(figsize=(5, 4))
    plt.plot(z, vx_z, lw=2)
    plt.xlabel("z")
    plt.ylabel(r"$v_x$")
    plt.tight_layout()

    outpath_v = os.path.join(folder, "velocity_profile_planey4.png")
    plt.savefig(outpath_v, dpi=300)
    plt.close()
    print(f"Saved: {outpath_v}")

    # gradient dvx/dz
    dvdz = np.gradient(vx_z, z) * 10 ** 4

    plt.figure(figsize=(5, 4))
    plt.plot(z, dvdz, lw=2)
    plt.xlabel("z")
    plt.ylabel(r"$dv_x/dz$")
    plt.axvline(wall_z + a, color="green", label="Polymer position")
    plt.axhline(expected_shear, color="red", label="Expected shear")
    plt.legend()
    plt.tight_layout()

    outpath_grad = os.path.join(folder, "velocity_gradient_dvdz_planey4.png")
    plt.savefig(outpath_grad, dpi=300)
    plt.close()
    print(f"Saved: {outpath_grad}")
    
    # ------------------------------------------------
    # 2D velocity plane: v_x(x, z)
    # ------------------------------------------------
    Nx, Nz = vx_plane.shape
    x = np.arange(Nx) * step
    z = np.arange(Nz) * step

    plt.figure(figsize=(x_size/10, z_size/10))
    plt.imshow(
        vx_plane.T,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], z[0], z[-1]]
    )
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar(label=r"$v_x$")
    plt.tight_layout()

    outpath_plane = os.path.join(folder, "velocity_plane_xz_planey4.png")
    plt.savefig(outpath_plane, dpi=300)
    plt.close()
    print(f"Saved: {outpath_plane}")

    # ------------------------------------------------
    # 2D gradient plane: dvx/dz (x, z)
    # ------------------------------------------------
    # gradient along z-axis (axis=1 because vx_plane.shape = (Nx, Nz))
    dvdz_plane = np.gradient(vx_plane, step, axis=1) * 1e4

    plt.figure(figsize=(x_size/10, z_size/10))
    plt.imshow(
        dvdz_plane.T,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], z[0], z[-1]])
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar(label=r"$dv_x/dz$")
    plt.tight_layout()

    outpath_grad_plane = os.path.join(folder, "velocity_gradient_plane_xz_planey4.png")
    plt.savefig(outpath_grad_plane, dpi=300)
    plt.close()
    print(f"Saved: {outpath_grad_plane}")




    # # print velocities at selected z
    # print(f"\nResults for folder: {folder}")
    # for z_target in z_targets:
    #     idx = int(round(z_target / step))
    #     vx_val = vx_z[idx]
    #     shear_est = vx_val / (z_target - wall_z) * 1e4
    #     print(f"  v_x at z = {z_target:.2f} : {vx_val:.6e}")
    #     print(f"  estimated shear        : {shear_est:.6e}")
    
    # ----------------------------
    # shear ratio at z = 0.53
    # ----------------------------
    z_ref = 0.53
    idx_ref = int(round(z_ref / step))

    shear_at_ref = dvdz[idx_ref]        # already scaled by 1e4
    ratio = shear_at_ref / expected_shear

    print( f"Shear ratio at z = {z_ref:.2f}: "
    f"{shear_at_ref:.6e} / {expected_shear:.6e} = {ratio:.4f}")



base_folder = "/home/nroebersen/espresso_project/validation/flowfields_testing/sphere/sphere_7_5_LBR"
step=0.01
x_target=5

# base_folder = "/home/nroebersen/espresso_project/validation/flowfields_testing/adjusted_runs"
# #base_folder = "/home/nroebersen/espresso_project/validation/flowfields_testing/unadjusted_runs"
# step=0.01
# x_target=7

# base_folder = "/home/nroebersen/espresso_project/validation/single_runs/validation_uncoil_timeseries/k1000/LJ_01800/gamma_00050"
# step=0.01
# x_target=7

for name in sorted(os.listdir(base_folder)):
    subfolder = os.path.join(base_folder, name)

    if not os.path.isdir(subfolder):
        continue

    # extract first floating-point number from folder name
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", name)
    if match is None:
        continue

    expected_shear = float(match.group())

    analyze_velocity_plane(
        folder=subfolder,
        expected_shear=expected_shear,
        step=step,
        x_target=x_target)

