import os
import numpy as np
import vtk
from vtkmodules.util import numpy_support

def save_results(config, lbf, accumulator_forces, drag_forces, tensile_forces, tensile_vec_forces, added_bonds_per_step, accumulator_positions):
    shear = config["simulation"]["SHEAR"]
    loops = config["simulation"]["LOOPS"]
    save_VTK = config["simulation"]["SAVE_VTK"]
    BOX_Lx = config["geometry"]["BOX_Lx"]
    BOX_Ly = config["geometry"]["BOX_Ly"]
    BOX_Lz = config["geometry"]["BOX_Lz"]
    outdir = config["output"]["DIRECTORY"]

    os.makedirs(outdir, exist_ok=True)

    np.savez_compressed(os.path.join(outdir, "positions.npz"), positions=accumulator_positions.time_series())
    np.savez_compressed(os.path.join(outdir, "drag_forces.npz"), forces=drag_forces)
    np.savez_compressed(os.path.join(outdir, "forces.npz"), forces=accumulator_forces.time_series())
    np.savez_compressed(os.path.join(outdir, "tensile_forces.npz"), forces=tensile_forces)
    # np.savez_compressed(os.path.join(outdir, "tensile_forces_vec.npz"), forces=tensile_vec_forces)
    np.savez_compressed(os.path.join(outdir, "added_bonds.npz"), bonds=added_bonds_per_step)

    # Compute velocity profile
    step = 0.1
    x = np.arange(0., BOX_Lx + step, step) # (N,)
    y = np.arange(0., BOX_Ly + step, step) # (N,)
    z = np.arange(0., BOX_Lz + step, step) # (N,)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    pos_grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    velocities = np.array([lbf.get_interpolated_velocity(p) for p in pos_grid])
    velocities_reshaped = velocities.reshape(len(x), len(y), len(z), 3)
    vx_only = velocities_reshaped[..., 0]
    z_avg_vx = vx_only.mean(axis=(0, 1))

    # np.savez_compressed(os.path.join(outdir, "velocity_profile.npz"), z_avg_vx=z_avg_vx)

    # ----------------------------
    # new: one x–z plane at y = 4 (same style)
    # ----------------------------
    y_slice = int(BOX_Ly/2)
    step_fine = 0.01

    x_fine = np.arange(0., BOX_Lx + step_fine, step_fine)   # (Nx,)
    z_fine = np.arange(0., BOX_Lz + step_fine, step_fine)   # (Nz,)

    xx2, zz2 = np.meshgrid(x_fine, z_fine, indexing='ij')   # both (Nx, Nz)
    yy2 = np.full_like(xx2, y_slice, dtype=float)           # (Nx, Nz)

    pos_plane = np.vstack([xx2.ravel(), yy2.ravel(), zz2.ravel()]).T  # (Nx*Nz, 3)
    vel_plane = np.array([lbf.get_interpolated_velocity(p) for p in pos_plane])

    vel_plane_reshaped = vel_plane.reshape(len(x_fine), len(z_fine), 3)  # (Nx, Nz, 3)
    vx_plane = vel_plane_reshaped[..., 0]                                # (Nx, Nz)

    # np.savez_compressed(os.path.join(outdir, "velocity_plane_y4_step0p01.npz"),
    #         vx_plane=vx_plane)

    # save vti for visualization in paraview
    if save_VTK:
        velocities_reshaped = np.ascontiguousarray(velocities_reshaped)
        vel = np.reshape(velocities_reshaped, (-1, 3), order="F")  # (Nx*Ny*Nz, 3)
        vtk_vel = numpy_support.numpy_to_vtk(
        num_array=vel,
        deep=True,
        array_type=vtk.VTK_DOUBLE)
        vtk_vel.SetName("velocity")
        Nx, Ny, Nz, _ = velocities_reshaped.shape

        grid = vtk.vtkImageData()
        grid.SetDimensions(Nx, Ny, Nz)
        grid.SetSpacing(step, step, step)
        grid.SetOrigin(0.0, 0.0, 0.0)

        grid.GetPointData().SetVectors(vtk_vel)

        writer = vtk.vtkXMLImageDataWriter()
        vtk_path = os.path.join(outdir, "velocity_field.vti")
        writer.SetFileName(vtk_path)
        writer.SetInputData(grid)
        writer.Write()

    print(f"Simulation complete. Forces shape: {accumulator_forces.time_series().shape}")
    print(f"Positions shape: {accumulator_positions.time_series().shape}")
    print(f"Results saved in: {outdir}")

def save_velocity_field(config, lbf):
    shear = config["simulation"]["SHEAR"]
    loops = config["simulation"]["LOOPS"]
    BOX_Lx = config["geometry"]["BOX_Lx"]
    BOX_Ly = config["geometry"]["BOX_Ly"]
    BOX_Lz = config["geometry"]["BOX_Lz"]
    outdir = config["output"]["DIRECTORY"]

    os.makedirs(outdir, exist_ok=True)

    # Compute velocity profile
    if config["geometry"]["SPHERE"]:
        step = 1
    if not config["geometry"]["SPHERE"]:
        step = 0.1
    x = np.arange(0., BOX_Lx + step, step) # (N,)
    y = np.arange(0., BOX_Ly + step, step) # (N,)
    z = np.arange(0., BOX_Lz + step, step) # (N,)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    pos_grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    velocities = np.array([lbf.get_interpolated_velocity(p) for p in pos_grid])
    velocities_reshaped = velocities.reshape(len(x), len(y), len(z), 3)
    vx_only = velocities_reshaped[..., 0]
    z_avg_vx = vx_only.mean(axis=(0, 1))

    # np.savez_compressed(os.path.join(outdir, "velocity_profile.npz"), z_avg_vx=z_avg_vx)

    # ----------------------------
    # new: one x–z plane at y = 4 (same style)
    # ----------------------------
    if not config["geometry"]["SPHERE"]:
        y_slice = int(BOX_Ly/2)
        step_fine = 0.01
    
    if config["geometry"]["SPHERE"]:
        y_slice = int(BOX_Ly/2)
        step_fine = 0.1

    x_fine = np.arange(0., BOX_Lx + step_fine, step_fine)   # (Nx,)
    z_fine = np.arange(0., BOX_Lz + step_fine, step_fine)   # (Nz,)

    xx2, zz2 = np.meshgrid(x_fine, z_fine, indexing='ij')   # both (Nx, Nz)
    yy2 = np.full_like(xx2, y_slice, dtype=float)           # (Nx, Nz)

    pos_plane = np.vstack([xx2.ravel(), yy2.ravel(), zz2.ravel()]).T  # (Nx*Nz, 3)
    vel_plane = np.array([lbf.get_interpolated_velocity(p) for p in pos_plane])

    vel_plane_reshaped = vel_plane.reshape(len(x_fine), len(z_fine), 3)  # (Nx, Nz, 3)
    vx_plane = vel_plane_reshaped[..., 0]                                # (Nx, Nz)

    np.savez_compressed(os.path.join(outdir, "velocity_plane_y4_step0p01.npz"), vx_plane=vx_plane)

    # save vti for visualization in paraview
    velocities_reshaped = np.ascontiguousarray(velocities_reshaped)
    vel = np.reshape(velocities_reshaped, (-1, 3), order="F")  # (Nx*Ny*Nz, 3)
    vtk_vel = numpy_support.numpy_to_vtk(
    num_array=vel,
    deep=True,
    array_type=vtk.VTK_DOUBLE)
    vtk_vel.SetName("velocity")
    Nx, Ny, Nz, _ = velocities_reshaped.shape

    grid = vtk.vtkImageData()
    grid.SetDimensions(Nx, Ny, Nz)
    grid.SetSpacing(step, step, step)
    grid.SetOrigin(0.0, 0.0, 0.0)

    grid.GetPointData().SetVectors(vtk_vel)

    writer = vtk.vtkXMLImageDataWriter()
    vtk_path = os.path.join(outdir, "velocity_field.vti")
    writer.SetFileName(vtk_path)
    writer.SetInputData(grid)
    writer.Write()

    print(f"Results saved in: {outdir}")



