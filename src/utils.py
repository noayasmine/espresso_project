import os
import numpy as np

def save_results(config, lbf, accumulator_forces, accumulator_positions):
    shear = config["simulation"]["SHEAR"]
    loops = config["simulation"]["LOOPS"]
    BOX_Lx = config["geometry"]["BOX_Lx"]
    BOX_Ly = config["geometry"]["BOX_Ly"]
    BOX_Lz = config["geometry"]["BOX_Lz"]
    outdir = config["output"]["DIRECTORY"]

    os.makedirs(outdir, exist_ok=True)

    np.savez_compressed(os.path.join(outdir, "positions.npz"), positions=accumulator_positions.time_series())
    np.savez_compressed(os.path.join(outdir, "forces.npz"), forces=accumulator_forces.time_series())

    # Compute velocity profile
    # step = 0.1
    # x = np.arange(0, BOX_Lx + step, step)
    # y = np.arange(0, BOX_Ly + step, step)
    # z = np.arange(0, BOX_Lz + step, step)
    # xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # pos_grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    # velocities = np.array([lbf.get_interpolated_velocity(p) for p in pos_grid])
    # velocities_reshaped = velocities.reshape(len(x), len(y), len(z), 3)
    # vx_only = velocities_reshaped[..., 0]
    # z_avg_vx = vx_only.mean(axis=(0, 1))

    # np.savez_compressed(os.path.join(outdir, "velocity_profile.npz"), z_avg_vx=z_avg_vx)
    # np.savez_compressed(os.path.join(outdir, "z_positions.npz"), z=z)

    print(f"Simulation complete. Forces shape: {accumulator_forces.time_series().shape}")
    print(f"Positions shape: {accumulator_positions.time_series().shape}")
    print(f"Results saved in: {outdir}")


