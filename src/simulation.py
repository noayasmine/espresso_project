import logging
import sys
import numpy as np
import espressomd
import espressomd.lb
import espressomd.polymer
import espressomd.io.writer.vtf 
import espressomd.lbboundaries
import espressomd.shapes
import espressomd.observables
import os
import time

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def update_moving_average(buffer, new_value, window_size):
    """Update list buffer with new_value and return mean over window."""
    buffer.append(new_value)
    if len(buffer) > window_size:
        buffer.pop(0)  # remove oldest
    return np.mean(buffer, axis=0)


def build_polymers(system, run_type, n_monomers, n_polymers, start_positions, positions, fene, bond_length, TETHERED=True):

    if run_type=="validation_uncoiling":
        logging.info("Building coiled up linear polymer (validation uncoiling).")
        positions = espressomd.polymer.linear_polymer_positions(n_polymers=n_polymers, 
            beads_per_chain=n_monomers, start_positions=start_positions, respect_constraints=True,
            bond_length=bond_length, seed=42)
    
    for j in range(n_polymers):
        p_previous = None
        first_bead = None

        for i, pos in enumerate(positions[j]):
            p = system.part.add(pos=pos, type=0)
            if i == 0:
                first_bead = p
            if p_previous is not None:
                p.add_bond((fene, p_previous))
            p_previous = p

            if TETHERED==True:
                first_bead.fix = [True, True, True]


def run_simulation(params,run_type):
    # Unpack from structured config
    sim = params["simulation"]
    phys = params["physical"]
    geom = params["geometry"]
    fluid = params["fluid"]
    out = params["output"]

    # Geometry and runtime
    BOX_Lx, BOX_Ly, BOX_Lz = geom["BOX_Lx"], geom["BOX_Ly"], geom["BOX_Lz"]
    AGRID = geom["AGRID"]

    TIME_STEP, LOOPS, STEPS = sim["TIME_STEP"], sim["LOOPS"], sim["STEPS"]
    N_POLYMERS = sim["N_POLYMERS"]
    POSITIONS = sim["POSITIONS"]
    START_POSITIONS = sim["START_POSITIONS"]
    TETHERED = sim["TETHERED"]
    SHEAR = sim["SHEAR"] * 1e-4  # Convert units if needed
    save_VTK = sim["SAVE_VTK"]
    # minimization_steps = int(100 * (1 / AGRID) ** 2)
    minimization_steps = sim["MINIMIZATION_STEPS"]
    # print(sim["MONOMER_COUNTS"])

    # Physical parameters
    KT = phys["KT"]
    GAMMA = phys["GAMMA"]
    N_MONOMERS = phys["N_MONOMERS"]
    k, r_0, d_r_max = phys["k"], phys["r_0"], phys["d_r_max"]
    epsilon, sigma, cutoff, shift = phys["epsilon"], phys["sigma"], phys["cutoff"], phys["shift"]
    
    # Fluid properties
    VISC, DENS = fluid["VISC"], fluid["DENS"]

    # Output and boundary
    DIRECTORY = out["DIRECTORY"]
    geo_sphere = geom.get("SPHERE", False)
    geo_cylinder = geom.get("CYLINDER", False)
    radius_object = geom.get("RADIUS_OBJECT", 3)

    VELOCITY = SHEAR * (BOX_Lz - AGRID)
    VELOCITY2 = sim["VELOCITY"]
    outdir = DIRECTORY + f'vtk_data/'
    if save_VTK:
        os.makedirs(outdir, exist_ok=True)

    logging.info(f"Running simulation: shear={int(SHEAR*1e4)}, n_polymers={N_POLYMERS}, n_monomers={N_MONOMERS}, gridsize={AGRID}, timestep={TIME_STEP}")
    logging.info(f"Running simulation: VELOCITY={VELOCITY}, VELOCITY={VELOCITY2}")

    system = espressomd.System(box_l=[BOX_Lx, BOX_Ly, BOX_Lz], periodicity=[True, True, False])
    
    # print("Total ranks:", system.cell_system.get_state()["n_nodes"])
    print(system.cell_system.get_state())

    system.cell_system.skin = 0.4
    system.time_step = TIME_STEP

    # Non-bonded interactions
    system.non_bonded_inter[0, 0].lennard_jones.set_params(epsilon=epsilon, sigma=sigma, shift=shift, cutoff=cutoff)
    fene = espressomd.interactions.FeneBond(k=k, r_0=r_0, d_r_max=d_r_max)
    system.bonded_inter.add(fene)
    logging.info(f"LJ parameters: epsilon={epsilon}, sigma={sigma}, shift={shift}, cutoff={cutoff}")

    # Wall constraints
    wall_shape_z_up = espressomd.shapes.Wall(normal=[0, 0, 1], dist=AGRID * 0.5)
    wall_shape_z_down = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(BOX_Lz - AGRID * 0.5))
    system.constraints.add(particle_type=1, penetrable=False, shape=wall_shape_z_up)
    system.constraints.add(particle_type=1, penetrable=False, shape=wall_shape_z_down)

    if geo_sphere:
        logging.info("Adding Sphere.")
        sphere_constraint = espressomd.shapes.Sphere(radius=radius_object - 0.5, center=[BOX_Lx / 4, BOX_Ly / 2, 0.5], direction=1)
        system.constraints.add(particle_type=1, penetrable=False, shape=sphere_constraint)
        sphere = espressomd.shapes.Sphere(radius=radius_object, center=[BOX_Lx / 4, BOX_Ly / 2, 0.5])
        system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=sphere, velocity=[0, 0, 0]))

    if geo_cylinder:
        logging.info("Adding Cylinder.")
        cylinder = espressomd.shapes.Cylinder(center=[BOX_Lx / 2, BOX_Ly / 2, 0], axis=[0.0, 0.0, 1.0], radius=radius_object, direction=+1)
        system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=cylinder, velocity=[0, 0, 0]))
        system.constraints.add(particle_type=1, penetrable=False, shape=cylinder)

    # Polymers
    logging.info("Build Polymer(s).")
    build_polymers(system, run_type, N_MONOMERS, N_POLYMERS, START_POSITIONS, POSITIONS, fene, r_0, TETHERED)
    system.non_bonded_inter[0, 1].soft_sphere.set_params(a=1e-3, n=1.2, offset=0, cutoff=2 * phys["a"])

    particle_ids = [p.id for p in system.part.select(type=0)]
    forces = espressomd.observables.ParticleForces(ids=particle_ids)
    part_pos = espressomd.observables.ParticlePositions(ids=particle_ids)

    accumulator_forces = espressomd.accumulators.TimeSeries(obs=forces, delta_N=STEPS)
    accumulator_positions = espressomd.accumulators.TimeSeries(obs=part_pos, delta_N=STEPS)
    system.auto_update_accumulators.add(accumulator_forces)
    system.auto_update_accumulators.add(accumulator_positions)

    # Minimization and integrator
    # what value for gamma is suited here?
    # system.integrator.set_steepest_descent(f_max=0, gamma=GAMMA, max_displacement=0.001 * sigma)
    system.integrator.set_steepest_descent(f_max=0, gamma=0.001, max_displacement=0.0001 * sigma)

    if run_type=="validation_uncoiling":
        logging.info("Energy minimization.")

        if save_VTK:
            for i in range(minimization_steps):
                print(f"step {i + 1} of {minimization_steps}", end='\r', flush=True)
                system.part.writevtk(outdir + f"particles_prep{i:05d}.vtk")
                system.integrator.run(STEPS)

        if not save_VTK:
            for i in range(minimization_steps):
                print(f"step {i + 1} of {minimization_steps}", end='\r', flush=True)
                system.integrator.run(STEPS)


    if run_type=="validation_coiling" or run_type=="test_gridsizes":
        logging.info("Energy minimization.")

        if save_VTK:
            for i in range(minimization_steps):
                print(f"step {i + 1} of {minimization_steps}", end='\r', flush=True)
                system.part.writevtk(outdir + f"particles_prep{i:05d}.vtk")
                system.integrator.run(STEPS)

        if not save_VTK:
            for i in range(minimization_steps):
                print(f"step {i + 1} of {minimization_steps}", end='\r', flush=True)
                system.integrator.run(STEPS)

    system.integrator.set_vv()

    # LB Fluid setup
    # lbf = espressomd.lb.LBFluid(kT=0, seed=42, agrid=AGRID, dens=DENS, visc=VISC, tau=TIME_STEP)
    lbf = espressomd.lb.LBFluid(kT=0, seed=42, agrid=AGRID, dens=DENS, visc=VISC, tau=TIME_STEP*5)
    system.actors.add(lbf)
    system.thermostat.set_lb(LB_fluid=lbf, gamma=GAMMA, seed=42)

    wall_shape_z_up_lb = espressomd.shapes.Wall(normal=[0, 0, 1], dist=AGRID)
    wall_shape_z_down_lb = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(BOX_Lz - AGRID))

    if run_type=="validation_coiling":
        logging.info("LB setup for coiling validation (zero fluid velocity)")
        lb_boundary_up = system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=wall_shape_z_up_lb, velocity=[0, 0, 0]))
        lb_boundary_down = system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=wall_shape_z_down_lb, velocity=[0, 0, 0]))
    
    if run_type=="validation_uncoiling" or run_type=="test_gridsizes":
        logging.info(f"LB setup for uncoiling validation (nonzero fluid velocity)")
        lb_boundary_up = system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=wall_shape_z_up_lb, velocity=[0, 0, 0]))
        lb_boundary_down = system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=wall_shape_z_down_lb, velocity=[VELOCITY, 0, 0]))
    
    # Observables
    # particle_ids = [p.id for p in system.part.select(type=0)]
    # forces = espressomd.observables.ParticleForces(ids=particle_ids)
    # part_pos = espressomd.observables.ParticlePositions(ids=particle_ids)
    # accumulator_forces = espressomd.accumulators.TimeSeries(obs=forces, delta_N=STEPS)
    # accumulator_positions = espressomd.accumulators.TimeSeries(obs=part_pos, delta_N=STEPS)
    # system.auto_update_accumulators.add(accumulator_forces)
    # system.auto_update_accumulators.add(accumulator_positions)

    # for i in range(500):
    #     print(f"step {i + 1} of {500}", end='\r', flush=True)
    #     lbf.write_vtk_velocity(outdir + f"velocity_prep{i:05d}.vtk")
    #     lbf.write_vtk_boundary(outdir + f"boundary_prep{i:05d}.vtk")
    #     system.part.writevtk(outdir + f"particles_prep{i:05d}.vtk")
    #     system.integrator.run(STEPS)

    # system.lbboundaries.remove(lb_boundary_up)
    # system.lbboundaries.remove(lb_boundary_down)

    # wall_shape_z_up_lb_vel = espressomd.shapes.Wall(normal=[0, 0, 1], dist=AGRID)
    # wall_shape_z_down_lb_vel = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(BOX_Lz - AGRID))

    # system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=wall_shape_z_up_lb_vel, velocity=[0, 0, 0]))
    # system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=wall_shape_z_down_lb_vel, velocity=[0, 0, 0]))

    # Main simulation loop
    logging.info("Run simulation.")
    start_time = time.time()
    WINDOW_FRAMES = 10

    if save_VTK:
        for i in range(int(LOOPS)):
            print(f"step {i + 1} of {int(LOOPS)}", end='\r', flush=True)
            lbf.write_vtk_velocity(outdir + f"velocity_{i:05d}.vtk")
            lbf.write_vtk_boundary(outdir + f"boundary_{i:05d}.vtk")
            system.part.writevtk(outdir + f"particles_{i:05d}.vtk")
            system.integrator.run(STEPS)

    if not save_VTK:
        for i in range(int(LOOPS)):
            print(f"step {i + 1} of {int(LOOPS)}", end='\r', flush=True)
            system.integrator.run(STEPS)
            forces_all = accumulator_forces.time_series()
            F_avg = np.mean(forces_all[-WINDOW_FRAMES:], axis=0)



    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    
    return lbf, accumulator_forces, accumulator_positions