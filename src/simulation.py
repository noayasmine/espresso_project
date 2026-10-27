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
from collections import defaultdict

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def lb_units_and_gammas(agrid, tau, nu_md, xi_md=None, trt=False):
    # MD -> LB conversions
    nu_lb = nu_md * (tau / (agrid**2))
    xi_lb = (xi_md * (tau / (agrid**2))) if (xi_md is not None) else None

    # Gammas (MRT)
    gamma_shear = 1.0 - 2.0 / (6.0 * nu_lb + 1.0)
    if xi_lb is None:
        gamma_bulk = 0.0   # ESPResSo default when bulk viscosity not provided
        xi_lb_eff = 1.0/9.0
    else:
        gamma_bulk = 1.0 - 2.0 / (9.0 * xi_lb + 1.0)
        xi_lb_eff = xi_lb

    if trt:
        gamma_even = gamma_shear
        gamma_odd  = -(7.0 * gamma_even + 1.0) / (gamma_even + 7.0)
    else:
        gamma_even = 0.0
        gamma_odd  = 0.0

    # BGK-style taus (per-mode intuition)
    tau_bkg_shear = 1.0 / (1.0 - gamma_shear)
    tau_bkg_bulk  = 1.0 / (1.0 - gamma_bulk) if gamma_bulk != 1.0 else float("inf")

    return {
        "nu_LB": nu_lb,
        "xi_LB": xi_lb_eff,
        "gamma_shear": gamma_shear,
        "gamma_bulk": gamma_bulk,
        "gamma_even": gamma_even,
        "gamma_odd": gamma_odd,
        "tau_BGK_shear": tau_bkg_shear,
        "tau_BGK_bulk": tau_bkg_bulk,
    }

def lb_drag_forces(system, lb, gamma, ids):
    """
    Compute F_drag = -gamma * (v_i - u(x_i)) for selected particles.

    system : espressomd.System
    lb : LB fluid object (LBFluid)
    gamma : float (LB coupling constant)
    ids : list/array of particle ids
    """
    ids = np.asarray(ids, dtype=int)
    pos = espressomd.observables.ParticlePositions(ids=ids).calculate()   # Nx3 numpy
    vel = espressomd.observables.ParticleVelocities(ids=ids).calculate()  # Nx3 numpy

    # interpolated fluid velocities
    u = np.empty_like(pos)
    gi = lb.get_interpolated_velocity  # local ref for speed
    for n in range(ids.size):
        u[n] = gi(pos[n])

    # F_drag
    return -gamma * (vel - u)

def build_bond_list(existing_pairs):
    """
    Build a consistent ordered bond list (i < j) from a set of frozensets.
    """
    bond_list = []
    for pair in existing_pairs:
        i, j = tuple(pair)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        bond_list.append((i, j))

    bond_list = sorted(set(bond_list))
    return bond_list

def compute_tensile_forces_from_drag(forces_per_particle, N_POLYMERS, N_MONOMERS):
    """
    forces_per_particle: (N_POLYMERS*N_MONOMERS, 3) drag forces per monomer
    
    returns: (N_POLYMERS*N_MONOMERS, 3) tensile forces per monomer
    """
    # reshape to (polymers, monomers, xyz)
    F = forces_per_particle.reshape(int(N_POLYMERS), int(N_MONOMERS), 3)
    # reverse along monomer axis, cumsum, then flip back -> reverse-cumsum
    tensile = np.flip(np.cumsum(np.flip(F, axis=1), axis=1), axis=1)
    return tensile.reshape(-1, 3)

def compute_tensile_forces_general(total_forces,drag_forces,N_POLYMERS,N_MONOMERS,other_forces=None):
    """
    Approximate tensile forces per monomer as:

        F_tensile_i = F_total_i - F_drag_i - F_other_i

    Parameters
    ----------
    total_forces : (N_POLYMERS*N_MONOMERS, 3)
        Total forces per monomer (from ParticleForces).
    drag_forces  : (N_POLYMERS*N_MONOMERS, 3)
        Drag forces per monomer (lb_drag_forces).
    other_forces : same shape or None
        Any additional known external forces (e.g. gravity, walls).

    Returns
    -------
    tensile_forces_flat : (N_POLYMERS*N_MONOMERS, 3)
        Approximate tensile forces per monomer.
    """
    # reshape to (polymers, monomers, 3) if you like, though not strictly needed
    TF = total_forces.reshape(int(N_POLYMERS), int(N_MONOMERS), 3)
    DF = drag_forces.reshape(int(N_POLYMERS), int(N_MONOMERS), 3)

    if other_forces is not None:
        OF = other_forces.reshape(int(N_POLYMERS), int(N_MONOMERS), 3)
    else:
        OF = 0.0

    # F_tensile ≈ F_total - F_drag - F_other
    T = TF - DF - OF     # shape: (N_POLYMERS, N_MONOMERS, 3)

    return T.reshape(-1, 3)

def compute_fene_tensile_forces(positions, N_POLYMERS, N_MONOMERS, k, r_0, d_r_max):
    """
    Compute scalar and vector FENE tensile forces per monomer.

    Returns
    -------
    tensile_scalar_flat : (N_POLYMERS*N_MONOMERS,)
    tensile_vector_flat : (N_POLYMERS*N_MONOMERS, 3)
    """

    pos = positions.reshape(N_POLYMERS, N_MONOMERS, 3)

    r_vec = pos[:, 1:, :] - pos[:, :-1, :]
    r = np.linalg.norm(r_vec, axis=-1)
    x = r - r_0

    F_mag = np.zeros_like(r)
    valid = np.abs(x) < d_r_max
    F_mag[valid] = k * x[valid] / (1.0 - (x[valid] / d_r_max)**2)
    
    t_hat = np.zeros_like(r_vec)
    nonzero = r > 0
    t_hat[nonzero] = r_vec[nonzero] / r[nonzero, None]

    F_vec_bonds = F_mag[..., None] * t_hat

    tensile_scalar = np.zeros((N_POLYMERS, N_MONOMERS))
    tensile_vector = np.zeros((N_POLYMERS, N_MONOMERS, 3))

    tensile_scalar[:, :-1] = F_mag
    tensile_vector[:, :-1, :] = F_vec_bonds

    return tensile_scalar.reshape(-1), tensile_vector.reshape(-1, 3)

def add_tensile_forces_to_vtk_scalars(vtk_path, tensile_force_scalar):
    """
    Insert per-particle scalar tensile force magnitudes into an existing VTK legacy file.

    Parameters
    ----------
    vtk_path : str
        Path to the VTK file written by Espresso.
    tensile_force_scalar : array-like, shape (N,)
        Scalar tensile force per particle.
    """

    # Read file
    with open(vtk_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    n_particles = None

    while i < len(lines):
        line = lines[i]
        new_lines.append(line)

        # Detect POINT_DATA line
        if line.startswith("POINT_DATA"):
            n_particles = int(line.split()[1])

        # Detect end of previous dataset (velocity lookup table)
        if line.strip() == "LOOKUP_TABLE default":
            # Skip velocity entries
            for k in range(n_particles):
                i += 1
                new_lines.append(lines[i])

            # Insert scalar tensile forces
            new_lines.append("SCALARS tensile_force float\n")
            new_lines.append("LOOKUP_TABLE default\n")

            for p in range(n_particles):
                val = tensile_force_scalar[p]
                new_lines.append(f"{val}\n")

        i += 1

    with open(vtk_path, "w") as f:
        f.writelines(new_lines)

def add_tensile_forces_to_vtk(vtk_path, tensile_forces):
    """
    Insert tensile force vectors into an existing VTK legacy file produced by Espresso.

    Parameters
    ----------
    vtk_path : str
        Path to the VTK file written by Espresso.
    tensile_forces : array-like, shape (N, 3)
        Tensile force vectors, indexed by particle id.
    """

    # Read the file
    with open(vtk_path, "r") as f:
        lines = f.readlines()

    # Find where velocity data ends
    # After "LOOKUP_TABLE default", there are N lines (one per particle)
    # We want to insert right after those.
    new_lines = []
    i = 0
    n_particles = None

    while i < len(lines):
        line = lines[i]
        new_lines.append(line)

        # Detect POINT_DATA line so we know N
        if line.startswith("POINT_DATA"):
            n_particles = int(line.split()[1])

        # Detect the LOOKUP_TABLE for velocity
        if line.strip() == "LOOKUP_TABLE default":
            # Skip velocity entries
            for k in range(n_particles):
                i += 1
                new_lines.append(lines[i])

            # Now insert tensile forces
            new_lines.append("VECTORS tensile_force float\n")
            for p in range(n_particles):
                fx, fy, fz = tensile_forces[p]
                new_lines.append(f"{fx} {fy} {fz}\n")

        i += 1

    # Overwrite file
    with open(vtk_path, "w") as f:
        f.writelines(new_lines)

def add_bonds(system, exceeding_ids, fene, type_dictionary, distance, type_to_ids, bonds, sa_initiated):
    """
    Create FENE self-association bonds. Each particle above threshold initiates
    at most one bond, to the nearest particle that:
      - is within 'distance',
      - belongs to a different polymer,
      - is not already bonded to it in either direction.
    A particle can still receive bonds initiated by others.
    """

    exc_ids = sorted(int(i) for i in exceeding_ids)
    print(f"exceeding ids: {exc_ids}")
    added_bonds = []

    for pid in exc_ids:
        # each particle initiates at most one self-association bond
        if sa_initiated[pid]:
            continue

        p_handle = system.part.by_id(pid)
        particle_type = type_dictionary[pid]
        same_type_ids = type_to_ids[particle_type]

        # Neighbor search
        neighbor_ids = system.cell_system.get_neighbors(p_handle, distance)
        neighbor_ids_set = {int(nid) for nid in neighbor_ids}

        # Candidates: neighbors not in same polymer and not bonded in either direction
        candidates = [nid for nid in neighbor_ids_set - same_type_ids
                      if nid not in bonds[pid] and pid not in bonds[nid]]
        if not candidates:
            continue

        # bond to the nearest candidate
        pid_neighbor = min(candidates, key=lambda nid: system.distance(p_handle, system.part.by_id(nid)))
        p_handle_neighbor = system.part.by_id(pid_neighbor)
        p_handle.add_bond((fene, p_handle_neighbor))

        # update global state
        bonds[pid].add(pid_neighbor)
        sa_initiated[pid] = True

        added_bonds.append((pid, pid_neighbor))

    # print(f"Added bonds: {added_bonds}")
    return added_bonds

def remove_bonds(system, below_ids, fene, bonds, degree, type_dictionary):
    """
    Remove directed bonds pid -> pid_neighbor ONLY if pid and pid_neighbor are NOT in the same polymer (different type).

    below_ids: particle ids whose tensile force is below threshold (or below remove-threshold).
    bonds: directed adjacency (outgoing) dict: bonds[i] = set(j) meaning i -> j exists
    degree: outgoing bond count (degree[i] == len(bonds[i]))
    type_dictionary: maps particle id -> particle type (polymer id)
    """

    below_set = set(int(i) for i in below_ids)
    removed_bonds = []

    for pid in below_set:
        if pid not in bonds or len(bonds[pid]) == 0:
            continue

        p_handle = system.part.by_id(pid)
        pid_type = type_dictionary[pid]

        # copy because we'll mutate bonds[pid]
        outgoing = list(bonds[pid])

        for pid_neighbor in outgoing:
            # only remove self-association bonds: cross-polymer (different type)
            if type_dictionary[pid_neighbor] == pid_type:
                continue

            p_handle_neighbor = system.part.by_id(pid_neighbor)

            # remove in ESPResSo
            p_handle.delete_bond((fene, p_handle_neighbor))

            # update bookkeeping (directed!)
            bonds[pid].remove(pid_neighbor)
            degree[pid] -= 1
            removed_bonds.append((pid, pid_neighbor))
    
    print(f"Removed bonds: {removed_bonds}")

    return removed_bonds

def build_polymers(system, n_monomers, n_polymers, start_positions, positions, fene, bond_length, SEED, TETHERED=True, COILED=True, COMPUTE_POSITIONS=True):
    
    # compute positions here only if we are not using precomputed positions
    if COILED and COMPUTE_POSITIONS:
        logging.info("Building coiled up linear polymer.")
        positions = espressomd.polymer.linear_polymer_positions(n_polymers=n_polymers, 
            beads_per_chain=n_monomers, start_positions=start_positions, respect_constraints=True,
            bond_length=bond_length, seed=SEED)

    type_dictionary = {}
    for j in range(n_polymers):
        p_previous = None
        first_bead = None

        # type per polymer, each monomer has unique id
        for i, pos in enumerate(positions[j]):
            p = system.part.add(pos=pos, type=j)
            type_dictionary.update({p.id: j})

            if i == 0:
                first_bead = p
            if p_previous is not None:
                p.add_bond((fene, p_previous))
            p_previous = p

            if TETHERED==True:
                first_bead.fix = [True, True, True]
    
    type_to_ids = defaultdict(set)
    for pid, ptype in type_dictionary.items():
        type_to_ids[ptype].add(pid)

    return type_dictionary, type_to_ids

def run_simulation(params):
    # unpack from structured config
    sim = params["simulation"]
    phys = params["physical"]
    geom = params["geometry"]
    fluid = params["fluid"]
    out = params["output"]

    # geometry and runtime
    BOX_Lx, BOX_Ly, BOX_Lz = geom["BOX_Lx"], geom["BOX_Ly"], geom["BOX_Lz"]
    AGRID = geom["AGRID"]

    TIME_STEP, LB_TIME_STEP, LOOPS, STEPS = sim["TIME_STEP"], sim["LB_TIME_STEP"], sim["LOOPS"], sim["STEPS"]

    N_POLYMERS = sim["N_POLYMERS"]
    if N_POLYMERS > 0:
        POSITIONS = sim["POSITIONS"]
        START_POSITIONS = sim["START_POSITIONS"]

    TETHERED = sim["TETHERED"]
    COILED = sim["COILED"]

    # NEARWALL_FIX needs the patched ESPResSo build (wall-aware velocity readout): particle walls
    # then sit on the halfway bounce-back LB walls at AGRID instead of at the boundary node centre.
    NEARWALL_FIX = sim.get("NEARWALL_FIX", False)
    wall_z = AGRID if NEARWALL_FIX else 0.5 * AGRID
    logging.info(f"NEARWALL_FIX={NEARWALL_FIX}, particle walls at z={wall_z}, espressomd from {espressomd.__file__}")

    # set shear times 4/3 to account for boundary condition
    SHEAR = sim["SHEAR"] * 1e-4  # convert to sim units
    save_VTK = sim["SAVE_VTK"]
    minimization_steps = sim["MINIMIZATION_STEPS"]
    SEED = sim["SEED"]
    FORCE_THRESHOLD = sim["FORCE_THRESHOLD"]
    COMPUTE_POSITIONS = sim["COMPUTE_POSITIONS"]
    SELF_ASSOCIATE = sim["SELF_ASSOCIATE"]

    # physical parameters
    KT = phys["KT"]
    GAMMA = phys["GAMMA"]
    N_MONOMERS = phys["N_MONOMERS"]
    a = phys["a"]
    k, r_0, d_r_max = phys["k"], phys["r_0"], phys["d_r_max"]
    epsilon, sigma, cutoff, shift = phys["epsilon"], phys["sigma"], phys["cutoff"], phys["shift"]
    
    # fluid properties
    VISC, DENS = fluid["VISC"], fluid["DENS"]

    # output and boundary
    DIRECTORY = out["DIRECTORY"]
    geo_sphere = geom.get("SPHERE", False)
    geo_microcontraction = geom.get("MICROCONTRACTION", False)

    radius_object = geom["RADIUS_OBJECT"]
    xc = geom["SPHERE_CENTER_X"]
    yc = geom["SPHERE_CENTER_Y"]
    zc = geom["SPHERE_CENTER_Z"]

    VELOCITY = sim["VELOCITY"]
    outdir = DIRECTORY + f'vtk_data/'
    
    os.makedirs(DIRECTORY, exist_ok=True)
    start_time_init = time.time()
    if save_VTK:
        logging.info(f"Save vtk files to {outdir}")
        os.makedirs(outdir, exist_ok=True)

    MA = (np.sqrt(3) * VELOCITY * LB_TIME_STEP) / AGRID
    logging.info(f"Running simulation: shear={int(SHEAR*1e4)}, n_polymers={N_POLYMERS}, n_monomers={N_MONOMERS}, a={a}, "\
    f"gridsize={AGRID}, MD_timestep={TIME_STEP}, LB_timestep={LB_TIME_STEP}, viscosity={VISC}, seed={SEED}, velocity={VELOCITY}, Ma={MA}")
    logging.info(f"FENE parameters: k={k}, r_0={r_0}, dr_max={d_r_max}")

    vals = lb_units_and_gammas(agrid=AGRID, tau=LB_TIME_STEP, nu_md=VISC)
    for val, v in vals.items():
        logging.info(f"{val}: {v}")

    system = espressomd.System(box_l=[BOX_Lx, BOX_Ly, BOX_Lz], periodicity=[True, True, False])
    
    # print("Total ranks:", system.cell_system.get_state()["n_nodes"])
    logging.info(system.cell_system.get_state())

    system.cell_system.skin = 0.4
    system.time_step = TIME_STEP

    # Non-bonded interactions
    if N_POLYMERS > 0:
        all_types = np.arange(N_POLYMERS, dtype=int)
        logging.info(f"All particle types: {all_types}")

        # add LJ potential between all monomers, but only one way (no two-way bonds, the bonding is symmetric)
        # upper triangle,  covers (i,i) and each unordered (i,j)
        ii, jj = np.triu_indices(all_types.size, k=0)
        for i, j in zip(all_types[ii], all_types[jj]):
            system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=epsilon, sigma=sigma, shift=shift, cutoff=cutoff)

        fene = espressomd.interactions.FeneBond(k=k, r_0=r_0, d_r_max=d_r_max)
        system.bonded_inter.add(fene)

        logging.info(f"LJ parameters: epsilon={epsilon}, sigma={sigma}, shift={shift}, cutoff={cutoff}")

    # Wall constraints
    wall_shape_z_up = espressomd.shapes.Wall(normal=[0, 0, 1], dist=wall_z)
    wall_shape_z_down = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(BOX_Lz - wall_z))

    # particle type per polymer range from [0, N_POLYMER-1], giving LB shapes "particle type" N_POLYMERS
    system.constraints.add(particle_type=N_POLYMERS, penetrable=False, shape=wall_shape_z_up)
    system.constraints.add(particle_type=N_POLYMERS, penetrable=False, shape=wall_shape_z_down)

    if geo_sphere:
        # only adding the sphere as a constraint here, later for LB
        logging.info("Adding Sphere.")
        # add sphere as constraint 0.5 agrid from the wall
        # sphere_constraint = espressomd.shapes.Sphere(radius=(radius_object + AGRID), center=[25, 25, radius_object + 0.5*AGRID], direction=1)
        print(f"Adding sphere constraint at center={xc, yc, zc} with R={radius_object}")
        sphere_constraint = espressomd.shapes.Sphere(radius=radius_object, center=[xc, yc, zc], direction=1)
        # particle type per polymer range from 0 to N_POLYMER-1, giving LB shapes "particle type" N_POLYMERS
        system.constraints.add(particle_type=N_POLYMERS, penetrable=False, shape=sphere_constraint)

    if geo_microcontraction:
        # only add constraint for micro contraction here, later for LB too
        logging.info("Adding Microcontraction.")
        # add sphere as constraint 0.5 agrid from the wall
        microcontraction_constraint_left = espressomd.shapes.Rhomboid(corner=np.array([2.5, 0, 0]), a=np.array([2, 0, 0]),
            b=np.array([0, 8, 0]), c=np.array([2, 0, 4.]), direction=1)
        microcontraction_constraint_right = espressomd.shapes.Rhomboid(corner=np.array([8.5, 0, 0]), a=np.array([2, 0, 0]),
            b=np.array([0, 8, 0]), c=np.array([-2, 0, 4.]), direction=1)
        # particle type per polymer range from 0 to N_POLYMER-1, giving LB shapes "particle type" N_POLYMERS
        system.constraints.add(particle_type=N_POLYMERS, penetrable=False, shape=microcontraction_constraint_left)
        system.constraints.add(particle_type=N_POLYMERS, penetrable=False, shape=microcontraction_constraint_right)

    # Polymers
    if N_POLYMERS > 0:
        logging.info("Build Polymer(s).")
        print(START_POSITIONS.shape)
        type_dictionary, type_to_ids = build_polymers(system, N_MONOMERS, N_POLYMERS, START_POSITIONS, POSITIONS, fene, r_0, SEED, \
            TETHERED, COILED, COMPUTE_POSITIONS)
        
        # Add soft sphere potential between walls and particles
        for particle_type in all_types:
            system.non_bonded_inter[particle_type, N_POLYMERS].soft_sphere.set_params(a=1e-3, n=1.2, offset=0, cutoff=phys["a"])
        
        particle_ids = np.arange(N_MONOMERS * N_POLYMERS, dtype=int)
        # particle_ids = np.array([p.id for p in system.part.all()],dtype=int)
        logging.info(f"Particle ids: {particle_ids}")
        forces = espressomd.observables.ParticleForces(ids=particle_ids)
        part_pos = espressomd.observables.ParticlePositions(ids=particle_ids)
        print()

        accumulator_forces = espressomd.accumulators.TimeSeries(obs=forces, delta_N=STEPS)
        accumulator_positions = espressomd.accumulators.TimeSeries(obs=part_pos, delta_N=STEPS)
        system.auto_update_accumulators.add(accumulator_forces)
        system.auto_update_accumulators.add(accumulator_positions)

    if COMPUTE_POSITIONS and N_POLYMERS > 0:
        # energy minimization with steepest descent integrator
        # to coil up polymers without overlap
        start_time_energy_minimization = time.time()
        system.integrator.set_steepest_descent(f_max=0, gamma=0.001, max_displacement=0.0001 * sigma)

        logging.info(f"Energy minimization: {minimization_steps} steps")

        if save_VTK:
            for i in range(minimization_steps):
                print(f"step {i + 1} of {minimization_steps}", end='\r', flush=True)
                system.part.writevtk(outdir + f"particles_energy_minimization_{i:05d}.vtk")
                system.integrator.run(STEPS)

        if not save_VTK:
            for i in range(minimization_steps):
                print(f"step {i + 1} of {minimization_steps}", end='\r', flush=True)
                system.integrator.run(STEPS)

        end_time_energy_minimization = time.time()
        print(f"Runtime energy minimization step: {end_time_energy_minimization - start_time_energy_minimization:.2f} seconds")
        np.savez_compressed(os.path.join(DIRECTORY, "init_positions.npz"), positions=accumulator_positions.time_series())
        print(f"Positions shape: {accumulator_positions.time_series().shape}")

    system.integrator.set_vv()
    logging.info(f"Initialize LB fluid.")
    
    # adding all LB stuff after the minimization of the energy (positions)
    # because this uses a different integrator
    lbf = espressomd.lb.LBFluid(kT=0, seed=42, agrid=AGRID, dens=DENS, visc=VISC, tau=LB_TIME_STEP)
    system.actors.add(lbf)
    system.thermostat.set_lb(LB_fluid=lbf, gamma=GAMMA, seed=42)

    wall_shape_z_up_lb = espressomd.shapes.Wall(normal=[0, 0, 1], dist=AGRID)
    wall_shape_z_down_lb = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(BOX_Lz - AGRID))

    logging.info(f"Set LB setup with velocity = {VELOCITY}")
    lb_boundary_up = system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=wall_shape_z_up_lb, velocity=[0, 0, 0]))
    lb_boundary_down = system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=wall_shape_z_down_lb, velocity=[VELOCITY, 0, 0]))
    
    if geo_sphere:
        # effecitvely placing the sphere at 
        # sphere = espressomd.shapes.Sphere(radius=radius_object + AGRID*0.5, center=[25, 25, radius_object + 0.5 * AGRID])
        # with NEARWALL_FIX the LB sphere coincides with the particle sphere (no half-cell shift)
        sphere = espressomd.shapes.Sphere(radius=radius_object, center=[xc, yc, zc if NEARWALL_FIX else zc + 0.5 * AGRID])
        system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=sphere, velocity=[0, 0, 0]))
    
    if geo_microcontraction:
        # placing LB smicrocontraction
        logging.info("Adding Microcontraction.")
        # add sphere as constraint 0.5 agrid from the wall
        microcontraction_left = espressomd.shapes.Rhomboid(corner=np.array([2.5, 0, 0]), a=np.array([2.25, 0, 0]),
            b=np.array([-0.5, 8.5, 0]), c=np.array([2.5, 0, 4.5]), direction=1)
        microcontraction_right = espressomd.shapes.Rhomboid(corner=np.array([8.5, 0, 0]), a=np.array([2.25, 0, 0]),
            b=np.array([-0.5, 8.5, 0]), c=np.array([-2.5, 0, 4.5]), direction=1)
        # particle type per polymer range from 0 to N_POLYMER-1, giving LB shapes "particle type" N_POLYMERS
        system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=microcontraction_left, velocity=[0, 0, 0]))
        system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=microcontraction_right, velocity=[0, 0, 0]))

    if N_POLYMERS > 0:
        # drag forces per monomer
        drag_forces_traj = np.empty((int(LOOPS), int(N_POLYMERS * N_MONOMERS), 3), dtype=np.float32)
        tensile_forces_traj = np.empty((int(LOOPS), int(N_POLYMERS * N_MONOMERS)), dtype=np.float32)
        tensile_forces_vec_traj = np.empty((int(LOOPS), int(N_POLYMERS * N_MONOMERS), 3), dtype=np.float32)
        added_bonds_per_step_traj = np.zeros(int(LOOPS), dtype=np.int32)

        WINDOW = 20

        # Directed bond dictionary
        bonds = defaultdict(set)            # i -> {j, k, ...}
        sa_initiated = defaultdict(bool)    # True once i has initiated its self-association bond

        for p in system.part:
            pid = int(p.id)
            for _, partner in p.bonds:
                qid = int(getattr(partner, "id", partner))

                # add both directions (initially symmetric)
                bonds[pid].add(qid)
                bonds[qid].add(pid)
    
    print(system.actors.active_actors)
    end_time_init = time.time()
    total_init_time = end_time_init - start_time_init
    logging.info(f"Total initialization time is {total_init_time}")
    # Main simulation loop
    logging.info("Run simulation.")

    start_time = time.time()

    if save_VTK and N_POLYMERS == 0:
        # Run to get velocity field only 
        print(f"Run with N_pol=0")
        for i in range(int(LOOPS)):
            if i==1:
                lbf.write_vtk_boundary(outdir + f"boundary_{i:05d}.vtk")

            print(f"step {i + 1} of {int(LOOPS)}", end='\r', flush=True)
            #lbf.write_vtk_velocity(outdir + f"velocity_{i:05d}.vtk")
            system.integrator.run(STEPS)

    if save_VTK and N_POLYMERS > 0:
        for i in range(int(LOOPS)):
            
            if i==1:
                np.savez_compressed(os.path.join(DIRECTORY, "init_positions.npz"), positions=accumulator_positions.time_series())
                print(f"Positions shape: {accumulator_positions.time_series().shape}")

                lbf.write_vtk_boundary(outdir + f"boundary_{i:05d}.vtk")

            print(f"step {i + 1} of {int(LOOPS)}", end='\r', flush=True)
            #lbf.write_vtk_velocity(outdir + f"velocity_{i:05d}.vtk")
            system.part.writevtk(outdir + f"particles_{i:05d}.vtk")

            system.integrator.run(STEPS)

            drag_forces_per_particle = lb_drag_forces(system, lbf, GAMMA, particle_ids)
            drag_forces_traj[i] = drag_forces_per_particle

            ###### compute tensile forces from the fene over extension ######
            positions = espressomd.observables.ParticlePositions(ids=particle_ids).calculate()
            net_tensile_forces, vec_tensile_forces = compute_fene_tensile_forces(
            positions, N_POLYMERS, N_MONOMERS, k=k, r_0=r_0, d_r_max=d_r_max)

            # store per-timestep
            tensile_forces_traj[i]     = net_tensile_forces          # (N, )
            tensile_forces_vec_traj[i] = vec_tensile_forces          # (N, 3)

            start = max(0, i - WINDOW + 1)

            # vector moving average
            tensile_vec_window = tensile_forces_vec_traj[start:i+1]  # (W, N, 3)
            tensile_vec_avg = tensile_vec_window.mean(axis=0)        # (N, 3)

            # magnitude AFTER vector averaging
            tensile_net_from_vec_avg = np.linalg.norm(tensile_vec_avg, axis=-1)  # (N,)

            # the moving average of the magnitudes
            tensile_window = tensile_forces_traj[start:i+1]          # (W, N)
            tensile_net_avg = tensile_window.mean(axis=0)            # (N,)

            vtk_file = outdir + f"particles_{i:05d}.vtk"
            # adding the net tensile forces MOVING AVG to the vtk files
            add_tensile_forces_to_vtk_scalars(vtk_file, tensile_net_avg)

            if SELF_ASSOCIATE and i >= WINDOW:
                exceed_mask = tensile_net_avg > FORCE_THRESHOLD
                exceeding_ids = particle_ids[exceed_mask]
                R_CAPTURE = 2 * a

                added_pairs = add_bonds(system, exceeding_ids, fene=fene, type_dictionary=type_dictionary, distance=R_CAPTURE,
                    type_to_ids=type_to_ids, bonds=bonds, sa_initiated=sa_initiated)
                
                added_bonds_per_step_traj[i] = len(added_pairs)
                # print(added_bonds_per_step_traj)
                # below_mask = tensile_avg < FORCE_THRESHOLD
                # below_ids = particle_ids[below_mask]

                #removed_pairs = remove_bonds(system, below_ids, fene=fene, bonds=bonds, degree=degree, type_dictionary=type_dictionary)

    if not save_VTK and N_POLYMERS > 0:
        for i in range(int(LOOPS)):
            if i==1:
                np.savez_compressed(os.path.join(DIRECTORY, "init_positions.npz"), positions=accumulator_positions.time_series())
                print(f"Positions shape: {accumulator_positions.time_series().shape}")

            print(f"step {i + 1} of {int(LOOPS)}", end='\r', flush=True)

            system.integrator.run(STEPS)

            drag_forces_per_particle = lb_drag_forces(system, lbf, GAMMA, particle_ids)
            drag_forces_traj[i] = drag_forces_per_particle

            ###### compute tensile forces from the fene over extension ######
            positions = espressomd.observables.ParticlePositions(ids=particle_ids).calculate()
            net_tensile_forces, vec_tensile_forces = compute_fene_tensile_forces(
            positions, N_POLYMERS, N_MONOMERS, k=k, r_0=r_0, d_r_max=d_r_max)

            # store per-timestep
            tensile_forces_traj[i]     = net_tensile_forces          # (N, )
            tensile_forces_vec_traj[i] = vec_tensile_forces          # (N, 3)

            start = max(0, i - WINDOW + 1)

            # vector moving average
            tensile_vec_window = tensile_forces_vec_traj[start:i+1]  # (W, N, 3)
            tensile_vec_avg = tensile_vec_window.mean(axis=0)        # (N, 3)

            # magnitude AFTER vector averaging
            tensile_net_from_vec_avg = np.linalg.norm(tensile_vec_avg, axis=-1)  # (N,)

            # the moving average of the magnitudes
            tensile_window = tensile_forces_traj[start:i+1]          # (W, N)
            tensile_net_avg = tensile_window.mean(axis=0)            # (N,)

            if SELF_ASSOCIATE and i >= WINDOW:
                exceed_mask = tensile_net_avg > FORCE_THRESHOLD
                exceeding_ids = particle_ids[exceed_mask]
                R_CAPTURE = 2 * a

                added_pairs = add_bonds(system, exceeding_ids, fene=fene, type_dictionary=type_dictionary, distance=R_CAPTURE,
                    type_to_ids=type_to_ids, bonds=bonds, sa_initiated=sa_initiated)
                
                added_bonds_per_step_traj[i] = len(added_pairs)
                # print(added_bonds_per_step_traj)
                # below_mask = tensile_avg < FORCE_THRESHOLD
                # below_ids = particle_ids[below_mask]

                #removed_pairs = remove_bonds(system, below_ids, fene=fene, bonds=bonds, degree=degree, type_dictionary=type_dictionary)


    end_time = time.time()
    print(f"Runtime simulation: {end_time - start_time:.2f} seconds")
    
    if N_POLYMERS > 0:
        return lbf, accumulator_forces, accumulator_positions, drag_forces_traj, tensile_forces_traj, tensile_forces_vec_traj, added_bonds_per_step_traj

    elif N_POLYMERS == 0:
        return lbf
