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

def compute_tensile_forces_general_old(total_forces, drag_forces, N_POLYMERS, N_MONOMERS, other_forces=None):
    """
    total_forces : (N_POLYMERS*N_MONOMERS, 3) total forces per monomer (from ParticleForces)
    drag_forces  : (N_POLYMERS*N_MONOMERS, 3) drag forces per monomer (your lb_drag_forces)
    other_forces : same shape or None (e.g. gravity, wall, etc.)

    Returns:
      bond_tensions_per_monomer : (N_POLYMERS*N_MONOMERS, 3)

    Convention:
      For each polymer, we compute bond tensions T[0..N_MONOMERS-2].
      We store them at monomer index i (bond to i+1).
      The last monomer's tension is set to 0.
    """
    # reshape to (polymers, monomers, 3)
    TF = total_forces.reshape(int(N_POLYMERS), int(N_MONOMERS), 3)
    DF = drag_forces.reshape(int(N_POLYMERS), int(N_MONOMERS), 3)

    if other_forces is not None:
        OF = other_forces.reshape(int(N_POLYMERS), int(N_MONOMERS), 3)
    else:
        OF = 0.0

    # residuals R_i = F_tot_i - F_drag_i - F_other_i
    R = TF - DF - OF   # shape (N_POLYMERS, N_MONOMERS, 3)

    T = np.zeros((int(N_POLYMERS), int(N_MONOMERS), 3))

    # reconstruct tension for each polymer separately
    for p in range(int(N_POLYMERS)):
        # bond N-2 (between monomers N-2 and N-1)
        T[p, N_MONOMERS-2] = -R[p, N_MONOMERS-1]

        # walk backward along the chain: i = N-2, ..., 1
        for i in range(N_MONOMERS-2, 0, -1):
            T[p, i-1] = T[p, i] + R[p, i]

        # last monomer has no bond downstream, keep T[p, N_MONOMERS-1] = 0

    return T.reshape(-1, 3)


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

def compute_fene_tensile_forces(positions,N_POLYMERS,N_MONOMERS,k,r_0,d_r_max):
    """
    Compute vector FENE tensile forces and return them per monomer.

    Returns
    -------
    tensile_forces_flat : (N_POLYMERS*N_MONOMERS, 3)
        Vector tensile forces per monomer.
        Monomer i gets the force from the bond (i -> i+1).
        Last monomer in each polymer gets [0,0,0].
    """
    # reshape positions
    pos = positions.reshape(N_POLYMERS, N_MONOMERS, 3)

    # bond vectors r_vec[p,i] = pos[p,i+1] - pos[p,i]
    r_vec = pos[:, 1:, :] - pos[:, :-1, :]        # (P, M-1, 3)

    # bond lengths
    r = np.linalg.norm(r_vec, axis=-1)            # (P, M-1)
    x = r - r_0                                   # overextension

    # scalar FENE tension (zero if compressed or invalid)
    F_mag = np.zeros_like(r)
    mask = (x > 0) & (x < d_r_max)
    xv = x[mask]
    F_mag[mask] = k * xv / (1.0 - (xv / d_r_max)**2)

    # unit vectors
    r_expanded = r[:, :, None]                    # (P, M-1, 1)
    t_hat = np.zeros_like(r_vec)
    nonzero = r > 0
    t_hat[nonzero] = r_vec[nonzero] / r_expanded[nonzero]

    # vector tensile forces on monomer i from bond (i→i+1)
    F_vec_bonds = F_mag[..., None] * t_hat        # (P, M-1, 3)

    # build per-monomer forces (last monomer has zero force)
    tensile = np.zeros((N_POLYMERS, N_MONOMERS, 3))
    tensile[:, :-1, :] = F_vec_bonds              # assign bond-force to monomer i

    # flatten
    tensile_flat = tensile.reshape(-1, 3)

    return tensile_flat


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

def add_bonds_old(system, exceeding_ids, fene, type_dictionary, distance):
    """
    Create FENE bonds between particles that:
      - are within 'distance' (via get_neighbors),
      - and have different particle ids (not same polymer).

    Parameters
    exceeding_ids : array-like of int
        Particle IDs whose net tensile force exceeded the threshold.
    fene : object
        Preconfigured FENE interaction (e.g., espressomd.interactions.FeneBond(...)).
    distance : float
        Neighbor cutoff radius.

    Returns
    added_pairs : list[tuple[int, int]]
    """

    # set of ids that exceed the tension threshold
    exc_ids_set  = set(exceeding_ids)
    # print(f"Ids that exceed tension threshold:{exc_ids_set}")

    for pid in exc_ids_set:
        # get particle handle from id
        p_handle = system.part.by_id(pid)

        # print(p_handle)
        neighbor_ids = system.cell_system.get_neighbors(p_handle, distance)
        neighbor_ids_set = set(neighbor_ids)
        # print(pid, p_handle.bonds)

        # check to which particles this pid is already bonded
        bonded_ids = set([int(getattr(partner, "id", partner)) for _, partner in p_handle.bonds])
        # print(f"Already bonded ids set: {bonded_ids}")
        # if particle already has two bonds (i.e. neighboring monomer + self-association bond)
        # do NOT add more bonds
        if len(bonded_ids) >= 2: 
            print(f"Particle {pid} is already bonded")
            continue
        # print(f"Already bonded ids set: {bonded_ids}")
        # check for all particles with the same type
        particle_type = type_dictionary[pid]
        # print(f"particle type {particle_type}")

        particles_same_type = set(system.part.select(type=particle_type).id)

        # particles_same_type = set(system.part.select(type=particle_type).id)
        # print(f"Same type ids: {particles_same_type}")

        new_bond_ids = neighbor_ids_set - bonded_ids - particles_same_type
        # print(f"New bonds for: {new_bond_ids}")
        
        for pid_neighbor in new_bond_ids:
            # print(f"Adding bond between {pid} and {pid_neighbor}")
            p_handle_neighbor = system.part.by_id(pid_neighbor)
            # Check if this pid is not already bonded to this particle (other way)
            if pid not in set([int(getattr(partner, "id", partner)) for _, partner in p_handle_neighbor.bonds]):
                p_handle.add_bond((fene, p_handle_neighbor))
                print(f"Add bond {pid} and {pid_neighbor}")
    
    return 

def add_bonds(system, exceeding_ids, fene, type_dictionary, distance, type_to_ids, existing_pairs, degree):
    """
    Create FENE bonds between particles that:
      - are within 'distance' (via get_neighbors),
      - belong to different polymers (different particle type),
      - are not already bonded in either direction,
      - and respect a max-2-bonds-per-particle rule.

    Parameters
    ----------
    system : espressomd.System
    exceeding_ids : array-like of int
        Particle IDs whose net tensile force exceeded the threshold.
    fene : object
        Preconfigured FENE interaction.
    type_dictionary : dict[int, int]
        Maps particle id -> particle type (polymer id).
    distance : float
        Neighbor cutoff radius.
    type_to_ids : dict[int, set[int]]
        Precomputed mapping: type -> set of particle ids.
    existing_pairs : set[frozenset[int, int]]
        Global set of existing bonds (unordered pairs). Updated in-place.
    degree : dict[int, int]
        Bond count per particle. Updated in-place.

    Returns
    -------
    added_pairs : list[tuple[int, int]]
        List of newly created bonds (pid, pid_neighbor).
    """

    exc_ids_set = set(int(i) for i in exceeding_ids)
    print(exc_ids_set)
    added_pairs = []

    for pid in exc_ids_set:
        print(f"{pid} exceeded threshold")
        # Respect max-2 rule for this particle
        if degree[pid] >= 3:
            print(f"Particle {pid} already has three bonds.")
            continue

        p_handle = system.part.by_id(pid)
        particle_type = type_dictionary[pid]
        same_type_ids = type_to_ids[particle_type]

        # Neighbor search
        neighbor_ids = system.cell_system.get_neighbors(p_handle, distance)
        neighbor_ids_set = {int(nid) for nid in neighbor_ids}

        # Candidates: neighbors not in same polymer
        candidates = neighbor_ids_set - same_type_ids

        for pid_neighbor in candidates:
            # Respect max-2 rule on neighbor as well
            if degree[pid_neighbor] >= 3:
                continue

            pair = frozenset((pid, pid_neighbor))

            # if pair already in existing_pairs, there is already a bond
            # in either direction between these two. skip.
            if pair in existing_pairs:
                print(f"This pair {pair} is already bonded")
                continue

            # add bond
            print(f"Add bond between {pair}")
            p_handle_neighbor = system.part.by_id(pid_neighbor)
            p_handle.add_bond((fene, p_handle_neighbor))

            # update global state
            existing_pairs.add(pair)
            degree[pid] += 1
            degree[pid_neighbor] += 1
            added_pairs.append((pid, pid_neighbor))

            # if pid now has 2 bonds, don't add more from it
            if degree[pid] >= 2:
                break

    return added_pairs

def build_polymers(system, n_monomers, n_polymers, n_polymers_free, start_positions, positions, fene, bond_length, SEED, TETHERED=True, COILED=True, COMPUTE_POSITIONS=True):
    
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
    N_POLYMERS_FREE = sim["N_POLYMERS_FREE"]
    POSITIONS = sim["POSITIONS"]
    START_POSITIONS = sim["START_POSITIONS"]
    TETHERED = sim["TETHERED"]
    COILED = sim["COILED"]

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
    geo_cylinder = geom.get("CYLINDER", False)
    geo_microcontraction = geom.get("MICROCONTRACTION", False)
    radius_object = geom.get("RADIUS_OBJECT", 7.5)

    VELOCITY = sim["VELOCITY"]
    outdir = DIRECTORY + f'vtk_data/'
    
    os.makedirs(DIRECTORY, exist_ok=True)

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
    wall_shape_z_up = espressomd.shapes.Wall(normal=[0, 0, 1], dist=AGRID * 0.5)
    wall_shape_z_down = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(BOX_Lz - AGRID * 0.5))
    # particle type per polymer range from 0 to N_POLYMER-1, giving LB shapes "particle type" N_POLYMERS
    system.constraints.add(particle_type=N_POLYMERS, penetrable=False, shape=wall_shape_z_up)
    system.constraints.add(particle_type=N_POLYMERS, penetrable=False, shape=wall_shape_z_down)

    if geo_sphere:
        logging.info("Adding Sphere.")
        # add sphere as constraint 0.5 agrid from the wall
        sphere_constraint = espressomd.shapes.Sphere(radius=radius_object, center=[25, 25, radius_object + 0.5 * AGRID], direction=1)
        # particle type per polymer range from 0 to N_POLYMER-1, giving LB shapes "particle type" N_POLYMERS
        system.constraints.add(particle_type=N_POLYMERS, penetrable=False, shape=sphere_constraint)
        
        sphere = espressomd.shapes.Sphere(radius=radius_object - 0.5 * AGRID, center=[25, 25, radius_object + 0.5 * AGRID])
        system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=sphere, velocity=[0, 0, 0]))

    if geo_microcontraction:
        logging.info("Adding Microcontraction.")
        # add sphere as constraint 0.5 agrid from the wall
        microcontraction_constraint = espressomd.shapes.Rhomboid(corner=np.array([AGRID * 0.5, AGRID * 0.5, AGRID * 0.5]), a=np.array([20.0, 0.0, 0.0]),
            b=np.array([0.0, 130.0, 0.0]), c=np.array([40.0, 0.0, 80.0]), direction=1)
        # particle type per polymer range from 0 to N_POLYMER-1, giving LB shapes "particle type" N_POLYMERS
        system.constraints.add(particle_type=N_POLYMERS, penetrable=False, shape=microcontraction_constraint)
        
        microcontraction = espressomd.shapes.Rhomboid(corner=np.array([AGRID * 0.5, AGRID * 0.5, AGRID * 0.5]), a=np.array([20.0, 0.0, 0.0]),
            b=np.array([0.0, 130.0, 0.0]), c=np.array([40.0, 0.0, 80.0]), direction=1)
        system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=microcontraction, velocity=[0, 0, 0]))

    # Polymers
    logging.info("Build Polymer(s).")
    type_dictionary, type_to_ids = build_polymers(system, N_MONOMERS, N_POLYMERS, N_POLYMERS_FREE, START_POSITIONS, POSITIONS, fene, r_0, SEED, \
        TETHERED, COILED, COMPUTE_POSITIONS)
    # Add soft sphere potential between wall and particles
    for particle_type in all_types:
        system.non_bonded_inter[particle_type, N_POLYMERS].soft_sphere.set_params(a=1e-3, n=1.2, offset=0, cutoff=phys["a"])
    
    particle_ids = np.arange(N_MONOMERS * N_POLYMERS, dtype=int)
    # particle_ids = np.array([p.id for p in system.part.all()],dtype=int)
    logging.info(f"Particle ids: {particle_ids}")
    forces = espressomd.observables.ParticleForces(ids=particle_ids)
    part_pos = espressomd.observables.ParticlePositions(ids=particle_ids)

    accumulator_forces = espressomd.accumulators.TimeSeries(obs=forces, delta_N=STEPS)
    accumulator_positions = espressomd.accumulators.TimeSeries(obs=part_pos, delta_N=STEPS)
    system.auto_update_accumulators.add(accumulator_forces)
    system.auto_update_accumulators.add(accumulator_positions)

    if COMPUTE_POSITIONS:
        # Minimization and integrator
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

        np.savez_compressed(os.path.join(outdir, "init_positions.npz"), positions=accumulator_positions.time_series())
        print(f"Positions shape: {accumulator_positions.time_series().shape}")

    system.integrator.set_vv()
    logging.info(f"Initialize LB fluid.")

    lbf = espressomd.lb.LBFluid(kT=0, seed=42, agrid=AGRID, dens=DENS, visc=VISC, tau=LB_TIME_STEP)
    system.actors.add(lbf)
    system.thermostat.set_lb(LB_fluid=lbf, gamma=GAMMA, seed=42)

    wall_shape_z_up_lb = espressomd.shapes.Wall(normal=[0, 0, 1], dist=AGRID)
    wall_shape_z_down_lb = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(BOX_Lz - AGRID))

    logging.info(f"Set LB setup with velocity = {VELOCITY}")
    lb_boundary_up = system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=wall_shape_z_up_lb, velocity=[0, 0, 0]))
    lb_boundary_down = system.lbboundaries.add(espressomd.lbboundaries.LBBoundary(shape=wall_shape_z_down_lb, velocity=[VELOCITY, 0, 0]))

    # Main simulation loop
    logging.info("Run simulation.")
    start_time = time.time()

    # drag forces per monomer
    drag_forces_traj = np.empty((int(LOOPS), int(N_POLYMERS * N_MONOMERS), 3), dtype=np.float32)
    tensile_forces_traj = np.empty((int(LOOPS), int(N_POLYMERS * N_MONOMERS), 3), dtype=np.float32)
    total_forces_traj = np.empty((int(LOOPS), int(N_POLYMERS * N_MONOMERS), 3), dtype=np.float32)
    WINDOW = 100

    # Build global view of existing bonds (unordered pairs) + degrees
    existing_pairs = set()          # frozenset({i, j})
    degree = defaultdict(int)       # how many bonds each particle already has

    for p in system.part:
        pid = int(p.id)
        for _, partner in p.bonds:
            qid = int(getattr(partner, "id", partner))
            pair = frozenset((pid, qid))
            if pair not in existing_pairs:
                existing_pairs.add(pair)
                degree[pid] += 1
                degree[qid] += 1   # we treat bond as undirected, even if stored once
    
    print("The degree dict:")
    print(degree)
    print("The existing pairs set:")
    print(existing_pairs)

    if save_VTK:
        for i in range(int(LOOPS)):
            if i==1:
                np.savez_compressed(os.path.join(DIRECTORY, "init_positions.npz"), positions=accumulator_positions.time_series())
                print(f"Positions shape: {accumulator_positions.time_series().shape}")

                lbf.write_vtk_boundary(outdir + f"boundary_{i:05d}.vtk")

            print(f"step {i + 1} of {int(LOOPS)}", end='\r', flush=True)
            lbf.write_vtk_velocity(outdir + f"velocity_{i:05d}.vtk")
            # lbf.write_vtk_boundary(outdir + f"boundary_{i:05d}.vtk")
            system.part.writevtk(outdir + f"particles_{i:05d}.vtk")

            system.integrator.run(STEPS)

            drag_forces_per_particle = lb_drag_forces(system, lbf, GAMMA, particle_ids)
            drag_forces_traj[i] = drag_forces_per_particle
            start = max(0, i - WINDOW + 1)              # inclusive

            ###### computing tensile from total - cumulative drag force ######
            drag_window = drag_forces_traj[start:i+1]    # shape: (window_len, N_particles, 3)
            drag_avg = drag_window.mean(axis=0)          # shape: (N_particles, 3)
            total_forces_per_particle = espressomd.observables.ParticleForces(ids=particle_ids).calculate()
            total_forces_traj[i] = total_forces_per_particle
            total_window = total_forces_traj[start:i+1]        # (window_len, N_particles, 3)
            total_avg   = total_window.mean(axis=0) 
            tensile_forces = compute_tensile_forces_general(total_avg, drag_avg, N_POLYMERS, N_MONOMERS)
            tensile_forces_traj[i] = tensile_forces
            tensile_window = tensile_forces_traj[start:i+1]
            tensile_moving_avg = tensile_window.mean(axis=0)
            ######

            ###### compute tensile forces from the fene over extension ######
            # positions = espressomd.observables.ParticlePositions(ids=particle_ids).calculate()
            # tensile_forces = compute_fene_tensile_forces(positions,N_POLYMERS,N_MONOMERS,k=k,r_0=r_0,d_r_max=d_r_max)
            # total_forces_traj[i] = tensile_forces
            # start = max(0, i - WINDOW + 1)    
            # tensile_window = tensile_forces_traj[start:i+1]
            # tensile_moving_avg = tensile_window.mean(axis=0)
            #######

            vtk_file = outdir + f"particles_{i:05d}.vtk"
            add_tensile_forces_to_vtk(vtk_file, tensile_moving_avg)

            net_tensile = np.linalg.norm(tensile_moving_avg, axis=1)

            if SELF_ASSOCIATE and i >= WINDOW:
                exceed_mask = net_tensile > FORCE_THRESHOLD
                exceeding_ids = particle_ids[exceed_mask]
                R_CAPTURE = 2 * a

                added = add_bonds(system, exceeding_ids, fene=fene, type_dictionary=type_dictionary, distance=R_CAPTURE,
                    type_to_ids=type_to_ids, existing_pairs=existing_pairs, degree=degree)

    if not save_VTK:
        for i in range(int(LOOPS)):
            if i==1:
                np.savez_compressed(os.path.join(DIRECTORY, "init_positions.npz"), positions=accumulator_positions.time_series())
                print(f"Positions shape: {accumulator_positions.time_series().shape}")
              
            print(f"step {i + 1} of {int(LOOPS)}", end='\r', flush=True)
            system.integrator.run(STEPS)

            drag_forces_per_particle = lb_drag_forces(system, lbf, GAMMA, particle_ids)
            drag_forces_traj[i] = drag_forces_per_particle

            tensile_forces = compute_tensile_forces(drag_forces_per_particle, N_POLYMERS, N_MONOMERS)
            tensile_forces_traj[i] = tensile_forces
            net_tensile = np.abs(np.sum(tensile_forces, axis=1))

            if SELF_ASSOCIATE:
                # boolean mask of monomers exceeding threshold tension 
                exceed_mask = net_tensile > FORCE_THRESHOLD
                exceeding_ids = particle_ids[exceed_mask]      
                R_CAPTURE = 2 * a                 
                added = add_bonds(system, exceeding_ids, fene=fene, type_dictionary=type_dictionary, distance=R_CAPTURE)

            # add removal of bonds? 

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    
    return lbf, accumulator_forces, accumulator_positions, drag_forces_traj