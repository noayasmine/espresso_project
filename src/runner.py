import json
import sys
import numpy as np
import time
from simulation import run_simulation
from utils import save_results
import os

def load_config(path):
    with open(path) as f:
        return json.load(f)
    
def add_dependent_parameters(config, LJ_strength=18):
    config["physical"]["r_0"] = 2 * config["physical"]["a"]
    config["physical"]["sigma"] = (2 ** (-1/6)) * config["physical"]["r_0"]
    config["physical"]["epsilon"] = LJ_strength * config["physical"]["KT"]
    config["physical"]["cutoff"] = 6 * config["physical"]["a"]
    # Compute velocity (in MD units, i.e. 1e2 m/s )
    config["simulation"]["VELOCITY"] = config["simulation"]["SHEAR"] * 1e-4 * (config["geometry"]["BOX_Lz"] - config["geometry"]["AGRID"])

def sample_points_min_dist_2d(n_points, bounds, min_dist, max_attempts=100000):
    """
    Sample n_points in 2D within given bounds, with pairwise distance >= min_dist.
    bounds = ((x_min, x_max), (y_min, y_max))
    """
    (x_min, x_max), (y_min, y_max) = bounds
    pts = []
    attempts = 0

    while len(pts) < n_points and attempts < max_attempts:
        attempts += 1
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        p = np.array([x, y])

        # accept if far enough from all existing points
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)

    if len(pts) < n_points:
        raise RuntimeError("Could not place all points with the given min_dist.")

    pts = np.array(pts)
    return pts[:, 0], pts[:, 1]

def compute_start_positions(config):
    """Computes positions of first monomers.

    Returns
    -------
    start_positions : (N_POLYMERS, 3)
    + adds start_positions to json file (config)
    """
    np.random.seed(config["simulation"]["SEED"])
    N_POLYMERS = config["simulation"]["N_POLYMERS"]

    a = config["physical"]["a"]
    AGRID = config["geometry"]["AGRID"]
    BOX_Lx = config["geometry"]["BOX_Lx"]
    BOX_Ly = config["geometry"]["BOX_Ly"]
    BOX_Lz = config["geometry"]["BOX_Lz"]
    min_dist = 15.0 * a
    offset = 0.5 * AGRID + a + 1e-07
    # offset = 3 * AGRID + a + 1e-07

    if config["geometry"]["SPHERE"]:
        # ---- minimum distance in xy ----
        
        x_bounds = (20, 35)
        y_bounds = (15,35)

        random_x, random_y = sample_points_min_dist_2d(N_POLYMERS, (x_bounds, y_bounds), min_dist)

        x = random_x.copy()
        y = random_y.copy()

        # ---- sphere geometry ----
        xc = config["geometry"].get("SPHERE_CENTER_X", 25.0)
        yc = config["geometry"].get("SPHERE_CENTER_Y", 25.0)
        zc = config["geometry"].get("SPHERE_CENTER_Z",config["geometry"].get("RADIUS_OBJECT", 7.5))
        zc += 0.5 * AGRID

        R = float(config["geometry"]["RADIUS_OBJECT"])

        dx = x - xc
        dy = y - yc
        r2_xy = dx*dx + dy*dy

        inside = r2_xy <= R*R

        # default: flat wall
        z = np.full(N_POLYMERS, offset)

        # lift points inside sphere footprint
        if np.any(inside):
            print("point inside sphere")
            under = R*R - r2_xy[inside]
            under = np.maximum(under, 0.0)

            z_surf = zc + np.sqrt(under)
            # z[inside] = z_surf
            nx = dx[inside] / R
            ny = dy[inside] / R
            nz = (z_surf - zc) / R

            # increased offset due to non perfect sphere
            offset_sphere = AGRID + a + 1e-07
            x[inside] = x[inside] + nx * offset_sphere
            y[inside] = y[inside] + ny * offset_sphere
            z[inside] = z_surf     + nz * offset_sphere

        start_positions = np.stack([x, y, z], axis=1)

    elif N_POLYMERS == 1:
        random_x = np.array([BOX_Lx / 5])
        random_y = np.array([BOX_Ly / 2])
        z = np.full(N_POLYMERS, offset)
        start_positions = np.stack([random_x, random_y, z], axis=1)

    elif N_POLYMERS == 2:
        spacing = 6 * a
        center_x = BOX_Lx / 4
        center_y = BOX_Ly / 2
        random_x = np.array([center_x, center_x])
        random_y = np.array([center_y - spacing / 2, center_y + spacing / 2])
        z = np.full(N_POLYMERS, offset)
        start_positions = np.stack([random_x, random_y, z], axis=1)

    else:
        x_bounds = (AGRID + 2.0 * a, BOX_Lx - (AGRID + 2.0 * a))
        y_bounds = (AGRID + 2.0 * a, BOX_Ly - (AGRID + 2.0 * a))

        random_x, random_y = sample_points_min_dist_2d(N_POLYMERS, (x_bounds, y_bounds), min_dist)

        z = np.full(N_POLYMERS, offset)
        start_positions = np.stack([random_x, random_y, z], axis=1)

    config["simulation"]["START_POSITIONS"] = start_positions
    print("Positions first monomers:")
    print(start_positions)

    return start_positions

def rotation_matrix_from_vectors(a, b):
    """
    Compute rotation matrix that rotates vector a onto vector b.
    a, b must be normalized.
    """
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    vx = np.array([[    0, -v[2],  v[1]],
                   [ v[2],     0, -v[0]],
                   [-v[1],  v[0],    0]])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return R

def load_polymer_positions(start_positions, config):
    """
    Places pre-defined polymer conformations at start_positions.
    Polymers whose start position lies on the sphere are rotated
    so that they align with the local sphere normal.
    """
    
    N_MONOMERS = config["physical"]["N_MONOMERS"]
    seed = config["simulation"]["SEED"]
    k = int(config["physical"]["k"])
    AGRID = config["geometry"]["AGRID"]

    outdir = f"./configs/initial_polymer_positions_k_{k}"
    # outdir = f"./configs/initial_polymer_positions_k_1000"
    # path_to_file = os.path.join(outdir, f"init_positions_{N_MONOMERS}_seed_{seed}.npz")
    # print(f"Read out position file: {path_to_file}")
    # data = np.load(path_to_file)
    # init_positions = data["positions"]      # (1, N_MONOMERS, 3)

    # # normalize polymer to origin
    # origin = init_positions[0, 0]
    # shifted = init_positions - origin       # (1, N_MONOMERS, 3)

    N_POLYMERS = start_positions.shape[0]
    positions = np.zeros((N_POLYMERS, N_MONOMERS, 3))

    # sphere geometry
    sphere = config["geometry"]["SPHERE"]
    xc = config["geometry"].get("SPHERE_CENTER_X", 25.0)
    yc = config["geometry"].get("SPHERE_CENTER_Y", 25.0)
    zc = config["geometry"].get("SPHERE_CENTER_Z",config["geometry"].get("RADIUS_OBJECT", 0.0))
    zc += 0.5 * config["geometry"]["AGRID"]
    R = float(config["geometry"]["RADIUS_OBJECT"])
    print(xc,yc,zc)

    ez = np.array([0.0, 0.0, 1.0])  # original polymer direction
   
    for i in range(N_POLYMERS):
        # random_seed = np.random.randint(0,10)
        random_seed = 1

        path_to_file = os.path.join(outdir, f"init_positions_{N_MONOMERS}_seed_{random_seed}.npz")
        print(f"Read out position file: {path_to_file}")
        data = np.load(path_to_file)
        init_positions = data["positions"]      # (1, N_MONOMERS, 3)

        # normalize polymer to origin
        origin = init_positions[0, 0]
        shifted = init_positions - origin       # (1, N_MONOMERS, 3)

        base = shifted[0]           # (N_MONOMERS, 3)
        pos0 = start_positions[i]   # (3,)

        polymer = base.copy()

        if sphere:
            # this part of code doesnt work properly, it is not rotating the polymers properly
            dx = pos0[0] - xc
            dy = pos0[1] - yc

            # check if start position lies on sphere footprint (from top)
            if dx*dx + dy*dy <= R*R + AGRID:
                print("Rotation for sphere")
                # compute normal at surface point
                r_vec = pos0 - np.array([xc, yc, zc])
                n = r_vec / np.linalg.norm(r_vec)
                print(n)
                # rotation that maps +z to normal
                Rmat = rotation_matrix_from_vectors(ez, n)

                polymer = polymer @ Rmat.T

        # translate polymer to its start position
        positions[i] = polymer + pos0

    config["simulation"]["POSITIONS"] = positions
    print("Added positions with shape:")
    print(positions.shape)

def add_polymer_positions(config):
    start_positions = compute_start_positions(config)

    N_POLYMERS = config["simulation"]["N_POLYMERS"]
    N_MONOMERS = config["physical"]["N_MONOMERS"]
    a = config["physical"]["a"]
    AGRID = config["geometry"]["AGRID"]
    COILED = config["simulation"]["COILED"]
    bond_length = 2 * config["physical"]["a"]
    COMPUTE_POSITIONS = config["simulation"]["COMPUTE_POSITIONS"]

    distribution_type = config["simulation"].get("DISTRIBUTION", "UNIFORM")
    std_dev_monomer_direction = config["simulation"]["stdev_initial_pos"]
    
    if not COILED:
        monomer_counts = np.full(N_POLYMERS, N_MONOMERS, dtype=int)
        print("Adding custom positions (uncoiled starting state).")
        positions = []

        for j in range(N_POLYMERS):
            n_monomers = monomer_counts[j]
            start_pos = start_positions[j]

            direction = np.array([1.0, 0.0, 0.0])  # initial direction

            polymer_positions = []
            current_pos = start_pos.copy()

            for i in range(n_monomers):
                polymer_positions.append(current_pos.copy())
                if i < n_monomers - 1:
                    # Add small noise to direction (not to position!)
                    noise = np.random.normal(loc=0.0, scale=std_dev_monomer_direction, size=3)
                    noisy_direction = direction + noise
                    noisy_direction /= np.linalg.norm(noisy_direction)  # keep unit length
                    current_pos = current_pos + bond_length * noisy_direction

            polymer_positions = np.array(polymer_positions)
            # ensure constraint of wall in z-direction (above soft sphere 2 * a + 1e-07)
            # polymer_positions[:, 2] = np.maximum(polymer_positions[:, 2], 0.5 * AGRID + 0.001)
            # (above soft sphere 2 * a + 1e-07)
            polymer_positions[:, 2] = np.maximum(polymer_positions[:, 2], 0.5 * AGRID + a + 1e-07)
            positions.append(polymer_positions)
        
        config["simulation"]["POSITIONS"] = positions

    if COILED and not COMPUTE_POSITIONS:
        load_polymer_positions(start_positions, config)

def add_timeloops_from_gridsize(config):
    """ Calculate number of loops for specified number of 
    steps and total time (in seconds)

    Adds loops [int] to json file (config) 
    """
    steps = config["simulation"]["STEPS"]
    total_time = config["simulation"]["TOTAL_TIME"] # seconds

    time_unit = 1e-4 # seconds
    total_sim_time = total_time / time_unit

    time_step = config["simulation"]["TIME_STEP"] 

    LOOPS = total_sim_time / (time_step * steps)
    config["simulation"]["LOOPS"] = int(LOOPS)
    
if __name__ == "__main__":
    path = "configs/default.json"
    config = load_config(path)
    run_type = str(sys.argv[1])

    if run_type == "test_gridsizes":
        if len(sys.argv) == 7:
            shear = float(sys.argv[2])
            gridsize = float(sys.argv[3])
            run = int(sys.argv[4])
            k_vwf = float(sys.argv[5])
            save_dir = str(sys.argv[6])
            config["simulation"]["SEED"] = run
            np.random.seed(config["simulation"]["SEED"])
            config["output"]["DIRECTORY"] = f"./{save_dir}/grid_{int(gridsize*1e2):05d}/shear_{int(shear):05d}/run_{run}/"

        if len(sys.argv) == 6:
            shear = float(sys.argv[2])
            gridsize = float(sys.argv[3])
            k_vwf = float(sys.argv[4])
            save_dir = str(sys.argv[5])
            config["output"]["DIRECTORY"] = f"./{save_dir}/grid_{int(gridsize*1e2):05d}/shear_{int(shear):05d}/"

        config["simulation"]["COILED"] = False
        config["simulation"]["SHEAR"] = shear
        config["geometry"]["AGRID"] = gridsize
        config["simulation"]["N_POLYMERS"] = 2
        config["simulation"]["stdev_initial_pos"] = 0.01
        config["simulation"]["TOTAL_TIME"] = 0.1
        config["physical"]["k"] = k_vwf
        config["simulation"]["MINIMIZATION_STEPS"] = 0
        
        if gridsize==0.5:
            config["simulation"]["LB_TIME_STEP"] = 0.001

        # config["simulation"]["SAVE_VTK"] = True

        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength=14)
        add_polymer_positions(config)

    if run_type == "validation_uncoiling":
        if len(sys.argv) == 8:
            shear = float(sys.argv[2])
            gamma = float(sys.argv[3])
            LJ_strength = float(sys.argv[4])
            N_monomers = int(sys.argv[5])
            config["physical"]["N_MONOMERS"] = N_monomers
            run = int(sys.argv[6])
            save_dir = str(sys.argv[7])
            config["simulation"]["TOTAL_TIME"] = 0.2

            # set run_id as seed
            config["simulation"]["SEED"] = run
            config["output"]["DIRECTORY"] = f"./{save_dir}/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/shear_{int(shear):05d}/run_{run}/"

        if len(sys.argv) == 6:
            shear = float(sys.argv[2])
            gamma = float(sys.argv[3])
            LJ_strength = float(sys.argv[4])
            save_dir = str(sys.argv[5])
            config["physical"]["N_MONOMERS"] = 50
            #config["simulation"]["SAVE_VTK"] = True
            config["simulation"]["SEED"] = 15
            config["simulation"]["TOTAL_TIME"] = 0.4

            config["output"]["DIRECTORY"] = f"./{save_dir}/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/shear_{int(shear):05d}/"
        
        # set parameters to match the validation data from Fu et al.
        config["simulation"]["SHEAR"] = shear
        config["physical"]["GAMMA"] = gamma
        config["simulation"]["N_POLYMERS"] = 1
        config["geometry"]["AGRID"] = 1
        # change number of monomers in the polymer
        
        config["physical"]["a"] = 0.03
        # config["physical"]["k"] = 10
        # can set total time to 0.2 for multiple runs / validation uncoiling
        # config["simulation"]["TOTAL_TIME"] = 0.2
        # config["simulation"]["SAVE_VTK"] = True

        if config["physical"]["N_MONOMERS"] == 100:
            config["simulation"]["MINIMIZATION_STEPS"] = 600
            config["simulation"]["TIME_STEP"] = 0.001
            config["simulation"]["LB_TIME_STEP"] = 0.001

        if config["physical"]["N_MONOMERS"] == 50:
            config["simulation"]["MINIMIZATION_STEPS"] = 200

        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength)
        add_polymer_positions(config)

    if run_type == "validation_uncoiling_free":
        shear = float(sys.argv[2])
        gamma = float(sys.argv[3])
        LJ_strength = float(sys.argv[4])
        run = int(sys.argv[5])
        save_dir = str(sys.argv[6])
        
        # set run id as random seed
        config["simulation"]["SEED"] = run
        # set parameters to match the validation data from Schneider et al.
        config["simulation"]["SHEAR"] = shear
        config["physical"]["GAMMA"] = gamma
        # 15 micrometer polymer
        config["physical"]["N_MONOMERS"] = 250
        
        # large box to fit N_monomer=250 polymer (15 micrometer)
        config["geometry"]["BOX_Lx"] = 64
        config["geometry"]["BOX_Ly"] = 32
        config["geometry"]["BOX_Lz"] = 32
        config["simulation"]["N_POLYMERS"] = 1

        config["geometry"]["AGRID"] = 1
        # need to decrease timestep due to large boxsize 
        # (large Ma number for LB for high velocity)
        # also because larger polymer needs more timesteps to be stable
        config["simulation"]["TIME_STEP"] = 0.0005
        config["simulation"]["LB_TIME_STEP"] = 0.0005

        # to test
        config["simulation"]["COMPUTE_POSITIONS"] = False
        if run==1:
            config["simulation"]["SAVE_VTK"] = True
        
        # if 5000 > shear > 10:
        #     config["simulation"]["TIME_STEP"] = 0.00025
        #     config["simulation"]["LB_TIME_STEP"] = 0.00025
        
        # if shear >= 5000:
        #     config["simulation"]["TIME_STEP"] = 0.00025/2
        #     config["simulation"]["LB_TIME_STEP"] = 0.00025/2

        if shear > 5000:
            config["simulation"]["TIME_STEP"] = 0.00025
            config["simulation"]["LB_TIME_STEP"] = 0.00025

        config["simulation"]["TETHERED"] = False
        config["simulation"]["TOTAL_TIME"] = 0.3
        # extra minimization steps needed for the large polymer
        config["simulation"]["MINIMIZATION_STEPS"] = 800
        # might need to increase fene bond for large these large free flowing polymers?
        # config["physical"]["k"] = 10
        config["physical"]["k"] = 1000

        # increased steps in simulation to save the data (to run for vtk)
        # since otherwise the file will be too big
        config["simulation"]["SAVE_VTK"] = False
        config["simulation"]["STEPS"] = 10000
        
        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength)
        config["output"]["DIRECTORY"] = f"./{save_dir}/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/shear_{int(shear):05d}/run_{run}"
        add_polymer_positions(config)

    if run_type == "validation_coiling":
        if len(sys.argv) == 6:
            gamma = float(sys.argv[2])
            LJ_strength = float(sys.argv[3])
            run = int(sys.argv[4])
            save_dir = str(sys.argv[5])
            # set run id as random seed
            config["simulation"]["SEED"] = run
            config["output"]["DIRECTORY"] = f"./{save_dir}/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/run_{run}/"

        if len(sys.argv) == 7:
            gamma = float(sys.argv[2])
            LJ_strength = float(sys.argv[3])
            stdev_init = float(sys.argv[4])
            N_mono = float(sys.argv[5])
            save_dir = str(sys.argv[6])
            config["simulation"]["stdev_initial_pos"] = stdev_init
            config["output"]["DIRECTORY"] = f"./{save_dir}/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/"

        config["simulation"]["COILED"] = False
        # for recoiling, the experimental data added 60% sucrose, increased viscosity and density
        config["fluid"]["VISC"] = config["fluid"]["VISC"] * 58
        config["fluid"]["DENS"] = 0.00013
        config["physical"]["GAMMA"] = gamma
        config["geometry"]["AGRID"] = 1
        config["simulation"]["N_POLYMERS"] = 1
        config["simulation"]["TOTAL_TIME"] = 0.8
        config["physical"]["k"] = 10
        config["physical"]["N_MONOMERS"] = 100
        config["simulation"]["MINIMIZATION_STEPS"] = 0
        
        # config["simulation"]["SAVE_VTK"] = True

        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength)
        add_polymer_positions(config)
    
    if run_type == "experiment":
        shear = float(sys.argv[2])
        # config["simulation"]["SELF_ASSOCIATE"] = sys.argv[3].lower() == "true"
        # N_monomers = int(sys.argv[3])
        seed = int(sys.argv[3])
        save_dir = str(sys.argv[4])

        # if config["simulation"]["SELF_ASSOCIATE"]:
        #     config["output"]["DIRECTORY"] = f"./{save_dir}/selfassociation/"

        # else:
        #     config["output"]["DIRECTORY"] = f"./{save_dir}/no_selfassociation/"
        config["output"]["DIRECTORY"] = f"./{save_dir}/N_monomers_{50}_seed_{seed}/"

        config["simulation"]["SAVE_VTK"] = True
        config["simulation"]["SHEAR"] = shear
        config["simulation"]["TOTAL_TIME"] = 0.01
        config["FORCE_THRESHOLD"] =  0.01

        config["simulation"]["COMPUTE_POSITIONS"] = True

        config["simulation"]["N_POLYMERS"] = 1
        config["simulation"]["MINIMIZATION_STEPS"] = 600
        config["physical"]["k"] = 1000
        config["physical"]["N_MONOMERS"] = 50
        config["simulation"]["SEED"] = seed

        config["simulation"]["LB_TIME_STEP"] = 0.001
        config["simulation"]["TIME_STEP"] = 0.001
        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config)
        add_polymer_positions(config)
    
    if run_type == "experiment_density":
        shear = float(sys.argv[2])
        N_POLYMERS = int(sys.argv[3])
        save_dir = str(sys.argv[4])

        config["output"]["DIRECTORY"] = f"./{save_dir}/N_polymers_{N_POLYMERS}/"
        
        config["simulation"]["SELF_ASSOCIATE"] =  True
        config["simulation"]["SAVE_VTK"] = True
        config["simulation"]["SHEAR"] = shear
        config["simulation"]["TOTAL_TIME"] = 0.3
        config["FORCE_THRESHOLD"] =  0.01

        config["simulation"]["COMPUTE_POSITIONS"] = False

        config["simulation"]["N_POLYMERS"] = N_POLYMERS

        config["physical"]["k"] = 1000
        config["physical"]["N_MONOMERS"] = 50
        config["simulation"]["SEED"] = 1

        config["simulation"]["LB_TIME_STEP"] = 0.001
        config["simulation"]["TIME_STEP"] = 0.001
        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config)
        add_polymer_positions(config)
    
    if run_type == "experiment_sphere":
        shear = float(sys.argv[2])
        config["simulation"]["SELF_ASSOCIATE"] = sys.argv[3].lower() == "true"
        save_dir = str(sys.argv[4])

        print("Run simulation with sphere")

        if config["simulation"]["SELF_ASSOCIATE"]:
            config["output"]["DIRECTORY"] = f"./{save_dir}/selfassociation/"

        else:
            config["output"]["DIRECTORY"] = f"./{save_dir}/no_selfassociation/"
    
        config["simulation"]["SAVE_VTK"] = True
        config["simulation"]["SHEAR"] = shear
        config["simulation"]["TOTAL_TIME"] = 0.02
        config["simulation"]["COMPUTE_POSITIONS"] = True
        config["simulation"]["SEED"] = 1

        config["simulation"]["N_POLYMERS"] = 3
        config["geometry"]["BOX_Lx"] = 100
        config["geometry"]["BOX_Ly"] = 50
        config["geometry"]["BOX_Lz"] = 30
        config["geometry"]["AGRID"] = 1
        config["geometry"]["SPHERE"] = True
        config["physical"]["k"] = 1000
        config["physical"]["N_MONOMERS"] = 50

        config["simulation"]["LB_TIME_STEP"] = 0.001
        config["simulation"]["TIME_STEP"] = 0.001
        
        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config)
        add_polymer_positions(config)

    if run_type == "experiment_microcontraction":
        shear = float(sys.argv[2])
        save_dir = str(sys.argv[3])

        config["output"]["DIRECTORY"] = f"./{save_dir}/"
    
        config["simulation"]["SAVE_VTK"] = True
        config["simulation"]["SHEAR"] = shear
        config["simulation"]["TOTAL_TIME"] = 0.1

        # config["simulation"]["N_POLYMERS"] = 50
        config["simulation"]["N_POLYMERS"] = 1
        config["geometry"]["BOX_Lx"] = 300
        config["geometry"]["BOX_Ly"] = 130
        config["geometry"]["BOX_Lz"] = 100
        config["geometry"]["MICROCONTRACTION"] = True
        config["physical"]["k"] = 100
        # config["physical"]["N_MONOMERS"] = 120
        config["physical"]["N_MONOMERS"] = 50
        # N_POLYMERS_FREE <= N_POLYMERS
        # config["simulation"]["N_POLYMERS_FREE"] = 20
        config["simulation"]["TIME_STEP"] = 0.0005
        config["simulation"]["LB_TIME_STEP"] = 0.0005

        if config["physical"]["N_MONOMERS"] > 150:
            config["simulation"]["MINIMIZATION_STEPS"] = 600
            # config["simulation"]["TIME_STEP"] = 0.001
            # config["simulation"]["LB_TIME_STEP"] = 0.001

        if 150 >= config["physical"]["N_MONOMERS"] > 50:
            config["simulation"]["MINIMIZATION_STEPS"] = 400
            # config["simulation"]["TIME_STEP"] = 0.001
            # config["simulation"]["LB_TIME_STEP"] = 0.001

        if config["physical"]["N_MONOMERS"] <= 50:
            config["simulation"]["MINIMIZATION_STEPS"] = 200

        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config)
        add_polymer_positions(config)

    lbf, forces, positions, drag_forces, tensile_forces, added_bonds_per_step = run_simulation(config)

    save_results(config, lbf, forces, drag_forces, tensile_forces, added_bonds_per_step, positions)