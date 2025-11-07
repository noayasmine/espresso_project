import json
import sys
import numpy as np
import time
from simulation import run_simulation
from utils import save_results

def load_config(path):
    with open(path) as f:
        return json.load(f)
    
def add_dependent_parameters(config, LJ_strength=1):
    config["physical"]["r_0"] = 2 * config["physical"]["a"]
    config["physical"]["sigma"] = (2 ** (-1/6)) * config["physical"]["r_0"]
    config["physical"]["epsilon"] = LJ_strength * config["physical"]["KT"]
    config["physical"]["cutoff"] = 6 * config["physical"]["a"]
    # Compute velocity (in MD units, i.e. 1e2 m/s )
    config["simulation"]["VELOCITY"] = config["simulation"]["SHEAR"] * 1e-4 * (config["geometry"]["BOX_Lz"] - config["geometry"]["AGRID"])

def compute_start_positions(config):
    n_polymers = config["simulation"]["N_POLYMERS"]
    a = config["physical"]["a"]
    agrid = config["geometry"]["AGRID"]
    box_lx = config["geometry"]["BOX_Lx"]
    box_ly = config["geometry"]["BOX_Ly"]
    radius = config["geometry"]["RADIUS_OBJECT"] - agrid / 2
    
    center = np.array([box_lx / 4, box_ly / 2])
    spacing = 6*a 

    if config["geometry"]["SPHERE"] or config["geometry"]["CYLINDER"]:
        n_attempts = 10 * n_polymers  # oversample to filter later

        lower_bounds = [agrid, 2 * agrid]
        upper_bounds = [box_lx - 7, box_ly - 2 * agrid]
        points = np.random.uniform(lower_bounds, upper_bounds, (n_attempts, 2))

        mask = np.sum((points - center) ** 2, axis=1) >= radius ** 2
        filtered_points = points[mask][:n_polymers]

        random_x, random_y = filtered_points.T

    elif n_polymers == 1:
        random_x = np.array([box_lx / 5])
        random_y = np.array([box_ly / 2])
    
    elif n_polymers == 2:
        center_x = box_lx / 4
        center_y = box_ly / 2
        random_x = np.array([center_x, center_x])
        random_y = np.array([center_y - spacing / 2, center_y + spacing / 2])
        
    else:
        random_x = np.random.uniform(agrid, box_lx - agrid, n_polymers)
        random_y = np.random.uniform(agrid, box_ly - agrid, n_polymers)

    # place monomers just above the wall
    # place monomers a (above the soft sphere potential)
    z = np.full(n_polymers, 0.5 * agrid + a + 1e-07)

    start_positions = np.stack([random_x, random_y, z], axis=1)
    config["simulation"]["START_POSITIONS"] = start_positions
    return start_positions

def add_polymer_positions(config, run_type):
    start_positions = compute_start_positions(config)

    N_POLYMERS = config["simulation"]["N_POLYMERS"]
    N_MONOMERS = config["physical"]["N_MONOMERS"]
    a = config["physical"]["a"]
    AGRID = config["geometry"]["AGRID"]
    bond_length = 2 * config["physical"]["a"]

    distribution_type = config["simulation"].get("DISTRIBUTION", "UNIFORM")
    min_monomers = config["simulation"].get("MIN_MONOMERS", 50)
    std_dev = config["simulation"].get("STD_MONOMERS", 10)

    std_dev_monomer_direction = config["simulation"]["stdev_initial_pos"]

    if distribution_type == "UNIFORM":
        monomer_counts = np.full(N_POLYMERS, N_MONOMERS, dtype=int)

    elif distribution_type == "NORMAL":
        raw_counts = np.random.normal(loc=N_MONOMERS, scale=std_dev, size=N_POLYMERS)
        monomer_counts = np.clip(np.round(raw_counts), min_monomers, None).astype(int)

    elif distribution_type == "SKEWED":
        # Skewed right (longer tail to the right), using exponential distribution
        scale = std_dev  # bigger means more skew
        skewed = np.random.exponential(scale=scale, size=N_POLYMERS)
        monomer_counts = np.round(skewed + min_monomers).astype(int)

    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    positions = []

    for j in range(N_POLYMERS):
        n_monomers = monomer_counts[j]
        start_pos = start_positions[j]
        direction = np.array([1.0, 0.0, 0.0])  # initial direction
        direction = direction / np.linalg.norm(direction)

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
    config["simulation"]["MONOMER_COUNTS"] = monomer_counts.tolist()

def add_timeloops_from_gridsize(config):
    steps = config["simulation"]["STEPS"]
    gridsize = config["geometry"]["AGRID"]
    total_time = config["simulation"]["TOTAL_TIME"] # seconds
    time_unit = 1e-4 # seconds
    total_sim_time = total_time / time_unit
    time_step = config["simulation"]["TIME_STEP"] 
    
    config["simulation"]["TIME_STEP"] = time_step
    LOOPS = total_sim_time / (time_step * steps)
    config["simulation"]["LOOPS"] = LOOPS

    print(f"gridsize={gridsize}, loops={LOOPS}, steps={steps}")
    
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
        add_polymer_positions(config, run_type)

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
            # set rud_id as seed
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
        
        # config["simulation"]["SAVE_VTK"] = True
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
        add_polymer_positions(config, run_type)

    if run_type == "validation_uncoiling_free":
        run_type = "validation_uncoiling"
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
        # large box to fit N_monomer=250 polymer
        config["geometry"]["BOX_Lx"] = 64
        config["geometry"]["BOX_Ly"] = 32
        config["geometry"]["BOX_Lz"] = 32
        config["simulation"]["N_POLYMERS"] = 1
        config["geometry"]["AGRID"] = 1
        # need to decrease timestep due to large boxsize 
        # (i.e. large Ma number for LB for high shear rates)
        config["simulation"]["TIME_STEP"] = 0.0005
        config["simulation"]["LB_TIME_STEP"] = 0.0005
        
        if shear == 10000:
            config["simulation"]["TIME_STEP"] = 0.00025
            config["simulation"]["LB_TIME_STEP"] = 0.00025

        config["simulation"]["TETHERED"] = False
        config["simulation"]["TOTAL_TIME"] = 0.3
        # extra minimization steps needed for the large polymer
        config["simulation"]["MINIMIZATION_STEPS"] = 400
        # might need to increase fene bond for large these large free flowing polymers?
        # config["physical"]["k"] = 10
        config["physical"]["k"] = 100

        # increased steps in simulation to save the data (to run for vtk)
        # since otherwise the file will be too big
        config["simulation"]["SAVE_VTK"] = False
        config["simulation"]["STEPS"] = 10000
        
        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength)
        config["output"]["DIRECTORY"] = f"./{save_dir}/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/shear_{int(shear):05d}/run_{run}"
        add_polymer_positions(config, run_type)

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
        add_polymer_positions(config, run_type)
    
    if run_type == "experiment":
        shear = float(sys.argv[2])
        save_dir = str(sys.argv[3])

        config["output"]["DIRECTORY"] = f"./{save_dir}/"
    
        config["simulation"]["SAVE_VTK"] = True
        config["simulation"]["SHEAR"] = shear
        config["simulation"]["TOTAL_TIME"] = 0.1
        initialize="coiled"
        LJ_strength=18

        config["geometry"]["BOX_Lx"] = 20
        config["geometry"]["BOX_Ly"] = 8
        config["geometry"]["BOX_Lz"] = 8
        config["geometry"]["SPHERE"] = True

        if config["physical"]["N_MONOMERS"] == 100:
            config["simulation"]["MINIMIZATION_STEPS"] = 600
            config["simulation"]["TIME_STEP"] = 0.001
            config["simulation"]["LB_TIME_STEP"] = 0.001

        if config["physical"]["N_MONOMERS"] == 50:
            config["simulation"]["MINIMIZATION_STEPS"] = 200

        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength)
        add_polymer_positions(config, run_type)

    lbf, forces, positions, drag_forces = run_simulation(config,run_type)

    save_results(config, lbf, forces, drag_forces, positions)