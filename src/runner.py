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
    config["physical"]["epsilon"] = LJ_strength * 4 * config["physical"]["KT"]
    config["physical"]["cutoff"] = 8 * config["physical"]["a"]
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
    spacing = 0.3

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

    # Place monomers just above the wall
    z = np.full(n_polymers, 0.5 * agrid + a/2)

    start_positions = np.stack([random_x, random_y, z], axis=1)
    config["simulation"]["START_POSITIONS"] = start_positions
    return start_positions

def add_polymer_positions(config, run_type):
    start_positions = compute_start_positions(config)

    N_POLYMERS = config["simulation"]["N_POLYMERS"]
    N_MONOMERS = config["physical"]["N_MONOMERS"]
    AGRID = config["geometry"]["AGRID"]
    bond_length = 2 * config["physical"]["a"]
    distribution_type = config["simulation"].get("DISTRIBUTION", "UNIFORM")
    min_monomers = config["simulation"].get("MIN_MONOMERS", 25)
    std_dev = config["simulation"].get("STD_MONOMERS", 10)

    std_dev_monomer_direction = 0.5

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
        # ensure constraint of wall in z-direction
        polymer_positions[:, 2] = np.maximum(polymer_positions[:, 2], 0.5 * AGRID + 0.001)
        positions.append(polymer_positions)

    config["simulation"]["POSITIONS"] = positions
    config["simulation"]["MONOMER_COUNTS"] = monomer_counts.tolist()

def add_timeloops_from_gridsize(config):
    steps = config["simulation"]["STEPS"]
    gridsize = config["geometry"]["AGRID"]
    total_time = 0.1 # seconds
    time_unit = 1e-4 # seconds
    total_sim_time = total_time / time_unit
    time_step = config["simulation"]["TIME_STEP"] * gridsize ** 3
    
    config["simulation"]["TIME_STEP"] = time_step
    LOOPS = total_sim_time / (time_step * steps)
    config["simulation"]["LOOPS"] = LOOPS
    
    print(f"gridsize={gridsize}, loops={LOOPS}, steps={steps}")
    
if __name__ == "__main__":
    path = "configs/default.json"
    config = load_config(path)
    run_type = str(sys.argv[1])
    # np.random.seed(config["simulation"]["SEED"])

    if run_type == "test_gridsizes":
        config["simulation"]["SHEAR"] = float(sys.argv[2])
        config["simulation"]["N_POLYMERS"] = int(sys.argv[3])
        config["geometry"]["AGRID"] = float(sys.argv[4])
        run = int(sys.argv[5])
        config["simulation"]["SEED"] = run
        shear = config["simulation"]["SHEAR"]
        gridsize = config["geometry"]["AGRID"]
        np.random.seed(config["simulation"]["SEED"])

        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength=1)
        config["output"]["DIRECTORY"] = f"./data/test_grids/grid_{int(gridsize*1e2):05d}/shear_{int(shear):05d}/run_{run}"
        add_polymer_positions(config, run_type)

    if run_type == "validation_uncoiling":
        config["simulation"]["SHEAR"] = float(sys.argv[2])
        config["simulation"]["N_POLYMERS"] = int(sys.argv[3])
        config["physical"]["GAMMA"] = float(sys.argv[4])
        shear = config["simulation"]["SHEAR"]
        gamma = config["physical"]["GAMMA"]
        LJ_strength = float(sys.argv[5])

        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength)
        config["output"]["DIRECTORY"] = f"./data/validation_uncoil/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/shear_{int(shear):05d}"
        add_polymer_positions(config, run_type)
    
    if run_type == "validation_coiling":
        config["simulation"]["SHEAR"] = float(sys.argv[2])
        config["simulation"]["N_POLYMERS"] = int(sys.argv[3])
        gamma = float(sys.argv[4])
        LJ_strength = float(sys.argv[5])
        config["physical"]["GAMMA"] = gamma
        
        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength)
        config["output"]["DIRECTORY"] = f"./data/validation_coil/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}"
        add_polymer_positions(config, run_type)
    
    lbf, forces, positions = run_simulation(config,run_type)

    save_results(config, lbf, forces, positions)