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
    config["physical"]["cutoff"] = 4 * config["physical"]["a"]
    config["simulation"]["VELOCITY"] = config["simulation"]["SHEAR"] * 1e-4 * (config["geometry"]["BOX_Lz"] - config["geometry"]["AGRID"])

def compute_start_positions(config):
    n_polymers = config["simulation"]["N_POLYMERS"]
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
        random_x = np.array([box_lx / 4])
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
    z = np.full(n_polymers, 0.5 * agrid + 0.001)

    start_positions = np.stack([random_x, random_y, z], axis=1)

    return start_positions

def add_polymer_positions(config):
    start_positions = compute_start_positions(config)

    N_POLYMERS = config["simulation"]["N_POLYMERS"]
    N_MONOMERS = config["physical"]["N_MONOMERS"]
    bond_length = 2 * config["physical"]["a"]
    distribution_type = config["simulation"].get("DISTRIBUTION", "UNIFORM")
    min_monomers = config["simulation"].get("MIN_MONOMERS", 25)
    std_dev = config["simulation"].get("STD_MONOMERS", 10)

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
        direction = np.array([1.0, 0.0, 0.0])  # along x-axis

        polymer_positions = np.array([
            start_pos + i * bond_length/2 * direction for i in range(n_monomers)
        ])
        positions.append(polymer_positions)
    print(positions)

    config["simulation"]["POSITIONS"] = positions
    config["simulation"]["MONOMER_COUNTS"] = monomer_counts.tolist()

def add_config_from_args_validation(config, shear, n_polymers, gamma):
    config["simulation"]["SHEAR"] = shear
    config["simulation"]["N_POLYMERS"] = n_polymers
    config["physical"]["GAMMA"] = gamma

def add_config_from_args_gridsize(config, shear, n_polymers, gridsize):
    config["geometry"]["AGRID"] = gridsize
    config["simulation"]["SHEAR"] = shear
    config["simulation"]["N_POLYMERS"] = n_polymers

    if gridsize==1:
        config["simulation"]["TIME_STEP"] = 0.002
        config["simulation"]["LOOPS"] = 250
    if gridsize==0.5:
        config["simulation"]["TIME_STEP"] = 0.001
        config["simulation"]["LOOPS"] = 1000
    if gridsize==0.25:
        config["simulation"]["TIME_STEP"] = 0.00025
        config["simulation"]["LOOPS"] = 4000
    

if __name__ == "__main__":
    path = "configs/default.json"
    config = load_config(path)

    if len(sys.argv) != 4:
        print("EXIT: Check number of arguments")
        sys.exit(1)

    shear = float(sys.argv[1])
    n_polymers = int(sys.argv[2])
    gridsize = float(sys.argv[3])

    # add_config_from_args_validation(config, shear, n_polymers, gamma)
    add_config_from_args_gridsize(config, shear, n_polymers, gridsize)
    add_dependent_parameters(config, LJ_strength=1)
    config["output"]["DIRECTORY"] = f"./data/grid_{int(gridsize*100):05d}/" + f"shear_{int(shear):05d}/"

    add_polymer_positions(config)
    
    lbf, forces, positions = run_simulation(config)

    save_results(config, lbf, forces, positions)