import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

def load_config(path):
    with open(path) as f:
        return json.load(f)

def compute_avg_polymer_distance(npz_path, n_polymers=2, monomers_per_polymer=30):
    try:
        data = np.load(npz_path)
        if "positions" not in data:
            print(f"Warning: 'positions' key not found in {npz_path}")
            return None

        positions = data["positions"]  # shape: (timesteps, total_monomers, 3)
        ntimesteps, total_monomers, _ = positions.shape
        expected = n_polymers * monomers_per_polymer
        if total_monomers != expected:
            print(f"Warning: Unexpected number of monomers in {npz_path}")
            return None

        # Reshape and split polymers
        reshaped = positions.reshape((ntimesteps, n_polymers, monomers_per_polymer, 3))
        polymer1 = reshaped[:, 0]
        polymer2 = reshaped[:, 1]

        distances = np.linalg.norm(polymer1 - polymer2, axis=-1)
        avg_distance = np.mean(distances, axis=1)

        return avg_distance
    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        return None

def plot_grid_results(config_path, base_dir):
    plt.figure(figsize=(12, 6))
    config = load_config(config_path)

    grid_pattern = re.compile(r"grid_(\d+)")
    shear_pattern = re.compile(r"shear_(\d+)")

    for grid_folder in sorted(os.listdir(base_dir)):
        grid_match = grid_pattern.match(grid_folder)
        if not grid_match:
            continue

        grid_value = int(grid_match.group(1)) / 100.0
        grid_path = os.path.join(base_dir, grid_folder)

        for shear_folder in sorted(os.listdir(grid_path)):
            shear_match = shear_pattern.match(shear_folder)
            if not shear_match:
                continue

            shear_value = int(shear_match.group(1))
            shear_path = os.path.join(grid_path, shear_folder)

            npz_path = os.path.join(shear_path, "positions.npz")
            print(f"Processing: gridsize={grid_value}, shear={shear_value}")

            avg_distances = compute_avg_polymer_distance(npz_path)
            if avg_distances is not None:
                label = f"grid={grid_value}, shear={shear_value}"
                plt.plot(avg_distances, label=label)

    plt.xlabel("Timestep")
    plt.ylabel("Avg. distance between polymers")
    plt.title("Polymer separation over time")
    plt.legend(fontsize='small', loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(base_dir, "polymer_distances.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()