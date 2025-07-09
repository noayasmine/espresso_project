import os
import numpy as np
import matplotlib.pyplot as plt

# Parent data folder
base_root = "data/test_grids"

# List of grid size folders
grid_folders = sorted(
    [f for f in os.listdir(base_root) if f.startswith("grid_")]
)

# For consistent color/legend
grid_labels = []

plt.figure(figsize=(8, 5))

# Loop through each grid size folder
for grid_folder in grid_folders:
    full_grid_path = os.path.join(base_root, grid_folder)
    
    if not os.path.exists(full_grid_path):
        continue

    shear_folders = sorted(
        [f for f in os.listdir(full_grid_path) if f.startswith("shear_")],
        key=lambda x: int(x.split("_")[1])
    )

    shear_rates = []
    avg_distances = []

    for shear_folder in shear_folders:
        shear_rate = int(shear_folder.split("_")[1])
        shear_rates.append(shear_rate)

        file_path = os.path.join(full_grid_path, shear_folder, "files", "positions.npz")
        if not os.path.exists(file_path):
            continue

        data = np.load(file_path)
        positions = data["arr_0"]  # shape: (ntimesteps, 60, 3)

        # Compute distances between monomer i and i+30
        distances = np.linalg.norm(positions[:, :30, :] - positions[:, 30:, :], axis=2)
        avg_distance = np.mean(distances)
        avg_distances.append(avg_distance)

    # Label: extract numeric grid size from folder name
    grid_size_value = int(grid_folder.split("_")[2]) / 100
    grid_labels.append(f"Grid {grid_size_value:.2f}")

    # Plot shear vs avg distance
    plt.plot(shear_rates, avg_distances, marker='o', label=f"Grid {grid_size_value:.2f}")

# Finalize plot
plt.xlabel("Shear Rate")
plt.ylabel("Average Distance Between Matching Monomers")
plt.title("Distance vs Shear Rate for Different Grid Sizes (Gamma=0.005)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = os.path.join(base_root, "avg_distance_vs_shear_and_grid.png")
plt.savefig(plot_path, dpi=300)
