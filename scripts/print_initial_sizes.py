import os
import numpy as np

data_dir = "/home/nroebersen/espresso_project/configs/initial_polymer_positions"

for fname in sorted(os.listdir(data_dir)):
    if fname.endswith(".npz"):
        path = os.path.join(data_dir, fname)
        data = np.load(path)

        # assume one array per file
        arr = data[list(data.keys())[0]]

        print(f"{fname} : {arr.shape}")
