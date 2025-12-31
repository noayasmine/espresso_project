#!/usr/bin/env python3
# compare_tensile_moving_avg_minimal.py

import os
import numpy as np
import matplotlib.pyplot as plt


# =====================
# USER SETTINGS
# =====================
BASE_DIR = "/home/nroebersen/espresso_project/experiments/test_vec_forces/no_selfassociation" 
WINDOW   = 20                             # moving average window
MONOMER  = 0                               # monomer index (0-based)

NET_FILE = "tensile_forces.npz"
VEC_FILE = "tensile_forces_vec.npz"


def load_npz_first(path):
    d = np.load(path)
    return d[d.files[0]]


def moving_avg_trailing(arr, window):
    if window <= 1:
        return arr
    T = arr.shape[0]
    out = np.empty_like(arr, dtype=float)
    csum = np.cumsum(arr, axis=0, dtype=float)
    for i in range(T):
        start = i - window + 1
        if start <= 0:
            out[i] = csum[i] / (i + 1)
        else:
            out[i] = (csum[i] - csum[start - 1]) / window
    return out


# =====================
# LOAD DATA
# =====================
net_path = os.path.join(BASE_DIR, NET_FILE)
vec_path = os.path.join(BASE_DIR, VEC_FILE)

net = load_npz_first(net_path)   # (T, N)
vec = load_npz_first(vec_path)   # (T, N, 3)


# =====================
# COMPUTE MOVING AVGS
# =====================
# (A) avg(|F|)
net_avg = moving_avg_trailing(net[:, MONOMER], WINDOW)

# (B) |avg(F_vec)|
vec_avg = moving_avg_trailing(vec[:, MONOMER, :], WINDOW)
net_from_vec_avg = np.linalg.norm(vec_avg, axis=-1)


# =====================
# PLOT
# =====================
t = np.arange(net.shape[0])

plt.figure()
plt.plot(t, net_avg/0.001, label=r"$\langle |F| \rangle_{\mathrm{window}}$")
plt.plot(t, net_from_vec_avg/0.001, label=r"$|\langle \vec F \rangle_{\mathrm{window}}|$")
plt.xlabel("timestep")
plt.ylabel("tensile force (pN)")
plt.title(f"Moving avg tensile force (monomer={MONOMER}, window={WINDOW})")
plt.legend()
plt.tight_layout()

outname = os.path.join(
    BASE_DIR,
    f"compare_moving_avg_monomer{MONOMER}_W{WINDOW}.png"
)
plt.savefig(outname, dpi=200)


print(f"Saved plot to: {outname}")

