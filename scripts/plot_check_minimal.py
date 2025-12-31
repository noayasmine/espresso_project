#!/usr/bin/env python3
# minimal_debug_tensile_raw_and_MA.py

import os
import numpy as np
import matplotlib.pyplot as plt

# =====================
# USER SETTINGS
# =====================
BASE_DIR = "/home/nroebersen/espresso_project/experiments/test_vec_forces/no_selfassociation"
NET_FILE = "tensile_forces.npz"         # (T, N) signed
VEC_FILE = "tensile_forces_vec.npz"     # (T, N, 3) signed vectors
MONOMER  = 0
WINDOW   = 20

def load_npz_first(path):
    d = np.load(path)
    return d[d.files[0]]

# =====================
# LOAD DATA
# =====================
net = load_npz_first(os.path.join(BASE_DIR, NET_FILE))  # (T, N)
vec = load_npz_first(os.path.join(BASE_DIR, VEC_FILE))  # (T, N, 3)
print("net shape:", net.shape, "vec shape:", vec.shape)

t = np.arange(net.shape[0])
f = net[:, MONOMER]          # signed scalar timeseries

# =====================
# MOVING AVG (same simple trailing-window logic)
# =====================
f_ma = np.empty_like(f, dtype=float)
f_vec_ma_abs = np.empty_like(f, dtype=float)

for i in range(len(f)):
    start = max(0, i - WINDOW + 1)

    # scalar moving avg (SIGNED)
    f_ma[i] = f[start:i+1].mean()

    # vector moving avg -> then magnitude
    v_avg = vec[start:i+1, MONOMER, :].mean(axis=0)      # (3,)
    f_vec_ma_abs[i] = np.linalg.norm(v_avg)              # >= 0

# =====================
# PLOTS
# =====================

# 1) raw signed net force
plt.figure()
plt.plot(t, f)
plt.xlabel("timestep")
plt.ylabel("tensile force")
plt.title(f"Raw signed tensile force (monomer {MONOMER})")
plt.tight_layout()
out_raw = os.path.join(BASE_DIR, f"minimal_check_mon{MONOMER}_raw.png")
plt.savefig(out_raw, dpi=200)
plt.close()
print("saved:", out_raw)

# 2) moving averages: signed scalar MA vs |vector MA|
plt.figure()
plt.plot(t, f_ma, label="MA of signed net force")
plt.plot(t, f_vec_ma_abs, label="|MA of vector force|")
plt.xlabel("timestep")
plt.ylabel("tensile force")
plt.title(f"Moving avg comparison (monomer {MONOMER}, window {WINDOW})")
plt.legend()
plt.tight_layout()
out_ma = os.path.join(BASE_DIR, f"minimal_check_mon{MONOMER}_maW{WINDOW}_compare.png")
plt.savefig(out_ma, dpi=200)
plt.close()
print("saved:", out_ma)
