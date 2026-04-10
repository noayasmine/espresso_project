import os
import numpy as np

def load_npz_first(path):
    d = np.load(path)
    return d[d.files[0]]

def trailing_moving_avg_axis0(arr_TN, window):
    arr = np.asarray(arr_TN, dtype=float)
    T, N = arr.shape
    out = np.empty_like(arr, dtype=float)
    for t in range(T):
        s = max(0, t - window + 1)
        out[t] = np.nanmean(arr[s:t+1], axis=0)
    return out

# ----------------------------
# settings
# ----------------------------
dir_self   = "/home/nroebersen/espresso_project/experiments/sphere_geometry_experiment/sphere_only/dens28_Nmono50/1875/selfassociation"
dir_noself = "/home/nroebersen/espresso_project/experiments/sphere_geometry_experiment/sphere_only/dens28_Nmono50/1875/no_selfassociation"

THRESHOLDS = [20.0, 27.5]  # pN
N_LAST = 20
N_MOV  = 20

IGNORE_Z_BELOW_1 = True   # <<< switch this on/off
Z_CUTOFF = 1.0

# ----------------------------
# main computation
# ----------------------------
def compute_fractions(run_dir, thresholds):
    tens_path = os.path.join(run_dir, "tensile_forces.npz")
    pos_path  = os.path.join(run_dir, "positions.npz")

    if not os.path.exists(tens_path):
        raise FileNotFoundError(f"Missing tensile_forces.npz in: {run_dir}")

    tens = load_npz_first(tens_path).astype(float)   # (T, N_total)
    tens_pN = tens * 1000.0                          # pN

    ma_monomer = trailing_moving_avg_axis0(tens_pN, N_MOV)

    last = min(N_LAST, ma_monomer.shape[0])
    ma_last = ma_monomer[-last:]                           # (last, N_total)
    ma_last_mean_per_monomer = np.nanmean(ma_last, axis=0) # (N_total,)

    # -------------------------------------------------
    # optional: ignore monomers with z < Z_CUTOFF
    # -------------------------------------------------
    if IGNORE_Z_BELOW_1:
        if not os.path.exists(pos_path):
            raise FileNotFoundError(f"Missing positions.npz in: {run_dir}")

        pos = load_npz_first(pos_path).astype(float)
        # assume shape (T, N_total, 3)
        z_last = pos[-last:, :, 2]                 # (last, N_total)
        z_mean = np.nanmean(z_last, axis=0)        # (N_total,)

        valid_mask = z_mean >= Z_CUTOFF

        # apply mask
        ma_last_mean_per_monomer = ma_last_mean_per_monomer[valid_mask]

        if ma_last_mean_per_monomer.size == 0:
            raise ValueError("All monomers filtered out by z-cut.")

    # -------------------------------------------------
    # compute fractions
    # -------------------------------------------------
    fractions = {}
    for thr in thresholds:
        frac = np.sum(ma_last_mean_per_monomer > thr) / ma_last_mean_per_monomer.size
        fractions[thr] = float(frac)

    return fractions

frac_self   = compute_fractions(dir_self, THRESHOLDS)
frac_noself = compute_fractions(dir_noself, THRESHOLDS)

print("=== Fraction of monomers above threshold ===")
print(f"(IGNORE_Z_BELOW_1 = {IGNORE_Z_BELOW_1}, Z_CUTOFF = {Z_CUTOFF})\n")

for thr in THRESHOLDS:
    print(f"Threshold = {thr} pN")
    print(f"  selfassociation      : {frac_self[thr]:.6f}")
    print(f"  no selfassociation   : {frac_noself[thr]:.6f}")
    print()
