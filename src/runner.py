import json
import sys
import numpy as np
import time
from simulation import run_simulation
from utils import save_results, save_velocity_field
import os

def load_config(path):
    with open(path) as f:
        return json.load(f)
    
def wall_height(config):
    """z of the particle walls and tether plane. With NEARWALL_FIX (patched ESPResSo velocity readout)
    this is the halfway bounce-back LB wall at AGRID; otherwise the boundary node centre at 0.5*AGRID."""
    AGRID = config["geometry"]["AGRID"]
    return AGRID if config["simulation"].get("NEARWALL_FIX", False) else 0.5 * AGRID

def add_dependent_parameters(config, LJ_strength=18):
    config["physical"]["r_0"] = 2 * config["physical"]["a"]
    config["physical"]["sigma"] = (2 ** (-1/6)) * config["physical"]["r_0"]
    config["physical"]["epsilon"] = LJ_strength * config["physical"]["KT"]
    config["physical"]["cutoff"] = 6 * config["physical"]["a"]
    # Compute velocity (in MD units, i.e. 1e2 m/s )
    if config["simulation"].get("NEARWALL_FIX", False):
        # Patched readout: interpolated profile is linear from the LB walls at AGRID and BOX_Lz - AGRID.
        config["simulation"]["VELOCITY"] = config["simulation"]["SHEAR"] * 1e-4 * (config["geometry"]["BOX_Lz"] - 2 * config["geometry"]["AGRID"])
    elif config["simulation"]["TETHERED"]:
        # Halfway bounce-back puts the LB walls at AGRID and BOX_Lz - AGRID, so the fluid gap is
        # BOX_Lz - 2*AGRID. Tethered beads sit in the first cell, where the interpolated velocity
        # runs from the boundary node (u_wall) to the first fluid node and has half the bulk slope.
        # Doubling the velocity makes that first-cell shear equal to SHEAR (same as the old 4/3 for Lz=4, AGRID=1).
        config["simulation"]["VELOCITY"] = 2 * config["simulation"]["SHEAR"] * 1e-4 * (config["geometry"]["BOX_Lz"] - 2 * config["geometry"]["AGRID"])
    else:
        config["simulation"]["SHEAR"] = (4/3) * config["simulation"]["SHEAR"]
        config["simulation"]["VELOCITY"] = config["simulation"]["SHEAR"] * 1e-4 * (config["geometry"]["BOX_Lz"] - config["geometry"]["AGRID"])

def sample_points_min_dist_2d(n_points, bounds, min_dist, offset, max_attempts=100000, xc=None, yc=None, r_excl=1.8):
    """
    Sample n_points in 2D within given bounds, with pairwise distance >= min_dist.
    Optionally exclude points within radius r_excl around (xc, yc).

    bounds = ((x_min, x_max), (y_min, y_max))
    """
    (x_min, x_max), (y_min, y_max) = bounds
    pts = []
    attempts = 0

    use_exclusion = (xc is not None) and (yc is not None)

    while len(pts) < n_points and attempts < max_attempts:
        attempts += 1
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        # optional exclusion disk
        if use_exclusion and ((x - xc)**2 + (y - yc)**2 <= r_excl**2):
            continue

        p = np.array([x, y])

        # accept if far enough from all existing points
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)

    if len(pts) < n_points:
        raise RuntimeError("Could not place all points with the given min_dist / exclusion region.")

    pts = np.array(pts)
    random_x, random_y = pts[:, 0], pts[:, 1]
    z = np.full(n_points, offset)
    start_positions = np.stack([random_x, random_y, z], axis=1)

    return start_positions

def sample_sphere_filtered_min_dist(npoints, xc, yc, zc, R, min_dist, min_z_val_sphere, max_attempts=1000000):
    
    xs, ys, zs = [], [], []
    attempts = 0

    while len(xs) < npoints and attempts < max_attempts:
        attempts += 1

        v = np.random.randn(3)
        v /= np.linalg.norm(v)

        x = xc + R * v[0]
        y = yc + R * v[1]
        z = zc + R * v[2]

        if (z < min_z_val_sphere):
            continue

        # min-distance check (3D)
        ok = True
        for xi, yi, zi in zip(xs, ys, zs):
            if np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) < min_dist:
                ok = False
                break

        if ok:
            xs.append(x)
            ys.append(y)
            zs.append(z)

    if len(xs) < npoints:
        raise RuntimeError(
            f"Could only sample {len(xs)} / {npoints} points "
            f"(min_dist={min_dist}, R={R}).")

    x = np.array(xs)
    y = np.array(ys)
    z = np.array(zs)

    start_positions = np.stack([x, y, z], axis=1)
    return start_positions


def sample_plane_with_xy_bounds_min_dist(n_points, p0, v1, v2, x_bounds, y_bounds, min_dist, normal_offset=0.0, uv_patch=((0.0, 1.0), (0.0, 1.0)), max_attempts=500000, xc=None, yc=None, r_excl=None):
    """
    Sample n_points on the plane p(u,v)=p0+u*v1+v*v2, but ONLY keep points whose (x,y)
    are inside x_bounds and y_bounds (world-space). Bounds are separate from plane definition.

    Additionally offsets each accepted point by `normal_offset` along the plane unit normal.

    Steps:
      - sample (x,y) uniformly in the given bounds
      - solve for (u,v) from the 2x2 system using the XY components of v1,v2
      - compute point p on plane from u,v
      - optionally require (u,v) inside uv_patch (default: [0,1]x[0,1])
      - offset by normal: p <- p + normal_offset * n_hat
      - enforce 3D min_dist on the OFFSET points
      - optional exclusion disk in world (x,y)
    """
    p0 = np.asarray(p0, float); v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
    (x_min, x_max) = x_bounds
    (y_min, y_max) = y_bounds
    (u_min, u_max), (v_min, v_max) = uv_patch

    # plane unit normal
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-14:
        raise ValueError("v1 and v2 are collinear; plane normal is undefined.")
    n_hat = n / n_norm
    n_hat = -n_hat
    print(n_hat)

    # Build 2x2 map from (u,v) -> (x,y):  [x;y] = [p0x;p0y] + u*[v1x;v1y] + v*[v2x;v2y]
    A = np.array([[v1[0], v2[0]],
                  [v1[1], v2[1]]], dtype=float)

    det = np.linalg.det(A)
    if abs(det) < 1e-14:
        raise ValueError("Cannot solve (u,v) from (x,y): the plane's XY projection is singular (det ~ 0).")

    Ainv = np.linalg.inv(A)

    use_excl = (xc is not None) and (yc is not None) and (r_excl is not None)

    pts = []
    attempts = 0
    while len(pts) < n_points and attempts < max_attempts:
        attempts += 1

        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        # optional exclusion disk in world (x,y)
        if use_excl and ((x - xc)**2 + (y - yc)**2 <= r_excl**2):
            continue

        rhs = np.array([x - p0[0], y - p0[1]], dtype=float)
        u, v = Ainv @ rhs

        # keep only if within desired patch on the plane
        if not (u_min <= u <= u_max and v_min <= v <= v_max):
            continue

        # point on plane, then offset along normal
        p = p0 + u * v1 + v * v2
        p = p + normal_offset * n_hat

        # 3D min-dist check (on the OFFSET points)
        ok = True
        for q in pts:
            if np.linalg.norm(p - q) < min_dist:
                ok = False
                break
        if ok:
            pts.append(p)

    if len(pts) < n_points:
        raise RuntimeError(f"Could only place {len(pts)} / {n_points} points. Try lowering min_dist or increasing bounds/attempts.")

    return np.asarray(pts)

def compute_start_positions(config):
    """Computes positions of first monomers.

    Returns
    -------
    start_positions : (N_POLYMERS, 3)
    + adds start_positions to json file (config)
    """
    np.random.seed(config["simulation"]["SEED"])
    N_POLYMERS = config["simulation"]["N_POLYMERS"]
    N_MONOMERS = config["physical"]["N_MONOMERS"]

    a = config["physical"]["a"]
    AGRID = config["geometry"]["AGRID"]
    BOX_Lx = config["geometry"]["BOX_Lx"]
    BOX_Ly = config["geometry"]["BOX_Ly"]
    BOX_Lz = config["geometry"]["BOX_Lz"]
    if N_MONOMERS == 50:
        min_dist = 15.0 * a
    if N_MONOMERS == 100:
        min_dist = 16.0 * a

    wall_z = wall_height(config)
    offset = wall_z + a + 1e-07
    offset_sphere = 2*a + 1e-07

    if N_POLYMERS == 1 and not (config["geometry"]["SPHERE"] or config["geometry"]["MICROCONTRACTION"]):
        random_x = np.array([BOX_Lx / 5])
        random_y = np.array([BOX_Ly / 2])
        z = np.full(N_POLYMERS, offset)
        start_positions = np.stack([random_x, random_y, z], axis=1)

    elif N_POLYMERS == 2 and not (config["geometry"]["SPHERE"] or config["geometry"]["MICROCONTRACTION"]):
        spacing = 6 * a
        center_x = BOX_Lx / 4
        center_y = BOX_Ly / 2
        random_x = np.array([center_x, center_x])
        random_y = np.array([center_y - spacing / 2, center_y + spacing / 2])
        z = np.full(N_POLYMERS, offset)
        start_positions = np.stack([random_x, random_y, z], axis=1)
    
    elif config["geometry"]["SPHERE"]:
        R = config["geometry"].get("RADIUS_OBJECT")
        xc = config["geometry"].get("SPHERE_CENTER_X")
        yc = config["geometry"].get("SPHERE_CENTER_Y")
        zc = config["geometry"].get("RADIUS_OBJECT") + wall_z
        config["geometry"]["SPHERE_CENTER_Z"] = zc
        density = config["simulation"]["POLYMER_DENSITY"]
        # setting minimal value of z=1.1 (0.6 above the wall) for tethered polymers on the sphere to satisfy wall overlapping monomers constraints
        min_z_val_sphere = wall_z + 0.6
        # excluding area with radius r_excl around sphere center for polymers on ground 
        # to satisfy wall overlapping monomers constraints
        r_excl = 1.8 

        x_bounds = (xc - 3.5 * AGRID, xc + 3.5 * AGRID)
        y_bounds = (yc - 3.5 * AGRID, yc + 3.5 * AGRID)

        radius_particle_sampling = config["geometry"].get("RADIUS_OBJECT") + offset_sphere

        cut_distance = min_z_val_sphere - wall_z
        sphere_area = 4 * np.pi * radius_particle_sampling ** 2 - (2 * np.pi * radius_particle_sampling * (radius_particle_sampling - cut_distance))
        ground_area = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0]) - np.pi * r_excl ** 2
        
        N_POLYMERS_SPHERE = int(sphere_area * density)
        N_POLYMERS_GROUND = int(ground_area * density)
        N_POLYMERS = N_POLYMERS_SPHERE + N_POLYMERS_GROUND
        config["simulation"]["N_POLYMERS_SPHERE"] = N_POLYMERS_SPHERE
        config["simulation"]["N_POLYMERS"] = N_POLYMERS

        print("sphere radius, particle sphere radius, xc,yc,zc")
        print(R, radius_particle_sampling, xc, yc, zc)
        print("N_POLYMERS_SPHERE, N_POLYMERS_GROUND, N_POLYMERS")
        print(N_POLYMERS_SPHERE, N_POLYMERS_GROUND, N_POLYMERS)
        print("DENSITY")
        print(density)

        start_positions_sphere = sample_sphere_filtered_min_dist(N_POLYMERS_SPHERE, xc, yc, zc, R=radius_particle_sampling, min_dist=min_dist, min_z_val_sphere=min_z_val_sphere, max_attempts=100000)

        if N_POLYMERS > N_POLYMERS_SPHERE:
            start_positions_ground = sample_points_min_dist_2d(N_POLYMERS_GROUND, (x_bounds, y_bounds), min_dist, offset, xc=xc, yc=yc, r_excl=r_excl)

        start_positions = np.vstack([start_positions_sphere, start_positions_ground])
    
    elif config["geometry"]["MICROCONTRACTION"]:

        print("Adding positions from microcontraction")
        density = config["simulation"]["POLYMER_DENSITY"]
        y_bounds = (AGRID + 2.0 * a, BOX_Ly - (AGRID + 2.0 * a))
        offset = wall_z + 3*a + 1e-07

        # first bottom ground area
        x_bounds_1 = (AGRID, 2.5 - 15*a)

        ground_area_1 = (x_bounds_1[1] - x_bounds_1[0]) * (y_bounds[1] - y_bounds[0])
        N_POLYMERS_GROUND_1 = int(ground_area_1 * density)
        print(N_POLYMERS_GROUND_1)
        start_positions_ground_1 = sample_points_min_dist_2d(N_POLYMERS_GROUND_1, (x_bounds_1, y_bounds), min_dist, offset)

        # flat area on top of the microcontraction
        x_bounds_2 = (4.5, 5.5)

        ground_area_2 = (x_bounds_2[1] - x_bounds_2[0]) * (y_bounds[1] - y_bounds[0])
        N_POLYMERS_GROUND_2 = int(ground_area_2 * density)
        print(N_POLYMERS_GROUND_2)
        start_positions_ground_2 = sample_points_min_dist_2d(N_POLYMERS_GROUND_2, (x_bounds_2, y_bounds), min_dist, offset=offset+4.25)

        # second bottom ground area
        x_bounds_3 = (7.5 + 15*a, 10)
        ground_area_3 = (x_bounds_3[1] - x_bounds_3[0]) * (y_bounds[1] - y_bounds[0])
        N_POLYMERS_GROUND_3 = int(ground_area_3 * density)
        print(N_POLYMERS_GROUND_3)
        start_positions_ground_3 = sample_points_min_dist_2d(N_POLYMERS_GROUND_3, (x_bounds_3, y_bounds), min_dist, offset)

        area_sides = np.sqrt((2 - AGRID)**2 + (4 - AGRID)**2) * (y_bounds[1] - y_bounds[0]) 
        N_POLYMERS_MICRO = int(area_sides * density)

        # LEFT plane
        p0 = np.array([2.5, 0.0, 0.0])
        v1 = np.array([0.0, 8.0, 0.0])
        v2 = np.array([2., 0.0, 4.])

        start_positions_micro_left = sample_plane_with_xy_bounds_min_dist(
            n_points=N_POLYMERS_MICRO,
            p0=p0, v1=v1, v2=v2,
            x_bounds=(2.5 + AGRID, 4.5),
            y_bounds=y_bounds,            # or your y_bounds variable
            min_dist=min_dist,
            normal_offset=offset,
            uv_patch=((0.0, 1.0), (0.0, 1.0)))

        # RIGHT plane
        p0 = np.array([5.5, 0.0, 4.])
        v1 = np.array([0.0, 8.0, 0.0])
        v2 = np.array([2., 0.0, -4.])   # downward tilt

        start_positions_micro_right = sample_plane_with_xy_bounds_min_dist(
            n_points=N_POLYMERS_MICRO,
            p0=p0, v1=v1, v2=v2,
            x_bounds=(5.5, 7.5 - AGRID),
            y_bounds=y_bounds,            # or your y_bounds variable
            min_dist=min_dist,
            normal_offset=offset,
            uv_patch=((0.0, 1.0), (0.0, 1.0)))

        start_positions = np.vstack([start_positions_micro_left, start_positions_micro_right, start_positions_ground_1, start_positions_ground_2, start_positions_ground_3])
        
        config["simulation"]["N_POLYMERS_MICROLEFT"] = N_POLYMERS_MICRO
        config["simulation"]["N_POLYMERS_MICRORIGHT"] = N_POLYMERS_MICRO
        config["simulation"]["N_POLYMERS"] = N_POLYMERS_GROUND_1 + N_POLYMERS_GROUND_2 + N_POLYMERS_GROUND_3 + 2 * N_POLYMERS_MICRO

    else:
        x_bounds = (AGRID + 2.0 * a, BOX_Lx - (AGRID + 2.0 * a))
        y_bounds = (AGRID + 2.0 * a, BOX_Ly - (AGRID + 2.0 * a))

        start_positions = sample_points_min_dist_2d(N_POLYMERS, (x_bounds, y_bounds), min_dist, offset)

    config["simulation"]["START_POSITIONS"] = start_positions
    print("Positions first monomers:")
    print(start_positions)

    return start_positions

def _unit(v):
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n

def rotation_matrix_from_vectors(a, b):
    """
    Return R such that R @ a == b for unit vectors a, b (in R^3).
    Uses Rodrigues' rotation formula, with careful handling of a ~ +/- b.
    """
    a = _unit(np.asarray(a, dtype=float))
    b = _unit(np.asarray(b, dtype=float))

    v = np.cross(a, b)
    c = float(np.dot(a, b))  # cos(theta)
    s = np.linalg.norm(v)    # sin(theta)

    if s < 1e-12:
        # a and b are (anti-)parallel
        if c > 0.0:
            return np.eye(3)  # same direction
        # opposite direction: rotate 180 degrees about any axis orthogonal to a
        # choose an axis not parallel to a
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, tmp))
        # Rodrigues for theta = pi: R = I + 2*K^2 (since sin(pi)=0, 1-cos(pi)=2)
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        return np.eye(3) + 2.0 * (K @ K)

    # general case
    axis = v / s
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    # Rodrigues: R = I + sinθ K + (1-cosθ) K^2
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)

def check_polymer_geometry(positions, a, max_consecutive=1.25, min_pairwise=0.85, check_all_pairs=True, raise_on_fail=True):
    """
    positions: (N_POLYMERS, N_MONOMERS, 3)
    max_consecutive: consecutive distance must be <= max_consecutive*(2a)
    min_pairwise: any pairwise distance must be >= min_pairwise*(2a)
    """
    Np, Nm, _ = positions.shape
    twoa = 2.0 * float(a)
    max_bond = max_consecutive * twoa
    min_sep = min_pairwise * twoa

    bad = False

    for i in range(Np):
        r = positions[i]  # (Nm, 3)

        # 1) consecutive monomer distances: validates "order" matches a linear chain
        dr = r[1:] - r[:-1]
        d = np.linalg.norm(dr, axis=1)
        dmax = float(d.max()) if d.size else 0.0
        dmin = float(d.min()) if d.size else 0.0
        if np.any(d > max_bond):
            bad = True
            idx = np.where(d > max_bond)[0]
            print(f"[FAIL] polymer {i}: consecutive distances too large at bonds {idx.tolist()}. "
                  f"max={dmax:.6g}, threshold={max_bond:.6g} (2a={twoa:.6g})")
        else:
            print(f"[OK] polymer {i}: consecutive distances in [{dmin:.6g}, {dmax:.6g}] (2a={twoa:.6g})")

        # 2) intra-polymer overlap / too-close check
        if check_all_pairs and Nm >= 2:
            # full pairwise distance matrix without loops
            diff = r[:, None, :] - r[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            iu = np.triu_indices(Nm, k=1)
            d_pairs = dist[iu]
            pmin = float(d_pairs.min()) if d_pairs.size else np.inf
            if np.any(d_pairs < min_sep):
                bad = True
                w = np.where(d_pairs < min_sep)[0]
                # show up to first few offending pairs
                show = w[:8]
                pairs = list(zip(iu[0][show].tolist(), iu[1][show].tolist()))
                print(f"[FAIL] polymer {i}: monomers too close. min_pairwise={pmin:.6g}, "
                      f"threshold={min_sep:.6g} (2a={twoa:.6g}). Example pairs {pairs}")
            else:
                print(f"[OK] polymer {i}: min pairwise distance = {pmin:.6g} (threshold={min_sep:.6g})")

    if bad and raise_on_fail:
        raise ValueError("Polymer geometry check failed: order/bond lengths or intra-polymer overlaps detected.")

def load_polymer_positions(start_positions, config, pos_dir=None):
    """
    Places pre-defined polymer conformations at start_positions.
    Polymers whose start position lies on the sphere are rotated
    so that they align with the local sphere normal.
    """
    
    N_MONOMERS = config["physical"]["N_MONOMERS"]
    seed = config["simulation"]["SEED"]
    k = int(config["physical"]["k"])
    AGRID = config["geometry"]["AGRID"]
    a = config["physical"]["a"]
    N_POLYMERS_SPHERE = config["simulation"]["N_POLYMERS_SPHERE"]

    microcontraction = bool(config["geometry"].get("MICROCONTRACTION", False))
    N_POLYMERS_MICROLEFT  = int(config["simulation"].get("N_POLYMERS_MICROLEFT", 0))
    N_POLYMERS_MICRORIGHT = int(config["simulation"].get("N_POLYMERS_MICRORIGHT", 0))  # this should be TOTAL micro (left+right) or right-end index as you use it below

    if N_MONOMERS==50:
        pos_dir = f"./configs/initial_polymer_positions_k_1000"
        path_to_file = os.path.join(pos_dir, f"init_positions_{N_MONOMERS}_seed_{seed}.npz")
    if N_MONOMERS==100:
        pos_dir = f"./configs/init_polymer_positions_multiple_runs_N100/run_{seed}"
        path_to_file = os.path.join(pos_dir, f"init_positions.npz")

    print(f"Read out position file: {path_to_file}")
    data = np.load(path_to_file)
    init_positions = data["positions"]      # (1, N_MONOMERS, 3)

    # normalize polymer to origin
    origin = init_positions[0, 0]
    shifted = init_positions - origin       # (1, N_MONOMERS, 3)

    N_POLYMERS = start_positions.shape[0]
    positions = np.zeros((N_POLYMERS, N_MONOMERS, 3))

    # sphere geometry
    sphere = config["geometry"]["SPHERE"]
    xc = config["geometry"].get("SPHERE_CENTER_X")
    yc = config["geometry"].get("SPHERE_CENTER_Y")
    zc = config["geometry"].get("SPHERE_CENTER_Z")
    offset_sphere = 2*a + 1e-07
    R = float(config["geometry"]["RADIUS_OBJECT"]) + offset_sphere
    center = np.array([xc, yc, zc], dtype=float)
    ez = np.array([0.0, 0.0, 1.0], dtype=float)  # original polymer "normal"

    # microcontraction geometry
    #n_left  = _unit(np.array([-2.0, 0.0, 4.0], dtype=float))
    n_left  = _unit(np.array([-4.0, 0.0, 2.0], dtype=float))
    print(n_left)
    #n_right = _unit(np.array([ 2.0, 0.0, 4.0], dtype=float))
    n_right = _unit(np.array([ 4.0, 0.0, 2.0], dtype=float))
    print(n_right)

    # precompute rotation matrices once (constant normals)
    R_left  = rotation_matrix_from_vectors(ez, n_left)
    R_right = rotation_matrix_from_vectors(ez, n_right)
    
    for i in range(N_POLYMERS):
        base = shifted[0]           # (N_MONOMERS, 3)
        pos0 = start_positions[i]   # (3,)

        polymer = base.copy()

        if sphere and i < N_POLYMERS_SPHERE:
            n = _unit(pos0 - center)                 # outward normal at pos0
            R = rotation_matrix_from_vectors(ez, n)  # rotate +z -> n
            polymer = (R @ polymer.T).T              # apply to all monomers
        
        if microcontraction and i < N_POLYMERS_MICROLEFT:
            # rotate for left plane
            polymer = (R_left @ polymer.T).T

        if microcontraction and N_POLYMERS_MICROLEFT <= i < N_POLYMERS_MICRORIGHT + N_POLYMERS_MICROLEFT:
            # rotate for right plane
            polymer = (R_right @ polymer.T).T

        # translate polymer to its start position
        positions[i] = polymer + pos0

    # check_polymer_geometry(positions, a=a, max_consecutive=1.1, min_pairwise=0.95, check_all_pairs=True, raise_on_fail=True)

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
            polymer_positions[:, 2] = np.maximum(polymer_positions[:, 2], wall_height(config) + a + 1e-07)
            positions.append(polymer_positions)
        
        config["simulation"]["POSITIONS"] = positions

    elif COILED and not COMPUTE_POSITIONS:
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
            config["simulation"]["COMPUTE_POSITIONS"] = False
            config["output"]["DIRECTORY"] = f"./{save_dir}/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/shear_{int(shear):05d}/run_{run}/"

        if len(sys.argv) == 6:
            shear = float(sys.argv[2])
            gamma = float(sys.argv[3])
            LJ_strength = float(sys.argv[4])
            save_dir = str(sys.argv[5])
            config["physical"]["N_MONOMERS"] = 50
            config["simulation"]["SAVE_VTK"] = True
            config["simulation"]["SEED"] = 1
            config["simulation"]["TOTAL_TIME"] = 0.3
            config["output"]["DIRECTORY"] = f"./{save_dir}/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/shear_{int(shear):05d}/"
        
        # set parameters to match the validation data from Fu et al.
        config["simulation"]["SHEAR"] = shear
        config["physical"]["GAMMA"] = gamma
        config["simulation"]["N_POLYMERS"] = 1
        config["geometry"]["AGRID"] = 1
        # change number of monomers in the polymer
        
        config["physical"]["a"] = 0.03
        config["physical"]["k"] = 50

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
            LJ_strength = float(sys.argv[2])
            N_monomers = int(sys.argv[3])
            run = int(sys.argv[4])
            save_dir = str(sys.argv[5])
            # set run id as random seed
            gamma = config["physical"]["GAMMA"]
            config["simulation"]["SEED"] = run
            config["output"]["DIRECTORY"] = f"./{save_dir}/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/run_{run}/"

        if len(sys.argv) == 7:
            gamma = float(sys.argv[2])
            LJ_strength = float(sys.argv[3])
            stdev_init = float(sys.argv[4])
            N_monomers = float(sys.argv[5])
            save_dir = str(sys.argv[6])
            config["simulation"]["stdev_initial_pos"] = stdev_init
            # config["simulation"]["SAVE_VTK"] = True
            config["output"]["DIRECTORY"] = f"./{save_dir}/LJ_{int(LJ_strength*1e2):05d}/gamma_{int(gamma*1e4):05d}/"

        config["simulation"]["COILED"] = False
        # for recoiling, the experimental data added 60% sucrose, increased viscosity and density
        config["fluid"]["VISC"] = config["fluid"]["VISC"] * 58
        config["fluid"]["DENS"] = 0.00013
        config["geometry"]["AGRID"] = 1
        config["simulation"]["N_POLYMERS"] = 1
        config["simulation"]["TOTAL_TIME"] = 0.8
        config["simulation"]["MINIMIZATION_STEPS"] = 0
        config["physical"]["N_MONOMERS"] = N_monomers

        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength)
        add_polymer_positions(config)
    
    if run_type == "check_forces":
        shear = float(sys.argv[2])
        k = float(sys.argv[3])
        save_dir = str(sys.argv[4])

        config["output"]["DIRECTORY"] = f"./{save_dir}/{int(k)}/{int(shear)}/"

        config["simulation"]["SAVE_VTK"] = True
        config["simulation"]["SHEAR"] = shear
        config["simulation"]["TOTAL_TIME"] = 0.2

        config["simulation"]["COMPUTE_POSITIONS"] = False
        config["simulation"]["N_POLYMERS"] = 1
        config["physical"]["k"] = k
        config["physical"]["N_MONOMERS"] = 50

        if config["physical"]["N_MONOMERS"] == 250:
            config["geometry"]["BOX_Lx"] = 20

        config["simulation"]["SEED"] = 1

        config["physical"]["GAMMA"] = 0.011
        config["simulation"]["LB_TIME_STEP"] = 0.001
        config["simulation"]["TIME_STEP"] = 0.001
        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config, LJ_strength=0)
        add_polymer_positions(config)
    
    if run_type == "experiment":
        shear = float(sys.argv[2])
        config["simulation"]["SELF_ASSOCIATE"] = sys.argv[3].lower() == "true"
        save_dir = str(sys.argv[4])

        if config["simulation"]["SELF_ASSOCIATE"]:
            config["output"]["DIRECTORY"] = f"./{save_dir}/selfassociation/"

        else:
            config["output"]["DIRECTORY"] = f"./{save_dir}/no_selfassociation/"

        config["simulation"]["SAVE_VTK"] = True
        config["simulation"]["SHEAR"] = shear
        config["simulation"]["TOTAL_TIME"] = 0.02
        config["FORCE_THRESHOLD"] =  0.01

        config["simulation"]["COMPUTE_POSITIONS"] = False
        config["simulation"]["N_POLYMERS"] = 1
        # config["simulation"]["MINIMIZATION_STEPS"] = 600
        config["physical"]["k"] = 1000
        config["physical"]["N_MONOMERS"] = 50
        config["simulation"]["SEED"] = 1

        config["simulation"]["LB_TIME_STEP"] = 0.001
        config["simulation"]["TIME_STEP"] = 0.001
        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config)
        add_polymer_positions(config)
    
    if run_type == "check_velocity_field":
        shear = float(sys.argv[2])
        save_dir = str(sys.argv[3])

        config["output"]["DIRECTORY"] = f"./{save_dir}/{int(shear)}/"

        config["simulation"]["SAVE_VTK"] = True
        config["simulation"]["SHEAR"] = shear
        config["simulation"]["TOTAL_TIME"] = 0.001
        config["FORCE_THRESHOLD"] =  0.01

        config["geometry"]["SPHERE"] = False
        if config["geometry"]["SPHERE"]:
            config["geometry"]["BOX_Lx"] = 100
            config["geometry"]["BOX_Ly"] = 50
            config["geometry"]["BOX_Lz"] = 30

        config["simulation"]["COMPUTE_POSITIONS"] = False
        config["simulation"]["N_POLYMERS"] = 0
        config["physical"]["k"] = 1000
        config["physical"]["N_MONOMERS"] = 50
        config["simulation"]["SEED"] = 1

        config["simulation"]["LB_TIME_STEP"] = 0.001
        config["simulation"]["TIME_STEP"] = 0.001
        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config)

    if run_type == "experiment_density":
        shear = float(sys.argv[2])
        N_POLYMERS = int(sys.argv[3])
        config["simulation"]["SELF_ASSOCIATE"] = sys.argv[4].lower() == "true"
        run = int(sys.argv[5])
        save_dir = str(sys.argv[6])

        if config["simulation"]["SELF_ASSOCIATE"]:
            config["output"]["DIRECTORY"] = f"./{save_dir}/{int(shear)}/selfassociation/N_polymers_{N_POLYMERS}/"

        else:
            config["output"]["DIRECTORY"] = f"./{save_dir}/{int(shear)}/no_selfassociation/N_polymers_{N_POLYMERS}/"
        
        config["simulation"]["SAVE_VTK"] = False
        config["simulation"]["SAVE_EVERY"] = 5
        config["simulation"]["SHEAR"] = shear
        config["simulation"]["TOTAL_TIME"] = 0.15
        # 4 um of fluid between the halfway bounce-back LB walls at AGRID and BOX_Lz - AGRID (as in the paper)
        config["geometry"]["BOX_Lz"] = 4 + 2 * config["geometry"]["AGRID"]
        config["FORCE_THRESHOLD"] =  0.01

        config["simulation"]["COMPUTE_POSITIONS"] = False

        config["simulation"]["N_POLYMERS"] = N_POLYMERS

        config["physical"]["k"] = 1000
        config["physical"]["N_MONOMERS"] = 50
        config["simulation"]["SEED"] = run

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
            config["output"]["DIRECTORY"] = f"./{save_dir}/{int(shear)}/selfassociation/"

        else:
            config["output"]["DIRECTORY"] = f"./{save_dir}/{int(shear)}/no_selfassociation/"

        config["simulation"]["POLYMER_DENSITY"] = 1.8

        config["simulation"]["SAVE_VTK"] = True
        config["simulation"]["SAVE_EVERY"] = 5
        config["simulation"]["SHEAR"] = shear

        config["simulation"]["TOTAL_TIME"] = 0.12
        config["simulation"]["COMPUTE_POSITIONS"] = False
        config["simulation"]["SEED"] = 1

        config["geometry"]["BOX_Lx"] = 32
        config["geometry"]["BOX_Ly"] = 12
        config["geometry"]["BOX_Lz"] = 12

        config["geometry"]["AGRID"] = 1
        config["geometry"]["SPHERE"] = True
        config["physical"]["k"] = 1000
        config["physical"]["N_MONOMERS"] = 100

        config["simulation"]["TIME_STEP"] = 0.001/4
        config["simulation"]["LB_TIME_STEP"] = config["simulation"]["TIME_STEP"]
        config["simulation"]["STEPS"] = int(1000*4)
        
        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config)
        add_polymer_positions(config)

    if run_type == "experiment_microcontraction":
        shear = float(sys.argv[2])
        config["simulation"]["SELF_ASSOCIATE"] = sys.argv[3].lower() == "true"
        save_dir = str(sys.argv[4])

        print("Run simulation with microcontraction")

        if config["simulation"]["SELF_ASSOCIATE"]:
            config["output"]["DIRECTORY"] = f"./{save_dir}/{int(shear)}/selfassociation/"

        else:
            config["output"]["DIRECTORY"] = f"./{save_dir}/{int(shear)}/no_selfassociation/"

        config["simulation"]["COMPUTE_POSITIONS"] = False
        config["simulation"]["SAVE_VTK"] = True
        config["simulation"]["SAVE_EVERY"] = 5
        config["simulation"]["SHEAR"] = shear
        config["simulation"]["SEED"] = 1
        config["simulation"]["TOTAL_TIME"] = 0.005

        config["geometry"]["BOX_Lx"] = 14
        config["geometry"]["BOX_Ly"] = 8
        config["geometry"]["BOX_Lz"] = 6

        config["simulation"]["POLYMER_DENSITY"] = 2
        #config["simulation"]["POLYMER_DENSITY"] = 0.5
        config["simulation"]["N_POLYMERS"] = 1
        config["geometry"]["AGRID"] = 0.5

        config["geometry"]["MICROCONTRACTION"] = True

        config["physical"]["N_MONOMERS"] = 50

        config["simulation"]["TIME_STEP"] = 0.001/4
        # config["simulation"]["LB_TIME_STEP"] = config["simulation"]["TIME_STEP"]
        # config["simulation"]["STEPS"] = int(1000*4)

        add_timeloops_from_gridsize(config)
        add_dependent_parameters(config)
        add_polymer_positions(config)
        #config["simulation"]["N_POLYMERS"] = 0
        config["geometry"]["MICROCONTRACTION"] = False

    if config["simulation"]["N_POLYMERS"] > 0:
        lbf, forces, positions, drag_forces, tensile_forces, tensile_vec_forces, added_bonds_per_step = run_simulation(config)
        save_results(config, lbf, forces, drag_forces, tensile_forces, tensile_vec_forces, added_bonds_per_step, positions)
    
    if config["simulation"]["N_POLYMERS"] == 0:
        lbf = run_simulation(config)
        save_velocity_field(config,lbf)

