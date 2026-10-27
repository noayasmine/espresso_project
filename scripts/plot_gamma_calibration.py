import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# settings
# ------------------------------------------------
# gamma sweep of the tethered N=100 calibration (Fig. 5 setup) at 1280 dyn/cm2 = 128,000 1/s in 1 cP buffer,
# one plot per monomer mass and LJ strength, plus a summary of tether force vs gamma for all masses.
# Two folder layouts are read:
#   <base>/LJ_*/gamma_*/shear_*/run_*        gamma from the folder name (resolution 1e-4)
#   <base>/g_<gamma>/LJ_*/gamma_*/shear_*/run_*   gamma read exactly from g_<gamma> (for gamma < 1e-3)
experiments = "/home/nroebersen/espresso_project/experiments"
mass_sets = {  # monomer mass: (folders, largest gamma to plot or None)
    1.0: ([os.path.join(experiments, "gamma_calibration"), os.path.join(experiments, "gamma_low"),
           os.path.join(experiments, "gamma_eps")], 0.0003),
    0.01: ([os.path.join(experiments, "gamma_mass", "m_0.01")], None),
    0.001: ([os.path.join(experiments, "gamma_mass", "m_0.001")], None),
}
out_dir = os.path.join(experiments, "gamma_calibration")
target_shear = 128000
avg_window = 20       # trailing moving average over saved points
time_avg_steps = 100  # average over the last saved points
ref_per_mono, sigma_per_mono = 0.5, 0.1  # Fu et al. [pN per monomer]
unstable_drag = 100.0  # [pN] runs with a drag spike above this are reported and left out of the average


def moving_average_trailing_matrix(x, window):
    out = np.empty_like(x, dtype=float)
    for i in range(x.shape[0]):
        start = max(0, i - window + 1)
        out[i] = x[start:i + 1].mean(axis=0)
    return out


def tensile_profiles(shear_dir):
    """Cumulative drag force from the free end, one row per stable run [pN]."""
    runs = sorted(d for d in os.listdir(shear_dir) if re.match(r"^run_\d+$", d))
    profiles = []
    for r in runs:
        path = os.path.join(shear_dir, r, "drag_forces.npz")
        if not os.path.exists(path):
            print(f"  missing: {path}")
            continue
        Fmag = np.linalg.norm(np.load(path)["forces"], axis=2) * 1000.0
        per_run = moving_average_trailing_matrix(Fmag, avg_window)[-time_avg_steps:].mean(axis=0)
        cum = np.cumsum(per_run[::-1])[::-1]
        if Fmag.max() > unstable_drag:
            print(f"  unstable: {r} (max |F_drag| = {Fmag.max():.0f} pN, tether force {cum[0]:.1f} pN)")
            continue
        profiles.append(cum)
    return np.vstack(profiles) if profiles else None


def subdirs(path, pattern):
    return [d for d in os.listdir(path) if re.match(pattern, d)] if os.path.isdir(path) else []


def collect_cases(base_dirs):
    """{LJ folder name: {gamma: shear_dir}} from both folder layouts."""
    cases = {}
    for base in base_dirs:
        roots = [(None, base)] + [(float(g[2:]), os.path.join(base, g)) for g in subdirs(base, r"^g_[0-9.eE+-]+$")]
        for gamma_exact, root in roots:
            for lj in subdirs(root, r"^LJ_\d+$"):
                for gdir in subdirs(os.path.join(root, lj), r"^gamma_\d+$"):
                    shear_dir = os.path.join(root, lj, gdir, f"shear_{target_shear}")
                    if os.path.isdir(shear_dir):
                        gamma = gamma_exact if gamma_exact is not None else int(gdir.split("_")[1]) / 1e4
                        cases.setdefault(lj, {})[gamma] = shear_dir
    return cases


summary = {}  # (LJ, mass): [(gamma, tether mean, tether std, t2)]
for mass, (base_dirs, gamma_max) in mass_sets.items():
    for lj, by_gamma in sorted(collect_cases(base_dirs).items()):
        print(f"===== m = {mass}, {lj} =====")
        gammas = sorted(g for g in by_gamma if gamma_max is None or g <= gamma_max)
        colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, max(len(gammas), 1)))

        plt.figure(figsize=(6, 4))
        N = None
        for gamma, color in zip(gammas, colors):
            print(f"gamma = {gamma:.6g}")
            cum = tensile_profiles(by_gamma[gamma])
            if cum is None:
                print("  no stable runs")
                continue
            N = cum.shape[1]
            steps = N - np.arange(N)
            ref, sigma_ref = ref_per_mono * steps, sigma_per_mono * steps
            mu = cum.mean(axis=0)
            sig = cum.std(axis=0, ddof=1) if cum.shape[0] > 1 else np.zeros(N)
            t2 = np.mean(((mu - ref) / (sig + sigma_ref)) ** 2)
            print(f"  stable runs={cum.shape[0]}, tether force={mu[0]:.1f} +- {sig[0]:.1f} pN (ref {ref[0]:.0f}), "
                  f"drag per monomer={mu[0] / N:.3f} pN, t2={t2:.3f}")
            summary.setdefault((lj, mass), []).append((gamma, mu[0], sig[0], t2, N))
            plt.errorbar(np.arange(N), mu, yerr=sig, color=color, lw=1.2, elinewidth=0.5, capsize=2,
                         label=fr"$\gamma$={gamma:.6g} ($t^2_{{{N}}}$={t2:.3f})")

        if N is None:
            print("  no results yet, no plot")
            plt.close()
            continue
        plt.plot(np.arange(N), ref, "--", color="grey", label=r"Fu et al. $0.5 \pm 0.1$ pN/mono")
        plt.fill_between(np.arange(N), ref - sigma_ref, ref + sigma_ref, color="grey", alpha=0.2)
        plt.xlabel("Monomer index")
        plt.ylabel("Tensile force [pN]")
        plt.title(fr"monomer mass {mass}, $\epsilon_{{LJ}}$ = {int(lj.split('_')[1]) / 100:g} $k_BT$", fontsize=9)
        plt.legend(fontsize=7)
        plt.tight_layout()
        out = os.path.join(out_dir, f"gamma_calibration_{lj}_m{mass}_shear_{target_shear}.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved: {out}")

# ------------------------------------------------
# summary: tether force vs gamma for all masses
# ------------------------------------------------
for lj in sorted({lj for lj, _ in summary}):
    plt.figure(figsize=(5, 4))
    for mass in mass_sets:
        rows = summary.get((lj, mass))
        if not rows:
            continue
        g, f, s, _, n_mono = np.array(rows).T
        plt.errorbar(g, f, yerr=s, marker="o", ms=4, capsize=2, label=f"m = {mass}")
    n_mono = n_mono[0]
    plt.axhline(ref_per_mono * n_mono, color="grey", ls="--", label="Fu et al.")
    plt.axhspan((ref_per_mono - sigma_per_mono) * n_mono, (ref_per_mono + sigma_per_mono) * n_mono, color="grey", alpha=0.2)
    plt.xscale("log")
    plt.xlabel(r"$\gamma$ [MD]")
    plt.ylabel("Tensile force at tether [pN]")
    plt.legend(fontsize=7)
    plt.tight_layout()
    out = os.path.join(out_dir, f"gamma_calibration_{lj}_mass_summary_shear_{target_shear}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")
