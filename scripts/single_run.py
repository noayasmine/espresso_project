import subprocess
import sys

def main():
    shear = 7000
    gamma = 0.005
    LJ_strength = 1
    n_polymers = 1

    subprocess.run([
        sys.executable,
        "src/runner.py",
        str(shear),
        str(n_polymers),
        str(gamma),
        str(LJ_strength)
        ])

if __name__ == "__main__":
    main()