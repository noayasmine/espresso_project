import subprocess
import sys

def main():
    shears = [1000, 5000]
    # shears = [1000,2000,3000,4000,5000,6000,7000,8000,9000]
    gammas = [0.005]
    LJ_strengths = [1]
    n_polymers = 1
    
    for LJ_strength in LJ_strengths:
        for gamma in gammas:
            for shear in shears:
                print(f"Launching shear={shear}, gamma={gamma}")
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
