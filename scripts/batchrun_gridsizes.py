import subprocess
import sys

def main():
    shears = [1000, 5000]
    gridsizes = [1,0.5,0.25]

    for gridsize in gridsizes:
        for shear in shears:
        # print(f"Launching shear={shear}, gamma={gamma}")
            subprocess.run([
                sys.executable,
                "src/runner.py",
                str(shear),
                str(gridsize)
                ])

if __name__ == "__main__":
    main()