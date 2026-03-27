# Simulation Runner

This repository contains a simulation workflow where `src/runner.py` configures and launches simulations implemented in `src/simulation.py`.

## Structure
- `src/runner.py` — main entry point
- `src/simulation.py` — core simulation code
- `src/utils.py` — helper functions (saving, etc.)
- `configs/default.json` — base configuration
- `scripts/` — plotting and analysis scripts

## Usage
Run simulations with:
```bash
srun -n 8 python src/runner.py <run_type> <arguments>
```

## Run types

```bash
# test_gridsizes
python src/runner.py test_gridsizes <shear> <gridsize> <k_vwf> <save_dir>
python src/runner.py test_gridsizes <shear> <gridsize> <run> <k_vwf> <save_dir>

# validation_uncoiling
python src/runner.py validation_uncoiling <shear> <gamma> <LJ_strength> <save_dir>
python src/runner.py validation_uncoiling <shear> <gamma> <LJ_strength> <N_monomers> <run> <save_dir>

# validation_uncoiling_free
python src/runner.py validation_uncoiling_free <shear> <gamma> <LJ_strength> <run> <save_dir>

# validation_coiling
python src/runner.py validation_coiling <LJ_strength> <N_monomers> <run> <save_dir>
python src/runner.py validation_coiling <gamma> <LJ_strength> <stdev_init> <N_monomers> <save_dir>

# check_forces
python src/runner.py check_forces <shear> <k> <save_dir>

# experiment
python src/runner.py experiment <shear> <true|false> <save_dir>

# check_velocity_field
python src/runner.py check_velocity_field <shear> <save_dir>

# experiment_density
python src/runner.py experiment_density <shear> <N_POLYMERS> <true|false> <save_dir>

# experiment_sphere
python src/runner.py experiment_sphere <shear> <true|false> <save_dir>

# experiment_microcontraction
python src/runner.py experiment_microcontraction <shear> <true|false> <save_dir>
```

## Plotting / Analysis

Plotting scripts are located in the `scripts/` folder. Examples:

```bash
python scripts/plot_experiment_sphere.py
python scripts/plot_density_experiment.py
python scripts/plot_validation.py
```

These scripts read simulation outputs and generate figures for analysis.

## Example

```bash
srun -n 8 python src/runner.py experiment_sphere 2000 true results
```

## Notes
- Default parameters are defined in `configs/default.json`
- Parameters are modified internally depending on `run_type`
- Output directories are created automatically
- Plot scripts are separate and should be run after simulations
