# Nankai Trough Earthquake Simulation - Developer Guide

This document provides instructions for developers and AI agents working on this project.

## Project Overview

This project simulates earthquake cycles along the Nankai Trough using a rate-and-state friction model. It is based on Hirose et al. (2022).

### Key Components

- **`simulation_realistic.py`**: The main entry point for running the realistic simulation.
- **`src/simulation/realistic.py`**: Contains the `RealisticSimulation` class, which encapsulates the simulation logic.
- **`src/physics/`**: Contains physical models (StressKernel, RateStateFriction, QuasiDynamicEquations).
- **`src/geometry/`**: Contains geometry utilities (CoordinateSystem, Mesh, optimized `estimate_normals`).
- **`plate_geometry.py`**: Defines the specific geometry of the Nankai Trough (segments, asperities).

## Running the Simulation

To run the realistic simulation:

```bash
python simulation_realistic.py --years 500
```

Options:
- `--years N`: Simulation duration in years.
- `--nx N`: Number of cells along strike (e.g., 60).
- `--ny N`: Number of cells down dip (e.g., 8).
- `--polygon-data FILE`: Use polygon data for mesh generation.

## Recent Optimizations

1.  **Parallel Stress Kernel**: The `StressKernel` calculation is optimized using Numba's `parallel=True` directive. This significantly speeds up the $O(N^2)$ interaction calculation.
2.  **KDTree for Geometry**: Normal vector estimation uses `scipy.spatial.KDTree` for $O(N \log N)$ performance instead of $O(N^2)$.
3.  **Refactoring**: The simulation logic is now modularized in `src.simulation.realistic.RealisticSimulation`.

## Development Guidelines

- **Performance**: Always use Numba (`@njit`) for computationally intensive loops. Avoid Python loops in the critical path (derivative function).
- **Modularity**: Keep physics, geometry, and simulation logic separate.
- **Visualization**: Visualization code should be decoupled from the core simulation loop where possible (use callbacks).

## Troubleshooting

- If `scipy` is missing, install it: `pip install scipy`.
- If Numba compilation is slow, be patient on the first run. `cache=True` is enabled to speed up subsequent runs.
