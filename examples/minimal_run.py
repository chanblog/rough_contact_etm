"""Small local smoke-test configuration.

Run from the repository root with:

    python examples/minimal_run.py

This script intentionally uses a small grid and a short load sweep so that you
can check the installation and file-writing workflow before launching the full
reference simulation.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rough_contact_etm import config

# Lightweight overrides for a local smoke test.
config.n_x = 128
config.n_y = 128
config.N_LOAD_STEPS = 3
config.F_min = 50.0
config.F_max = 300.0
config.solver_tolerance = 1e-7
config.solver_max_iter = 5000
config.electrical_tolerance = 1e-7
config.electrical_max_iter = 300
config.coupling_tolerance = 5e-4
config.coupling_max_iters = 40
config.coupling_relaxation = 0.1
config.output_dir = "minimal_contact_results"
config.basename = f"{config.output_dir}/contact_solution"
config.equiv_surf_path = f"{config.output_dir}/equivalent_surface.npy"
config.rigid_surf_path = f"{config.output_dir}/rigid_surface.npy"
config.hdf5_dump_dir = f"hdf5/{config.output_dir}"

from rough_contact_etm.run_simulation import main


if __name__ == "__main__":
    main()
