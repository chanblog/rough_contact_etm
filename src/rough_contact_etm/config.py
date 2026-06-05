"""Default configuration for rough-contact-etm.

The values in this module define a moderately sized reference simulation. For
quick local smoke tests, use ``examples/minimal_run.py`` or override these values
before importing ``rough_contact_etm.run_simulation``.
"""

from __future__ import annotations

import os

# -----------------------------------------------------------------------------
# Geometry and material parameters
# -----------------------------------------------------------------------------
L_x = 10e-3  # domain length in x [m]
L_y = 10e-3  # domain length in y [m]

# Rigid rough surface
h_rms_rigid = 1.0e-6  # root-mean-square height [m]
seed_rigid = 2024
hurst = 0.8

# Elastic half-space / finite layer parameters
E_elastic = 80e9  # Young's modulus [Pa]
nu_elastic = 0.3  # Poisson's ratio [-]
thermal_expansion = 1.77e-5  # thermal expansion coefficient [1/K]
thermal_conductivity = 400.0  # thermal conductivity [W/(m K)]
rho_elastic = 1.75e-8  # electrical resistivity [Ohm m]

# Interfacial film model
rho_film = 1.75e-2  # film resistivity [Ohm m]
l_film = 10e-9  # effective film thickness [m]

# Layer height used by the electrical and thermal Green's functions
h0 = 2e-3  # [m]

# -----------------------------------------------------------------------------
# Simulation controls
# -----------------------------------------------------------------------------
F_min = 50.0  # minimum normal load [N]
F_max = 3000.0  # maximum normal load [N]
n_x = 512
n_y = 512
N_LOAD_STEPS = 20
Delta_V_total = 0.1  # applied voltage drop [V]

# Mechanical solver controls
solver_tolerance = 1e-8
solver_max_iter = 20_000

# Electrical fixed-point solver controls
electrical_tolerance = 1e-8
electrical_max_iter = 1000
electrical_relaxation = 0.5

# Thermo-electro-mechanical fixed-point controls
coupling_max_iters = 200
coupling_tolerance = 1e-4
coupling_relaxation = 0.1
coupling_memory_depth = 15
evaluate_final_state_on_failure = True

# Joule heat partition coefficient. eta=0.5 assigns half of the interfacial
# heat generation to the modeled body.
heat_partition = 0.5

# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------
output_dir = "rigid_on_elastic_contact_results"
basename = os.path.join(output_dir, "contact_solution")
equiv_surf_path = os.path.join(output_dir, "equivalent_surface.npy")
rigid_surf_path = os.path.join(output_dir, "rigid_surface.npy")
hdf5_dump_dir = os.path.join("hdf5", output_dir)


def validate() -> None:
    """Validate the most important scalar configuration values."""
    if L_x <= 0 or L_y <= 0:
        raise ValueError("Domain lengths L_x and L_y must be positive.")
    if n_x <= 0 or n_y <= 0:
        raise ValueError("Grid sizes n_x and n_y must be positive.")
    if E_elastic <= 0:
        raise ValueError("E_elastic must be positive.")
    if not (0.0 <= nu_elastic < 0.5):
        raise ValueError("nu_elastic must satisfy 0 <= nu < 0.5.")
    if rho_film <= 0 or l_film <= 0:
        raise ValueError("rho_film and l_film must be positive.")
    if h0 <= 0:
        raise ValueError("h0 must be positive.")
    if not (0.0 < electrical_relaxation <= 1.0):
        raise ValueError("electrical_relaxation must be in (0, 1].")
    if not (0.0 < coupling_relaxation <= 1.0):
        raise ValueError("coupling_relaxation must be in (0, 1].")
    if not (0.0 <= heat_partition <= 1.0):
        raise ValueError("heat_partition must be between 0 and 1.")
