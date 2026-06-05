"""Mechanical contact solve based on Tamaas."""

from __future__ import annotations

import numpy as np

from . import config


def _require_tamaas():
    try:
        import tamaas as tm
    except ImportError as exc:
        raise ImportError(
            "Tamaas is required for the mechanical contact solve. Install Tamaas "
            "before running full simulations."
        ) from exc
    return tm


def solve_mechanical_step(surface_profile: np.ndarray, target_pressure: float):
    """Solve one normal-contact problem for a target mean pressure.

    Parameters
    ----------
    surface_profile:
        Effective surface height map, i.e. geometric height minus the current
        thermal normal displacement.
    target_pressure:
        Target spatially averaged normal pressure in Pa.

    Returns
    -------
    tamaas.Model
        A Tamaas model containing the converged traction field.
    """
    config.validate()
    tm = _require_tamaas()

    surface_profile = np.asarray(surface_profile, dtype=float)
    expected_shape = (config.n_x, config.n_y)
    if surface_profile.shape != expected_shape:
        raise ValueError(f"surface_profile has shape {surface_profile.shape}, expected {expected_shape}.")
    if target_pressure < 0:
        raise ValueError("target_pressure must be non-negative.")
    if not np.all(np.isfinite(surface_profile)):
        raise ValueError("surface_profile contains non-finite values.")

    effective_modulus = config.E_elastic / (1.0 - config.nu_elastic**2)

    model = tm.Model(tm.model_type.basic_2d, [config.L_x, config.L_y], [config.n_x, config.n_y])
    model.E = effective_modulus
    model.nu = 0.0

    solver = tm.PolonskyKeerRey(model, surface_profile, tolerance=config.solver_tolerance)
    solver.max_iter = config.solver_max_iter

    print(f"  > (M) Solving target mean pressure: {target_pressure / 1e6:.3f} MPa")
    err = solver.solve(target_pressure)

    if not np.isfinite(err):
        raise RuntimeError("Mechanical solver returned a non-finite residual.")
    if err > solver.tolerance:
        raise RuntimeError(
            f"Mechanical solver residual {err:.3e} exceeds tolerance {solver.tolerance:.3e}."
        )

    print(f"    [OK] Mechanical solve converged with residual {err:.3e}")
    return model
