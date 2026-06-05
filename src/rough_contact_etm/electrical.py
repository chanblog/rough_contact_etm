"""Electrical contact post-processing on a fixed contact mask."""

from __future__ import annotations

import numpy as np

from . import config


def create_q_grids(nx: int, ny: int, Lx: float, Ly: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create Fourier-space wave-number grids."""
    qx_vec = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    qy_vec = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    qx, qy = np.meshgrid(qx_vec, qy_vec, indexing="ij")
    q = np.sqrt(qx**2 + qy**2)
    return qx, qy, q


def calculate_surface_voltage_hat(J_hat: np.ndarray, q: np.ndarray, rho: float, h0: float) -> np.ndarray:
    """Return the Fourier coefficients of the surface potential at z=0.

    The kernel corresponds to a finite layer of height ``h0`` with a grounded
    boundary at the opposite face. The zero mode is treated by its analytical
    limit, ``rho * h0``.
    """
    kernel = np.empty_like(q, dtype=float)
    nonzero = q > 0.0
    kernel[nonzero] = (rho / q[nonzero]) * np.tanh(q[nonzero] * h0)
    kernel[~nonzero] = rho * h0
    return kernel * J_hat


def _validate_pressure_map(pressure_map: np.ndarray) -> np.ndarray:
    pressure_map = np.asarray(pressure_map, dtype=float)
    expected_shape = (config.n_x, config.n_y)
    if pressure_map.shape != expected_shape:
        raise ValueError(f"pressure_map has shape {pressure_map.shape}, expected {expected_shape}.")
    if not np.all(np.isfinite(pressure_map)):
        raise ValueError("pressure_map contains non-finite values.")
    return pressure_map


def run_electrical_analysis(pressure_map: np.ndarray, verbose: bool = True):
    """Solve the electrical fixed-point problem on the real contact mask.

    Parameters
    ----------
    pressure_map:
        Contact pressure field. Nodes with positive pressure are treated as
        electrically active contact nodes.
    verbose:
        Print iteration information when ``True``.

    Returns
    -------
    tuple
        ``(V_z0, J_z0, Q_s, total_current, ECR)`` where ``Q_s`` is the heat flux
        assigned to the modeled solid.
    """
    config.validate()
    pressure_map = _validate_pressure_map(pressure_map)

    nx, ny = config.n_x, config.n_y
    Lx, Ly = config.L_x, config.L_y
    dx, dy = Lx / nx, Ly / ny

    delta_v = config.Delta_V_total
    rho = config.rho_elastic
    h0 = config.h0
    k_E = 1.0 / (config.rho_film * config.l_film)
    relaxation = config.electrical_relaxation

    if verbose:
        print("\n--- Electrical field solve ---")
        print(f"  > Interfacial film conductance k_E: {k_E:.3e} S/m^2")

    contact_mask = pressure_map > 0.0
    if not np.any(contact_mask):
        if verbose:
            print("  > No contact nodes found. Returning zero current and infinite resistance.")
        zeros = np.zeros((nx, ny), dtype=float)
        return zeros, zeros.copy(), zeros.copy(), 0.0, np.inf

    _, _, q = create_q_grids(nx, ny, Lx, Ly)
    V_z0 = np.zeros((nx, ny), dtype=float)
    V_z_h0 = 0.0

    converged = False
    for iteration in range(config.electrical_max_iter):
        V_old = V_z0.copy()
        delta_v_f = delta_v - (V_old - V_z_h0)
        J_z0 = k_E * delta_v_f
        J_z0[~contact_mask] = 0.0

        J_hat = np.fft.fft2(J_z0)
        V_candidate = np.fft.ifft2(calculate_surface_voltage_hat(J_hat, q, rho, h0)).real
        V_z0 = (1.0 - relaxation) * V_old + relaxation * V_candidate

        update_norm = np.linalg.norm((V_z0 - V_old)[contact_mask])
        reference_norm = max(np.linalg.norm(V_z0[contact_mask]), abs(delta_v) * np.sqrt(np.count_nonzero(contact_mask)), 1e-30)
        error = update_norm / reference_norm

        if not np.isfinite(error):
            raise RuntimeError("Electrical fixed-point iteration produced a non-finite residual.")

        if verbose and (iteration + 1) % 50 == 0:
            print(f"    iter {iteration + 1}, relative update: {error:.3e}")

        if error < config.electrical_tolerance and iteration > 1:
            converged = True
            if verbose:
                print(f"  > Electrical solve converged in {iteration + 1} iterations. Residual: {error:.3e}")
            break

    if not converged and verbose:
        print(
            f"  > WARNING: electrical solve reached {config.electrical_max_iter} iterations "
            f"without meeting tolerance {config.electrical_tolerance:.1e}."
        )

    delta_v_f = delta_v - (V_z0 - V_z_h0)
    J_z0 = k_E * delta_v_f
    J_z0[~contact_mask] = 0.0

    # Joule heating per unit area is J * DeltaV_f = k_E * DeltaV_f^2 on contact
    # nodes. Numerical roundoff can create tiny negative values; clip those only.
    Q_s = config.heat_partition * np.maximum(J_z0 * delta_v_f, 0.0)
    total_current = float(np.sum(J_z0) * dx * dy)
    ECR = delta_v / total_current if total_current > 0.0 else np.inf

    if verbose:
        print("--- Electrical field solve complete ---")
        print(f"  > Applied voltage: {delta_v:.4g} V")
        print(f"  > Total current:   {total_current:.6g} A")
        print(f"  > ECR:             {ECR * 1000.0:.6g} mOhm")

    return V_z0, J_z0, Q_s, total_current, ECR
