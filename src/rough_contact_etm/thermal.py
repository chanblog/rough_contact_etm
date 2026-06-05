"""Thermal Green's-function solve for surface heat flux."""

from __future__ import annotations

import numpy as np

from . import config
from .electrical import create_q_grids


def solve_thermal_field(Q_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute surface temperature and thermal normal displacement.

    Parameters
    ----------
    Q_s:
        Surface heat flux density assigned to the modeled solid, in W/m^2.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Surface temperature rise and thermally induced normal displacement.
    """
    config.validate()
    Q_s = np.asarray(Q_s, dtype=float)
    expected_shape = (config.n_x, config.n_y)
    if Q_s.shape != expected_shape:
        raise ValueError(f"Q_s has shape {Q_s.shape}, expected {expected_shape}.")
    if not np.all(np.isfinite(Q_s)):
        raise ValueError("Q_s contains non-finite values.")

    nx, ny = config.n_x, config.n_y
    Lx, Ly = config.L_x, config.L_y
    h0 = config.h0
    kappa = config.thermal_conductivity
    alpha = config.thermal_expansion
    nu = config.nu_elastic

    _, _, q = create_q_grids(nx, ny, Lx, Ly)
    Q_hat = np.fft.fft2(Q_s)

    kernel_T = np.empty_like(q, dtype=float)
    nonzero = q > 0.0
    kernel_T[nonzero] = np.tanh(q[nonzero] * h0) / (kappa * q[nonzero])
    kernel_T[~nonzero] = h0 / kappa
    T_hat = kernel_T * Q_hat

    kernel_U = np.empty_like(q, dtype=float)
    qh = q[nonzero] * h0
    exp_term = np.exp(-qh)
    ratio = (1.0 - exp_term) ** 2 / (1.0 + exp_term**2)
    kernel_U[nonzero] = (alpha * (1.0 + nu)) * ratio / (kappa * q[nonzero] ** 2)
    kernel_U[~nonzero] = alpha * (1.0 + nu) * h0**2 / (2.0 * kappa)
    u_hat = kernel_U * Q_hat

    T_map = np.fft.ifft2(T_hat).real
    u_thermal = np.fft.ifft2(u_hat).real
    return T_map, u_thermal
