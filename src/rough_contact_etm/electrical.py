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




def _contact_neighbors_8(i: int, j: int, nx: int, ny: int):
    """Yield 8-connected in-domain neighbours of a contact node."""
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ii = i + di
            jj = j + dj
            if 0 <= ii < nx and 0 <= jj < ny:
                yield ii, jj


def build_contact_conductance_map(pressure_map: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build the local interfacial conductance map ``k_E(x, y)``.

    The default model follows the rough-surface validation setting in Li et al.
    and assigns one conductance value to every connected contact spot,

        k_E,j = 1 / (rho_film * l_film + rho_elastic * pi * r_j / 2),

    where ``r_j = sqrt(A_j / pi)`` is the equivalent circular radius of the
    connected contact spot.  If ``config.include_constriction_resistance`` is
    false, the model reduces to the thin-film-only conductance
    ``1 / (rho_film * l_film)``.
    """
    pressure_map = _validate_pressure_map(pressure_map)
    nx, ny = pressure_map.shape
    dx = config.L_x / nx
    dy = config.L_y / ny

    contact_mask = pressure_map > 0.0
    k_map = np.zeros_like(pressure_map, dtype=float)
    film_specific_resistance = config.rho_film * config.l_film

    if not np.any(contact_mask):
        return k_map, {
            "electrical_contact_clusters": 0,
            "electrical_min_spot_radius_m": 0.0,
            "electrical_max_spot_radius_m": 0.0,
            "electrical_mean_spot_radius_m": 0.0,
            "electrical_min_kE_S_per_m2": 0.0,
            "electrical_max_kE_S_per_m2": 0.0,
        }

    if not bool(getattr(config, "include_constriction_resistance", False)):
        k_value = 1.0 / film_specific_resistance
        k_map[contact_mask] = k_value
        return k_map, {
            "electrical_contact_clusters": 1,
            "electrical_min_spot_radius_m": 0.0,
            "electrical_max_spot_radius_m": 0.0,
            "electrical_mean_spot_radius_m": 0.0,
            "electrical_min_kE_S_per_m2": float(k_value),
            "electrical_max_kE_S_per_m2": float(k_value),
        }

    visited = np.zeros_like(contact_mask, dtype=bool)
    radii: list[float] = []
    conductances: list[float] = []

    contact_indices = np.argwhere(contact_mask)
    for i0, j0 in contact_indices:
        i0 = int(i0)
        j0 = int(j0)
        if visited[i0, j0]:
            continue

        stack = [(i0, j0)]
        visited[i0, j0] = True
        nodes: list[tuple[int, int]] = []

        while stack:
            i, j = stack.pop()
            nodes.append((i, j))
            for ii, jj in _contact_neighbors_8(i, j, nx, ny):
                if contact_mask[ii, jj] and not visited[ii, jj]:
                    visited[ii, jj] = True
                    stack.append((ii, jj))

        spot_area = len(nodes) * dx * dy
        # Equivalent radius used in Li et al.'s film-plus-constriction model.
        r_eq = float(np.sqrt(max(spot_area, 0.0) / np.pi))
        specific_resistance = film_specific_resistance + config.rho_elastic * np.pi * r_eq / 2.0
        k_value = 1.0 / specific_resistance

        for i, j in nodes:
            k_map[i, j] = k_value

        radii.append(r_eq)
        conductances.append(float(k_value))

    radii_arr = np.asarray(radii, dtype=float)
    k_arr = np.asarray(conductances, dtype=float)
    return k_map, {
        "electrical_contact_clusters": int(len(radii)),
        "electrical_min_spot_radius_m": float(np.min(radii_arr)),
        "electrical_max_spot_radius_m": float(np.max(radii_arr)),
        "electrical_mean_spot_radius_m": float(np.mean(radii_arr)),
        "electrical_min_kE_S_per_m2": float(np.min(k_arr)),
        "electrical_max_kE_S_per_m2": float(np.max(k_arr)),
    }

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
    k_E, k_stats = build_contact_conductance_map(pressure_map)
    relaxation = config.electrical_relaxation

    if verbose:
        print("\n--- Electrical field solve ---")
        if bool(getattr(config, "include_constriction_resistance", False)):
            print("  > Interfacial conductance model: film + constriction")
            print(f"  > Contact spots for k_E: {k_stats['electrical_contact_clusters']}")
            print(
                "  > Equivalent spot radius range: "
                f"{k_stats['electrical_min_spot_radius_m'] * 1e6:.3g}--"
                f"{k_stats['electrical_max_spot_radius_m'] * 1e6:.3g} um"
            )
            print(
                "  > Interfacial conductance k_E range: "
                f"{k_stats['electrical_min_kE_S_per_m2']:.3e}--"
                f"{k_stats['electrical_max_kE_S_per_m2']:.3e} S/m^2"
            )
        else:
            print(f"  > Interfacial film conductance k_E: {k_stats['electrical_max_kE_S_per_m2']:.3e} S/m^2")

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
