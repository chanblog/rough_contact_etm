"""Thermo-electro-mechanical fixed-point coupling algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .electrical import run_electrical_analysis
from .mechanical import solve_mechanical_step
from .thermal import solve_thermal_field


@dataclass
class IterationResult:
    x: np.ndarray
    converged: bool
    history: list[float]


class AndersonAcceleration:
    """Memory-limited Anderson acceleration for flattened fixed-point residuals."""

    def __init__(self, dimension: int, memory: int = 5, beta: float = 1.0, tolerance: float = 1e-6):
        self.dimension = dimension
        self.m = max(1, int(memory))
        self.beta = float(beta)
        self.tolerance = float(tolerance)
        self.F_history: list[np.ndarray] = []
        self.X_history: list[np.ndarray] = []

    def solve(self, x0: np.ndarray, residual_func, max_iter: int = 100) -> IterationResult:
        x = np.asarray(x0, dtype=float).copy()
        history: list[float] = []

        print(f"  > (Anderson) dimension={self.dimension}, memory={self.m}")
        for k in range(max_iter):
            F_current = np.asarray(residual_func(x), dtype=float)
            if F_current.shape != x.shape:
                raise ValueError("residual_func returned an array with an incompatible shape.")
            if not np.all(np.isfinite(F_current)):
                raise RuntimeError("Anderson residual contains non-finite values.")

            F_norm = float(np.linalg.norm(F_current))
            history.append(F_norm)

            if (k + 1) % 10 == 0:
                print(f"    Anderson iter {k + 1}, residual norm: {F_norm:.3e}")
            if F_norm < self.tolerance:
                print(f"  > Anderson converged in {k + 1} iterations. Residual: {F_norm:.3e}")
                return IterationResult(x=x, converged=True, history=history)

            x = self._update(x, F_current)

            while len(self.F_history) > self.m + 1:
                self.F_history.pop(0)
                self.X_history.pop(0)

        print(f"  > WARNING: Anderson did not converge. Final residual: {history[-1]:.3e}")
        return IterationResult(x=x, converged=False, history=history)

    def _update(self, x_current: np.ndarray, F_current: np.ndarray) -> np.ndarray:
        self.F_history.append(F_current.copy())
        self.X_history.append(x_current.copy())

        m_effective = min(len(self.F_history) - 1, self.m)
        if m_effective <= 0:
            return x_current + self.beta * F_current

        start = len(self.F_history) - m_effective - 1
        dF_cols = []
        dX_cols = []
        for j in range(start, len(self.F_history) - 1):
            dF_cols.append(self.F_history[j + 1] - self.F_history[j])
            dX_cols.append(self.X_history[j + 1] - self.X_history[j])

        dF = np.column_stack(dF_cols)
        dX = np.column_stack(dX_cols)

        try:
            alpha, *_ = np.linalg.lstsq(dF, F_current, rcond=None)
            return x_current + self.beta * F_current - (dX + self.beta * dF) @ alpha
        except np.linalg.LinAlgError:
            print("    Anderson least-squares update failed; falling back to relaxed fixed-point step.")
            return x_current + self.beta * F_current


class AdaptiveRelaxationSolver:
    """Simple fixed-point iteration with adaptive relaxation."""

    def __init__(self, tolerance: float = 1e-6, min_relax: float = 0.05, max_relax: float = 0.8):
        self.tolerance = float(tolerance)
        self.min_relax = float(min_relax)
        self.max_relax = float(max_relax)
        self.relaxation = 0.3

    def solve(self, x0: np.ndarray, residual_func, max_iter: int = 200) -> IterationResult:
        x = np.asarray(x0, dtype=float).copy()
        history: list[float] = []
        error_prev: float | None = None

        print(f"  > (Adaptive relaxation) initial relaxation={self.relaxation:.3f}")
        for k in range(max_iter):
            F_current = np.asarray(residual_func(x), dtype=float)
            if F_current.shape != x.shape:
                raise ValueError("residual_func returned an array with an incompatible shape.")
            if not np.all(np.isfinite(F_current)):
                raise RuntimeError("Adaptive residual contains non-finite values.")

            x_candidate = x + F_current
            x_new = (1.0 - self.relaxation) * x + self.relaxation * x_candidate
            error = float(np.linalg.norm(x_new - x) / max(np.linalg.norm(x_new), 1e-30))
            history.append(error)

            if error_prev is not None:
                if error < 0.9 * error_prev:
                    self.relaxation = min(self.relaxation * 1.1, self.max_relax)
                elif error > 1.1 * error_prev:
                    self.relaxation = max(self.relaxation * 0.8, self.min_relax)
            error_prev = error

            if (k + 1) % 20 == 0:
                print(f"    adaptive iter {k + 1}, error={error:.3e}, relaxation={self.relaxation:.3f}")
            if error < self.tolerance:
                print(f"  > Adaptive relaxation converged in {k + 1} iterations. Error: {error:.3e}")
                return IterationResult(x=x_new, converged=True, history=history)

            x = x_new

        print(f"  > WARNING: adaptive relaxation did not converge. Final error: {history[-1]:.3e}")
        return IterationResult(x=x, converged=False, history=history)


def high_precision_coupling_iteration(
    s_geom: np.ndarray,
    target_pressure: float,
    method: str = "adaptive",
    tolerance: float = 1e-6,
):
    """Run one coupled thermo-electro-mechanical solve for a given load step."""
    config.validate()
    s_geom = np.asarray(s_geom, dtype=float)
    expected_shape = (config.n_x, config.n_y)
    if s_geom.shape != expected_shape:
        raise ValueError(f"s_geom has shape {s_geom.shape}, expected {expected_shape}.")

    print(f"\n--- Coupled E-T-M iteration ({method}) ---")
    u_initial = np.zeros_like(s_geom)
    n_dof = s_geom.size

    def residual_function(u_thermal_flat: np.ndarray) -> np.ndarray:
        u_thermal = u_thermal_flat.reshape(s_geom.shape)
        s_effective = s_geom - u_thermal
        model = solve_mechanical_step(s_effective, target_pressure)
        pressure_map = np.asarray(model.traction, dtype=float)
        _, _, Q_s, _, _ = run_electrical_analysis(pressure_map)
        _, u_thermal_new = solve_thermal_field(Q_s)
        return (u_thermal_new - u_thermal).ravel()

    if method == "anderson":
        solver = AndersonAcceleration(
            n_dof,
            memory=config.coupling_memory_depth,
            beta=config.coupling_relaxation,
            tolerance=tolerance,
        )
        result = solver.solve(u_initial.ravel(), residual_function, max_iter=config.coupling_max_iters)
    elif method == "adaptive":
        solver = AdaptiveRelaxationSolver(tolerance=tolerance)
        solver.relaxation = config.coupling_relaxation
        result = solver.solve(u_initial.ravel(), residual_function, max_iter=config.coupling_max_iters)
    else:
        raise ValueError("method must be either 'adaptive' or 'anderson'.")

    u_thermal_final = result.x.reshape(s_geom.shape)
    final_state = {"converged": result.converged, "thermal_displacement": u_thermal_final}

    if not result.converged and not config.evaluate_final_state_on_failure:
        print("  > Coupled iteration did not converge; final state evaluation was skipped.")
        return final_state, result.history

    if not result.converged:
        print("  > Coupled iteration did not fully converge; evaluating the last iterate for diagnostics.")

    try:
        s_effective = s_geom - u_thermal_final
        model = solve_mechanical_step(s_effective, target_pressure)
        pressure_map = np.asarray(model.traction, dtype=float)
        V_z0, J_z0, Q_s, total_current, ECR = run_electrical_analysis(pressure_map)
        T_map, _ = solve_thermal_field(Q_s)

        final_state.update(
            {
                "model": model,
                "temperature": T_map,
                "heat_flux": Q_s,
                "voltage": V_z0,
                "current_density": J_z0,
                "total_current": total_current,
                "ECR": ECR,
            }
        )
        print("  > Coupled solve final state evaluated successfully.")
    except Exception as exc:
        print(f"  > ERROR: final state evaluation failed: {exc}")
        final_state["final_state_error"] = str(exc)

    return final_state, result.history
