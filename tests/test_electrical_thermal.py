from __future__ import annotations

import numpy as np

from rough_contact_etm import config
from rough_contact_etm.electrical import run_electrical_analysis
from rough_contact_etm.thermal import solve_thermal_field


def _set_small_grid(monkeypatch):
    monkeypatch.setattr(config, "n_x", 16)
    monkeypatch.setattr(config, "n_y", 16)
    monkeypatch.setattr(config, "L_x", 1e-3)
    monkeypatch.setattr(config, "L_y", 1e-3)
    monkeypatch.setattr(config, "electrical_max_iter", 20)
    monkeypatch.setattr(config, "electrical_tolerance", 1e-6)


def test_no_contact_returns_zero_current(monkeypatch):
    _set_small_grid(monkeypatch)
    pressure = np.zeros((config.n_x, config.n_y))
    V, J, Q, total_current, ecr = run_electrical_analysis(pressure, verbose=False)
    assert V.shape == pressure.shape
    assert np.allclose(J, 0.0)
    assert np.allclose(Q, 0.0)
    assert total_current == 0.0
    assert np.isinf(ecr)


def test_thermal_solver_returns_finite_fields(monkeypatch):
    _set_small_grid(monkeypatch)
    heat_flux = np.zeros((config.n_x, config.n_y))
    heat_flux[4:8, 4:8] = 1.0e5
    temperature, displacement = solve_thermal_field(heat_flux)
    assert temperature.shape == heat_flux.shape
    assert displacement.shape == heat_flux.shape
    assert np.all(np.isfinite(temperature))
    assert np.all(np.isfinite(displacement))
