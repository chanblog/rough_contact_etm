"""Random rough-surface generation utilities."""

from __future__ import annotations

import os

import numpy as np

from . import config


def _require_tamaas():
    try:
        import tamaas as tm
    except ImportError as exc:
        raise ImportError(
            "Tamaas is required for surface generation. Install Tamaas before "
            "running mechanical simulations. See the README for installation notes."
        ) from exc
    return tm


def generate_surface(shape: tuple[int, int] | list[int], h_rms: float, seed: int, hurst: float) -> np.ndarray:
    """Generate a periodic random-phase rough surface with a power-law spectrum.

    Parameters
    ----------
    shape:
        Number of grid points in ``x`` and ``y``.
    h_rms:
        Target root-mean-square height in metres.
    seed:
        Random seed used by Tamaas.
    hurst:
        Hurst exponent used in the isotropic power-law spectrum.
    """
    tm = _require_tamaas()

    if h_rms < 0:
        raise ValueError("h_rms must be non-negative.")

    spectrum = tm.Isopowerlaw2D()
    spectrum.q0 = 4
    spectrum.q1 = 32
    spectrum.q2 = 128
    spectrum.hurst = hurst

    generator = tm.SurfaceGeneratorRandomPhase2D(list(shape))
    generator.spectrum = spectrum
    generator.random_seed = seed
    surface = np.asarray(generator.buildSurface(), dtype=float)

    theoretical_rms = spectrum.rmsHeights()
    if theoretical_rms > 0 and h_rms > 0:
        surface *= h_rms / theoretical_rms
    elif h_rms == 0:
        surface.fill(0.0)

    return surface


def build_equivalent_surface() -> np.ndarray:
    """Build and save the equivalent rough surface for a rigid-on-elastic setup.

    In the current model the equivalent surface is the rigid rough surface after
    removing its spatial mean. The resulting arrays are saved to the output
    directory and returned for the coupled solver.
    """
    config.validate()
    os.makedirs(config.output_dir, exist_ok=True)

    shape = (config.n_x, config.n_y)
    s_rigid = generate_surface(
        shape=shape,
        h_rms=config.h_rms_rigid,
        seed=config.seed_rigid,
        hurst=config.hurst,
    )

    s_eq = s_rigid - np.mean(s_rigid)
    np.save(config.rigid_surf_path, s_rigid)
    np.save(config.equiv_surf_path, s_eq)

    print(f"Equivalent rough surface saved to: {config.equiv_surf_path}")
    return s_eq
