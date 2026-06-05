"""Journal-style post-processing for rough-contact-etm HDF5 outputs.

This script is intentionally self-contained so it can replace the top-level
``post_processing.py`` file in the project root.  It reads the scalar attribute
files written in ``config.output_dir`` and the corresponding field files written
in ``config.hdf5_dump_dir``.
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Iterable

import h5py
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rough_contact_etm import config  # noqa: E402


# -----------------------------------------------------------------------------
# Global plotting style
# -----------------------------------------------------------------------------


def set_journal_style() -> None:
    """Apply a compact journal-style Matplotlib configuration."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.9,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.minorticks_on()
    if grid:
        ax.grid(True, which="major", color="0.82", linestyle="--", linewidth=0.45)
        ax.grid(True, which="minor", color="0.90", linestyle=":", linewidth=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )


def _save_figure(fig: plt.Figure, path_without_suffix: str, save_pdf: bool = True) -> None:
    png_path = f"{path_without_suffix}.png"
    fig.savefig(png_path)
    print(f"  > Saved: {png_path}")
    if save_pdf:
        pdf_path = f"{path_without_suffix}.pdf"
        fig.savefig(pdf_path)
        print(f"  > Saved: {pdf_path}")


# -----------------------------------------------------------------------------
# HDF5 data access
# -----------------------------------------------------------------------------


def get_data_filepath(attr_filepath: str) -> str:
    """Map a scalar-attribute HDF5 path to the corresponding field-data HDF5 path."""
    basename = os.path.basename(attr_filepath)
    data_basename = basename.replace(".h5", "_0000.h5")
    return os.path.join(config.hdf5_dump_dir, data_basename)


def parse_load_from_filename(path: str) -> float:
    """Extract the load value from names such as ``step_001_F_50.0N_solution.h5``."""
    match = re.search(r"_F_([0-9.+\-eE]+)N_", os.path.basename(path))
    if match is None:
        raise ValueError(f"Could not parse load from filename: {path}")
    return float(match.group(1))


def _temperature_field_is_valid(temperature: np.ndarray) -> bool:
    if temperature.size == 0:
        return False
    if not np.all(np.isfinite(temperature)):
        return False
    return float(np.nanmax(np.abs(temperature))) < 1.0e12


def _read_temperature_or_recompute(h5_data: h5py.File, data_filepath: str) -> np.ndarray:
    """Read temperature, or reconstruct it from heat_flux if the stored field is corrupt."""
    if "temperature" in h5_data:
        temperature = np.asarray(h5_data["temperature"][:], dtype=float)
        if _temperature_field_is_valid(temperature):
            return temperature

    if "heat_flux" not in h5_data:
        raise ValueError(f"No valid temperature dataset and no heat_flux dataset in {data_filepath}")

    from rough_contact_etm.thermal import solve_thermal_field

    heat_flux = np.asarray(h5_data["heat_flux"][:], dtype=float)
    temperature, _ = solve_thermal_field(heat_flux)
    if "temperature" in h5_data:
        del h5_data["temperature"]
    h5_data.create_dataset("temperature", data=temperature)
    print(f"  > Recomputed and repaired corrupted temperature dataset in: {data_filepath}")
    return temperature


@dataclass
class FieldData:
    load_N: float
    pressure_Pa: np.ndarray
    temperature_K: np.ndarray
    current_density_A_m2: np.ndarray
    heat_flux_W_m2: np.ndarray | None


# -----------------------------------------------------------------------------
# Scalar data extraction
# -----------------------------------------------------------------------------


def extract_scan_data(attr_files: list[str]) -> dict[str, list[float]]:
    """Extract scalar scan data from HDF5 attribute and field-data file pairs."""
    print(f"--- Extracting data from {len(attr_files)} file pairs ---")

    results: dict[str, list[float]] = {
        "loads_N": [],
        "loads_kN": [],
        "currents_A": [],
        "currents_kA": [],
        "resistances_mOhm": [],
        "area_fractions": [],
        "areas_mm2": [],
        "cluster_counts": [],
        "max_temps_K": [],
        "avg_temps_K": [],
    }

    dx = config.L_x / config.n_x
    dy = config.L_y / config.n_y
    dA = dx * dy
    nominal_area_mm2 = config.L_x * config.L_y * 1.0e6
    voltage = config.Delta_V_total

    for attr_filepath in attr_files:
        basename = os.path.basename(attr_filepath)
        try:
            load_val = parse_load_from_filename(attr_filepath)
            data_filepath = get_data_filepath(attr_filepath)
            if not os.path.exists(data_filepath):
                raise FileNotFoundError(f"Matching field-data file not found: {data_filepath}")

            with h5py.File(attr_filepath, "r") as h5_attr:
                area_frac = float(h5_attr.attrs.get("contact_area_fraction", np.nan))
                clusters = float(h5_attr.attrs.get("contact_cluster_count", np.nan))
                stored_current = h5_attr.attrs.get("total_current_A", np.nan)
                stored_resistance = h5_attr.attrs.get("ECR_ohm", np.nan)

            with h5py.File(data_filepath, "r+") as h5_data:
                temperature = _read_temperature_or_recompute(h5_data, data_filepath)
                current_density = np.asarray(h5_data["current_density"][:], dtype=float)

            total_current = (
                float(stored_current)
                if np.isfinite(stored_current)
                else float(np.nansum(current_density) * dA)
            )
            if np.isfinite(stored_resistance):
                resistance_ohm = float(stored_resistance)
            elif total_current > 0.0:
                resistance_ohm = voltage / total_current
            else:
                resistance_ohm = np.inf

            max_temp = float(np.nanmax(temperature))
            avg_temp = float(np.nanmean(temperature))
            area_mm2 = area_frac * nominal_area_mm2

            results["loads_N"].append(load_val)
            results["loads_kN"].append(load_val / 1000.0)
            results["currents_A"].append(total_current)
            results["currents_kA"].append(total_current / 1000.0)
            results["resistances_mOhm"].append(resistance_ohm * 1000.0)
            results["area_fractions"].append(area_frac)
            results["areas_mm2"].append(area_mm2)
            results["cluster_counts"].append(clusters)
            results["max_temps_K"].append(max_temp)
            results["avg_temps_K"].append(avg_temp)

            print(
                f"  > Processed F={load_val:.1f} N | "
                f"A_r={area_mm2:.4f} mm^2 | I={total_current:.6g} A | "
                f"R={resistance_ohm * 1000.0:.6g} mOhm | Tmax={max_temp:.4g} K"
            )
        except Exception as exc:
            print(f"  > WARNING: failed to process {basename}: {exc}")

    order = np.argsort(results["loads_N"]) if results["loads_N"] else []
    if len(order):
        for key, values in results.items():
            results[key] = [values[i] for i in order]

    return results


# -----------------------------------------------------------------------------
# Scalar evolution figure
# -----------------------------------------------------------------------------


def plot_evolution_curves(results: dict[str, list[float]], plot_dir: str, save_pdf: bool = True) -> None:
    """Plot load-dependent scalar quantities in a journal-style 2x2 layout."""
    print("\n--- Generating scalar evolution plots ---")
    loads_kN = np.asarray(results["loads_kN"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(loads_kN, results["areas_mm2"], marker="s", color="C0")
    ax.set_xlabel(r"Contact force $P$ (kN)")
    ax.set_ylabel(r"Real contact area $A_r$ (mm$^2$)")
    _style_axes(ax)
    _panel_label(ax, "(a)")

    ax = axes[0, 1]
    resistance = np.asarray(results["resistances_mOhm"], dtype=float)
    positive = resistance[np.isfinite(resistance) & (resistance > 0.0)]
    ax.plot(loads_kN, resistance, marker="o", color="C1")
    ax.set_yscale("log")
    if positive.size:
        ax.set_ylim(10 ** np.floor(np.log10(positive.min()) - 0.1), 10 ** np.ceil(np.log10(positive.max()) + 0.1))
    ax.set_xlabel(r"Contact force $P$ (kN)")
    ax.set_ylabel(r"ECR $R_c$ (m$\Omega$)")
    _style_axes(ax)
    _panel_label(ax, "(b)")

    ax = axes[1, 0]
    ax.plot(loads_kN, results["currents_kA"], marker="^", color="C2")
    ax.set_xlabel(r"Contact force $P$ (kN)")
    ax.set_ylabel(r"Electric current $I$ (kA)")
    _style_axes(ax)
    _panel_label(ax, "(c)")

    ax = axes[1, 1]
    ax.plot(loads_kN, results["max_temps_K"], marker="D", color="C3", label=r"$T_{\max}$")
    ax.plot(
        loads_kN,
        results["avg_temps_K"],
        marker="o",
        linestyle="--",
        color="0.25",
        label=r"$\overline{T}$",
    )
    ax.set_xlabel(r"Contact force $P$ (kN)")
    ax.set_ylabel(r"Temperature rise $T$ (K)")
    ax.legend(loc="best")
    _style_axes(ax)
    _panel_label(ax, "(d)")

    stem = os.path.join(plot_dir, "summary_evolution_curves")
    _save_figure(fig, stem, save_pdf=save_pdf)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Field figure helpers
# -----------------------------------------------------------------------------


def _read_field_data(attr_filepath: str) -> FieldData | None:
    data_filepath = get_data_filepath(attr_filepath)
    if not os.path.exists(data_filepath):
        print(f"  > WARNING: field-data file not found: {data_filepath}")
        return None

    with h5py.File(data_filepath, "r+") as h5:
        pressure = np.asarray(h5["traction"][:], dtype=float)
        temperature = _read_temperature_or_recompute(h5, data_filepath)
        current_density = np.asarray(h5["current_density"][:], dtype=float)
        heat_flux = np.asarray(h5["heat_flux"][:], dtype=float) if "heat_flux" in h5 else None

    return FieldData(
        load_N=parse_load_from_filename(attr_filepath),
        pressure_Pa=np.maximum(pressure, 0.0),
        temperature_K=temperature,
        current_density_A_m2=current_density,
        heat_flux_W_m2=heat_flux,
    )


def _finite_limits(data: np.ndarray, lower: float = 1.0, upper: float = 99.5) -> tuple[float, float]:
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(values, lower))
    vmax = float(np.percentile(values, upper))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _positive_limits(data: np.ndarray, upper: float = 99.8) -> tuple[float, float]:
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 0.0, 1.0
    vmin = 0.0
    vmax = float(np.percentile(values, upper))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return vmin, vmax


def _masked_cmap(name: str, bad: str = "#f7f7f7") -> mcolors.Colormap:
    cmap = mpl.colormaps[name].copy()
    cmap.set_bad(bad)
    return cmap


def _field_colorbar(image, ax: plt.Axes, label: str) -> None:
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label(label)
    cbar.ax.tick_params(direction="in", length=3.0, width=0.8, labelsize=8.5)
    cbar.outline.set_linewidth(0.8)


def _format_field_axis(ax: plt.Axes) -> None:
    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"$y$ (mm)")
    ax.set_aspect("equal")
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def plot_2d_fields(attr_filepath: str, plot_dir: str, save_pdf: bool = True) -> None:
    """Plot pressure, temperature, current-density, and contact maps for one load step."""
    basename = os.path.basename(attr_filepath)
    print(f"\n--- Generating 2D field plots for {basename} ---")
    fields = _read_field_data(attr_filepath)
    if fields is None:
        return

    pressure_MPa = fields.pressure_Pa / 1.0e6
    temperature = fields.temperature_K
    current_1e8 = fields.current_density_A_m2 / 1.0e8
    heat_flux_MW_m2 = None if fields.heat_flux_W_m2 is None else fields.heat_flux_W_m2 / 1.0e6

    contact_mask = fields.pressure_Pa > 0.0
    current_masked = np.ma.masked_where(~contact_mask | ~np.isfinite(current_1e8) | (current_1e8 <= 0.0), current_1e8)
    pressure_masked = np.ma.masked_where(~contact_mask | ~np.isfinite(pressure_MPa) | (pressure_MPa <= 0.0), pressure_MPa)

    extent_mm = [0.0, config.L_x * 1000.0, 0.0, config.L_y * 1000.0]
    load_label = f"{fields.load_N / 1000.0:.3g} kN" if fields.load_N >= 1000.0 else f"{fields.load_N:.3g} N"
    area_fraction = float(np.count_nonzero(contact_mask) / contact_mask.size)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.05, hspace=0.10, wspace=0.05)

    ax = axes[0, 0]
    contact_cmap = ListedColormap(["white", "black"])
    image = ax.imshow(contact_mask.astype(int), origin="lower", extent=extent_mm, cmap=contact_cmap, interpolation="nearest")
    ax.set_title(rf"Contact map, $A_d/A_0={100.0 * area_fraction:.2f}\%$")
    _format_field_axis(ax)
    _panel_label(ax, "(a)")

    ax = axes[0, 1]
    _, pmax = _positive_limits(pressure_MPa, upper=99.8)
    image = ax.imshow(
        pressure_masked,
        origin="lower",
        extent=extent_mm,
        cmap=_masked_cmap("magma"),
        vmin=0.0,
        vmax=pmax,
        interpolation="nearest",
    )
    ax.set_title(rf"Contact pressure, $p_{{\max}}={np.nanmax(pressure_MPa):.0f}$ MPa")
    _format_field_axis(ax)
    _field_colorbar(image, ax, r"$p$ (MPa)")
    _panel_label(ax, "(b)")

    ax = axes[1, 0]
    tmin, tmax = 0.0, float(np.nanmax(temperature))
    if not np.isfinite(tmax) or tmax <= tmin:
        tmin, tmax = _finite_limits(temperature, lower=0.0, upper=100.0)
    image = ax.imshow(
        temperature,
        origin="lower",
        extent=extent_mm,
        cmap="inferno",
        vmin=tmin,
        vmax=tmax,
        interpolation="bilinear",
    )
    ax.set_title(rf"Temperature rise, $T_{{\max}}={np.nanmax(temperature):.2f}$ K")
    _format_field_axis(ax)
    _field_colorbar(image, ax, r"$T$ (K)")
    _panel_label(ax, "(c)")

    ax = axes[1, 1]
    _, jmax = _positive_limits(current_1e8, upper=99.8)
    image = ax.imshow(
        current_masked,
        origin="lower",
        extent=extent_mm,
        cmap=_masked_cmap("viridis"),
        vmin=0.0,
        vmax=jmax,
        interpolation="nearest",
    )
    ax.set_title(rf"Current density, $J_{{\max}}={np.nanmax(current_1e8):.2f}\times10^8$ A/m$^2$")
    _format_field_axis(ax)
    _field_colorbar(image, ax, r"$J$ ($10^8$ A m$^{-2}$)")
    _panel_label(ax, "(d)")

    stem = os.path.join(plot_dir, f"{os.path.splitext(basename)[0]}_2D_fields")
    _save_figure(fig, stem, save_pdf=save_pdf)
    plt.close(fig)

    if heat_flux_MW_m2 is not None:
        plot_heat_flux_field(fields, plot_dir, save_pdf=save_pdf)


def plot_heat_flux_field(fields: FieldData, plot_dir: str, save_pdf: bool = True) -> None:
    """Save an additional single-panel heat-flux map for diagnostics."""
    heat_flux_MW_m2 = fields.heat_flux_W_m2 / 1.0e6
    heat_masked = np.ma.masked_where(~np.isfinite(heat_flux_MW_m2) | (heat_flux_MW_m2 <= 0.0), heat_flux_MW_m2)
    extent_mm = [0.0, config.L_x * 1000.0, 0.0, config.L_y * 1000.0]
    _, qmax = _positive_limits(heat_flux_MW_m2, upper=99.8)

    fig, ax = plt.subplots(figsize=(3.55, 3.15), constrained_layout=True)
    image = ax.imshow(
        heat_masked,
        origin="lower",
        extent=extent_mm,
        cmap=_masked_cmap("plasma"),
        vmin=0.0,
        vmax=qmax,
        interpolation="nearest",
    )
    load_label = f"{fields.load_N / 1000.0:.3g} kN" if fields.load_N >= 1000.0 else f"{fields.load_N:.3g} N"
    ax.set_title(rf"Heat flux at $P={load_label}$")
    _format_field_axis(ax)
    _field_colorbar(image, ax, r"$Q_s$ (MW m$^{-2}$)")
    stem = os.path.join(plot_dir, f"heat_flux_F_{fields.load_N:.1f}N")
    _save_figure(fig, stem, save_pdf=save_pdf)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def _select_field_files(attr_files: list[str], mode: str) -> list[str]:
    if mode == "none":
        return []
    if mode == "all":
        return attr_files
    if len(attr_files) <= 1:
        return attr_files
    mid = attr_files[len(attr_files) // 2]
    selected = [attr_files[0], mid, attr_files[-1]] if mode == "first-middle-last" else [attr_files[0], attr_files[-1]]
    # Preserve order while removing duplicates.
    unique = []
    for item in selected:
        if item not in unique:
            unique.append(item)
    return unique


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate journal-style post-processing figures.")
    parser.add_argument(
        "--fields",
        choices=["first-last", "first-middle-last", "all", "none"],
        default="first-last",
        help="Which load steps should be used for 2D field figures.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Only save PNG files. By default both PNG and PDF are saved.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    set_journal_style()
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    save_pdf = not args.no_pdf

    plot_output_dir = os.path.join(config.output_dir, "post_processing_plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    attr_files = sorted(glob.glob(os.path.join(config.output_dir, "step_*_solution.h5")))
    if not attr_files:
        print(f"ERROR: no step attribute files found in {config.output_dir!r}.")
        print("Run the load-sweep simulation first.")
        return

    scan_results = extract_scan_data(attr_files)
    if not scan_results["loads_N"]:
        print("ERROR: no valid data could be extracted.")
        return

    plot_evolution_curves(scan_results, plot_output_dir, save_pdf=save_pdf)

    for attr_file in _select_field_files(attr_files, args.fields):
        plot_2d_fields(attr_file, plot_output_dir, save_pdf=save_pdf)

    print("\nPost-processing complete.")


if __name__ == "__main__":
    main()
