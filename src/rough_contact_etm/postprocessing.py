"""Post-processing utilities for rough-contact-etm HDF5 outputs."""

from __future__ import annotations

import glob
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np

from . import config


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
    """Return True when a stored temperature field is finite and numerically sane."""
    if temperature.size == 0:
        return False
    if not np.all(np.isfinite(temperature)):
        return False
    # Temperatures much above this threshold are not credible for this reduced
    # model and usually indicate a corrupted HDF5 custom-field dump.
    return float(np.nanmax(np.abs(temperature))) < 1.0e12


def _read_temperature_or_recompute(h5_data: h5py.File, data_filepath: str) -> np.ndarray:
    """Read temperature, or reconstruct it from heat_flux if the stored field is corrupt."""
    temperature = None
    if "temperature" in h5_data:
        temperature = np.asarray(h5_data["temperature"][:], dtype=float)
        if _temperature_field_is_valid(temperature):
            return temperature

    if "heat_flux" not in h5_data:
        raise ValueError(f"No valid temperature dataset and no heat_flux dataset in {data_filepath}")

    from .thermal import solve_thermal_field

    heat_flux = np.asarray(h5_data["heat_flux"][:], dtype=float)
    temperature, _ = solve_thermal_field(heat_flux)
    if "temperature" in h5_data:
        del h5_data["temperature"]
    h5_data.create_dataset("temperature", data=temperature)
    print(f"  > Recomputed and repaired corrupted temperature dataset in: {data_filepath}")
    return temperature

def extract_scan_data(attr_files: list[str]) -> dict[str, list[float]]:
    """Extract scalar scan data from HDF5 attribute and field-data file pairs."""
    print(f"--- Extracting data from {len(attr_files)} file pairs ---")

    results: dict[str, list[float]] = {
        "loads": [],
        "currents": [],
        "resistances_mOhm": [],
        "area_fractions": [],
        "cluster_counts": [],
        "max_temps": [],
        "avg_temps": [],
    }

    dx = config.L_x / config.n_x
    dy = config.L_y / config.n_y
    dA = dx * dy
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

            total_current = float(stored_current) if np.isfinite(stored_current) else float(np.nansum(current_density) * dA)
            if np.isfinite(stored_resistance):
                resistance_ohm = float(stored_resistance)
            elif total_current > 0.0:
                resistance_ohm = voltage / total_current
            else:
                resistance_ohm = np.inf

            max_temp = float(np.nanmax(temperature))
            avg_temp = float(np.nanmean(temperature))

            results["loads"].append(load_val)
            results["currents"].append(total_current)
            results["resistances_mOhm"].append(resistance_ohm * 1000.0)
            results["area_fractions"].append(area_frac)
            results["cluster_counts"].append(clusters)
            results["max_temps"].append(max_temp)
            results["avg_temps"].append(avg_temp)

            print(
                f"  > Processed F={load_val:.1f} N | "
                f"A_r/A_0={area_frac:.6f} | I={total_current:.6g} A | "
                f"R={resistance_ohm * 1000.0:.6g} mOhm"
            )
        except Exception as exc:
            print(f"  > WARNING: failed to process {basename}: {exc}")

    order = np.argsort(results["loads"]) if results["loads"] else []
    if len(order):
        for key, values in results.items():
            results[key] = [values[i] for i in order]

    return results


def plot_evolution_curves(results: dict[str, list[float]], plot_dir: str) -> None:
    """Plot load-dependent scalar quantities."""
    print("\n--- Generating scalar evolution plots ---")
    loads = np.asarray(results["loads"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("E-T-M coupling simulation results vs. load", fontsize=18)

    ax = axes[0, 0]
    ax.plot(loads, results["currents"], "o-")
    ax.set_title("Total current vs. load")
    ax.set_xlabel("Applied load (N)")
    ax.set_ylabel("Total current (A)")
    ax.grid(True, linestyle=":")

    ax = axes[0, 1]
    ax.plot(loads, results["resistances_mOhm"], "s-")
    ax.set_title("Electrical contact resistance vs. load")
    ax.set_xlabel("Applied load (N)")
    ax.set_ylabel("ECR (mOhm)")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", which="both")

    ax = axes[1, 0]
    ax.plot(loads, 100.0 * np.asarray(results["area_fractions"]), "^-")
    ax.set_title("Real contact area fraction vs. load")
    ax.set_xlabel("Applied load (N)")
    ax.set_ylabel("Real contact area A_r/A_0 (%)")
    ax.grid(True, linestyle=":")

    ax = axes[1, 1]
    ax.plot(loads, results["max_temps"], "o-", label="Max temperature")
    ax.plot(loads, results["avg_temps"], "o--", label="Average temperature")
    ax.set_title("Surface temperature vs. load")
    ax.set_xlabel("Applied load (N)")
    ax.set_ylabel("Temperature rise (K)")
    ax.legend()
    ax.grid(True, linestyle=":")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = os.path.join(plot_dir, "summary_evolution_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  > Evolution plot saved to: {save_path}")


def plot_2d_fields(attr_filepath: str, plot_dir: str) -> None:
    """Plot contact, temperature, and current-density fields for one load step."""
    basename = os.path.basename(attr_filepath)
    print(f"\n--- Generating 2D field plots for {basename} ---")
    data_filepath = get_data_filepath(attr_filepath)
    if not os.path.exists(data_filepath):
        print(f"  > WARNING: field-data file not found: {data_filepath}")
        return

    with h5py.File(data_filepath, "r+") as h5:
        temperature = _read_temperature_or_recompute(h5, data_filepath)
        current_density = np.asarray(h5["current_density"][:], dtype=float)
        pressure = np.asarray(h5["traction"][:], dtype=float)

    contact_map = pressure > 0.0
    extent_mm = [0.0, config.L_x * 1000.0, 0.0, config.L_y * 1000.0]
    load_label = f"{parse_load_from_filename(attr_filepath):.1f} N"

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f"2D field distributions (load: {load_label})", fontsize=16)

    ax = axes[0]
    ax.imshow(contact_map, origin="lower", extent=extent_mm, cmap="gray")
    ax.set_title("Contact spots (traction > 0)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    ax = axes[1]
    image = ax.imshow(temperature, origin="lower", extent=extent_mm, cmap="hot")
    ax.set_title(f"Temperature field (max: {np.nanmax(temperature):.3g} K)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(image, ax=ax, label="Temperature rise (K)", fraction=0.046, pad=0.04)

    ax = axes[2]
    current_on_contact = np.where(contact_map, current_density, 0.0)
    image = ax.imshow(current_on_contact, origin="lower", extent=extent_mm, cmap="viridis")
    ax.set_title(f"Current density (max: {np.nanmax(current_on_contact):.3e} A/m^2)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(image, ax=ax, label="Current density (A/m^2)", fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(plot_dir, f"{os.path.splitext(basename)[0]}_2D_fields.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  > 2D field plot saved to: {save_path}")


def main() -> None:
    plot_output_dir = os.path.join(config.output_dir, "post_processing_plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    attr_files = sorted(glob.glob(os.path.join(config.output_dir, "step_*_solution.h5")))
    if not attr_files:
        print(f"ERROR: no step attribute files found in {config.output_dir!r}.")
        print("Run the load-sweep simulation first.")
        return

    scan_results = extract_scan_data(attr_files)
    if not scan_results["loads"]:
        print("ERROR: no valid data could be extracted.")
        return

    plot_evolution_curves(scan_results, plot_output_dir)
    plot_2d_fields(attr_files[0], plot_output_dir)
    if len(attr_files) > 1:
        plot_2d_fields(attr_files[-1], plot_output_dir)

    print("\nPost-processing complete.")


if __name__ == "__main__":
    main()
