"""Command-line entry point for load-sweep E-T-M simulations."""

from __future__ import annotations

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .coupling import high_precision_coupling_iteration
from .surface import build_equivalent_surface


def _require_tamaas():
    try:
        import tamaas as tm
        from tamaas.dumpers import H5Dumper
    except ImportError as exc:
        raise ImportError("Tamaas is required to run and dump full contact simulations.") from exc
    return tm, H5Dumper


def _estimate_contact_area(model, domain_area: float) -> dict[str, float | int | bool]:
    tm, _ = _require_tamaas()

    # Keep the original Tamaas GridWrap for Tamaas statistics.  Converting it to
    # np.ndarray is fine for simple node counting, but tm.Statistics2D.contact()
    # requires the original GridWrap object.
    traction_grid = model.traction
    pressure = np.asarray(traction_grid, dtype=float)

    contact_nodes = int(np.count_nonzero(pressure > 0.0))
    total_nodes = int(pressure.size)
    node_area_fraction = contact_nodes / total_nodes if total_nodes else 0.0

    output = {
        "contact_area_fraction": node_area_fraction,
        "contact_area_absolute_m2": domain_area * node_area_fraction,
        "contact_cluster_count": 0,
        "contact_perimeter_segments": 0,
        "contact_nodes": contact_nodes,
        "total_nodes": total_nodes,
        "contact_area_uses_tamaas_correction": False,
    }

    try:
        # Follow the Tamaas API: FloodFill works on the GridWrap contact mask,
        # and Statistics2D.contact expects (tractions: GridWrap, perimeter: int).
        try:
            contact_mask = traction_grid > 0.0
        except TypeError:
            # Some Tamaas/Python combinations expose only the NumPy comparison.
            contact_mask = pressure > 0.0
        clusters = tm.FloodFill.getClusters(contact_mask, False)
        perimeter_segments = int(sum(int(cluster.perimeter) for cluster in clusters))

        output.update(
            {
                "contact_cluster_count": int(len(clusters)),
                "contact_perimeter_segments": perimeter_segments,
            }
        )

        area_fraction = float(tm.Statistics2D.contact(traction_grid, perimeter_segments))
        output.update(
            {
                "contact_area_fraction": area_fraction,
                "contact_area_absolute_m2": area_fraction * domain_area,
                "contact_area_uses_tamaas_correction": True,
            }
        )
    except Exception as exc:
        print(f"    > WARNING: Tamaas contact-area statistics failed; using node-count fallback. Reason: {exc}")

    return output


def _plot_convergence(history: list[float], step_num: int, load: float) -> None:
    if not history:
        return

    plt.figure(figsize=(10, 6))
    plt.semilogy(history, marker="o", linestyle="-", markersize=3, linewidth=1.5, label="Relative update")
    plt.axhline(y=config.coupling_tolerance, linestyle="--", linewidth=1.5, label=f"Tolerance ({config.coupling_tolerance:.1e})")
    plt.xlabel("Iteration")
    plt.ylabel("Residual / relative update")
    plt.title(f"E-T-M coupling convergence (step {step_num}, F={load:.1f} N)")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.7)

    plot_path = os.path.join(config.output_dir, f"step_{step_num:03d}_F_{load:.1f}N_convergence.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  > Convergence plot saved to: {plot_path}")


def _save_step_results(final_state: dict, history: list[float], s_geom: np.ndarray, step_num: int, load: float, domain_area: float) -> None:
    _, H5Dumper = _require_tamaas()
    model = final_state.get("model")
    if model is None:
        print(f"\n--- Step {step_num} failed: no mechanical model is available. ---")
        return

    print(f"\n--- Saving step {step_num} results ---")
    stats = _estimate_contact_area(model, domain_area)
    print(f"  > Contact clusters: {stats['contact_cluster_count']}")
    print(f"  > Real contact area fraction: {stats['contact_area_fraction']:.6f}")
    print(f"  > Real contact area: {stats['contact_area_absolute_m2'] * 1e6:.6f} mm^2")

    step_basename = os.path.join(config.output_dir, f"step_{step_num:03d}_F_{load:.1f}N_solution")

    # Let Tamaas dump only its native mechanical fields.  Custom NumPy fields are
    # written explicitly below with h5py.  This avoids a Tamaas H5Dumper field-name
    # collision/uninitialised-buffer issue observed for the name "temperature",
    # which can otherwise produce values of order 1e308 in the saved HDF5 file even
    # though the thermal solver returned a finite field.
    dumper = H5Dumper(step_basename, all_fields=True)
    model.addDumper(dumper)
    model.dump()

    # Tamaas writes the actual field data under config.hdf5_dump_dir with a
    # _0000 suffix.  Overwrite/add custom fields there directly so post-processing
    # reads exactly the arrays produced by the coupled solver.
    data_hdf5_path = os.path.join(config.hdf5_dump_dir, f"{os.path.basename(step_basename)}_0000.h5")
    custom_fields = {
        "temperature": final_state["temperature"],
        "heat_flux": final_state["heat_flux"],
        "voltage": final_state["voltage"],
        "current_density": final_state["current_density"],
        "thermal_displacement": final_state["thermal_displacement"],
        "geom_surface": s_geom,
    }
    with h5py.File(data_hdf5_path, "a") as h5_data:
        for field_name, field_value in custom_fields.items():
            array = np.asarray(field_value, dtype=float)
            if field_name in h5_data:
                del h5_data[field_name]
            h5_data.create_dataset(field_name, data=array)

    hdf5_path = f"{step_basename}.h5"
    final_error = history[-1] if history else float("inf")
    with h5py.File(hdf5_path, "a") as h5:
        for key, value in stats.items():
            h5.attrs[key] = value
        h5.attrs["load_N"] = load
        h5.attrs["target_pressure_Pa"] = load / domain_area
        h5.attrs["total_current_A"] = final_state.get("total_current", np.nan)
        h5.attrs["ECR_ohm"] = final_state.get("ECR", np.nan)
        h5.attrs["coupled_converged"] = bool(final_state.get("converged", False))
        h5.attrs["coupled_final_error"] = final_error

    print(f"  > Field data and scalar attributes saved for step {step_num}.")
    print(f"  > Final coupled residual/update: {final_error:.3e}")
    print(f"  > Coupled convergence: {bool(final_state.get('converged', False))}")


def main() -> None:
    config.validate()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.hdf5_dump_dir, exist_ok=True)

    print("--- Step 1: generating the rough surface ---")
    s_geom = build_equivalent_surface()

    domain_area = config.L_x * config.L_y
    loads = np.linspace(config.F_min, config.F_max, config.N_LOAD_STEPS)

    print(f"\n--- Running load sweep with {config.N_LOAD_STEPS} load steps ---")
    print(f"    Load range: {config.F_min:.1f} N to {config.F_max:.1f} N")
    print(f"    Applied voltage: {config.Delta_V_total:.4g} V")

    for i, current_force in enumerate(loads):
        step_num = i + 1
        target_pressure = current_force / domain_area

        print("\n" + "=" * 70)
        print(f"  Load step {step_num} / {config.N_LOAD_STEPS}")
        print(f"  Normal load: {current_force:.3f} N")
        print(f"  Target mean pressure: {target_pressure / 1e6:.3f} MPa")
        print("=" * 70)

        try:
            final_state, history = high_precision_coupling_iteration(
                s_geom,
                target_pressure,
                method="adaptive",
                tolerance=config.coupling_tolerance,
            )
        except Exception as exc:
            print(f"  > ERROR: coupled solve failed at step {step_num}: {exc}")
            continue

        _plot_convergence(history, step_num, current_force)
        _save_step_results(final_state, history, s_geom, step_num, current_force, domain_area)

    print("\n" + "=" * 70)
    print("Load sweep complete.")
    print(f"Results are stored in: {config.output_dir}")


if __name__ == "__main__":
    main()
