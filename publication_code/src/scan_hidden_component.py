#!/usr/bin/env python3
"""Minimize combined dust mass at every selected hot-component temperature.

For every temperature T and every trial tau_h,100:

1. The projected area is set so the model closes the 100-micron deficit.
2. tau_f,100 is fitted to baseline+hidden versus observed flux over 5--7 micron.
3. tau_h,100 is varied to minimize M_dust,hot + M_dust,foreground.

This is deterministic nested one-dimensional optimization, not an MCMC.
The default tau_h,100 >= 1 preserves the assumption that the hidden component
is optically thick at 100 micron.  Change --tau-hot-min to explore otherwise.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar

import hidden_component_model as model

from project_paths import HIDDEN_COMPONENT_RESULTS


DEFAULT_OUTPUT = HIDDEN_COMPONENT_RESULTS / "rerun_temperature_scan.csv"
DEFAULT_VALIDATION_OUTPUT = (
    HIDDEN_COMPONENT_RESULTS / "rerun_validation.csv"
)


def bounded_global_minimum(
    function,
    lower: float,
    upper: float,
    coarse_points: int = 41,
) -> tuple[float, float]:
    """Coarse global scan followed by a bounded local refinement."""
    grid = np.linspace(lower, upper, coarse_points)
    values = np.asarray([function(value) for value in grid], dtype=float)
    best_index = int(np.argmin(values))
    candidates = [(float(grid[best_index]), float(values[best_index]))]

    if best_index > 0 and best_index < len(grid) - 1:
        result = minimize_scalar(
            function,
            bounds=(float(grid[best_index - 1]), float(grid[best_index + 1])),
            method="bounded",
            options={"xatol": 1.0e-8, "maxiter": 200},
        )
        candidates.append((float(result.x), float(result.fun)))

    candidates.extend(
        [
            (float(lower), float(function(lower))),
            (float(upper), float(function(upper))),
        ]
    )
    return min(candidates, key=lambda item: item[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature-min-K", type=float, default=100.0)
    parser.add_argument("--temperature-max-K", type=float, default=2000.0)
    parser.add_argument("--temperature-step-K", type=float, default=20.0)
    parser.add_argument("--tau-hot-min", type=float, default=1.0)
    parser.add_argument("--tau-hot-max", type=float, default=100.0)
    parser.add_argument("--tau-foreground-min", type=float, default=0.01)
    parser.add_argument("--tau-foreground-max", type=float, default=100.0)
    parser.add_argument("--foreground-fit-min-micron", type=float, default=5.0)
    parser.add_argument("--foreground-fit-max-micron", type=float, default=7.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation-output", type=Path, default=DEFAULT_VALIDATION_OUTPUT
    )
    args = parser.parse_args()

    if not 0.0 < args.temperature_min_K <= args.temperature_max_K:
        parser.error("temperature bounds must be positive and ordered")
    if args.temperature_step_K <= 0.0:
        parser.error("temperature step must be positive")
    if not 0.0 < args.tau_hot_min < args.tau_hot_max:
        parser.error("hot-tau bounds must be positive and ordered")
    if not 0.0 < args.tau_foreground_min < args.tau_foreground_max:
        parser.error("foreground-tau bounds must be positive and ordered")
    if not args.foreground_fit_min_micron < args.foreground_fit_max_micron:
        parser.error("foreground fit-band bounds must be ordered")

    wavelength, observed, baseline, fitted_kappa = model.load_area2_beta1_case()
    kappa_foreground = fitted_kappa.copy()
    kappa_hot = fitted_kappa.copy()
    calculation_settings = SimpleNamespace(
        foreground_reference_micron=100.0,
        hot_reference_micron=100.0,
        closure_micron=100.0,
        total_to_dust=100.0,
        shortwave_max_micron=50.0,
        foreground_fit_min_micron=args.foreground_fit_min_micron,
        foreground_fit_max_micron=args.foreground_fit_max_micron,
    )

    log_tau_foreground_bounds = (
        float(np.log10(args.tau_foreground_min)),
        float(np.log10(args.tau_foreground_max)),
    )
    log_tau_hot_bounds = (
        float(np.log10(args.tau_hot_min)),
        float(np.log10(args.tau_hot_max)),
    )

    def fit_foreground(
        temperature_K: float,
        tau_hot_100: float,
    ) -> dict[str, float | np.ndarray]:
        def foreground_objective(log10_tau_foreground: float) -> float:
            trial = model.evaluate_emission(
                calculation_settings,
                wavelength,
                observed,
                baseline,
                kappa_foreground,
                kappa_hot,
                temperature_K,
                10.0**log10_tau_foreground,
                tau_hot_100,
            )
            return float(trial["foreground_fit_rms_log_residual"])

        best_log_tau_foreground, _ = bounded_global_minimum(
            foreground_objective,
            *log_tau_foreground_bounds,
        )
        return model.evaluate_emission(
            calculation_settings,
            wavelength,
            observed,
            baseline,
            kappa_foreground,
            kappa_hot,
            temperature_K,
            10.0**best_log_tau_foreground,
            tau_hot_100,
        )

    temperatures = np.arange(
        args.temperature_min_K,
        args.temperature_max_K + 0.5 * args.temperature_step_K,
        args.temperature_step_K,
    )
    rows: list[dict[str, float]] = []
    validation_rows: list[dict[str, float | bool]] = []

    for temperature in temperatures:
        cache: dict[float, dict[str, float | np.ndarray]] = {}

        def mass_objective(log10_tau_hot: float) -> float:
            key = float(log10_tau_hot)
            if key not in cache:
                cache[key] = fit_foreground(temperature, 10.0**key)
            return float(cache[key]["combined_dust_mass_Msun"])

        best_log_tau_hot, _ = bounded_global_minimum(
            mass_objective,
            *log_tau_hot_bounds,
        )
        best = fit_foreground(temperature, 10.0**best_log_tau_hot)
        row = {
            "T_K": float(temperature),
            "tau_h_100": float(best["tau_hot_reference"]),
            "tau_f_100": float(best["tau_foreground_reference"]),
            # Projected solid-angle area required for the hidden component to
            # close the observed-minus-baseline deficit at 100 microns.
            "required_area_arcsec2": float(best["projected_area_arcsec2"]),
            "Mdust_hot_Msun": float(best["hot_dust_mass_Msun"]),
            "Mdust_combined_Msun": float(best["combined_dust_mass_Msun"]),
        }
        rows.append(row)
        validation_rows.append(
            {
                "T_K": float(temperature),
                "foreground_fit_rms_log_residual": float(
                    best["foreground_fit_rms_log_residual"]
                ),
                "foreground_fit_maximum_absolute_fractional_residual": float(
                    best["foreground_fit_maximum_absolute_fractional_residual"]
                ),
                "tau_f_at_lower_bound": bool(
                    np.isclose(
                        row["tau_f_100"],
                        args.tau_foreground_min,
                        rtol=1.0e-5,
                        atol=1.0e-10,
                    )
                ),
                "max_fractional_residual_le_10_percent": bool(
                    float(
                        best[
                            "foreground_fit_maximum_absolute_fractional_residual"
                        ]
                    )
                    <= 0.10
                ),
            }
        )
        print(
            f"T={temperature:7.1f} K  "
            f"tau_h,100={row['tau_h_100']:.7g}  "
            f"tau_f,100={row['tau_f_100']:.7g}  "
            f"area={row['required_area_arcsec2']:.7g} arcsec^2  "
            f"M_hot={row['Mdust_hot_Msun']:.7g} Msun  "
            f"M_combined={row['Mdust_combined_Msun']:.7g} Msun"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    with args.validation_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)
    print(f"\nSaved requested table: {args.output}")
    print(f"Saved 5--7 micron fit validation: {args.validation_output}")


if __name__ == "__main__":
    main()
