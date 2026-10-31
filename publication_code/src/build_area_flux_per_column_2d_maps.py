#!/usr/bin/env python3
"""Build two-dimensional empirical 18-micron maps from Area 1 and Area 2.

This is the 2-D companion to
``build_area_flux_per_column_latitude_profiles.py``.  It deliberately keeps
the reconstructed observed map unchanged.  The hard latitude cutoff is
applied only to the two Smith-distributed predictions.

For each aperture calibration, the already-established normalization is

    eta_18 = F_18,ap / (Sigma_LOS * A_1arcsec2),
    dF_18/dOmega_star = eta_18 * M_dust,global * p(lambda).

The same source-solid-angle cells used for the one-dimensional profile are
projected into the observed (R.A.-display offset, declination offset) grid.
Area 1 and Area 2 remain independent alternatives; they are never added.

The saved residuals are

    observed - Area-1-derived model
    observed - Area-2-derived model.

Positive residuals are flux not supplied by the relevant lobe model.
Residuals are defined only inside the observed reconstruction footprint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter

from project_paths import (
    MODEL_INPUT_DIR,
    OBSERVED_DATA_DIR,
    SMITH_GEOMETRY_DIR,
    TWO_PHASE_RESULTS,
    WHOLE_NEBULA_RESULTS,
)

import build_area_flux_per_column_latitude_profiles as one_d


PA_DEG = 132.0
DISPLAY_SMOOTH_FWHM_ARCSEC = 0.15
OBSERVED_COLOR = "#9f2f1f"
AREA1_COLOR = "#126782"
AREA2_COLOR = "#6f3f8f"


def angular_grid(
    shape: tuple[int, int], decisions: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return increasing x/y centers and edges for the observed native grid.

    The reconstruction stores rows from positive to negative declination.
    We flip the observed arrays vertically and work with an increasing-y
    representation throughout this script.
    """
    height, width = shape
    registration = decisions["registration"]
    star_x, star_y = map(float, registration["fig3_star_xy"])
    dx = float(registration["fig3_arcsec_per_pixel_x"])
    dy = float(registration["fig3_arcsec_per_pixel_y"])
    x = (np.arange(width, dtype=float) - star_x) * dx
    y_descending = (np.arange(height, dtype=float) - star_y) * (-dy)
    y = y_descending[::-1]
    return x, y, one_d.bin_edges_from_centers(x), one_d.bin_edges_from_centers(y)


def project_mass_fraction_to_xy(
    latitude_result: dict[str, np.ndarray | float],
    velocity_curve: PchipInterpolator,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    pa_deg: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Project the retained Smith mass fractions into the observed x-y grid.

    Each source-solid-angle element carries a fixed mass fraction.  The
    empirical flux-per-mass assumption makes its flux fraction identical.
    The latitude cutoff is already encoded in ``latitude_result``; the
    observed map is not involved in this calculation.
    """
    latitude = np.asarray(latitude_result["latitude_deg"], dtype=float)
    mass_fraction = np.asarray(latitude_result["mass_fraction"], dtype=float)
    phi = np.deg2rad(
        np.arange(
            one_d.PHI_STEP_DEG / 2.0,
            360.0,
            one_d.PHI_STEP_DEG,
            dtype=float,
        )
    )
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    inclination = np.deg2rad(one_d.INCLINATION_FROM_LOS_DEG)
    pa = np.deg2rad(pa_deg)
    ages_year = one_d.BALLISTIC_AGE_YR + np.linspace(
        -0.5 * one_d.EJECTION_DURATION_YR,
        0.5 * one_d.EJECTION_DURATION_YR,
        one_d.EJECTION_LAYERS,
    )
    ages_seconds = ages_year[:, None] * one_d.SECONDS_PER_YEAR

    fraction_map = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)
    outside_fraction = 0.0
    minimum_x = np.inf
    maximum_x = -np.inf
    minimum_y = np.inf
    maximum_y = -np.inf

    elements_per_latitude = 2 * len(phi) * one_d.EJECTION_LAYERS
    for latitude_abs, latitude_fraction in zip(latitude, mass_fraction):
        radius_cm = float(velocity_curve(latitude_abs)) * 1.0e5 * ages_seconds
        cell_weight = float(latitude_fraction) / elements_per_latitude
        for hemisphere in (-1.0, 1.0):
            lat = np.deg2rad(hemisphere * latitude_abs)
            along_cm = radius_cm * (
                -np.cos(lat) * cos_phi[None, :] * np.cos(inclination)
                + np.sin(lat) * np.sin(inclination)
            )
            perpendicular_cm = (
                radius_cm * np.cos(lat) * sin_phi[None, :]
            )
            along = along_cm / one_d.DISTANCE_CM / one_d.ARCSEC_TO_RAD
            perpendicular = (
                perpendicular_cm / one_d.DISTANCE_CM / one_d.ARCSEC_TO_RAD
            )

            # Invert the same orthonormal transform used by the observed
            # major-axis profile:
            #   along = -x sin(PA) + y cos(PA)
            #   cross =  x cos(PA) + y sin(PA).
            x = -along * np.sin(pa) + perpendicular * np.cos(pa)
            y = along * np.cos(pa) + perpendicular * np.sin(pa)
            x_flat = x.ravel()
            y_flat = y.ravel()
            minimum_x = min(minimum_x, float(np.min(x_flat)))
            maximum_x = max(maximum_x, float(np.max(x_flat)))
            minimum_y = min(minimum_y, float(np.min(y_flat)))
            maximum_y = max(maximum_y, float(np.max(y_flat)))

            weights = np.full(x_flat.size, cell_weight, dtype=float)
            inside = (
                (x_flat >= x_edges[0])
                & (x_flat < x_edges[-1])
                & (y_flat >= y_edges[0])
                & (y_flat < y_edges[-1])
            )
            outside_fraction += float(np.sum(weights[~inside]))
            fraction_map += np.histogram2d(
                y_flat,
                x_flat,
                bins=(y_edges, x_edges),
                weights=weights,
            )[0]

    extent = {
        "projected_min_x_arcsec": minimum_x,
        "projected_max_x_arcsec": maximum_x,
        "projected_min_y_arcsec": minimum_y,
        "projected_max_y_arcsec": maximum_y,
    }
    return fraction_map, outside_fraction, extent


def mask_normalized_smooth(
    data: np.ndarray,
    footprint: np.ndarray,
    sigma_y: float,
    sigma_x: float,
) -> np.ndarray:
    """Apply display-only smoothing without depressing footprint edges."""
    support = footprint.astype(float)
    numerator = gaussian_filter(data * support, (sigma_y, sigma_x), mode="constant")
    denominator = gaussian_filter(support, (sigma_y, sigma_x), mode="constant")
    result = numerator / np.maximum(denominator, 1.0e-12)
    result[~footprint] = np.nan
    return result


def major_axis_profile_from_flux_map(
    flux_per_pixel: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    offset_edges: np.ndarray,
    pa_deg: float,
) -> np.ndarray:
    """Reintegrate a 2-D flux map into the pre-existing 1-D offset bins."""
    xx, yy = np.meshgrid(x, y)
    pa = np.deg2rad(pa_deg)
    along = -xx * np.sin(pa) + yy * np.cos(pa)
    integrated = np.histogram(
        along.ravel(),
        bins=offset_edges,
        weights=flux_per_pixel.ravel(),
    )[0]
    return integrated / np.diff(offset_edges)


def draw_primary_maps(
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    footprint: np.ndarray,
    observed_display: np.ndarray,
    area1_display: np.ndarray,
    area2_display: np.ndarray,
) -> None:
    """Draw only the observed map and the two area-calibrated alternatives."""
    extent = [x[0], x[-1], y[0], y[-1]]
    finite_intensity = np.concatenate(
        [
            observed_display[np.isfinite(observed_display)],
            area1_display[np.isfinite(area1_display)],
            area2_display[np.isfinite(area2_display)],
        ]
    )
    intensity_max = float(np.percentile(finite_intensity, 99.5))
    intensity_norm = Normalize(vmin=0.0, vmax=intensity_max, clip=True)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 5.35),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panels = [
        (observed_display, "Observed 18 $\\mu$m map (unchanged)"),
        (area1_display, "Area 1-derived whole-nebula model"),
        (area2_display, "Area 2-derived whole-nebula model"),
    ]
    pa = np.deg2rad(PA_DEG)
    distance = np.linspace(-9.0, 9.0, 101)
    axis_x = -distance * np.sin(pa)
    axis_y = distance * np.cos(pa)
    images: list[matplotlib.image.AxesImage] = []
    for axis, (data, title) in zip(axes, panels, strict=True):
        images.append(
            axis.imshow(
                data,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap="magma",
                norm=intensity_norm,
                aspect="equal",
            )
        )
        axis.contour(
            x,
            y,
            footprint.astype(float),
            levels=[0.5],
            colors="0.55",
            linewidths=0.45,
        )
        axis.plot(axis_x, axis_y, color="white", lw=0.8, ls="--", alpha=0.85)
        axis.plot(0.0, 0.0, marker="+", color="white", ms=7, mew=1.0)
        axis.set_title(title, fontsize=11)
        axis.set_xlim(x[0], x[-1])
        axis.set_ylim(y[0], y[-1])
        axis.tick_params(direction="in", top=True, right=True, labelsize=8)
        axis.set_xlabel("R.A.-display offset (arcsec; right positive)")
    axes[0].set_ylabel("Declination offset (arcsec)")

    intensity_colorbar = fig.colorbar(
        images[0],
        ax=axes,
        location="bottom",
        shrink=0.72,
        pad=0.035,
    )
    intensity_colorbar.set_label("18 $\\mu$m surface brightness (Jy arcsec$^{-2}$)")
    fig.suptitle(
        "Observed 18 $\\mu$m emission and Smith-distributed lobe predictions\n"
        "Hard source-latitude cutoff $|\\lambda|<10^\\circ$ applied only to the two model panels",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=240, facecolor="white")
    plt.close(fig)


def draw_residual_maps(
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    footprint: np.ndarray,
    residual1_display: np.ndarray,
    residual2_display: np.ndarray,
) -> None:
    """Draw residuals separately so they do not complicate the principal figure."""
    extent = [x[0], x[-1], y[0], y[-1]]
    finite_residual = np.concatenate(
        [
            residual1_display[np.isfinite(residual1_display)],
            residual2_display[np.isfinite(residual2_display)],
        ]
    )
    residual_limit = float(np.percentile(np.abs(finite_residual), 99.0))
    residual_norm = TwoSlopeNorm(
        vmin=-residual_limit,
        vcenter=0.0,
        vmax=residual_limit,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 5.35),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panels = [
        (residual1_display, "Observed $-$ Area 1-derived model"),
        (residual2_display, "Observed $-$ Area 2-derived model"),
    ]
    pa = np.deg2rad(PA_DEG)
    distance = np.linspace(-9.0, 9.0, 101)
    axis_x = -distance * np.sin(pa)
    axis_y = distance * np.cos(pa)
    images: list[matplotlib.image.AxesImage] = []
    for axis, (data, title) in zip(axes, panels, strict=True):
        images.append(
            axis.imshow(
                data,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap="RdBu_r",
                norm=residual_norm,
                aspect="equal",
            )
        )
        axis.contour(
            x,
            y,
            footprint.astype(float),
            levels=[0.5],
            colors="0.45",
            linewidths=0.45,
        )
        axis.plot(axis_x, axis_y, color="black", lw=0.8, ls="--", alpha=0.8)
        axis.plot(0.0, 0.0, marker="+", color="black", ms=7, mew=1.0)
        axis.set_title(title, fontsize=11)
        axis.set_xlim(x[0], x[-1])
        axis.set_ylim(y[0], y[-1])
        axis.tick_params(direction="in", top=True, right=True, labelsize=8)
        axis.set_xlabel("R.A.-display offset (arcsec; right positive)")
    axes[0].set_ylabel("Declination offset (arcsec)")
    colorbar = fig.colorbar(
        images[0],
        ax=axes,
        location="bottom",
        shrink=0.72,
        pad=0.035,
    )
    colorbar.set_label(
        "Residual surface brightness: observed $-$ model (Jy arcsec$^{-2}$)"
    )
    fig.suptitle(
        "Residual diagnostics for the Smith-distributed lobe predictions",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=240, facecolor="white")
    plt.close(fig)


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameter-summary",
        type=Path,
        default=TWO_PHASE_RESULTS / "parameter_summary.csv",
    )
    parser.add_argument(
        "--mass-summary",
        type=Path,
        default=TWO_PHASE_RESULTS / "mass_summary.csv",
    )
    parser.add_argument("--area1-data", type=Path, default=MODEL_INPUT_DIR / "lobe.edge.data")
    parser.add_argument("--area2-data", type=Path, default=MODEL_INPUT_DIR / "lobeedgedata2")
    parser.add_argument(
        "--mass-curve",
        type=Path,
        default=SMITH_GEOMETRY_DIR / "Mass-per-SolidAngle.csv",
    )
    parser.add_argument(
        "--velocity-curve",
        type=Path,
        default=SMITH_GEOMETRY_DIR / "v_vs_theta.csv",
    )
    parser.add_argument(
        "--observed-map",
        type=Path,
        default=OBSERVED_DATA_DIR / "i18_map.npz",
    )
    parser.add_argument(
        "--observed-decisions",
        type=Path,
        default=root
        / "Infrared-three-colors-from-Smith"
        / "outputs"
        / "decisions.json",
    )
    parser.add_argument(
        "--observed-profile",
        type=Path,
        default=root
        / "Infrared-three-colors-from-Smith"
        / "outputs"
        / "i18_major_axis_profile.csv",
    )
    parser.add_argument("--latitude-min-deg", type=float, default=10.0)
    parser.add_argument("--pa-deg", type=float, default=PA_DEG)
    parser.add_argument(
        "--display-smooth-fwhm-arcsec",
        type=float,
        default=DISPLAY_SMOOTH_FWHM_ARCSEC,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "area_flux_per_column_2d_cutoff10_output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = one_d.read_parameter_medians(args.parameter_summary)
    masses = one_d.read_latest_corrected_masses(args.mass_summary)
    calibrations = one_d.build_calibrations(
        parameters,
        masses,
        one_d.nearest_flux_measurement(args.area1_data),
        one_d.nearest_flux_measurement(args.area2_data),
    )
    mass_latitude, mass_q = one_d.read_named_curve(args.mass_curve)
    velocity_latitude, velocity = one_d.read_named_curve(args.velocity_curve)
    mass_curve = PchipInterpolator(mass_latitude, mass_q, extrapolate=False)
    velocity_curve = PchipInterpolator(
        velocity_latitude, velocity, extrapolate=False
    )
    latitude_result = one_d.latitude_distribution(
        mass_curve,
        calibrations,
        args.latitude_min_deg,
    )

    with np.load(args.observed_map, allow_pickle=True) as product:
        observed_original = np.asarray(product["i18_map"], dtype=float)
        footprint_original = np.asarray(product["footprint"], dtype=bool)
    decisions = json.loads(args.observed_decisions.read_text(encoding="utf-8"))
    x, y, x_edges, y_edges = angular_grid(observed_original.shape, decisions)
    observed = np.flipud(observed_original)
    footprint = np.flipud(footprint_original)
    pixel_area_arcsec2 = float(np.diff(x_edges)[0] * np.diff(y_edges)[0])
    observed_flux_per_pixel = np.where(
        footprint,
        observed * pixel_area_arcsec2,
        0.0,
    )

    fraction_map, outside_fraction, projected_extent = project_mass_fraction_to_xy(
        latitude_result,
        velocity_curve,
        x_edges,
        y_edges,
        args.pa_deg,
    )
    predicted_full = {
        area: float(calibrations[area]["predicted_full_Smith_shell_flux_jy"])
        for area in ("area1", "area2")
    }
    flux_map = {
        area: fraction_map * predicted_full[area]
        for area in ("area1", "area2")
    }
    surface_brightness = {
        area: flux_map[area] / pixel_area_arcsec2
        for area in ("area1", "area2")
    }
    residual = {
        area: np.where(
            footprint,
            observed - surface_brightness[area],
            np.nan,
        )
        for area in ("area1", "area2")
    }

    # Use the same display-only smoothing for observation, predictions, and
    # residuals.  Saved raw arrays and all integrated validation values remain
    # unsmoothed.
    dx = float(np.diff(x)[0])
    dy = float(np.diff(y)[0])
    sigma_x = args.display_smooth_fwhm_arcsec / 2.355 / dx
    sigma_y = args.display_smooth_fwhm_arcsec / 2.355 / dy
    observed_display = mask_normalized_smooth(
        observed,
        footprint,
        sigma_y,
        sigma_x,
    )
    area_display = {
        area: mask_normalized_smooth(
            surface_brightness[area],
            footprint,
            sigma_y,
            sigma_x,
        )
        for area in ("area1", "area2")
    }
    residual_display = {
        area: observed_display - area_display[area]
        for area in ("area1", "area2")
    }

    offset, _, offset_edges = one_d.read_observed_profile(args.observed_profile)
    existing_profile_data = np.genfromtxt(
        Path(__file__).resolve().parent
        / "area_flux_per_column_profiles_cutoff10_output"
        / "whole_nebula_vs_area_calibrated_profiles.csv",
        delimiter=",",
        names=True,
    )
    validation: dict[str, dict[str, float]] = {}
    recovered_profiles: dict[str, np.ndarray] = {}
    for area in ("area1", "area2"):
        recovered = major_axis_profile_from_flux_map(
            flux_map[area],
            x,
            y,
            offset_edges,
            args.pa_deg,
        )
        reference = np.asarray(
            existing_profile_data[f"{area}_derived_flux_jy_per_arcsec"],
            dtype=float,
        )
        recovered_profiles[area] = recovered
        validation[area] = {
            "map_integral_jy": float(np.sum(flux_map[area])),
            "expected_cutoff_integral_jy": float(
                np.sum(np.asarray(latitude_result[f"{area}_flux_jy_per_bin"]))
            ),
            "flux_outside_observed_xy_grid_jy": float(
                outside_fraction * predicted_full[area]
            ),
            "flux_outside_observed_footprint_jy": float(
                np.sum(flux_map[area][~footprint])
            ),
            "major_axis_profile_integral_jy": float(
                np.sum(recovered * np.diff(offset_edges))
            ),
            "major_axis_profile_rmse_jy_per_arcsec": float(
                np.sqrt(np.mean((recovered - reference) ** 2))
            ),
            "major_axis_profile_rmse_fraction_of_reference_peak": float(
                np.sqrt(np.mean((recovered - reference) ** 2))
                / np.max(reference)
            ),
        }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    map_npz = output_dir / "observed_and_area_models_2d.npz"
    primary_figure_png = output_dir / "observed_and_area_models_2d.png"
    residual_figure_png = output_dir / "observed_minus_area_models_residuals_2d.png"
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "integrated_flux_and_residual_summary.csv"
    validation_csv = output_dir / "major_axis_reprojection_validation.csv"

    np.savez_compressed(
        map_npz,
        x_ra_display_offset_arcsec=x,
        y_declination_offset_arcsec=y,
        footprint=footprint,
        observed_jy_per_arcsec2=observed,
        area1_model_jy_per_arcsec2=surface_brightness["area1"],
        area2_model_jy_per_arcsec2=surface_brightness["area2"],
        observed_minus_area1_jy_per_arcsec2=residual["area1"],
        observed_minus_area2_jy_per_arcsec2=residual["area2"],
        observed_display_smoothed_jy_per_arcsec2=observed_display,
        area1_display_smoothed_jy_per_arcsec2=area_display["area1"],
        area2_display_smoothed_jy_per_arcsec2=area_display["area2"],
        observed_minus_area1_display_smoothed_jy_per_arcsec2=residual_display[
            "area1"
        ],
        observed_minus_area2_display_smoothed_jy_per_arcsec2=residual_display[
            "area2"
        ],
        area1_recovered_major_axis_jy_per_arcsec=recovered_profiles["area1"],
        area2_recovered_major_axis_jy_per_arcsec=recovered_profiles["area2"],
        major_axis_offset_arcsec=offset,
    )

    draw_primary_maps(
        primary_figure_png,
        x,
        y,
        footprint,
        observed_display,
        area_display["area1"],
        area_display["area2"],
    )
    draw_residual_maps(
        residual_figure_png,
        x,
        y,
        footprint,
        residual_display["area1"],
        residual_display["area2"],
    )

    observed_total = float(np.sum(observed_flux_per_pixel))
    summary_rows: list[dict[str, float | str]] = []
    for area in ("area1", "area2"):
        model_total = float(np.sum(flux_map[area]))
        model_inside_footprint = float(np.sum(flux_map[area][footprint]))
        observed_minus_model_inside = float(
            np.sum(
                (observed - surface_brightness[area])[footprint]
                * pixel_area_arcsec2
            )
        )
        positive_residual = float(
            np.sum(
                np.clip(
                    (observed - surface_brightness[area])[footprint],
                    0.0,
                    None,
                )
                * pixel_area_arcsec2
            )
        )
        negative_residual = float(
            np.sum(
                np.clip(
                    (observed - surface_brightness[area])[footprint],
                    None,
                    0.0,
                )
                * pixel_area_arcsec2
            )
        )
        summary_rows.append(
            {
                "calibration": "Area 1" if area == "area1" else "Area 2",
                "observed_map_integral_jy": observed_total,
                "model_integral_full_xy_grid_jy": model_total,
                "model_integral_inside_observed_footprint_jy": model_inside_footprint,
                "observed_minus_model_inside_footprint_jy": observed_minus_model_inside,
                "positive_residual_integral_jy": positive_residual,
                "negative_residual_integral_jy": negative_residual,
                "model_fraction_of_observed_inside_footprint": model_inside_footprint
                / observed_total,
            }
        )
    write_summary_csv(summary_csv, summary_rows)

    write_summary_csv(
        validation_csv,
        [
            {"calibration": "Area 1", **validation["area1"]},
            {"calibration": "Area 2", **validation["area2"]},
        ],
    )

    summary = {
        "method": "empirical flux per aperture dust mass distributed by the Smith dM/dOmega curve and projected into the native observed x-y grid",
        "observed_map_status": "unchanged provisional figure-derived 18-micron morphology",
        "latitude_cutoff_application": "models only; observed map unchanged",
        "minimum_model_absolute_latitude_deg": float(args.latitude_min_deg),
        "retained_smith_mass_and_flux_fraction": float(
            latitude_result["retained_mass_fraction"]
        ),
        "display_only_smoothing_fwhm_arcsec": float(
            args.display_smooth_fwhm_arcsec
        ),
        "pixel_area_arcsec2": pixel_area_arcsec2,
        "observed_map_integrated_flux_jy": observed_total,
        "area1": summary_rows[0] | validation["area1"],
        "area2": summary_rows[1] | validation["area2"],
        "projected_extent": projected_extent,
        "files": {
            "primary_figure": str(primary_figure_png),
            "residual_diagnostic_figure": str(residual_figure_png),
            "arrays": str(map_npz),
            "integrated_summary": str(summary_csv),
            "one_dimensional_reprojection_validation": str(validation_csv),
        },
        "interpretation": "positive residual means observed flux not supplied by the chosen Smith-distributed lobe calibration; it is not by itself a mass map",
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Observed map integral: {observed_total:.6f} Jy")
    for area in ("area1", "area2"):
        print(
            f"{area}: map={validation[area]['map_integral_jy']:.6f} Jy; "
            f"expected={validation[area]['expected_cutoff_integral_jy']:.6f} Jy; "
            f"outside grid={validation[area]['flux_outside_observed_xy_grid_jy']:.6e} Jy; "
            f"1-D reprojection RMSE/peak="
            f"{validation[area]['major_axis_profile_rmse_fraction_of_reference_peak']:.6%}"
        )
    print(primary_figure_png)
    print(residual_figure_png)
    print(map_npz)
    print(summary_json)


if __name__ == "__main__":
    main()
