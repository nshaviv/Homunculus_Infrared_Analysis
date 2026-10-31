#!/usr/bin/env python3
"""Build total 18-micron emission and whole-nebula model spectra.

The spatial result is the complete two-dimensional integral over the observed
Homunculus or over either modeled image.  Infinitesimal slices may be used to
describe that integral, but no slice coordinate remains in the final result:
each map produces one total flux density in Jy.

The spectra use exactly the same two independent universal-fraction models as
``build_universal_fraction_whole_nebula_flux.py``.  Smith dM/dOmega fixes the
latitude shape, the chosen aperture fixes each phase-column normalization,
temperatures scale as R^-0.4, Mie opacity is recomputed at every wavelength,
and rear-wall emission is attenuated by the independently overlapping front
wall.  No spectral flux renormalization is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

import build_universal_fraction_whole_nebula_flux as universal
from build_front_effective_temperature_2d import read_parameter_medians
from build_smith_surface_effective_temperature_2d import (
    apertures_in_sky_coordinates,
    read_velocity_curve,
)
import two_phase_aligned as physical_model


PA_DEG = 132.0
PROFILE_BIN_WIDTH_ARCSEC = 0.10
JY_CGS = 1.0e-23
ARCSEC2_SR = (np.pi / (180.0 * 3600.0)) ** 2


def projected_coordinates(
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    pa_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates along and perpendicular to the adopted major axis."""
    xx, yy = np.meshgrid(x_arcsec, y_arcsec)
    pa = np.deg2rad(pa_deg)
    along = -xx * np.sin(pa) + yy * np.cos(pa)
    perpendicular = xx * np.cos(pa) + yy * np.sin(pa)
    return along, perpendicular


def profile_parallel_to_major_axis(
    surface_brightness_jy_arcsec2: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    bin_width_arcsec: float,
    footprint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate narrow major-axis-parallel slices and retain their offsets."""
    _, perpendicular = projected_coordinates(x_arcsec, y_arcsec, PA_DEG)
    low = np.floor(np.nanmin(perpendicular) / bin_width_arcsec) * bin_width_arcsec
    high = np.ceil(np.nanmax(perpendicular) / bin_width_arcsec) * bin_width_arcsec
    edges = np.arange(low, high + 0.5 * bin_width_arcsec, bin_width_arcsec)

    pixel_area_arcsec2 = float(
        abs(np.median(np.diff(x_arcsec)) * np.median(np.diff(y_arcsec)))
    )
    valid = np.isfinite(surface_brightness_jy_arcsec2)
    if footprint is not None:
        valid &= footprint
    flux_per_pixel_jy = np.where(
        valid,
        surface_brightness_jy_arcsec2 * pixel_area_arcsec2,
        0.0,
    )
    flux_per_bin_jy = np.histogram(
        perpendicular.ravel(),
        bins=edges,
        weights=flux_per_pixel_jy.ravel(),
    )[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, flux_per_bin_jy / np.diff(edges), edges


def draw_parallel_slice_profile(
    path: Path,
    offset: np.ndarray,
    observed: np.ndarray,
    area1: np.ndarray,
    area2: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 6.1), constrained_layout=True)
    axis.plot(offset, observed, color="#9f2f1f", lw=2.4, label="Observed whole nebula")
    axis.plot(offset, area1, color="#126782", lw=2.1, label="Area 1-derived model")
    axis.plot(offset, area2, color="#6f3f8f", lw=2.1, label="Area 2-derived model")
    axis.axvline(0.0, color="0.35", ls="--", lw=1.0)
    axis.set(
        xlabel="Perpendicular offset from the major axis (arcsec)",
        ylabel=r"Flux integrated along each parallel slice (Jy arcsec$^{-1}$)",
        title=r"18 $\mu$m emission in slices parallel to the major axis",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def draw_total_flux_comparison(
    path: Path,
    observed_jy: float,
    area1_jy: float,
    area2_jy: float,
) -> None:
    labels = ["Observed\nwhole nebula", "Area 1-anchored\nmodel", "Area 2-anchored\nmodel"]
    values = np.asarray([observed_jy, area1_jy, area2_jy], dtype=float)
    colors = ["#9f2f1f", "#126782", "#6f3f8f"]
    figure, axis = plt.subplots(figsize=(9.2, 6.1), constrained_layout=True)
    bars = axis.bar(labels, values, color=colors, width=0.62)
    axis.set(
        ylabel=r"Spatially integrated flux density $F_\nu$ (Jy)",
        title=r"Total 18 $\mu$m emission integrated over the whole Homunculus",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(0.0, 1.16 * float(np.max(values)))
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.025 * float(np.max(values)),
            f"{value:,.0f} Jy",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def grain_size_latitude_curves(
    config: dict[str, float | str],
    latitude_grid: np.ndarray,
    mass_curve: PchipInterpolator,
    radius_curve: PchipInterpolator,
    q_density_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(radius_curve(latitude_grid), dtype=float)
    radial_column_shape = np.asarray(mass_curve(latitude_grid), dtype=float) / radius**2
    anchor_latitude = float(config["anchor_latitude_deg"])
    anchor_shape = float(mass_curve(anchor_latitude)) / float(
        radius_curve(anchor_latitude)
    ) ** 2
    ratio = radial_column_shape / anchor_shape
    dense = float(config["a_dense_anchor_micron"]) * ratio**q_density_size
    rare = float(config["a_rare_anchor_micron"]) * ratio**q_density_size
    return (
        np.clip(dense, physical_model.A_MIN_MICRON, physical_model.A_MAX_MICRON),
        np.clip(rare, physical_model.A_MIN_MICRON, physical_model.A_MAX_MICRON),
    )


def integrated_model_spectrum(
    config: dict[str, float | str],
    spatial: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    mass_curve: PchipInterpolator,
    velocity_curve: PchipInterpolator,
    p_size: float,
    q_density_size: float,
    wavelength_micron: np.ndarray,
    pixel_area_arcsec2: float,
) -> np.ndarray:
    """Integrate the two-wall radiative-transfer model over the image plane."""
    latitude_grid = np.arange(0.25, 90.0, 0.5)
    radius_grid = (
        np.asarray(velocity_curve(latitude_grid), dtype=float)
        * 1.0e5
        * universal.one_d.BALLISTIC_AGE_YR
        * universal.one_d.SECONDS_PER_YEAR
    )
    radius_curve = PchipInterpolator(latitude_grid, radius_grid, extrapolate=True)
    dense_size, rare_size = grain_size_latitude_curves(
        config,
        latitude_grid,
        mass_curve,
        radius_curve,
        q_density_size,
    )
    corundum = float(config["corundum_fraction"])
    dense_kappa = np.vstack(
        [
            physical_model.powerlaw_mixed_kappa(
                p_size, size, wavelength_micron, corundum
            )
            for size in dense_size
        ]
    )
    rare_kappa = np.vstack(
        [
            physical_model.powerlaw_mixed_kappa(
                p_size, size, wavelength_micron, corundum
            )
            for size in rare_size
        ]
    )

    front_latitude = geometry["front_abs_latitude_deg"]
    back_latitude = geometry["back_abs_latitude_deg"]
    valid = (
        np.isfinite(front_latitude)
        & np.isfinite(back_latitude)
        & np.isfinite(spatial["sigma_dense_front_los_g_cm2"])
        & np.isfinite(spatial["sigma_dense_back_los_g_cm2"])
    )
    front_latitude = front_latitude[valid]
    back_latitude = back_latitude[valid]
    sigma_dense_front = spatial["sigma_dense_front_los_g_cm2"][valid]
    sigma_dense_back = spatial["sigma_dense_back_los_g_cm2"][valid]
    sigma_rare_front = spatial["sigma_rare_front_los_g_cm2"][valid]
    sigma_rare_back = spatial["sigma_rare_back_los_g_cm2"][valid]
    temperature_dense_front = spatial["T_dense_front_K"][valid]
    temperature_dense_back = spatial["T_dense_back_K"][valid]
    temperature_rare_front = spatial["T_rare_front_K"][valid]
    temperature_rare_back = spatial["T_rare_back_K"][valid]
    fraction = float(config["fraction"])

    result_jy = np.empty_like(wavelength_micron, dtype=float)
    pixel_solid_angle_sr = pixel_area_arcsec2 * ARCSEC2_SR
    for index, wavelength in enumerate(wavelength_micron):
        kappa_dense_front = np.interp(
            front_latitude, latitude_grid, dense_kappa[:, index]
        )
        kappa_dense_back = np.interp(
            back_latitude, latitude_grid, dense_kappa[:, index]
        )
        kappa_rare_front = np.interp(
            front_latitude, latitude_grid, rare_kappa[:, index]
        )
        kappa_rare_back = np.interp(
            back_latitude, latitude_grid, rare_kappa[:, index]
        )
        tau_dense_front = kappa_dense_front * sigma_dense_front
        tau_dense_back = kappa_dense_back * sigma_dense_back
        tau_rare_front = kappa_rare_front * sigma_rare_front
        tau_rare_back = kappa_rare_back * sigma_rare_back

        front_emission = (
            fraction
            * universal.planck_nu(wavelength, temperature_dense_front)
            * -np.expm1(-tau_dense_front)
            + (1.0 - fraction)
            * universal.planck_nu(wavelength, temperature_rare_front)
            * -np.expm1(-tau_rare_front)
        )
        front_transmission = fraction * np.exp(-tau_dense_front) + (
            1.0 - fraction
        ) * np.exp(-tau_rare_front)
        back_emission = (
            fraction
            * universal.planck_nu(wavelength, temperature_dense_back)
            * -np.expm1(-tau_dense_back)
            + (1.0 - fraction)
            * universal.planck_nu(wavelength, temperature_rare_back)
            * -np.expm1(-tau_rare_back)
        )
        intensity = front_emission + front_transmission * back_emission
        result_jy[index] = float(np.sum(intensity) * pixel_solid_angle_sr / JY_CGS)
    return result_jy


def draw_spectrum(
    path: Path,
    observed_wavelength: np.ndarray,
    observed_flux_jy: np.ndarray,
    model_wavelength: np.ndarray,
    area1_flux_jy: np.ndarray,
    area2_flux_jy: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 6.1), constrained_layout=True)
    axis.plot(
        observed_wavelength,
        observed_flux_jy,
        color="#9f2f1f",
        lw=2.2,
        label="Observed whole nebula",
    )
    axis.plot(
        model_wavelength,
        area1_flux_jy,
        color="#126782",
        lw=2.1,
        label="Area 1-derived model",
    )
    axis.plot(
        model_wavelength,
        area2_flux_jy,
        color="#6f3f8f",
        lw=2.1,
        label="Area 2-derived model",
    )
    axis.axvline(100.0, color="0.35", lw=1.0, ls="--")
    axis.text(
        100.0,
        0.04,
        "opacity limit",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        xlabel=r"Wavelength ($\mu$m)",
        ylabel=r"Whole-nebula flux density $F_\nu$ (Jy)",
        title="Observed and independently anchored whole-nebula spectra",
    )
    axis.grid(alpha=0.22, which="both")
    axis.legend(frameon=False)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-product",
        type=Path,
        default=root
        / "universal_fraction_whole_nebula_flux_beta1_output"
        / "universal_fraction_whole_nebula_flux.npz",
    )
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        default=root / "universal_fraction_whole_nebula_flux_beta1_output",
    )
    parser.add_argument(
        "--parameter-summary",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results" / "whole_nebula" / "parameters" / "beta_p1p00"
        / "parameter_summary.csv",
    )
    parser.add_argument(
        "--mass-curve",
        type=Path,
        default=root / "homunculus-IR-smith" / "Mass-per-SolidAngle.csv",
    )
    parser.add_argument(
        "--velocity-curve",
        type=Path,
        default=root / "homunculus-IR-smith" / "v_vs_theta.csv",
    )
    parser.add_argument(
        "--observed-spectrum",
        type=Path,
        default=root / "homunculus_obs_flux_cgs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "parallel_major_axis_slices_and_spectrum_beta1_output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(args.map_product.expanduser().resolve()) as product:
        x = np.asarray(product["x"], dtype=float)
        y = np.asarray(product["y"], dtype=float)
        footprint = np.asarray(product["footprint"], dtype=bool)
        observed_map = np.asarray(product["observed_jy_arcsec2"], dtype=float)
        area1_map = np.asarray(product["area1_jy_arcsec2"], dtype=float)
        area2_map = np.asarray(product["area2_jy_arcsec2"], dtype=float)

    pixel_area_arcsec2 = float(abs(np.median(np.diff(x)) * np.median(np.diff(y))))
    map_totals = {
        "observed": float(
            np.sum(np.where(footprint, observed_map, 0.0)) * pixel_area_arcsec2
        ),
        "area1": float(np.sum(area1_map) * pixel_area_arcsec2),
        "area2": float(np.sum(area2_map) * pixel_area_arcsec2),
    }
    total_flux_png = output / "whole_homunculus_integrated_18um_flux.png"
    draw_total_flux_comparison(
        total_flux_png,
        map_totals["observed"],
        map_totals["area1"],
        map_totals["area2"],
    )
    total_flux_csv = output / "whole_homunculus_integrated_18um_flux.csv"
    with total_flux_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "integrated_18um_flux_jy", "fraction_of_observed"])
        for name in ("observed", "area1", "area2"):
            writer.writerow(
                [name, map_totals[name], map_totals[name] / map_totals["observed"]]
            )

    medians = read_parameter_medians(args.parameter_summary.expanduser().resolve())
    theta = np.asarray([medians[name] for name in physical_model.PARAMETER_NAMES])
    state = physical_model.derive_state(theta)
    mass_latitude, mass_value = universal.read_named_curve(
        args.mass_curve.expanduser().resolve()
    )
    mass_curve = PchipInterpolator(mass_latitude, mass_value, extrapolate=False)
    velocity_curve = read_velocity_curve(args.velocity_curve.expanduser().resolve())
    geometry = universal.load_geometry(args.geometry_dir.expanduser().resolve())
    apertures = apertures_in_sky_coordinates(velocity_curve)
    configs = {
        name: universal.area_configuration(name, state)
        for name in ("area1", "area2")
    }
    spatial: dict[str, dict[str, np.ndarray]] = {}
    for name, aperture_name in (("area1", "Area 1"), ("area2", "Area 2")):
        spatial[name], _ = universal.build_area_model(
            configs[name],
            geometry,
            apertures[aperture_name],
            x,
            y,
            mass_curve,
            velocity_curve,
            state["p_size"],
            state["q_density_size"],
        )

    wavelength_model = np.unique(
        np.r_[np.logspace(np.log10(2.0), np.log10(100.0), 180), 18.0]
    )
    spectra = {
        name: integrated_model_spectrum(
            configs[name],
            spatial[name],
            geometry,
            mass_curve,
            velocity_curve,
            state["p_size"],
            state["q_density_size"],
            wavelength_model,
            pixel_area_arcsec2,
        )
        for name in ("area1", "area2")
    }

    observed_table = np.genfromtxt(
        args.observed_spectrum.expanduser().resolve(), delimiter=",", names=True
    )
    observed_wavelength = np.asarray(
        observed_table[observed_table.dtype.names[0]], dtype=float
    )
    observed_flux_jy = (
        np.asarray(observed_table[observed_table.dtype.names[1]], dtype=float) / JY_CGS
    )
    order = np.argsort(observed_wavelength)
    observed_wavelength = observed_wavelength[order]
    observed_flux_jy = observed_flux_jy[order]

    spectrum_png = output / "observed_vs_area_anchored_total_spectra.png"
    draw_spectrum(
        spectrum_png,
        observed_wavelength,
        observed_flux_jy,
        wavelength_model,
        spectra["area1"],
        spectra["area2"],
    )
    spectrum_csv = output / "area_anchored_total_model_spectra.csv"
    with spectrum_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "wavelength_micron",
                "area1_derived_whole_nebula_flux_jy",
                "area2_derived_whole_nebula_flux_jy",
            ]
        )
        writer.writerows(
            zip(wavelength_model, spectra["area1"], spectra["area2"])
        )

    index_18 = int(np.where(wavelength_model == 18.0)[0][0])
    spectral_totals_18 = {
        "area1": float(spectra["area1"][index_18]),
        "area2": float(spectra["area2"][index_18]),
    }
    validations = {
        f"{name}_spectrum_18um_relative_error": abs(
            spectral_totals_18[name] / map_totals[name] - 1.0
        )
        for name in ("area1", "area2")
    }
    if max(validations.values()) > 5.0e-10:
        raise RuntimeError(f"Projection/spectral consistency check failed: {validations}")

    observed_18_jy = float(
        np.interp(18.0, observed_wavelength, observed_flux_jy)
    )
    summary = {
        "spatial_integral_definition": (
            "sum the 18-micron surface brightness over every pixel in the "
            "whole observed or modeled Homunculus; no slice coordinate remains"
        ),
        "map_integrals_jy": map_totals,
        "spectrum_18um_model_jy": spectral_totals_18,
        "observed_spectrum_18um_interpolated_jy": observed_18_jy,
        "observed_spectrum_source": (
            "homunculus_obs_flux_cgs.csv; Smith et al. (2003) Figure 8 digitization"
        ),
        "temperature_emissivity_beta": universal.BETA,
        "temperature_distance_exponent": universal.TEMPERATURE_EXPONENT,
        "model_spectrum_range_micron": [2.0, 100.0],
        "model_flux_renormalized": False,
        "validations": validations,
        "files": {
            "integrated_18um_flux_png": str(total_flux_png),
            "integrated_18um_flux_csv": str(total_flux_csv),
            "spectrum_png": str(spectrum_png),
            "model_spectrum_csv": str(spectrum_csv),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(total_flux_png)
    print(spectrum_png)
    print(summary_path)
    print(json.dumps(summary["validations"], indent=2))


if __name__ == "__main__":
    main()
