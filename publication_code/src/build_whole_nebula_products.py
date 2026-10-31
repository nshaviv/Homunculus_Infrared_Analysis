#!/usr/bin/env python3
"""Propagate the fixed-beta MCMC sweep through the whole-nebula model.

For each requested beta this program builds two independent alternatives:

* the Smith latitude distribution normalized by the Area-1 MCMC result;
* the same distribution normalized by the Area-2 MCMC result.

The alternatives are never added.  Each uses its own fitted covering fraction,
phase columns, temperatures, grain sizes, composition, and Mie opacity.  Beta
changes only T(R)=T_anchor*(R_anchor/R)**[2/(4+beta)].  Model crossings below
the default absolute-latitude cutoff of 10 degrees emit no flux.  The observed
18-micron map is never cut or rescaled.

The program creates, for every beta: 18-micron maps and residuals; profiles in
both projected directions; whole-nebula spectra; and wavelength-versus-major-
axis spectral profiles after integration across the minor-axis direction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

import build_area_flux_per_column_latitude_profiles as one_d
from build_area_flux_per_column_2d_maps import (
    angular_grid,
    major_axis_profile_from_flux_map,
)
import build_parallel_major_axis_slices_and_total_spectrum as spectral_tools
from build_smith_surface_effective_temperature_2d import (
    aperture_mask,
    apertures_in_sky_coordinates,
    read_velocity_curve,
)
import build_universal_fraction_whole_nebula_flux as universal
import two_phase_aligned as physical_model
import two_phase_fit as accepted_model

from project_paths import (
    MODEL_INPUT_DIR,
    OBSERVED_DATA_DIR,
    SMITH_GEOMETRY_DIR,
    WHOLE_NEBULA_GEOMETRY,
    WHOLE_NEBULA_RESULTS,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BETAS = (0.0, 1.0)
PA_DEG = 132.0
JY_CGS = 1.0e-23
ARCSEC2_SR = (np.pi / (180.0 * 3600.0)) ** 2
PROFILE_BIN_WIDTH_ARCSEC = 0.10
OBSERVED_COLOR = "#9f2f1f"
AREA1_COLOR = "#126782"
AREA2_COLOR = "#6f3f8f"


def beta_label(beta: float) -> str:
    sign = "m" if beta < 0.0 else "p"
    return f"beta_{sign}{abs(beta):.2f}".replace(".", "p")


def bin_edges_from_range(values: np.ndarray, width: float) -> np.ndarray:
    low = np.floor(np.nanmin(values) / width) * width
    high = np.ceil(np.nanmax(values) / width) * width
    return np.arange(low, high + 0.5 * width, width)


def wavelength_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    log_centres = np.log(centres)
    edges = np.empty(len(centres) + 1)
    edges[1:-1] = 0.5 * (log_centres[:-1] + log_centres[1:])
    edges[0] = log_centres[0] - 0.5 * (log_centres[1] - log_centres[0])
    edges[-1] = log_centres[-1] + 0.5 * (
        log_centres[-1] - log_centres[-2]
    )
    return np.exp(edges)


def read_parameter_vector(path: Path, column: str) -> np.ndarray:
    values: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values[row["parameter"]] = float(row[column])
    missing = [name for name in physical_model.PARAMETER_NAMES if name not in values]
    if missing:
        raise ValueError(f"Missing parameters in {path}: {missing}")
    return np.asarray([values[name] for name in physical_model.PARAMETER_NAMES])


def read_observed_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    wavelength = np.asarray(table[table.dtype.names[0]], dtype=float)
    flux_jy = np.asarray(table[table.dtype.names[1]], dtype=float) / JY_CGS
    keep = np.isfinite(wavelength) & np.isfinite(flux_jy) & (flux_jy > 0.0)
    order = np.argsort(wavelength[keep])
    return wavelength[keep][order], flux_jy[keep][order]


def write_csv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def apply_latitude_cutoff(
    config: dict[str, float | str],
    spatial: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    cutoff_deg: float,
) -> dict[str, np.ndarray]:
    """Zero each shell crossing independently below the model-only cutoff."""
    result = {name: np.array(value, copy=True) for name, value in spatial.items()}
    front_active = (
        np.isfinite(geometry["front_abs_latitude_deg"])
        & (geometry["front_abs_latitude_deg"] >= cutoff_deg)
    )
    back_active = (
        np.isfinite(geometry["back_abs_latitude_deg"])
        & (geometry["back_abs_latitude_deg"] >= cutoff_deg)
    )
    for phase in ("dense", "rare"):
        for side, active in (("front", front_active), ("back", back_active)):
            sigma_name = f"sigma_{phase}_{side}_los_g_cm2"
            tau_name = f"tau_{phase}_{side}"
            result[sigma_name] = np.where(active, result[sigma_name], 0.0)
            result[tau_name] = np.where(active, result[tau_name], 0.0)

    result["front_crossing_retained"] = front_active
    result["back_crossing_retained"] = back_active
    result["surface_brightness_jy_arcsec2"] = surface_brightness_from_spatial(
        config, result
    )
    return result


def surface_brightness_from_spatial(
    config: dict[str, float | str],
    spatial: dict[str, np.ndarray],
) -> np.ndarray:
    """Re-evaluate the 18-micron independent-overlap transfer equation."""
    fraction = float(config["fraction"])
    front_active = spatial["front_crossing_retained"]
    back_active = spatial["back_crossing_retained"]
    dense_front = np.where(
        front_active,
        fraction
        * universal.planck_nu(18.0, spatial["T_dense_front_K"])
        * -np.expm1(-spatial["tau_dense_front"]),
        0.0,
    )
    rare_front = np.where(
        front_active,
        (1.0 - fraction)
        * universal.planck_nu(18.0, spatial["T_rare_front_K"])
        * -np.expm1(-spatial["tau_rare_front"]),
        0.0,
    )
    front_transmission = np.where(
        front_active,
        fraction * np.exp(-spatial["tau_dense_front"])
        + (1.0 - fraction) * np.exp(-spatial["tau_rare_front"]),
        1.0,
    )
    dense_back = np.where(
        back_active,
        fraction
        * universal.planck_nu(18.0, spatial["T_dense_back_K"])
        * -np.expm1(-spatial["tau_dense_back"]),
        0.0,
    )
    rare_back = np.where(
        back_active,
        (1.0 - fraction)
        * universal.planck_nu(18.0, spatial["T_rare_back_K"])
        * -np.expm1(-spatial["tau_rare_back"]),
        0.0,
    )
    intensity = dense_front + rare_front + front_transmission * (
        dense_back + rare_back
    )
    return np.nan_to_num(intensity, nan=0.0) * ARCSEC2_SR / JY_CGS


def scaled_spatial_columns(
    config: dict[str, float | str],
    spatial: dict[str, np.ndarray],
    scale: float,
) -> dict[str, np.ndarray]:
    """Scale all phase columns together and recompute nonlinear transfer."""
    result = {name: np.array(value, copy=True) for name, value in spatial.items()}
    for phase in ("dense", "rare"):
        for side in ("front", "back"):
            result[f"sigma_{phase}_{side}_los_g_cm2"] *= scale
            result[f"tau_{phase}_{side}"] *= scale
    result["surface_brightness_jy_arcsec2"] = surface_brightness_from_spatial(
        config, result
    )
    return result


def close_anchor_aperture(
    config: dict[str, float | str],
    spatial: dict[str, np.ndarray],
    aperture: np.ndarray,
    target_flux_jy: float,
) -> tuple[dict[str, np.ndarray], float, float]:
    """Solve one common column scale so the mapped aperture closes at 18 um."""
    strict_flux = float(
        np.mean(spatial["surface_brightness_jy_arcsec2"][aperture])
    )

    def residual(log10_scale: float) -> float:
        candidate = scaled_spatial_columns(config, spatial, 10.0**log10_scale)
        return float(
            np.mean(candidate["surface_brightness_jy_arcsec2"][aperture])
            - target_flux_jy
        )

    grid = np.linspace(-6.0, 6.0, 97)
    values = np.asarray([residual(value) for value in grid])
    crossings = np.where(values[:-1] * values[1:] <= 0.0)[0]
    if len(crossings) == 0:
        raise RuntimeError(
            f"Could not close {config['name']} aperture to {target_flux_jy:.6g} Jy"
        )
    roots = [
        brentq(residual, grid[index], grid[index + 1])
        for index in crossings
    ]
    chosen_log_scale = min(roots, key=abs)
    scale = 10.0**chosen_log_scale
    closed = scaled_spatial_columns(config, spatial, scale)
    achieved = float(
        np.mean(closed["surface_brightness_jy_arcsec2"][aperture])
    )
    if abs(achieved / target_flux_jy - 1.0) > 1.0e-10:
        raise RuntimeError("Aperture-closure solve did not reach its target.")
    return closed, scale, strict_flux


def profile_from_brightness(
    surface_brightness: np.ndarray,
    coordinate: np.ndarray,
    edges: np.ndarray,
    pixel_area: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    valid = np.isfinite(surface_brightness)
    if mask is not None:
        valid &= mask
    flux_per_pixel = np.where(valid, surface_brightness * pixel_area, 0.0)
    integrated = np.histogram(
        coordinate.ravel(), bins=edges, weights=flux_per_pixel.ravel()
    )[0]
    return integrated / np.diff(edges)


def grain_opacity_tables(
    config: dict[str, float | str],
    mass_curve: PchipInterpolator,
    velocity_curve: PchipInterpolator,
    p_size: float,
    q_density_size: float,
    wavelength: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitude_grid = np.arange(0.25, 90.0, 0.5)
    radius_grid = (
        np.asarray(velocity_curve(latitude_grid), dtype=float)
        * 1.0e5
        * one_d.BALLISTIC_AGE_YR
        * one_d.SECONDS_PER_YEAR
    )
    radius_curve = PchipInterpolator(latitude_grid, radius_grid, extrapolate=True)
    dense_size, rare_size = spectral_tools.grain_size_latitude_curves(
        config, latitude_grid, mass_curve, radius_curve, q_density_size
    )
    corundum = float(config["corundum_fraction"])
    dense_kappa = np.vstack(
        [
            physical_model.powerlaw_mixed_kappa(
                p_size, size, wavelength, corundum
            )
            for size in dense_size
        ]
    )
    rare_kappa = np.vstack(
        [
            physical_model.powerlaw_mixed_kappa(
                p_size, size, wavelength, corundum
            )
            for size in rare_size
        ]
    )
    return latitude_grid, dense_kappa, rare_kappa


def spectral_products(
    config: dict[str, float | str],
    spatial: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    mass_curve: PchipInterpolator,
    velocity_curve: PchipInterpolator,
    p_size: float,
    q_density_size: float,
    wavelength: np.ndarray,
    profile_indices: np.ndarray,
    along_coordinate: np.ndarray,
    perpendicular_coordinate: np.ndarray,
    major_edges: np.ndarray,
    minor_edges: np.ndarray,
    pixel_area: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total spectrum and profiles after integration across each axis."""
    latitude_grid, dense_kappa, rare_kappa = grain_opacity_tables(
        config,
        mass_curve,
        velocity_curve,
        p_size,
        q_density_size,
        wavelength,
    )
    valid = (
        np.isfinite(geometry["front_abs_latitude_deg"])
        & np.isfinite(geometry["back_abs_latitude_deg"])
        & np.isfinite(spatial["sigma_dense_front_los_g_cm2"])
        & np.isfinite(spatial["sigma_dense_back_los_g_cm2"])
    )
    front_latitude = geometry["front_abs_latitude_deg"][valid]
    back_latitude = geometry["back_abs_latitude_deg"][valid]
    sigma_dense_front = spatial["sigma_dense_front_los_g_cm2"][valid]
    sigma_dense_back = spatial["sigma_dense_back_los_g_cm2"][valid]
    sigma_rare_front = spatial["sigma_rare_front_los_g_cm2"][valid]
    sigma_rare_back = spatial["sigma_rare_back_los_g_cm2"][valid]
    temperature_dense_front = spatial["T_dense_front_K"][valid]
    temperature_dense_back = spatial["T_dense_back_K"][valid]
    temperature_rare_front = spatial["T_rare_front_K"][valid]
    temperature_rare_back = spatial["T_rare_back_K"][valid]
    along = along_coordinate[valid]
    perpendicular = perpendicular_coordinate[valid]
    fraction = float(config["fraction"])
    pixel_solid_angle = pixel_area * ARCSEC2_SR

    total = np.empty(len(wavelength), dtype=float)
    wanted = {int(index): position for position, index in enumerate(profile_indices)}
    major_profiles = np.empty((len(profile_indices), len(major_edges) - 1))
    minor_profiles = np.empty((len(profile_indices), len(minor_edges) - 1))
    for index, one_wavelength in enumerate(wavelength):
        kdf = np.interp(front_latitude, latitude_grid, dense_kappa[:, index])
        kdb = np.interp(back_latitude, latitude_grid, dense_kappa[:, index])
        krf = np.interp(front_latitude, latitude_grid, rare_kappa[:, index])
        krb = np.interp(back_latitude, latitude_grid, rare_kappa[:, index])
        tau_df = kdf * sigma_dense_front
        tau_db = kdb * sigma_dense_back
        tau_rf = krf * sigma_rare_front
        tau_rb = krb * sigma_rare_back
        front_emission = (
            fraction
            * universal.planck_nu(one_wavelength, temperature_dense_front)
            * -np.expm1(-tau_df)
            + (1.0 - fraction)
            * universal.planck_nu(one_wavelength, temperature_rare_front)
            * -np.expm1(-tau_rf)
        )
        front_transmission = fraction * np.exp(-tau_df) + (
            1.0 - fraction
        ) * np.exp(-tau_rf)
        back_emission = (
            fraction
            * universal.planck_nu(one_wavelength, temperature_dense_back)
            * -np.expm1(-tau_db)
            + (1.0 - fraction)
            * universal.planck_nu(one_wavelength, temperature_rare_back)
            * -np.expm1(-tau_rb)
        )
        flux_per_pixel = (
            (front_emission + front_transmission * back_emission)
            * pixel_solid_angle
            / JY_CGS
        )
        total[index] = float(np.sum(flux_per_pixel))
        if index in wanted:
            row = wanted[index]
            major_profiles[row] = np.histogram(
                along, bins=major_edges, weights=flux_per_pixel
            )[0] / np.diff(major_edges)
            minor_profiles[row] = np.histogram(
                perpendicular, bins=minor_edges, weights=flux_per_pixel
            )[0] / np.diff(minor_edges)
    return total, major_profiles, minor_profiles


def draw_maps(
    path: Path,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
    footprint: np.ndarray,
    observed: np.ndarray,
    area1: np.ndarray,
    area2: np.ndarray,
) -> None:
    extent = [x[0], x[-1], y[0], y[-1]]
    observed_display = np.where(footprint, observed, np.nan)
    emission_values = np.concatenate(
        [
            observed_display[np.isfinite(observed_display)],
            area1[np.isfinite(area1) & (area1 > 0.0)],
            area2[np.isfinite(area2) & (area2 > 0.0)],
        ]
    )
    emission_norm = Normalize(
        vmin=0.0, vmax=float(np.percentile(emission_values, 99.5)), clip=True
    )
    residual1 = np.where(footprint, observed - area1, np.nan)
    residual2 = np.where(footprint, observed - area2, np.nan)
    residual_limit = float(
        np.percentile(
            np.abs(
                np.concatenate(
                    [
                        residual1[np.isfinite(residual1)],
                        residual2[np.isfinite(residual2)],
                    ]
                )
            ),
            99.5,
        )
    )
    residual_norm = TwoSlopeNorm(
        vmin=-residual_limit, vcenter=0.0, vmax=residual_limit
    )
    magma = plt.get_cmap("magma").copy()
    magma.set_bad("white")
    residual_cmap = plt.get_cmap("coolwarm").copy()
    residual_cmap.set_bad("white")
    figure, axes = plt.subplots(
        2, 3, figsize=(14.0, 9.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    panels = (
        (observed_display, "Observed 18 $\mu$m", magma, emission_norm),
        (np.where(area1 > 0.0, area1, np.nan), "Area 1-anchored", magma, emission_norm),
        (residual1, "Observed − Area 1", residual_cmap, residual_norm),
        (observed_display, "Observed 18 $\mu$m", magma, emission_norm),
        (np.where(area2 > 0.0, area2, np.nan), "Area 2-anchored", magma, emission_norm),
        (residual2, "Observed − Area 2", residual_cmap, residual_norm),
    )
    emission_image = None
    residual_image = None
    for axis, (data, title, cmap, norm) in zip(axes.ravel(), panels, strict=True):
        image = axis.imshow(
            data, origin="lower", extent=extent, cmap=cmap, norm=norm,
            interpolation="nearest", aspect="equal",
        )
        if "−" in title:
            residual_image = image
        else:
            emission_image = image
        axis.contour(x, y, footprint.astype(float), levels=[0.5], colors="0.55", linewidths=0.35)
        axis.plot(0.0, 0.0, "+", color="white", ms=6, mew=1)
        axis.set_title(title)
        axis.set_xlabel("R.A.-display offset (arcsec; right positive)")
        axis.tick_params(direction="in", top=True, right=True, labelsize=8)
    axes[0, 0].set_ylabel("Declination offset (arcsec)")
    axes[1, 0].set_ylabel("Declination offset (arcsec)")
    figure.colorbar(
        emission_image, ax=axes[:, :2], location="bottom", shrink=0.70,
        pad=0.025, label=r"Surface brightness (Jy arcsec$^{-2}$)",
    )
    figure.colorbar(
        residual_image, ax=axes[:, 2], location="bottom", shrink=0.85,
        pad=0.025, label=r"Observed − model (Jy arcsec$^{-2}$)",
    )
    figure.suptitle(
        rf"18 $\mu$m whole-nebula emission; fixed $\beta={beta:g}$; model cutoff $|\lambda|<10^\circ$"
    )
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def draw_three_panel_emission(
    path: Path,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
    footprint: np.ndarray,
    observed: np.ndarray,
    area2: np.ndarray,
    area1: np.ndarray,
) -> None:
    """Draw the requested observed | Area 2 | Area 1 emission comparison."""
    extent = [x[0], x[-1], y[0], y[-1]]
    observed_display = np.where(footprint, observed, np.nan)
    area2_display = np.where(area2 > 0.0, area2, np.nan)
    area1_display = np.where(area1 > 0.0, area1, np.nan)
    finite = np.concatenate(
        [
            observed_display[np.isfinite(observed_display)],
            area2_display[np.isfinite(area2_display)],
            area1_display[np.isfinite(area1_display)],
        ]
    )
    # A shared robust ceiling prevents a few reconstructed-map outliers from
    # compressing all physically relevant structure into the darkest colors.
    norm = Normalize(vmin=0.0, vmax=float(np.percentile(finite, 99.5)), clip=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.2, 5.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panels = (
        (observed_display, r"Observed 18 $\mu$m emission"),
        (area2_display, "Area 2-anchored model"),
        (area1_display, "Area 1-anchored model"),
    )
    images = []
    contour_levels = np.linspace(0.1, 0.9, 9) * norm.vmax
    for axis, (data, title) in zip(axes, panels, strict=True):
        image = axis.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        images.append(image)
        axis.contour(
            x,
            y,
            data,
            levels=contour_levels,
            colors="white",
            linewidths=0.35,
            alpha=0.35,
        )
        axis.plot(0.0, 0.0, "+", color="cyan", ms=7, mew=1.0)
        axis.set_title(title)
        axis.set_xlabel("R.A.-display offset (arcsec; right positive)")
        axis.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel("Declination offset (arcsec)")
    colorbar = figure.colorbar(
        images[0], ax=axes, location="bottom", shrink=0.76, pad=0.035
    )
    colorbar.set_label(r"18 $\mu$m surface brightness (Jy arcsec$^{-2}$)")
    figure.suptitle(
        rf"Observed and modeled whole-nebula emission; fixed $\beta={beta:g}$; "
        rf"model cutoff $|\mathrm{{latitude}}|<10^\circ$"
    )
    figure.savefig(path, dpi=240, facecolor="white")
    plt.close(figure)


def draw_profiles(
    path: Path,
    beta: float,
    major_centres: np.ndarray,
    observed_major: np.ndarray,
    area1_major: np.ndarray,
    area2_major: np.ndarray,
    minor_centres: np.ndarray,
    observed_minor: np.ndarray,
    area1_minor: np.ndarray,
    area2_minor: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), constrained_layout=True)
    for axis, coordinate, observed, area1, area2, title, xlabel in (
        (
            axes[0], major_centres, observed_major, area1_major, area2_major,
            "Minor-axis integrated: profile along major axis",
            "Major-axis offset (arcsec; southeast positive)",
        ),
        (
            axes[1], minor_centres, observed_minor, area1_minor, area2_minor,
            "Major-axis integrated: profile across major axis",
            "Perpendicular offset from major axis (arcsec)",
        ),
    ):
        axis.plot(coordinate, observed, color=OBSERVED_COLOR, lw=2.3, label="Observed")
        axis.plot(coordinate, area1, color=AREA1_COLOR, lw=2.0, label="Area 1-anchored")
        axis.plot(coordinate, area2, color=AREA2_COLOR, lw=2.0, label="Area 2-anchored")
        axis.axvline(0.0, color="0.4", ls="--", lw=0.8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"Integrated flux per offset (Jy arcsec$^{-1}$)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(rf"18 $\mu$m projected profiles; fixed $\beta={beta:g}$")
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def log_interpolate(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.exp(np.interp(np.log(target), np.log(x), np.log(y)))


def draw_spectrum(
    path: Path,
    beta: float,
    observed_wavelength: np.ndarray,
    observed_flux: np.ndarray,
    observed_map_18: float,
    model_wavelength: np.ndarray,
    area1: np.ndarray,
    area2: np.ndarray,
) -> None:
    comparable = observed_wavelength <= model_wavelength[-1]
    observed_x = observed_wavelength[comparable]
    observed_y = observed_flux[comparable]
    area1_at_observed = log_interpolate(model_wavelength, area1, observed_x)
    area2_at_observed = log_interpolate(model_wavelength, area2, observed_x)
    figure, axes = plt.subplots(
        2, 1, figsize=(10.8, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]}, constrained_layout=True,
    )
    axes[0].plot(observed_wavelength, observed_flux, color="black", lw=2.1, label="Observed spectrum (digitized)", zorder=5)
    axes[0].scatter([18.0], [observed_map_18], marker="*", s=130, facecolors="none", edgecolors="black", linewidths=1.4, label="Observed 18 $\mu$m map integral", zorder=8)
    axes[0].plot(model_wavelength, area1, color=AREA1_COLOR, lw=2.0, label="Area 1-anchored")
    axes[0].plot(model_wavelength, area2, color=AREA2_COLOR, lw=2.0, label="Area 2-anchored")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Whole-nebula $F_\nu$ (Jy)")
    axes[0].set_title(rf"Whole-nebula spectrum; fixed $\beta={beta:g}$")
    axes[0].grid(alpha=0.22, which="both")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(observed_x, area1_at_observed / observed_y, color=AREA1_COLOR, lw=1.8, label="Area 1 / observed")
    axes[1].plot(observed_x, area2_at_observed / observed_y, color=AREA2_COLOR, lw=1.8, label="Area 2 / observed")
    axes[1].axhline(1.0, color="black", ls="--", lw=1.0)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Wavelength ($\mu$m)")
    axes[1].set_ylabel("Model / observed")
    axes[1].grid(alpha=0.22, which="both")
    axes[1].legend(frameon=False, fontsize=9)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def draw_spectral_profiles(
    path: Path,
    beta: float,
    major_edges: np.ndarray,
    wavelengths: np.ndarray,
    area1_profiles: np.ndarray,
    area2_profiles: np.ndarray,
) -> None:
    positive = np.concatenate(
        [area1_profiles[area1_profiles > 0.0], area2_profiles[area2_profiles > 0.0]]
    )
    norm = LogNorm(
        vmin=float(np.percentile(positive, 2.0)),
        vmax=float(np.percentile(positive, 99.5)),
        clip=True,
    )
    wave_edges = wavelength_edges(wavelengths)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharex=True, sharey=True, constrained_layout=True)
    images = []
    for axis, profiles, title in (
        (axes[0], area1_profiles, "Area 1-anchored"),
        (axes[1], area2_profiles, "Area 2-anchored"),
    ):
        images.append(axis.pcolormesh(major_edges, wave_edges, profiles, cmap="magma", norm=norm, shading="flat"))
        axis.axhline(18.0, color="white", ls="--", lw=0.8)
        axis.set_yscale("log")
        axis.set_xlabel("Major-axis offset (arcsec; southeast positive)")
        axis.set_title(title)
    axes[0].set_ylabel(r"Wavelength ($\mu$m)")
    figure.colorbar(images[0], ax=axes, location="bottom", shrink=0.72, pad=0.04, label=r"Minor-axis-integrated flux (Jy arcsec$^{-1}$)")
    figure.suptitle(rf"Spatially resolved spectrum after minor-axis integration; fixed $\beta={beta:g}$")
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def draw_all_beta_spectra(
    path: Path,
    betas: list[float],
    wavelength: np.ndarray,
    spectra: dict[float, dict[str, np.ndarray]],
    observed_wavelength: np.ndarray,
    observed_flux: np.ndarray,
    observed_map_18: float,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(betas)))
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 6.0), sharex=True, sharey=True, constrained_layout=True)
    for axis, area, title in (
        (axes[0], "area1", "Area 1-anchored whole-nebula spectra"),
        (axes[1], "area2", "Area 2-anchored whole-nebula spectra"),
    ):
        axis.plot(observed_wavelength, observed_flux, color="black", lw=2.0, label="Observed spectrum", zorder=5)
        axis.scatter([18.0], [observed_map_18], marker="*", s=100, facecolors="none", edgecolors="black", label="Observed map integral", zorder=8)
        for beta, color in zip(betas, colors, strict=True):
            axis.plot(wavelength, spectra[beta][area], color=color, lw=1.7, label=rf"$\beta={beta:g}$")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"Wavelength ($\mu$m)")
        axis.set_title(title)
        axis.grid(alpha=0.2, which="both")
    axes[0].set_ylabel(r"Whole-nebula $F_\nu$ (Jy)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def draw_all_beta_profiles(
    path: Path,
    betas: list[float],
    offset: np.ndarray,
    observed: np.ndarray,
    profiles: dict[float, dict[str, np.ndarray]],
) -> None:
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(betas)))
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), sharex=True, sharey=True, constrained_layout=True)
    for axis, area, title in (
        (axes[0], "area1", "Area 1-anchored"),
        (axes[1], "area2", "Area 2-anchored"),
    ):
        axis.plot(offset, observed, color="black", lw=2.2, label="Observed", zorder=6)
        for beta, color in zip(betas, colors, strict=True):
            axis.plot(offset, profiles[beta][area], color=color, lw=1.6, label=rf"$\beta={beta:g}$")
        axis.axvline(0.0, color="0.4", ls="--", lw=0.8)
        axis.set_xlabel("Major-axis offset (arcsec; southeast positive)")
        axis.set_title(title)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel(r"Minor-axis-integrated 18 $\mu$m flux (Jy arcsec$^{-1}$)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def draw_all_beta_totals(path: Path, summary_rows: list[dict[str, float]]) -> None:
    beta = np.asarray([row["beta"] for row in summary_rows])
    area1 = np.asarray([row["area1_total_18um_jy"] for row in summary_rows])
    area2 = np.asarray([row["area2_total_18um_jy"] for row in summary_rows])
    observed_map = float(summary_rows[0]["observed_map_total_18um_jy"])
    observed_spectrum = float(summary_rows[0]["observed_spectrum_18um_jy"])
    figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    axis.plot(beta, area1, "o-", color=AREA1_COLOR, label="Area 1-anchored")
    axis.plot(beta, area2, "s-", color=AREA2_COLOR, label="Area 2-anchored")
    axis.axhline(observed_map, color="black", lw=1.5, label="Observed 18 $\mu$m map integral")
    axis.axhline(observed_spectrum, color="black", lw=1.2, ls="--", label="Observed spectrum at 18 $\mu$m")
    axis.set_xlabel(r"Fixed emissivity index $\beta$")
    axis.set_ylabel(r"Integrated 18 $\mu$m flux (Jy)")
    axis.set_title("Whole-nebula 18 $\mu$m flux versus temperature-law emissivity")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS))
    parser.add_argument("--latitude-cutoff-deg", type=float, default=10.0)
    parser.add_argument("--parameter-root", type=Path, default=WHOLE_NEBULA_RESULTS / "parameters")
    parser.add_argument("--geometry-dir", type=Path, default=WHOLE_NEBULA_GEOMETRY)
    parser.add_argument("--mass-curve", type=Path, default=SMITH_GEOMETRY_DIR / "Mass-per-SolidAngle.csv")
    parser.add_argument("--velocity-curve", type=Path, default=SMITH_GEOMETRY_DIR / "v_vs_theta.csv")
    parser.add_argument("--observed-map", type=Path, default=OBSERVED_DATA_DIR / "i18_map.npz")
    parser.add_argument("--observed-decisions", type=Path, default=OBSERVED_DATA_DIR / "i18_map_decisions.json")
    parser.add_argument("--observed-profile", type=Path, default=OBSERVED_DATA_DIR / "i18_major_axis_profile.csv")
    parser.add_argument("--observed-spectrum", type=Path, default=OBSERVED_DATA_DIR / "homunculus_obs_flux_cgs.csv")
    parser.add_argument("--area1-data", type=Path, default=MODEL_INPUT_DIR / "lobe.edge.data")
    parser.add_argument("--area2-data", type=Path, default=MODEL_INPUT_DIR / "lobeedgedata2")
    parser.add_argument("--output-dir", type=Path, default=WHOLE_NEBULA_RESULTS / "rerun")
    parser.add_argument("--parameter-column", choices=("posterior_median", "maximum_posterior"), default="maximum_posterior")
    arguments = parser.parse_args()
    if any((not np.isfinite(beta)) or beta <= -4.0 for beta in arguments.betas):
        parser.error("every beta must be finite and greater than -4")
    if arguments.latitude_cutoff_deg < 0.0 or arguments.latitude_cutoff_deg >= 90.0:
        parser.error("latitude cutoff must satisfy 0 <= cutoff < 90 degrees")
    return arguments


def main() -> None:
    args = parse_arguments()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(args.observed_map.expanduser().resolve(), allow_pickle=True) as product:
        observed_original = np.asarray(product["i18_map"], dtype=float)
        footprint_original = np.asarray(product["footprint"], dtype=bool)
    decisions = json.loads(args.observed_decisions.read_text(encoding="utf-8"))
    x, y, x_edges, y_edges = angular_grid(observed_original.shape, decisions)
    observed = np.flipud(observed_original)
    footprint = np.flipud(footprint_original)
    geometry = universal.load_geometry(args.geometry_dir.expanduser().resolve())
    expected_shape = (len(y), len(x))
    if any(array.shape != expected_shape for array in geometry.values()):
        raise ValueError("Geometry grid does not match the observed 18-micron map.")

    pixel_area = float(abs(np.median(np.diff(x)) * np.median(np.diff(y))))
    along, perpendicular = spectral_tools.projected_coordinates(x, y, PA_DEG)
    observed_major_centres, observed_major, major_edges = one_d.read_observed_profile(
        args.observed_profile.expanduser().resolve()
    )
    minor_edges = bin_edges_from_range(perpendicular, PROFILE_BIN_WIDTH_ARCSEC)
    minor_centres = 0.5 * (minor_edges[:-1] + minor_edges[1:])
    observed_minor = profile_from_brightness(
        observed, perpendicular, minor_edges, pixel_area, footprint
    )
    observed_map_total = float(np.sum(np.where(footprint, observed, 0.0)) * pixel_area)
    observed_wavelength, observed_spectrum = read_observed_spectrum(
        args.observed_spectrum.expanduser().resolve()
    )
    observed_spectrum_18 = float(
        np.interp(18.0, observed_wavelength, observed_spectrum)
    )

    mass_latitude, mass_value = universal.read_named_curve(
        args.mass_curve.expanduser().resolve()
    )
    mass_curve = PchipInterpolator(mass_latitude, mass_value, extrapolate=False)
    velocity_curve = read_velocity_curve(args.velocity_curve.expanduser().resolve())
    apertures = apertures_in_sky_coordinates(velocity_curve)
    area1_observed_wavelength, area1_observed_flux = universal.nearest_18um_flux(
        args.area1_data.expanduser().resolve()
    )
    area2_observed_wavelength, area2_observed_flux = universal.nearest_18um_flux(
        args.area2_data.expanduser().resolve()
    )

    model_wavelength = np.unique(
        np.r_[np.logspace(np.log10(2.0), np.log10(100.0), 120), 18.0]
    )
    profile_targets = np.unique(
        np.r_[np.logspace(np.log10(5.0), np.log10(100.0), 38), 18.0]
    )
    profile_indices = np.unique(
        [int(np.argmin(np.abs(model_wavelength - target))) for target in profile_targets]
    )
    profile_wavelength = model_wavelength[profile_indices]

    summary_rows: list[dict[str, float]] = []
    all_spectra: dict[float, dict[str, np.ndarray]] = {}
    all_major_profiles: dict[float, dict[str, np.ndarray]] = {}
    all_maps: dict[float, dict[str, np.ndarray]] = {}

    for case_index, beta in enumerate(args.betas, start=1):
        print(f"[{case_index}/{len(args.betas)}] beta={beta:g}", flush=True)
        case_output = output / beta_label(beta)
        case_output.mkdir(parents=True, exist_ok=True)
        parameter_path = (
            args.parameter_root.expanduser().resolve()
            / beta_label(beta)
            / "parameter_summary.csv"
        )
        theta = read_parameter_vector(parameter_path, args.parameter_column)
        universal.configure_temperature_beta(beta)
        state = physical_model.derive_state(theta)
        configs = {
            name: universal.area_configuration(name, state)
            for name in ("area1", "area2")
        }

        local_flux1, local_flux2, _, _ = accepted_model.evaluate_models(
            theta, np.asarray([18.0]), np.asarray([18.0])
        )
        local_mcmc_flux = {
            "area1": float(local_flux1[0] / JY_CGS),
            "area2": float(local_flux2[0] / JY_CGS),
        }

        spatial: dict[str, dict[str, np.ndarray]] = {}
        strict_maps: dict[str, np.ndarray] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        for area, aperture_name in (("area1", "Area 1"), ("area2", "Area 2")):
            raw_spatial, area_diagnostics = universal.build_area_model(
                configs[area],
                geometry,
                apertures[aperture_name],
                x,
                y,
                mass_curve,
                velocity_curve,
                state["p_size"],
                state["q_density_size"],
            )
            cutoff_spatial = apply_latitude_cutoff(
                configs[area], raw_spatial, geometry, args.latitude_cutoff_deg
            )
            mask = aperture_mask(apertures[aperture_name], x, y)
            strict_maps[area] = np.array(
                cutoff_spatial["surface_brightness_jy_arcsec2"], copy=True
            )
            strict_whole_flux = float(np.sum(strict_maps[area]) * pixel_area)
            closed_spatial, column_scale, strict_aperture_flux = close_anchor_aperture(
                configs[area], cutoff_spatial, mask, local_mcmc_flux[area]
            )
            spatial[area] = closed_spatial
            closed_aperture_flux = float(
                np.mean(spatial[area]["surface_brightness_jy_arcsec2"][mask])
            )
            area_diagnostics["strict_anchor_aperture_flux_after_cutoff_Jy"] = (
                strict_aperture_flux
            )
            area_diagnostics["strict_whole_nebula_18um_flux_Jy"] = strict_whole_flux
            area_diagnostics["aperture_closure_column_scale"] = column_scale
            area_diagnostics["closed_anchor_aperture_flux_Jy"] = closed_aperture_flux
            area_diagnostics["local_MCMC_model_flux_at_18um_Jy"] = local_mcmc_flux[area]
            area_diagnostics["strict_anchor_over_local_MCMC_flux"] = (
                strict_aperture_flux
                / local_mcmc_flux[area]
            )
            area_diagnostics["closed_anchor_over_local_MCMC_flux"] = (
                closed_aperture_flux / local_mcmc_flux[area]
            )
            diagnostics[area] = area_diagnostics

        maps = {
            area: spatial[area]["surface_brightness_jy_arcsec2"]
            for area in ("area1", "area2")
        }
        totals18 = {
            area: float(np.sum(maps[area]) * pixel_area)
            for area in ("area1", "area2")
        }
        major_profiles = {
            area: major_axis_profile_from_flux_map(
                maps[area] * pixel_area, x, y, major_edges, PA_DEG
            )
            for area in ("area1", "area2")
        }
        minor_profiles = {
            area: profile_from_brightness(
                maps[area], perpendicular, minor_edges, pixel_area
            )
            for area in ("area1", "area2")
        }
        profile_integrals = {
            area: {
                "major": float(np.sum(major_profiles[area] * np.diff(major_edges))),
                "minor": float(np.sum(minor_profiles[area] * np.diff(minor_edges))),
            }
            for area in ("area1", "area2")
        }
        for area in ("area1", "area2"):
            for direction in ("major", "minor"):
                relative_error = abs(
                    profile_integrals[area][direction] / totals18[area] - 1.0
                )
                if relative_error > 5.0e-10:
                    raise RuntimeError(
                        f"{direction}-profile integral failed for beta={beta}, "
                        f"{area}: {relative_error}"
                    )

        spectra: dict[str, np.ndarray] = {}
        spectral_major: dict[str, np.ndarray] = {}
        spectral_minor: dict[str, np.ndarray] = {}
        for area in ("area1", "area2"):
            spectra[area], spectral_major[area], spectral_minor[area] = spectral_products(
                configs[area],
                spatial[area],
                geometry,
                mass_curve,
                velocity_curve,
                state["p_size"],
                state["q_density_size"],
                model_wavelength,
                profile_indices,
                along,
                perpendicular,
                major_edges,
                minor_edges,
                pixel_area,
            )

        index18 = int(np.where(model_wavelength == 18.0)[0][0])
        closure = {
            area: abs(spectra[area][index18] / totals18[area] - 1.0)
            for area in ("area1", "area2")
        }
        if max(closure.values()) > 5.0e-10:
            raise RuntimeError(
                f"18-micron map/spectrum closure failed for beta={beta}: {closure}"
            )

        map_path = case_output / "observed_models_and_residuals_18um.png"
        clean_map_path = case_output / "observed_area2_area1_emission_18um.png"
        profile_path = case_output / "projected_profiles_18um.png"
        spectrum_path = case_output / "whole_nebula_spectrum.png"
        spectral_profile_path = case_output / "minor_axis_integrated_spectral_profiles.png"
        draw_maps(map_path, beta, x, y, footprint, observed, maps["area1"], maps["area2"])
        draw_three_panel_emission(
            clean_map_path,
            beta,
            x,
            y,
            footprint,
            observed,
            maps["area2"],
            maps["area1"],
        )
        draw_profiles(
            profile_path,
            beta,
            observed_major_centres,
            observed_major,
            major_profiles["area1"],
            major_profiles["area2"],
            minor_centres,
            observed_minor,
            minor_profiles["area1"],
            minor_profiles["area2"],
        )
        draw_spectrum(
            spectrum_path,
            beta,
            observed_wavelength,
            observed_spectrum,
            observed_map_total,
            model_wavelength,
            spectra["area1"],
            spectra["area2"],
        )
        draw_spectral_profiles(
            spectral_profile_path,
            beta,
            major_edges,
            profile_wavelength,
            spectral_major["area1"],
            spectral_major["area2"],
        )

        np.savez_compressed(
            case_output / "whole_nebula_products.npz",
            x_arcsec=x,
            y_arcsec=y,
            footprint=footprint,
            observed_18um_jy_arcsec2=observed,
            area1_strict_18um_jy_arcsec2=strict_maps["area1"],
            area2_strict_18um_jy_arcsec2=strict_maps["area2"],
            area1_18um_jy_arcsec2=maps["area1"],
            area2_18um_jy_arcsec2=maps["area2"],
            major_axis_offset_arcsec=observed_major_centres,
            observed_major_profile_jy_arcsec=observed_major,
            area1_major_profile_jy_arcsec=major_profiles["area1"],
            area2_major_profile_jy_arcsec=major_profiles["area2"],
            minor_axis_offset_arcsec=minor_centres,
            observed_minor_profile_jy_arcsec=observed_minor,
            area1_minor_profile_jy_arcsec=minor_profiles["area1"],
            area2_minor_profile_jy_arcsec=minor_profiles["area2"],
            model_wavelength_micron=model_wavelength,
            area1_whole_spectrum_jy=spectra["area1"],
            area2_whole_spectrum_jy=spectra["area2"],
            spectral_profile_wavelength_micron=profile_wavelength,
            area1_minor_axis_integrated_spectral_profile_jy_arcsec=spectral_major["area1"],
            area2_minor_axis_integrated_spectral_profile_jy_arcsec=spectral_major["area2"],
        )
        write_csv(
            case_output / "projected_profiles_18um.csv",
            [
                "major_axis_offset_arcsec",
                "observed_minor_axis_integrated_jy_arcsec",
                "area1_minor_axis_integrated_jy_arcsec",
                "area2_minor_axis_integrated_jy_arcsec",
            ],
            zip(
                observed_major_centres,
                observed_major,
                major_profiles["area1"],
                major_profiles["area2"],
            ),
        )
        write_csv(
            case_output / "whole_nebula_spectrum.csv",
            [
                "wavelength_micron",
                "area1_whole_nebula_flux_jy",
                "area2_whole_nebula_flux_jy",
            ],
            zip(model_wavelength, spectra["area1"], spectra["area2"]),
        )
        comparable = observed_wavelength <= model_wavelength[-1]
        observed_x = observed_wavelength[comparable]
        observed_y = observed_spectrum[comparable]
        area1_at_observed = log_interpolate(
            model_wavelength, spectra["area1"], observed_x
        )
        area2_at_observed = log_interpolate(
            model_wavelength, spectra["area2"], observed_x
        )
        write_csv(
            case_output / "spectrum_at_observed_wavelengths.csv",
            [
                "wavelength_micron",
                "observed_flux_jy",
                "area1_model_flux_jy",
                "area2_model_flux_jy",
                "area1_over_observed",
                "area2_over_observed",
            ],
            zip(
                observed_x,
                observed_y,
                area1_at_observed,
                area2_at_observed,
                area1_at_observed / observed_y,
                area2_at_observed / observed_y,
            ),
        )

        row = {
            "beta": float(beta),
            "temperature_exponent": float(universal.TEMPERATURE_EXPONENT),
            "latitude_cutoff_deg": float(args.latitude_cutoff_deg),
            "observed_map_total_18um_jy": observed_map_total,
            "observed_spectrum_18um_jy": observed_spectrum_18,
            "observed_map_over_spectrum_at_18um": observed_map_total / observed_spectrum_18,
            "area1_total_18um_jy": totals18["area1"],
            "area2_total_18um_jy": totals18["area2"],
            "area1_fraction_of_observed_map": totals18["area1"] / observed_map_total,
            "area2_fraction_of_observed_map": totals18["area2"] / observed_map_total,
            "area1_MCMC_model_18um_jy": local_mcmc_flux["area1"],
            "area2_MCMC_model_18um_jy": local_mcmc_flux["area2"],
            "area1_nearest_observed_wavelength_micron": area1_observed_wavelength,
            "area1_nearest_observed_flux_jy": area1_observed_flux,
            "area2_nearest_observed_wavelength_micron": area2_observed_wavelength,
            "area2_nearest_observed_flux_jy": area2_observed_flux,
            "area1_strict_total_18um_jy": diagnostics["area1"]["strict_whole_nebula_18um_flux_Jy"],
            "area2_strict_total_18um_jy": diagnostics["area2"]["strict_whole_nebula_18um_flux_Jy"],
            "area1_aperture_closure_column_scale": diagnostics["area1"]["aperture_closure_column_scale"],
            "area2_aperture_closure_column_scale": diagnostics["area2"]["aperture_closure_column_scale"],
            "area1_strict_anchor_aperture_18um_jy": diagnostics["area1"]["strict_anchor_aperture_flux_after_cutoff_Jy"],
            "area2_strict_anchor_aperture_18um_jy": diagnostics["area2"]["strict_anchor_aperture_flux_after_cutoff_Jy"],
            "area1_closed_anchor_aperture_18um_jy": diagnostics["area1"]["closed_anchor_aperture_flux_Jy"],
            "area2_closed_anchor_aperture_18um_jy": diagnostics["area2"]["closed_anchor_aperture_flux_Jy"],
            "area1_strict_anchor_over_MCMC": diagnostics["area1"]["strict_anchor_over_local_MCMC_flux"],
            "area2_strict_anchor_over_MCMC": diagnostics["area2"]["strict_anchor_over_local_MCMC_flux"],
            "area1_closed_anchor_over_MCMC": diagnostics["area1"]["closed_anchor_over_local_MCMC_flux"],
            "area2_closed_anchor_over_MCMC": diagnostics["area2"]["closed_anchor_over_local_MCMC_flux"],
            "area1_major_profile_integral_relative_error": abs(profile_integrals["area1"]["major"] / totals18["area1"] - 1.0),
            "area2_major_profile_integral_relative_error": abs(profile_integrals["area2"]["major"] / totals18["area2"] - 1.0),
            "area1_minor_profile_integral_relative_error": abs(profile_integrals["area1"]["minor"] / totals18["area1"] - 1.0),
            "area2_minor_profile_integral_relative_error": abs(profile_integrals["area2"]["minor"] / totals18["area2"] - 1.0),
            "area1_map_spectrum_18um_relative_error": closure["area1"],
            "area2_map_spectrum_18um_relative_error": closure["area2"],
        }
        summary_rows.append(row)
        summary = {
            **row,
            "parameter_source": str(parameter_path),
            "parameter_column": args.parameter_column,
            "observed_map_source": str(args.observed_map),
            "observed_spectrum_source": str(args.observed_spectrum),
            "spectral_flux_posthoc_renormalized": False,
            "aperture_column_closure_applied": True,
            "aperture_column_closure_explanation": (
                "One common LOS-column multiplier is solved per area so the "
                "spatialized 1-arcsec^2 anchor aperture reproduces the local "
                "MCMC 18-micron model. Dense/rare contrast, temperatures, grain "
                "distributions, Smith latitude shape, and nonlinear transfer "
                "remain unchanged."
            ),
            "observational_calibrations_forced_to_match": False,
            "area1_diagnostics": diagnostics["area1"],
            "area2_diagnostics": diagnostics["area2"],
            "files": {
                "maps": str(map_path),
                "clean_three_panel_emission": str(clean_map_path),
                "profiles": str(profile_path),
                "spectrum": str(spectrum_path),
                "spectral_profiles": str(spectral_profile_path),
            },
        }
        (case_output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        all_spectra[beta] = spectra
        all_major_profiles[beta] = major_profiles
        all_maps[beta] = maps

    summary_header = list(summary_rows[0].keys())
    write_csv(
        output / "all_beta_summary.csv",
        summary_header,
        ([row[name] for name in summary_header] for row in summary_rows),
    )
    draw_all_beta_spectra(
        output / "all_beta_whole_nebula_spectra.png",
        list(args.betas),
        model_wavelength,
        all_spectra,
        observed_wavelength,
        observed_spectrum,
        observed_map_total,
    )
    draw_all_beta_profiles(
        output / "all_beta_major_axis_profiles_18um.png",
        list(args.betas),
        observed_major_centres,
        observed_major,
        all_major_profiles,
    )
    draw_all_beta_totals(
        output / "all_beta_integrated_18um_flux.png", summary_rows
    )
    print(output / "all_beta_summary.csv")
    print(output / "all_beta_whole_nebula_spectra.png")
    print(output / "all_beta_major_axis_profiles_18um.png")


if __name__ == "__main__":
    main()
