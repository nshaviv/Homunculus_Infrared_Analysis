#!/usr/bin/env python3
"""Empirically extrapolate the two aperture fluxes over the Homunculus.

This is deliberately *not* a new radiative-transfer calculation.  It performs
the normalization requested for the spatial comparison:

    (1) take the measured 18-micron flux in one aperture;
    (2) divide it by that aperture's fitted effective LOS dust column;
    (3) convert the latest Smith-normalized whole-nebula mass into a flux;
    (4) distribute that flux with Smith's dM/dOmega latitude curve; and
    (5) project the latitude/azimuth cells into major-axis-offset bins.

Area 1 and Area 2 are two alternative calibrations of the same Smith
distribution.  They are never added together.

For a one-square-arcsecond aperture at distance D,

    M_ap,dust = Sigma_LOS A_ap,       A_ap = [D (1 arcsec)]^2,
    eta_18    = F_18,ap / M_ap,dust,
    dF_18/dOmega_star = eta_18 M_dust,total p(lambda),

where p(lambda) is Smith's q(lambda)=dM/dOmega shape normalized over both
lobes.  The latest saved masses include the adopted factor 100, so they are
divided by 100 before multiplying by the dust-specific eta_18.

The fitted Area-2 phase columns are TOTAL LOS columns (front plus rear).  Its
effective calibration column is therefore

    Sigma_LOS,2 = f Sigma_dense,total + (1-f) Sigma_rare,total,

with f=0.444.  The Area-2 mass is the latest FRONT-normalized global Smith
mass; the rear wall is the second crossing of that same global distribution,
not an extra global mass to add.
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

from project_paths import (
    MODEL_INPUT_DIR,
    OBSERVED_DATA_DIR,
    SMITH_GEOMETRY_DIR,
    TWO_PHASE_RESULTS,
    WHOLE_NEBULA_RESULTS,
)


M_SUN_G = 1.98847e33
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
DISTANCE_CM = 7.1e21
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
BALLISTIC_AGE_YR = 152.9
EJECTION_DURATION_YR = 20.0
EJECTION_LAYERS = 101
INCLINATION_FROM_LOS_DEG = 41.0
TOTAL_TO_DUST_FACTOR = 100.0
AREA2_DENSE_COVERING_FRACTION = 0.444
TARGET_WAVELENGTH_MICRON = 18.0
LATITUDE_STEP_DEG = 0.25
PHI_STEP_DEG = 0.5
ERROR_FRACTION = 0.05
BACKGROUND_NOISE_JY = 5.0

OBSERVED_COLOR = "#9f2f1f"
AREA1_COLOR = "#126782"
AREA2_COLOR = "#6f3f8f"


def bin_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    spacing = np.diff(centers)
    if centers.ndim != 1 or len(centers) < 2 or np.any(spacing <= 0.0):
        raise ValueError("Profile centers must be one-dimensional and increasing.")
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * spacing[0]
    edges[-1] = centers[-1] + 0.5 * spacing[-1]
    return edges


def read_named_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None or len(data.dtype.names) != 2:
        raise ValueError(f"Expected a named two-column CSV: {path}")
    x = np.asarray(data[data.dtype.names[0]], dtype=float)
    y = np.asarray(data[data.dtype.names[1]], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y > 0.0)
    order = np.argsort(x[keep])
    return x[keep][order], y[keep][order]


def read_observed_profile(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    offset = np.asarray(data["major_axis_offset_arcsec"], dtype=float)
    profile = np.asarray(
        data["minor_axis_integrated_flux_jy_per_arcsec"], dtype=float
    )
    edges = bin_edges_from_centers(offset)
    if np.any(~np.isfinite(offset)) or np.any(~np.isfinite(profile)):
        raise ValueError(f"Non-finite observed-profile values in {path}")
    return offset, profile, edges


def read_parameter_medians(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values[row["parameter"]] = float(row["posterior_median"])
    return values


def read_latest_corrected_masses(path: Path) -> dict[str, float]:
    wanted = {
        "area1": "area1_total_mass_with_LOS_correction",
        "area2": "area2_FRONT_ONLY_mass_with_LOS_correction",
    }
    masses: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for short, saved_name in wanted.items():
                if row["mass_quantity"] == saved_name:
                    masses[short] = float(row["posterior_median_Msun"])
    missing = sorted(set(wanted) - set(masses))
    if missing:
        raise ValueError(f"Missing corrected masses {missing} in {path}")
    return masses


def nearest_flux_measurement(path: Path) -> dict[str, float]:
    data = np.loadtxt(path, delimiter=",")
    wavelength = np.asarray(data[:, 0], dtype=float)
    flux_jy = np.asarray(data[:, 1], dtype=float)
    index = int(np.argmin(np.abs(wavelength - TARGET_WAVELENGTH_MICRON)))
    uncertainty_jy = float(
        np.hypot(ERROR_FRACTION * flux_jy[index], BACKGROUND_NOISE_JY)
    )
    return {
        "wavelength_micron": float(wavelength[index]),
        "flux_jy": float(flux_jy[index]),
        "uncertainty_jy": uncertainty_jy,
    }


def build_calibrations(
    parameters: dict[str, float],
    masses_total_msun: dict[str, float],
    area1_flux: dict[str, float],
    area2_flux: dict[str, float],
) -> dict[str, dict[str, float | str]]:
    """Build the two independent empirical Jy-per-dust-mass calibrations."""
    sigma1 = 10.0 ** parameters["area1_log10_sigma_total_LOS_g_cm2"]
    sigma2_dense = 10.0 ** parameters[
        "area2_log10_sigma_dense_total_LOS_g_cm2"
    ]
    sigma2_rare = 10.0 ** parameters[
        "area2_log10_sigma_rare_total_LOS_g_cm2"
    ]
    sigma2 = (
        AREA2_DENSE_COVERING_FRACTION * sigma2_dense
        + (1.0 - AREA2_DENSE_COVERING_FRACTION) * sigma2_rare
    )
    aperture_area_cm2 = (DISTANCE_CM * ARCSEC_TO_RAD) ** 2

    rows: dict[str, dict[str, float | str]] = {}
    for area, latitude, sigma, flux, column_note, mass_note in (
        (
            "area1",
            60.0,
            sigma1,
            area1_flux,
            "posterior-median Area-1 total effective LOS dust column",
            "latest LOS-corrected whole-nebula total mass from Area 1",
        ),
        (
            "area2",
            80.0,
            sigma2,
            area2_flux,
            "f*Sigma_dense,total_LOS + (1-f)*Sigma_rare,total_LOS; front+rear",
            "latest LOS-corrected FRONT-normalized whole-nebula total mass",
        ),
    ):
        aperture_dust_mass_msun = sigma * aperture_area_cm2 / M_SUN_G
        flux_per_sigma = float(flux["flux_jy"]) / sigma
        flux_per_dust_mass = float(flux["flux_jy"]) / aperture_dust_mass_msun
        whole_dust_mass_msun = masses_total_msun[area] / TOTAL_TO_DUST_FACTOR
        predicted_total_flux = flux_per_dust_mass * whole_dust_mass_msun
        relative_flux_error = float(flux["uncertainty_jy"]) / float(
            flux["flux_jy"]
        )
        rows[area] = {
            "area": "Area 1" if area == "area1" else "Area 2",
            "anchor_latitude_deg": latitude,
            "observed_wavelength_micron": float(flux["wavelength_micron"]),
            "observed_integrated_flux_jy": float(flux["flux_jy"]),
            "observed_flux_uncertainty_jy": float(flux["uncertainty_jy"]),
            "effective_sigma_LOS_dust_g_cm2": sigma,
            "flux_per_sigma_LOS_jy_cm2_g": flux_per_sigma,
            "one_arcsec2_physical_area_cm2": aperture_area_cm2,
            "dust_mass_in_one_arcsec2_aperture_Msun": aperture_dust_mass_msun,
            "flux_per_aperture_dust_mass_jy_per_Msun": flux_per_dust_mass,
            "latest_corrected_global_total_mass_Msun": masses_total_msun[area],
            "latest_corrected_global_dust_mass_Msun": whole_dust_mass_msun,
            "predicted_full_Smith_shell_flux_jy": predicted_total_flux,
            "relative_anchor_flux_uncertainty": relative_flux_error,
            "column_definition": column_note,
            "mass_definition": mass_note,
        }
    return rows


def latitude_distribution(
    mass_curve: PchipInterpolator,
    calibrations: dict[str, dict[str, float | str]],
    latitude_min_deg: float,
) -> dict[str, np.ndarray | float]:
    """Return dF/d|latitude| without renormalizing an excluded latitude band.

    Smith's q(lambda) is always normalized over the complete 0--90 degree
    two-lobe shell.  If ``latitude_min_deg`` is positive, the omitted fraction
    is genuinely removed from the flux prediction rather than redistributed
    over the retained latitudes.
    """
    if not (0.0 <= latitude_min_deg < 90.0):
        raise ValueError("latitude_min_deg must satisfy 0 <= value < 90 deg")
    if not np.isclose(latitude_min_deg / LATITUDE_STEP_DEG, round(latitude_min_deg / LATITUDE_STEP_DEG)):
        raise ValueError("latitude_min_deg must align with the latitude grid")

    full_edges = np.arange(0.0, 90.0, LATITUDE_STEP_DEG, dtype=float)
    full_edges = np.append(full_edges, 90.0)
    full_latitude = 0.5 * (full_edges[:-1] + full_edges[1:])
    full_widths_rad = np.deg2rad(np.diff(full_edges))
    full_q = np.asarray(mass_curve(full_latitude), dtype=float)
    full_raw_weight = (
        4.0
        * np.pi
        * full_q
        * np.cos(np.deg2rad(full_latitude))
        * full_widths_rad
    )
    full_discrete_integral = float(np.sum(full_raw_weight))

    latitude_edges = np.arange(
        latitude_min_deg, 90.0, LATITUDE_STEP_DEG, dtype=float
    )
    latitude_edges = np.append(latitude_edges, 90.0)
    latitude = 0.5 * (latitude_edges[:-1] + latitude_edges[1:])
    widths_deg = np.diff(latitude_edges)
    widths_rad = np.deg2rad(widths_deg)
    q = np.asarray(mass_curve(latitude), dtype=float)

    # Smith q is defined per source solid angle.  Both hemispheres and all
    # azimuths give dOmega = 4*pi*cos(lambda)*d|lambda|.
    raw_weight = 4.0 * np.pi * q * np.cos(np.deg2rad(latitude)) * widths_rad
    retained_integral = float(np.sum(raw_weight))
    fraction = raw_weight / full_discrete_integral
    flux: dict[str, np.ndarray] = {}
    flux_per_deg: dict[str, np.ndarray] = {}
    for area in ("area1", "area2"):
        total = float(calibrations[area]["predicted_full_Smith_shell_flux_jy"])
        flux[area] = total * fraction
        flux_per_deg[area] = flux[area] / widths_deg
    return {
        "edges_deg": latitude_edges,
        "latitude_deg": latitude,
        "width_deg": widths_deg,
        "q_raw": q,
        "mass_fraction": fraction,
        "latitude_min_deg": latitude_min_deg,
        "smith_full_discrete_integral": full_discrete_integral,
        "smith_retained_discrete_integral": retained_integral,
        "retained_mass_fraction": retained_integral / full_discrete_integral,
        "area1_flux_jy_per_bin": flux["area1"],
        "area2_flux_jy_per_bin": flux["area2"],
        "area1_flux_jy_per_deg": flux_per_deg["area1"],
        "area2_flux_jy_per_deg": flux_per_deg["area2"],
    }


def project_to_major_axis(
    latitude_result: dict[str, np.ndarray | float],
    velocity_curve: PchipInterpolator,
    offset_edges_arcsec: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Project Smith source-solid-angle cells into observed offset bins.

    Flux is assigned by source solid angle, not projected surface area.  That
    is the consistent projection for this empirical flux-per-mass model: each
    cell carries a fixed mass fraction and therefore a fixed flux fraction.
    """
    latitude = np.asarray(latitude_result["latitude_deg"], dtype=float)
    phi = np.deg2rad(
        np.arange(PHI_STEP_DEG / 2.0, 360.0, PHI_STEP_DEG, dtype=float)
    )
    cos_phi = np.cos(phi)
    inclination = np.deg2rad(INCLINATION_FROM_LOS_DEG)
    # Match the finite-duration geometry used by the LOS calculation: the
    # same mass is spread uniformly over 101 launch times spanning 20 years.
    ages_year = BALLISTIC_AGE_YR + np.linspace(
        -0.5 * EJECTION_DURATION_YR,
        0.5 * EJECTION_DURATION_YR,
        EJECTION_LAYERS,
    )
    ages_seconds = ages_year[:, None] * SECONDS_PER_YEAR

    histogram = {
        "area1": np.zeros(len(offset_edges_arcsec) - 1, dtype=float),
        "area2": np.zeros(len(offset_edges_arcsec) - 1, dtype=float),
    }
    overflow = {"area1": 0.0, "area2": 0.0}
    all_min = np.inf
    all_max = -np.inf

    for index, latitude_abs in enumerate(latitude):
        radius_cm = float(velocity_curve(latitude_abs)) * 1.0e5 * ages_seconds
        per_cell_flux = {
            area: float(
                np.asarray(latitude_result[f"{area}_flux_jy_per_bin"])[index]
            )
            / (2.0 * len(phi) * EJECTION_LAYERS)
            for area in ("area1", "area2")
        }
        for hemisphere in (-1.0, 1.0):
            lat = np.deg2rad(hemisphere * latitude_abs)
            along_cm = (
                -radius_cm * np.cos(lat) * cos_phi[None, :] * np.cos(inclination)
                + radius_cm * np.sin(lat) * np.sin(inclination)
            )
            along_arcsec = (
                along_cm / DISTANCE_CM / ARCSEC_TO_RAD
            ).ravel()
            all_min = min(all_min, float(np.min(along_arcsec)))
            all_max = max(all_max, float(np.max(along_arcsec)))
            inside = (along_arcsec >= offset_edges_arcsec[0]) & (
                along_arcsec < offset_edges_arcsec[-1]
            )
            for area in ("area1", "area2"):
                weights = np.full(
                    len(phi) * EJECTION_LAYERS,
                    per_cell_flux[area],
                    dtype=float,
                )
                histogram[area] += np.histogram(
                    along_arcsec, bins=offset_edges_arcsec, weights=weights
                )[0]
                overflow[area] += float(np.sum(weights[~inside]))

    widths = np.diff(offset_edges_arcsec)
    return {
        "area1_flux_jy_per_bin": histogram["area1"],
        "area2_flux_jy_per_bin": histogram["area2"],
        "area1_flux_jy_per_arcsec": histogram["area1"] / widths,
        "area2_flux_jy_per_arcsec": histogram["area2"] / widths,
        "area1_overflow_jy": overflow["area1"],
        "area2_overflow_jy": overflow["area2"],
        "projected_min_offset_arcsec": all_min,
        "projected_max_offset_arcsec": all_max,
    }


def write_dict_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_aligned_csv(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(zip(*columns))


def make_figure(
    path: Path,
    latitude_result: dict[str, np.ndarray | float],
    offset: np.ndarray,
    widths: np.ndarray,
    observed: np.ndarray,
    projected: dict[str, np.ndarray | float],
    calibrations: dict[str, dict[str, float | str]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.1), constrained_layout=True)
    latitude = np.asarray(latitude_result["latitude_deg"])

    for area, color, label in (
        ("area1", AREA1_COLOR, "Area 1 calibration"),
        ("area2", AREA2_COLOR, "Area 2 calibration"),
    ):
        y = np.asarray(latitude_result[f"{area}_flux_jy_per_deg"])
        fractional_error = float(
            calibrations[area]["relative_anchor_flux_uncertainty"]
        )
        axes[0].fill_between(
            latitude,
            y * (1.0 - fractional_error),
            y * (1.0 + fractional_error),
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )
        axes[0].plot(latitude, y, color=color, lw=2.3, label=label)

    axes[0].set(
        xlim=(0.0, 90.0),
        ylim=(0.0, None),
        xlabel="Absolute latitude |λ| (deg)",
        ylabel="18 μm flux per degree of |latitude| (Jy deg⁻¹)",
        title="Smith mass distribution converted to flux",
    )
    axes[0].legend(frameon=False)

    observed_total = float(np.sum(observed * widths))
    axes[1].fill_between(offset, observed, color=OBSERVED_COLOR, alpha=0.12)
    axes[1].plot(
        offset,
        observed,
        color=OBSERVED_COLOR,
        lw=2.5,
        label=f"Observed whole nebula ({observed_total:,.0f} Jy)",
    )
    for area, color, label in (
        ("area1", AREA1_COLOR, "Area 1-derived Smith shell"),
        ("area2", AREA2_COLOR, "Area 2-derived Smith shell"),
    ):
        y = np.asarray(projected[f"{area}_flux_jy_per_arcsec"])
        fractional_error = float(
            calibrations[area]["relative_anchor_flux_uncertainty"]
        )
        total = float(
            np.sum(np.asarray(projected[f"{area}_flux_jy_per_bin"]))
        )
        axes[1].fill_between(
            offset,
            y * (1.0 - fractional_error),
            y * (1.0 + fractional_error),
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )
        axes[1].plot(
            offset, y, color=color, lw=2.2, label=f"{label} ({total:,.0f} Jy)"
        )
    axes[1].axvline(0.0, color="0.35", ls="--", lw=1.0)
    axes[1].set(
        xlim=(float(offset[0]), float(offset[-1])),
        ylim=(0.0, None),
        xlabel="Offset along major axis (arcsec; southeast positive)",
        ylabel="Minor-axis-integrated flux (Jy arcsec⁻¹)",
        title="Projection onto the observed major-axis profile",
    )
    axes[1].legend(frameon=False, fontsize=9)

    for axis in axes:
        axis.grid(axis="y", color="0.88", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Independent empirical 18 μm flux-per-mass extrapolations "
        f"(|λ| ≥ {float(latitude_result['latitude_min_deg']):g}°)",
        fontsize=16,
    )
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def write_method(
    path: Path,
    calibrations: dict[str, dict[str, float | str]],
    observed_total: float,
    projected: dict[str, np.ndarray | float],
    latitude_result: dict[str, np.ndarray | float],
) -> None:
    a1 = calibrations["area1"]
    a2 = calibrations["area2"]
    p1 = float(a1["predicted_full_Smith_shell_flux_jy"])
    p2 = float(a2["predicted_full_Smith_shell_flux_jy"])
    retained = float(latitude_result["retained_mass_fraction"])
    converted1 = float(np.sum(np.asarray(projected["area1_flux_jy_per_bin"])))
    converted2 = float(np.sum(np.asarray(projected["area2_flux_jy_per_bin"])))
    lines = [
        "# Area-normalized 18 micron flux distribution",
        "",
        "## What is calculated",
        "",
        "Area 1 and Area 2 are independent calibrations; they are not added. For each area:",
        "",
        "`eta_18 = F_18,ap / (Sigma_LOS A_1arcsec2)`",
        "",
        "`dF_18/dOmega_star = eta_18 M_dust,total p(lambda)`",
        "",
        "Here `p(lambda)` is the Smith `dM/dOmega` shape normalized over both lobes and over the complete 0--90 degree distribution, and the latest saved factor-100 total mass is divided by 100 to recover the emitting dust mass. Each retained source-solid-angle cell is then projected into the observed major-axis-offset bins. No Mie opacity, fitted temperature, or radiative-transfer curve is re-evaluated in this empirical graph.",
        "",
        "## Exact anchor inputs",
        "",
        "| Quantity | Area 1 | Area 2 |",
        "|---|---:|---:|",
        f"| Anchor latitude | {float(a1['anchor_latitude_deg']):.1f} deg | {float(a2['anchor_latitude_deg']):.1f} deg |",
        f"| Nearest measured wavelength | {float(a1['observed_wavelength_micron']):.6f} micron | {float(a2['observed_wavelength_micron']):.6f} micron |",
        f"| Measured aperture flux | {float(a1['observed_integrated_flux_jy']):.6f} Jy | {float(a2['observed_integrated_flux_jy']):.6f} Jy |",
        f"| Effective LOS dust column | {float(a1['effective_sigma_LOS_dust_g_cm2']):.6e} g cm^-2 | {float(a2['effective_sigma_LOS_dust_g_cm2']):.6e} g cm^-2 |",
        f"| Flux / LOS column | {float(a1['flux_per_sigma_LOS_jy_cm2_g']):.6e} | {float(a2['flux_per_sigma_LOS_jy_cm2_g']):.6e} Jy cm^2 g^-1 |",
        f"| Dust mass in 1 arcsec^2 | {float(a1['dust_mass_in_one_arcsec2_aperture_Msun']):.6e} Msun | {float(a2['dust_mass_in_one_arcsec2_aperture_Msun']):.6e} Msun |",
        f"| Latest LOS-corrected global total mass | {float(a1['latest_corrected_global_total_mass_Msun']):.6f} Msun | {float(a2['latest_corrected_global_total_mass_Msun']):.6f} Msun |",
        f"| Implied full-Smith-shell flux | {p1:.3f} Jy | {p2:.3f} Jy |",
        "",
        "Area 2 uses the total observed LOS column (front plus rear) and total observed Area-2 flux for the empirical emissivity. Its saved global mass is the front-normalized mass; the rear crossing is not added as a second nebular mass.",
        "",
        "## Comparison",
        "",
        f"The Smith fraction retained above the hard cutoff is **{retained:.8f}**. The reconstructed whole-nebula profile integrates to **{observed_total:.3f} Jy**. After removing the excluded latitude band without renormalizing, the Area-1 calibration predicts **{converted1:.3f} Jy** ({100.0*converted1/observed_total:.2f}%), and Area 2 predicts **{converted2:.3f} Jy** ({100.0*converted2/observed_total:.2f}%). The corresponding full 0--90 degree values before the cutoff are {p1:.3f} and {p2:.3f} Jy.",
        "",
        "## Important algebraic check",
        "",
        "The LOS column is explicitly used to form `F/Sigma_LOS`. Because the latest Smith mass is itself proportional to the fitted column for a fixed LOS geometry, that same posterior column scale cancels in `eta_18 M_dust,total`. This is expected, not an omission. The absolute graph is controlled by the measured aperture flux and the fixed Smith/LOS geometric extrapolation. Therefore no artificial MCMC-column uncertainty band is drawn; the displayed band propagates only the adopted aperture-flux uncertainty.",
        "",
        "## Assumption and limitation",
        "",
        "This calculation assumes that the aperture's measured 18-micron flux per unit dust mass applies at every latitude. It is an empirical mass-weighted extrapolation, not a temperature-dependent radiative-transfer prediction. Smith-curve, distance, inclination, aperture-registration, gas-to-dust-ratio, and spatially varying temperature/opacity uncertainties are not propagated.",
        "",
        "## Numerical validation",
        "",
        f"- Area 1 flux outside the observed offset grid: {float(projected['area1_overflow_jy']):.6e} Jy.",
        f"- Area 2 flux outside the observed offset grid: {float(projected['area2_overflow_jy']):.6e} Jy.",
        f"- Projected shell extent: {float(projected['projected_min_offset_arcsec']):.6f} to {float(projected['projected_max_offset_arcsec']):.6f} arcsec.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "--observed-profile",
        type=Path,
        default=OBSERVED_DATA_DIR / "i18_major_axis_profile.csv",
    )
    parser.add_argument(
        "--latitude-min-deg",
        type=float,
        default=0.0,
        help="Hard absolute-latitude cutoff; excluded flux is not renormalized.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WHOLE_NEBULA_RESULTS / "profile_diagnostic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = read_parameter_medians(args.parameter_summary)
    masses = read_latest_corrected_masses(args.mass_summary)
    calibrations = build_calibrations(
        parameters,
        masses,
        nearest_flux_measurement(args.area1_data),
        nearest_flux_measurement(args.area2_data),
    )

    mass_latitude, mass_q = read_named_curve(args.mass_curve)
    velocity_latitude, velocity = read_named_curve(args.velocity_curve)
    mass_curve = PchipInterpolator(mass_latitude, mass_q, extrapolate=False)
    velocity_curve = PchipInterpolator(
        velocity_latitude, velocity, extrapolate=False
    )
    latitude_result = latitude_distribution(
        mass_curve, calibrations, args.latitude_min_deg
    )

    offset, observed, offset_edges = read_observed_profile(args.observed_profile)
    widths = np.diff(offset_edges)
    projected = project_to_major_axis(
        latitude_result, velocity_curve, offset_edges
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_csv = output_dir / "area_flux_per_column_calibrations.csv"
    latitude_csv = output_dir / "flux_distribution_by_latitude.csv"
    profile_csv = output_dir / "whole_nebula_vs_area_calibrated_profiles.csv"
    figure_png = output_dir / "whole_nebula_vs_area_calibrated_profiles.png"
    method_md = output_dir / "METHOD_AND_RESULTS.md"
    summary_json = output_dir / "summary.json"

    write_dict_rows(calibration_csv, [calibrations["area1"], calibrations["area2"]])
    write_aligned_csv(
        latitude_csv,
        [
            "absolute_latitude_deg",
            "latitude_bin_width_deg",
            "smith_dM_dOmega_raw",
            "normalized_full_shell_mass_fraction_per_latitude_bin",
            "area1_derived_flux_jy_per_latitude_bin",
            "area1_derived_flux_jy_per_degree",
            "area2_derived_flux_jy_per_latitude_bin",
            "area2_derived_flux_jy_per_degree",
        ],
        [
            np.asarray(latitude_result["latitude_deg"]),
            np.asarray(latitude_result["width_deg"]),
            np.asarray(latitude_result["q_raw"]),
            np.asarray(latitude_result["mass_fraction"]),
            np.asarray(latitude_result["area1_flux_jy_per_bin"]),
            np.asarray(latitude_result["area1_flux_jy_per_deg"]),
            np.asarray(latitude_result["area2_flux_jy_per_bin"]),
            np.asarray(latitude_result["area2_flux_jy_per_deg"]),
        ],
    )
    write_aligned_csv(
        profile_csv,
        [
            "major_axis_offset_arcsec",
            "offset_bin_width_arcsec",
            "observed_whole_nebula_flux_jy_per_arcsec",
            "observed_whole_nebula_flux_jy_per_bin",
            "area1_derived_flux_jy_per_arcsec",
            "area1_derived_flux_jy_per_bin",
            "area2_derived_flux_jy_per_arcsec",
            "area2_derived_flux_jy_per_bin",
        ],
        [
            offset,
            widths,
            observed,
            observed * widths,
            np.asarray(projected["area1_flux_jy_per_arcsec"]),
            np.asarray(projected["area1_flux_jy_per_bin"]),
            np.asarray(projected["area2_flux_jy_per_arcsec"]),
            np.asarray(projected["area2_flux_jy_per_bin"]),
        ],
    )
    make_figure(
        figure_png,
        latitude_result,
        offset,
        widths,
        observed,
        projected,
        calibrations,
    )
    observed_total = float(np.sum(observed * widths))
    write_method(
        method_md,
        calibrations,
        observed_total,
        projected,
        latitude_result,
    )
    retained_fraction = float(latitude_result["retained_mass_fraction"])
    area1_converted = float(
        np.sum(np.asarray(projected["area1_flux_jy_per_bin"]))
    )
    area2_converted = float(
        np.sum(np.asarray(projected["area2_flux_jy_per_bin"]))
    )
    summary = {
        "minimum_absolute_latitude_deg": args.latitude_min_deg,
        "retained_smith_mass_and_flux_fraction": retained_fraction,
        "observed_whole_nebula_integrated_flux_jy": observed_total,
        "area1_predicted_full_smith_shell_flux_jy": float(
            calibrations["area1"]["predicted_full_Smith_shell_flux_jy"]
        ),
        "area2_predicted_full_smith_shell_flux_jy": float(
            calibrations["area2"]["predicted_full_Smith_shell_flux_jy"]
        ),
        "area1_cutoff_converted_flux_jy": area1_converted,
        "area2_cutoff_converted_flux_jy": area2_converted,
        "area1_fraction_of_observed": area1_converted / observed_total,
        "area2_fraction_of_observed": area2_converted / observed_total,
        "area1_profile_integral_inside_observed_grid_jy": area1_converted,
        "area2_profile_integral_inside_observed_grid_jy": area2_converted,
        "area1_overflow_jy": float(projected["area1_overflow_jy"]),
        "area2_overflow_jy": float(projected["area2_overflow_jy"]),
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Observed whole-nebula integral: {observed_total:.6f} Jy")
    print(
        f"Hard cutoff |latitude| >= {args.latitude_min_deg:g} deg retains "
        f"{100.0*retained_fraction:.6f}% of the Smith mass/flux weight"
    )
    for area in ("area1", "area2"):
        calibration = calibrations[area]
        predicted_full = float(
            calibration["predicted_full_Smith_shell_flux_jy"]
        )
        predicted = area1_converted if area == "area1" else area2_converted
        print(
            f"{calibration['area']}: Sigma_LOS="
            f"{float(calibration['effective_sigma_LOS_dust_g_cm2']):.6e} "
            f"g/cm^2, F18={float(calibration['observed_integrated_flux_jy']):.6f} "
            f"Jy, full-shell prediction={predicted_full:.6f} Jy, "
            f"cutoff prediction={predicted:.6f} Jy "
            f"({100.0*predicted/observed_total:.3f}% of observed)"
        )
    print(f"Saved {figure_png}")
    print(f"Saved {latitude_csv}")
    print(f"Saved {profile_csv}")
    print(f"Saved {calibration_csv}")
    print(f"Saved {method_md}")


if __name__ == "__main__":
    main()
