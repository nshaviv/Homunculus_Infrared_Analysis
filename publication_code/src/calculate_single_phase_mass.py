#!/usr/bin/env python3
"""Infer the two-lobe mass from the homogeneous single-phase model.

The observed optical depth is first converted to a line-of-sight dust column,
``Sigma_dust_LOS = tau_18 / kappa_18``.  A fixed total-to-dust factor converts
that column to total material.  The uncorrected calculation temporarily treats
the LOS column as radial.  The corrected calculation applies the deterministic
Figure-2 conversion exactly once before normalizing and integrating the Smith
et al. (2006) mass-per-solid-angle profile.

This script reproduces the four single-phase masses reported in the paper:
4.65 and 2.63 solar masses before the LOS correction, and 1.38 and 1.09 solar
masses afterward for Areas 1 and 2, respectively.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from project_paths import RESULTS_ROOT, SMITH_GEOMETRY_DIR


M_SUN_G = 1.989e33
PC_CM = 3.086e18
ARCSEC_PER_RADIAN = 180.0 * 3600.0 / np.pi


@dataclass(frozen=True)
class AreaAnchor:
    """Fixed quantities for one local mass normalization."""

    name: str
    latitude_deg: float
    tau_18: float
    k_radial_g_cm2_per_msun: float
    k_los_g_cm2_per_msun: float

    @property
    def los_to_radial(self) -> float:
        return self.k_radial_g_cm2_per_msun / self.k_los_g_cm2_per_msun


AREA_1 = AreaAnchor(
    name="Area 1",
    latitude_deg=60.0,
    tau_18=0.30,
    k_radial_g_cm2_per_msun=0.005758514010801346,
    k_los_g_cm2_per_msun=0.01944101868276169,
)
AREA_2 = AreaAnchor(
    name="Area 2",
    latitude_deg=80.0,
    tau_18=0.10,
    k_radial_g_cm2_per_msun=0.0034847470034176735,
    k_los_g_cm2_per_msun=0.008414650086086446,
)


def load_two_column_csv(path: Path, x_name: str, y_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read two named numeric columns from a CSV file."""
    table = np.genfromtxt(path, delimiter=",", names=True)
    names = set(table.dtype.names or ())
    missing = {x_name, y_name} - names
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    x = np.asarray(table[x_name], dtype=float)
    y = np.asarray(table[y_name], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        raise ValueError(f"{path} does not contain enough finite rows")
    order = np.argsort(x[valid])
    return x[valid][order], y[valid][order]


def radius_from_velocity(
    velocity_km_s: float,
    distance_pc: float,
    age_years: float,
) -> tuple[float, float]:
    """Return ballistic radius in centimetres and projected arcseconds."""
    age_s = age_years * 365.25 * 24.0 * 3600.0
    radius_cm = velocity_km_s * 1.0e5 * age_s
    angular_radius_arcsec = radius_cm / (distance_pc * PC_CM) * ARCSEC_PER_RADIAN
    return radius_cm, angular_radius_arcsec


def latitude_shape_integral(
    latitude_deg: np.ndarray,
    relative_dmdomega: np.ndarray,
    anchor_latitude_deg: float,
) -> float:
    """Integrate S(theta)/S(theta_anchor) over d[sin(theta)]."""
    anchor = float(np.interp(anchor_latitude_deg, latitude_deg, relative_dmdomega))
    if not np.isfinite(anchor) or anchor <= 0.0:
        raise ValueError("Mass-profile value at the anchor latitude is not positive")
    normalized = relative_dmdomega / anchor
    mu = np.sin(np.radians(latitude_deg))
    order = np.argsort(mu)
    return float(np.trapezoid(normalized[order], x=mu[order]))


def calculate_area_mass(
    anchor: AreaAnchor,
    velocity_latitude_deg: np.ndarray,
    velocity_km_s: np.ndarray,
    mass_latitude_deg: np.ndarray,
    relative_dmdomega: np.ndarray,
    *,
    kappa_18_cm2_g_dust: float,
    total_to_dust: float,
    distance_pc: float,
    age_years: float,
) -> dict[str, float | str]:
    """Calculate uncorrected and LOS-corrected global lobe masses."""
    velocity = float(
        np.interp(anchor.latitude_deg, velocity_latitude_deg, velocity_km_s)
    )
    radius_cm, radius_arcsec = radius_from_velocity(
        velocity, distance_pc, age_years
    )
    sigma_dust_los = anchor.tau_18 / kappa_18_cm2_g_dust
    sigma_total_los = total_to_dust * sigma_dust_los
    sigma_total_radial = anchor.los_to_radial * sigma_total_los
    shape_integral = latitude_shape_integral(
        mass_latitude_deg, relative_dmdomega, anchor.latitude_deg
    )

    geometric_factor = 4.0 * np.pi * radius_cm**2 * shape_integral / M_SUN_G
    mass_uncorrected = geometric_factor * sigma_total_los
    mass_corrected = geometric_factor * sigma_total_radial

    if not np.isclose(
        mass_corrected / mass_uncorrected,
        anchor.los_to_radial,
        rtol=1.0e-13,
    ):
        raise RuntimeError("LOS correction was not applied exactly once")

    return {
        "area": anchor.name,
        "latitude_deg": anchor.latitude_deg,
        "tau_18": anchor.tau_18,
        "kappa_18_cm2_g_dust": kappa_18_cm2_g_dust,
        "velocity_km_s": velocity,
        "radius_arcsec": radius_arcsec,
        "sigma_total_los_g_cm2": sigma_total_los,
        "los_to_radial_factor": anchor.los_to_radial,
        "sigma_total_radial_g_cm2": sigma_total_radial,
        "mass_uncorrected_msun": mass_uncorrected,
        "mass_los_corrected_msun": mass_corrected,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--velocity-curve",
        type=Path,
        default=SMITH_GEOMETRY_DIR / "v_vs_theta.csv",
    )
    parser.add_argument(
        "--mass-curve",
        type=Path,
        default=SMITH_GEOMETRY_DIR / "Mass-per-SolidAngle.csv",
    )
    parser.add_argument("--kappa-18", type=float, default=1.2083e3)
    parser.add_argument("--total-to-dust", type=float, default=100.0)
    parser.add_argument("--distance-pc", type=float, default=2300.0)
    parser.add_argument("--age-years", type=float, default=160.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "single_phase_mass.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    velocity_latitude, velocity = load_two_column_csv(
        args.velocity_curve, "theta", "v"
    )
    mass_latitude, mass_shape = load_two_column_csv(
        args.mass_curve, "theta", "dMdOmega"
    )
    results = [
        calculate_area_mass(
            area,
            velocity_latitude,
            velocity,
            mass_latitude,
            mass_shape,
            kappa_18_cm2_g_dust=args.kappa_18,
            total_to_dust=args.total_to_dust,
            distance_pc=args.distance_pc,
            age_years=args.age_years,
        )
        for area in (AREA_1, AREA_2)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    for result in results:
        print(
            f"{result['area']}: "
            f"M_uncorrected={result['mass_uncorrected_msun']:.6f} Msun, "
            f"C_LOS={result['los_to_radial_factor']:.12f}, "
            f"M_corrected={result['mass_los_corrected_msun']:.6f} Msun"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
