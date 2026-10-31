#!/usr/bin/env python3
"""Fast numerical validation of the archived publication release."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import calculate_single_phase_mass as single_mass  # noqa: E402


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"{label}: got {actual}, expected {expected}")


def validate_single_phase_mass() -> None:
    velocity_latitude, velocity = single_mass.load_two_column_csv(
        single_mass.SMITH_GEOMETRY_DIR / "v_vs_theta.csv", "theta", "v"
    )
    mass_latitude, mass_shape = single_mass.load_two_column_csv(
        single_mass.SMITH_GEOMETRY_DIR / "Mass-per-SolidAngle.csv",
        "theta",
        "dMdOmega",
    )
    expected = {
        "Area 1": (4.652401, 1.378061),
        "Area 2": (2.631216, 1.089662),
    }
    for anchor in (single_mass.AREA_1, single_mass.AREA_2):
        result = single_mass.calculate_area_mass(
            anchor,
            velocity_latitude,
            velocity,
            mass_latitude,
            mass_shape,
            kappa_18_cm2_g_dust=1.2083e3,
            total_to_dust=100.0,
            distance_pc=2300.0,
            age_years=160.0,
        )
        uncorr, corr = expected[anchor.name]
        assert_close(float(result["mass_uncorrected_msun"]), uncorr, 1.0e-6, anchor.name)
        assert_close(float(result["mass_los_corrected_msun"]), corr, 1.0e-6, anchor.name)


def validate_two_phase_archive() -> None:
    rows = read_csv_rows(ROOT / "results" / "two_phase" / "mass_summary.csv")
    medians = {
        row["mass_quantity"]: float(row["posterior_median_Msun"])
        for row in rows
    }
    expected = {
        "area1_total_mass_without_LOS_correction": 8.942000542552375,
        "area1_total_mass_with_LOS_correction": 2.6486593243459775,
        "area2_FRONT_ONLY_mass_without_LOS_correction": 9.880667117289462,
        "area2_FRONT_ONLY_mass_with_LOS_correction": 6.928915841062993,
    }
    for key, value in expected.items():
        assert_close(medians[key], value, 1.0e-12, key)

    with np.load(
        ROOT / "results" / "two_phase" / "posterior_samples.npz",
        allow_pickle=False,
    ) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
    density_ratio = samples[:, 5] - samples[:, 6]
    assert_close(float(np.median(density_ratio)), 2.006, 0.002, "log density ratio")


def validate_hidden_component() -> None:
    rows = read_csv_rows(
        ROOT / "results" / "hidden_component" / "selected_800K.csv"
    )
    if len(rows) != 1:
        raise AssertionError("Expected exactly one selected 800 K row")
    row = rows[0]
    expected = {
        "T_h_K": 800.0,
        "tau_h_100": 1.1815039383170023,
        "tau_f_100": 0.6714739310835411,
        "required_area_arcsec2": 2.397315816851676,
        "Mdust_hot_Msun": 0.03390659919975044,
        "Mdust_combined_Msun": 0.05317644394251243,
    }
    for key, value in expected.items():
        assert_close(float(row[key]), value, 1.0e-10, key)


def validate_portability() -> None:
    forbidden = ("/Users/", "Documents/ChatGPT", "ηCar ")
    offenders: list[str] = []
    for path in sorted(SRC.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(path.name)
    if offenders:
        raise AssertionError(f"Absolute local paths remain in: {offenders}")


def main() -> None:
    validate_single_phase_mass()
    validate_two_phase_archive()
    validate_hidden_component()
    validate_portability()
    print("Publication release validation passed.")


if __name__ == "__main__":
    main()
