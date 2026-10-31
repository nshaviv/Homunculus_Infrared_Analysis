#!/usr/bin/env python3
"""Infer one Area-1 18-micron effective temperature and map the STL front.

The Area-1 anchor is obtained from the observed one-square-arcsecond aperture
flux and the latest MCMC LOS optical depth.  Dense and rare columns/opacities
are used only to obtain that Area-1 optical depth.  Afterward there is one
effective-temperature field and no nebula-wide filling-fraction prescription:

    I_nu = B_nu(T_eff,1) [1-exp(-tau_18)]
    T_eff(R) = T_eff,1 (R_1/R)^0.4                 (beta=1).

Only the nearest visible mean-age STL surface is displayed.  No flux map is
calculated by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AREA1_REFERENCE_RADIUS_CM = 2.7970595086102778e17
TEMPERATURE_EXPONENT_BETA1 = 0.4
PA_DEG = 132.0


def read_parameter_medians(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return {
            row["parameter"]: float(row["posterior_median"])
            for row in csv.DictReader(stream)
        }


def infer_area1_effective_temperature(
    parameter_summary: Path,
    area1_data: Path,
) -> dict[str, float]:
    # Importing here keeps the scientific definitions in the exact MCMC code
    # that generated the saved posterior rather than duplicating Mie physics.
    import two_phase_aligned as model

    medians = read_parameter_medians(parameter_summary)
    theta = np.asarray([medians[name] for name in model.PARAMETER_NAMES])
    state = model.derive_state(theta)
    data = np.loadtxt(area1_data, delimiter=",")
    index = int(np.argmin(np.abs(data[:, 0] - 18.0)))
    wavelength_micron = float(data[index, 0])
    observed_flux_jy = float(data[index, 1])
    wavelength = np.asarray([wavelength_micron])
    kappas = model.build_kappas(state, wavelength, wavelength)
    kappa_dense = float(kappas["dense_1"][0])
    kappa_rare = float(kappas["rare_1"][0])
    sigma_dense = float(state["sigma_dense_1_weighted_los"])
    sigma_rare = float(state["sigma_rare_1_weighted_los"])
    sigma_total = float(state["sigma_total_1_los"])
    tau = kappa_dense * sigma_dense + kappa_rare * sigma_rare
    intensity = (
        observed_flux_jy * 1.0e-23 / model.base.PIXEL_SOLID_ANGLE_SR
    )
    planck_required = intensity / (-np.expm1(-tau))
    wavelength_cm = wavelength_micron * 1.0e-4
    frequency_hz = model.base.C_CGS / wavelength_cm
    effective_temperature = (
        model.base.H_CGS
        * frequency_hz
        / model.base.K_B_CGS
        / np.log1p(
            2.0
            * model.base.H_CGS
            * frequency_hz**3
            / (model.base.C_CGS**2 * planck_required)
        )
    )
    return {
        "area1_data_wavelength_micron": wavelength_micron,
        "area1_observed_flux_Jy_in_1_arcsec2": observed_flux_jy,
        "area1_sigma_total_LOS_g_cm2": sigma_total,
        "area1_sigma_dense_weighted_LOS_g_cm2": sigma_dense,
        "area1_sigma_rare_weighted_LOS_g_cm2": sigma_rare,
        "area1_kappa_dense_cm2_g": kappa_dense,
        "area1_kappa_rare_cm2_g": kappa_rare,
        "area1_tau": tau,
        "area1_observed_intensity_cgs_per_sr": intensity,
        "required_planck_function_cgs_per_sr": planck_required,
        "area1_effective_temperature_K": float(effective_temperature),
    }


def plot_temperature(
    output_png: Path,
    output_pdf: Path,
    axis: np.ndarray,
    temperature: np.ndarray,
    anchor: dict[str, float],
) -> None:
    temperature = np.flipud(temperature)
    finite = temperature[np.isfinite(temperature)]
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    figure, panel = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    image = panel.imshow(
        temperature,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="inferno",
        vmin=float(np.min(finite)),
        vmax=float(np.max(finite)),
        aspect="equal",
    )
    s = np.linspace(-10.0, 10.0, 101)
    pa = np.deg2rad(PA_DEG)
    panel.plot(-s * np.sin(pa), s * np.cos(pa), "w--", lw=0.75, alpha=0.8)
    panel.plot(0.0, 0.0, marker="+", color="white", ms=7, mew=1.0)
    panel.set(
        xlabel="R.A.-display offset (arcsec; right positive)",
        ylabel="Declination offset (arcsec)",
        title="Front-surface effective temperature",
    )
    panel.tick_params(direction="in", top=True, right=True)
    colorbar = figure.colorbar(image, ax=panel, location="bottom", pad=0.04)
    colorbar.set_label("Effective temperature (K)")
    figure.suptitle(
        "Area-1 18 µm brightness-temperature anchor: "
        f"{anchor['area1_effective_temperature_K']:.2f} K\n"
        r"$T(R)=T_1(R_1/R)^{0.4}$ ($\beta=1$); nearest STL surface only; no flux map",
        fontsize=13,
    )
    figure.savefig(output_png, dpi=240, facecolor="white")
    figure.savefig(output_pdf, facecolor="white")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    from project_paths import MODEL_INPUT_DIR, WHOLE_NEBULA_RESULTS

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameter-summary",
        type=Path,
        default=WHOLE_NEBULA_RESULTS
        / "parameters" / "beta_p1p00"
        / "parameter_summary.csv",
    )
    parser.add_argument("--area1-data", type=Path, default=MODEL_INPUT_DIR / "lobe.edge.data")
    parser.add_argument("--pixels", type=int, default=600)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "front_effective_temperature_2d_output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    root = Path(__file__).resolve().parent
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor = infer_area1_effective_temperature(
        args.parameter_summary.expanduser().resolve(),
        args.area1_data.expanduser().resolve(),
    )
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/mpl_eta_front_temperature")
    subprocess.run(
        [
            "julia",
            str(root / "build_stl_front_effective_temperature.jl"),
            str(output_dir),
            str(anchor["area1_effective_temperature_K"]),
            str(args.pixels),
        ],
        check=True,
        cwd=root,
        env=environment,
    )
    axis = np.loadtxt(output_dir / "sky_axis_arcsec.csv", delimiter=",")
    temperature = np.loadtxt(
        output_dir / "front_effective_temperature_K.csv", delimiter=","
    )
    radius = np.loadtxt(output_dir / "front_radius_cm.csv", delimiter=",")
    png = output_dir / "front_effective_temperature_2d.png"
    pdf = output_dir / "front_effective_temperature_2d.pdf"
    npz = output_dir / "front_effective_temperature_2d.npz"
    summary = output_dir / "summary.json"
    plot_temperature(png, pdf, axis, temperature, anchor)
    np.savez_compressed(
        npz,
        sky_axis_arcsec=axis,
        front_effective_temperature_K=np.flipud(temperature),
        front_radius_cm=np.flipud(radius),
    )
    result = {
        "calculation": "Area-1 18-micron effective brightness temperature propagated over the nearest STL surface",
        "no_flux_map_calculated": True,
        "anchor": anchor,
        "temperature_law": {
            "emissivity_beta": 1.0,
            "distance_exponent_2_over_4_plus_beta": TEMPERATURE_EXPONENT_BETA1,
            "formula": "T(R)=T_eff,Area1*(R_Area1/R)^0.4",
            "area1_reference_radius_cm": AREA1_REFERENCE_RADIUS_CM,
        },
        "projection": {
            "surface": "nearest visible mean-age STL intersection",
            "inclination_from_line_of_sight_deg": 41.0,
            "position_angle_deg": PA_DEG,
            "dense_rare_fraction_outside_area1": "not used",
        },
        "files": {
            "figure_png": str(png),
            "figure_pdf": str(pdf),
            "arrays_npz": str(npz),
            "projection_metadata": str(output_dir / "projection_metadata.txt"),
        },
    }
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, value in anchor.items():
        print(f"{name} = {value:.12e}")
    for path in (png, pdf, npz, summary):
        print(path)


if __name__ == "__main__":
    main()
