#!/usr/bin/env python3
"""Interactive calculation for a hot component behind a foreground shell.

This is a deterministic calculation, not an MCMC.  You choose the temperature
and the background optical depth.  The projected area is set by the 100-micron
deficit, while the foreground optical depth is automatically chosen to give
the best baseline+hidden agreement with the observation from 5 to 7 micron.

Radiative-transfer equation
---------------------------

    F_hidden(nu) = Omega * B_nu(T_hot)
                   * exp[-tau_foreground(nu)]
                   * {1 - exp[-tau_hot(nu)]}

where

    tau_foreground(nu) = kappa_foreground(nu) * Sigma_foreground
    tau_hot(nu)        = kappa_hot(nu)        * Sigma_hot.

The default experiment is:

    tau_foreground(100 micron) = solved from the 5--7 micron spectrum
    tau_hot(100 micron)        = 1

The present version uses the same Area-2 dense-component Mie opacity curve for
the foreground and hot component.  The two opacity arrays are assigned on
separate lines below, so they can easily be replaced independently later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from project_paths import HIDDEN_COMPONENT_RESULTS, OBSERVED_DATA_DIR

# ---------------------------------------------------------------------------
# USER SETTINGS
# These are also available as command-line options; see --help.
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE_K = 688.745

# Optical depth of the foreground shell at 100 micron.
DEFAULT_TAU_FOREGROUND = 1.0
DEFAULT_FOREGROUND_REFERENCE_MICRON = 100.0

# The foreground tau is fitted in this band for every selected temperature.
DEFAULT_FOREGROUND_FIT_MIN_MICRON = 5.0
DEFAULT_FOREGROUND_FIT_MAX_MICRON = 7.0

# Optical depth of the hot background component at 100 micron.
DEFAULT_TAU_HOT = 1.0
DEFAULT_HOT_REFERENCE_MICRON = 100.0

# The area is solved so that the missing flux is closed here.
DEFAULT_CLOSURE_MICRON = 100.0

# This conversion is not constrained by the infrared fit.  It follows the
# paper convention: total material mass = 100 * dust mass.
DEFAULT_TOTAL_TO_DUST = 100.0

# Display the graph interactively.  Set --save-figure to also save a PNG.
DEFAULT_OUTPUT = HIDDEN_COMPONENT_RESULTS / "interactive_experiment"


# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS AND LOCAL INPUT TABLE
# The script is intentionally self-contained apart from NumPy, Matplotlib, and
# one CSV containing wavelength, observed flux, baseline flux, and Mie opacity.
# ---------------------------------------------------------------------------
# Keep the same constants used by the MCMC/forward-model files so the
# standalone calculation reproduces their numerical results exactly.
H_CGS = 6.626e-27
C_CGS = 3.0e10
K_B_CGS = 1.38e-16
JY_CGS = 1.0e-23
ARCSEC2_SR = (np.pi / (180.0 * 3600.0)) ** 2
DISTANCE_PC = 2300.0
PARSEC_CM = 3.0856775814913673e18
DISTANCE_CM = DISTANCE_PC * PARSEC_CM
M_SUN_G = 1.98847e33
INPUT_FILENAME = "hidden_component_area2_beta1_input.csv"


def planck_nu(wavelength_micron: np.ndarray, temperature_K: float) -> np.ndarray:
    """Planck specific intensity B_nu in cgs per steradian."""
    wavelength_cm = np.asarray(wavelength_micron, dtype=float) * 1.0e-4
    exponent = H_CGS * C_CGS / (wavelength_cm * K_B_CGS * temperature_K)
    exponent = np.minimum(exponent, 700.0)
    b_lambda = (
        2.0
        * H_CGS
        * C_CGS**2
        / wavelength_cm**5
        / np.expm1(exponent)
    )
    return b_lambda * wavelength_cm**2 / C_CGS


def positive_log_interpolate(
    wavelength_micron: np.ndarray,
    values: np.ndarray,
    requested_wavelength_micron: float,
) -> float:
    """Log-log interpolate a positive spectrum or opacity curve."""
    if requested_wavelength_micron < wavelength_micron.min() or requested_wavelength_micron > wavelength_micron.max():
        raise ValueError(
            f"Requested wavelength {requested_wavelength_micron:g} micron is "
            f"outside {wavelength_micron.min():g}--{wavelength_micron.max():g} micron."
        )
    valid = np.isfinite(wavelength_micron) & np.isfinite(values) & (values > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Log interpolation requires at least two positive values.")
    valid_wavelength = wavelength_micron[valid]
    valid_values = values[valid]
    if (
        requested_wavelength_micron < valid_wavelength.min()
        or requested_wavelength_micron > valid_wavelength.max()
    ):
        raise ValueError(
            "Requested wavelength is outside the positive part of the curve."
        )
    return float(
        np.exp(
            np.interp(
                np.log(requested_wavelength_micron),
                np.log(valid_wavelength),
                np.log(valid_values),
            )
        )
    )


def load_area2_beta1_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the frozen Area-2 baseline, observation, and fitted Mie opacity."""
    script_directory = Path(__file__).resolve().parent
    candidates = (
        OBSERVED_DATA_DIR / INPUT_FILENAME,
        script_directory / INPUT_FILENAME,
        script_directory
        / "hottest_opaque_hidden_component_output"
        / "hottest_solution_spectrum.csv",
    )
    input_path = next((path for path in candidates if path.exists()), None)
    if input_path is None:
        searched = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            f"Could not find {INPUT_FILENAME}. Searched:\n{searched}\n"
            "Keep the CSV beside this script."
        )

    table = np.genfromtxt(input_path, delimiter=",", names=True)
    required = {
        "wavelength_micron",
        "observed_flux_Jy",
        "baseline_flux_Jy",
        "kappa_cm2_g_dust",
    }
    names = set(table.dtype.names or ())
    if not required.issubset(names):
        raise ValueError(
            f"Input table {input_path} is missing columns: {sorted(required - names)}"
        )

    wavelength = np.asarray(table["wavelength_micron"], dtype=float)
    observed_flux = np.asarray(table["observed_flux_Jy"], dtype=float)
    baseline_flux = np.asarray(table["baseline_flux_Jy"], dtype=float)
    kappa = np.asarray(table["kappa_cm2_g_dust"], dtype=float)
    keep = (
        np.isfinite(wavelength)
        & np.isfinite(observed_flux)
        & np.isfinite(baseline_flux)
        & np.isfinite(kappa)
        & (wavelength > 0.0)
        & (observed_flux > 0.0)
        & (baseline_flux > 0.0)
        & (kappa > 0.0)
    )
    order = np.argsort(wavelength[keep])
    print(f"Input spectrum and opacity: {input_path}")
    print("Opacity curve: Area 2 front-dense fitted Mie opacity")
    return (
        wavelength[keep][order],
        observed_flux[keep][order],
        baseline_flux[keep][order],
        kappa[keep][order],
    )


def evaluate_emission(
    args: argparse.Namespace,
    wavelength: np.ndarray,
    observed_flux: np.ndarray,
    baseline_flux: np.ndarray,
    kappa_foreground: np.ndarray,
    kappa_hot: np.ndarray,
    temperature_K: float,
    tau_foreground_reference: float,
    tau_hot_reference: float,
) -> dict[str, float | np.ndarray]:
    """Evaluate one user-selected point; no fitting or optimization occurs."""
    kappa_foreground_reference = positive_log_interpolate(
        wavelength,
        kappa_foreground,
        args.foreground_reference_micron,
    )
    kappa_hot_reference = positive_log_interpolate(
        wavelength,
        kappa_hot,
        args.hot_reference_micron,
    )

    # tau = kappa * Sigma, so Sigma = tau / kappa.
    sigma_foreground = tau_foreground_reference / kappa_foreground_reference
    sigma_hot = tau_hot_reference / kappa_hot_reference

    tau_foreground = kappa_foreground * sigma_foreground
    tau_hot = kappa_hot * sigma_hot

    # Intrinsic hot-component intensity and the intensity transmitted through
    # the foreground.  Keeping both lets the plot show exactly what the screen
    # removes at every wavelength.
    intrinsic_source_per_sr_jy = (
        planck_nu(wavelength, temperature_K)
        * (-np.expm1(-tau_hot))
        / JY_CGS
    )
    transmitted_source_per_sr_jy = intrinsic_source_per_sr_jy * np.exp(
        -tau_foreground
    )

    observed_at_closure = positive_log_interpolate(
        wavelength,
        observed_flux,
        args.closure_micron,
    )
    baseline_at_closure = positive_log_interpolate(
        wavelength,
        baseline_flux,
        args.closure_micron,
    )
    source_at_closure = positive_log_interpolate(
        wavelength,
        transmitted_source_per_sr_jy,
        args.closure_micron,
    )
    missing_flux = observed_at_closure - baseline_at_closure
    if missing_flux <= 0.0:
        raise ValueError(
            "The baseline has no positive flux deficit at the closure wavelength."
        )

    # This is the only solved quantity.  It closes the selected flux deficit.
    omega_sr = missing_flux / source_at_closure
    area_arcsec2 = omega_sr / ARCSEC2_SR

    intrinsic_hot_flux = omega_sr * intrinsic_source_per_sr_jy
    hidden_flux = omega_sr * transmitted_source_per_sr_jy
    total_flux = baseline_flux + hidden_flux

    physical_projected_area_cm2 = omega_sr * DISTANCE_CM**2
    hot_dust_mass = sigma_hot * physical_projected_area_cm2 / M_SUN_G
    foreground_dust_mass = (
        sigma_foreground * physical_projected_area_cm2 / M_SUN_G
    )
    combined_dust_mass = hot_dust_mass + foreground_dust_mass
    total_multiplier = args.total_to_dust

    short = wavelength <= args.shortwave_max_micron
    short_fraction = hidden_flux[short] / observed_flux[short]
    worst_short_index_local = int(np.argmax(short_fraction))
    short_wavelengths = wavelength[short]

    foreground_fit_min = getattr(
        args, "foreground_fit_min_micron", DEFAULT_FOREGROUND_FIT_MIN_MICRON
    )
    foreground_fit_max = getattr(
        args, "foreground_fit_max_micron", DEFAULT_FOREGROUND_FIT_MAX_MICRON
    )
    foreground_fit_band = (
        (wavelength >= foreground_fit_min)
        & (wavelength <= foreground_fit_max)
    )
    if not np.any(foreground_fit_band):
        raise ValueError("No wavelength samples lie inside the foreground-fit band.")
    foreground_fit_log_residual = np.log(
        total_flux[foreground_fit_band] / observed_flux[foreground_fit_band]
    )
    foreground_fit_fractional_residual = (
        total_flux[foreground_fit_band] / observed_flux[foreground_fit_band] - 1.0
    )

    tau_foreground_at_closure = positive_log_interpolate(
        wavelength,
        tau_foreground,
        args.closure_micron,
    )
    tau_hot_at_closure = positive_log_interpolate(
        wavelength,
        tau_hot,
        args.closure_micron,
    )

    return {
        "temperature_K": float(temperature_K),
        "tau_foreground_reference": float(tau_foreground_reference),
        "tau_hot_reference": float(tau_hot_reference),
        "wavelength": wavelength,
        "observed_flux": observed_flux,
        "baseline_flux": baseline_flux,
        "kappa_foreground": kappa_foreground,
        "kappa_hot": kappa_hot,
        "intrinsic_hot_flux": intrinsic_hot_flux,
        "hidden_flux": hidden_flux,
        "total_flux": total_flux,
        "tau_foreground": tau_foreground,
        "tau_hot": tau_hot,
        "sigma_foreground_g_cm2": sigma_foreground,
        "sigma_hot_g_cm2": sigma_hot,
        "tau_foreground_at_closure": tau_foreground_at_closure,
        "tau_hot_at_closure": tau_hot_at_closure,
        "foreground_transmission_at_closure": np.exp(-tau_foreground_at_closure),
        "projected_area_arcsec2": area_arcsec2,
        "physical_projected_area_cm2": physical_projected_area_cm2,
        "missing_flux_Jy": missing_flux,
        "hot_dust_mass_Msun": hot_dust_mass,
        "foreground_dust_mass_Msun": foreground_dust_mass,
        "combined_dust_mass_Msun": combined_dust_mass,
        "hot_total_mass_Msun": total_multiplier * hot_dust_mass,
        "foreground_total_mass_Msun": total_multiplier
        * foreground_dust_mass,
        "combined_total_mass_Msun": total_multiplier
        * combined_dust_mass,
        "maximum_shortwave_fraction": float(short_fraction[worst_short_index_local]),
        "maximum_shortwave_fraction_wavelength": float(
            short_wavelengths[worst_short_index_local]
        ),
        "foreground_fit_rms_log_residual": float(
            np.sqrt(np.mean(foreground_fit_log_residual**2))
        ),
        "foreground_fit_maximum_absolute_fractional_residual": float(
            np.max(np.abs(foreground_fit_fractional_residual))
        ),
    }


def fit_foreground_tau(
    args: argparse.Namespace,
    wavelength: np.ndarray,
    observed_flux: np.ndarray,
    baseline_flux: np.ndarray,
    kappa_foreground: np.ndarray,
    kappa_hot: np.ndarray,
    temperature_K: float,
    tau_hot_reference: float,
) -> dict[str, float | np.ndarray]:
    """Solve tau_foreground,100 from the 5--7 micron spectrum.

    One scalar optical depth cannot generally force every wavelength sample to
    agree exactly.  Therefore the objective is the RMS logarithmic residual of
    baseline+hidden relative to observed flux across the entire selected band.
    A coarse scan followed by a golden-section refinement keeps the script
    independent of SciPy and makes the global one-dimensional search explicit.
    """

    lower = args.log10_tau_slider_min
    upper = args.log10_tau_slider_max

    def objective(log10_tau_foreground: float) -> float:
        trial = evaluate_emission(
            args,
            wavelength,
            observed_flux,
            baseline_flux,
            kappa_foreground,
            kappa_hot,
            temperature_K,
            10.0**log10_tau_foreground,
            tau_hot_reference,
        )
        return float(trial["foreground_fit_rms_log_residual"])

    grid = np.linspace(lower, upper, 161)
    grid_objective = np.asarray([objective(value) for value in grid])
    best_index = int(np.argmin(grid_objective))
    left = grid[max(0, best_index - 1)]
    right = grid[min(len(grid) - 1, best_index + 1)]

    golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
    c_value = right - golden_ratio * (right - left)
    d_value = left + golden_ratio * (right - left)
    c_objective = objective(c_value)
    d_objective = objective(d_value)
    for _ in range(60):
        if c_objective <= d_objective:
            right = d_value
            d_value = c_value
            d_objective = c_objective
            c_value = right - golden_ratio * (right - left)
            c_objective = objective(c_value)
        else:
            left = c_value
            c_value = d_value
            c_objective = d_objective
            d_value = left + golden_ratio * (right - left)
            d_objective = objective(d_value)

    best_log10_tau = 0.5 * (left + right)
    return evaluate_emission(
        args,
        wavelength,
        observed_flux,
        baseline_flux,
        kappa_foreground,
        kappa_hot,
        temperature_K,
        10.0**best_log10_tau,
        tau_hot_reference,
    )


def calculate(args: argparse.Namespace) -> dict[str, float | np.ndarray]:
    wavelength, observed_flux, baseline_flux, fitted_dense_kappa = (
        load_area2_beta1_case()
    )

    # CURRENT OPACITY ASSUMPTION:
    # The foreground and background use the same fitted dense Mie opacity.
    # To try different foreground grains, replace only kappa_foreground.
    kappa_foreground = fitted_dense_kappa.copy()
    kappa_hot = fitted_dense_kappa.copy()

    if args.manual_foreground:
        return evaluate_emission(
            args,
            wavelength,
            observed_flux,
            baseline_flux,
            kappa_foreground,
            kappa_hot,
            args.temperature_K,
            args.tau_foreground,
            args.tau_hot,
        )
    return fit_foreground_tau(
        args,
        wavelength,
        observed_flux,
        baseline_flux,
        kappa_foreground,
        kappa_hot,
        args.temperature_K,
        args.tau_hot,
    )


def print_results(args: argparse.Namespace, result: dict[str, float | np.ndarray]) -> None:
    print("\n" + "=" * 66)
    print("FIXED-OPTICAL-DEPTH HIDDEN-COMPONENT CALCULATION")
    print("=" * 66)
    print(
        f"Hot temperature                         : "
        f"{float(result['temperature_K']):.6g} K"
    )
    foreground_mode = "manual" if args.manual_foreground else (
        f"best fit over {args.foreground_fit_min_micron:g}--"
        f"{args.foreground_fit_max_micron:g} um"
    )
    print(
        f"Foreground tau({args.foreground_reference_micron:g} um)              : "
        f"{float(result['tau_foreground_reference']):.8g} ({foreground_mode})"
    )
    print(
        f"Hot-component condition                 : tau({args.hot_reference_micron:g} um) "
        f"= {args.tau_hot:g}"
    )
    print(f"Flux-closure wavelength                 : {args.closure_micron:g} um")
    print(
        f"RMS log residual in foreground-fit band : "
        f"{float(result['foreground_fit_rms_log_residual']):.8g}"
    )
    print(
        f"Maximum fractional residual in band     : "
        f"{float(result['foreground_fit_maximum_absolute_fractional_residual']):.8g}"
    )
    print("-" * 66)
    print(
        f"Foreground tau at closure wavelength    : "
        f"{float(result['tau_foreground_at_closure']):.8g}"
    )
    print(
        f"Foreground transmission at closure      : "
        f"{float(result['foreground_transmission_at_closure']):.8g}"
    )
    print(
        f"Hot tau at closure wavelength           : "
        f"{float(result['tau_hot_at_closure']):.8g}"
    )
    print(f"Flux deficit closed                     : {float(result['missing_flux_Jy']):.8g} Jy")
    print(f"Required projected area                 : {float(result['projected_area_arcsec2']):.8g} arcsec^2")
    print("-" * 66)
    print(f"Hot dust column                         : {float(result['sigma_hot_g_cm2']):.8g} g/cm^2")
    print(f"Foreground dust column                  : {float(result['sigma_foreground_g_cm2']):.8g} g/cm^2")
    print(f"HOT DUST MASS                           : {float(result['hot_dust_mass_Msun']):.8g} Msun")
    print(f"FOREGROUND DUST MASS                    : {float(result['foreground_dust_mass_Msun']):.8g} Msun")
    print(f"COMBINED DUST MASS                      : {float(result['combined_dust_mass_Msun']):.8g} Msun")
    print(
        f"Hot total mass (total/dust={args.total_to_dust:g})       : "
        f"{float(result['hot_total_mass_Msun']):.8g} Msun"
    )
    print(
        f"Combined total mass                    : "
        f"{float(result['combined_total_mass_Msun']):.8g} Msun"
    )
    print("-" * 66)
    print(
        "Largest hidden/observed fraction below "
        f"{args.shortwave_max_micron:g} um       : "
        f"{float(result['maximum_shortwave_fraction']):.8g} "
        f"at {float(result['maximum_shortwave_fraction_wavelength']):.6g} um"
    )
    print("A value above 1 means the hidden component alone exceeds the observation.")
    print("=" * 66 + "\n")


def plot_results(
    args: argparse.Namespace,
    result: dict[str, float | np.ndarray],
) -> None:
    wavelength = np.asarray(result["wavelength"])
    figure, axis = plt.subplots(figsize=(12.0, 7.6))
    figure.subplots_adjust(
        left=0.09,
        right=0.97,
        top=0.90,
        bottom=0.29 if not args.no_sliders else 0.12,
    )
    observed_line, = axis.plot(
        wavelength,
        result["observed_flux"],
        color="black",
        lw=2.4,
        label="Observed",
    )
    baseline_line, = axis.plot(
        wavelength,
        result["baseline_flux"],
        color="#126782",
        lw=2.0,
        label="Area 2 baseline",
    )
    intrinsic_line, = axis.plot(
        wavelength,
        result["intrinsic_hot_flux"],
        color="#e76f51",
        lw=1.5,
        ls=":",
        label="Hot component before foreground",
    )
    transmitted_line, = axis.plot(
        wavelength,
        result["hidden_flux"],
        color="#7b2cbf",
        lw=1.8,
        ls="--",
        label="Hidden component after foreground",
    )
    total_line, = axis.plot(
        wavelength,
        result["total_flux"],
        color="#c92a2a",
        lw=2.1,
        label="Baseline + hidden",
    )
    fit_band = (
        (wavelength >= args.foreground_fit_min_micron)
        & (wavelength <= args.foreground_fit_max_micron)
    )
    fit_axis = axis.inset_axes([0.58, 0.55, 0.39, 0.40])
    fit_axis.plot(
        wavelength[fit_band],
        np.asarray(result["observed_flux"])[fit_band],
        color="black",
        lw=2.0,
        label="Observed",
    )
    fit_axis.plot(
        wavelength[fit_band],
        np.asarray(result["baseline_flux"])[fit_band],
        color="#126782",
        lw=1.5,
        label="Baseline",
    )
    fit_total_line, = fit_axis.plot(
        wavelength[fit_band],
        np.asarray(result["total_flux"])[fit_band],
        color="#c92a2a",
        lw=1.8,
        label="Baseline + hidden",
    )
    fit_axis.set_title("Foreground-tau fit: 5–7 µm", fontsize=9)
    fit_axis.set_xlabel(r"Wavelength ($\mu$m)", fontsize=8)
    fit_axis.set_ylabel(r"$F_\nu$ (Jy)", fontsize=8)
    fit_axis.tick_params(labelsize=8)
    fit_axis.grid(alpha=0.20)
    axis.axvspan(
        args.foreground_fit_min_micron,
        args.foreground_fit_max_micron,
        color="0.75",
        alpha=0.18,
        label="Foreground-tau fit band",
    )
    axis.axvline(
        args.foreground_reference_micron,
        color="0.45",
        lw=1.0,
        ls=":",
        label="Foreground tau reference",
    )
    axis.axvline(
        args.hot_reference_micron,
        color="0.65",
        lw=1.0,
        ls="--",
        label="Hot tau reference",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(wavelength.min(), wavelength.max())
    axis.grid(alpha=0.20, which="both")
    axis.set_xlabel(r"Wavelength ($\mu$m)")
    axis.set_ylabel(r"Whole-nebula $F_\nu$ (Jy)")
    title = axis.set_title("")
    axis.legend(frameon=False, fontsize=9)

    diagnostics = axis.text(
        0.015,
        0.025,
        "",
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
    )

    def update_labels(current: dict[str, float | np.ndarray]) -> None:
        tau_label = "Manual" if args.manual_foreground else "Fitted"
        title.set_text(
            rf"$T={float(current['temperature_K']):.1f}$ K; "
            rf"$\tau_f({args.foreground_reference_micron:g}\,\mu m)="
            rf"{float(current['tau_foreground_reference']):.3g}$; "
            rf"$\tau_h({args.hot_reference_micron:g}\,\mu m)="
            rf"{float(current['tau_hot_reference']):.3g}$"
        )
        diagnostics.set_text(
            rf"{tau_label} $\tau_{{f,100}}$ = "
            f"{float(current['tau_foreground_reference']):.5g}\n"
            "5–7 µm RMS log residual = "
            f"{float(current['foreground_fit_rms_log_residual']):.4g}\n"
            "5–7 µm maximum residual = "
            f"{100.0 * float(current['foreground_fit_maximum_absolute_fractional_residual']):.3g}%\n"
            "Area = "
            f"{float(current['projected_area_arcsec2']):.4g} arcsec²\n"
            "Hot dust mass = "
            f"{float(current['hot_dust_mass_Msun']):.4g} M☉\n"
            "Combined dust mass = "
            f"{float(current['combined_dust_mass_Msun']):.4g} M☉\n"
            "Maximum hidden/observed below "
            f"{args.shortwave_max_micron:g} µm = "
            f"{float(current['maximum_shortwave_fraction']):.4g}"
        )

    update_labels(result)

    if not args.no_sliders:
        temperature_axis = figure.add_axes([0.16, 0.185, 0.67, 0.027])
        if args.manual_foreground:
            foreground_axis = figure.add_axes([0.16, 0.125, 0.67, 0.027])
            hot_axis = figure.add_axes([0.16, 0.065, 0.67, 0.027])
        else:
            hot_axis = figure.add_axes([0.16, 0.105, 0.67, 0.027])
        reset_axis = figure.add_axes([0.86, 0.058, 0.09, 0.045])

        temperature_slider = Slider(
            temperature_axis,
            r"$T_{hot}$ [K]",
            args.temperature_slider_min_K,
            args.temperature_slider_max_K,
            valinit=args.temperature_K,
            valstep=1.0,
        )
        foreground_slider = None
        if args.manual_foreground:
            foreground_slider = Slider(
                foreground_axis,
                rf"$\log_{{10}}\tau_f({args.foreground_reference_micron:g}\,\mu m)$",
                args.log10_tau_slider_min,
                args.log10_tau_slider_max,
                valinit=np.log10(args.tau_foreground),
            )
        hot_slider = Slider(
            hot_axis,
            rf"$\log_{{10}}\tau_h({args.hot_reference_micron:g}\,\mu m)$",
            args.log10_tau_slider_min,
            args.log10_tau_slider_max,
            valinit=np.log10(args.tau_hot),
        )
        reset_button = Button(reset_axis, "Reset")

        def slider_update(_value: float) -> None:
            if args.manual_foreground:
                assert foreground_slider is not None
                current = evaluate_emission(
                    args,
                    wavelength,
                    np.asarray(result["observed_flux"]),
                    np.asarray(result["baseline_flux"]),
                    np.asarray(result["kappa_foreground"]),
                    np.asarray(result["kappa_hot"]),
                    temperature_slider.val,
                    10.0**foreground_slider.val,
                    10.0**hot_slider.val,
                )
            else:
                current = fit_foreground_tau(
                    args,
                    wavelength,
                    np.asarray(result["observed_flux"]),
                    np.asarray(result["baseline_flux"]),
                    np.asarray(result["kappa_foreground"]),
                    np.asarray(result["kappa_hot"]),
                    temperature_slider.val,
                    10.0**hot_slider.val,
                )
            intrinsic_line.set_ydata(current["intrinsic_hot_flux"])
            transmitted_line.set_ydata(current["hidden_flux"])
            total_line.set_ydata(current["total_flux"])
            fit_total_line.set_ydata(np.asarray(current["total_flux"])[fit_band])
            update_labels(current)
            axis.relim(visible_only=True)
            axis.autoscale_view(scalex=False, scaley=True)
            fit_axis.relim(visible_only=True)
            fit_axis.autoscale_view(scalex=False, scaley=True)
            figure.canvas.draw_idle()

        temperature_slider.on_changed(slider_update)
        if foreground_slider is not None:
            foreground_slider.on_changed(slider_update)
        hot_slider.on_changed(slider_update)

        def reset(_event: object) -> None:
            temperature_slider.reset()
            if foreground_slider is not None:
                foreground_slider.reset()
            hot_slider.reset()

        reset_button.on_clicked(reset)

    if args.save_figure:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "fixed_optical_depth_spectrum.png"
        figure.savefig(path, dpi=220, facecolor="white")
        print(f"Saved figure: {path}")

    plt.show()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature-K", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument(
        "--tau-foreground",
        type=float,
        default=DEFAULT_TAU_FOREGROUND,
        help="used only with --manual-foreground",
    )
    parser.add_argument(
        "--foreground-reference-micron",
        type=float,
        default=DEFAULT_FOREGROUND_REFERENCE_MICRON,
    )
    parser.add_argument("--tau-hot", type=float, default=DEFAULT_TAU_HOT)
    parser.add_argument(
        "--hot-reference-micron",
        type=float,
        default=DEFAULT_HOT_REFERENCE_MICRON,
    )
    parser.add_argument(
        "--closure-micron",
        type=float,
        default=DEFAULT_CLOSURE_MICRON,
    )
    parser.add_argument(
        "--total-to-dust",
        type=float,
        default=DEFAULT_TOTAL_TO_DUST,
        help="total material mass divided by dust mass (default: 100)",
    )
    parser.add_argument("--shortwave-max-micron", type=float, default=50.0)
    parser.add_argument(
        "--foreground-fit-min-micron",
        type=float,
        default=DEFAULT_FOREGROUND_FIT_MIN_MICRON,
    )
    parser.add_argument(
        "--foreground-fit-max-micron",
        type=float,
        default=DEFAULT_FOREGROUND_FIT_MAX_MICRON,
    )
    parser.add_argument(
        "--manual-foreground",
        action="store_true",
        help="do not fit foreground tau in 5--7 micron; use --tau-foreground",
    )
    parser.add_argument("--temperature-slider-min-K", type=float, default=30.0)
    parser.add_argument("--temperature-slider-max-K", type=float, default=2000.0)
    parser.add_argument("--log10-tau-slider-min", type=float, default=-2.0)
    parser.add_argument("--log10-tau-slider-max", type=float, default=2.0)
    parser.add_argument(
        "--no-sliders",
        action="store_true",
        help="show the same curves without interactive sliders",
    )
    parser.add_argument("--save-figure", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.temperature_K <= 0.0:
        parser.error("temperature must be positive")
    if args.tau_foreground <= 0.0 or args.tau_hot <= 0.0:
        parser.error("optical depths must be positive")
    if args.total_to_dust <= 0.0:
        parser.error("total-to-dust conversion must be positive")
    if not args.foreground_fit_min_micron < args.foreground_fit_max_micron:
        parser.error("foreground fit-band minimum must be below its maximum")
    if not args.temperature_slider_min_K < args.temperature_slider_max_K:
        parser.error("temperature slider minimum must be below its maximum")
    if not args.log10_tau_slider_min < args.log10_tau_slider_max:
        parser.error("log-tau slider minimum must be below its maximum")
    if not args.temperature_slider_min_K <= args.temperature_K <= args.temperature_slider_max_K:
        parser.error("initial temperature must lie inside the slider range")
    if not args.log10_tau_slider_min <= np.log10(args.tau_foreground) <= args.log10_tau_slider_max:
        parser.error("initial foreground tau must lie inside the slider range")
    if not args.log10_tau_slider_min <= np.log10(args.tau_hot) <= args.log10_tau_slider_max:
        parser.error("initial hot-component tau must lie inside the slider range")
    return args


def main() -> None:
    args = parse_arguments()
    result = calculate(args)
    print_results(args, result)
    plot_results(args, result)


if __name__ == "__main__":
    main()
