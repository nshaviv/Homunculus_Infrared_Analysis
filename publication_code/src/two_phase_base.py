"""
Joint MCMC for Eta Carinae Areas 1 and 2 with independent area parameters.

This file deliberately returns to the two independently successful models:

* Area 1: sideways mixed slab with fitted rho_dense, rho_rare, sigma_total,
  temperatures, and its own two-size Mie opacity distribution.
* Area 2: side-by-side standard model with fitted dense/rare columns,
  temperatures, composition trace fraction, and its own two-size Mie opacity
  distribution.

The likelihoods are combined only as

    ln L_joint = ln L_area1 + ln L_area2.

There are NO shared temperatures, grain sizes, columns, or compositions.  The
only cross-area physical condition is imposed in the prior, phase by phase:

    T_dense,2 < T_dense,1
    T_rare,2  < T_rare,1.

Inside each area the original shielded-dense ordering is also retained:

    T_dense,i < T_rare,i.

The phase-by-phase condition is used instead of an averaged temperature.  An
average-only inequality could be satisfied by making one Area-2 phase hotter
than its Area-1 counterpart and compensating with the other phase.

IMPORTANT IDENTIFIABILITY NOTE
------------------------------
The model has 16 fitted parameters and only 11 positive flux measurements.
The temperature-ordering prior restricts the allowed region but does not
reduce the parameter dimension.  The script therefore reports no reduced
chi-square and saves the full posterior for inspection.  No autocorrelation-
time or 50-tau rule is used.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import scipy.integrate

# Compatibility required by some PyMieScatt releases.
scipy.integrate.trapz = scipy.integrate.trapezoid

import emcee
import numpy as np
from scipy.interpolate import interp1d

from project_paths import MODEL_INPUT_DIR, TWO_PHASE_RESULTS


# ---------------------------------------------------------------------------
# Constants and fixed assumptions retained from the independent models
# ---------------------------------------------------------------------------
H_CGS = 6.626e-27
C_CGS = 3.0e10
K_B_CGS = 1.38e-16
PIXEL_SOLID_ANGLE_SR = (np.pi / (180.0 * 3600.0)) ** 2
BACKGROUND_NOISE_CGS = 5.0e-23
ERROR_FRACTION = 0.05

AREA1_FILLING_FRACTION = 0.500
AREA2_FILLING_FRACTION = 0.444

# Approved Figure-2 LOS-to-radial conversions and radial-column response.
# The fitted columns remain observed LOS columns in the flux likelihood.
AREA1_LOS_TO_RADIAL = 0.2962043350077848
AREA2_LOS_TO_RADIAL = 0.4141285695503457
AREA1_K_RADIAL_G_CM2_PER_MSUN = 0.005758514010801346
AREA2_K_RADIAL_G_CM2_PER_MSUN = 0.0034847470034176735
TOTAL_TO_DUST_FACTOR = 100.0

SILICATE_FRACTION = 0.990
AREA1_CORUNDUM_FRACTION = 0.005

WAVELENGTH_MIN_MICRON = 2.0
WAVELENGTH_MAX_MICRON = 100.0
A_GRID_MIN_MICRON = 0.1
A_GRID_MAX_MICRON = 10.0

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

PARAMETER_NAMES = (
    "area1_log10_rho_dense_g_cm3",
    "area1_log10_rho_rare_g_cm3",
    "area1_T_dense_K",
    "area1_T_rare_K",
    "area1_w_small",
    "area1_a_small_micron",
    "area1_a_large_micron",
    "area1_log10_sigma_total_g_cm2",
    "area2_T_dense_K",
    "area2_T_rare_K",
    "area2_log10_sigma_dense_g_cm2",
    "area2_log10_sigma_rare_g_cm2",
    "area2_corundum_fraction",
    "area2_w_small",
    "area2_a_small_micron",
    "area2_a_large_micron",
)

INITIAL_GUESS = np.asarray(
    [
        -16.0,  # Area 1 log10 rho_dense [g/cm^3]
        -18.0,  # Area 1 log10 rho_rare [g/cm^3]
        160.0,  # Area 1 T_dense [K]
        420.0,  # Area 1 T_rare [K]
        0.60,   # Area 1 small-grain weight
        0.20,   # Area 1 small radius [micron]
        3.80,   # Area 1 large radius [micron]
        -4.00,  # Area 1 log10 sigma_total [g/cm^2]
        120.0,  # Area 2 T_dense [K], initially below Area 1
        280.0,  # Area 2 T_rare [K], initially below Area 1
        -3.50,  # Area 2 log10 sigma_dense [g/cm^2]
        -5.50,  # Area 2 log10 sigma_rare [g/cm^2]
        0.005,  # Area 2 corundum fraction
        0.50,   # Area 2 small-grain weight
        0.18,   # Area 2 small radius [micron]
        4.00,   # Area 2 large radius [micron]
    ],
    dtype=float,
)

WALKER_SCATTER = np.asarray(
    [
        0.30,
        0.30,
        6.0,
        8.0,
        0.05,
        0.03,
        0.40,
        0.15,
        6.0,
        8.0,
        0.15,
        0.15,
        0.001,
        0.05,
        0.03,
        0.40,
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------
def resolve_input(filename: str) -> Path:
    """Find an input file without requiring a particular working directory."""
    candidates = (
        MODEL_INPUT_DIR / filename,
        SCRIPT_DIRECTORY / filename,
        Path.cwd() / filename,
    )
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find required file {filename!r}. Searched:\n{searched}"
    )


# ---------------------------------------------------------------------------
# Mie opacity lookup, preserving the original two-discrete-size model
# ---------------------------------------------------------------------------
def load_optical_constants():
    silicate_path = resolve_input("eps_Sil")
    corundum_path = resolve_input("corundumAl203data")
    iron_path = resolve_input("iron_ordaldata")

    lam_sil_list, n_sil_list, k_sil_list = [], [], []
    with silicate_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                wavelength = float(parts[0])
                n_value = float(parts[3]) + 1.0
                k_value = float(parts[4])
            except ValueError:
                continue
            lam_sil_list.append(wavelength)
            n_sil_list.append(n_value)
            k_sil_list.append(k_value)

    lam_sil = np.asarray(lam_sil_list, dtype=float)
    n_sil = np.asarray(n_sil_list, dtype=float)
    k_sil = np.asarray(k_sil_list, dtype=float)
    order = np.argsort(lam_sil)
    lam_sil, n_sil, k_sil = (
        lam_sil[order],
        n_sil[order],
        k_sil[order],
    )

    lam_cor, n_cor, k_cor = np.loadtxt(corundum_path, unpack=True)
    lam_fe, n_fe, k_fe = np.loadtxt(iron_path, unpack=True)
    cor_order = np.argsort(lam_cor)
    fe_order = np.argsort(lam_fe)
    lam_cor, n_cor, k_cor = (
        lam_cor[cor_order],
        n_cor[cor_order],
        k_cor[cor_order],
    )
    lam_fe, n_fe, k_fe = (
        lam_fe[fe_order],
        n_fe[fe_order],
        k_fe[fe_order],
    )

    wavelength_grid = np.logspace(
        np.log10(WAVELENGTH_MIN_MICRON),
        np.log10(WAVELENGTH_MAX_MICRON),
        1000,
    )
    return (
        wavelength_grid,
        (
            np.interp(wavelength_grid, lam_sil, n_sil),
            np.interp(wavelength_grid, lam_sil, k_sil),
        ),
        (
            np.interp(wavelength_grid, lam_cor, n_cor),
            np.interp(wavelength_grid, lam_cor, k_cor),
        ),
        (
            np.interp(wavelength_grid, lam_fe, n_fe),
            np.interp(wavelength_grid, lam_fe, k_fe),
        ),
    )


def monodisperse_kappa(
    wavelength_grid, n_values, k_values, bulk_density, radius_micron
):
    # Delay PyMieScatt (which imports plotting support) until a flux run
    # actually builds opacities.  The --mass-only path does not need it.
    import PyMieScatt as ps

    result = np.zeros_like(wavelength_grid)
    radius_cm = radius_micron * 1.0e-4
    diameter_micron = 2.0 * radius_micron
    for index, wavelength in enumerate(wavelength_grid):
        refractive_index = n_values[index] + 1j * k_values[index]
        q_abs = ps.MieQ(
            refractive_index, wavelength, diameter_micron
        )[2]
        result[index] = 3.0 * q_abs / (4.0 * radius_cm * bulk_density)
    return result


def build_opacity_interpolators():
    wavelength_grid, silicate_nk, corundum_nk, iron_nk = (
        load_optical_constants()
    )
    radius_grid = np.logspace(
        np.log10(A_GRID_MIN_MICRON), np.log10(A_GRID_MAX_MICRON), 50
    )
    interpolators = []
    print("Pre-calculating the original two-size Mie opacity grids...")
    for (n_values, k_values), bulk_density in (
        (silicate_nk, 3.2),
        (corundum_nk, 3.9),
        (iron_nk, 7.9),
    ):
        table = np.empty((len(radius_grid), len(wavelength_grid)))
        for index, radius in enumerate(radius_grid):
            table[index] = monodisperse_kappa(
                wavelength_grid,
                n_values,
                k_values,
                bulk_density,
                radius,
            )
        interpolators.append(
            interp1d(
                radius_grid,
                table,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )
        )
    print("Mie opacity grids ready.\n")
    return wavelength_grid, interpolators


WAVELENGTH_GRID = None
OPACITY_INTERPOLATORS = None


def ensure_opacity_interpolators():
    """Build the Mie tables only when a flux model actually needs them."""
    global WAVELENGTH_GRID
    global OPACITY_INTERPOLATORS
    if OPACITY_INTERPOLATORS is None:
        WAVELENGTH_GRID, OPACITY_INTERPOLATORS = (
            build_opacity_interpolators()
        )


def mixed_opacity(w_small, a_small, a_large, corundum_fraction):
    """Return one area's independently fitted two-size opacity curve."""
    ensure_opacity_interpolators()
    iron_fraction = 1.0 - SILICATE_FRACTION - corundum_fraction
    size_mixtures = [
        w_small * interpolator(a_small)
        + (1.0 - w_small) * interpolator(a_large)
        for interpolator in OPACITY_INTERPOLATORS
    ]
    return (
        SILICATE_FRACTION * size_mixtures[0]
        + corundum_fraction * size_mixtures[1]
        + iron_fraction * size_mixtures[2]
    )


# ---------------------------------------------------------------------------
# Data and radiation functions
# ---------------------------------------------------------------------------
def load_flux_data(filename: str):
    table = np.loadtxt(resolve_input(filename), delimiter=",")
    wavelength = np.asarray(table[:, 0], dtype=float)
    flux = np.asarray(table[:, 1], dtype=float) * 1.0e-23
    keep = np.isfinite(wavelength) & np.isfinite(flux) & (flux > 0.0)
    wavelength = wavelength[keep]
    flux = flux[keep]
    uncertainty = np.sqrt(
        (ERROR_FRACTION * flux) ** 2 + BACKGROUND_NOISE_CGS**2
    )
    return wavelength, flux, uncertainty


def planck_nu_from_wavelength(wavelength_micron, temperature_K):
    wavelength_cm = np.asarray(wavelength_micron, dtype=float) * 1.0e-4
    exponent = H_CGS * C_CGS / (
        wavelength_cm * K_B_CGS * float(temperature_K)
    )
    safe_exponent = np.minimum(exponent, 700.0)
    b_lambda = (
        2.0
        * H_CGS
        * C_CGS**2
        / wavelength_cm**5
        / np.expm1(safe_exponent)
    )
    b_lambda = np.where(exponent < 700.0, b_lambda, 0.0)
    return b_lambda * wavelength_cm**2 / C_CGS


def derive_state(theta):
    (
        log_rho_d_1,
        log_rho_r_1,
        T_d_1,
        T_r_1,
        w_small_1,
        a_small_1,
        a_large_1,
        log_sigma_total_1,
        T_d_2,
        T_r_2,
        log_sigma_d_2,
        log_sigma_r_2,
        d2_2,
        w_small_2,
        a_small_2,
        a_large_2,
    ) = theta

    rho_d_1 = 10.0**log_rho_d_1
    rho_r_1 = 10.0**log_rho_r_1
    sigma_total_1 = 10.0**log_sigma_total_1
    effective_density_1 = (
        AREA1_FILLING_FRACTION * rho_d_1
        + (1.0 - AREA1_FILLING_FRACTION) * rho_r_1
    )
    x_total_1 = sigma_total_1 / effective_density_1
    sigma_d_1 = (
        AREA1_FILLING_FRACTION * rho_d_1 * x_total_1
    )
    sigma_r_1 = (
        (1.0 - AREA1_FILLING_FRACTION) * rho_r_1 * x_total_1
    )

    sigma_d_2 = 10.0**log_sigma_d_2
    sigma_r_2 = 10.0**log_sigma_r_2
    sigma_effective_2 = (
        AREA2_FILLING_FRACTION * sigma_d_2
        + (1.0 - AREA2_FILLING_FRACTION) * sigma_r_2
    )

    return {
        "rho_d_1": rho_d_1,
        "rho_r_1": rho_r_1,
        "T_d_1": T_d_1,
        "T_r_1": T_r_1,
        "w_small_1": w_small_1,
        "a_small_1": a_small_1,
        "a_large_1": a_large_1,
        "sigma_total_1": sigma_total_1,
        "x_total_1": x_total_1,
        "sigma_d_1": sigma_d_1,
        "sigma_r_1": sigma_r_1,
        "T_d_2": T_d_2,
        "T_r_2": T_r_2,
        "sigma_d_2": sigma_d_2,
        "sigma_r_2": sigma_r_2,
        "sigma_effective_2": sigma_effective_2,
        "d2_2": d2_2,
        "d3_2": 1.0 - SILICATE_FRACTION - d2_2,
        "w_small_2": w_small_2,
        "a_small_2": a_small_2,
        "a_large_2": a_large_2,
    }


# ---------------------------------------------------------------------------
# The two unchanged transfer equations
# ---------------------------------------------------------------------------
def evaluate_models(theta, wavelength_area1, wavelength_area2):
    state = derive_state(theta)

    opacity_1_grid = mixed_opacity(
        state["w_small_1"],
        state["a_small_1"],
        state["a_large_1"],
        AREA1_CORUNDUM_FRACTION,
    )
    kappa_1 = np.interp(
        wavelength_area1, WAVELENGTH_GRID, opacity_1_grid
    )

    # Area 1: sideways mixed slab, unchanged from the independent fit.
    chi_effective_1 = kappa_1 * (
        AREA1_FILLING_FRACTION * state["rho_d_1"]
        + (1.0 - AREA1_FILLING_FRACTION) * state["rho_r_1"]
    )
    source_1 = kappa_1 * (
        AREA1_FILLING_FRACTION
        * state["rho_d_1"]
        * planck_nu_from_wavelength(wavelength_area1, state["T_d_1"])
        + (1.0 - AREA1_FILLING_FRACTION)
        * state["rho_r_1"]
        * planck_nu_from_wavelength(wavelength_area1, state["T_r_1"])
    )
    intensity_1 = (
        -np.expm1(-chi_effective_1 * state["x_total_1"])
        * source_1
        / chi_effective_1
    )
    flux_1 = intensity_1 * PIXEL_SOLID_ANGLE_SR

    opacity_2_grid = mixed_opacity(
        state["w_small_2"],
        state["a_small_2"],
        state["a_large_2"],
        state["d2_2"],
    )
    kappa_2 = np.interp(
        wavelength_area2, WAVELENGTH_GRID, opacity_2_grid
    )

    # Area 2: standard side-by-side projected-area model, unchanged.
    tau_d_2 = kappa_2 * state["sigma_d_2"]
    tau_r_2 = kappa_2 * state["sigma_r_2"]
    flux_2 = (
        AREA2_FILLING_FRACTION
        * planck_nu_from_wavelength(wavelength_area2, state["T_d_2"])
        * -np.expm1(-tau_d_2)
        + (1.0 - AREA2_FILLING_FRACTION)
        * planck_nu_from_wavelength(wavelength_area2, state["T_r_2"])
        * -np.expm1(-tau_r_2)
    ) * PIXEL_SOLID_ANGLE_SR

    return flux_1, flux_2, state


# ---------------------------------------------------------------------------
# Joint prior and likelihood
# ---------------------------------------------------------------------------
def log_prior(theta):
    state = derive_state(theta)

    log_rho_d_1 = theta[0]
    log_rho_r_1 = theta[1]
    log_sigma_total_1 = theta[7]
    log_sigma_d_2 = theta[10]
    log_sigma_r_2 = theta[11]

    if not (-20.0 <= log_rho_d_1 <= -14.0):
        return -np.inf
    if not (-22.0 <= log_rho_r_1 <= -16.0):
        return -np.inf
    if not (log_rho_d_1 > log_rho_r_1):
        return -np.inf
    if not (-6.0 <= log_sigma_total_1 <= -1.0):
        return -np.inf

    for temperature in (
        state["T_d_1"],
        state["T_r_1"],
        state["T_d_2"],
        state["T_r_2"],
    ):
        if not (50.0 <= temperature <= 800.0):
            return -np.inf

    # Original within-area ordering: dense material is cooler.
    if not (state["T_d_1"] < state["T_r_1"]):
        return -np.inf
    if not (state["T_d_2"] < state["T_r_2"]):
        return -np.inf

    # THE ONLY CROSS-AREA BIND requested by the user.
    # Every corresponding Area-2 phase must be cooler than Area 1.
    if not (state["T_d_2"] < state["T_d_1"]):
        return -np.inf
    if not (state["T_r_2"] < state["T_r_1"]):
        return -np.inf

    for area in (1, 2):
        if not (0.2 <= state[f"w_small_{area}"] <= 1.0):
            return -np.inf
        if not (0.1 <= state[f"a_small_{area}"] <= 0.4):
            return -np.inf
        if not (0.1 <= state[f"a_large_{area}"] <= 7.0):
            return -np.inf
        if not (
            state[f"a_small_{area}"] < state[f"a_large_{area}"]
        ):
            return -np.inf

    if not (-3.0 <= log_sigma_d_2 <= -1.0):
        return -np.inf
    if not (-6.0 <= log_sigma_r_2 <= -3.5):
        return -np.inf
    if not (0.0 <= state["d2_2"] <= 0.010):
        return -np.inf
    if not (0.0 <= state["d3_2"] <= 0.010):
        return -np.inf

    return 0.0


def one_area_log_likelihood(model_flux, observed_flux, linear_error):
    if np.any(model_flux <= 0.0) or not np.all(np.isfinite(model_flux)):
        return -np.inf
    log10_error = linear_error / (observed_flux * np.log(10.0))
    residual = (
        np.log10(model_flux) - np.log10(observed_flux)
    ) / log10_error
    return -0.5 * float(np.sum(residual**2))


def log_likelihood_components(theta, area1_data, area2_data):
    model_1, model_2, _ = evaluate_models(
        theta, area1_data[0], area2_data[0]
    )
    return (
        one_area_log_likelihood(model_1, area1_data[1], area1_data[2]),
        one_area_log_likelihood(model_2, area2_data[1], area2_data[2]),
    )


def log_probability(theta, area1_data, area2_data):
    prior = log_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    ln_l_1, ln_l_2 = log_likelihood_components(theta, area1_data, area2_data)
    if not np.isfinite(ln_l_1 + ln_l_2):
        return -np.inf
    return prior + ln_l_1 + ln_l_2


def initialize_walkers(nwalkers, generator):
    positions = []
    attempts = 0
    while len(positions) < nwalkers:
        trial = INITIAL_GUESS + WALKER_SCATTER * generator.standard_normal(
            len(PARAMETER_NAMES)
        )
        attempts += 1
        if np.isfinite(log_prior(trial)):
            positions.append(trial)
        if attempts > 200000:
            raise RuntimeError("Could not initialize valid joint walkers.")
    return np.asarray(positions)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def calculate_mass_arrays(samples):
    """Derive both whole-lobe mass normalizations for every sample.

    The MCMC columns are LOS columns because those determine optical depth.
    For the physical mass, the approved deterministic factors convert each
    LOS column to its radial counterpart exactly once.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(
            f"Expected N x {len(PARAMETER_NAMES)} posterior samples."
        )

    sigma_los_1 = 10.0 ** samples[:, 7]
    sigma_los_2 = (
        AREA2_FILLING_FRACTION * 10.0 ** samples[:, 10]
        + (1.0 - AREA2_FILLING_FRACTION) * 10.0 ** samples[:, 11]
    )
    sigma_radial_1 = AREA1_LOS_TO_RADIAL * sigma_los_1
    sigma_radial_2 = AREA2_LOS_TO_RADIAL * sigma_los_2

    return {
        "sigma_los_1": sigma_los_1,
        "sigma_radial_1": sigma_radial_1,
        "dust_uncorrected_1": sigma_los_1
        / AREA1_K_RADIAL_G_CM2_PER_MSUN,
        "dust_corrected_1": sigma_radial_1
        / AREA1_K_RADIAL_G_CM2_PER_MSUN,
        "sigma_los_2": sigma_los_2,
        "sigma_radial_2": sigma_radial_2,
        "dust_uncorrected_2": sigma_los_2
        / AREA2_K_RADIAL_G_CM2_PER_MSUN,
        "dust_corrected_2": sigma_radial_2
        / AREA2_K_RADIAL_G_CM2_PER_MSUN,
    }


def print_mass_summary(samples, maximum_posterior_sample):
    """Print posterior mass intervals and the coherent best-fit sample."""
    arrays = calculate_mass_arrays(samples)
    best_arrays = calculate_mass_arrays(
        np.asarray(maximum_posterior_sample, dtype=float)[None, :]
    )

    print("\n--- Whole-nebula mass normalizations ---")
    print(
        "Each area independently normalizes the complete Smith two-lobe "
        "distribution; these are not masses confined to the apertures."
    )
    print("Total masses below use the requested total/dust factor of 100.")

    for area_number in (1, 2):
        sigma_los = arrays[f"sigma_los_{area_number}"]
        sigma_radial = arrays[f"sigma_radial_{area_number}"]
        dust_uncorrected = arrays[f"dust_uncorrected_{area_number}"]
        dust_corrected = arrays[f"dust_corrected_{area_number}"]
        total_uncorrected = TOTAL_TO_DUST_FACTOR * dust_uncorrected
        total_corrected = TOTAL_TO_DUST_FACTOR * dust_corrected

        q_sigma_los = np.percentile(sigma_los, [16.0, 50.0, 84.0])
        q_sigma_radial = np.percentile(
            sigma_radial, [16.0, 50.0, 84.0]
        )
        q_uncorrected = np.percentile(
            total_uncorrected, [16.0, 50.0, 84.0]
        )
        q_corrected = np.percentile(
            total_corrected, [16.0, 50.0, 84.0]
        )
        best_los = best_arrays[f"sigma_los_{area_number}"][0]
        best_radial = best_arrays[f"sigma_radial_{area_number}"][0]
        best_uncorrected = (
            TOTAL_TO_DUST_FACTOR
            * best_arrays[f"dust_uncorrected_{area_number}"][0]
        )
        best_corrected = (
            TOTAL_TO_DUST_FACTOR
            * best_arrays[f"dust_corrected_{area_number}"][0]
        )
        correction = (
            AREA1_LOS_TO_RADIAL
            if area_number == 1
            else AREA2_LOS_TO_RADIAL
        )

        print(f"\nArea {area_number}:")
        print(f"  LOS-to-radial factor C = {correction:.15f}")
        print(
            "  Sigma_LOS posterior q16/median/q84    = "
            f"{q_sigma_los[0]:.6e} / {q_sigma_los[1]:.6e} / "
            f"{q_sigma_los[2]:.6e} g/cm^2"
        )
        print(
            "  Sigma_radial posterior q16/median/q84 = "
            f"{q_sigma_radial[0]:.6e} / {q_sigma_radial[1]:.6e} / "
            f"{q_sigma_radial[2]:.6e} g/cm^2"
        )
        print(
            "  WITHOUT LOS correction total mass      = "
            f"{q_uncorrected[1]:.6f} "
            f"(-{q_uncorrected[1]-q_uncorrected[0]:.6f}, "
            f"+{q_uncorrected[2]-q_uncorrected[1]:.6f}) Msun"
        )
        print(
            "  WITH LOS correction total mass         = "
            f"{q_corrected[1]:.6f} "
            f"(-{q_corrected[1]-q_corrected[0]:.6f}, "
            f"+{q_corrected[2]-q_corrected[1]:.6f}) Msun"
        )
        print(
            "  Maximum-posterior LOS/radial columns    = "
            f"{best_los:.6e} / {best_radial:.6e} g/cm^2"
        )
        print(
            "  Maximum-posterior total mass, before/after = "
            f"{best_uncorrected:.6f} / {best_corrected:.6f} Msun"
        )


def run_mass_only(args):
    """Print masses from an already completed chain without rerunning it."""
    output_directory = Path(args.output_dir).expanduser().resolve()
    posterior_path = output_directory / "posterior_samples.npz"
    if not posterior_path.is_file():
        raise FileNotFoundError(
            "Cannot use --mass-only because the posterior is missing: "
            f"{posterior_path}"
        )
    with np.load(posterior_path, allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        log_probabilities = np.asarray(
            archive["log_probability"], dtype=float
        )
    maximum = samples[int(np.argmax(log_probabilities))]
    print(f"Loaded {len(samples)} samples from {posterior_path}")
    print_mass_summary(samples, maximum)
    print("\nNo MCMC steps were rerun.\n")


def run_joint_mcmc(args):
    # Import the interactive plotting backend only for a full fit.  The
    # --mass-only recovery path therefore opens no GUI and starts quickly.
    import matplotlib.pyplot as plt

    area1_data = load_flux_data("lobe.edge.data")
    area2_data = load_flux_data("lobeedgedata2")
    n_data = len(area1_data[0]) + len(area2_data[0])
    n_parameters = len(PARAMETER_NAMES)

    generator = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    positions = initialize_walkers(args.walkers, generator)

    print("\n============================================================")
    print("JOINT MCMC: INDEPENDENT AREA MODELS + TEMPERATURE ORDER")
    print("============================================================")
    print("ln L_joint = ln L_area1 + ln L_area2")
    print("Only cross-area condition: Td2 < Td1 AND Tr2 < Tr1")
    print("No parameters are shared between the two areas.")
    print(f"Data points / parameters = {n_data} / {n_parameters}")
    print("CAUTION: parameter count exceeds data count; reduced chi2 is undefined.")
    print(f"Running {args.walkers} walkers for {args.steps} steps...")

    sampler = emcee.EnsembleSampler(
        args.walkers,
        n_parameters,
        log_probability,
        args=(area1_data, area2_data),
    )
    sampler.run_mcmc(positions, args.steps, progress=True)
    samples = sampler.get_chain(discard=args.burn, thin=args.thin, flat=True)
    log_probabilities = sampler.get_log_prob(
        discard=args.burn, thin=args.thin, flat=True
    )
    if len(samples) == 0:
        raise RuntimeError("No post-burn samples were retained.")

    q16, median, q84 = np.percentile(samples, [16.0, 50.0, 84.0], axis=0)
    maximum = samples[int(np.argmax(log_probabilities))]
    best_flux_1, best_flux_2, best_state = evaluate_models(
        maximum, area1_data[0], area2_data[0]
    )
    ln_l_1, ln_l_2 = log_likelihood_components(
        maximum, area1_data, area2_data
    )

    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    parameter_path = output_directory / "parameter_summary.csv"
    derived_path = output_directory / "maximum_posterior_derived.csv"
    posterior_path = output_directory / "posterior_samples.npz"

    write_csv(
        parameter_path,
        [
            "parameter",
            "posterior_q16",
            "posterior_median",
            "posterior_q84",
            "maximum_posterior_sample",
        ],
        zip(PARAMETER_NAMES, q16, median, q84, maximum),
    )
    write_csv(
        derived_path,
        ["derived_quantity", "maximum_posterior_value"],
        [
            ("area1_T_dense_K", best_state["T_d_1"]),
            ("area1_T_rare_K", best_state["T_r_1"]),
            ("area2_T_dense_K", best_state["T_d_2"]),
            ("area2_T_rare_K", best_state["T_r_2"]),
            ("area1_minus_area2_T_dense_K", best_state["T_d_1"] - best_state["T_d_2"]),
            ("area1_minus_area2_T_rare_K", best_state["T_r_1"] - best_state["T_r_2"]),
            ("area1_rho_dense_g_cm3", best_state["rho_d_1"]),
            ("area1_rho_rare_g_cm3", best_state["rho_r_1"]),
            ("area1_sigma_total_g_cm2", best_state["sigma_total_1"]),
            ("area1_sigma_dense_component_g_cm2", best_state["sigma_d_1"]),
            ("area1_sigma_rare_component_g_cm2", best_state["sigma_r_1"]),
            ("area1_x_total_cm", best_state["x_total_1"]),
            ("area2_sigma_dense_g_cm2", best_state["sigma_d_2"]),
            ("area2_sigma_rare_g_cm2", best_state["sigma_r_2"]),
            ("area2_sigma_effective_g_cm2", best_state["sigma_effective_2"]),
        ],
    )
    np.savez_compressed(
        posterior_path,
        samples=samples,
        log_probability=log_probabilities,
        parameter_names=np.asarray(PARAMETER_NAMES),
    )

    wavelength_fit = np.logspace(
        np.log10(WAVELENGTH_MIN_MICRON),
        np.log10(WAVELENGTH_MAX_MICRON),
        1000,
    )
    fit_1, fit_2, _ = evaluate_models(maximum, wavelength_fit, wavelength_fit)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, title, data, fit in (
        (axes[0], "Area 1: sideways slab", area1_data, fit_1),
        (axes[1], "Area 2: standard", area2_data, fit_2),
    ):
        axis.errorbar(
            data[0], data[1], yerr=data[2], fmt="o", color="black",
            label="Data", zorder=5,
        )
        axis.plot(
            wavelength_fit,
            fit,
            color="#9f2f1f",
            lw=2.0,
            label="Maximum-posterior fit",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(WAVELENGTH_MIN_MICRON, WAVELENGTH_MAX_MICRON)
        axis.set_ylim(1.0e-26, 1.0e-19)
        axis.set_xlabel(r"Wavelength [$\mu$m]")
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Flux [CGS]")
    figure.suptitle("Joint MCMC with Td2 < Td1 and Tr2 < Tr1")
    figure.tight_layout()

    print("\n--- Marginal posterior summaries ---")
    for name, low, middle, high in zip(PARAMETER_NAMES, q16, median, q84):
        print(f"{name:42s}: {middle:.9g} (-{middle-low:.3g}, +{high-middle:.3g})")

    print("\n--- Maximum-posterior temperature audit ---")
    print(f"Area 1 Td / Tr = {best_state['T_d_1']:.6f} / {best_state['T_r_1']:.6f} K")
    print(f"Area 2 Td / Tr = {best_state['T_d_2']:.6f} / {best_state['T_r_2']:.6f} K")
    print(f"Td1 - Td2      = {best_state['T_d_1'] - best_state['T_d_2']:.6f} K")
    print(f"Tr1 - Tr2      = {best_state['T_r_1'] - best_state['T_r_2']:.6f} K")
    print("Both differences must be positive by construction.")

    print("\n--- Maximum-posterior fit audit ---")
    print(f"chi2 Area 1 / Area 2 / total = {-2.0*ln_l_1:.6f} / {-2.0*ln_l_2:.6f} / {-2.0*(ln_l_1+ln_l_2):.6f}")
    print(f"mean acceptance fraction      = {np.mean(sampler.acceptance_fraction):.6f}")
    print("Reduced chi2 is not reported because 11 - 16 < 0.")

    print_mass_summary(samples, maximum)

    print("\n--- Saved outputs ---")
    for path in (parameter_path, derived_path, posterior_path):
        print(path)
    print("The fit graph is displayed interactively and is not saved.")
    print("============================================================\n")

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Joint Area-1/Area-2 MCMC with independent parameters and "
            "phase-by-phase Area-2-cooler temperature ordering."
        )
    )
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument(
        "--output-dir",
        default=str(
            TWO_PHASE_RESULTS / "base_model"
        ),
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--mass-only",
        action="store_true",
        help="print masses from the saved posterior without rerunning MCMC",
    )
    args = parser.parse_args()

    if args.walkers < 2 * len(PARAMETER_NAMES):
        parser.error("walkers must be at least 32 for the 16 parameters.")
    if args.steps < 2:
        parser.error("steps must be at least 2.")
    if args.burn < 0 or args.burn >= args.steps:
        parser.error("burn must satisfy 0 <= burn < steps.")
    if args.thin < 1:
        parser.error("thin must be at least 1.")
    return args


if __name__ == "__main__":
    command_line_arguments = parse_arguments()
    if command_line_arguments.mass_only:
        run_mass_only(command_line_arguments)
    else:
        run_joint_mcmc(command_line_arguments)
