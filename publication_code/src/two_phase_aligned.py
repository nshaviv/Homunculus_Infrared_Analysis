"""Joint two-wall Eta Carinae MCMC with density-dependent grain growth.

This is a new model variant.  It combines two previously requested pieces of
physics without modifying their meaning:

1. Area 2 is one transparent sightline with an 80-degree foreground wall and
   a 28.4287-degree rear wall.  The fitted dense and rare columns are TOTAL LOS
   columns.  Smith's radial mass distribution and the two independent LOS
   projections split each fitted total into foreground and rear columns.  The
   same two-phase radiative-transfer model is evaluated at both temperatures,
   with rear emission attenuated by the foreground.

2. Every dust phase has a continuous number distribution

       dn/da = C a^(-p),      0.05 micron <= a <= a_max.

   The slope p and empirical density-size exponent q are shared.  Area 1 has
   one fitted dense a_max.  Its rare a_max and all foreground/rear Area-2
   a_max values are derived from rho ratios through a_max proportional to
   rho^q.  Cross-location density ratios use the physical RADIAL columns, not
   the projected LOS columns.

There are eleven fitted parameters for eleven positive photometric points.
The common multiplicative scale of the two Area-1 volume densities remains an
exact degeneracy (only their ratio is spectrally identifiable); this is stated
in the output.  No autocorrelation-time or 50-tau rule is used.

The shared utilities and geometry are provided by ``two_phase_base.py`` and
``two_phase_geometry.py`` in this directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import emcee
import numpy as np
import scipy.integrate

import two_phase_base as base
import two_phase_geometry as geometry

from project_paths import TWO_PHASE_RESULTS


# ---------------------------------------------------------------------------
# Continuous grain distributions and Mie opacity lookup
# ---------------------------------------------------------------------------
A_MIN_MICRON = 0.05
A_MAX_MICRON = 10.0
P_MIN = 2.0
P_MAX = 5.0
Q_MIN = 0.0
Q_MAX = 2.0
P_GRID = np.linspace(P_MIN, P_MAX, 31)
RADIUS_GRID = np.logspace(np.log10(A_MIN_MICRON), np.log10(A_MAX_MICRON), 80)
LOG_RADIUS_GRID = np.log10(RADIUS_GRID)

POWERLAW_WAVELENGTH_GRID = None
POWERLAW_LOG_WAVELENGTH_GRID = None
POWERLAW_SPECIES_OPACITY = None


def ensure_powerlaw_opacity_table():
    """Build the mass-weighted Mie table once, on first flux evaluation."""
    global POWERLAW_WAVELENGTH_GRID
    global POWERLAW_LOG_WAVELENGTH_GRID
    global POWERLAW_SPECIES_OPACITY
    if POWERLAW_SPECIES_OPACITY is not None:
        return

    wavelength, silicate_nk, corundum_nk, iron_nk = base.load_optical_constants()
    monodisperse = np.empty((3, len(RADIUS_GRID), len(wavelength)))
    print("Building continuous power-law Mie opacity lookup...")
    for species_index, ((n_values, k_values), bulk_density) in enumerate(
        (
            (silicate_nk, 3.2),
            (corundum_nk, 3.9),
            (iron_nk, 7.9),
        )
    ):
        for radius_index, radius in enumerate(RADIUS_GRID):
            monodisperse[species_index, radius_index] = (
                base.monodisperse_kappa(
                    wavelength,
                    n_values,
                    k_values,
                    bulk_density,
                    radius,
                )
            )

    table = np.empty(
        (3, len(P_GRID), len(RADIUS_GRID), len(wavelength)), dtype=float
    )
    for p_index, p_value in enumerate(P_GRID):
        # dn/da is proportional to a^-p and grain mass is proportional to a^3,
        # so opacity per dust mass is weighted by a^(3-p).
        mass_weight = RADIUS_GRID ** (3.0 - p_value)
        denominator = scipy.integrate.cumulative_trapezoid(
            mass_weight, RADIUS_GRID, initial=0.0
        )
        numerator = scipy.integrate.cumulative_trapezoid(
            monodisperse * mass_weight[None, :, None],
            RADIUS_GRID,
            axis=1,
            initial=0.0,
        )
        table[:, p_index, 0, :] = monodisperse[:, 0, :]
        table[:, p_index, 1:, :] = (
            numerator[:, 1:, :] / denominator[None, 1:, None]
        )

    POWERLAW_WAVELENGTH_GRID = wavelength
    POWERLAW_LOG_WAVELENGTH_GRID = np.log10(wavelength)
    POWERLAW_SPECIES_OPACITY = table
    print("Continuous power-law opacity lookup ready.\n")


def powerlaw_species_kappas(p_value, a_max_micron, wavelength_micron):
    """Interpolate all three species in p, log(a_max), and log(lambda)."""
    ensure_powerlaw_opacity_table()
    wavelength = np.atleast_1d(np.asarray(wavelength_micron, dtype=float))
    log_a = np.log10(float(a_max_micron))
    log_wavelength = np.log10(wavelength)

    p_index = int(np.searchsorted(P_GRID, p_value, side="right") - 1)
    p_index = int(np.clip(p_index, 0, len(P_GRID) - 2))
    p_fraction = (
        (p_value - P_GRID[p_index])
        / (P_GRID[p_index + 1] - P_GRID[p_index])
    )

    a_index = int(np.searchsorted(LOG_RADIUS_GRID, log_a, side="right") - 1)
    a_index = int(np.clip(a_index, 0, len(LOG_RADIUS_GRID) - 2))
    a_fraction = (
        (log_a - LOG_RADIUS_GRID[a_index])
        / (LOG_RADIUS_GRID[a_index + 1] - LOG_RADIUS_GRID[a_index])
    )

    wavelength_index = (
        np.searchsorted(
            POWERLAW_LOG_WAVELENGTH_GRID, log_wavelength, side="right"
        )
        - 1
    )
    wavelength_index = np.clip(
        wavelength_index, 0, len(POWERLAW_LOG_WAVELENGTH_GRID) - 2
    )
    wavelength_fraction = (
        (log_wavelength - POWERLAW_LOG_WAVELENGTH_GRID[wavelength_index])
        / (
            POWERLAW_LOG_WAVELENGTH_GRID[wavelength_index + 1]
            - POWERLAW_LOG_WAVELENGTH_GRID[wavelength_index]
        )
    )[None, :]

    table = POWERLAW_SPECIES_OPACITY
    c000 = table[:, p_index, a_index, wavelength_index]
    c001 = table[:, p_index, a_index, wavelength_index + 1]
    c010 = table[:, p_index, a_index + 1, wavelength_index]
    c011 = table[:, p_index, a_index + 1, wavelength_index + 1]
    c100 = table[:, p_index + 1, a_index, wavelength_index]
    c101 = table[:, p_index + 1, a_index, wavelength_index + 1]
    c110 = table[:, p_index + 1, a_index + 1, wavelength_index]
    c111 = table[:, p_index + 1, a_index + 1, wavelength_index + 1]

    lower_p_lower_a = c000 * (1.0 - wavelength_fraction) + c001 * wavelength_fraction
    lower_p_upper_a = c010 * (1.0 - wavelength_fraction) + c011 * wavelength_fraction
    upper_p_lower_a = c100 * (1.0 - wavelength_fraction) + c101 * wavelength_fraction
    upper_p_upper_a = c110 * (1.0 - wavelength_fraction) + c111 * wavelength_fraction
    lower_p = lower_p_lower_a * (1.0 - a_fraction) + lower_p_upper_a * a_fraction
    upper_p = upper_p_lower_a * (1.0 - a_fraction) + upper_p_upper_a * a_fraction
    return lower_p * (1.0 - p_fraction) + upper_p * p_fraction


def powerlaw_mixed_kappa(
    p_value,
    a_max_micron,
    wavelength_micron,
    corundum_fraction,
):
    """Return Mie opacity for one continuous size distribution."""
    iron_fraction = 1.0 - base.SILICATE_FRACTION - corundum_fraction
    species = powerlaw_species_kappas(
        p_value, a_max_micron, wavelength_micron
    )
    fractions = np.asarray(
        [base.SILICATE_FRACTION, corundum_fraction, iron_fraction]
    )
    return np.sum(fractions[:, None] * species, axis=0)


# ---------------------------------------------------------------------------
# Eleven-parameter density-dependent model
# ---------------------------------------------------------------------------
PARAMETER_NAMES = (
    "area1_T_dense_K",
    "area1_T_rare_K",
    "p_size_distribution_shared",
    "q_density_size_shared",
    "area1_a_dense_max_micron",
    "area1_log10_rho_dense_g_cm3",
    "area1_log10_rho_rare_g_cm3",
    "area1_log10_sigma_total_LOS_g_cm2",
    "area2_log10_sigma_dense_total_LOS_g_cm2",
    "area2_log10_sigma_rare_total_LOS_g_cm2",
    "area2_corundum_fraction",
)

INITIAL_GUESS = np.asarray(
    [
        160.0,
        420.0,
        3.5,
        0.5,
        5.0,
        -16.0,
        -18.0,
        -4.0,
        -3.5,
        -5.5,
        0.005,
    ],
    dtype=float,
)

WALKER_SCATTER = np.asarray(
    [5.0, 8.0, 0.08, 0.10, 0.50, 0.30, 0.30, 0.15, 0.15, 0.15, 0.001],
    dtype=float,
)


def derive_state(theta):
    """Derive all columns, densities, grain sizes, and temperatures."""
    (
        temperature_dense_1,
        temperature_rare_1,
        p_size,
        q_density_size,
        a_dense_1,
        log_rho_dense_1,
        log_rho_rare_1,
        log_sigma_total_1,
        log_sigma_dense_2_total,
        log_sigma_rare_2_total,
        corundum_2,
    ) = np.asarray(theta, dtype=float)

    f1 = base.AREA1_FILLING_FRACTION
    f2 = base.AREA2_FILLING_FRACTION
    rho_dense_1 = 10.0**log_rho_dense_1
    rho_rare_1 = 10.0**log_rho_rare_1
    sigma_total_1_los = 10.0**log_sigma_total_1
    sigma_dense_2_total_los = 10.0**log_sigma_dense_2_total
    sigma_rare_2_total_los = 10.0**log_sigma_rare_2_total

    effective_density_1 = f1 * rho_dense_1 + (1.0 - f1) * rho_rare_1
    path_length_1 = sigma_total_1_los / effective_density_1
    sigma_dense_1_weighted_los = f1 * rho_dense_1 * path_length_1
    sigma_rare_1_weighted_los = (
        (1.0 - f1) * rho_rare_1 * path_length_1
    )
    sigma_dense_1_weighted_radial = (
        base.AREA1_LOS_TO_RADIAL * sigma_dense_1_weighted_los
    )
    sigma_rare_1_weighted_radial = (
        base.AREA1_LOS_TO_RADIAL * sigma_rare_1_weighted_los
    )

    # The fitted Area-2 phase columns are sums over the two LOS crossings.
    # Smith fixes the RADIAL ratio first; each wall then receives its own LOS
    # projection exactly as in the preceding two-wall implementation.
    sigma_dense_2_front_radial = (
        sigma_dense_2_total_los
        / geometry.AREA2_TOTAL_LOS_PER_FRONT_RADIAL
    )
    sigma_rare_2_front_radial = (
        sigma_rare_2_total_los
        / geometry.AREA2_TOTAL_LOS_PER_FRONT_RADIAL
    )
    sigma_dense_2_back_radial = (
        geometry.AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO
        * sigma_dense_2_front_radial
    )
    sigma_rare_2_back_radial = (
        geometry.AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO
        * sigma_rare_2_front_radial
    )
    sigma_dense_2_front_los = (
        sigma_dense_2_front_radial
        / geometry.AREA2_FRONT_LOS_TO_RADIAL
    )
    sigma_rare_2_front_los = (
        sigma_rare_2_front_radial
        / geometry.AREA2_FRONT_LOS_TO_RADIAL
    )
    sigma_dense_2_back_los = (
        sigma_dense_2_back_radial
        / geometry.AREA2_BACK_LOS_TO_RADIAL
    )
    sigma_rare_2_back_los = (
        sigma_rare_2_back_radial
        / geometry.AREA2_BACK_LOS_TO_RADIAL
    )

    # The earlier growth prescription compared filling-factor-weighted phase
    # columns.  Here the comparison is made using RADIAL material columns so
    # projected geometry cannot masquerade as a density difference.
    dense_front_weighted_radial = f2 * sigma_dense_2_front_radial
    rare_front_weighted_radial = (1.0 - f2) * sigma_rare_2_front_radial
    dense_back_weighted_radial = f2 * sigma_dense_2_back_radial
    rare_back_weighted_radial = (1.0 - f2) * sigma_rare_2_back_radial

    rho_dense_2_front = rho_dense_1 * (
        dense_front_weighted_radial / sigma_dense_1_weighted_radial
    )
    rho_rare_2_front = rho_rare_1 * (
        rare_front_weighted_radial / sigma_rare_1_weighted_radial
    )
    rho_dense_2_back = rho_dense_1 * (
        dense_back_weighted_radial / sigma_dense_1_weighted_radial
    )
    rho_rare_2_back = rho_rare_1 * (
        rare_back_weighted_radial / sigma_rare_1_weighted_radial
    )

    a_rare_1 = a_dense_1 * (rho_rare_1 / rho_dense_1) ** q_density_size
    a_dense_2_front = a_dense_1 * (
        rho_dense_2_front / rho_dense_1
    ) ** q_density_size
    a_rare_2_front = a_rare_1 * (
        rho_rare_2_front / rho_rare_1
    ) ** q_density_size
    a_dense_2_back = a_dense_1 * (
        rho_dense_2_back / rho_dense_1
    ) ** q_density_size
    a_rare_2_back = a_rare_1 * (
        rho_rare_2_back / rho_rare_1
    ) ** q_density_size

    return {
        "T_dense_1": temperature_dense_1,
        "T_rare_1": temperature_rare_1,
        "T_dense_2_front": geometry.AREA2_FRONT_TEMPERATURE_SCALE * temperature_dense_1,
        "T_rare_2_front": geometry.AREA2_FRONT_TEMPERATURE_SCALE * temperature_rare_1,
        "T_dense_2_back": geometry.AREA2_BACK_TEMPERATURE_SCALE * temperature_dense_1,
        "T_rare_2_back": geometry.AREA2_BACK_TEMPERATURE_SCALE * temperature_rare_1,
        "p_size": p_size,
        "q_density_size": q_density_size,
        "a_dense_1": a_dense_1,
        "a_rare_1": a_rare_1,
        "a_dense_2_front": a_dense_2_front,
        "a_rare_2_front": a_rare_2_front,
        "a_dense_2_back": a_dense_2_back,
        "a_rare_2_back": a_rare_2_back,
        "rho_dense_1": rho_dense_1,
        "rho_rare_1": rho_rare_1,
        "rho_dense_2_front": rho_dense_2_front,
        "rho_rare_2_front": rho_rare_2_front,
        "rho_dense_2_back": rho_dense_2_back,
        "rho_rare_2_back": rho_rare_2_back,
        "sigma_total_1_los": sigma_total_1_los,
        "path_length_1": path_length_1,
        "sigma_dense_1_weighted_los": sigma_dense_1_weighted_los,
        "sigma_rare_1_weighted_los": sigma_rare_1_weighted_los,
        "sigma_dense_1_weighted_radial": sigma_dense_1_weighted_radial,
        "sigma_rare_1_weighted_radial": sigma_rare_1_weighted_radial,
        "sigma_dense_2_total_los": sigma_dense_2_total_los,
        "sigma_rare_2_total_los": sigma_rare_2_total_los,
        "sigma_dense_2_front_los": sigma_dense_2_front_los,
        "sigma_rare_2_front_los": sigma_rare_2_front_los,
        "sigma_dense_2_back_los": sigma_dense_2_back_los,
        "sigma_rare_2_back_los": sigma_rare_2_back_los,
        "sigma_dense_2_front_radial": sigma_dense_2_front_radial,
        "sigma_rare_2_front_radial": sigma_rare_2_front_radial,
        "sigma_dense_2_back_radial": sigma_dense_2_back_radial,
        "sigma_rare_2_back_radial": sigma_rare_2_back_radial,
        "corundum_2": corundum_2,
        "iron_2": 1.0 - base.SILICATE_FRACTION - corundum_2,
    }


def build_kappas(state, wavelength_area1, wavelength_area2):
    """Build the six phase/location opacities implied by one state."""
    p_value = state["p_size"]
    area1_corundum = base.AREA1_CORUNDUM_FRACTION
    return {
        "dense_1": powerlaw_mixed_kappa(
            p_value, state["a_dense_1"], wavelength_area1, area1_corundum
        ),
        "rare_1": powerlaw_mixed_kappa(
            p_value, state["a_rare_1"], wavelength_area1, area1_corundum
        ),
        "dense_2_front": powerlaw_mixed_kappa(
            p_value, state["a_dense_2_front"], wavelength_area2, state["corundum_2"]
        ),
        "rare_2_front": powerlaw_mixed_kappa(
            p_value, state["a_rare_2_front"], wavelength_area2, state["corundum_2"]
        ),
        "dense_2_back": powerlaw_mixed_kappa(
            p_value, state["a_dense_2_back"], wavelength_area2, state["corundum_2"]
        ),
        "rare_2_back": powerlaw_mixed_kappa(
            p_value, state["a_rare_2_back"], wavelength_area2, state["corundum_2"]
        ),
    }


def evaluate_models(theta, wavelength_area1, wavelength_area2):
    """Evaluate the Area-1 slab and the two-wall Area-2 sightline."""
    state = derive_state(theta)
    kappas = build_kappas(state, wavelength_area1, wavelength_area2)

    tau_1 = (
        kappas["dense_1"] * state["sigma_dense_1_weighted_los"]
        + kappas["rare_1"] * state["sigma_rare_1_weighted_los"]
    )
    source_integral_1 = (
        kappas["dense_1"]
        * state["sigma_dense_1_weighted_los"]
        * base.planck_nu_from_wavelength(wavelength_area1, state["T_dense_1"])
        + kappas["rare_1"]
        * state["sigma_rare_1_weighted_los"]
        * base.planck_nu_from_wavelength(wavelength_area1, state["T_rare_1"])
    )
    intensity_1 = np.divide(
        -np.expm1(-tau_1) * source_integral_1,
        tau_1,
        out=np.zeros_like(tau_1),
        where=tau_1 > 0.0,
    )
    flux_1 = intensity_1 * base.PIXEL_SOLID_ANGLE_SR

    tau_dense_front = (
        kappas["dense_2_front"] * state["sigma_dense_2_front_los"]
    )
    tau_rare_front = (
        kappas["rare_2_front"] * state["sigma_rare_2_front_los"]
    )
    tau_dense_back = (
        kappas["dense_2_back"] * state["sigma_dense_2_back_los"]
    )
    tau_rare_back = (
        kappas["rare_2_back"] * state["sigma_rare_2_back_los"]
    )

    dense_front = base.planck_nu_from_wavelength(
        wavelength_area2, state["T_dense_2_front"]
    ) * -np.expm1(-tau_dense_front)
    rare_front = base.planck_nu_from_wavelength(
        wavelength_area2, state["T_rare_2_front"]
    ) * -np.expm1(-tau_rare_front)
    dense_back_seen = (
        np.exp(-tau_dense_front)
        * base.planck_nu_from_wavelength(
            wavelength_area2, state["T_dense_2_back"]
        )
        * -np.expm1(-tau_dense_back)
    )
    rare_back_seen = (
        np.exp(-tau_rare_front)
        * base.planck_nu_from_wavelength(
            wavelength_area2, state["T_rare_2_back"]
        )
        * -np.expm1(-tau_rare_back)
    )

    front_intensity = (
        base.AREA2_FILLING_FRACTION * dense_front
        + (1.0 - base.AREA2_FILLING_FRACTION) * rare_front
    )
    back_seen_intensity = (
        base.AREA2_FILLING_FRACTION * dense_back_seen
        + (1.0 - base.AREA2_FILLING_FRACTION) * rare_back_seen
    )
    front_flux = front_intensity * base.PIXEL_SOLID_ANGLE_SR
    back_seen_flux = back_seen_intensity * base.PIXEL_SOLID_ANGLE_SR
    flux_2 = front_flux + back_seen_flux
    state["front_flux"] = front_flux
    state["back_seen_flux"] = back_seen_flux
    state["front_flux_fraction"] = np.divide(
        front_flux, flux_2, out=np.zeros_like(front_flux), where=flux_2 > 0.0
    )
    state["back_flux_fraction"] = np.divide(
        back_seen_flux,
        flux_2,
        out=np.zeros_like(back_seen_flux),
        where=flux_2 > 0.0,
    )
    state["tau_dense_front"] = tau_dense_front
    state["tau_rare_front"] = tau_rare_front
    state["tau_dense_back"] = tau_dense_back
    state["tau_rare_back"] = tau_rare_back
    return flux_1, flux_2, state, kappas


def log_prior(theta):
    """Uniform bounds plus derived-size and temperature constraints."""
    (
        temperature_dense_1,
        temperature_rare_1,
        p_size,
        q_density_size,
        a_dense_1,
        log_rho_dense_1,
        log_rho_rare_1,
        log_sigma_total_1,
        log_sigma_dense_2_total,
        log_sigma_rare_2_total,
        corundum_2,
    ) = np.asarray(theta, dtype=float)
    if not (50.0 <= temperature_dense_1 < temperature_rare_1 <= 800.0):
        return -np.inf
    if not (P_MIN < p_size < P_MAX):
        return -np.inf
    if not (Q_MIN <= q_density_size <= Q_MAX):
        return -np.inf
    if not (A_MIN_MICRON < a_dense_1 <= A_MAX_MICRON):
        return -np.inf
    if not (-20.0 <= log_rho_dense_1 <= -4.0):
        return -np.inf
    if not (-22.0 <= log_rho_rare_1 <= -5.0):
        return -np.inf
    if not (log_rho_dense_1 > log_rho_rare_1):
        return -np.inf
    if not (-6.0 <= log_sigma_total_1 <= -1.0):
        return -np.inf
    if not (-6.0 <= log_sigma_dense_2_total <= -1.0):
        return -np.inf
    if not (-6.0 <= log_sigma_rare_2_total <= -3.5):
        return -np.inf
    if not (log_sigma_dense_2_total > log_sigma_rare_2_total):
        return -np.inf
    if not (0.0 <= corundum_2 <= 0.010):
        return -np.inf

    state = derive_state(theta)
    for name in (
        "T_dense_2_front",
        "T_rare_2_front",
        "T_dense_2_back",
        "T_rare_2_back",
    ):
        if not (50.0 <= state[name] <= 800.0):
            return -np.inf
    sizes = np.asarray(
        [
            state["a_dense_1"],
            state["a_rare_1"],
            state["a_dense_2_front"],
            state["a_rare_2_front"],
            state["a_dense_2_back"],
            state["a_rare_2_back"],
        ]
    )
    if not np.all(np.isfinite(sizes)):
        return -np.inf
    if np.any(sizes < A_MIN_MICRON) or np.any(sizes > A_MAX_MICRON):
        return -np.inf
    return 0.0


def log_likelihood_components(theta, area1_data, area2_data):
    model_1, model_2, _, _ = evaluate_models(
        theta, area1_data[0], area2_data[0]
    )
    return (
        base.one_area_log_likelihood(model_1, area1_data[1], area1_data[2]),
        base.one_area_log_likelihood(model_2, area2_data[1], area2_data[2]),
    )


def log_probability(theta, area1_data, area2_data):
    prior = log_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    likelihood_1, likelihood_2 = log_likelihood_components(
        theta, area1_data, area2_data
    )
    if not np.isfinite(likelihood_1 + likelihood_2):
        return -np.inf
    return prior + likelihood_1 + likelihood_2


def initialize_walkers(number_walkers, generator):
    positions = []
    attempts = 0
    while len(positions) < number_walkers:
        trial = INITIAL_GUESS + WALKER_SCATTER * generator.standard_normal(
            len(PARAMETER_NAMES)
        )
        attempts += 1
        if np.isfinite(log_prior(trial)):
            positions.append(trial)
        if attempts > 200000:
            raise RuntimeError("Could not initialize valid walkers.")
    return np.asarray(positions)


# ---------------------------------------------------------------------------
# Mass calculation: Area 2 reports the 80-degree foreground normalization
# ---------------------------------------------------------------------------
def calculate_mass_arrays(samples):
    samples = np.asarray(samples, dtype=float)
    sigma_los_1 = 10.0 ** samples[:, 7]
    sigma_los_2_total = (
        base.AREA2_FILLING_FRACTION * 10.0 ** samples[:, 8]
        + (1.0 - base.AREA2_FILLING_FRACTION) * 10.0 ** samples[:, 9]
    )
    sigma_radial_2_front = (
        sigma_los_2_total / geometry.AREA2_TOTAL_LOS_PER_FRONT_RADIAL
    )
    sigma_radial_2_back = (
        geometry.AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO
        * sigma_radial_2_front
    )
    sigma_los_2_front = (
        sigma_radial_2_front / geometry.AREA2_FRONT_LOS_TO_RADIAL
    )
    sigma_los_2_back = (
        sigma_radial_2_back / geometry.AREA2_BACK_LOS_TO_RADIAL
    )
    sigma_radial_1 = base.AREA1_LOS_TO_RADIAL * sigma_los_1

    dust_corrected_2_front = (
        sigma_radial_2_front
        / geometry.AREA2_FRONT_K_RADIAL_G_CM2_PER_MSUN
    )
    dust_corrected_2_back = (
        sigma_radial_2_back
        / geometry.AREA2_BACK_K_RADIAL_G_CM2_PER_MSUN
    )
    dust_from_summed_los = sigma_los_2_total / (
        geometry.AREA2_FRONT_K_LOS_G_CM2_PER_MSUN
        + geometry.AREA2_BACK_K_LOS_G_CM2_PER_MSUN
    )
    if not np.allclose(
        sigma_los_2_front + sigma_los_2_back,
        sigma_los_2_total,
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise RuntimeError("Area-2 LOS partition failed.")
    if not np.allclose(
        dust_corrected_2_front,
        dust_corrected_2_back,
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise RuntimeError("Front/back corrected mass identity failed.")
    if not np.allclose(
        dust_corrected_2_front,
        dust_from_summed_los,
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise RuntimeError("Summed-LOS mass identity failed.")
    return {
        "sigma_los_1": sigma_los_1,
        "sigma_radial_1": sigma_radial_1,
        "dust_uncorrected_1": sigma_los_1 / base.AREA1_K_RADIAL_G_CM2_PER_MSUN,
        "dust_corrected_1": sigma_radial_1 / base.AREA1_K_RADIAL_G_CM2_PER_MSUN,
        "sigma_los_2_total": sigma_los_2_total,
        "sigma_los_2_front": sigma_los_2_front,
        "sigma_los_2_back": sigma_los_2_back,
        "sigma_radial_2_front": sigma_radial_2_front,
        "sigma_radial_2_back": sigma_radial_2_back,
        "dust_uncorrected_2_front": (
            sigma_los_2_front
            / geometry.AREA2_FRONT_K_RADIAL_G_CM2_PER_MSUN
        ),
        "dust_corrected_2_front": dust_corrected_2_front,
        "dust_corrected_2_back": dust_corrected_2_back,
    }


def mass_rows(samples, maximum):
    arrays = calculate_mass_arrays(samples)
    best = calculate_mass_arrays(np.asarray(maximum)[None, :])
    factor = base.TOTAL_TO_DUST_FACTOR
    quantities = (
        ("area1_total_mass_without_LOS_correction", "dust_uncorrected_1"),
        ("area1_total_mass_with_LOS_correction", "dust_corrected_1"),
        ("area2_FRONT_ONLY_mass_without_LOS_correction", "dust_uncorrected_2_front"),
        ("area2_FRONT_ONLY_mass_with_LOS_correction", "dust_corrected_2_front"),
        ("area2_back_corrected_mass_CONSISTENCY_AUDIT_NOT_ADDED", "dust_corrected_2_back"),
    )
    rows = []
    for label, key in quantities:
        values = factor * arrays[key]
        q16, median, q84 = np.percentile(values, [16.0, 50.0, 84.0])
        rows.append((label, q16, median, q84, factor * best[key][0], "Msun"))
    return rows


def print_mass_summary(samples, maximum):
    print("\n--- Mass summary (factor 100 total/dust) ---")
    for label, q16, median, q84, map_value, _ in mass_rows(samples, maximum):
        print(
            f"{label}: {median:.6f} (-{median-q16:.6f}, "
            f"+{q84-median:.6f}) Msun; MAP={map_value:.6f} Msun"
        )
    print("The Area-2 back audit is not added to the foreground mass.")


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def derived_rows(state):
    names = (
        "T_dense_1", "T_rare_1", "T_dense_2_front", "T_rare_2_front",
        "T_dense_2_back", "T_rare_2_back", "p_size", "q_density_size",
        "a_dense_1", "a_rare_1", "a_dense_2_front", "a_rare_2_front",
        "a_dense_2_back", "a_rare_2_back", "rho_dense_1", "rho_rare_1",
        "rho_dense_2_front", "rho_rare_2_front", "rho_dense_2_back",
        "rho_rare_2_back", "sigma_total_1_los", "path_length_1",
        "sigma_dense_2_total_los", "sigma_rare_2_total_los",
        "sigma_dense_2_front_los", "sigma_rare_2_front_los",
        "sigma_dense_2_back_los", "sigma_rare_2_back_los",
        "sigma_dense_2_front_radial", "sigma_rare_2_front_radial",
        "sigma_dense_2_back_radial", "sigma_rare_2_back_radial",
    )
    rows = [(name, state[name]) for name in names]
    rows.extend(
        [
            ("temperature_emissivity_beta", geometry.EMISSIVITY_BETA),
            ("temperature_distance_exponent", geometry.TEMPERATURE_DISTANCE_EXPONENT),
            ("area2_front_temperature_scale", geometry.AREA2_FRONT_TEMPERATURE_SCALE),
            ("area2_back_temperature_scale", geometry.AREA2_BACK_TEMPERATURE_SCALE),
            ("C_front_80deg", geometry.AREA2_FRONT_LOS_TO_RADIAL),
            ("C_back_28p4287deg", geometry.AREA2_BACK_LOS_TO_RADIAL),
            ("Smith_back_to_front_radial_ratio", geometry.AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO),
            ("front_LOS_fraction_of_total", geometry.AREA2_FRONT_LOS_FRACTION_OF_TOTAL),
            ("back_LOS_fraction_of_total", geometry.AREA2_BACK_LOS_FRACTION_OF_TOTAL),
        ]
    )
    return rows


def print_derived_state_summary(state):
    """Expose the density/size construction used at maximum posterior."""
    print("\n--- Maximum-posterior density-dependent grain construction ---")
    print(
        "temperature beta / distance exponent : "
        f"{geometry.EMISSIVITY_BETA:.8g} / "
        f"{geometry.TEMPERATURE_DISTANCE_EXPONENT:.8g}"
    )
    print(
        "shared p / q                         : "
        f"{state['p_size']:.8g} / {state['q_density_size']:.8g}"
    )
    print(" location          rho_dense       rho_rare        a_dense     a_rare")
    print("                   [g cm^-3]       [g cm^-3]       [micron]    [micron]")
    for label, suffix in (
        ("Area 1", "1"),
        ("Area 2 front", "2_front"),
        ("Area 2 back", "2_back"),
    ):
        print(
            f" {label:16s} "
            f"{state[f'rho_dense_{suffix}']:13.6e}  "
            f"{state[f'rho_rare_{suffix}']:13.6e}  "
            f"{state[f'a_dense_{suffix}']:10.6f}  "
            f"{state[f'a_rare_{suffix}']:10.6f}"
        )

    print("\n--- Maximum-posterior Area-2 column partition ---")
    print(" phase       total LOS       front LOS      back LOS       front radial   back radial")
    print("             [g cm^-2]       [g cm^-2]      [g cm^-2]      [g cm^-2]      [g cm^-2]")
    for phase in ("dense", "rare"):
        print(
            f" {phase:10s} "
            f"{state[f'sigma_{phase}_2_total_los']:13.6e}  "
            f"{state[f'sigma_{phase}_2_front_los']:13.6e}  "
            f"{state[f'sigma_{phase}_2_back_los']:13.6e}  "
            f"{state[f'sigma_{phase}_2_front_radial']:13.6e}  "
            f"{state[f'sigma_{phase}_2_back_radial']:13.6e}"
        )
    print(
        "LOS closure check (front + back - total): dense="
        f"{state['sigma_dense_2_front_los'] + state['sigma_dense_2_back_los'] - state['sigma_dense_2_total_los']:.3e}, "
        "rare="
        f"{state['sigma_rare_2_front_los'] + state['sigma_rare_2_back_los'] - state['sigma_rare_2_total_los']:.3e} g cm^-2"
    )


def run_mass_only(args):
    output_directory = Path(args.output_dir).expanduser().resolve()
    posterior_path = output_directory / "posterior_samples.npz"
    if not posterior_path.is_file():
        raise FileNotFoundError(f"Missing posterior: {posterior_path}")
    with np.load(posterior_path, allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        log_probability_values = np.asarray(archive["log_probability"], dtype=float)
        saved_names = tuple(str(value) for value in archive["parameter_names"])
    if saved_names != PARAMETER_NAMES:
        raise ValueError("Saved posterior belongs to a different parameterization; rerun MCMC.")
    maximum = samples[int(np.argmax(log_probability_values))]
    print_mass_summary(samples, maximum)


def run_mcmc(args):
    import matplotlib.pyplot as plt

    area1_data = base.load_flux_data("lobe.edge.data")
    area2_data = base.load_flux_data("lobeedgedata2")
    generator = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    positions = initialize_walkers(args.walkers, generator)

    print("\n============================================================")
    print("TWO-WALL MCMC WITH DENSITY-DEPENDENT POWER-LAW GRAINS")
    print("============================================================")
    print("dn/da proportional to a^-p; a_min=0.05 micron")
    print("a_max proportional to rho^q; p and q shared everywhere")
    print("Area-2 total LOS = foreground LOS + rear LOS")
    print("Each wall uses its own LOS-to-radial correction and grain opacity")
    print("Rear emission is attenuated by the foreground")
    print("ln L = ln L_area1 + ln L_area2")
    number_data = len(area1_data[0]) + len(area2_data[0])
    print(f"Data points / fitted parameters = {number_data} / {len(PARAMETER_NAMES)}")
    print("CAUTION: the common Area-1 density scale is exactly degenerate.")
    print("No autocorrelation-time or 50-tau rule is used.")

    sampler = emcee.EnsembleSampler(
        args.walkers,
        len(PARAMETER_NAMES),
        log_probability,
        args=(area1_data, area2_data),
    )
    sampler.run_mcmc(positions, args.steps, progress=True)
    samples = sampler.get_chain(discard=args.burn, thin=args.thin, flat=True)
    log_probability_values = sampler.get_log_prob(
        discard=args.burn, thin=args.thin, flat=True
    )
    if len(samples) == 0:
        raise RuntimeError("No samples remain after burn-in/thinning.")

    q16, median, q84 = np.percentile(samples, [16.0, 50.0, 84.0], axis=0)
    maximum = samples[int(np.argmax(log_probability_values))]
    best_flux_1, best_flux_2, state, _ = evaluate_models(
        maximum, area1_data[0], area2_data[0]
    )
    likelihood_1, likelihood_2 = log_likelihood_components(
        maximum, area1_data, area2_data
    )

    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    parameter_path = output_directory / "parameter_summary.csv"
    derived_path = output_directory / "maximum_posterior_derived.csv"
    mass_path = output_directory / "mass_summary.csv"
    spectrum_path = output_directory / "maximum_posterior_area2_components.csv"
    posterior_path = output_directory / "posterior_samples.npz"

    write_csv(
        parameter_path,
        ["parameter", "posterior_q16", "posterior_median", "posterior_q84", "maximum_posterior"],
        zip(PARAMETER_NAMES, q16, median, q84, maximum),
    )
    write_csv(derived_path, ["derived_quantity", "maximum_posterior_value"], derived_rows(state))
    write_csv(
        mass_path,
        ["mass_quantity", "posterior_q16_Msun", "posterior_median_Msun", "posterior_q84_Msun", "maximum_posterior_Msun", "units"],
        mass_rows(samples, maximum),
    )
    required_front = area2_data[1] - state["back_seen_flux"]
    write_csv(
        spectrum_path,
        [
            "wavelength_micron", "observed_flux_cgs", "model_total_flux_cgs",
            "model_front_flux_cgs", "model_attenuated_back_flux_cgs",
            "observed_minus_back_flux_cgs", "front_flux_fraction",
            "attenuated_back_flux_fraction", "tau_dense_front",
            "tau_rare_front", "tau_dense_back", "tau_rare_back",
        ],
        zip(
            area2_data[0], area2_data[1], best_flux_2, state["front_flux"],
            state["back_seen_flux"], required_front, state["front_flux_fraction"],
            state["back_flux_fraction"], state["tau_dense_front"],
            state["tau_rare_front"], state["tau_dense_back"], state["tau_rare_back"],
        ),
    )
    np.savez_compressed(
        posterior_path,
        samples=samples,
        log_probability=log_probability_values,
        parameter_names=np.asarray(PARAMETER_NAMES),
    )

    wavelength_plot = np.logspace(
        np.log10(base.WAVELENGTH_MIN_MICRON),
        np.log10(base.WAVELENGTH_MAX_MICRON),
        1000,
    )
    plot_flux_1, plot_flux_2, plot_state, _ = evaluate_models(
        maximum, wavelength_plot, wavelength_plot
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    axes[0].errorbar(area1_data[0], area1_data[1], yerr=area1_data[2], fmt="o", color="black", label="Data")
    axes[0].plot(wavelength_plot, plot_flux_1, color="#9f2f1f", lw=2.0, label="Fit")
    axes[0].set_title("Area 1: density-dependent grain populations")
    axes[1].errorbar(area2_data[0], area2_data[1], yerr=area2_data[2], fmt="o", color="black", label="Data")
    axes[1].plot(wavelength_plot, plot_flux_2, color="#9f2f1f", lw=2.2, label="Total")
    axes[1].plot(wavelength_plot, plot_state["front_flux"], "--", color="#2f6f9f", lw=1.8, label="80 deg foreground")
    axes[1].plot(wavelength_plot, plot_state["back_seen_flux"], ":", color="#d17b20", lw=2.0, label="Attenuated 28.43 deg rear")
    axes[1].set_title("Area 2: summed LOS, two temperatures")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(base.WAVELENGTH_MIN_MICRON, base.WAVELENGTH_MAX_MICRON)
        axis.set_ylim(1.0e-26, 1.0e-19)
        axis.set_xlabel(r"Wavelength [$\mu$m]")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Flux [CGS]")
    figure.tight_layout()

    print("\n--- Posterior summaries ---")
    for name, low, middle, high in zip(PARAMETER_NAMES, q16, median, q84):
        print(f"{name:52s}: {middle:.9g} (-{middle-low:.3g}, +{high-middle:.3g})")
    print("\n--- Maximum-posterior Area-2 flux split ---")
    print(" wavelength  front fraction  attenuated-rear fraction  observed-minus-rear")
    for wavelength, front_fraction, back_fraction, remaining in zip(
        area2_data[0], state["front_flux_fraction"], state["back_flux_fraction"], required_front
    ):
        print(f" {wavelength:9.4f}  {front_fraction:14.6%}  {back_fraction:24.6%}  {remaining:.6e}")
    print(
        "\nchi2 Area 1 / Area 2 / total = "
        f"{-2.0*likelihood_1:.6f} / {-2.0*likelihood_2:.6f} / "
        f"{-2.0*(likelihood_1+likelihood_2):.6f}"
    )
    print(f"mean acceptance fraction = {np.mean(sampler.acceptance_fraction):.6f}")
    print_derived_state_summary(state)
    print_mass_summary(samples, maximum)
    print("\nSaved:")
    for path in (parameter_path, derived_path, mass_path, spectrum_path, posterior_path):
        print(path)
    print("The graph is displayed interactively and is not saved.\n")

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Two-wall MCMC with density-dependent power-law grains."
    )
    parser.add_argument("--walkers", type=int, default=50)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument(
        "--output-dir",
        default=str(
            TWO_PHASE_RESULTS / "aligned_overlap_diagnostic"
        ),
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--mass-only", action="store_true")
    args = parser.parse_args()
    if args.walkers < 2 * len(PARAMETER_NAMES):
        parser.error(f"walkers must be at least {2*len(PARAMETER_NAMES)}")
    if args.steps < 2:
        parser.error("steps must be at least 2")
    if args.burn < 0 or args.burn >= args.steps:
        parser.error("burn must satisfy 0 <= burn < steps")
    if args.thin < 1:
        parser.error("thin must be at least 1")
    return args


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.mass_only:
        run_mass_only(arguments)
    else:
        run_mcmc(arguments)
