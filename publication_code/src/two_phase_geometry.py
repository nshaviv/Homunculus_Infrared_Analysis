"""
Joint Eta Carinae MCMC with Area-2 temperatures bound to Area 1.

Physical interpretation
-----------------------
Area 1 is represented by the shell at absolute latitude 60 degrees.  The
Area-2 aperture is transparent and its line of sight intersects the shell at
two locations:

* foreground polar wall: absolute latitude 80 degrees;
* background wall: absolute latitude 28.4287 degrees.

Only the Area-1 dense and rare temperatures are fitted.  During EVERY
posterior evaluation the four Area-2 temperatures are derived from

    T(r) = T_area1 * (R_area1 / r)^[2/(4+beta)].

The effective emissivity index beta is fixed within any one run and varied
only between controlled sensitivity runs.  It is used only for this
deterministic temperature--distance relation; the spectral opacity remains
the full Mie opacity inherited from the original model.  beta=0 is the
default and exactly reproduces the original square-root distance law.

Consequently the 80-degree wall is colder than Area 1, while the background
wall is warmer than Area 1.  The foreground and attenuated background
intensities are calculated separately and added before evaluating the Area-2
likelihood.  This is not an after-the-fit temperature correction.

The fitted Area-2 phase columns are the TOTAL observed LOS columns through the
two shell crossings.  The report's normalized mass-per-solid-angle model first
fixes the radial-column ratio between 28.4287 and 80 degrees.  The two local
projection factors then convert those radial pieces to LOS columns.  In every
posterior evaluation the front and back LOS columns are solved by requiring
their sum to equal the fitted total LOS column.  Thus the warmer background
flux is included in the likelihood without adding a background-column
parameter and without redefining the fitted total as a front-only quantity.

Both crossings are fitted to the observed Area-2 flux.  The reported Area-2
mass uses only the radial column assigned to the 80-degree foreground after
the exact two-wall partition.  No fixed percentage is subtracted after fitting.

Area 1 retains its sideways mixed-slab transfer equation.  Area 2 retains its
dense/rare projected filling fraction and independent Mie size/composition
parameters.  The joint likelihood remains

    ln L_joint = ln L_area1 + ln L_area2.

No autocorrelation-time or 50-tau stopping rule is used.

This file imports the opacity and data utilities from ``two_phase_base.py``.
The modules are distributed together in the publication package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import emcee
import numpy as np

import two_phase_base as original
import temperature_scaling as temperature_law

from project_paths import TWO_PHASE_RESULTS


# ---------------------------------------------------------------------------
# Fixed geometry and deterministic temperature/column transformations
# ---------------------------------------------------------------------------
# The 60- and 80-degree radii come from the adopted PCHIP velocity curve and
# the common 152.9 yr ballistic age.  Only their ratio affects temperature.
AREA1_LATITUDE_DEG = 60.0
AREA2_FRONT_LATITUDE_DEG = 80.0
AREA2_BACK_LATITUDE_DEG = 28.4287

AREA1_RADIUS_CM = 2.7970595086102778e17
AREA2_FRONT_RADIUS_CM = 3.1466426279867880e17

# The STL ray calculation gives R_80/R_back = 1.2910403499325787.  This
# geometric radius is used only for the user-requested temperature scaling.
AREA2_FRONT_TO_BACK_RADIUS_RATIO = 1.2910403499325787
AREA2_BACK_TEMPERATURE_RADIUS_CM = (
    AREA2_FRONT_RADIUS_CM / AREA2_FRONT_TO_BACK_RADIUS_RATIO
)

# The default beta=0 reproduces the original square-root distance law.  The
# values below are deterministic globals, not fitted MCMC parameters.  The
# sensitivity runner updates them once before each independent MCMC run.
EMISSIVITY_BETA = temperature_law.emissivity_beta_from_environment()
TEMPERATURE_DISTANCE_EXPONENT = np.nan
AREA2_FRONT_TEMPERATURE_SCALE = np.nan
AREA2_BACK_TEMPERATURE_SCALE = np.nan


def configure_temperature_beta(beta):
    """Set the fixed temperature law used by every posterior evaluation."""
    global EMISSIVITY_BETA
    global TEMPERATURE_DISTANCE_EXPONENT
    global AREA2_FRONT_TEMPERATURE_SCALE
    global AREA2_BACK_TEMPERATURE_SCALE

    EMISSIVITY_BETA = temperature_law.validate_emissivity_beta(beta)
    TEMPERATURE_DISTANCE_EXPONENT = (
        temperature_law.temperature_distance_exponent(EMISSIVITY_BETA)
    )
    AREA2_FRONT_TEMPERATURE_SCALE = float(
        temperature_law.temperature_scale(
            AREA1_RADIUS_CM, AREA2_FRONT_RADIUS_CM, EMISSIVITY_BETA
        )
    )
    AREA2_BACK_TEMPERATURE_SCALE = float(
        temperature_law.temperature_scale(
            AREA1_RADIUS_CM,
            AREA2_BACK_TEMPERATURE_RADIUS_CM,
            EMISSIVITY_BETA,
        )
    )


configure_temperature_beta(EMISSIVITY_BETA)

# Separate per-latitude responses from the report's equations.  The PCHIP
# dM/dOmega curve is normalized to 1 Msun, R(lambda)=v(lambda)*152.9 yr, and
# Sigma_LOS=Sigma_radial*|rhat.nhat|/|nhat.ellhat|.  The local surface normals
# use the report's axisymmetric velocity surface at the azimuths of the two
# STL ray intersections.  The 101 homologous layers have the same projection
# ratio, so the LOS-to-radial factors are independent of launch-time layer.
AREA2_FRONT_K_RADIAL_G_CM2_PER_MSUN = 0.003484747003417672
AREA2_BACK_K_RADIAL_G_CM2_PER_MSUN = 0.006892305619494989
# The report's mass response uses its Smith ballistic radius at 28.4287 deg,
# not the STL radius used by the separate temperature prescription.
AREA2_BACK_SMITH_MASS_RADIUS_CM = 9.83683342039328e16
AREA2_FRONT_LOS_TO_RADIAL = 0.701259920895279
AREA2_BACK_LOS_TO_RADIAL = 1.5443629301639
AREA2_FRONT_K_LOS_G_CM2_PER_MSUN = (
    AREA2_FRONT_K_RADIAL_G_CM2_PER_MSUN / AREA2_FRONT_LOS_TO_RADIAL
)
AREA2_BACK_K_LOS_G_CM2_PER_MSUN = (
    AREA2_BACK_K_RADIAL_G_CM2_PER_MSUN / AREA2_BACK_LOS_TO_RADIAL
)

# Smith first fixes the radial-column ratio.  Only after imposing that physical
# ratio do we project each wall into LOS space.  The final LOS ratio below is a
# derived diagnostic, not the starting point of the partition.
AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO = (
    AREA2_BACK_K_RADIAL_G_CM2_PER_MSUN
    / AREA2_FRONT_K_RADIAL_G_CM2_PER_MSUN
)
AREA2_BACK_TO_FRONT_LOS_COLUMN_RATIO = (
    AREA2_BACK_K_LOS_G_CM2_PER_MSUN
    / AREA2_FRONT_K_LOS_G_CM2_PER_MSUN
)

# If S_total_LOS is the fitted column for one phase, write
#
#   S_radial,back = q_radial S_radial,front
#   S_total_LOS = S_radial,front/C_front
#               + S_radial,back/C_back.
#
# These constants solve that equation.  They also guarantee that the two LOS
# pieces sum exactly to the fitted total.
AREA2_TOTAL_LOS_PER_FRONT_RADIAL = (
    1.0 / AREA2_FRONT_LOS_TO_RADIAL
    + AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO
    / AREA2_BACK_LOS_TO_RADIAL
)
AREA2_FRONT_LOS_FRACTION_OF_TOTAL = (
    (1.0 / AREA2_FRONT_LOS_TO_RADIAL)
    / AREA2_TOTAL_LOS_PER_FRONT_RADIAL
)
AREA2_BACK_LOS_FRACTION_OF_TOTAL = (
    (
        AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO
        / AREA2_BACK_LOS_TO_RADIAL
    )
    / AREA2_TOTAL_LOS_PER_FRONT_RADIAL
)


# ---------------------------------------------------------------------------
# Fourteen fitted parameters: the two independent Area-2 temperatures from
# the old model are deliberately absent.
# ---------------------------------------------------------------------------
PARAMETER_NAMES = (
    "area1_log10_rho_dense_g_cm3",
    "area1_log10_rho_rare_g_cm3",
    "area1_T_dense_K",
    "area1_T_rare_K",
    "area1_w_small",
    "area1_a_small_micron",
    "area1_a_large_micron",
    "area1_log10_sigma_total_g_cm2",
    "area2_log10_sigma_dense_total_LOS_g_cm2",
    "area2_log10_sigma_rare_total_LOS_g_cm2",
    "area2_corundum_fraction",
    "area2_w_small",
    "area2_a_small_micron",
    "area2_a_large_micron",
)

INITIAL_GUESS = np.asarray(
    [
        -16.0,  # Area 1 log10 rho_dense [g/cm^3]
        -18.0,  # Area 1 log10 rho_rare [g/cm^3]
        160.0,  # fitted Area 1 T_dense [K]
        420.0,  # fitted Area 1 T_rare [K]
        0.60,   # Area 1 small-grain weight
        0.20,   # Area 1 small radius [micron]
        3.80,   # Area 1 large radius [micron]
        -4.00,  # Area 1 log10 sigma_total LOS [g/cm^2]
        -3.50,  # Area 2 TOTAL (front + back) log10 sigma_dense LOS [g/cm^2]
        -5.50,  # Area 2 TOTAL (front + back) log10 sigma_rare LOS [g/cm^2]
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
        0.15,
        0.15,
        0.001,
        0.05,
        0.03,
        0.40,
    ],
    dtype=float,
)


def derive_state(theta):
    """Expand one 14-parameter vector into fitted and derived quantities."""
    (
        log_rho_d_1,
        log_rho_r_1,
        T_d_1,
        T_r_1,
        w_small_1,
        a_small_1,
        a_large_1,
        log_sigma_total_1,
        log_sigma_d_2,
        log_sigma_r_2,
        d2_2,
        w_small_2,
        a_small_2,
        a_large_2,
    ) = np.asarray(theta, dtype=float)

    rho_d_1 = 10.0**log_rho_d_1
    rho_r_1 = 10.0**log_rho_r_1
    sigma_total_1 = 10.0**log_sigma_total_1
    effective_density_1 = (
        original.AREA1_FILLING_FRACTION * rho_d_1
        + (1.0 - original.AREA1_FILLING_FRACTION) * rho_r_1
    )
    x_total_1 = sigma_total_1 / effective_density_1
    sigma_d_1 = (
        original.AREA1_FILLING_FRACTION * rho_d_1 * x_total_1
    )
    sigma_r_1 = (
        (1.0 - original.AREA1_FILLING_FRACTION) * rho_r_1 * x_total_1
    )

    # The MCMC parameters are the TOTAL observed LOS columns through both
    # crossings.  Split each phase in radial space using the Smith ratio, then
    # project the two radial pieces separately.  Algebraically, for j=dense or
    # rare,
    #
    #   sigma_rad_back,j = q_rad sigma_rad_front,j
    #   sigma_LOS_total,j = sigma_rad_front,j/C_front
    #                     + sigma_rad_back,j/C_back.
    #
    # This is the requested normalization: front LOS + back LOS equals the
    # fitted total exactly.  Neither wall is itself a fitted column.
    sigma_d_2_total_LOS = 10.0**log_sigma_d_2
    sigma_r_2_total_LOS = 10.0**log_sigma_r_2
    sigma_d_2_front_radial = (
        sigma_d_2_total_LOS / AREA2_TOTAL_LOS_PER_FRONT_RADIAL
    )
    sigma_r_2_front_radial = (
        sigma_r_2_total_LOS / AREA2_TOTAL_LOS_PER_FRONT_RADIAL
    )
    sigma_d_2_back_radial = (
        AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO
        * sigma_d_2_front_radial
    )
    sigma_r_2_back_radial = (
        AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO
        * sigma_r_2_front_radial
    )
    sigma_d_2_front = (
        sigma_d_2_front_radial / AREA2_FRONT_LOS_TO_RADIAL
    )
    sigma_r_2_front = (
        sigma_r_2_front_radial / AREA2_FRONT_LOS_TO_RADIAL
    )
    sigma_d_2_back = (
        sigma_d_2_back_radial / AREA2_BACK_LOS_TO_RADIAL
    )
    sigma_r_2_back = (
        sigma_r_2_back_radial / AREA2_BACK_LOS_TO_RADIAL
    )
    sigma_effective_2_front = (
        original.AREA2_FILLING_FRACTION * sigma_d_2_front
        + (1.0 - original.AREA2_FILLING_FRACTION) * sigma_r_2_front
    )
    sigma_effective_2_back = (
        original.AREA2_FILLING_FRACTION * sigma_d_2_back
        + (1.0 - original.AREA2_FILLING_FRACTION) * sigma_r_2_back
    )
    sigma_effective_2_total_LOS = (
        original.AREA2_FILLING_FRACTION * sigma_d_2_total_LOS
        + (1.0 - original.AREA2_FILLING_FRACTION)
        * sigma_r_2_total_LOS
    )
    sigma_effective_2_front_radial = (
        original.AREA2_FILLING_FRACTION * sigma_d_2_front_radial
        + (1.0 - original.AREA2_FILLING_FRACTION)
        * sigma_r_2_front_radial
    )
    sigma_effective_2_back_radial = (
        original.AREA2_FILLING_FRACTION * sigma_d_2_back_radial
        + (1.0 - original.AREA2_FILLING_FRACTION)
        * sigma_r_2_back_radial
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
        # Both Area-2 temperature pairs are deterministic, not fitted.
        "T_d_2_front": AREA2_FRONT_TEMPERATURE_SCALE * T_d_1,
        "T_r_2_front": AREA2_FRONT_TEMPERATURE_SCALE * T_r_1,
        "T_d_2_back": AREA2_BACK_TEMPERATURE_SCALE * T_d_1,
        "T_r_2_back": AREA2_BACK_TEMPERATURE_SCALE * T_r_1,
        "sigma_d_2_front": sigma_d_2_front,
        "sigma_r_2_front": sigma_r_2_front,
        "sigma_d_2_total_LOS": sigma_d_2_total_LOS,
        "sigma_r_2_total_LOS": sigma_r_2_total_LOS,
        "sigma_d_2_front_radial": sigma_d_2_front_radial,
        "sigma_r_2_front_radial": sigma_r_2_front_radial,
        "sigma_d_2_back_radial": sigma_d_2_back_radial,
        "sigma_r_2_back_radial": sigma_r_2_back_radial,
        "sigma_effective_2_front": sigma_effective_2_front,
        "sigma_effective_2_front_radial": sigma_effective_2_front_radial,
        "sigma_d_2_back": sigma_d_2_back,
        "sigma_r_2_back": sigma_r_2_back,
        "sigma_effective_2_back": sigma_effective_2_back,
        "sigma_effective_2_back_radial": sigma_effective_2_back_radial,
        "sigma_effective_2_total_LOS": sigma_effective_2_total_LOS,
        "d2_2": d2_2,
        "d3_2": 1.0 - original.SILICATE_FRACTION - d2_2,
        "w_small_2": w_small_2,
        "a_small_2": a_small_2,
        "a_large_2": a_large_2,
    }


def evaluate_models(theta, wavelength_area1, wavelength_area2):
    """Return both model spectra and all fitted/derived state quantities."""
    state = derive_state(theta)

    opacity_1_grid = original.mixed_opacity(
        state["w_small_1"],
        state["a_small_1"],
        state["a_large_1"],
        original.AREA1_CORUNDUM_FRACTION,
    )
    kappa_1 = np.interp(
        wavelength_area1, original.WAVELENGTH_GRID, opacity_1_grid
    )

    # Area 1: the original sideways mixed slab.
    chi_effective_1 = kappa_1 * (
        original.AREA1_FILLING_FRACTION * state["rho_d_1"]
        + (1.0 - original.AREA1_FILLING_FRACTION) * state["rho_r_1"]
    )
    source_1 = kappa_1 * (
        original.AREA1_FILLING_FRACTION
        * state["rho_d_1"]
        * original.planck_nu_from_wavelength(
            wavelength_area1, state["T_d_1"]
        )
        + (1.0 - original.AREA1_FILLING_FRACTION)
        * state["rho_r_1"]
        * original.planck_nu_from_wavelength(
            wavelength_area1, state["T_r_1"]
        )
    )
    intensity_1 = (
        -np.expm1(-chi_effective_1 * state["x_total_1"])
        * source_1
        / chi_effective_1
    )
    flux_1 = intensity_1 * original.PIXEL_SOLID_ANGLE_SR

    opacity_2_grid = original.mixed_opacity(
        state["w_small_2"],
        state["a_small_2"],
        state["a_large_2"],
        state["d2_2"],
    )
    kappa_2 = np.interp(
        wavelength_area2, original.WAVELENGTH_GRID, opacity_2_grid
    )

    # Area 2 consists of two emitting surfaces along the same sightline.
    # For each projected phase, background emission is attenuated by the
    # corresponding foreground phase.  No optically-thin approximation is
    # made here.
    tau_d_front = kappa_2 * state["sigma_d_2_front"]
    tau_r_front = kappa_2 * state["sigma_r_2_front"]
    tau_d_back = kappa_2 * state["sigma_d_2_back"]
    tau_r_back = kappa_2 * state["sigma_r_2_back"]

    dense_front = original.planck_nu_from_wavelength(
        wavelength_area2, state["T_d_2_front"]
    ) * -np.expm1(-tau_d_front)
    rare_front = original.planck_nu_from_wavelength(
        wavelength_area2, state["T_r_2_front"]
    ) * -np.expm1(-tau_r_front)

    dense_back_seen = (
        np.exp(-tau_d_front)
        * original.planck_nu_from_wavelength(
            wavelength_area2, state["T_d_2_back"]
        )
        * -np.expm1(-tau_d_back)
    )
    rare_back_seen = (
        np.exp(-tau_r_front)
        * original.planck_nu_from_wavelength(
            wavelength_area2, state["T_r_2_back"]
        )
        * -np.expm1(-tau_r_back)
    )

    area2_front_intensity = (
        original.AREA2_FILLING_FRACTION * dense_front
        + (1.0 - original.AREA2_FILLING_FRACTION) * rare_front
    )
    area2_back_seen_intensity = (
        original.AREA2_FILLING_FRACTION * dense_back_seen
        + (1.0 - original.AREA2_FILLING_FRACTION) * rare_back_seen
    )
    flux_2_front = area2_front_intensity * original.PIXEL_SOLID_ANGLE_SR
    flux_2_back_seen = (
        area2_back_seen_intensity * original.PIXEL_SOLID_ANGLE_SR
    )
    flux_2 = flux_2_front + flux_2_back_seen
    front_flux_fraction = np.divide(
        flux_2_front,
        flux_2,
        out=np.zeros_like(flux_2_front, dtype=float),
        where=flux_2 > 0.0,
    )
    back_flux_fraction = np.divide(
        flux_2_back_seen,
        flux_2,
        out=np.zeros_like(flux_2_back_seen, dtype=float),
        where=flux_2 > 0.0,
    )

    # Arrays are stored only for diagnostic plotting/output; priors ignore them.
    state["flux_2_front"] = flux_2_front
    state["flux_2_back_seen"] = flux_2_back_seen
    state["front_flux_fraction"] = front_flux_fraction
    state["back_flux_fraction"] = back_flux_fraction
    state["tau_d_2_front"] = tau_d_front
    state["tau_r_2_front"] = tau_r_front
    state["tau_d_2_back"] = tau_d_back
    state["tau_r_2_back"] = tau_r_back
    return flux_1, flux_2, state


def log_prior(theta):
    """Uniform priors inherited from the independent-area fit."""
    state = derive_state(theta)

    log_rho_d_1 = theta[0]
    log_rho_r_1 = theta[1]
    log_sigma_total_1 = theta[7]
    log_sigma_d_2 = theta[8]
    log_sigma_r_2 = theta[9]

    if not (-20.0 <= log_rho_d_1 <= -14.0):
        return -np.inf
    if not (-22.0 <= log_rho_r_1 <= -16.0):
        return -np.inf
    if not (log_rho_d_1 > log_rho_r_1):
        return -np.inf
    if not (-6.0 <= log_sigma_total_1 <= -1.0):
        return -np.inf

    for name in (
        "T_d_1",
        "T_r_1",
        "T_d_2_front",
        "T_r_2_front",
        "T_d_2_back",
        "T_r_2_back",
    ):
        if not (50.0 <= state[name] <= 800.0):
            return -np.inf

    # Dense remains cooler than rare at all three physical locations.  Because
    # the scale factors are positive, the Area-1 condition guarantees both
    # derived Area-2 conditions as well; all are kept explicit for auditing.
    if not (state["T_d_1"] < state["T_r_1"]):
        return -np.inf
    if not (state["T_d_2_front"] < state["T_r_2_front"]):
        return -np.inf
    if not (state["T_d_2_back"] < state["T_r_2_back"]):
        return -np.inf

    for area in (1, 2):
        if not (0.2 <= state[f"w_small_{area}"] <= 1.0):
            return -np.inf
        if not (0.1 <= state[f"a_small_{area}"] <= 0.4):
            return -np.inf
        if not (0.1 <= state[f"a_large_{area}"] <= 7.0):
            return -np.inf
        if not (state[f"a_small_{area}"] < state[f"a_large_{area}"]):
            return -np.inf

    # The previous -3 lower bound incorrectly excluded lower-column solutions.
    # These are now total (front + back) LOS columns; keep a broad range and
    # let the combined two-wall spectrum constrain them.
    if not (-6.0 <= log_sigma_d_2 <= -1.0):
        return -np.inf
    if not (-6.0 <= log_sigma_r_2 <= -3.5):
        return -np.inf
    if not (log_sigma_d_2 > log_sigma_r_2):
        return -np.inf
    if not (0.0 <= state["d2_2"] <= 0.010):
        return -np.inf
    if not (0.0 <= state["d3_2"] <= 0.010):
        return -np.inf
    return 0.0


def log_likelihood_components(theta, area1_data, area2_data):
    model_1, model_2, _ = evaluate_models(
        theta, area1_data[0], area2_data[0]
    )
    return (
        original.one_area_log_likelihood(
            model_1, area1_data[1], area1_data[2]
        ),
        original.one_area_log_likelihood(
            model_2, area2_data[1], area2_data[2]
        ),
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


def calculate_mass_arrays(samples):
    """Calculate Area-1 and partitioned foreground Area-2 mass arrays.

    The Area-2 fit parameters are total LOS phase columns.  Smith's radial
    ratio and the two local projection factors partition each total into
    front/back LOS columns.  The reported Area-2 mass uses the resulting
    80-degree foreground radial column only.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(f"Expected N x {len(PARAMETER_NAMES)} samples.")

    sigma_los_1 = 10.0 ** samples[:, 7]
    sigma_los_2_total = (
        original.AREA2_FILLING_FRACTION * 10.0 ** samples[:, 8]
        + (1.0 - original.AREA2_FILLING_FRACTION)
        * 10.0 ** samples[:, 9]
    )
    sigma_radial_2_front = (
        sigma_los_2_total / AREA2_TOTAL_LOS_PER_FRONT_RADIAL
    )
    sigma_radial_2_back = (
        AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO
        * sigma_radial_2_front
    )
    sigma_los_2_front = (
        sigma_radial_2_front / AREA2_FRONT_LOS_TO_RADIAL
    )
    sigma_los_2_back = (
        sigma_radial_2_back / AREA2_BACK_LOS_TO_RADIAL
    )
    sigma_radial_1 = original.AREA1_LOS_TO_RADIAL * sigma_los_1

    dust_uncorrected_2_front = (
        sigma_los_2_front / AREA2_FRONT_K_RADIAL_G_CM2_PER_MSUN
    )
    dust_corrected_2_front = (
        sigma_radial_2_front / AREA2_FRONT_K_RADIAL_G_CM2_PER_MSUN
    )
    dust_uncorrected_2_back = (
        sigma_los_2_back / AREA2_BACK_K_RADIAL_G_CM2_PER_MSUN
    )
    dust_corrected_2_back = (
        sigma_radial_2_back / AREA2_BACK_K_RADIAL_G_CM2_PER_MSUN
    )
    # This is the same mass normalization written directly in the user's
    # requested summed-LOS form:
    # sigma_total_LOS = M * (K_LOS,front + K_LOS,back).
    dust_corrected_2_from_summed_los = sigma_los_2_total / (
        AREA2_FRONT_K_LOS_G_CM2_PER_MSUN
        + AREA2_BACK_K_LOS_G_CM2_PER_MSUN
    )

    # The LOS pieces must reconstruct the fitted total, the radial pieces must
    # obey Smith's ratio, and either crossing must recover the same global mass
    # normalization.  These checks catch both an incorrect partition and use
    # of the old aggregate C2 factor.
    if not np.allclose(
        sigma_los_2_front + sigma_los_2_back,
        sigma_los_2_total,
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise RuntimeError("Area-2 front/back LOS columns do not sum to total.")
    if not np.allclose(
        sigma_radial_2_back / sigma_radial_2_front,
        AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO,
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise RuntimeError("Area-2 Smith radial-column ratio failed.")
    if not np.allclose(
        dust_corrected_2_front,
        dust_corrected_2_back,
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise RuntimeError(
            "Independent front/back LOS corrections do not recover the same mass."
        )
    if not np.allclose(
        dust_corrected_2_front,
        dust_corrected_2_from_summed_los,
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise RuntimeError(
            "Summed-LOS normalization disagrees with the foreground radial mass."
        )

    return {
        "sigma_los_1": sigma_los_1,
        "sigma_radial_1": sigma_radial_1,
        "dust_uncorrected_1": (
            sigma_los_1 / original.AREA1_K_RADIAL_G_CM2_PER_MSUN
        ),
        "dust_corrected_1": (
            sigma_radial_1 / original.AREA1_K_RADIAL_G_CM2_PER_MSUN
        ),
        "sigma_los_2_total": sigma_los_2_total,
        "sigma_los_2_front": sigma_los_2_front,
        "sigma_los_2_back": sigma_los_2_back,
        "sigma_radial_2_front": sigma_radial_2_front,
        "sigma_radial_2_back": sigma_radial_2_back,
        "dust_uncorrected_2_back": dust_uncorrected_2_back,
        "dust_uncorrected_2_front": dust_uncorrected_2_front,
        "dust_corrected_2_back": dust_corrected_2_back,
        "dust_corrected_2_front": dust_corrected_2_front,
        "dust_corrected_2_from_summed_los": (
            dust_corrected_2_from_summed_los
        ),
    }


def print_mass_summary(samples, maximum_posterior_sample):
    arrays = calculate_mass_arrays(samples)
    best = calculate_mass_arrays(
        np.asarray(maximum_posterior_sample, dtype=float)[None, :]
    )
    factor = original.TOTAL_TO_DUST_FACTOR
    print("\n--- Requested mass calculations ---")
    print("Area 1 is calculated normally from its fitted LOS column.")
    print("Area 2 fits total dense/rare LOS columns through both walls.")
    print("Smith + the two projections split each total into front and back.")
    print("Only the resulting Area-2 foreground mass is retained below.")
    print("The 80-degree and 28.43-degree walls use separate LOS corrections.")
    print("Total masses use the requested total/dust factor of 100.")

    def quantiles(values):
        return np.percentile(values, [16.0, 50.0, 84.0])

    def interval_text(values):
        low, middle, high = quantiles(values)
        return (
            f"{middle:.6f} (-{middle-low:.6f}, +{high-middle:.6f})"
        )

    area1_before = factor * arrays["dust_uncorrected_1"]
    area1_after = factor * arrays["dust_corrected_1"]
    print("\nArea 1 mass:")
    print(f"  LOS-to-radial factor C1 = {original.AREA1_LOS_TO_RADIAL:.15f}")
    print(
        "  Sigma_LOS q16/median/q84    = "
        + " / ".join(f"{value:.6e}" for value in quantiles(arrays["sigma_los_1"]))
        + " g/cm^2"
    )
    print(
        "  Sigma_radial q16/median/q84 = "
        + " / ".join(f"{value:.6e}" for value in quantiles(arrays["sigma_radial_1"]))
        + " g/cm^2"
    )
    print(f"  Total mass WITHOUT correction = {interval_text(area1_before)} Msun")
    print(f"  Total mass WITH correction    = {interval_text(area1_after)} Msun")
    print(
        "  Maximum-posterior mass before/after = "
        f"{factor * best['dust_uncorrected_1'][0]:.6f} / "
        f"{factor * best['dust_corrected_1'][0]:.6f} Msun"
    )

    area2_back_before = factor * arrays["dust_uncorrected_2_back"]
    area2_front_before = factor * arrays["dust_uncorrected_2_front"]
    area2_back_after = factor * arrays["dust_corrected_2_back"]
    area2_front_after = factor * arrays["dust_corrected_2_front"]

    print("\nArea 2 foreground-only mass:")
    print(f"  Foreground C_80       = {AREA2_FRONT_LOS_TO_RADIAL:.15f}")
    print(f"  Background C_28.43    = {AREA2_BACK_LOS_TO_RADIAL:.15f}")
    print(
        f"  Smith back/front RADIAL-column ratio = "
        f"{AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO:.15f}"
    )
    print(
        f"  K_LOS,80 / K_LOS,28.43 = "
        f"{AREA2_FRONT_K_LOS_G_CM2_PER_MSUN:.12e} / "
        f"{AREA2_BACK_K_LOS_G_CM2_PER_MSUN:.12e} g/cm^2/Msun"
    )
    print(
        f"  Resulting back/front LOS-column ratio = "
        f"{AREA2_BACK_TO_FRONT_LOS_COLUMN_RATIO:.15f}"
    )
    print(
        "  FITTED TOTAL LOS column q16/median/q84      = "
        + " / ".join(
            f"{value:.6e}" for value in quantiles(arrays["sigma_los_2_total"])
        )
        + " g/cm^2"
    )
    print(
        "  SOLVED foreground LOS column q16/median/q84 = "
        + " / ".join(
            f"{value:.6e}" for value in quantiles(arrays["sigma_los_2_front"])
        )
        + " g/cm^2"
    )
    print(
        "  SOLVED background LOS column q16/median/q84 = "
        + " / ".join(
            f"{value:.6e}" for value in quantiles(arrays["sigma_los_2_back"])
        )
        + " g/cm^2"
    )
    print("\n  WITHOUT LOS correction:")
    print(f"    FRONT-ONLY mass estimate    = {interval_text(area2_front_before)} Msun")
    print(f"    Back-wall audit estimate    = {interval_text(area2_back_before)} Msun")
    print("\n  WITH LOS correction:")
    print(f"    FRONT-ONLY FINAL MASS       = {interval_text(area2_front_after)} Msun")
    print(f"    Back-wall consistency audit = {interval_text(area2_back_after)} Msun")
    print("    (The audit value must match; it is not added to the foreground mass.)")
    print(
        "  Maximum-posterior FRONT-ONLY mass before/after = "
        f"{factor * best['dust_uncorrected_2_front'][0]:.6f} / "
        f"{factor * best['dust_corrected_2_front'][0]:.6f} Msun"
    )


def mass_summary_rows(samples, maximum_posterior_sample):
    """Rows for a machine-readable posterior mass audit."""
    arrays = calculate_mass_arrays(samples)
    best = calculate_mass_arrays(
        np.asarray(maximum_posterior_sample, dtype=float)[None, :]
    )
    factor = original.TOTAL_TO_DUST_FACTOR
    requested = (
        ("area1_total_mass_without_LOS_correction", factor * arrays["dust_uncorrected_1"], factor * best["dust_uncorrected_1"][0]),
        ("area1_total_mass_with_LOS_correction", factor * arrays["dust_corrected_1"], factor * best["dust_corrected_1"][0]),
        ("area2_FRONT_ONLY_mass_without_LOS_correction", factor * arrays["dust_uncorrected_2_front"], factor * best["dust_uncorrected_2_front"][0]),
        ("area2_back_wall_mass_without_its_LOS_correction_AUDIT_ONLY", factor * arrays["dust_uncorrected_2_back"], factor * best["dust_uncorrected_2_back"][0]),
        ("area2_FRONT_ONLY_mass_with_LOS_correction", factor * arrays["dust_corrected_2_front"], factor * best["dust_corrected_2_front"][0]),
        ("area2_back_wall_mass_with_its_LOS_correction_CONSISTENCY_AUDIT", factor * arrays["dust_corrected_2_back"], factor * best["dust_corrected_2_back"][0]),
    )
    rows = []
    for name, values, maximum_value in requested:
        q16, median, q84 = np.percentile(values, [16.0, 50.0, 84.0])
        rows.append((name, q16, median, q84, maximum_value, "Msun"))
    return rows


def derived_temperature_rows(samples, maximum):
    """Return posterior summaries for temperatures absent from the vector."""
    values = {
        "area1_T_dense_K": samples[:, 2],
        "area1_T_rare_K": samples[:, 3],
        "area2_front_80deg_T_dense_K": (
            AREA2_FRONT_TEMPERATURE_SCALE * samples[:, 2]
        ),
        "area2_front_80deg_T_rare_K": (
            AREA2_FRONT_TEMPERATURE_SCALE * samples[:, 3]
        ),
        "area2_back_28p4287deg_T_dense_K": (
            AREA2_BACK_TEMPERATURE_SCALE * samples[:, 2]
        ),
        "area2_back_28p4287deg_T_rare_K": (
            AREA2_BACK_TEMPERATURE_SCALE * samples[:, 3]
        ),
    }
    maximum_state = derive_state(maximum)
    maximum_values = {
        "area1_T_dense_K": maximum_state["T_d_1"],
        "area1_T_rare_K": maximum_state["T_r_1"],
        "area2_front_80deg_T_dense_K": maximum_state["T_d_2_front"],
        "area2_front_80deg_T_rare_K": maximum_state["T_r_2_front"],
        "area2_back_28p4287deg_T_dense_K": maximum_state["T_d_2_back"],
        "area2_back_28p4287deg_T_rare_K": maximum_state["T_r_2_back"],
    }
    rows = []
    for name, array in values.items():
        q16, median, q84 = np.percentile(array, [16.0, 50.0, 84.0])
        rows.append((name, q16, median, q84, maximum_values[name]))
    return rows


def run_mass_only(args):
    output_directory = Path(args.output_dir).expanduser().resolve()
    posterior_path = output_directory / "posterior_samples.npz"
    if not posterior_path.is_file():
        raise FileNotFoundError(f"Missing posterior: {posterior_path}")
    with np.load(posterior_path, allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        log_probabilities = np.asarray(archive["log_probability"], dtype=float)
        saved_parameter_names = tuple(
            str(name) for name in archive["parameter_names"]
        )
    if saved_parameter_names != PARAMETER_NAMES:
        raise ValueError(
            "This posterior uses a different Area-2 column definition. Rerun "
            "the MCMC: the current parameters are total front+back LOS columns "
            "and are partitioned by the Smith radial ratio."
        )
    maximum = samples[int(np.argmax(log_probabilities))]
    print(f"Loaded {len(samples)} samples from {posterior_path}")
    print_mass_summary(samples, maximum)
    mass_path = (
        output_directory / "mass_summary_area1_and_area2_front_only.csv"
    )
    original.write_csv(
        mass_path,
        [
            "mass_quantity",
            "posterior_q16_Msun",
            "posterior_median_Msun",
            "posterior_q84_Msun",
            "maximum_posterior_Msun",
            "units",
        ],
        mass_summary_rows(samples, maximum),
    )
    print(f"\nSaved front-only mass summary: {mass_path}\n")


def run_joint_mcmc(args):
    import matplotlib.pyplot as plt

    area1_data = original.load_flux_data("lobe.edge.data")
    area2_data = original.load_flux_data("lobeedgedata2")
    generator = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    positions = initialize_walkers(args.walkers, generator)

    print("\n============================================================")
    print("JOINT MCMC: AREA 1 TEMPERATURES + TWO-WALL AREA 2")
    print("============================================================")
    print("ln L_joint = ln L_area1 + ln L_area2")
    print("Only Td1 and Tr1 are fitted temperatures.")
    print(
        f"T_80/T_60 = {AREA2_FRONT_TEMPERATURE_SCALE:.15f}; "
        f"T_back/T_60 = {AREA2_BACK_TEMPERATURE_SCALE:.15f}"
    )
    print(
        "Area-2 Smith background/foreground RADIAL-column ratio = "
        f"{AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO:.15f}"
    )
    print(
        "Resulting background/foreground LOS-column ratio = "
        f"{AREA2_BACK_TO_FRONT_LOS_COLUMN_RATIO:.15f}; "
        "their sum equals each fitted total LOS column."
    )
    print(
        "Separate LOS-to-radial factors C_80 / C_28.43 = "
        f"{AREA2_FRONT_LOS_TO_RADIAL:.15f} / "
        f"{AREA2_BACK_LOS_TO_RADIAL:.15f}"
    )
    n_data = len(area1_data[0]) + len(area2_data[0])
    print(f"Data points / free parameters = {n_data} / {len(PARAMETER_NAMES)}")
    print("No autocorrelation-time or 50-tau stopping rule is used.")
    print(f"Running {args.walkers} walkers for {args.steps} steps...")

    sampler = emcee.EnsembleSampler(
        args.walkers,
        len(PARAMETER_NAMES),
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
    temperature_path = output_directory / "derived_temperature_summary.csv"
    mass_path = output_directory / "mass_summary_area1_and_area2_front_only.csv"
    derived_path = output_directory / "maximum_posterior_derived.csv"
    spectrum_path = output_directory / "maximum_posterior_area2_components.csv"
    posterior_path = output_directory / "posterior_samples.npz"

    original.write_csv(
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
    original.write_csv(
        temperature_path,
        [
            "temperature",
            "posterior_q16_K",
            "posterior_median_K",
            "posterior_q84_K",
            "maximum_posterior_K",
        ],
        derived_temperature_rows(samples, maximum),
    )
    original.write_csv(
        mass_path,
        [
            "mass_quantity",
            "posterior_q16_Msun",
            "posterior_median_Msun",
            "posterior_q84_Msun",
            "maximum_posterior_Msun",
            "units",
        ],
        mass_summary_rows(samples, maximum),
    )
    original.write_csv(
        derived_path,
        ["derived_quantity", "maximum_posterior_value"],
        [
            ("area1_radius_cm", AREA1_RADIUS_CM),
            ("area2_front_radius_cm", AREA2_FRONT_RADIUS_CM),
            ("area2_back_temperature_STL_radius_cm", AREA2_BACK_TEMPERATURE_RADIUS_CM),
            ("area2_back_Smith_mass_radius_cm", AREA2_BACK_SMITH_MASS_RADIUS_CM),
            ("temperature_emissivity_beta", EMISSIVITY_BETA),
            ("temperature_distance_exponent", TEMPERATURE_DISTANCE_EXPONENT),
            ("area2_front_temperature_scale", AREA2_FRONT_TEMPERATURE_SCALE),
            ("area2_back_temperature_scale", AREA2_BACK_TEMPERATURE_SCALE),
            ("area2_front_K_radial_g_cm2_per_Msun", AREA2_FRONT_K_RADIAL_G_CM2_PER_MSUN),
            ("area2_back_K_radial_g_cm2_per_Msun", AREA2_BACK_K_RADIAL_G_CM2_PER_MSUN),
            ("area2_front_K_LOS_g_cm2_per_Msun", AREA2_FRONT_K_LOS_G_CM2_PER_MSUN),
            ("area2_back_K_LOS_g_cm2_per_Msun", AREA2_BACK_K_LOS_G_CM2_PER_MSUN),
            ("area2_front_LOS_to_radial", AREA2_FRONT_LOS_TO_RADIAL),
            ("area2_back_LOS_to_radial", AREA2_BACK_LOS_TO_RADIAL),
            ("area2_back_to_front_radial_column_ratio", AREA2_BACK_TO_FRONT_RADIAL_COLUMN_RATIO),
            ("area2_back_to_front_LOS_column_ratio", AREA2_BACK_TO_FRONT_LOS_COLUMN_RATIO),
            ("area2_front_LOS_fraction_of_total", AREA2_FRONT_LOS_FRACTION_OF_TOTAL),
            ("area2_back_LOS_fraction_of_total", AREA2_BACK_LOS_FRACTION_OF_TOTAL),
            ("area1_T_dense_K", best_state["T_d_1"]),
            ("area1_T_rare_K", best_state["T_r_1"]),
            ("area2_front_T_dense_K", best_state["T_d_2_front"]),
            ("area2_front_T_rare_K", best_state["T_r_2_front"]),
            ("area2_back_T_dense_K", best_state["T_d_2_back"]),
            ("area2_back_T_rare_K", best_state["T_r_2_back"]),
            ("area1_sigma_total_LOS_g_cm2", best_state["sigma_total_1"]),
            ("area2_sigma_dense_total_LOS_g_cm2", best_state["sigma_d_2_total_LOS"]),
            ("area2_sigma_dense_front_LOS_g_cm2", best_state["sigma_d_2_front"]),
            ("area2_sigma_dense_back_LOS_g_cm2", best_state["sigma_d_2_back"]),
            ("area2_sigma_dense_front_radial_g_cm2", best_state["sigma_d_2_front_radial"]),
            ("area2_sigma_dense_back_radial_g_cm2", best_state["sigma_d_2_back_radial"]),
            ("area2_sigma_rare_total_LOS_g_cm2", best_state["sigma_r_2_total_LOS"]),
            ("area2_sigma_rare_front_LOS_g_cm2", best_state["sigma_r_2_front"]),
            ("area2_sigma_rare_back_LOS_g_cm2", best_state["sigma_r_2_back"]),
            ("area2_sigma_rare_front_radial_g_cm2", best_state["sigma_r_2_front_radial"]),
            ("area2_sigma_rare_back_radial_g_cm2", best_state["sigma_r_2_back_radial"]),
            ("area2_sigma_effective_total_LOS_g_cm2", best_state["sigma_effective_2_total_LOS"]),
            ("area2_sigma_effective_front_LOS_g_cm2", best_state["sigma_effective_2_front"]),
            ("area2_sigma_effective_back_LOS_g_cm2", best_state["sigma_effective_2_back"]),
            ("area2_sigma_effective_front_radial_g_cm2", best_state["sigma_effective_2_front_radial"]),
            ("area2_sigma_effective_back_radial_g_cm2", best_state["sigma_effective_2_back_radial"]),
        ],
    )
    original.write_csv(
        spectrum_path,
        [
            "wavelength_micron",
            "observed_flux_cgs",
            "model_total_flux_cgs",
            "model_front_80deg_flux_cgs",
            "model_attenuated_back_28p4287deg_flux_cgs",
            "model_front_flux_fraction",
            "model_attenuated_back_flux_fraction",
        ],
        zip(
            area2_data[0],
            area2_data[1],
            best_flux_2,
            best_state["flux_2_front"],
            best_state["flux_2_back_seen"],
            best_state["front_flux_fraction"],
            best_state["back_flux_fraction"],
        ),
    )
    np.savez_compressed(
        posterior_path,
        samples=samples,
        log_probability=log_probabilities,
        parameter_names=np.asarray(PARAMETER_NAMES),
    )

    wavelength_fit = np.logspace(
        np.log10(original.WAVELENGTH_MIN_MICRON),
        np.log10(original.WAVELENGTH_MAX_MICRON),
        1000,
    )
    fit_1, fit_2, fit_state = evaluate_models(
        maximum, wavelength_fit, wavelength_fit
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    axes[0].errorbar(
        area1_data[0], area1_data[1], yerr=area1_data[2],
        fmt="o", color="black", label="Data", zorder=5,
    )
    axes[0].plot(wavelength_fit, fit_1, color="#9f2f1f", lw=2.0,
                 label="Maximum-posterior fit")
    axes[0].set_title("Area 1: 60-degree sideways slab")

    axes[1].errorbar(
        area2_data[0], area2_data[1], yerr=area2_data[2],
        fmt="o", color="black", label="Data", zorder=5,
    )
    axes[1].plot(wavelength_fit, fit_2, color="#9f2f1f", lw=2.2,
                 label="Total Area 2")
    axes[1].plot(
        wavelength_fit, fit_state["flux_2_front"], "--",
        color="#2f6f9f", lw=1.8, label="Front wall: 80 deg",
    )
    axes[1].plot(
        wavelength_fit, fit_state["flux_2_back_seen"], ":",
        color="#d17b20", lw=2.0, label="Attenuated back wall: 28.43 deg",
    )
    axes[1].set_title("Area 2: two LOS shell crossings")

    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(original.WAVELENGTH_MIN_MICRON,
                      original.WAVELENGTH_MAX_MICRON)
        axis.set_ylim(1.0e-26, 1.0e-19)
        axis.set_xlabel(r"Wavelength [$\mu$m]")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Flux [CGS]")
    figure.suptitle(
        "Joint MCMC: Area-2 temperatures bound to Area 1 by "
        rf"$R^{{-2/(4+\beta)}}$, $\beta={EMISSIVITY_BETA:g}$"
    )
    figure.tight_layout()

    print("\n--- Marginal posterior summaries: fitted parameters ---")
    for name, low, middle, high in zip(PARAMETER_NAMES, q16, median, q84):
        print(
            f"{name:52s}: {middle:.9g} "
            f"(-{middle-low:.3g}, +{high-middle:.3g})"
        )

    print("\n--- Maximum-posterior temperature audit ---")
    print(
        f"beta={EMISSIVITY_BETA:.6f}; temperature-distance exponent="
        f"{TEMPERATURE_DISTANCE_EXPONENT:.6f}"
    )
    print(
        f"Area 1 (60 deg) Td / Tr  = {best_state['T_d_1']:.6f} / "
        f"{best_state['T_r_1']:.6f} K"
    )
    print(
        f"Area 2 front (80 deg)    = {best_state['T_d_2_front']:.6f} / "
        f"{best_state['T_r_2_front']:.6f} K"
    )
    print(
        f"Area 2 back (28.43 deg)  = {best_state['T_d_2_back']:.6f} / "
        f"{best_state['T_r_2_back']:.6f} K"
    )
    print("The 80-degree wall is colder; the back wall is warmer.")

    print("\n--- Maximum-posterior Area 2 flux decomposition ---")
    print(" wavelength  front fraction  attenuated-back fraction")
    for wavelength, front_fraction, back_fraction in zip(
        area2_data[0],
        best_state["front_flux_fraction"],
        best_state["back_flux_fraction"],
    ):
        print(
            f" {wavelength:9.4f}  {front_fraction:14.6%}  "
            f"{back_fraction:24.6%}"
        )

    print("\n--- Maximum-posterior fit audit ---")
    print(
        "chi2 Area 1 / Area 2 / total = "
        f"{-2.0*ln_l_1:.6f} / {-2.0*ln_l_2:.6f} / "
        f"{-2.0*(ln_l_1+ln_l_2):.6f}"
    )
    print(f"mean acceptance fraction = {np.mean(sampler.acceptance_fraction):.6f}")
    if n_data <= len(PARAMETER_NAMES):
        print("Reduced chi2 is not reported: data count <= parameter count.")

    print_mass_summary(samples, maximum)
    print("\n--- Saved outputs ---")
    for path in (
        parameter_path,
        temperature_path,
        mass_path,
        derived_path,
        spectrum_path,
        posterior_path,
    ):
        print(path)
    print("The graph is displayed interactively and is not saved.")
    print("============================================================\n")

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Joint MCMC with Area-1 temperature anchors and a two-wall "
            "Area-2 LOS model."
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
            TWO_PHASE_RESULTS / "geometry_model"
        ),
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--mass-only",
        action="store_true",
        help="print masses from the saved posterior without rerunning MCMC",
    )
    args = parser.parse_args()
    minimum_walkers = 2 * len(PARAMETER_NAMES)
    if args.walkers < minimum_walkers:
        parser.error(f"walkers must be at least {minimum_walkers}.")
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
