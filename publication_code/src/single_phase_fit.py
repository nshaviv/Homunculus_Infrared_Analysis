#!/usr/bin/env python3
"""Joint two-region MCMC with a power-law grain distribution and fixed tau.

The two regions are fitted simultaneously.  They share one continuous grain-
size distribution,

    dN/da proportional to a**(-slope),  a_small <= a <= a_large,

and have independent temperatures.  The optical depths at the reference
wavelength are fixed: by default tau_1(18 um)=0.3 and tau_2(18 um)=0.1.
For every MCMC proposal the code derives an independent surface mass density
for each region from

    Sigma_i = tau_i(18 um) / kappa_eff(18 um),

then uses

    tau_i(lambda) = tau_i(18 um) * kappa_eff(lambda) / kappa_eff(18 um).

The column densities are region-specific but are derived, not independently
sampled, because fixing tau and kappa fixes Sigma.

The likelihood deliberately preserves the original code's observational error
model.  For every observed flux F it constructs

    sigma_F = sqrt((error_fraction * F)**2 + background_noise**2)
    sigma_log10F = sigma_F / (F * ln(10)),

and uses Gaussian residuals in log10 flux.  The defaults are the original
error_fraction=0.05 and background_noise=5e-23 cgs (=5 Jy).  Thus the additive
floor downweights faint points while the fractional term controls bright
points.  These values are analysis assumptions and must not be described as
Smith's published error prescription unless independently justified.

Unlike the original sequential code, both regions are evaluated at every MCMC
proposal and the joint probability is exactly

    ln L_total = ln L_region1 + ln L_region2.

The MCMC parameters are therefore

    [T_1, T_2, log10(a_small), log10(a_large), slope].

The reported radii are in microns.  The radius transformation makes the prior
uniform in log radius; ``a_small`` and ``a_large`` remain the physical lower
and upper cutoffs, not two discrete grain populations.

The likelihood currently treats the quoted errors as independent.  Because
the two regions were measured in the same filter images, common calibration
errors may correlate same-filter measurements across regions.  That covariance
is not available in the input data and is therefore not modeled here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import scipy.integrate

# PyMieScatt still imports the retired scipy.integrate.trapz name.
scipy.integrate.trapz = scipy.integrate.trapezoid

import emcee
import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import PyMieScatt as ps

from project_paths import MODEL_INPUT_DIR, SINGLE_PHASE_RESULTS


# =============================================================================
# PHYSICAL AND NUMERICAL CONSTANTS
# =============================================================================

H_CGS = 6.626e-27
C_CGS = 3.0e10
K_B_CGS = 1.38e-16

WAVELENGTH_MIN_MICRON = 2.0
WAVELENGTH_MAX_MICRON = 100.0
WAVELENGTH_GRID = np.logspace(
    np.log10(WAVELENGTH_MIN_MICRON),
    np.log10(WAVELENGTH_MAX_MICRON),
    1000,
)
RADIUS_GRID_MICRON = np.logspace(np.log10(0.1), np.log10(10.0), 50)

SILICATE_MASS_FRACTION = 0.990
CORUNDUM_MASS_FRACTION = 0.005
IRON_MASS_FRACTION = 0.005

SILICATE_DENSITY_G_CM3 = 3.2
CORUNDUM_DENSITY_G_CM3 = 3.9
IRON_DENSITY_G_CM3 = 7.9

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
ONE_ARCSEC2_SR = ARCSEC_TO_RAD**2

PARAMETER_NAMES = (
    "T_region1_K",
    "T_region2_K",
    "a_small_micron",
    "a_large_micron",
    "slope_q",
)


@dataclass(frozen=True)
class RegionData:
    name: str
    filename: str
    wavelength_micron: np.ndarray
    flux_cgs: np.ndarray
    error_cgs: np.ndarray


@dataclass(frozen=True)
class OpacityGrid:
    wavelength_micron: np.ndarray
    radius_micron: np.ndarray
    silicate: np.ndarray
    corundum: np.ndarray
    iron: np.ndarray

    @property
    def mixture(self) -> np.ndarray:
        return (
            SILICATE_MASS_FRACTION * self.silicate
            + CORUNDUM_MASS_FRACTION * self.corundum
            + IRON_MASS_FRACTION * self.iron
        )


@dataclass(frozen=True)
class JointFitResult:
    samples_physical: np.ndarray
    log_probabilities: np.ndarray
    chain_physical: np.ndarray
    q16: np.ndarray
    median: np.ndarray
    q84: np.ndarray
    maximum_posterior: np.ndarray
    derived_summary: dict[str, tuple[float, float, float]]
    wavelength_model: np.ndarray
    kappa_model: np.ndarray
    tau_region1: np.ndarray
    tau_region2: np.ndarray
    flux_model_region1: np.ndarray
    flux_model_region2: np.ndarray
    chi2_region1: float
    chi2_region2: float
    chi2_total: float
    reduced_chi2: float
    rms_log10_region1: float
    rms_log10_region2: float
    log_likelihood_region1: float
    log_likelihood_region2: float
    acceptance_fraction: float


# =============================================================================
# INPUT DATA AND MIE OPACITY GRID
# =============================================================================

def _sorted_optical_constants(
    wavelength: np.ndarray,
    refractive_n: np.ndarray,
    refractive_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(wavelength)
    return wavelength[order], refractive_n[order], refractive_k[order]


def load_optical_constants(
    data_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    silicate_wavelength: list[float] = []
    silicate_n: list[float] = []
    silicate_k: list[float] = []
    with (data_dir / "eps_Sil").open("r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                wavelength_value = float(parts[0])
                refractive_n_value = float(parts[3]) + 1.0
                refractive_k_value = float(parts[4])
            except ValueError:
                continue
            silicate_wavelength.append(wavelength_value)
            silicate_n.append(refractive_n_value)
            silicate_k.append(refractive_k_value)

    corundum_wavelength, corundum_n, corundum_k = np.loadtxt(
        data_dir / "corundumAl203data", unpack=True
    )
    iron_wavelength, iron_n, iron_k = np.loadtxt(
        data_dir / "iron_ordaldata", unpack=True
    )

    return {
        "silicate": _sorted_optical_constants(
            np.asarray(silicate_wavelength),
            np.asarray(silicate_n),
            np.asarray(silicate_k),
        ),
        "corundum": _sorted_optical_constants(
            corundum_wavelength, corundum_n, corundum_k
        ),
        "iron": _sorted_optical_constants(iron_wavelength, iron_n, iron_k),
    }


def kappa_from_nk(
    wavelength_micron: np.ndarray,
    refractive_n: np.ndarray,
    refractive_k: np.ndarray,
    material_density_g_cm3: float,
    radius_micron: float,
) -> np.ndarray:
    """Return monodisperse absorption opacity in cm^2/g."""
    radius_cm = radius_micron * 1.0e-4
    diameter_micron = 2.0 * radius_micron
    result = np.empty_like(wavelength_micron)
    for index, wavelength in enumerate(wavelength_micron):
        refractive_index = refractive_n[index] + 1j * refractive_k[index]
        _, _, q_abs, _, _, _, _ = ps.MieQ(
            refractive_index, wavelength, diameter_micron
        )
        result[index] = 3.0 * q_abs / (
            4.0 * radius_cm * material_density_g_cm3
        )
    return result


def opacity_signature(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for filename in ("eps_Sil", "corundumAl203data", "iron_ordaldata"):
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(filename.encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    digest.update(WAVELENGTH_GRID.tobytes())
    digest.update(RADIUS_GRID_MICRON.tobytes())
    digest.update(
        repr(
            (
                SILICATE_DENSITY_G_CM3,
                CORUNDUM_DENSITY_G_CM3,
                IRON_DENSITY_G_CM3,
            )
        ).encode("ascii")
    )
    return digest.hexdigest()


def _validate_opacity_grid(opacity: OpacityGrid) -> None:
    expected = (len(opacity.radius_micron), len(opacity.wavelength_micron))
    for name in ("silicate", "corundum", "iron"):
        values = getattr(opacity, name)
        if values.shape != expected:
            raise ValueError(f"Unexpected {name} opacity shape: {values.shape}")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"Invalid values in {name} opacity grid")


def _load_compatible_cache(path: Path) -> OpacityGrid:
    """Load this script's cache or the compatible existing composition cache."""
    with np.load(path) as cached:
        wavelength = (
            np.asarray(cached["wavelength_micron"])
            if "wavelength_micron" in cached
            else WAVELENGTH_GRID
        )
        radius = (
            np.asarray(cached["radius_micron"])
            if "radius_micron" in cached
            else RADIUS_GRID_MICRON
        )
        opacity = OpacityGrid(
            wavelength_micron=wavelength,
            radius_micron=radius,
            silicate=np.asarray(cached["kappa_silicate"]),
            corundum=np.asarray(cached["kappa_corundum"]),
            iron=np.asarray(cached["kappa_iron"]),
        )
    _validate_opacity_grid(opacity)
    if not np.array_equal(opacity.wavelength_micron, WAVELENGTH_GRID):
        raise ValueError(f"Incompatible wavelength grid in {path}")
    if not np.array_equal(opacity.radius_micron, RADIUS_GRID_MICRON):
        raise ValueError(f"Incompatible radius grid in {path}")
    return opacity


def build_or_load_opacity_grid(
    data_dir: Path,
    output_dir: Path,
    explicit_cache: Path | None,
    rebuild_cache: bool,
) -> OpacityGrid:
    if explicit_cache is not None:
        if not explicit_cache.exists():
            raise FileNotFoundError(explicit_cache)
        opacity = _load_compatible_cache(explicit_cache)
        print(f"Loaded explicit compatible opacity cache: {explicit_cache}")
        return opacity

    signature = opacity_signature(data_dir)
    cache_path = output_dir / "tau_dictated_powerlaw_opacity_cache.npz"
    if cache_path.exists() and not rebuild_cache:
        with np.load(cache_path) as cached:
            cached_signature = str(cached["signature"].item())
        if cached_signature == signature:
            opacity = _load_compatible_cache(cache_path)
            print(f"Loaded verified opacity cache: {cache_path}")
            return opacity

    optical_constants = load_optical_constants(data_dir)
    species = (
        ("silicate", SILICATE_DENSITY_G_CM3),
        ("corundum", CORUNDUM_DENSITY_G_CM3),
        ("iron", IRON_DENSITY_G_CM3),
    )
    grids: dict[str, np.ndarray] = {}
    print("Building silicate/corundum/iron Mie opacity grids...")
    for name, density in species:
        native_wavelength, native_n, native_k = optical_constants[name]
        n_on_grid = np.interp(WAVELENGTH_GRID, native_wavelength, native_n)
        k_on_grid = np.interp(WAVELENGTH_GRID, native_wavelength, native_k)
        grids[name] = np.vstack(
            [
                kappa_from_nk(
                    WAVELENGTH_GRID,
                    n_on_grid,
                    k_on_grid,
                    density,
                    radius,
                )
                for radius in RADIUS_GRID_MICRON
            ]
        )

    opacity = OpacityGrid(
        wavelength_micron=WAVELENGTH_GRID,
        radius_micron=RADIUS_GRID_MICRON,
        silicate=grids["silicate"],
        corundum=grids["corundum"],
        iron=grids["iron"],
    )
    _validate_opacity_grid(opacity)
    np.savez_compressed(
        cache_path,
        signature=np.asarray(signature),
        wavelength_micron=WAVELENGTH_GRID,
        radius_micron=RADIUS_GRID_MICRON,
        kappa_silicate=opacity.silicate,
        kappa_corundum=opacity.corundum,
        kappa_iron=opacity.iron,
    )
    print(f"Saved verified opacity cache: {cache_path}")
    return opacity


def load_region_data(
    data_dir: Path,
    filename: str,
    name: str,
    error_fraction: float,
    background_noise_cgs: float,
) -> RegionData:
    values = np.loadtxt(data_dir / filename, delimiter=",")
    wavelength = np.asarray(values[:, 0], dtype=float)
    flux = np.asarray(values[:, 1], dtype=float) * 1.0e-23
    keep = np.isfinite(wavelength) & np.isfinite(flux) & (flux > 0.0)
    wavelength = wavelength[keep]
    flux = flux[keep]
    error = np.sqrt(
        (error_fraction * flux) ** 2 + background_noise_cgs**2
    )
    return RegionData(name, filename, wavelength, flux, error)


# =============================================================================
# CONTINUOUS GRAIN-SIZE DISTRIBUTION AND RADIATIVE TRANSFER
# =============================================================================

def interpolate_wavelength_grid(
    opacity_by_radius: np.ndarray,
    wavelength_target: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [
            np.interp(wavelength_target, WAVELENGTH_GRID, radius_curve)
            for radius_curve in opacity_by_radius
        ]
    )


def interpolate_radius_grid(
    opacity_grid: np.ndarray,
    radius_target: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(RADIUS_GRID_MICRON, radius_target, side="right") - 1
    indices = np.clip(indices, 0, len(RADIUS_GRID_MICRON) - 2)
    radius_low = RADIUS_GRID_MICRON[indices]
    radius_high = RADIUS_GRID_MICRON[indices + 1]
    fraction = (radius_target - radius_low) / (radius_high - radius_low)
    return (
        opacity_grid[indices] * (1.0 - fraction[:, None])
        + opacity_grid[indices + 1] * fraction[:, None]
    )


def effective_kappa_powerlaw(
    a_small_micron: float,
    a_large_micron: float,
    slope: float,
    opacity_grid: np.ndarray,
    integration_points: int,
) -> np.ndarray:
    """Mass-weight kappa over dN/da proportional to a**(-slope)."""
    radius = np.logspace(
        np.log10(a_small_micron),
        np.log10(a_large_micron),
        integration_points,
    )
    kappa_at_radius = interpolate_radius_grid(opacity_grid, radius)
    mass_weight = radius ** (3.0 - slope)
    numerator = np.trapezoid(
        mass_weight[:, None] * kappa_at_radius,
        radius,
        axis=0,
    )
    denominator = np.trapezoid(mass_weight, radius)
    return numerator / denominator


def planck_nu(wavelength_micron: np.ndarray, temperature_k: float) -> np.ndarray:
    wavelength_cm = wavelength_micron * 1.0e-4
    exponent = H_CGS * C_CGS / (wavelength_cm * K_B_CGS * temperature_k)
    result = np.zeros_like(wavelength_micron)
    safe = exponent < 700.0
    result[safe] = (
        2.0 * H_CGS * C_CGS**2 / wavelength_cm[safe] ** 5
    ) / np.expm1(exponent[safe])
    return result * wavelength_cm**2 / C_CGS


def flux_from_dictated_tau(
    wavelength_micron: np.ndarray,
    temperature_k: float,
    kappa_at_wavelength: np.ndarray,
    kappa_at_reference: float,
    tau_at_reference: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    sigma_g_cm2 = tau_at_reference / kappa_at_reference
    tau = kappa_at_wavelength * sigma_g_cm2
    intensity = planck_nu(wavelength_micron, temperature_k) * (-np.expm1(-tau))
    return intensity * ONE_ARCSEC2_SR, tau, sigma_g_cm2


def distribution_quantile(
    a_small: np.ndarray | float,
    a_large: np.ndarray | float,
    slope: np.ndarray | float,
    quantile: float,
    mass_weighted: bool,
) -> np.ndarray:
    """Analytic quantile of dN/da or of its mass-weighted distribution."""
    a_small_array = np.asarray(a_small, dtype=float)
    a_large_array = np.asarray(a_large, dtype=float)
    slope_array = np.asarray(slope, dtype=float)
    exponent = (4.0 if mass_weighted else 1.0) - slope_array
    geometric = a_small_array * (a_large_array / a_small_array) ** quantile
    near_zero = np.isclose(exponent, 0.0, atol=1.0e-10)
    safe_exponent = np.where(near_zero, 1.0, exponent)
    regular = (
        (1.0 - quantile) * a_small_array**safe_exponent
        + quantile * a_large_array**safe_exponent
    ) ** (1.0 / safe_exponent)
    return np.where(near_zero, geometric, regular)


def transformed_to_physical(theta: np.ndarray) -> np.ndarray:
    result = np.asarray(theta, dtype=float).copy()
    result[..., 2] = 10.0 ** result[..., 2]
    result[..., 3] = 10.0 ** result[..., 3]
    return result


def physical_to_transformed(theta: np.ndarray) -> np.ndarray:
    result = np.asarray(theta, dtype=float).copy()
    result[..., 2] = np.log10(result[..., 2])
    result[..., 3] = np.log10(result[..., 3])
    return result


# =============================================================================
# JOINT MCMC
# =============================================================================

def percentile_triplet(values: np.ndarray) -> tuple[float, float, float]:
    q16, median, q84 = np.percentile(values, [16.0, 50.0, 84.0])
    return float(q16), float(median), float(q84)


def run_joint_mcmc(
    region1: RegionData,
    region2: RegionData,
    opacity: OpacityGrid,
    args: argparse.Namespace,
) -> JointFitResult:
    ndim = len(PARAMETER_NAMES)
    if args.walkers < 2 * ndim:
        raise ValueError(f"Need at least {2 * ndim} walkers for {ndim} dimensions")

    evaluation_wavelength = np.unique(
        np.concatenate(
            [
                region1.wavelength_micron,
                region2.wavelength_micron,
                np.asarray([args.lambda_reference]),
            ]
        )
    )
    mixture_evaluation_grid = interpolate_wavelength_grid(
        opacity.mixture, evaluation_wavelength
    )

    def log_prior(theta: np.ndarray) -> float:
        temperature1, temperature2, log_a_small, log_a_large, slope = theta
        if not (args.temperature_min <= temperature1 <= args.temperature_max):
            return -np.inf
        if not (args.temperature_min <= temperature2 <= args.temperature_max):
            return -np.inf
        if not (
            np.log10(args.radius_min) <= log_a_small < log_a_large
            <= np.log10(args.radius_max)
        ):
            return -np.inf
        if not (args.slope_min <= slope <= args.slope_max):
            return -np.inf
        return 0.0

    def evaluate(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        physical = transformed_to_physical(theta)
        temperature1, temperature2, a_small, a_large, slope = physical
        kappa_evaluation = effective_kappa_powerlaw(
            a_small,
            a_large,
            slope,
            mixture_evaluation_grid,
            args.size_integration_points,
        )
        kappa_reference = float(
            np.interp(args.lambda_reference, evaluation_wavelength, kappa_evaluation)
        )
        kappa_region1 = np.interp(
            region1.wavelength_micron, evaluation_wavelength, kappa_evaluation
        )
        kappa_region2 = np.interp(
            region2.wavelength_micron, evaluation_wavelength, kappa_evaluation
        )
        model1, _, sigma1 = flux_from_dictated_tau(
            region1.wavelength_micron,
            temperature1,
            kappa_region1,
            kappa_reference,
            args.tau_region1,
        )
        model2, _, sigma2 = flux_from_dictated_tau(
            region2.wavelength_micron,
            temperature2,
            kappa_region2,
            kappa_reference,
            args.tau_region2,
        )
        return model1, model2, kappa_reference, sigma1, sigma2

    def likelihood_components(
        theta: np.ndarray,
    ) -> tuple[float, float, float, float, float, float]:
        model1, model2, _, _, _ = evaluate(theta)
        if (
            not np.all(np.isfinite(model1))
            or not np.all(np.isfinite(model2))
            or np.any(model1 <= 0.0)
            or np.any(model2 <= 0.0)
        ):
            return np.inf, np.inf, -np.inf, -np.inf, np.inf, np.inf
        sigma_log1 = region1.error_cgs / (
            region1.flux_cgs * np.log(10.0)
        )
        sigma_log2 = region2.error_cgs / (
            region2.flux_cgs * np.log(10.0)
        )
        residual1 = np.log10(model1) - np.log10(region1.flux_cgs)
        residual2 = np.log10(model2) - np.log10(region2.flux_cgs)
        chi1 = float(np.sum((residual1 / sigma_log1) ** 2))
        chi2 = float(np.sum((residual2 / sigma_log2) ** 2))
        normalization1 = float(np.sum(np.log(2.0 * np.pi * sigma_log1**2)))
        normalization2 = float(np.sum(np.log(2.0 * np.pi * sigma_log2**2)))
        log_likelihood1 = float(-0.5 * (chi1 + normalization1))
        log_likelihood2 = float(-0.5 * (chi2 + normalization2))
        rms1 = float(np.sqrt(np.mean(residual1**2)))
        rms2 = float(np.sqrt(np.mean(residual2**2)))
        return chi1, chi2, log_likelihood1, log_likelihood2, rms1, rms2

    def log_probability(theta: np.ndarray) -> float:
        prior = log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        _, _, log_likelihood1, log_likelihood2, _, _ = likelihood_components(theta)
        if not np.isfinite(log_likelihood1 + log_likelihood2):
            return -np.inf
        return prior + log_likelihood1 + log_likelihood2

    initial_physical = np.asarray(
        [
            args.initial_temperature_region1,
            args.initial_temperature_region2,
            args.initial_a_small,
            args.initial_a_large,
            args.initial_slope,
        ],
        dtype=float,
    )
    initial = physical_to_transformed(initial_physical)
    scatter = np.asarray([5.0, 5.0, 0.035, 0.060, 0.10])
    generator = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    positions: list[np.ndarray] = []
    attempts = 0
    while len(positions) < args.walkers:
        trial = initial + scatter * generator.standard_normal(ndim)
        attempts += 1
        if np.isfinite(log_prior(trial)):
            positions.append(trial)
        if attempts > 100000:
            raise RuntimeError("Could not initialize walkers inside the prior")

    sampler = emcee.EnsembleSampler(args.walkers, ndim, log_probability)
    print(
        "Running joint dictated-tau MCMC: "
        f"{args.walkers} walkers x {args.steps} steps..."
    )
    sampler.run_mcmc(
        np.vstack(positions),
        args.steps,
        progress=not args.no_progress,
    )

    chain_transformed = sampler.get_chain()
    samples_transformed = sampler.get_chain(
        discard=args.burn,
        thin=args.thin,
        flat=True,
    )
    log_probabilities = sampler.get_log_prob(
        discard=args.burn,
        thin=args.thin,
        flat=True,
    )
    if len(samples_transformed) == 0:
        raise RuntimeError("No samples remain after burn-in and thinning")

    samples_physical = transformed_to_physical(samples_transformed)
    chain_physical = transformed_to_physical(chain_transformed)
    q16, median, q84 = np.percentile(
        samples_physical,
        [16.0, 50.0, 84.0],
        axis=0,
    )
    maximum_posterior = samples_physical[int(np.argmax(log_probabilities))]

    reference_grid = interpolate_wavelength_grid(
        opacity.mixture, np.asarray([args.lambda_reference])
    )
    kappa_reference_samples = np.empty(len(samples_physical))
    for index, sample in enumerate(samples_physical):
        _, _, a_small, a_large, slope = sample
        kappa_reference_samples[index] = effective_kappa_powerlaw(
            a_small,
            a_large,
            slope,
            reference_grid,
            args.size_integration_points,
        )[0]

    number_median_samples = distribution_quantile(
        samples_physical[:, 2],
        samples_physical[:, 3],
        samples_physical[:, 4],
        0.5,
        mass_weighted=False,
    )
    mass_median_samples = distribution_quantile(
        samples_physical[:, 2],
        samples_physical[:, 3],
        samples_physical[:, 4],
        0.5,
        mass_weighted=True,
    )
    sigma1_samples = args.tau_region1 / kappa_reference_samples
    sigma2_samples = args.tau_region2 / kappa_reference_samples
    derived_summary = {
        "a50_number_micron": percentile_triplet(number_median_samples),
        "a50_mass_micron": percentile_triplet(mass_median_samples),
        "kappa_reference_cm2_g": percentile_triplet(kappa_reference_samples),
        "sigma_region1_g_cm2": percentile_triplet(sigma1_samples),
        "sigma_region2_g_cm2": percentile_triplet(sigma2_samples),
    }

    median_transformed = physical_to_transformed(median)
    maximum_transformed = physical_to_transformed(maximum_posterior)
    (
        chi1,
        chi2,
        log_likelihood1,
        log_likelihood2,
        rms_log1,
        rms_log2,
    ) = likelihood_components(maximum_transformed)
    degrees_of_freedom = (
        len(region1.wavelength_micron) + len(region2.wavelength_micron) - ndim
    )
    reduced_chi2 = (chi1 + chi2) / degrees_of_freedom

    _, _, median_a_small, median_a_large, median_slope = median
    kappa_model = effective_kappa_powerlaw(
        median_a_small,
        median_a_large,
        median_slope,
        opacity.mixture,
        args.size_integration_points,
    )
    kappa_reference = float(
        np.interp(args.lambda_reference, WAVELENGTH_GRID, kappa_model)
    )
    flux_model1, tau1, sigma1 = flux_from_dictated_tau(
        WAVELENGTH_GRID,
        median[0],
        kappa_model,
        kappa_reference,
        args.tau_region1,
    )
    flux_model2, tau2, sigma2 = flux_from_dictated_tau(
        WAVELENGTH_GRID,
        median[1],
        kappa_model,
        kappa_reference,
        args.tau_region2,
    )

    tau1_at_reference = float(
        np.interp(args.lambda_reference, WAVELENGTH_GRID, tau1)
    )
    tau2_at_reference = float(
        np.interp(args.lambda_reference, WAVELENGTH_GRID, tau2)
    )
    if not np.isclose(tau1_at_reference, args.tau_region1, rtol=1.0e-12):
        raise AssertionError("Region-1 dictated tau was not preserved")
    if not np.isclose(tau2_at_reference, args.tau_region2, rtol=1.0e-12):
        raise AssertionError("Region-2 dictated tau was not preserved")
    if not np.isclose(sigma1, args.tau_region1 / kappa_reference):
        raise AssertionError("Region-1 surface density normalization failed")
    if not np.isclose(sigma2, args.tau_region2 / kappa_reference):
        raise AssertionError("Region-2 surface density normalization failed")

    return JointFitResult(
        samples_physical=samples_physical,
        log_probabilities=log_probabilities,
        chain_physical=chain_physical,
        q16=q16,
        median=median,
        q84=q84,
        maximum_posterior=maximum_posterior,
        derived_summary=derived_summary,
        wavelength_model=WAVELENGTH_GRID,
        kappa_model=kappa_model,
        tau_region1=tau1,
        tau_region2=tau2,
        flux_model_region1=flux_model1,
        flux_model_region2=flux_model2,
        chi2_region1=chi1,
        chi2_region2=chi2,
        chi2_total=chi1 + chi2,
        reduced_chi2=reduced_chi2,
        rms_log10_region1=rms_log1,
        rms_log10_region2=rms_log2,
        log_likelihood_region1=log_likelihood1,
        log_likelihood_region2=log_likelihood2,
        acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
    )


# =============================================================================
# OUTPUTS
# =============================================================================

def summary_payload(
    result: JointFitResult,
    args: argparse.Namespace,
    region1: RegionData,
    region2: RegionData,
) -> dict[str, Any]:
    parameters: dict[str, dict[str, float]] = {}
    for name, q16, median, q84, maximum in zip(
        PARAMETER_NAMES,
        result.q16,
        result.median,
        result.q84,
        result.maximum_posterior,
    ):
        parameters[name] = {
            "q16": float(q16),
            "median": float(median),
            "q84": float(q84),
            "maximum_posterior": float(maximum),
        }

    derived = {
        name: {"q16": q16, "median": median, "q84": q84}
        for name, (q16, median, q84) in result.derived_summary.items()
    }
    return {
        "model": {
            "size_distribution": "dN/da proportional to a^(-slope_q)",
            "shared_between_regions": True,
            "tau_is_dictated": True,
            "likelihood": (
                "Gaussian in log10 flux with the original propagated errors: "
                "sigma_F=sqrt((error_fraction*F)^2+background_noise_cgs^2), "
                "sigma_log10F=sigma_F/(F*ln(10)); lnL=lnL1+lnL2"
            ),
            "error_fraction": args.error_fraction,
            "background_noise_cgs": args.background_noise_cgs,
            "background_noise_jy": args.background_noise_cgs / 1.0e-23,
            "uncertainty_note": (
                "Preserved from the original code; an analysis assumption, "
                "not claimed to be Smith's original error treatment"
            ),
            "cross_region_calibration_covariance_included": False,
            "lambda_reference_micron": args.lambda_reference,
            "tau_region1": args.tau_region1,
            "tau_region2": args.tau_region2,
            "composition_mass_fractions": {
                "silicate": SILICATE_MASS_FRACTION,
                "corundum": CORUNDUM_MASS_FRACTION,
                "iron": IRON_MASS_FRACTION,
            },
        },
        "data": {
            "region1": region1.filename,
            "region2": region2.filename,
            "n_region1": len(region1.wavelength_micron),
            "n_region2": len(region2.wavelength_micron),
        },
        "parameters": parameters,
        "derived": derived,
        "fit_at_maximum_posterior": {
            "chi2_region1": result.chi2_region1,
            "chi2_region2": result.chi2_region2,
            "chi2_total": result.chi2_total,
            "degrees_of_freedom": (
                len(region1.wavelength_micron)
                + len(region2.wavelength_micron)
                - len(PARAMETER_NAMES)
            ),
            "reduced_chi2": result.reduced_chi2,
            "rms_log10_region1_dex": result.rms_log10_region1,
            "rms_log10_region2_dex": result.rms_log10_region2,
            "log_likelihood_region1": result.log_likelihood_region1,
            "log_likelihood_region2": result.log_likelihood_region2,
        },
        "chain_diagnostics": {
            "walkers": args.walkers,
            "steps": args.steps,
            "burn": args.burn,
            "thin": args.thin,
            "mean_acceptance_fraction": result.acceptance_fraction,
        },
    }


def save_summary_files(
    result: JointFitResult,
    args: argparse.Namespace,
    region1: RegionData,
    region2: RegionData,
) -> None:
    payload = summary_payload(result, args, region1, region2)
    json_path = args.output_dir / "tau_dictated_18um_joint_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "tau_dictated_18um_joint_parameters.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", "q16", "median", "q84", "maximum_posterior"])
        for name, q16, median, q84, maximum in zip(
            PARAMETER_NAMES,
            result.q16,
            result.median,
            result.q84,
            result.maximum_posterior,
        ):
            writer.writerow([name, q16, median, q84, maximum])
        for name, (q16, median, q84) in result.derived_summary.items():
            writer.writerow([name, q16, median, q84, ""])

    np.savez_compressed(
        args.output_dir / "tau_dictated_18um_posterior_samples.npz",
        parameter_names=np.asarray(PARAMETER_NAMES),
        samples=result.samples_physical,
        log_probability=result.log_probabilities,
    )

    kappa_tau_path = args.output_dir / "tau_dictated_18um_kappa_and_tau.csv"
    np.savetxt(
        kappa_tau_path,
        np.column_stack(
            [
                result.wavelength_model,
                result.kappa_model,
                result.tau_region1,
                result.tau_region2,
            ]
        ),
        delimiter=",",
        header=(
            "wavelength_micron,kappa_eff_cm2_g,"
            "tau_region1,tau_region2"
        ),
        comments="",
    )

    a_small = result.median[2]
    a_large = result.median[3]
    slope = result.median[4]
    radius = np.logspace(np.log10(a_small), np.log10(a_large), 400)
    number_per_log_radius = radius ** (1.0 - slope)
    mass_per_log_radius = radius ** (4.0 - slope)
    number_per_log_radius /= np.trapezoid(number_per_log_radius, np.log(radius))
    mass_per_log_radius /= np.trapezoid(mass_per_log_radius, np.log(radius))
    np.savetxt(
        args.output_dir / "tau_dictated_18um_grain_distribution.csv",
        np.column_stack([radius, number_per_log_radius, mass_per_log_radius]),
        delimiter=",",
        header=(
            "grain_radius_micron,normalized_dN_dln_a,"
            "normalized_dM_dln_a"
        ),
        comments="",
    )
    print(f"Saved summary: {json_path}")
    print(f"Saved parameters: {csv_path}")
    print(f"Saved opacity/tau curve: {kappa_tau_path}")


def print_summary(
    result: JointFitResult,
    args: argparse.Namespace,
    region1: RegionData,
    region2: RegionData,
) -> None:
    print("\n====== JOINT POWER-LAW FIT; FIXED TAU; ORIGINAL PROPAGATED ERRORS ======")
    for name, q16, median, q84, maximum in zip(
        PARAMETER_NAMES,
        result.q16,
        result.median,
        result.q84,
        result.maximum_posterior,
    ):
        print(
            f"{name:22s}: {median:.6g} "
            f"(-{median - q16:.3g}, +{q84 - median:.3g}); "
            f"MAP={maximum:.6g}"
        )
    kappa_q16, kappa_median, kappa_q84 = result.derived_summary[
        "kappa_reference_cm2_g"
    ]
    print(
        f"kappa_{args.lambda_reference:g}um_cm2_g     : {kappa_median:.6g} "
        f"(-{kappa_median - kappa_q16:.3g}, "
        f"+{kappa_q84 - kappa_median:.3g})"
    )
    print("Derived quantities:")
    for name, (q16, median, q84) in result.derived_summary.items():
        print(
            f"{name:26s}: {median:.6g} "
            f"(-{median - q16:.3g}, +{q84 - median:.3g})"
        )
    print(
        f"Dictated anchors: tau1({args.lambda_reference:g} um)="
        f"{args.tau_region1:g}, tau2({args.lambda_reference:g} um)="
        f"{args.tau_region2:g}"
    )
    print(
        "Original error model used by likelihood: "
        f"fractional={100.0 * args.error_fraction:.3g}%, "
        f"background floor={args.background_noise_cgs / 1.0e-23:.6g} Jy; "
        "sigma_log10=sigma_F/(F ln 10)"
    )
    print(
        f"chi2: region1={result.chi2_region1:.5g}, "
        f"region2={result.chi2_region2:.5g}, "
        f"total={result.chi2_total:.5g}"
    )
    degrees_of_freedom = (
        len(region1.wavelength_micron)
        + len(region2.wavelength_micron)
        - len(PARAMETER_NAMES)
    )
    print(
        f"data/parameters/dof: "
        f"{len(region1.wavelength_micron) + len(region2.wavelength_micron)}/"
        f"{len(PARAMETER_NAMES)}/{degrees_of_freedom}"
    )
    print(f"reduced chi2: {result.reduced_chi2:.5g}")
    print(
        "RMS log10 residuals [dex]: "
        f"region1={result.rms_log10_region1:.5g}, "
        f"region2={result.rms_log10_region2:.5g}"
    )
    print(
        "log likelihoods: "
        f"region1={result.log_likelihood_region1:.5g}, "
        f"region2={result.log_likelihood_region2:.5g}, "
        f"total={result.log_likelihood_region1 + result.log_likelihood_region2:.5g}"
    )
    print(f"mean acceptance fraction: {result.acceptance_fraction:.4f}")


def make_plots(
    result: JointFitResult,
    args: argparse.Namespace,
    region1: RegionData,
    region2: RegionData,
) -> None:
    # Primary comparison figure: both regions and their joint models are drawn
    # on one axis.  The gray curves are physical, optically thick blackbodies
    # subtending one square arcsecond, consistent with the model's solid angle.
    comparison_figure, comparison_axis = plt.subplots(figsize=(11, 7.5))
    blackbody_temperatures = np.arange(100.0, 321.0, 20.0)
    label_wavelength = 29.0
    for temperature in blackbody_temperatures:
        blackbody_flux = (
            planck_nu(result.wavelength_model, temperature)
            * ONE_ARCSEC2_SR
        )
        comparison_axis.plot(
            result.wavelength_model,
            blackbody_flux,
            color="0.65",
            linewidth=0.8,
            alpha=0.32,
            zorder=0,
        )
        label_flux = float(
            np.interp(
                label_wavelength,
                result.wavelength_model,
                blackbody_flux,
            )
        )
        comparison_axis.text(
            label_wavelength * 1.012,
            label_flux,
            f"{temperature:.0f} K",
            color="0.48",
            fontsize=6.5,
            alpha=0.85,
            va="center",
            zorder=1,
        )

    comparison_axis.plot(
        [],
        [],
        color="0.65",
        linewidth=0.9,
        alpha=0.45,
        label=r"Optically thick blackbodies, $1\,\mathrm{arcsec}^2$",
    )
    for region, model_flux, color, marker in (
        (region1, result.flux_model_region1, "#9b2226", "o"),
        (region2, result.flux_model_region2, "#005f73", "s"),
    ):
        comparison_axis.errorbar(
            region.wavelength_micron,
            region.flux_cgs,
            yerr=region.error_cgs,
            fmt=marker,
            markersize=6,
            capsize=2.5,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            label=f"{region.name} data",
            zorder=5,
        )
        comparison_axis.plot(
            result.wavelength_model,
            model_flux,
            color=color,
            linewidth=2.3,
            label=f"{region.name} joint model",
            zorder=4,
        )

    kappa_median = result.derived_summary["kappa_reference_cm2_g"][1]
    result_text = (
        rf"$T_1={result.median[0]:.1f}\,\mathrm{{K}}$" "\n"
        rf"$T_2={result.median[1]:.1f}\,\mathrm{{K}}$" "\n"
        rf"$a_{{\min}}={result.median[2]:.3g}\,\mu\mathrm{{m}}$" "\n"
        rf"$a_{{\max}}={result.median[3]:.3g}\,\mu\mathrm{{m}}$" "\n"
        rf"$q={result.median[4]:.3f}$" "\n"
        rf"$\kappa_{{18}}={kappa_median:.1f}\,\mathrm{{cm^2\,g^{{-1}}}}$"
    )
    comparison_axis.text(
        0.025,
        0.97,
        result_text,
        transform=comparison_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.86,
        },
        zorder=10,
    )
    comparison_axis.set_xscale("log")
    comparison_axis.set_yscale("log")
    comparison_axis.set_xlim(4.0, 32.0)
    comparison_axis.set_ylim(1.0e-25, 3.0e-19)
    comparison_axis.set_xlabel(r"Wavelength [$\mu$m]")
    comparison_axis.set_ylabel(
        r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ per arcsec$^2$]"
    )
    comparison_axis.set_title(
        "Simultaneous two-region fit with blackbody comparison curves"
    )
    comparison_axis.grid(alpha=0.18, which="both")
    comparison_axis.legend(loc="lower right", fontsize=8, framealpha=0.90)
    comparison_figure.tight_layout()
    comparison_path = (
        args.output_dir / "tau_dictated_18um_combined_sed.png"
    )
    comparison_figure.savefig(
        comparison_path,
        dpi=240,
        bbox_inches="tight",
    )
    print(f"Saved combined SED plot: {comparison_path}")

    # Additional diagnostic sheet retained from the earlier version.
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))

    for axis, region, model_flux, color in (
        (axes[0, 0], region1, result.flux_model_region1, "#9b2226"),
        (axes[0, 1], region2, result.flux_model_region2, "#005f73"),
    ):
        axis.errorbar(
            region.wavelength_micron,
            region.flux_cgs,
            yerr=region.error_cgs,
            fmt="o",
            color="black",
            label="Data",
            zorder=5,
        )
        axis.plot(
            result.wavelength_model,
            model_flux,
            color=color,
            linewidth=2,
            label="Joint posterior-median model",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(WAVELENGTH_MIN_MICRON, WAVELENGTH_MAX_MICRON)
        axis.set_ylim(1.0e-26, 1.0e-19)
        axis.set_xlabel(r"Wavelength [$\mu$m]")
        axis.set_ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
        axis.set_title(f"{region.name}: {region.filename}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    a_small = result.median[2]
    a_large = result.median[3]
    slope = result.median[4]
    radius = np.logspace(np.log10(a_small), np.log10(a_large), 400)
    number_per_log = radius ** (1.0 - slope)
    mass_per_log = radius ** (4.0 - slope)
    number_per_log /= np.max(number_per_log)
    mass_per_log /= np.max(mass_per_log)
    axes[0, 2].plot(radius, number_per_log, label=r"$dN/d\ln a$", linewidth=2)
    axes[0, 2].plot(radius, mass_per_log, label=r"$dM/d\ln a$", linewidth=2)
    axes[0, 2].axvline(
        result.derived_summary["a50_number_micron"][1],
        color="C0",
        linestyle="--",
        alpha=0.7,
        label=r"number $a_{50}$",
    )
    axes[0, 2].axvline(
        result.derived_summary["a50_mass_micron"][1],
        color="C1",
        linestyle="--",
        alpha=0.7,
        label=r"mass $a_{50}$",
    )
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlabel(r"Grain radius [$\mu$m]")
    axes[0, 2].set_ylabel("Relative density per log radius")
    axes[0, 2].set_title(rf"Shared $dN/da\propto a^{{-{slope:.2f}}}$")
    axes[0, 2].grid(alpha=0.25)
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(
        result.wavelength_model,
        result.kappa_model,
        color="black",
        linewidth=2,
    )
    axes[1, 0].axvline(args.lambda_reference, color="0.5", linestyle="--")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel(r"Wavelength [$\mu$m]")
    axes[1, 0].set_ylabel(r"$\kappa_{\rm eff}$ [cm$^2$ g$^{-1}$]")
    axes[1, 0].set_title("Size-averaged mixture opacity")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(
        result.wavelength_model,
        result.tau_region1,
        color="#9b2226",
        label=rf"Region 1: $\tau_{{18}}={args.tau_region1:g}$",
    )
    axes[1, 1].plot(
        result.wavelength_model,
        result.tau_region2,
        color="#005f73",
        label=rf"Region 2: $\tau_{{18}}={args.tau_region2:g}$",
    )
    axes[1, 1].axvline(args.lambda_reference, color="0.5", linestyle="--")
    axes[1, 1].axhline(1.0, color="0.3", linestyle=":")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel(r"Wavelength [$\mu$m]")
    axes[1, 1].set_ylabel(r"Optical depth $\tau$")
    axes[1, 1].set_title("Tau curves normalized at the dictated anchor")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    residual_axis = axes[1, 2]
    for region, model_flux, color, marker in (
        (region1, result.flux_model_region1, "#9b2226", "o"),
        (region2, result.flux_model_region2, "#005f73", "s"),
    ):
        model_at_data = np.interp(
            region.wavelength_micron,
            result.wavelength_model,
            model_flux,
        )
        residual = (
            np.log10(model_at_data) - np.log10(region.flux_cgs)
        ) / (region.error_cgs / (region.flux_cgs * np.log(10.0)))
        residual_axis.plot(
            region.wavelength_micron,
            residual,
            marker=marker,
            linestyle="none",
            color=color,
            label=region.name,
        )
    residual_axis.axhline(0.0, color="black", linewidth=1)
    residual_axis.axhline(2.0, color="0.5", linestyle=":")
    residual_axis.axhline(-2.0, color="0.5", linestyle=":")
    residual_axis.set_xscale("log")
    residual_axis.set_xlabel(r"Wavelength [$\mu$m]")
    residual_axis.set_ylabel("Log-flux residual / sigma")
    residual_axis.set_title("Posterior-median residuals")
    residual_axis.grid(alpha=0.25)
    residual_axis.legend(fontsize=8)

    figure.suptitle(
        "Joint power-law grain MCMC; fixed tau and original propagated errors"
    )
    figure.tight_layout()
    plot_path = args.output_dir / "tau_dictated_18um_joint_fit.png"
    figure.savefig(plot_path, dpi=220, bbox_inches="tight")
    print(f"Saved fit plot: {plot_path}")

    trace_figure, trace_axes = plt.subplots(
        len(PARAMETER_NAMES),
        1,
        figsize=(11, 10),
        sharex=True,
    )
    for parameter_index, (axis, name) in enumerate(
        zip(trace_axes, PARAMETER_NAMES)
    ):
        axis.plot(
            result.chain_physical[:, :, parameter_index],
            color="black",
            alpha=0.12,
            linewidth=0.5,
        )
        axis.axvline(args.burn, color="C3", linestyle="--", linewidth=1)
        axis.set_ylabel(name)
    trace_axes[-1].set_xlabel("MCMC step")
    trace_figure.suptitle("Joint dictated-tau MCMC traces")
    trace_figure.tight_layout()
    trace_path = args.output_dir / "tau_dictated_18um_trace.png"
    trace_figure.savefig(trace_path, dpi=180, bbox_inches="tight")
    print(f"Saved trace plot: {trace_path}")


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=MODEL_INPUT_DIR,
        help="Directory containing the two region files and optical constants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            SINGLE_PHASE_RESULTS / "rerun"
        ),
    )
    parser.add_argument(
        "--opacity-cache",
        type=Path,
        default=None,
        help="Optional compatible precomputed opacity cache.",
    )
    parser.add_argument("--rebuild-opacity-cache", action="store_true")
    parser.add_argument("--tau-region1", type=float, default=0.3)
    parser.add_argument("--tau-region2", type=float, default=0.1)
    parser.add_argument("--lambda-reference", type=float, default=18.0)
    parser.add_argument("--temperature-min", type=float, default=50.0)
    parser.add_argument("--temperature-max", type=float, default=800.0)
    parser.add_argument("--radius-min", type=float, default=0.1)
    parser.add_argument("--radius-max", type=float, default=10.0)
    parser.add_argument("--slope-min", type=float, default=2.0)
    parser.add_argument("--slope-max", type=float, default=5.0)
    parser.add_argument("--initial-temperature-region1", type=float, default=140.0)
    parser.add_argument("--initial-temperature-region2", type=float, default=250.0)
    parser.add_argument("--initial-a-small", type=float, default=0.2)
    parser.add_argument("--initial-a-large", type=float, default=4.0)
    parser.add_argument("--initial-slope", type=float, default=3.5)
    parser.add_argument(
        "--error-fraction",
        type=float,
        default=0.05,
        help="Original fractional flux-error term (default: 0.05).",
    )
    parser.add_argument(
        "--background-noise-cgs",
        type=float,
        default=5.0e-23,
        help="Original additive flux-error floor in cgs (default: 5e-23 = 5 Jy).",
    )
    parser.add_argument("--size-integration-points", type=int, default=64)
    parser.add_argument("--walkers", type=int, default=50)
    parser.add_argument(
        "--steps",
        type=int,
        default=25000,
        help="Fixed number of MCMC steps (default: 25000).",
    )
    parser.add_argument("--burn", type=int, default=5000)
    parser.add_argument("--thin", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.steps <= 1:
        raise ValueError("--steps must exceed 1")
    if not (0 <= args.burn < args.steps):
        raise ValueError("Require 0 <= --burn < --steps")
    if args.thin < 1:
        raise ValueError("--thin must be positive")
    if args.size_integration_points < 16:
        raise ValueError("Use at least 16 size-integration points")
    if args.tau_region1 <= 0.0 or args.tau_region2 <= 0.0:
        raise ValueError("Dictated optical depths must be positive")
    if not (
        WAVELENGTH_MIN_MICRON
        <= args.lambda_reference
        <= WAVELENGTH_MAX_MICRON
    ):
        raise ValueError("Reference wavelength lies outside the opacity grid")
    if not (0.1 <= args.radius_min < args.radius_max <= 10.0):
        raise ValueError("Radius bounds must lie inside [0.1, 10] microns")
    if args.temperature_min >= args.temperature_max:
        raise ValueError("Invalid temperature bounds")
    if args.slope_min >= args.slope_max:
        raise ValueError("Invalid slope bounds")
    if not (0.0 < args.error_fraction < 1.0):
        raise ValueError("--error-fraction must lie between 0 and 1")
    if args.background_noise_cgs < 0.0:
        raise ValueError("--background-noise-cgs must be non-negative")
    initial = np.asarray(
        [
            args.initial_temperature_region1,
            args.initial_temperature_region2,
            args.initial_a_small,
            args.initial_a_large,
            args.initial_slope,
        ]
    )
    if not (
        args.temperature_min <= initial[0] <= args.temperature_max
        and args.temperature_min <= initial[1] <= args.temperature_max
        and args.radius_min <= initial[2] < initial[3] <= args.radius_max
        and args.slope_min <= initial[4] <= args.slope_max
    ):
        raise ValueError("Initial values must lie inside the priors")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    opacity = build_or_load_opacity_grid(
        args.data_dir,
        args.output_dir,
        args.opacity_cache,
        args.rebuild_opacity_cache,
    )
    region1 = load_region_data(
        args.data_dir,
        "lobe.edge.data",
        "Region 1",
        args.error_fraction,
        args.background_noise_cgs,
    )
    region2 = load_region_data(
        args.data_dir,
        "lobeedgedata2",
        "Region 2",
        args.error_fraction,
        args.background_noise_cgs,
    )

    result = run_joint_mcmc(region1, region2, opacity, args)
    print_summary(result, args, region1, region2)
    save_summary_files(result, args, region1, region2)
    make_plots(result, args, region1, region2)
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
