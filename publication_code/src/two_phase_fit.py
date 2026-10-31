"""Independent-overlap variant of the two-wall density-dependent-grain MCMC.

This file changes one geometrical assumption in ``two_phase_aligned.py`` and
leaves its parameter
vector, Mie opacity construction, temperatures, LOS/radial conversions,
grain-growth relations, likelihood, and mass calculations unchanged.

The fixed number f=0.444 is interpreted as a PROJECTED AREA covering
fraction.  It is assumed to have the same value at the 80-degree foreground
wall and the 28.4287-degree rear wall because no independent measurement of
the rear covering fraction is available.

The foreground and rear clump patterns are NOT assumed to be spatially
aligned.  They are treated as statistically independent projected masks.
Consequently the four overlap probabilities are

    P(dense front, dense rear) = f*f
    P(dense front, rare  rear) = f*(1-f)
    P(rare  front, dense rear) = (1-f)*f
    P(rare  front, rare  rear) = (1-f)*(1-f).

For phase i in the foreground and phase j in the rear, one ray has

    I_ij = B(T_i,f)*(1-exp(-tau_i,f))
           + exp(-tau_i,f)*B(T_j,b)*(1-exp(-tau_j,b)).

Averaging those four rays is equivalently

    E_front = f*E_dense,front + (1-f)*E_rare,front
    T_front = f*exp(-tau_dense,front) + (1-f)*exp(-tau_rare,front)
    E_back  = f*E_dense,back + (1-f)*E_rare,back
    I_total = E_front + T_front*E_back.

This includes the exact blackbody limit within each covered fraction.  It
differs from the preceding file, which implicitly set P_dr=P_rd=0 and thus
paired dense foreground only with dense rear and rare foreground only with
rare rear.

Important remaining approximation
---------------------------------
The Smith radial ratio still scales the dense and rare rear columns from the
corresponding foreground columns.  Together with equal f this assumes the
dense/rare column contrast is the same at the two Area-2 latitudes.  Relaxing
that would require another independently constrained rear phase-contrast
parameter.  This assumption is printed explicitly at runtime.

No autocorrelation-time or 50-tau rule is used.  The supporting modules are
distributed beside this file in the publication package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import two_phase_aligned as aligned

from project_paths import TWO_PHASE_RESULTS


# Save the tested preceding implementation before replacing the function in
# its module namespace.  It still supplies Area 1 and all common derivations.
_ALIGNED_EVALUATE_MODELS = aligned.evaluate_models
_ORIGINAL_PRINT_MASS_SUMMARY = aligned.print_mass_summary


def evaluate_models(theta, wavelength_area1, wavelength_area2):
    """Evaluate Area 1 unchanged and Area 2 with independent wall overlap."""
    flux_1, _, state, kappas = _ALIGNED_EVALUATE_MODELS(
        theta, wavelength_area1, wavelength_area2
    )

    f_cover = aligned.base.AREA2_FILLING_FRACTION
    one_minus_f = 1.0 - f_cover

    # The preceding evaluator has already calculated the correct phase
    # optical depths, temperatures, and foreground emission.  Reconstruct
    # the unattenuated rear emission and average foreground transmission.
    dense_back_intrinsic = aligned.base.planck_nu_from_wavelength(
        wavelength_area2, state["T_dense_2_back"]
    ) * -np.expm1(-state["tau_dense_back"])
    rare_back_intrinsic = aligned.base.planck_nu_from_wavelength(
        wavelength_area2, state["T_rare_2_back"]
    ) * -np.expm1(-state["tau_rare_back"])

    mean_front_transmission = (
        f_cover * np.exp(-state["tau_dense_front"])
        + one_minus_f * np.exp(-state["tau_rare_front"])
    )
    intrinsic_back_intensity = (
        f_cover * dense_back_intrinsic
        + one_minus_f * rare_back_intrinsic
    )
    independent_back_seen_flux = (
        mean_front_transmission
        * intrinsic_back_intensity
        * aligned.base.PIXEL_SOLID_ANGLE_SR
    )

    # front_flux is identical for aligned and independent overlap.  Only the
    # foreground transmission applied to the rear average changes.
    front_flux = state["front_flux"]
    aligned_back_seen_flux = state["back_seen_flux"].copy()
    flux_2 = front_flux + independent_back_seen_flux

    state["aligned_back_seen_flux_diagnostic"] = aligned_back_seen_flux
    state["back_intrinsic_flux"] = (
        intrinsic_back_intensity * aligned.base.PIXEL_SOLID_ANGLE_SR
    )
    state["mean_front_transmission"] = mean_front_transmission
    state["back_seen_flux"] = independent_back_seen_flux
    state["front_flux_fraction"] = np.divide(
        front_flux,
        flux_2,
        out=np.zeros_like(front_flux),
        where=flux_2 > 0.0,
    )
    state["back_flux_fraction"] = np.divide(
        independent_back_seen_flux,
        flux_2,
        out=np.zeros_like(independent_back_seen_flux),
        where=flux_2 > 0.0,
    )
    state["overlap_probability_dense_dense"] = f_cover**2
    state["overlap_probability_dense_rare"] = f_cover * one_minus_f
    state["overlap_probability_rare_dense"] = one_minus_f * f_cover
    state["overlap_probability_rare_rare"] = one_minus_f**2
    return flux_1, flux_2, state, kappas


def print_mass_summary_with_consistency(samples, maximum):
    """Print inherited masses plus the cross-area normalization mismatch."""
    _ORIGINAL_PRINT_MASS_SUMMARY(samples, maximum)
    arrays = aligned.calculate_mass_arrays(np.asarray(samples, dtype=float))
    best = aligned.calculate_mass_arrays(np.asarray(maximum, dtype=float)[None, :])
    ratio = arrays["dust_corrected_2_front"] / arrays["dust_corrected_1"]
    best_ratio = (
        best["dust_corrected_2_front"][0] / best["dust_corrected_1"][0]
    )
    q16, median, q84 = np.percentile(ratio, [16.0, 50.0, 84.0])
    print("\n--- Cross-area Smith-normalization diagnostic ---")
    print(
        "M_from_Area2_front / M_from_Area1 = "
        f"{median:.6f} (-{median-q16:.6f}, +{q84-median:.6f}); "
        f"MAP={best_ratio:.6f}"
    )
    print("A physically consistent common normalization would equal 1.")
    print("This is a diagnostic only; equality is NOT forced in the likelihood.")


# Every likelihood call made inside the imported runner resolves these names
# in the imported module.  Replacing them here makes the entire sampler use
# the independent-overlap transfer equation, including plots and saved flux
# decompositions, without copying or silently changing the rest of the model.
aligned.evaluate_models = evaluate_models
aligned.print_mass_summary = print_mass_summary_with_consistency


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Two-wall density-dependent-grain MCMC with independent "
            "front/back projected clump overlap."
        )
    )
    parser.add_argument("--walkers", type=int, default=50)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument(
        "--temperature-beta",
        type=float,
        default=aligned.geometry.EMISSIVITY_BETA,
        help=(
            "fixed beta in T proportional to R^[-2/(4+beta)]; "
            "this does not replace the Mie opacity"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            TWO_PHASE_RESULTS / "rerun"
        ),
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--mass-only", action="store_true")
    args = parser.parse_args()
    number_parameters = len(aligned.PARAMETER_NAMES)
    if args.walkers < 2 * number_parameters:
        parser.error(f"walkers must be at least {2*number_parameters}")
    if args.steps < 2:
        parser.error("steps must be at least 2")
    if args.burn < 0 or args.burn >= args.steps:
        parser.error("burn must satisfy 0 <= burn < steps")
    if args.thin < 1:
        parser.error("thin must be at least 1")
    if not np.isfinite(args.temperature_beta) or args.temperature_beta <= -4.0:
        parser.error("temperature-beta must be finite and greater than -4")
    return args


def main():
    args = parse_arguments()
    # The temperature law is fixed for this run but remains inside every
    # posterior evaluation through derive_state().  It is not a free MCMC
    # parameter and it does not alter the Mie opacity calculation.
    aligned.geometry.configure_temperature_beta(args.temperature_beta)
    f_cover = aligned.base.AREA2_FILLING_FRACTION
    print("\nINDEPENDENT FRONT/BACK CLUMP-OVERLAP VARIANT")
    print(
        "Temperature law: beta="
        f"{aligned.geometry.EMISSIVITY_BETA:.6f}, "
        "T proportional to R^(-"
        f"{aligned.geometry.TEMPERATURE_DISTANCE_EXPONENT:.6f})"
    )
    print(f"f_front = f_back = {f_cover:.6f} (assumed)")
    print(
        "P_dd / P_dr / P_rd / P_rr = "
        f"{f_cover**2:.6f} / {f_cover*(1-f_cover):.6f} / "
        f"{(1-f_cover)*f_cover:.6f} / {(1-f_cover)**2:.6f}"
    )
    print("The two projected clump masks are assumed statistically independent.")
    print("The dense/rare radial-column contrast is still shared front-to-back.")
    if args.mass_only:
        aligned.run_mass_only(args)
    else:
        aligned.run_mcmc(args)


if __name__ == "__main__":
    main()
