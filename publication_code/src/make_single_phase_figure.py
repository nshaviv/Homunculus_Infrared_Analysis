#!/usr/bin/env python3
"""Create the publication figure for the joint fixed-tau single-phase fit.

The MCMC is not rerun.  This script reads the saved posterior produced by
joint_powerlaw_fixed_tau_og_errors_mcmc.py, evaluates the exact radiative-
transfer functions from that file, and draws the posterior-median spectrum
with a pointwise 68 per cent posterior band.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np


from project_paths import (
    FIGURE_DIR,
    MODEL_INPUT_DIR,
    SINGLE_PHASE_RESULTS,
)


SOURCE_SCRIPT = Path(__file__).resolve().parent / "single_phase_fit.py"
SOURCE_OUTPUT = SINGLE_PHASE_RESULTS
FIGURE_DIRECTORY = FIGURE_DIR

POSTERIOR_PATH = SOURCE_OUTPUT / "posterior_samples.npz"
SUMMARY_PATH = SOURCE_OUTPUT / "summary.json"
OPACITY_CACHE_PATH = SOURCE_OUTPUT / "opacity_cache.npz"

PDF_PATH = FIGURE_DIRECTORY / "single_phase_joint_fit_bw.pdf"
PNG_PATH = FIGURE_DIRECTORY / "single_phase_joint_fit_bw.png"

NUMBER_POSTERIOR_CURVES = 800
RANDOM_SEED = 20261028


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "joint_fixed_tau_model", SOURCE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_spectra(module, opacity, parameters, wavelength):
    temperature_1, temperature_2, a_small, a_large, slope = parameters
    kappa_grid = module.effective_kappa_powerlaw(
        a_small,
        a_large,
        slope,
        opacity.mixture,
        integration_points=64,
    )
    kappa_reference = float(
        np.interp(18.0, module.WAVELENGTH_GRID, kappa_grid)
    )
    kappa_wavelength = np.interp(
        wavelength, module.WAVELENGTH_GRID, kappa_grid
    )
    flux_1, _, _ = module.flux_from_dictated_tau(
        wavelength,
        temperature_1,
        kappa_wavelength,
        kappa_reference,
        0.30,
    )
    flux_2, _, _ = module.flux_from_dictated_tau(
        wavelength,
        temperature_2,
        kappa_wavelength,
        kappa_reference,
        0.10,
    )
    return flux_1, flux_2


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.75,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.2,
            "ytick.minor.size": 2.2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main():
    required = (
        SOURCE_SCRIPT,
        POSTERIOR_PATH,
        SUMMARY_PATH,
        OPACITY_CACHE_PATH,
        MODEL_INPUT_DIR / "lobe.edge.data",
        MODEL_INPUT_DIR / "lobeedgedata2",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))

    module = load_model_module()
    opacity = module._load_compatible_cache(OPACITY_CACHE_PATH)
    with np.load(POSTERIOR_PATH, allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        names = tuple(str(value) for value in archive["parameter_names"])
    if names != module.PARAMETER_NAMES:
        raise ValueError("The saved posterior has an unexpected parameter order.")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    median = np.asarray(
        [
            summary["parameters"][name]["median"]
            for name in module.PARAMETER_NAMES
        ],
        dtype=float,
    )

    region_1 = module.load_region_data(
        MODEL_INPUT_DIR,
        "lobe.edge.data",
        "Area 1",
        error_fraction=0.05,
        background_noise_cgs=5.0e-23,
    )
    region_2 = module.load_region_data(
        MODEL_INPUT_DIR,
        "lobeedgedata2",
        "Area 2",
        error_fraction=0.05,
        background_noise_cgs=5.0e-23,
    )

    wavelength = np.logspace(np.log10(4.0), np.log10(32.0), 600)
    median_flux_1, median_flux_2 = model_spectra(
        module, opacity, median, wavelength
    )

    generator = np.random.default_rng(RANDOM_SEED)
    draw_count = min(NUMBER_POSTERIOR_CURVES, len(samples))
    draw_indices = generator.choice(len(samples), size=draw_count, replace=False)
    posterior_flux_1 = np.empty((draw_count, len(wavelength)))
    posterior_flux_2 = np.empty((draw_count, len(wavelength)))
    for output_index, sample_index in enumerate(draw_indices):
        posterior_flux_1[output_index], posterior_flux_2[output_index] = (
            model_spectra(module, opacity, samples[sample_index], wavelength)
        )
    band_1 = np.percentile(posterior_flux_1, [16.0, 84.0], axis=0)
    band_2 = np.percentile(posterior_flux_2, [16.0, 84.0], axis=0)

    configure_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.45),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    panel_specs = (
        (
            axes[0],
            region_1,
            median_flux_1,
            band_1,
            "a",
            r"Area 1",
            r"$T_1=152.8\,\mathrm{K}$",
            r"$\tau_{18}=0.30$",
            "o",
        ),
        (
            axes[1],
            region_2,
            median_flux_2,
            band_2,
            "b",
            r"Area 2",
            r"$T_2=213.7\,\mathrm{K}$",
            r"$\tau_{18}=0.10$",
            "s",
        ),
    )

    cgs_to_jy = 1.0e23
    for (
        axis,
        region,
        median_flux,
        band,
        panel_letter,
        area_name,
        temperature_text,
        tau_text,
        marker,
    ) in panel_specs:
        axis.fill_between(
            wavelength,
            band[0] * cgs_to_jy,
            band[1] * cgs_to_jy,
            color="0.78",
            linewidth=0,
            label=r"68\% posterior interval",
            zorder=1,
        )
        axis.plot(
            wavelength,
            median_flux * cgs_to_jy,
            color="black",
            linewidth=1.35,
            label="Posterior-median model",
            zorder=2,
        )
        axis.errorbar(
            region.wavelength_micron,
            region.flux_cgs * cgs_to_jy,
            yerr=region.error_cgs * cgs_to_jy,
            fmt=marker,
            linestyle="none",
            markersize=4.6,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            ecolor="black",
            elinewidth=0.75,
            capsize=1.8,
            capthick=0.75,
            label="Observed flux",
            zorder=4,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(4.0, 32.0)
        axis.set_ylim(1.0e-2, 2.0e3)
        axis.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0)))
        axis.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(3.0, 10.0))
        )
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_major_locator(LogLocator(base=10.0))
        axis.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2.0, 10.0))
        )
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(False)
        axis.text(
            0.045,
            0.945,
            rf"$\mathbf{{({panel_letter})}}$ {area_name}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
        )
        axis.text(
            0.955,
            0.945,
            temperature_text + "\n" + tau_text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            linespacing=1.25,
        )
        axis.set_xlabel(r"Wavelength, $\lambda$ [$\mu$m]")

    axes[0].set_ylabel(r"$F_\nu$ [Jy arcsec$^{-2}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [2, 1, 0]
    figure.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        handlelength=2.2,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    fit_summary = summary["fit_at_maximum_posterior"]
    figure.text(
        0.5,
        0.902,
        (
            r"Joint fit: $\chi^2/\nu="
            f"{fit_summary['chi2_total']:.2f}/"
            f"{fit_summary['degrees_of_freedom']}="
            f"{fit_summary['reduced_chi2']:.2f}$"
        ),
        ha="center",
        va="center",
        fontsize=8.2,
    )
    figure.subplots_adjust(left=0.095, right=0.992, bottom=0.155, top=0.875)

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(PNG_PATH, dpi=450, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    print(PDF_PATH)
    print(PNG_PATH)
    print(f"posterior curves used: {draw_count}")


if __name__ == "__main__":
    main()
