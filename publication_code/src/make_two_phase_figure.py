#!/usr/bin/env python3
"""Create the publication figure for the final joint two-phase MCMC fit.

The sampler is not rerun.  The script reads the saved posterior from the
independent front/rear-overlap model and evaluates its exact radiative-transfer
functions.  The reported run uses beta=0 only in the deterministic
temperature-distance relation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

from make_single_phase_figure import configure_style

from project_paths import FIGURE_DIR, TWO_PHASE_RESULTS


OUTPUT_DIRECTORY = TWO_PHASE_RESULTS
POSTERIOR_PATH = OUTPUT_DIRECTORY / "posterior_samples.npz"
FIGURE_DIRECTORY = FIGURE_DIR
PDF_PATH = FIGURE_DIRECTORY / "two_phase_joint_fit_bw.pdf"
PNG_PATH = FIGURE_DIRECTORY / "two_phase_joint_fit_bw.png"

NUMBER_POSTERIOR_CURVES = 500
RANDOM_SEED = 20261028


def main():
    import two_phase_fit as model

    if not POSTERIOR_PATH.is_file():
        raise FileNotFoundError(POSTERIOR_PATH)

    # Reproduce the saved run.  Beta changes only the deterministic
    # temperature-distance scaling; Mie opacity remains fully wavelength
    # dependent.
    model.aligned.geometry.configure_temperature_beta(0.0)

    with np.load(POSTERIOR_PATH, allow_pickle=False) as archive:
        samples = np.asarray(archive["samples"], dtype=float)
        log_probability = np.asarray(archive["log_probability"], dtype=float)
        parameter_names = tuple(str(value) for value in archive["parameter_names"])
    if parameter_names != model.aligned.PARAMETER_NAMES:
        raise ValueError("Saved posterior has an unexpected parameter order.")

    median = np.percentile(samples, 50.0, axis=0)
    area1_data = model.aligned.base.load_flux_data("lobe.edge.data")
    area2_data = model.aligned.base.load_flux_data("lobeedgedata2")
    maximum_posterior = samples[int(np.argmax(log_probability))]
    likelihood_1, likelihood_2 = model.aligned.log_likelihood_components(
        maximum_posterior, area1_data, area2_data
    )
    chi2_total = -2.0 * (likelihood_1 + likelihood_2)
    number_data = len(area1_data[0]) + len(area2_data[0])
    degrees_of_freedom = number_data - len(parameter_names)
    wavelength = np.logspace(np.log10(4.0), np.log10(32.0), 600)

    median_flux_1, median_flux_2, median_state, _ = model.evaluate_models(
        median, wavelength, wavelength
    )

    generator = np.random.default_rng(RANDOM_SEED)
    draw_count = min(NUMBER_POSTERIOR_CURVES, len(samples))
    draw_indices = generator.choice(len(samples), size=draw_count, replace=False)
    posterior_flux_1 = np.empty((draw_count, len(wavelength)))
    posterior_flux_2 = np.empty((draw_count, len(wavelength)))
    for output_index, sample_index in enumerate(draw_indices):
        (
            posterior_flux_1[output_index],
            posterior_flux_2[output_index],
            _,
            _,
        ) = model.evaluate_models(samples[sample_index], wavelength, wavelength)

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

    cgs_to_jy = 1.0e23
    panel_specs = (
        (
            axes[0],
            area1_data,
            median_flux_1,
            band_1,
            "a",
            "Area 1",
            "o",
        ),
        (
            axes[1],
            area2_data,
            median_flux_2,
            band_2,
            "b",
            "Area 2",
            "s",
        ),
    )

    for axis, data, median_flux, band, letter, title, marker in panel_specs:
        wavelength_data, observed_flux, uncertainty = data
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
            zorder=3,
        )
        axis.errorbar(
            wavelength_data,
            observed_flux * cgs_to_jy,
            yerr=uncertainty * cgs_to_jy,
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
            zorder=5,
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
            rf"$\mathbf{{({letter})}}$ {title}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
        )
        axis.set_xlabel(r"Wavelength, $\lambda$ [$\mu$m]")

    axes[0].text(
        0.955,
        0.945,
        (
            rf"$T_\mathrm{{d}}={median[0]:.0f}\,\mathrm{{K}}$"
            "\n"
            rf"$T_\mathrm{{r}}={median[1]:.0f}\,\mathrm{{K}}$"
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        linespacing=1.25,
    )

    axes[1].text(
        0.955,
        0.945,
        (
            rf"$80^\circ:\ {median_state['T_dense_2_front']:.0f},"
            rf"\ {median_state['T_rare_2_front']:.0f}\,\mathrm{{K}}$"
            "\n"
            rf"$28.4^\circ:\ {median_state['T_dense_2_back']:.0f},"
            rf"\ {median_state['T_rare_2_back']:.0f}\,\mathrm{{K}}$"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        linespacing=1.20,
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )

    foreground_line, = axes[1].plot(
        wavelength,
        median_state["front_flux"] * cgs_to_jy,
        color="0.32",
        linewidth=1.0,
        linestyle=(0, (5, 2.5)),
        label=r"$80^\circ$ foreground",
        zorder=2,
    )
    rear_line, = axes[1].plot(
        wavelength,
        median_state["back_seen_flux"] * cgs_to_jy,
        color="0.48",
        linewidth=1.05,
        linestyle=(0, (1.2, 2.0)),
        label=r"Attenuated $28.4^\circ$ rear",
        zorder=2,
    )

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
    component_legend = axes[1].legend(
        handles=[foreground_line, rear_line],
        labels=[
            r"$80^\circ$ foreground",
            r"Attenuated $28.4^\circ$ rear",
        ],
        loc="lower right",
        fontsize=7.1,
        handlelength=2.8,
        borderaxespad=0.7,
        frameon=False,
    )
    axes[1].add_artist(component_legend)

    figure.text(
        0.5,
        0.902,
        (
            rf"Joint fit: $\chi^2={chi2_total:.2f}$; "
            rf"$\nu=N-k={degrees_of_freedom}$ "
            r"(reduced $\chi^2$ undefined)"
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
