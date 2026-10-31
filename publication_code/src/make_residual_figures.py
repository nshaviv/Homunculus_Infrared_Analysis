#!/usr/bin/env python3
"""Build the paper figures for the residual-flux section.

The major-axis profile uses the full beta=0 radiative-transfer products,
including the temperature--distance law.  The two-dimensional figure uses
the corresponding saved beta=1 products selected for Figure 7.  The spectral
figure compares the observed whole-nebula spectrum with the Area-2 baseline,
the transmitted hot component, and their sum.

Nothing is refitted here.  The profile and temperature-dependent maps are
read from saved calculations.  The hidden-component spectrum uses the saved
800 K row requested for the manuscript.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import json

import hidden_component_model as hidden_model

from project_paths import (
    FIGURE_DIR,
    HIDDEN_COMPONENT_RESULTS,
    WHOLE_NEBULA_RESULTS,
)


OUTPUT = FIGURE_DIR
HIDDEN_SCAN = HIDDEN_COMPONENT_RESULTS / "temperature_scan.csv"
HIDDEN_VALIDATION = HIDDEN_COMPONENT_RESULTS / "validation.csv"
PHYSICAL_BETA1_DIR = WHOLE_NEBULA_RESULTS / "beta_1"
PHYSICAL_MAP_DATA = PHYSICAL_BETA1_DIR / "whole_nebula_products.npz"
PHYSICAL_SUMMARY = PHYSICAL_BETA1_DIR / "summary.json"
PHYSICAL_BETA0_DIR = WHOLE_NEBULA_RESULTS / "beta_0"
PHYSICAL_BETA0_DATA = PHYSICAL_BETA0_DIR / "whole_nebula_products.npz"
HIDDEN_TEMPERATURE_K = 800.0


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.dpi": 350,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_both(figure: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight")


def build_integrated_profile() -> None:
    """Build Figure 6 from the saved beta=0 temperature-dependent model.

    The observed curve is the reconstructed 18-micron map integrated in the
    minor-axis direction.  Each model curve is obtained by applying the full
    two-phase radiative transfer to the spatialized Smith latitude profile,
    with the MCMC parameters of one aperture used as the nebula-wide anchor.
    No posterior refit is performed here.
    """
    data = np.load(PHYSICAL_BETA0_DATA)
    x = np.asarray(data["major_axis_offset_arcsec"], dtype=float)
    observed = np.asarray(data["observed_major_profile_jy_arcsec"], dtype=float)
    area1 = np.asarray(data["area1_major_profile_jy_arcsec"], dtype=float)
    area2 = np.asarray(data["area2_major_profile_jy_arcsec"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.15, 4.45), constrained_layout=True)
    ax.plot(x, observed, color="black", lw=1.75, label="Observed Homunculus")
    ax.plot(
        x,
        area1,
        color="0.28",
        lw=1.4,
        ls="--",
        label="Area 1-anchored model",
    )
    ax.plot(
        x,
        area2,
        color="0.55",
        lw=1.55,
        ls="-.",
        label="Area 2-anchored model",
    )
    ax.axvline(0.0, color="0.35", lw=0.75, ls=":")
    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Offset along the major axis (arcsec; southeast positive)")
    ax.set_ylabel(r"Minor-axis-integrated $18\,\mu$m flux (Jy arcsec$^{-1}$)")
    ax.legend(loc="upper right", frameon=False)
    save_both(fig, "residual_flux_major_axis_profile")
    plt.close(fig)


def build_2d_comparison() -> None:
    """Build Figure 7 from the temperature-dependent beta=1 RT model."""
    data = np.load(PHYSICAL_MAP_DATA)
    summary = json.loads(PHYSICAL_SUMMARY.read_text(encoding="utf-8"))
    x = np.asarray(data["x_arcsec"], dtype=float)
    y = np.asarray(data["y_arcsec"], dtype=float)
    footprint = np.asarray(data["footprint"], dtype=bool)
    maps = [
        np.asarray(data["observed_18um_jy_arcsec2"], dtype=float),
        np.asarray(data["area1_18um_jy_arcsec2"], dtype=float),
        np.asarray(data["area2_18um_jy_arcsec2"], dtype=float),
    ]
    area1 = summary["area1_diagnostics"]
    area2 = summary["area2_diagnostics"]
    titles = [
        "Observed",
        (
            "Area 1-anchored model\n"
            rf"$T_{{\rm d,1}}={area1['anchor_T_dense_K']:.0f}$ K, "
            rf"$T_{{\rm r,1}}={area1['anchor_T_rare_K']:.0f}$ K"
        ),
        (
            "Area 2-anchored model\n"
            rf"$T_{{\rm d,2}}={area2['anchor_T_dense_K']:.0f}$ K, "
            rf"$T_{{\rm r,2}}={area2['anchor_T_rare_K']:.0f}$ K"
        ),
    ]

    # A common linear scale is essential: it lets brightness differences among
    # the three panels be read directly rather than normalizing each map by its
    # own maximum.  The upper limit is the maximum smoothed brightness over all
    # three maps inside the measured footprint.
    finite = np.concatenate([image[footprint] for image in maps])
    vmax = float(np.nanpercentile(finite, 99.5))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.90),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image_artist = None
    for index, (axis, image, title) in enumerate(zip(axes, maps, titles)):
        masked = np.ma.array(image, mask=(~footprint) | ~np.isfinite(image))
        image_artist = axis.imshow(
            masked,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            aspect="equal",
            rasterized=True,
        )
        axis.contour(
            x,
            y,
            footprint.astype(float),
            levels=[0.5],
            colors="0.55",
            linewidths=0.45,
        )
        axis.plot(0.0, 0.0, marker="+", ms=5.0, mew=0.9, color="white")
        axis.set_title(f"({chr(97 + index)}) {title}", pad=5.0, fontsize=9.4)
        axis.set_xlim(-8.7, 8.5)
        axis.set_ylim(-8.0, 8.2)
        axis.set_xlabel("R.A. offset (arcsec)")
    axes[0].set_ylabel("Declination offset (arcsec)")
    assert image_artist is not None
    colorbar = fig.colorbar(
        image_artist,
        ax=axes,
        orientation="horizontal",
        fraction=0.075,
        pad=0.10,
        aspect=45,
    )
    colorbar.set_label(r"$18\,\mu$m surface brightness (Jy arcsec$^{-2}$)")
    save_both(fig, "residual_flux_2d_comparison")
    plt.close(fig)


def selected_hidden_solution(
    temperature_K: float = HIDDEN_TEMPERATURE_K,
) -> tuple[pd.Series, dict]:
    scan = pd.read_csv(HIDDEN_SCAN)
    validation = pd.read_csv(HIDDEN_VALIDATION)
    table = scan.merge(validation, on="T_K", validate="one_to_one")
    match = table[np.isclose(table["T_K"], temperature_K)]
    if len(match) != 1:
        raise ValueError(f"Expected one saved hidden-component row at {temperature_K:g} K")
    row = match.iloc[0]

    wavelength, observed, baseline, fitted_kappa = hidden_model.load_area2_beta1_case()
    settings = SimpleNamespace(
        foreground_reference_micron=100.0,
        hot_reference_micron=100.0,
        closure_micron=100.0,
        total_to_dust=100.0,
        shortwave_max_micron=50.0,
        foreground_fit_min_micron=5.0,
        foreground_fit_max_micron=7.0,
    )
    result = hidden_model.evaluate_emission(
        settings,
        wavelength,
        observed,
        baseline,
        fitted_kappa,
        fitted_kappa,
        float(row["T_K"]),
        float(row["tau_f_100"]),
        float(row["tau_h_100"]),
    )
    return row, result


def build_hidden_component_spectrum() -> None:
    row, result = selected_hidden_solution()
    wavelength = np.asarray(result["wavelength"], dtype=float)
    observed = np.asarray(result["observed_flux"], dtype=float)
    baseline = np.asarray(result["baseline_flux"], dtype=float)
    added_hot = np.asarray(result["hidden_flux"], dtype=float)
    total = np.asarray(result["total_flux"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.15, 4.45), constrained_layout=True)
    ax.plot(
        wavelength,
        observed,
        color="#a3342b",
        lw=1.9,
        label="Observed spectrum",
    )
    ax.plot(
        wavelength,
        baseline,
        color="0.58",
        lw=1.5,
        ls="--",
        label="Area 2 model before hidden component",
    )
    ax.plot(
        wavelength,
        added_hot,
        color="#4c6f8a",
        lw=1.45,
        ls=":",
        label="Added hot component after foreground attenuation",
    )
    ax.plot(
        wavelength,
        total,
        color="0.20",
        lw=1.7,
        ls="-.",
        label="Area 2 model after hidden component",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5.0, 100.0)
    ax.set_xlabel(r"Wavelength, $\lambda$ ($\mu$m)")
    ax.set_ylabel(r"Whole-nebula flux density, $F_\nu$ (Jy)")
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.025,
        0.95,
        rf"$T_{{\rm h}}={row['T_K']:.0f}\,$K",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    save_both(fig, "hidden_component_observed_before_after")
    plt.close(fig)

    print("Selected hidden-component solution")
    print(row.to_string())
    print(f"Recomputed combined dust mass: {float(result['combined_dust_mass_Msun']):.8f} Msun")
    print(f"Recomputed required area: {float(result['projected_area_arcsec2']):.8f} arcsec^2")

    table = pd.DataFrame(
        [
            {
                "T_h_K": float(result["temperature_K"]),
                "required_area_arcsec2": float(result["projected_area_arcsec2"]),
                "tau_h_100": float(result["tau_hot_at_closure"]),
                "tau_f_100": float(result["tau_foreground_at_closure"]),
                "Mdust_hot_Msun": float(result["hot_dust_mass_Msun"]),
                "Mdust_foreground_Msun": float(result["foreground_dust_mass_Msun"]),
                "Mdust_combined_Msun": float(result["combined_dust_mass_Msun"]),
                "Mtotal_combined_Msun_factor100": float(
                    result["combined_total_mass_Msun"]
                ),
            }
        ]
    )
    table.to_csv(HIDDEN_COMPONENT_RESULTS / "selected_800K.csv", index=False)


def main() -> None:
    set_paper_style()
    build_integrated_profile()
    build_2d_comparison()
    build_hidden_component_spectrum()
    print(f"Saved draft figures to {OUTPUT}")


if __name__ == "__main__":
    main()
