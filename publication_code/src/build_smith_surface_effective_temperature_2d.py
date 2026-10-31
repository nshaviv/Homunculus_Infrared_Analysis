#!/usr/bin/env python3
"""Build the Area-1-anchored temperature map on the Smith velocity surface.

This replaces the rejected STL projection.  The geometry comes directly from
``homunculus-IR-smith/v_vs_theta.csv`` and is displayed on the same native
R.A./declination grid as the earlier 2-D 18-micron flux distributions.

The only inferred temperature is the Area-1 effective 18-micron brightness
temperature.  It is obtained from the observed one-square-arcsecond Area-1
flux and the latest MCMC LOS optical depth.  It is then propagated across the
nearest observer-facing Smith surface with the beta=1 equilibrium law

    T(R) = T_eff,Area1 (R_Area1/R)^0.4.

No nebula-wide dense fraction, rare fraction, or flux-weighted temperature is
introduced.  Area 1 and Area 2 are shown as registered aperture outlines.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy.interpolate import PchipInterpolator

from build_area_flux_per_column_2d_maps import angular_grid
from build_front_effective_temperature_2d import (
    AREA1_REFERENCE_RADIUS_CM,
    infer_area1_effective_temperature,
)

from project_paths import (
    MODEL_INPUT_DIR,
    OBSERVED_DATA_DIR,
    SMITH_GEOMETRY_DIR,
    WHOLE_NEBULA_RESULTS,
)


PA_DEG = 132.0
INCLINATION_FROM_LOS_DEG = 41.0
CENTRAL_AGE_YR = 2000.0 - 1847.1
EJECTION_DURATION_YR = 20.0
DISTANCE_PC = 2300.0
PARSEC_CM = 3.0856775814913673e18
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
TEMPERATURE_EXPONENT_BETA1 = 0.4
APERTURE_PIXELS = 600


def read_velocity_curve(path: Path) -> PchipInterpolator:
    table = np.genfromtxt(path, delimiter=",", names=True)
    names = table.dtype.names
    if names is None or len(names) != 2:
        raise ValueError(f"Expected two named columns in {path}")
    latitude = np.asarray(table[names[0]], dtype=float)
    velocity = np.asarray(table[names[1]], dtype=float)
    return PchipInterpolator(latitude, velocity, extrapolate=False)


def registered_aperture(bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Reproduce the aperture registration used by the LOS calculation."""
    x0, y0, x1, y1 = bounds
    source = np.asarray(
        [
            [700.0 * x0, 700.0 * (1.0 - y0)],
            [700.0 * x1, 700.0 * (1.0 - y0)],
            [700.0 * x1, 700.0 * (1.0 - y1)],
            [700.0 * x0, 700.0 * (1.0 - y1)],
        ]
    )
    center_in = np.asarray([350.0, 350.0])
    center_out = np.asarray([327.3333333333333, 306.6666666666667])
    scale = 0.7422857142857143
    angle = np.deg2rad(-54.5)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return ((source - center_in) * scale) @ rotation.T + center_out


def apertures_in_sky_coordinates(
    velocity_curve: PchipInterpolator,
) -> dict[str, np.ndarray]:
    """Convert the established 600-pixel Smith apertures to the native sky frame."""
    age_max_s = (
        CENTRAL_AGE_YR + 0.5 * EJECTION_DURATION_YR
    ) * 365.25 * 24.0 * 3600.0
    latitude_dense = np.linspace(0.0, 90.0, 5001)
    extent_cm = (
        1.05
        * float(np.nanmax(velocity_curve(latitude_dense)))
        * 1.0e5
        * age_max_s
    )
    distance_cm = DISTANCE_PC * PARSEC_CM
    pa = np.deg2rad(PA_DEG)
    output: dict[str, np.ndarray] = {}
    bounds = {
        "Area 1": (0.19, 0.58, 0.25, 0.65),
        "Area 2": (0.20, 0.33, 0.30, 0.44),
    }
    for name, box in bounds.items():
        model_pixels = registered_aperture(box)
        # calculate_los_correction.jl maps x to the cross-axis and an upward
        # model coordinate to decreasing raster row.  Undo that map first.
        cross_cm = (
            2.0 * (model_pixels[:, 0] - 1.0) / (APERTURE_PIXELS - 1.0) - 1.0
        ) * extent_cm
        along_cm = (
            1.0 - 2.0 * (model_pixels[:, 1] - 1.0) / (APERTURE_PIXELS - 1.0)
        ) * extent_cm
        along = along_cm / distance_cm / ARCSEC_TO_RAD
        cross = cross_cm / distance_cm / ARCSEC_TO_RAD
        x = -along * np.sin(pa) + cross * np.cos(pa)
        y = along * np.cos(pa) + cross * np.sin(pa)
        # The published Smith aperture raster and the native reconstructed
        # sky display use opposite image-axis senses.  A 180-degree image
        # rotation is therefore required after undoing the model raster map.
        # This places Area 1 in the north-west aperture and Area 2 in the
        # south-west aperture exactly as in the supplied Smith figure.
        x = -x
        y = -y
        output[name] = np.column_stack([x, y])
    return output


def aperture_mask(
    polygon: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return PolygonPath(polygon).contains_points(points).reshape(xx.shape)


def draw_figure(
    output_png: Path,
    output_pdf: Path,
    x: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    footprint: np.ndarray,
    temperature: np.ndarray,
    apertures: dict[str, np.ndarray],
    anchor: dict[str, float],
) -> None:
    extent = [x[0], x[-1], y[0], y[-1]]
    observed_display = np.where(footprint, observed, np.nan)
    observed_limit = float(np.nanpercentile(observed_display, 99.5))
    finite_temperature = temperature[np.isfinite(temperature)]
    temperature_norm = Normalize(
        vmin=float(np.percentile(finite_temperature, 1.0)),
        vmax=float(np.percentile(finite_temperature, 99.0)),
        clip=True,
    )

    observed_cmap = plt.get_cmap("gray_r").copy()
    observed_cmap.set_bad("white")
    temperature_cmap = plt.get_cmap("inferno").copy()
    temperature_cmap.set_bad("white")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.55),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes[0].imshow(
        observed_display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=observed_cmap,
        vmin=0.0,
        vmax=observed_limit,
        aspect="equal",
    )
    image = axes[1].imshow(
        temperature,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=temperature_cmap,
        norm=temperature_norm,
        aspect="equal",
    )
    axes[1].contour(
        x,
        y,
        footprint.astype(float),
        levels=[0.5],
        colors="0.55",
        linewidths=0.45,
    )
    axes[0].set_title("Observed 18 $\\mu$m morphology")
    axes[1].set_title("Effective temperature on Smith front surface")

    aperture_style = {
        "Area 1": ("#df2f2f", "1"),
        "Area 2": ("#1f5fe0", "2"),
    }
    for axis in axes:
        for name, polygon in apertures.items():
            color, short_label = aperture_style[name]
            closed = np.vstack([polygon, polygon[0]])
            axis.plot(closed[:, 0], closed[:, 1], color=color, lw=1.8)
            center = np.mean(polygon, axis=0)
            axis.text(
                center[0],
                center[1],
                short_label,
                color=color,
                fontsize=10,
                weight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
            )
        axis.plot(0.0, 0.0, marker="+", color="white", ms=7, mew=1.1)
        axis.set_xlim(x[0], x[-1])
        axis.set_ylim(y[0], y[-1])
        axis.set_xlabel("R.A.-display offset (arcsec; right positive)")
        axis.tick_params(direction="in", top=True, right=True, labelsize=8)
    axes[0].set_ylabel("Declination offset (arcsec)")
    colorbar = figure.colorbar(
        image,
        ax=axes,
        location="bottom",
        shrink=0.72,
        pad=0.04,
    )
    colorbar.set_label("Effective temperature (K)")
    figure.suptitle(
        "Area-1-anchored temperature on the Smith velocity surface\n"
        rf"$T_{{\rm eff,1}}={anchor['area1_effective_temperature_K']:.2f}$ K; "
        r"$T(R)=T_{\rm eff,1}(R_1/R)^{0.4}$; nearest observer-facing surface",
        fontsize=13,
    )
    figure.savefig(output_png, dpi=240, facecolor="white")
    figure.savefig(output_pdf, facecolor="white")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameter-summary",
        type=Path,
        default=WHOLE_NEBULA_RESULTS / "parameters" / "beta_p1p00"
        / "parameter_summary.csv",
    )
    parser.add_argument("--area1-data", type=Path, default=MODEL_INPUT_DIR / "lobe.edge.data")
    parser.add_argument(
        "--velocity-curve",
        type=Path,
        default=SMITH_GEOMETRY_DIR / "v_vs_theta.csv",
    )
    parser.add_argument(
        "--observed-map",
        type=Path,
        default=OBSERVED_DATA_DIR / "i18_map.npz",
    )
    parser.add_argument(
        "--observed-decisions",
        type=Path,
        default=OBSERVED_DATA_DIR / "i18_map_decisions.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WHOLE_NEBULA_RESULTS / "surface_temperature_diagnostic",
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
    velocity_curve = read_velocity_curve(args.velocity_curve.expanduser().resolve())
    with np.load(args.observed_map.expanduser().resolve(), allow_pickle=True) as product:
        observed_original = np.asarray(product["i18_map"], dtype=float)
        footprint_original = np.asarray(product["footprint"], dtype=bool)
    decisions = json.loads(
        args.observed_decisions.expanduser().resolve().read_text(encoding="utf-8")
    )
    x, y, x_edges, y_edges = angular_grid(observed_original.shape, decisions)
    observed = np.flipud(observed_original)
    footprint = np.flipud(footprint_original)

    subprocess.run(
        [
            "julia",
            str(root / "build_smith_front_temperature_surface.jl"),
            str(output_dir),
            str(anchor["area1_effective_temperature_K"]),
            str(AREA1_REFERENCE_RADIUS_CM),
            str(x_edges[0]),
            str(x_edges[-1]),
            str(y_edges[0]),
            str(y_edges[-1]),
            str(len(y)),
            str(len(x)),
        ],
        check=True,
        cwd=root,
    )
    temperature = np.loadtxt(
        output_dir / "smith_front_temperature_K.csv", delimiter=","
    )
    radius = np.loadtxt(output_dir / "smith_front_radius_cm.csv", delimiter=",")
    depth = np.loadtxt(output_dir / "smith_front_depth_cm.csv", delimiter=",")
    latitude = np.loadtxt(
        output_dir / "smith_front_absolute_latitude_deg.csv", delimiter=","
    )
    apertures = apertures_in_sky_coordinates(velocity_curve)

    aperture_summary: dict[str, dict[str, float]] = {}
    for name, polygon in apertures.items():
        selected = aperture_mask(polygon, x, y) & np.isfinite(temperature)
        values = temperature[selected]
        latitudes = latitude[selected]
        aperture_summary[name] = {
            "finite_surface_pixels": int(np.count_nonzero(selected)),
            "median_effective_temperature_K": float(np.median(values)),
            "mean_effective_temperature_K": float(np.mean(values)),
            "median_visible_surface_absolute_latitude_deg": float(
                np.median(latitudes)
            ),
        }

    figure_png = output_dir / "smith_surface_effective_temperature_2d.png"
    figure_pdf = output_dir / "smith_surface_effective_temperature_2d.pdf"
    arrays_npz = output_dir / "smith_surface_effective_temperature_2d.npz"
    summary_json = output_dir / "summary.json"
    draw_figure(
        figure_png,
        figure_pdf,
        x,
        y,
        observed,
        footprint,
        temperature,
        apertures,
        anchor,
    )
    np.savez_compressed(
        arrays_npz,
        x_ra_display_offset_arcsec=x,
        y_declination_offset_arcsec=y,
        observed_18um_jy_per_arcsec2=observed,
        observed_footprint=footprint,
        smith_front_effective_temperature_K=temperature,
        smith_front_radius_cm=radius,
        smith_front_depth_cm=depth,
        smith_front_absolute_latitude_deg=latitude,
        area1_aperture_sky_arcsec=apertures["Area 1"],
        area2_aperture_sky_arcsec=apertures["Area 2"],
    )
    finite = temperature[np.isfinite(temperature)]
    result = {
        "supersedes": "the earlier STL-based front-temperature projection",
        "geometry": {
            "source": str(args.velocity_curve.expanduser().resolve()),
            "surface": "nearest observer-facing central-age Smith velocity surface",
            "stl_used": False,
            "central_ballistic_age_years": CENTRAL_AGE_YR,
            "inclination_from_line_of_sight_deg": INCLINATION_FROM_LOS_DEG,
            "position_angle_deg": PA_DEG,
            "native_observed_grid_source": str(args.observed_map.expanduser().resolve()),
        },
        "anchor": anchor,
        "temperature_law": {
            "emissivity_beta": 1.0,
            "distance_exponent": TEMPERATURE_EXPONENT_BETA1,
            "formula": "T(R)=T_eff,Area1*(R_Area1/R)^0.4",
            "area1_reference_radius_cm": AREA1_REFERENCE_RADIUS_CM,
            "no_nebula_wide_filling_fraction": True,
            "no_flux_weighted_or_linear_dense_rare_mixing": True,
        },
        "temperature_map": {
            "minimum_K": float(np.min(finite)),
            "maximum_K": float(np.max(finite)),
            "finite_pixels": int(finite.size),
        },
        "aperture_registration": {
            "source": "same registered Area 1/2 polygons used by homunculus-los-gui/tools/calculate_los_correction.jl",
            "statistics": aperture_summary,
        },
        "files": {
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
            "arrays_npz": str(arrays_npz),
            "julia_metadata": str(output_dir / "smith_projection_metadata.txt"),
        },
    }
    summary_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Area-1 effective anchor: {anchor['area1_effective_temperature_K']:.6f} K")
    print(f"Smith-surface temperature range: {np.min(finite):.6f}--{np.max(finite):.6f} K")
    for name, values in aperture_summary.items():
        print(
            f"{name}: median T={values['median_effective_temperature_K']:.6f} K; "
            f"median visible |latitude|={values['median_visible_surface_absolute_latitude_deg']:.6f} deg"
        )
    for path in (figure_png, figure_pdf, arrays_npz, summary_json):
        print(path)


if __name__ == "__main__":
    main()
