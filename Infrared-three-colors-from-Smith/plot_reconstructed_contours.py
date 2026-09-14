#!/usr/bin/env python3
"""Plot the reconstructed 18 um map with Smith et al. (2003) Fig. 1e levels."""

from __future__ import annotations

import argparse, json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from astropy.io import fits
from PIL import Image
from scipy.ndimage import gaussian_filter


LEVELS = np.array(
    [
        30, 60, 120, 200, 250, 300, 375, 450, 520, 590, 750,
        870, 970, 1050, 1200, 1350, 1450, 1550, 1700, 2000, 2500, 3200,
    ],
    dtype=float,
)

FIG1E_XREF = 35


def angular_axes(shape: tuple[int, int], header: fits.Header) -> tuple[np.ndarray, np.ndarray]:
    """Return linear angular offsets at FITS pixel centers."""
    height, width = shape
    x = (np.arange(width) + 1.0 - header["CRPIX1"]) * header["CDELT1"] + header["CRVAL1"]
    y = (np.arange(height) + 1.0 - header["CRPIX2"]) * header["CDELT2"] + header["CRVAL2"]
    return x, y


def smooth_for_display(
    data: np.ndarray,
    footprint: np.ndarray,
    header: fits.Header,
    fwhm_arcsec: float,
) -> np.ndarray:
    """Suppress publication-raster speckle with mask-normalized Gaussian smoothing."""
    if fwhm_arcsec <= 0:
        return data.copy()
    sigma_x = fwhm_arcsec / 2.355 / abs(header["CDELT1"])
    sigma_y = fwhm_arcsec / 2.355 / abs(header["CDELT2"])
    support = footprint.astype(float)
    numerator = gaussian_filter(data * support, (sigma_y, sigma_x))
    denominator = gaussian_filter(support, (sigma_y, sigma_x))
    smoothed = numerator / np.maximum(denominator, 1e-8)
    smoothed[~footprint] = 0.0
    return smoothed


def draw_reconstruction(
    ax: plt.Axes,
    data: np.ndarray,
    footprint: np.ndarray,
    header: fits.Header,
    smooth_fwhm_arcsec: float,
):
    x, y = angular_axes(data.shape, header)
    shown_levels = LEVELS[LEVELS <= np.nanmax(data)]
    masked = np.ma.masked_where(~footprint, data)

    # Fig. 1e uses darker shading for higher surface brightness.
    fill_levels = np.unique(np.r_[0.0, shown_levels, np.nanmax(data)])
    ax.contourf(x, y, masked, levels=fill_levels, cmap="Greys", extend="max", antialiased=True)
    contours = ax.contour(x, y, masked, levels=shown_levels, colors="black", linewidths=0.55)
    ax.plot(0, 0, marker="+", color="white", markeredgewidth=1.2, markersize=10)
    ax.plot(0, 0, marker="+", color="black", markeredgewidth=0.55, markersize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("R.A. offset (arcsec)")
    ax.set_ylabel("Declination offset (arcsec)")
    ax.set_title(f"Reconstructed 18 um map (display smoothing {smooth_fwhm_arcsec:.2f} arcsec FWHM)")
    ax.tick_params(direction="in", top=True, right=True)
    return contours, shown_levels


def fig1_angular_extent(shape: tuple[int, int], header: fits.Header) -> list[float]:
    """Extent at pixel edges on the same Fig1e-derived OFFSET grid as the FITS."""
    x, y = angular_axes(shape, header)
    dx, dy = abs(header["CDELT1"]), abs(header["CDELT2"])
    return [x[0] - dx / 2, x[-1] + dx / 2, y[-1] - dy / 2, y[0] + dy / 2]


def render_fig1e(pdf_path: Path, dpi_scale: float = 4.0) -> np.ndarray:
    """Exact embedded PBOX raster, excluding page labels and panel margins."""
    doc = pymupdf.open(pdf_path)
    return np.asarray(Image.open(BytesIO(doc.extract_image(FIG1E_XREF)["image"])).convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--pdf", type=Path, default=Path("Smith-Homonculus-mass-2003.pdf"))
    parser.add_argument("--smooth-fwhm", type=float, default=0.15, help="display-only smoothing in arcsec")
    args = parser.parse_args()

    product = np.load(args.outputs / "i18_map.npz", allow_pickle=True)
    data = product["i18_map"].astype(float)
    footprint = product["footprint"].astype(bool)
    header = fits.getheader(args.outputs / "i18_map.fits")
    display_data = smooth_for_display(data, footprint, header, args.smooth_fwhm)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    _, shown_levels = draw_reconstruction(ax, display_data, footprint, header, args.smooth_fwhm)
    fig.savefig(args.outputs / "i18_contours_angular.png", dpi=240)
    fig.savefig(args.outputs / "i18_contours_angular.pdf")
    plt.close(fig)

    original = render_fig1e(args.pdf)
    scale = abs(header["CDELT1"])
    reg = json.loads((args.outputs / "decisions.json").read_text())["registration"]
    # Exact xref35/PBOX interior uses its own measured Fig1 tick scale.
    fig1scale = reg["fig1_arcsec_per_pixel"]
    starx, stary = reg["fig1_star_pixel"]
    panel_extent = [(-starx - .5) * fig1scale, (220 - starx + .5) * fig1scale,
                    -(220 - stary + .5) * fig1scale, -(-stary - .5) * fig1scale]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    # The left panel is display-only, but its axes deliberately equal the FITS
    # axes so the formerly compressed extent can be inspected directly.
    axes[0].imshow(original, extent=panel_extent, origin="upper")
    axes[0].plot(0, 0, "+", color="white", markersize=10)
    axes[0].set_title("Smith et al. (2003), Fig. 1e (matched offsets)")
    axes[0].set(xlabel="R.A. offset (arcsec)", ylabel="Declination offset (arcsec)")
    axes[0].set_aspect("equal")
    axes[0].tick_params(direction="in", top=True, right=True)
    draw_reconstruction(axes[1], display_data, footprint, header, args.smooth_fwhm)
    axes[1].set_xlim(panel_extent[:2]); axes[1].set_ylim(panel_extent[2:])
    missing_display = LEVELS[LEVELS > np.nanmax(display_data)]
    missing_map = LEVELS[LEVELS > np.nanmax(data)]
    if missing_display.size:
        axes[1].text(
            0.02,
            0.02,
            f"display-only missing: {missing_display.astype(int).tolist()}\n"
            f"map missing: {missing_map.astype(int).tolist()}",
            transform=axes[1].transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.5"},
        )
    fig.savefig(args.outputs / "comparison_fig1e_reconstruction.png", dpi=220)
    plt.close(fig)

    print(f"Map range: {np.nanmin(data):.3f} to {np.nanmax(data):.3f} Jy arcsec^-2")
    print(f"Display smoothing: {args.smooth_fwhm:.3f} arcsec FWHM")
    print(f"Published levels drawn: {shown_levels.astype(int).tolist()}")
    print(f"Levels absent after display smoothing: {missing_display.astype(int).tolist()}")
    print(f"Levels absent from scientific map: {missing_map.astype(int).tolist()}")


if __name__ == "__main__":
    main()
