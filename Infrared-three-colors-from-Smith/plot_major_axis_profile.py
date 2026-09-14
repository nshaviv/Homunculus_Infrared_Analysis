#!/usr/bin/env python3
"""Integrate the reconstructed Homunculus map across its minor axis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def major_axis_profile(data, footprint, header, pa_deg=132.0, bin_width_arcsec=0.1):
    """Return major-axis coordinate and minor-axis-integrated flux density.

    Position angle is measured east of north. The histogram is normalized by
    its major-axis bin width, yielding Jy/arcsec from Jy/arcsec^2 pixels.
    """
    data = np.asarray(data, dtype=np.float64)
    footprint = np.asarray(footprint, dtype=bool)
    height, width = data.shape
    x = (np.arange(width) + 1 - header["CRPIX1"]) * header["CDELT1"]
    y = (np.arange(height) + 1 - header["CRPIX2"]) * header["CDELT2"]
    xx, yy = np.meshgrid(x, y)
    pa = np.deg2rad(pa_deg)
    # The displayed horizontal offset increases to the right, whereas R.A.
    # (east) increases to the left. Convert to east before applying sky PA.
    along = -xx * np.sin(pa) + yy * np.cos(pa)
    valid = footprint & np.isfinite(data)
    lo = np.floor(along[valid].min() / bin_width_arcsec) * bin_width_arcsec
    hi = np.ceil(along[valid].max() / bin_width_arcsec) * bin_width_arcsec
    edges = np.arange(lo, hi + bin_width_arcsec * 1.01, bin_width_arcsec)
    pixel_area = abs(header["CDELT1"] * header["CDELT2"])
    integrated, _ = np.histogram(along[valid], bins=edges, weights=data[valid] * pixel_area)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, integrated / np.diff(edges), edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--pa", type=float, default=132.0, help="major-axis position angle east of north")
    parser.add_argument("--bin-width", type=float, default=0.1, help="major-axis bin width in arcsec")
    args = parser.parse_args()

    product = np.load(args.outputs / "i18_map.npz")
    data = product["i18_map"].astype(float)
    footprint = product["footprint"].astype(bool)
    header = fits.getheader(args.outputs / "i18_map.fits")
    distance, profile, edges = major_axis_profile(data, footprint, header, args.pa, args.bin_width)
    total_flux = float(np.sum(profile * np.diff(edges)))

    csv_path = args.outputs / "i18_major_axis_profile.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["major_axis_offset_arcsec", "minor_axis_integrated_flux_jy_per_arcsec"])
        writer.writerows(zip(distance, profile))

    metadata = {
        "position_angle_deg_east_of_north": args.pa,
        "positive_direction": "southeast",
        "bin_width_arcsec": args.bin_width,
        "total_integrated_flux_jy": total_flux,
        "method": "native-pixel area-conserving sky-PA projection; right-positive display X is converted to east before projection; no map rotation or interpolation",
        "status": "provisional figure-derived flux profile",
    }
    (args.outputs / "i18_major_axis_profile.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.plot(distance, profile, color="#9d2f21", linewidth=1.8)
    ax.fill_between(distance, profile, color="#d97a55", alpha=0.22)
    ax.axvline(0, color="0.25", linewidth=0.8, linestyle="--")
    ax.set(
        xlabel="Offset along major axis (arcsec; southeast positive)",
        ylabel=r"Minor-axis integrated flux (Jy arcsec$^{-1}$)",
        title=f"Eta Carinae Homunculus at 18 um | major-axis PA = {args.pa:g} deg",
    )
    ax.text(
        0.99,
        0.96,
        f"Integrated profile: {total_flux:.1f} Jy\nFigure-derived provisional calibration",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.outputs / "i18_major_axis_profile.png", dpi=240)
    fig.savefig(args.outputs / "i18_major_axis_profile.pdf")
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Integrated 18 um flux: {total_flux:.3f} Jy")


if __name__ == "__main__":
    main()
