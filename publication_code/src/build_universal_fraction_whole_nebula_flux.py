#!/usr/bin/env python3
"""Whole-nebula 18-micron forward models anchored independently by Areas 1/2.

For model i, the fitted covering fraction f_i is assumed universal across the
Homunculus.  The fitted phase LOS columns normalise the Smith dM/dOmega shape,
the phase temperatures scale as R^[-2/(4+beta)] for a fixed beta,
density-dependent grain sizes and Mie opacities follow the current MCMC
prescription, and the rear wall is attenuated through an independently
overlapping foreground wall.  Beta changes only the temperature-distance
law; the spectral opacity remains the full Mie calculation.

The Area-1-anchored and Area-2-anchored models are alternatives.  They are
never added together.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import PchipInterpolator

import build_area_flux_per_column_latitude_profiles as one_d
from build_area_flux_per_column_2d_maps import (
    angular_grid,
    major_axis_profile_from_flux_map,
)
from build_smith_surface_effective_temperature_2d import (
    aperture_mask,
    apertures_in_sky_coordinates,
    read_velocity_curve,
)
from build_front_effective_temperature_2d import read_parameter_medians
import two_phase_aligned as physical_model
import temperature_scaling as temperature_law

from project_paths import (
    MODEL_INPUT_DIR,
    OBSERVED_DATA_DIR,
    SMITH_GEOMETRY_DIR,
    TWO_PHASE_RESULTS,
    WHOLE_NEBULA_RESULTS,
)


WAVELENGTH_MICRON = 18.0
PA_DEG = 132.0
AREA1_LATITUDE_DEG = 60.0
AREA2_LATITUDE_DEG = 80.0
ARCSEC2_SR = (np.pi / (180.0 * 3600.0)) ** 2
JY_CGS = 1.0e-23
BETA = temperature_law.DEFAULT_EMISSIVITY_BETA
TEMPERATURE_EXPONENT = temperature_law.temperature_distance_exponent(BETA)


def configure_temperature_beta(beta: float) -> None:
    """Fix beta for this forward-model evaluation and the imported MCMC."""
    global BETA
    global TEMPERATURE_EXPONENT
    BETA = temperature_law.validate_emissivity_beta(beta)
    TEMPERATURE_EXPONENT = temperature_law.temperature_distance_exponent(BETA)
    physical_model.geometry.configure_temperature_beta(BETA)


def read_named_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    names = table.dtype.names
    if names is None or len(names) != 2:
        raise ValueError(f"Expected a named two-column curve: {path}")
    x = np.asarray(table[names[0]], dtype=float)
    y = np.asarray(table[names[1]], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y > 0.0)
    order = np.argsort(x[keep])
    return x[keep][order], y[keep][order]


def nearest_18um_flux(path: Path) -> tuple[float, float]:
    data = np.loadtxt(path, delimiter=",")
    index = int(np.argmin(np.abs(data[:, 0] - WAVELENGTH_MICRON)))
    return float(data[index, 0]), float(data[index, 1])


def planck_nu(wavelength_micron: float, temperature_k: np.ndarray) -> np.ndarray:
    wavelength_cm = wavelength_micron * 1.0e-4
    exponent = (
        physical_model.base.H_CGS
        * physical_model.base.C_CGS
        / (wavelength_cm * physical_model.base.K_B_CGS * temperature_k)
    )
    exponent = np.minimum(exponent, 700.0)
    b_lambda = (
        2.0
        * physical_model.base.H_CGS
        * physical_model.base.C_CGS**2
        / wavelength_cm**5
        / np.expm1(exponent)
    )
    return b_lambda * wavelength_cm**2 / physical_model.base.C_CGS


def load_geometry(source: Path) -> dict[str, np.ndarray]:
    names = (
        "front_depth_cm",
        "back_depth_cm",
        "front_abs_latitude_deg",
        "back_abs_latitude_deg",
        "front_radius_cm",
        "back_radius_cm",
        "front_los_per_radial",
        "back_los_per_radial",
    )
    if source.is_file():
        with np.load(source, allow_pickle=False) as archive:
            missing = [name for name in names if name not in archive]
            if missing:
                raise ValueError(f"Geometry archive is missing: {missing}")
            return {name: np.asarray(archive[name], dtype=float) for name in names}
    return {name: np.loadtxt(source / f"{name}.csv", delimiter=",") for name in names}


def interpolate_map(
    latitude_map: np.ndarray,
    latitude_grid: np.ndarray,
    value_grid: np.ndarray,
) -> np.ndarray:
    result = np.full_like(latitude_map, np.nan, dtype=float)
    finite = np.isfinite(latitude_map)
    result[finite] = np.interp(
        latitude_map[finite], latitude_grid, value_grid
    )
    return result


def kappa_latitude_curve(
    latitude_grid: np.ndarray,
    shape_curve: PchipInterpolator,
    radius_curve: PchipInterpolator,
    anchor_latitude_deg: float,
    anchor_a_dense: float,
    anchor_a_rare: float,
    p_size: float,
    q_density_size: float,
    corundum_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    radius = np.asarray(radius_curve(latitude_grid), dtype=float)
    shape = np.asarray(shape_curve(latitude_grid), dtype=float) / radius**2
    anchor_shape = float(shape_curve(anchor_latitude_deg)) / float(
        radius_curve(anchor_latitude_deg)
    ) ** 2
    density_ratio = shape / anchor_shape
    raw_dense = anchor_a_dense * density_ratio**q_density_size
    raw_rare = anchor_a_rare * density_ratio**q_density_size
    dense_size = np.clip(
        raw_dense, physical_model.A_MIN_MICRON, physical_model.A_MAX_MICRON
    )
    rare_size = np.clip(
        raw_rare, physical_model.A_MIN_MICRON, physical_model.A_MAX_MICRON
    )
    dense_kappa = np.asarray(
        [
            physical_model.powerlaw_mixed_kappa(
                p_size, size, np.asarray([WAVELENGTH_MICRON]), corundum_fraction
            )[0]
            for size in dense_size
        ]
    )
    rare_kappa = np.asarray(
        [
            physical_model.powerlaw_mixed_kappa(
                p_size, size, np.asarray([WAVELENGTH_MICRON]), corundum_fraction
            )[0]
            for size in rare_size
        ]
    )
    clipping = {
        "dense_fraction_clipped_to_grain_bounds": float(
            np.mean(raw_dense != dense_size)
        ),
        "rare_fraction_clipped_to_grain_bounds": float(
            np.mean(raw_rare != rare_size)
        ),
        "dense_a_min_micron": float(np.min(dense_size)),
        "dense_a_max_micron": float(np.max(dense_size)),
        "rare_a_min_micron": float(np.min(rare_size)),
        "rare_a_max_micron": float(np.max(rare_size)),
    }
    return dense_kappa, rare_kappa, clipping


def area_configuration(
    area: str,
    state: dict[str, float],
) -> dict[str, float | str]:
    if area == "area1":
        fraction = physical_model.base.AREA1_FILLING_FRACTION
        return {
            "name": "Area 1",
            "fraction": fraction,
            "anchor_latitude_deg": AREA1_LATITUDE_DEG,
            "T_dense_anchor_K": state["T_dense_1"],
            "T_rare_anchor_K": state["T_rare_1"],
            "sigma_dense_total_LOS_g_cm2": state[
                "sigma_dense_1_weighted_los"
            ]
            / fraction,
            "sigma_rare_total_LOS_g_cm2": state[
                "sigma_rare_1_weighted_los"
            ]
            / (1.0 - fraction),
            "a_dense_anchor_micron": state["a_dense_1"],
            "a_rare_anchor_micron": state["a_rare_1"],
            "corundum_fraction": physical_model.base.AREA1_CORUNDUM_FRACTION,
        }
    fraction = physical_model.base.AREA2_FILLING_FRACTION
    return {
        "name": "Area 2",
        "fraction": fraction,
        "anchor_latitude_deg": AREA2_LATITUDE_DEG,
        "T_dense_anchor_K": state["T_dense_2_front"],
        "T_rare_anchor_K": state["T_rare_2_front"],
        "sigma_dense_total_LOS_g_cm2": state["sigma_dense_2_total_los"],
        "sigma_rare_total_LOS_g_cm2": state["sigma_rare_2_total_los"],
        "a_dense_anchor_micron": state["a_dense_2_front"],
        "a_rare_anchor_micron": state["a_rare_2_front"],
        "corundum_fraction": state["corundum_2"],
    }


def build_area_model(
    config: dict[str, float | str],
    geometry: dict[str, np.ndarray],
    aperture: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mass_curve: PchipInterpolator,
    velocity_curve: PchipInterpolator,
    p_size: float,
    q_density_size: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    front_lat = geometry["front_abs_latitude_deg"]
    back_lat = geometry["back_abs_latitude_deg"]
    front_factor = geometry["front_los_per_radial"]
    back_factor = geometry["back_los_per_radial"]
    front_radius = geometry["front_radius_cm"]
    back_radius = geometry["back_radius_cm"]
    support = np.isfinite(front_lat) & np.isfinite(back_lat)

    latitude_grid = np.arange(0.25, 90.0, 0.5)
    central_age_seconds = one_d.BALLISTIC_AGE_YR * one_d.SECONDS_PER_YEAR
    radius_grid = (
        np.asarray(velocity_curve(latitude_grid), dtype=float)
        * 1.0e5
        * central_age_seconds
    )
    radius_curve = PchipInterpolator(latitude_grid, radius_grid, extrapolate=True)
    smith_shape_grid = np.asarray(mass_curve(latitude_grid), dtype=float) / radius_grid**2
    front_shape = interpolate_map(front_lat, latitude_grid, smith_shape_grid)
    back_shape = interpolate_map(back_lat, latitude_grid, smith_shape_grid)
    unit_total_los = np.nan_to_num(front_shape * front_factor) + np.nan_to_num(
        back_shape * back_factor
    )
    area_mask = aperture_mask(aperture, x, y)
    unit_mean = float(np.mean(np.where(area_mask, unit_total_los, 0.0)[area_mask]))
    if not np.isfinite(unit_mean) or unit_mean <= 0.0:
        raise ValueError(f"Aperture normalisation failed for {config['name']}")
    dense_amplitude = float(config["sigma_dense_total_LOS_g_cm2"]) / unit_mean
    rare_amplitude = float(config["sigma_rare_total_LOS_g_cm2"]) / unit_mean

    sigma_dense_front = dense_amplitude * front_shape * front_factor
    sigma_dense_back = dense_amplitude * back_shape * back_factor
    sigma_rare_front = rare_amplitude * front_shape * front_factor
    sigma_rare_back = rare_amplitude * back_shape * back_factor

    dense_kappa_grid, rare_kappa_grid, clipping = kappa_latitude_curve(
        latitude_grid,
        mass_curve,
        radius_curve,
        float(config["anchor_latitude_deg"]),
        float(config["a_dense_anchor_micron"]),
        float(config["a_rare_anchor_micron"]),
        p_size,
        q_density_size,
        float(config["corundum_fraction"]),
    )
    kappa_dense_front = interpolate_map(front_lat, latitude_grid, dense_kappa_grid)
    kappa_dense_back = interpolate_map(back_lat, latitude_grid, dense_kappa_grid)
    kappa_rare_front = interpolate_map(front_lat, latitude_grid, rare_kappa_grid)
    kappa_rare_back = interpolate_map(back_lat, latitude_grid, rare_kappa_grid)

    anchor_radius = float(radius_curve(float(config["anchor_latitude_deg"])))
    dense_temperature_front = float(config["T_dense_anchor_K"]) * (
        anchor_radius / front_radius
    ) ** TEMPERATURE_EXPONENT
    dense_temperature_back = float(config["T_dense_anchor_K"]) * (
        anchor_radius / back_radius
    ) ** TEMPERATURE_EXPONENT
    rare_temperature_front = float(config["T_rare_anchor_K"]) * (
        anchor_radius / front_radius
    ) ** TEMPERATURE_EXPONENT
    rare_temperature_back = float(config["T_rare_anchor_K"]) * (
        anchor_radius / back_radius
    ) ** TEMPERATURE_EXPONENT

    tau_dense_front = kappa_dense_front * sigma_dense_front
    tau_dense_back = kappa_dense_back * sigma_dense_back
    tau_rare_front = kappa_rare_front * sigma_rare_front
    tau_rare_back = kappa_rare_back * sigma_rare_back
    f = float(config["fraction"])
    emission_front = (
        f
        * planck_nu(WAVELENGTH_MICRON, dense_temperature_front)
        * -np.expm1(-tau_dense_front)
        + (1.0 - f)
        * planck_nu(WAVELENGTH_MICRON, rare_temperature_front)
        * -np.expm1(-tau_rare_front)
    )
    transmission_front = f * np.exp(-tau_dense_front) + (1.0 - f) * np.exp(
        -tau_rare_front
    )
    emission_back = (
        f
        * planck_nu(WAVELENGTH_MICRON, dense_temperature_back)
        * -np.expm1(-tau_dense_back)
        + (1.0 - f)
        * planck_nu(WAVELENGTH_MICRON, rare_temperature_back)
        * -np.expm1(-tau_rare_back)
    )
    intensity = emission_front + transmission_front * emission_back
    surface_brightness = np.where(
        support, intensity * ARCSEC2_SR / JY_CGS, 0.0
    )

    fitted_dense = float(
        np.mean(
            np.where(
                area_mask,
                np.nan_to_num(sigma_dense_front) + np.nan_to_num(sigma_dense_back),
                0.0,
            )[area_mask]
        )
    )
    fitted_rare = float(
        np.mean(
            np.where(
                area_mask,
                np.nan_to_num(sigma_rare_front) + np.nan_to_num(sigma_rare_back),
                0.0,
            )[area_mask]
        )
    )
    aperture_brightness = float(np.mean(surface_brightness[area_mask]))
    diagnostics = {
        "universal_dense_covering_fraction": f,
        "anchor_latitude_deg": float(config["anchor_latitude_deg"]),
        "anchor_T_dense_K": float(config["T_dense_anchor_K"]),
        "anchor_T_rare_K": float(config["T_rare_anchor_K"]),
        "target_sigma_dense_total_LOS_g_cm2": float(
            config["sigma_dense_total_LOS_g_cm2"]
        ),
        "recovered_sigma_dense_total_LOS_g_cm2": fitted_dense,
        "target_sigma_rare_total_LOS_g_cm2": float(
            config["sigma_rare_total_LOS_g_cm2"]
        ),
        "recovered_sigma_rare_total_LOS_g_cm2": fitted_rare,
        "predicted_anchor_aperture_flux_Jy_for_1arcsec2": aperture_brightness,
        **clipping,
    }
    arrays = {
        "surface_brightness_jy_arcsec2": surface_brightness,
        "sigma_dense_front_los_g_cm2": sigma_dense_front,
        "sigma_dense_back_los_g_cm2": sigma_dense_back,
        "sigma_rare_front_los_g_cm2": sigma_rare_front,
        "sigma_rare_back_los_g_cm2": sigma_rare_back,
        "T_dense_front_K": dense_temperature_front,
        "T_dense_back_K": dense_temperature_back,
        "T_rare_front_K": rare_temperature_front,
        "T_rare_back_K": rare_temperature_back,
        "tau_dense_front": tau_dense_front,
        "tau_dense_back": tau_dense_back,
        "tau_rare_front": tau_rare_front,
        "tau_rare_back": tau_rare_back,
    }
    return arrays, diagnostics


def draw_maps(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    footprint: np.ndarray,
    area1: np.ndarray,
    area2: np.ndarray,
) -> None:
    extent = [x[0], x[-1], y[0], y[-1]]
    observed_display = np.where(footprint, observed, np.nan)
    combined = np.concatenate(
        [
            observed_display[np.isfinite(observed_display)],
            area1[np.isfinite(area1)],
            area2[np.isfinite(area2)],
        ]
    )
    norm = Normalize(vmin=0.0, vmax=float(np.percentile(combined, 99.5)), clip=True)
    cmap = plt.get_cmap("magma").copy(); cmap.set_bad("white")
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 5.2), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        (observed_display, "Observed whole nebula"),
        (np.where(area1 > 0.0, area1, np.nan), "Area 1 fraction universal"),
        (np.where(area2 > 0.0, area2, np.nan), "Area 2 fraction universal"),
    ]
    images=[]
    for axis,(data,title) in zip(axes,panels,strict=True):
        images.append(axis.imshow(data,origin="lower",extent=extent,cmap=cmap,norm=norm,interpolation="nearest",aspect="equal"))
        axis.contour(x,y,footprint.astype(float),levels=[0.5],colors="0.55",linewidths=0.4)
        axis.plot(0,0,"+",color="white",ms=7,mew=1)
        axis.set_title(title,fontsize=11); axis.set_xlabel("R.A.-display offset (arcsec; right positive)")
        axis.tick_params(direction="in",top=True,right=True,labelsize=8)
    axes[0].set_ylabel("Declination offset (arcsec)")
    cbar=figure.colorbar(images[0],ax=axes,location="bottom",shrink=0.72,pad=0.035)
    cbar.set_label("18 $\\mu$m surface brightness (Jy arcsec$^{-2}$)")
    figure.suptitle("Whole-nebula radiative-transfer predictions with the anchor-area fraction made universal",fontsize=13)
    figure.savefig(path,dpi=240,facecolor="white"); plt.close(figure)


def draw_profiles(path, offset, observed, area1, area2):
    figure,axis=plt.subplots(figsize=(9.0,5.1),constrained_layout=True)
    axis.plot(offset,observed,color="#9f2f1f",lw=2.2,label="Observed whole nebula")
    axis.plot(offset,area1,color="#126782",lw=2.0,label="Area 1 fraction universal")
    axis.plot(offset,area2,color="#6f3f8f",lw=2.0,label="Area 2 fraction universal")
    axis.axvline(0,color="0.4",lw=0.8,ls="--")
    axis.set(xlabel="Offset along major axis (arcsec; southeast positive)",ylabel="Integrated flux per offset (Jy arcsec$^{-1}$)",title="Observed and universal-fraction whole-nebula profiles")
    axis.grid(alpha=0.25); axis.legend(frameon=False); axis.tick_params(direction="in",top=True,right=True)
    figure.savefig(path,dpi=240,facecolor="white"); plt.close(figure)


def parse_args():
    root = Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-summary",type=Path,default=TWO_PHASE_RESULTS/"parameter_summary.csv")
    parser.add_argument("--area1-data",type=Path,default=MODEL_INPUT_DIR/"lobe.edge.data")
    parser.add_argument("--area2-data",type=Path,default=MODEL_INPUT_DIR/"lobeedgedata2")
    parser.add_argument("--mass-curve",type=Path,default=SMITH_GEOMETRY_DIR/"Mass-per-SolidAngle.csv")
    parser.add_argument("--velocity-curve",type=Path,default=SMITH_GEOMETRY_DIR/"v_vs_theta.csv")
    parser.add_argument("--observed-map",type=Path,default=OBSERVED_DATA_DIR/"i18_map.npz")
    parser.add_argument("--observed-decisions",type=Path,default=OBSERVED_DATA_DIR/"i18_map_decisions.json")
    parser.add_argument("--observed-profile",type=Path,default=OBSERVED_DATA_DIR/"i18_major_axis_profile.csv")
    parser.add_argument("--output-dir",type=Path,default=WHOLE_NEBULA_RESULTS/"single_beta_rerun")
    return parser.parse_args()


def main():
    args=parse_args(); root=Path(__file__).resolve().parent
    output=args.output_dir.expanduser().resolve(); output.mkdir(parents=True,exist_ok=True)
    medians=read_parameter_medians(args.parameter_summary.expanduser().resolve())
    theta=np.asarray([medians[name] for name in physical_model.PARAMETER_NAMES])
    state=physical_model.derive_state(theta)
    mass_lat,mass_val=read_named_curve(args.mass_curve); mass_curve=PchipInterpolator(mass_lat,mass_val,extrapolate=False)
    velocity_curve=read_velocity_curve(args.velocity_curve)
    with np.load(args.observed_map,allow_pickle=True) as product:
        observed_original=np.asarray(product["i18_map"],float); footprint_original=np.asarray(product["footprint"],bool)
    decisions=json.loads(args.observed_decisions.read_text(encoding="utf-8"))
    x,y,x_edges,y_edges=angular_grid(observed_original.shape,decisions)
    observed=np.flipud(observed_original); footprint=np.flipud(footprint_original)
    subprocess.run(["julia",str(root/"build_smith_two_wall_geometry.jl"),str(output),str(x_edges[0]),str(x_edges[-1]),str(y_edges[0]),str(y_edges[-1]),str(len(y)),str(len(x))],check=True,cwd=root)
    geometry=load_geometry(output); apertures=apertures_in_sky_coordinates(velocity_curve)
    configs={area:area_configuration(area,state) for area in ("area1","area2")}
    models={}; diagnostics={}
    for area,name in (("area1","Area 1"),("area2","Area 2")):
        models[area],diagnostics[area]=build_area_model(configs[area],geometry,apertures[name],x,y,mass_curve,velocity_curve,state["p_size"],state["q_density_size"])
    pixel_area=float(np.diff(x_edges)[0]*np.diff(y_edges)[0])
    observed_total=float(np.sum(np.where(footprint,observed,0.0))*pixel_area)
    for area in models:
        sb=models[area]["surface_brightness_jy_arcsec2"]
        diagnostics[area]["whole_nebula_integrated_flux_Jy"]=float(np.sum(sb)*pixel_area)
        diagnostics[area]["fraction_of_observed_whole_nebula_flux"]=diagnostics[area]["whole_nebula_integrated_flux_Jy"]/observed_total
    offset,observed_profile,offset_edges=one_d.read_observed_profile(args.observed_profile)
    profiles={area:major_axis_profile_from_flux_map(models[area]["surface_brightness_jy_arcsec2"]*pixel_area,x,y,offset_edges,PA_DEG) for area in models}
    map_png=output/"observed_vs_universal_fraction_models_2d.png"; profile_png=output/"observed_vs_universal_fraction_profiles.png"
    draw_maps(map_png,x,y,observed,footprint,models["area1"]["surface_brightness_jy_arcsec2"],models["area2"]["surface_brightness_jy_arcsec2"])
    draw_profiles(profile_png,offset,observed_profile,profiles["area1"],profiles["area2"])
    np.savez_compressed(output/"universal_fraction_whole_nebula_flux.npz",x=x,y=y,footprint=footprint,observed_jy_arcsec2=observed,area1_jy_arcsec2=models["area1"]["surface_brightness_jy_arcsec2"],area2_jy_arcsec2=models["area2"]["surface_brightness_jy_arcsec2"],major_axis_offset_arcsec=offset,observed_profile_jy_arcsec=observed_profile,area1_profile_jy_arcsec=profiles["area1"],area2_profile_jy_arcsec=profiles["area2"])
    with (output/"major_axis_profiles.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream); writer.writerow(["offset_arcsec","observed_jy_per_arcsec","area1_universal_fraction_jy_per_arcsec","area2_universal_fraction_jy_per_arcsec"])
        writer.writerows(zip(offset,observed_profile,profiles["area1"],profiles["area2"]))
    summary={"method":"two independent whole-nebula 18-micron radiative-transfer models; anchor-area covering fraction is universal","fractions":{"area1":physical_model.base.AREA1_FILLING_FRACTION,"area2":physical_model.base.AREA2_FILLING_FRACTION},"temperature_emissivity_beta":BETA,"temperature_exponent":TEMPERATURE_EXPONENT,"observed_whole_nebula_flux_Jy":observed_total,"area1":diagnostics["area1"],"area2":diagnostics["area2"],"files":{"map":str(map_png),"profile":str(profile_png),"arrays":str(output/"universal_fraction_whole_nebula_flux.npz"),"profiles_csv":str(output/"major_axis_profiles.csv")}}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(f"Observed total: {observed_total:.6f} Jy")
    for area in ("area1","area2"):
        d=diagnostics[area]; print(f"{area}: total={d['whole_nebula_integrated_flux_Jy']:.6f} Jy ({d['fraction_of_observed_whole_nebula_flux']:.4%}); anchor aperture={d['predicted_anchor_aperture_flux_Jy_for_1arcsec2']:.6f} Jy")
    print(map_png); print(profile_png); print(output/"summary.json")


if __name__=="__main__":
    main()
