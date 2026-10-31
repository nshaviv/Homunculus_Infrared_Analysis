# Input data and provenance

| Directory/file | Role | Scientific source |
|---|---|---|
| `model_inputs/lobe.edge.data` | Area 1 local infrared spectrum | Aperture photometry reported by Smith et al. (2003) |
| `model_inputs/lobeedgedata2` | Area 2 local infrared spectrum | Aperture photometry reported by Smith et al. (2003) |
| `model_inputs/eps_Sil` | Astronomical-silicate optical constants | Draine & Lee (1984); Laor & Draine (1993) |
| `model_inputs/corundumAl203data` | Temperature-dependent corundum optical constants | Zeidler et al. (2013), Jena laboratory database |
| `model_inputs/iron_ordaldata` | Metallic-iron optical constants | Ordal et al. (1985) |
| `smith_geometry/Mass-per-SolidAngle.csv` | Relative mass per solid angle versus latitude | Smith (2006) |
| `smith_geometry/v_vs_theta.csv` | Expansion velocity versus latitude | Smith (2006) |
| `observed/i18_map.npz` | Reconstructed 18-micron map used for the spatial comparison | Digitized/reconstructed from Smith et al. (2003) |
| `observed/i18_major_axis_profile.csv` | Minor-axis-integrated 18-micron profile | Derived from the reconstructed map |
| `observed/homunculus_obs_flux_cgs.csv` | Integrated Homunculus infrared spectrum | Digitized literature spectrum described in the manuscript |
| `observed/hidden_component_area2_beta1_input.csv` | Frozen observed/baseline/opacity table for the obscured-component calculation | Derived analysis product |
| `whole_nebula_geometry/geometry_maps.npz` | Projected front/rear shell crossings, radii, latitudes, and LOS-to-radial ratios | Derived from the finite-duration Smith (2006) geometry |
| `whole_nebula_geometry/geometry_metadata.txt` | Grid and physical conventions for the compressed geometry maps | Derived analysis metadata |

The numeric files are included to make the archived paper results exactly
reproducible.  Before making the repository public, verify that the original
optical-constant tables and digitized literature data may be redistributed.
If redistribution is restricted, replace those files with a documented
download-and-preparation script plus checksums; do not silently substitute a
different optical-constant release.

The source papers are cited in the manuscript and should also be cited by
users of these inputs.

The compressed whole-nebula geometry is included so the complete
two-dimensional forward model can be rerun without an absolute path to the
original CSV export.
