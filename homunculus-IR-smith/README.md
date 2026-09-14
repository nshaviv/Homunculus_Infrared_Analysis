# Thin-shell Homunculus column-density model

Run the first-stage model with:

```sh
julia thin_shell_column_density.jl output
```

Inputs are `Mass-per-SolidAngle.csv` and `v_vs_theta.csv`. The model interprets
their 0--90 degree angle as latitude from the equator, mirrors it across the
equator, normalizes the supplied
mass distribution to one solar mass, and uses the velocity curve with a
ballistic age of 152.9 yr (1847.1 to 2000.0) to set shell radii.

Results are written to `output/`:

- `thin_shell_column_density.ppm`: false-colour projected map.
- `thin_shell_column_density_g_cm2.csv`: numeric column density, in g cm^-2.
- `model_metadata.txt`: geometric and display values.

The current model is an infinitesimal, axisymmetric shell. It does not include
shell thickness, clumping, the equatorial skirt, extinction, or departures from
homologous expansion.

## STL finite-width model

Run the non-axisymmetric STL model with:

```sh
julia stl_column_density.jl output 41.0
```

The optional second argument is the polar-axis inclination from the line of
sight in degrees. The STL polar radius is scaled to the Smith velocity-law pole
at epoch 2000, and 101 homologous layers represent a uniform 20-year ejection.
The combined mass is one solar mass, distributed using the smoothed
`Mass-per-SolidAngle.csv` latitude profile.

- `output/stl_finite_duration_column_density.png`: rendered total column map.
- `output/stl_finite_duration_column_density_g_cm2.csv`: numeric map in g cm^-2.
- `output/stl_sky_axis_au.csv`: sky-coordinate pixel centres in AU, used for
  both CSV dimensions.
- `output/stl_model_metadata.txt`: scaling and mass-conservation diagnostics.
