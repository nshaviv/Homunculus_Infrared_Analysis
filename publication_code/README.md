# Homunculus infrared mass analysis: publication code

This directory contains the compact, reproducible analysis set used for the
paper *The Two-Phase Structure of the Homunculus Nebula and its Mass*.  It is
intentionally separate from the exploratory scripts in the repository root.
All paths are relative to this directory; no script depends on a particular
user name or working directory.

## Analysis map

| Paper calculation | Entry point | Frozen result used by the manuscript |
|---|---|---|
| Homogeneous single-phase joint MCMC | `src/single_phase_fit.py` | `results/single_phase/` |
| Final independent-overlap two-phase joint MCMC | `src/two_phase_fit.py` | `results/two_phase/` |
| Single-phase lobe-mass normalization and LOS correction | `src/calculate_single_phase_mass.py` | `results/single_phase_mass.csv` |
| Single- and two-phase fit figures | `src/make_single_phase_figure.py`, `src/make_two_phase_figure.py` | `figures/` |
| Major-axis profile, two-dimensional emission maps, and obscured-component spectrum | `src/make_residual_figures.py` | `figures/` |
| Full two-dimensional nebular emission and spectra | `src/build_whole_nebula_products.py` | `results/whole_nebula/` |
| Obscured-component temperature scan | `src/scan_hidden_component.py` | `results/hidden_component/` |
| Interactive obscured-component experiment | `src/hidden_component_model.py` | user-selected output |

The final two-phase model uses 11 fitted parameters for 11 positive local
flux measurements.  It treats the dense projected covering fraction as the
same on the foreground and rear Area-2 walls, while the two masks are
statistically independent.  No autocorrelation-time or “50-tau” stopping
rule is used.

## Installation

Python 3.11 or newer is recommended.

```bash
cd publication_code
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The LOS projection tools in the repository's `homunculus-IR-smith/` and
`homunculus-los-gui/` directories additionally require Julia.  Their numerical
responses are frozen as constants in the paper-facing Python calculations.

## Reproduce the manuscript products

Run these commands from `publication_code`.

```bash
# Deterministic single-phase masses and approved LOS-to-radial factors.
python3 src/calculate_single_phase_mass.py

# Recompute the two-phase mass summary from the archived posterior.
python3 src/two_phase_fit.py \
  --mass-only --no-show --output-dir results/two_phase

# Rebuild the paper figures from the archived posteriors/intermediate arrays.
python3 src/make_single_phase_figure.py
python3 src/make_two_phase_figure.py
python3 src/make_residual_figures.py

# Rebuild the complete beta=0 and beta=1 two-dimensional forward models.
# This is the slowest deterministic step; outputs are written under rerun/.
python3 src/build_whole_nebula_products.py --betas 0 1

# Validate the release against the numerical values quoted in the paper.
python3 tests/validate_publication_results.py
```

Expected central mass values are:

| Model and local normalization | Before LOS correction | After LOS correction |
|---|---:|---:|
| Single phase, Area 1 | 4.6524 | 1.3781 |
| Single phase, Area 2 | 2.6312 | 1.0897 |
| Two phase, Area 1 posterior median | 8.9420 | 2.6487 |
| Two phase, Area 2 foreground posterior median | 9.8807 | 6.9289 |

All masses in this table are in solar masses and include the adopted
total-to-dust conversion factor of 100.

## Rerun the MCMC fits

The archived posteriors are included so that the figures and tables do not
require rerunning expensive samplers.  New runs are written to `rerun/`
subdirectories and do not overwrite the archived results.

```bash
# Homogeneous five-parameter fit. The saved run used the fixed 18-micron
# optical depths and the original propagated-error prescription.
python3 src/single_phase_fit.py \
  --opacity-cache results/single_phase/opacity_cache.npz \
  --no-show --no-progress

# Final 11-parameter two-phase fit; beta=0 means T proportional to R^(-1/2)
# only in the deterministic temperature-distance relation. Opacity is still
# calculated wavelength by wavelength with Mie theory.
python3 src/two_phase_fit.py --temperature-beta 0 --no-show
```

Random seeds, priors, walker counts, burn-in, and thinning are exposed as
command-line arguments and have deterministic defaults matching the archived
runs.

## Hidden-component calculation

The added component is not an MCMC fit.  At each fixed temperature the code:

1. sets the projected area so the model closes the 100-micron deficit;
2. fits the foreground optical depth to the 5--7 micron spectrum;
3. varies the hot-component optical depth to minimize the combined dust mass.

The transfer term is

```text
Omega * B_nu(T_hot) * exp[-tau_foreground(nu)]
      * (1 - exp[-tau_hot(nu)]).
```

To reproduce only the 800 K row used in the paper:

```bash
python3 src/scan_hidden_component.py \
  --temperature-min-K 800 --temperature-max-K 800
```

The archived 800 K solution has `tau_h,100 = 1.181504`,
`tau_f,100 = 0.671474`, projected area `2.397316 arcsec^2`, hot dust mass
`0.0339066 Msun`, and hot-plus-foreground dust mass `0.0531764 Msun`.

## Important conventions

- `Sigma_LOS` enters the radiative-transfer likelihood because the measured
  optical depth is along the observer's sight line.
- `Sigma_radial` enters the mass normalization and radial physical relations.
- The deterministic conversion is applied exactly once.
- Area 2 contains an 80-degree foreground crossing and a 28.4-degree rear
  crossing.  Only the foreground radial column normalizes the Area-2 lobe
  mass; the rear crossing is not added as a second Homunculus.
- `beta=0` in the archived two-phase run refers only to the fixed
  temperature-distance law.  It does **not** replace the Mie opacity with a
  blackbody opacity.
- The two aperture normalizations are retained independently.  Their
  discrepancy is a diagnostic, not removed with an equality constraint.

## Repository hygiene

- Files under `results/` are the frozen numerical products used in the
  manuscript.
- New sampler runs should go into the existing `rerun/` locations.
- Do not commit `.venv`, `__pycache__`, notebook checkpoints, or temporary
  Matplotlib caches.
- No file here exceeds GitHub's 100 MB per-file limit; Git LFS is not required
  for the present release.
- A software license has not been selected in this package.  The repository
  owners should choose one before a public release.

See `data/README.md` for data provenance and redistribution notes.
