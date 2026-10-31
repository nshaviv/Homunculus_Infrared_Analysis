# Archived manuscript results

These files are frozen numerical products used by the manuscript figures and
tables.  They are not additional observational data.

- `single_phase/`: posterior samples, parameter summary, and cached Mie
  opacity grid for the homogeneous five-parameter joint fit.
- `two_phase/`: posterior samples and derived summaries for the final
  independent-overlap, density-dependent-grain joint fit.
- `whole_nebula/beta_0/`: full mapped products used for the integrated
  major-axis profile.
- `whole_nebula/beta_1/`: full mapped products used for the two-dimensional
  emission comparison.
- `whole_nebula/parameters/`: the matching beta=0 and beta=1 MCMC parameter
  summaries used by the forward-model generator.
- `hidden_component/`: deterministic temperature scan and selected 800 K
  solution.

The `rerun/` outputs created by new MCMC and forward-model runs are
intentionally separate from these archived products and ignored by Git.
