"""Portable paths for the publication analysis package.

All paths are resolved relative to ``publication_code``.  No script in this
directory should depend on a particular user name, working directory, or
absolute filesystem location.
"""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"
MODEL_INPUT_DIR = DATA_ROOT / "model_inputs"
SMITH_GEOMETRY_DIR = DATA_ROOT / "smith_geometry"
OBSERVED_DATA_DIR = DATA_ROOT / "observed"
WHOLE_NEBULA_GEOMETRY = DATA_ROOT / "whole_nebula_geometry" / "geometry_maps.npz"

RESULTS_ROOT = PACKAGE_ROOT / "results"
SINGLE_PHASE_RESULTS = RESULTS_ROOT / "single_phase"
TWO_PHASE_RESULTS = RESULTS_ROOT / "two_phase"
WHOLE_NEBULA_RESULTS = RESULTS_ROOT / "whole_nebula"
HIDDEN_COMPONENT_RESULTS = RESULTS_ROOT / "hidden_component"

FIGURE_DIR = PACKAGE_ROOT / "figures"
