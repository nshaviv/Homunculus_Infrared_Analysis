"""Configurable dust temperature--distance scaling for sensitivity tests.

For an effective absorption/emission efficiency proportional to frequency
raised to ``beta``, radiative equilibrium gives

    T(R) = T_ref * (R_ref / R)**[2 / (4 + beta)].

``beta`` is used only in this deterministic temperature relation.  The MCMC
continues to calculate its wavelength-dependent dust opacity with Mie theory.
The default is beta=0, reproducing the original square-root distance law.
"""

from __future__ import annotations

import os

import numpy as np


DEFAULT_EMISSIVITY_BETA = 0.0
ENVIRONMENT_VARIABLE = "ETA_TEMPERATURE_BETA"


def validate_emissivity_beta(beta):
    """Return a finite beta for which the equilibrium exponent is defined."""
    value = float(beta)
    if not np.isfinite(value):
        raise ValueError("Temperature emissivity beta must be finite.")
    if value <= -4.0:
        raise ValueError("Temperature emissivity beta must be greater than -4.")
    return value


def emissivity_beta_from_environment():
    """Read beta from the environment, defaulting to the original beta=0."""
    return validate_emissivity_beta(
        os.environ.get(ENVIRONMENT_VARIABLE, DEFAULT_EMISSIVITY_BETA)
    )


def temperature_distance_exponent(beta):
    """Return alpha=2/(4+beta) in T proportional to R**(-alpha)."""
    value = validate_emissivity_beta(beta)
    return 2.0 / (4.0 + value)


def temperature_scale(reference_radius, target_radius, beta):
    """Return T(target)/T(reference) for the requested beta."""
    reference = np.asarray(reference_radius, dtype=float)
    target = np.asarray(target_radius, dtype=float)
    if np.any(~np.isfinite(reference)) or np.any(~np.isfinite(target)):
        raise ValueError("Temperature-scaling radii must be finite.")
    if np.any(reference <= 0.0) or np.any(target <= 0.0):
        raise ValueError("Temperature-scaling radii must be strictly positive.")
    return (reference / target) ** temperature_distance_exponent(beta)


def scale_temperature(reference_temperature, reference_radius, target_radius, beta):
    """Scale a reference temperature to a target radius."""
    return np.asarray(reference_temperature, dtype=float) * temperature_scale(
        reference_radius, target_radius, beta
    )
