"""Qt-free assembly of run/analysis configs from GUI selections (task S7 §2).

The control widgets collect plain values into *payload* mappings; this module
turns them into the frozen, validated pydantic models the core consumes
(:class:`~optivibe.core.config.models.ScenarioConfig` and the analysis
:class:`~optivibe.analysis.SweepSpec` / :class:`~optivibe.analysis.MonteCarloSpec`).
Keeping it Qt-free means the "did the panel build the right scenario?" logic is
unit-testable without a display (task S7 §7), and validation stays in one place:
a typo or out-of-range value raises a ``pydantic.ValidationError`` (a
``ValueError``) that the worker reports as a failed run -- no silent fallback
(10 §7). The GUI introduces **no new physical quantity**: every value flows into
the existing config models (09 §9).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter

from optivibe.analysis import MonteCarloSpec, SweepSpec
from optivibe.core.config.models import ExcitationSpec, ScenarioConfig

__all__ = [
    "build_excitation_spec",
    "build_monte_carlo_spec",
    "build_scenario_config",
    "build_sweep_spec",
    "demo_scenario_payload",
]

_EXCITATION_ADAPTER: TypeAdapter[ExcitationSpec] = TypeAdapter(ExcitationSpec)


def build_excitation_spec(payload: Mapping[str, Any]) -> ExcitationSpec:
    """Validate an excitation payload into a discriminated :data:`ExcitationSpec`.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Excitation fields including the ``kind`` discriminator (``sine``,
        ``multitone``, ``sweep``, ``random``, ``shock``, ``csv`` or ``wav``).

    Returns
    -------
    ExcitationSpec
        The validated spec (one member of the S1 union).

    Raises
    ------
    pydantic.ValidationError
        If the payload is invalid for its ``kind``.
    """
    return _EXCITATION_ADAPTER.validate_python(dict(payload))


def build_scenario_config(payload: Mapping[str, Any]) -> ScenarioConfig:
    """Validate a scenario payload into a :class:`ScenarioConfig`.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Scenario fields (``name``, ``variant``, ``excitation``, ``stages``,
        ``detector``, ``dsp``, ``mechanics``, ``seed``).

    Returns
    -------
    ScenarioConfig
        The validated, frozen scenario.

    Raises
    ------
    pydantic.ValidationError
        If any field is missing or invalid.
    """
    return ScenarioConfig.model_validate(dict(payload))


def build_sweep_spec(payload: Mapping[str, Any]) -> SweepSpec:
    """Validate a sweep payload into an analysis :class:`SweepSpec`.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Sweep fields (``name``, ``mode``, ``variant``, ``parameter``, ``grid``,
        tone context and stage options).

    Returns
    -------
    SweepSpec
        The validated sweep specification.

    Raises
    ------
    pydantic.ValidationError
        If the spec is invalid (e.g. a response parameter under ``design`` mode).
    """
    return SweepSpec.model_validate(dict(payload))


def build_monte_carlo_spec(payload: Mapping[str, Any]) -> MonteCarloSpec:
    """Validate a Monte-Carlo payload into an analysis :class:`MonteCarloSpec`.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Monte-Carlo fields (``name``, ``variant``, ``n_draws``, ``tolerances``,
        ``cross_axis`` and tone context).

    Returns
    -------
    MonteCarloSpec
        The validated Monte-Carlo specification.

    Raises
    ------
    pydantic.ValidationError
        If a tolerance key or distribution is invalid.
    """
    return MonteCarloSpec.model_validate(dict(payload))


def demo_scenario_payload() -> dict[str, Any]:
    """Return a known-good demo scenario payload (the ``recover_sine`` preset).

    Variant B, a 1 g / 200 Hz tone through ``modal -> cylinder -> photodiode``
    and the ``standard`` calibrated inverse, so the live tab shows a faithful
    recovery and a non-empty NEA budget out of the box (task S7 §2).

    Returns
    -------
    dict[str, Any]
        A payload accepted by :func:`build_scenario_config`.
    """
    return {
        "name": "gui-demo",
        "variant": "B",
        "excitation": {
            "kind": "sine",
            "axis": "x",
            "fs_hz": 5000.0,
            "duration_s": 2.0,
            "frequency_hz": 200.0,
            "amplitude_g": 1.0,
        },
        "stages": {
            "excitation": "sine",
            "mechanics": "modal",
            "optics": "cylinder",
            "detector": "photodiode",
            "dsp": "standard",
        },
        "detector": {"balanced": True, "reference_arm": "matched"},
        "dsp": {
            "integrator": "frequency",
            "spectrum_method": "fft",
            "window": "hann",
            "sensitivity_model": "static",
            "sensitivity_freq": "plateau",
        },
        "seed": 7,
    }
