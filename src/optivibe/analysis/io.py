"""Analysis I/O: spec loading and result persistence (task S6 §B9).

Loads ``optivibe sweep`` specs (sweep or Monte-Carlo, dispatched on the ``kind``
field) and persists results as ``.npz`` (the always-available format; Parquet is
optional behind pandas, 11 §5). Figures are written by the caller through
``viz.analysis`` -- this module touches data only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from optivibe.analysis.monte_carlo import MonteCarloResult
from optivibe.analysis.spec import MonteCarloSpec, SweepSpec
from optivibe.analysis.sweep import SweepResult

__all__ = [
    "load_analysis_spec",
    "save_monte_carlo_npz",
    "save_sweep_npz",
]


def load_analysis_spec(path: Path | str) -> SweepSpec | MonteCarloSpec:
    """Load and validate a sweep or Monte-Carlo spec from YAML (S6 §B9).

    Parameters
    ----------
    path : pathlib.Path or str
        Path to a spec YAML file with a ``kind`` of ``"sweep"`` or
        ``"montecarlo"``.

    Returns
    -------
    SweepSpec or MonteCarloSpec
        The validated spec.

    Raises
    ------
    ValueError
        If the ``kind`` field is missing or unrecognised.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    kind = raw.get("kind")
    if kind == "sweep":
        return SweepSpec.model_validate(raw)
    if kind == "montecarlo":
        return MonteCarloSpec.model_validate(raw)
    msg = f"analysis spec needs kind in {{'sweep', 'montecarlo'}}, got {kind!r}"
    raise ValueError(msg)


def save_sweep_npz(result: SweepResult, path: Path | str) -> Path:
    """Persist a sweep result to a compressed ``.npz`` file.

    Parameters
    ----------
    result : SweepResult
        The sweep result.
    path : pathlib.Path or str
        Output path (``.npz`` suffix added if missing).

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    out = Path(path).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "axis_values": result.axis_values,
        "axis_labels": np.asarray(result.axis_labels, dtype=object),
        "_header": np.asarray(
            json.dumps(
                {
                    "name": result.name,
                    "mode": result.mode,
                    "parameter": result.parameter,
                    "variant": result.variant,
                    "meta": result.meta,
                }
            )
        ),
    }
    for key, values in result.metrics.items():
        payload[f"metric__{key}"] = values
    np.savez_compressed(out, **payload)
    return out


def save_monte_carlo_npz(result: MonteCarloResult, path: Path | str) -> Path:
    """Persist a Monte-Carlo result to a compressed ``.npz`` file.

    Parameters
    ----------
    result : MonteCarloResult
        The Monte-Carlo result.
    path : pathlib.Path or str
        Output path (``.npz`` suffix added if missing).

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    out = Path(path).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "_header": np.asarray(
            json.dumps(
                {
                    "name": result.name,
                    "variant": result.variant,
                    "n_draws": result.n_draws,
                    "seed": result.seed,
                    "tolerances": result.tolerances,
                    "stats": result.stats,
                    "meta": result.meta,
                }
            )
        ),
    }
    for key, values in result.samples.items():
        payload[f"sample__{key}"] = values
    np.savez_compressed(out, **payload)
    return out
