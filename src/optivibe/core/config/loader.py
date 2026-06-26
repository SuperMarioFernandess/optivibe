"""Loading and validation of YAML configuration files.

Functions here read ``configs/constants.yaml``, ``configs/variants/{A,B,C,D}.yaml``
and scenario files, and validate them with the pydantic models of
:mod:`optivibe.core.config.models`. The directory ``configs/`` lives at the
repository root (sibling of ``src/``); its location is resolved by
:func:`default_config_dir`, overridable with the ``OPTIVIBE_CONFIG_DIR``
environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from optivibe.core.config.models import Constants, ScenarioConfig, VariantConfig
from optivibe.core.config.presets import PresetStore
from optivibe.core.config.subsystems import SystemConfig
from optivibe.core.logging import get_logger

logger = get_logger(__name__)

_ENV_VAR = "OPTIVIBE_CONFIG_DIR"
_VALID_VARIANTS = ("A", "B", "C", "D")

# Keys that mark a YAML document as a composed SystemConfig rather than a flat
# VariantConfig. A composition references subsystem presets (``preset`` key) and
# carries a ``cantilever`` block, which the flat variant never has.
_COMPOSITION_MARKERS: tuple[str, ...] = ("cantilever",)


def _is_composition(data: dict[str, Any]) -> bool:
    """Return ``True`` if ``data`` is a composed :class:`SystemConfig` document.

    A composition is detected either by a top-level ``cantilever`` block (absent
    from the flat variant) or by a ``source`` mapping that references a preset
    (``source.preset``) rather than inlining the source fields.

    Parameters
    ----------
    data : dict
        Parsed YAML mapping.

    Returns
    -------
    bool
        Whether the document should be parsed as a composition.
    """
    if any(marker in data for marker in _COMPOSITION_MARKERS):
        return True
    source = data.get("source")
    return isinstance(source, dict) and "preset" in source


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path``.

    Parameters
    ----------
    path : pathlib.Path
        File to read.

    Returns
    -------
    dict
        Parsed mapping.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the top-level YAML document is not a mapping.
    """
    if not path.is_file():
        msg = f"configuration file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def default_config_dir() -> Path:
    """Resolve the repository ``configs/`` directory.

    Resolution order: the ``OPTIVIBE_CONFIG_DIR`` environment variable, then the
    repository layout (``<repo>/configs`` relative to this file), then a search
    upward from the current working directory.

    Returns
    -------
    pathlib.Path
        Existing configuration directory.

    Raises
    ------
    FileNotFoundError
        If no ``configs/`` directory can be located.
    """
    env = os.environ.get(_ENV_VAR)
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir():
            return candidate
        msg = f"{_ENV_VAR}={env!r} is not a directory"
        raise FileNotFoundError(msg)

    repo_candidate = Path(__file__).resolve().parents[3].parent / "configs"
    if repo_candidate.is_dir():
        return repo_candidate

    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / "configs"
        if candidate.is_dir():
            return candidate

    msg = "could not locate a 'configs' directory; set OPTIVIBE_CONFIG_DIR"
    raise FileNotFoundError(msg)


def load_constants(path: Path | None = None) -> Constants:
    """Load and validate the physical constants (doc 01).

    Parameters
    ----------
    path : pathlib.Path or None, optional
        Explicit path to ``constants.yaml``; defaults to
        ``default_config_dir() / "constants.yaml"``.

    Returns
    -------
    Constants
        Validated constants bundle.
    """
    if path is None:
        path = default_config_dir() / "constants.yaml"
    logger.debug("loading constants from %s", path)
    return Constants.model_validate(_read_yaml(path))


def load_variant_file(path: Path, config_dir: Path | None = None) -> VariantConfig:
    """Load and validate a single variant file (flat or composed).

    The file may be a legacy flat ``VariantConfig`` or a composed
    :class:`~optivibe.core.config.subsystems.SystemConfig` (subsystem presets +
    overrides). Compositions are resolved to the same flat ``VariantConfig`` the
    stages read, so the return type is identical in both cases.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``variants/{name}.yaml`` file.
    config_dir : pathlib.Path or None, optional
        Configuration directory used to locate subsystem presets when the file
        is a composition. Defaults to the file's grandparent (``<dir>/variants``
        -> ``<dir>``) when it holds a ``presets/`` tree, else
        :func:`default_config_dir`.

    Returns
    -------
    VariantConfig
        Validated (and, for compositions, resolved) variant configuration.
    """
    logger.debug("loading variant from %s", path)
    data = _read_yaml(path)
    if not _is_composition(data):
        return VariantConfig.model_validate(data)
    resolved_dir = _resolve_preset_root(path, config_dir)
    system = SystemConfig.model_validate(data)
    return system.resolve(PresetStore(resolved_dir))


def _resolve_preset_root(variant_path: Path, config_dir: Path | None) -> Path:
    """Pick the ``configs/`` root that holds the preset tiers for a composition.

    Parameters
    ----------
    variant_path : pathlib.Path
        Path of the composition file (under ``<config_dir>/variants/``).
    config_dir : pathlib.Path or None
        Explicit override; returned as-is when given.

    Returns
    -------
    pathlib.Path
        Directory expected to contain ``presets/`` and ``user/``.
    """
    if config_dir is not None:
        return config_dir
    inferred = variant_path.resolve().parent.parent
    if (inferred / "presets").is_dir():
        return inferred
    return default_config_dir()


def load_variant(name: str, config_dir: Path | None = None) -> VariantConfig:
    """Load a variant preset by name (``"A"``..``"D"``).

    Parameters
    ----------
    name : str
        Variant identifier, one of ``"A"``, ``"B"``, ``"C"``, ``"D"``.
    config_dir : pathlib.Path or None, optional
        Configuration directory; defaults to :func:`default_config_dir`.

    Returns
    -------
    VariantConfig
        Validated variant configuration (resolved if stored as a composition).

    Raises
    ------
    ValueError
        If ``name`` is not a known variant.
    """
    if name not in _VALID_VARIANTS:
        msg = f"unknown variant {name!r}; expected one of {_VALID_VARIANTS}"
        raise ValueError(msg)
    if config_dir is None:
        config_dir = default_config_dir()
    return load_variant_file(config_dir / "variants" / f"{name}.yaml", config_dir)


def load_scenario(path: Path) -> ScenarioConfig:
    """Load and validate a scenario file.

    Parameters
    ----------
    path : pathlib.Path
        Path to a scenario YAML file.

    Returns
    -------
    ScenarioConfig
        Validated scenario configuration.
    """
    logger.debug("loading scenario from %s", path)
    return ScenarioConfig.model_validate(_read_yaml(path))
