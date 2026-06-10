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
from optivibe.core.logging import get_logger

logger = get_logger(__name__)

_ENV_VAR = "OPTIVIBE_CONFIG_DIR"
_VALID_VARIANTS = ("A", "B", "C", "D")


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


def load_variant_file(path: Path) -> VariantConfig:
    """Load and validate a single variant file.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``variants/{name}.yaml`` file.

    Returns
    -------
    VariantConfig
        Validated variant configuration.
    """
    logger.debug("loading variant from %s", path)
    return VariantConfig.model_validate(_read_yaml(path))


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
        Validated variant configuration.

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
    return load_variant_file(config_dir / "variants" / f"{name}.yaml")


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
