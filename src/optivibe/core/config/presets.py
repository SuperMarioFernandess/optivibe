"""Preset store and composition persistence for the subsystem layer (S9-A).

This module is the *resolver back-end* for :class:`optivibe.core.config.subsystems.SystemConfig`.
It locates named subsystem presets on disk, merges per-subsystem overrides on
top of them, validates the result against the editable subsystem models, and
provides save/load helpers for whole compositions.

Directory layout (doc 09 §7; task S9-A §3)
------------------------------------------
Two tiers are scanned for every subsystem ``<sub>`` in
``{source, fiber, cantilever, reflector, detector}``::

    <config_dir>/presets/<sub>/*.yaml        # built-in building blocks
    <config_dir>/user/presets/<sub>/*.yaml   # user-defined building blocks

User compositions are stored under ``<config_dir>/user/systems/<name>.yaml``.

Name-collision policy (task S9-A §3)
------------------------------------
* **Same tier.** Two preset files resolving to the *same* stem in one tier
  (e.g. ``sld.yaml`` and ``sld.yml``) are a configuration error and raise
  loudly -- there is no defined winner.
* **Across tiers.** A user preset of the same name as a built-in **wins**
  (explicit, documented priority); the shadowing is reported at debug level so
  it is discoverable without being noisy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from pydantic import BaseModel

from optivibe.core.config.subsystems import (
    CantileverConfig,
    DetectorConfig,
    FiberConfig,
    ReflectorConfig,
    SourceConfig,
    SubsystemRef,
)
from optivibe.core.logging import get_logger

if TYPE_CHECKING:
    from optivibe.core.config.subsystems import SystemConfig

logger = get_logger(__name__)

#: Subsystem model type resolved by :meth:`PresetStore._build`.
_SubsystemT = TypeVar("_SubsystemT", bound=BaseModel)

# Recognised preset file extensions (both map to the same stem namespace).
_PRESET_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path``.

    Parameters
    ----------
    path : pathlib.Path
        File to read.

    Returns
    -------
    dict
        Parsed top-level mapping.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the top-level YAML document is not a mapping.
    """
    if not path.is_file():
        msg = f"preset file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}"
        raise ValueError(msg)
    return data


class PresetStore:
    """Resolver from ``(subsystem, preset name)`` to a validated subsystem model.

    The store knows the two preset tiers (built-in and user) under a single
    ``configs/`` directory and applies the documented collision policy. It does
    not cache the directory scan, so a preset saved during the same process is
    visible to the next :meth:`build_source`/... call.

    Parameters
    ----------
    config_dir : pathlib.Path
        Root configuration directory (the one that contains ``presets/`` and,
        optionally, ``user/``).
    """

    def __init__(self, config_dir: Path) -> None:
        self._builtin_dir = config_dir / "presets"
        self._user_dir = config_dir / "user" / "presets"

    # ------------------------------------------------------------------ #
    # Public per-subsystem builders (called by SystemConfig.resolve).
    # ------------------------------------------------------------------ #
    def build_source(self, ref: SubsystemRef) -> SourceConfig:
        """Build the :class:`SourceConfig` named by ``ref`` with its overrides."""
        return self._build(ref, "source", SourceConfig)

    def build_fiber(self, ref: SubsystemRef) -> FiberConfig:
        """Build the :class:`FiberConfig` named by ``ref`` with its overrides."""
        return self._build(ref, "fiber", FiberConfig)

    def build_cantilever(self, ref: SubsystemRef) -> CantileverConfig:
        """Build the :class:`CantileverConfig` named by ``ref`` with its overrides."""
        return self._build(ref, "cantilever", CantileverConfig)

    def build_reflector(self, ref: SubsystemRef) -> ReflectorConfig:
        """Build the :class:`ReflectorConfig` named by ``ref`` with its overrides."""
        return self._build(ref, "reflector", ReflectorConfig)

    def build_detector(self, ref: SubsystemRef) -> DetectorConfig:
        """Build the :class:`DetectorConfig` named by ``ref`` with its overrides."""
        return self._build(ref, "detector", DetectorConfig)

    # ------------------------------------------------------------------ #
    # Discovery.
    # ------------------------------------------------------------------ #
    def list_presets(self, subsystem: str) -> dict[str, Path]:
        """List the resolvable presets for ``subsystem`` (user shadowing applied).

        Parameters
        ----------
        subsystem : str
            Subsystem folder name (``"source"``, ``"fiber"``, ...).

        Returns
        -------
        dict
            Mapping of preset name to the file that would be loaded.
        """
        builtin = self._scan_tier(self._builtin_dir / subsystem)
        user = self._scan_tier(self._user_dir / subsystem)
        resolved = dict(builtin)
        for name, path in user.items():
            if name in resolved:
                logger.debug(
                    "user preset %r/%r shadows built-in %s", subsystem, name, resolved[name]
                )
            resolved[name] = path
        return resolved

    # ------------------------------------------------------------------ #
    # Internals.
    # ------------------------------------------------------------------ #
    def _build(
        self, ref: SubsystemRef, subsystem: str, model_cls: type[_SubsystemT]
    ) -> _SubsystemT:
        """Load ``ref.preset`` for ``subsystem`` and merge ``ref.overrides``.

        The overrides are merged at the top field level (a nested block such as
        ``cantilever.material`` is replaced wholesale, not deep-merged). Unknown
        keys are rejected by the subsystem model's ``extra="forbid"`` (10 §7).
        """
        path = self._resolve_path(subsystem, ref.preset)
        data = _read_yaml_mapping(path)
        merged: dict[str, Any] = {**data, **ref.overrides}
        try:
            return model_cls.model_validate(merged)
        except ValueError as exc:
            msg = f"invalid {subsystem} preset {ref.preset!r} ({path}) after overrides: {exc}"
            raise ValueError(msg) from exc

    def _resolve_path(self, subsystem: str, name: str) -> Path:
        """Resolve ``name`` to a preset file (user tier wins over built-in)."""
        user = self._scan_tier(self._user_dir / subsystem)
        if name in user:
            return user[name]
        builtin = self._scan_tier(self._builtin_dir / subsystem)
        if name in builtin:
            return builtin[name]
        available = sorted({*builtin, *user})
        listing = ", ".join(available) if available else "<none>"
        msg = f"unknown {subsystem} preset {name!r}; available: {listing}"
        raise ValueError(msg)

    @staticmethod
    def _scan_tier(directory: Path) -> dict[str, Path]:
        """Scan one tier directory, mapping preset stem to file path.

        Parameters
        ----------
        directory : pathlib.Path
            A ``<tier>/<subsystem>`` directory (may be absent).

        Returns
        -------
        dict
            Mapping of file stem to path; empty if the directory is absent.

        Raises
        ------
        ValueError
            If two files in this tier resolve to the same stem.
        """
        if not directory.is_dir():
            return {}
        found: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if path.suffix not in _PRESET_SUFFIXES or not path.is_file():
                continue
            stem = path.stem
            if stem in found:
                msg = (
                    f"duplicate preset name {stem!r} in {directory}: "
                    f"{found[stem].name} and {path.name}"
                )
                raise ValueError(msg)
            found[stem] = path
        return found


# --------------------------------------------------------------------------- #
# Composition persistence (save/load whole SystemConfig documents).
# --------------------------------------------------------------------------- #
def load_system_file(path: Path) -> SystemConfig:
    """Load and validate a composed :class:`SystemConfig` from YAML.

    Parameters
    ----------
    path : pathlib.Path
        Path to a composition file (a ``SystemConfig`` document).

    Returns
    -------
    SystemConfig
        Validated composition.
    """
    from optivibe.core.config.subsystems import SystemConfig

    logger.debug("loading system composition from %s", path)
    return SystemConfig.model_validate(_read_yaml_mapping(path))


def save_system_config(system: SystemConfig, path: Path) -> Path:
    """Serialise a :class:`SystemConfig` to YAML (round-trippable).

    The dump is exact: ``load_system_file(save_system_config(s, p))`` reproduces
    ``s`` and therefore the same resolved :class:`VariantConfig`. Parent
    directories are created if missing.

    Parameters
    ----------
    system : SystemConfig
        Composition to persist.
    path : pathlib.Path
        Destination file (typically ``configs/user/systems/<name>.yaml``).

    Returns
    -------
    pathlib.Path
        The path written (echoed for convenience).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = system.model_dump(mode="python")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    logger.debug("saved system composition %r to %s", system.name, path)
    return path
