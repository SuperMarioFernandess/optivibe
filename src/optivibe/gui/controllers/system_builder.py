"""Qt-free assembly of an editable system *composition* (task S7-mod §1).

Mirrors :mod:`optivibe.gui.controllers.scenario_builder`: the subsystem forms
collect plain values into a *payload* mapping and this module turns it into the
frozen, validated :class:`~optivibe.core.config.subsystems.SystemConfig` the core
already knows how to resolve into a flat
:class:`~optivibe.core.config.models.VariantConfig` (S9-A). Keeping it Qt-free
means the "did the forms build the right composition?" logic is unit-testable
without a display (task S7-mod §7) and validation stays in one place: a typo, an
out-of-range value or an unregistered reflector shape raises a
``pydantic.ValidationError`` / ``ValueError`` that the worker reports as a failed
run -- no silent fallback (10 §7). The GUI introduces **no new physical
quantity**: every value flows into the existing subsystem models (09 §9).

Two responsibilities
--------------------
* :func:`build_system_config` -- payload (form state) -> ``SystemConfig``.
* :func:`resolve_system_variant` -- ``SystemConfig`` + ``configs/`` -> resolved
  ``VariantConfig`` (this reads subsystem presets off disk and is therefore done
  on the worker thread, never in the UI thread; SW-06).

A helper :func:`subsystem_defaults` reads a preset (no overrides) into a plain
mapping so a form can seed its override fields from the chosen building block.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from optivibe.core.config.presets import PresetStore
from optivibe.core.config.subsystems import (
    CantileverConfig,
    DetectorConfig,
    FiberConfig,
    ReflectorConfig,
    SourceConfig,
    SubsystemRef,
    SystemConfig,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from optivibe.core.config.models import VariantConfig

__all__ = [
    "SUBSYSTEM_MODELS",
    "build_system_config",
    "resolve_system_variant",
    "subsystem_defaults",
    "system_to_payload",
]

#: Subsystem name -> editable model, used to seed forms from a bare preset.
SUBSYSTEM_MODELS: dict[str, type[BaseModel]] = {
    "source": SourceConfig,
    "fiber": FiberConfig,
    "cantilever": CantileverConfig,
    "reflector": ReflectorConfig,
    "detector": DetectorConfig,
}


def build_system_config(payload: Mapping[str, Any]) -> SystemConfig:
    """Validate a composition payload into a :class:`SystemConfig`.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Composition fields: the system scalars (``name``, ``mode``, ``band``,
        ``full_scale_g``, ``q_total``, ...) plus one ``{preset, overrides}``
        block per subsystem (``source``, ``fiber``, ``cantilever``,
        ``reflector``, ``detector``).

    Returns
    -------
    SystemConfig
        The validated, frozen composition.

    Raises
    ------
    pydantic.ValidationError
        If a field is missing or invalid (e.g. a band with ``f_max <= f_min``).
    """
    return SystemConfig.model_validate(dict(payload))


def resolve_system_variant(system: SystemConfig, config_dir: Path) -> VariantConfig:
    """Resolve a composition into the flat variant the stages read (S9-A).

    Reads the named subsystem presets from ``config_dir``, merges the overrides,
    checks the cross-subsystem geometry guards and re-flattens into a
    :class:`~optivibe.core.config.models.VariantConfig`. Because it touches the
    filesystem it is called on the worker thread, never in the UI (SW-06).

    Parameters
    ----------
    system : SystemConfig
        The composition to resolve.
    config_dir : pathlib.Path
        Configuration root holding ``presets/`` (and optionally ``user/``).

    Returns
    -------
    VariantConfig
        The resolved, validated flat variant.

    Raises
    ------
    ValueError
        If a preset is unknown, an override key is invalid, or the composed
        geometry violates a per-shape guard.
    """
    return system.resolve(PresetStore(config_dir))


def subsystem_defaults(store: PresetStore, subsystem: str, preset: str) -> dict[str, Any]:
    """Return the field values of a bare ``preset`` (no overrides) as a mapping.

    Used by a subsystem form to seed its override fields when the user selects a
    different building block.

    Parameters
    ----------
    store : PresetStore
        Preset resolver.
    subsystem : str
        Subsystem name (``"source"``, ``"fiber"``, ...).
    preset : str
        Preset name to read.

    Returns
    -------
    dict[str, Any]
        The validated subsystem model dumped to a plain mapping.

    Raises
    ------
    KeyError
        If ``subsystem`` is not a known subsystem.
    ValueError
        If the preset is unknown or invalid.
    """
    builder = {
        "source": store.build_source,
        "fiber": store.build_fiber,
        "cantilever": store.build_cantilever,
        "reflector": store.build_reflector,
        "detector": store.build_detector,
    }[subsystem]
    model = builder(SubsystemRef(preset=preset))
    return model.model_dump(mode="python")


def system_to_payload(system: SystemConfig) -> dict[str, Any]:
    """Dump a :class:`SystemConfig` to a plain, round-trippable payload mapping.

    Parameters
    ----------
    system : SystemConfig
        The composition to serialise (e.g. a freshly loaded A/B/C/D start).

    Returns
    -------
    dict[str, Any]
        A mapping accepted by :func:`build_system_config`.
    """
    return system.model_dump(mode="python")
