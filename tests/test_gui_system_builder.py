"""Qt-free unit tests for the editable composition seam (task S7-mod §7).

These import only the *Qt-free* layer (``system_builder``, ``jobs``,
``scenario_builder`` and the config models do not pull in PySide6), so they run
without a display or the ``gui`` extra. They prove that a composition payload
assembles and resolves to the right variant (bit-identical to the named A/B/C/D
for an unedited start), that subsystem overrides and the reflector *shape*
selection flow through correctly, that an *edited* composition runs through the
worker job, that compositions round-trip through save/load, and that the dynamic
multitone produces an arbitrary-N spec the frozen core accepts unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from optivibe.core.config.loader import load_variant
from optivibe.core.config.models import MultitoneSpec, ScenarioConfig
from optivibe.core.config.presets import PresetStore, load_system_file, save_system_config
from optivibe.gui.controllers.scenario_builder import build_excitation_spec
from optivibe.gui.controllers.system_builder import (
    build_system_config,
    resolve_system_variant,
    subsystem_defaults,
    system_to_payload,
)
from optivibe.gui.workers.jobs import ScenarioJob
from optivibe.pipeline import RunArtifacts

_NOOP = lambda *_a, **_k: None  # noqa: E731 - tiny test stub
_NO_CANCEL = lambda: False  # noqa: E731 - tiny test stub


def _starting_payload(config_dir: Path, key: str) -> dict:
    """Load an A/B/C/D starting composition as an editable payload."""
    return system_to_payload(load_system_file(config_dir / "variants" / f"{key}.yaml"))


def test_unedited_composition_resolves_bit_identically(config_dir: Path) -> None:
    """A B start (no edits) resolves to exactly the named variant B."""
    payload = _starting_payload(config_dir, "B")
    system = build_system_config(payload)
    resolved = resolve_system_variant(system, config_dir)
    assert resolved.model_dump() == load_variant("B", config_dir=config_dir).model_dump()


def test_subsystem_override_changes_resolved_variant(config_dir: Path) -> None:
    """Overriding the cantilever length changes the resolved variant length."""
    payload = _starting_payload(config_dir, "B")
    payload["cantilever"]["overrides"]["length_m"] = 1.5e-3
    resolved = resolve_system_variant(build_system_config(payload), config_dir)
    assert resolved.length_m == pytest.approx(1.5e-3)


def test_reflector_shape_selection(config_dir: Path) -> None:
    """Switching the reflector preset/shape resolves to that optics shape."""
    payload = _starting_payload(config_dir, "B")
    payload["reflector"] = {"preset": "plane_flat", "overrides": {}}
    resolved = resolve_system_variant(build_system_config(payload), config_dir)
    assert resolved.reflector.shape == "plane"
    assert resolved.reflector.radius_of_curvature_m is None


def test_plane_with_stale_curvature_override_is_cleared(config_dir: Path) -> None:
    """A plane override sending ``curvature_radius_m=None`` is accepted."""
    payload = _starting_payload(config_dir, "B")
    payload["reflector"] = {
        "preset": "cyl_rc31",
        "overrides": {"shape": "plane", "curvature_radius_m": None, "wedge_angle_rad": None},
    }
    resolved = resolve_system_variant(build_system_config(payload), config_dir)
    assert resolved.reflector.shape == "plane"


def test_invalid_composition_raises(config_dir: Path) -> None:
    """A curved shape without a curvature radius fails loudly (10 §7)."""
    payload = _starting_payload(config_dir, "B")
    payload["reflector"] = {
        "preset": "cyl_rc31",
        "overrides": {"shape": "sphere", "curvature_radius_m": None},
    }
    with pytest.raises(ValueError):
        resolve_system_variant(build_system_config(payload), config_dir)


def test_subsystem_defaults_reads_preset(config_dir: Path) -> None:
    """``subsystem_defaults`` returns the bare preset field values."""
    store = PresetStore(config_dir)
    values = subsystem_defaults(store, "reflector", "cyl_rc31")
    assert values["shape"] == "cylinder"
    assert values["curvature_radius_m"] == pytest.approx(31.0e-6)


def test_save_load_round_trip(config_dir: Path, tmp_path: Path) -> None:
    """A composition saved and reloaded resolves to the same variant."""
    payload = _starting_payload(config_dir, "C")
    system = build_system_config(payload)
    path = save_system_config(system, tmp_path / "mine.yaml")
    reloaded = load_system_file(path)
    a = resolve_system_variant(system, config_dir).model_dump()
    b = resolve_system_variant(reloaded, config_dir).model_dump()
    assert a == b


def test_edited_composition_runs_through_job(config_dir: Path) -> None:
    """A ScenarioJob with an edited composition runs the core off-thread-ready."""
    payload = _starting_payload(config_dir, "B")
    payload["reflector"] = {"preset": "sphere_rc62", "overrides": {}}
    system = build_system_config(payload)
    scenario = ScenarioConfig.model_validate(
        {
            "name": "edited",
            "variant": "B",
            "excitation": {
                "kind": "sine",
                "axis": "x",
                "fs_hz": 5000.0,
                "duration_s": 1.0,
                "frequency_hz": 120.0,
                "amplitude_g": 1.0,
            },
            "stages": {
                "excitation": "sine",
                "mechanics": "modal",
                "optics": "cylinder",
                "detector": "stub",
                "dsp": "stub",
            },
        }
    )
    job = ScenarioJob(scenario=scenario, config_dir=config_dir, system=system)
    artifacts = job.run(progress=_NOOP, is_cancelled=_NO_CANCEL)
    assert isinstance(artifacts, RunArtifacts)
    assert artifacts.variant.reflector.shape == "sphere"
    assert artifacts.result.dominant_freqs_hz[0] == pytest.approx(120.0, abs=1.0)


def test_multitone_accepts_arbitrary_n_and_phase() -> None:
    """The dynamic multitone payload validates for any N, with optional phase."""
    spec = build_excitation_spec(
        {
            "kind": "multitone",
            "axis": "x",
            "fs_hz": 5000.0,
            "duration_s": 1.0,
            "tones": [[120.0, 1.0], [240.0, 0.5], [360.0, 0.25, 1.57]],
        }
    )
    assert isinstance(spec, MultitoneSpec)
    assert len(spec.tones) == 3
    assert spec.tones[2].phase_rad == pytest.approx(1.57)
