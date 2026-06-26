"""Subsystem composition layer: equivalence, overrides, priority, validators (S9-A).

These tests pin the behaviour introduced in S9-A:

* composed variants resolve **bit-identically** to the pre-refactor flat
  variants (the committed golden in ``tests/data``);
* preset + override merging applies overrides and inherits the rest;
* user presets shadow built-ins, same-tier name clashes fail loudly;
* save/load of a whole composition round-trips exactly;
* the cross-subsystem geometry guards and the mode validator reject bad
  compositions at config time.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from optivibe.core.config import (
    PresetStore,
    SubsystemRef,
    SystemConfig,
    load_system_file,
    load_variant,
    save_system_config,
)

_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "resolved_variants_golden.json"
_GOLDEN: dict[str, dict[str, Any]] = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _base_system_dict() -> dict[str, Any]:
    """A minimal valid composition (variant B's shape) for mutation in tests."""
    return {
        "name": "T",
        "description": "test composition",
        "mode": "offresonance",
        "band": {"f_min_hz": 1.0, "f_max_hz": 10000.0},
        "full_scale_g": 50.0,
        "route": 2,
        "eta_bias": 0.25,
        "q_total": 2610.0,
        "vacuum": False,
        "source": {"preset": "sld"},
        "fiber": {"preset": "smf28"},
        "cantilever": {"preset": "silica", "overrides": {"length_m": 2.0e-3}},
        "reflector": {"preset": "cyl_rc31"},
        "detector": {"preset": "balanced_24bit"},
    }


def _mirror_presets(dst_config_dir: Path, src_config_dir: Path) -> None:
    """Copy the built-in preset tree into a writable temporary config dir."""
    shutil.copytree(src_config_dir / "presets", dst_config_dir / "presets")


# --------------------------------------------------------------------------- #
# Equivalence (bit-identity) and determinism.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_resolved_variant_is_bit_identical_to_golden(name: str, config_dir: Path) -> None:
    """Composed A/B/C/D resolve to the exact pre-refactor flat dump."""
    resolved = load_variant(name, config_dir=config_dir).model_dump(mode="python")
    assert resolved == _GOLDEN[name]


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_resolution_is_idempotent(name: str, config_dir: Path) -> None:
    """Resolving the same composition twice yields equal dumps (no hidden state)."""
    first = load_variant(name, config_dir=config_dir).model_dump(mode="python")
    second = load_variant(name, config_dir=config_dir).model_dump(mode="python")
    assert first == second


def test_resolved_name_is_preserved(config_dir: Path) -> None:
    """The widened ``name`` field carries the composition name through resolve."""
    assert load_variant("D", config_dir=config_dir).name == "D"


# --------------------------------------------------------------------------- #
# Preset + override mechanics.
# --------------------------------------------------------------------------- #
def test_override_is_applied_and_rest_inherited(config_dir: Path) -> None:
    """An override changes its field; the other preset fields are inherited."""
    store = PresetStore(config_dir)
    ref = SubsystemRef(preset="sld", overrides={"power_w": 0.020})
    source = store.build_source(ref)
    assert source.power_w == 0.020  # overridden (A's value)
    assert source.wavelength_m == 1550.0e-9  # inherited from the SLD preset
    assert source.rin_db_hz == -126.0  # inherited


def test_unknown_override_key_fails_loudly(config_dir: Path) -> None:
    """A misspelled override key is rejected (extra='forbid'), not dropped."""
    store = PresetStore(config_dir)
    ref = SubsystemRef(preset="sld", overrides={"powr_w": 0.02})
    with pytest.raises(ValueError, match="sld"):
        store.build_source(ref)


def test_unknown_preset_name_fails(config_dir: Path) -> None:
    """Referencing a non-existent preset names the subsystem and lists options."""
    store = PresetStore(config_dir)
    with pytest.raises(ValueError, match="unknown source preset"):
        store.build_source(SubsystemRef(preset="does_not_exist"))


# --------------------------------------------------------------------------- #
# Tier priority and collisions.
# --------------------------------------------------------------------------- #
def test_user_preset_shadows_builtin(tmp_path: Path, config_dir: Path) -> None:
    """A user preset of the same name wins over the built-in (documented priority)."""
    _mirror_presets(tmp_path, config_dir)
    user_src = tmp_path / "user" / "presets" / "source"
    user_src.mkdir(parents=True)
    # Same name as a built-in ('sld') but a different power.
    (user_src / "sld.yaml").write_text(
        "source_kind: SLD\nwavelength_m: 1550.0e-9\npower_w: 0.005\nrin_db_hz: -126.0\n",
        encoding="utf-8",
    )
    store = PresetStore(tmp_path)
    assert store.build_source(SubsystemRef(preset="sld")).power_w == 0.005
    assert store.list_presets("source")["sld"] == user_src / "sld.yaml"


def test_duplicate_name_same_tier_fails(tmp_path: Path, config_dir: Path) -> None:
    """Two files resolving to the same stem in one tier raise loudly."""
    _mirror_presets(tmp_path, config_dir)
    src = tmp_path / "presets" / "source"
    # 'sld.yaml' already exists from the mirror; add a colliding 'sld.yml'.
    (src / "sld.yml").write_text(
        "source_kind: SLD\nwavelength_m: 1550.0e-9\npower_w: 0.016\nrin_db_hz: -126.0\n",
        encoding="utf-8",
    )
    store = PresetStore(tmp_path)
    with pytest.raises(ValueError, match="duplicate preset name 'sld'"):
        store.build_source(SubsystemRef(preset="sld"))


# --------------------------------------------------------------------------- #
# Save / load round-trip.
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip_preserves_resolution(tmp_path: Path, config_dir: Path) -> None:
    """A composition saved and reloaded resolves to the same VariantConfig."""
    original = load_system_file(config_dir / "variants" / "A.yaml")
    store = PresetStore(config_dir)
    before = original.resolve(store).model_dump(mode="python")

    out = save_system_config(original, tmp_path / "user" / "systems" / "mine.yaml")
    assert out.is_file()
    reloaded = load_system_file(out)

    assert reloaded == original  # SystemConfig itself round-trips
    after = reloaded.resolve(store).model_dump(mode="python")
    assert after == before  # and so does the resolved variant


# --------------------------------------------------------------------------- #
# Validators / geometry guards (bad compositions fail at config time).
# --------------------------------------------------------------------------- #
def test_unregistered_reflector_shape_is_rejected(config_dir: Path) -> None:
    """A reflector shape without a registered optics model fails at resolve."""
    data = _base_system_dict()
    data["reflector"] = {"preset": "cyl_rc31", "overrides": {"shape": "sphere"}}
    store = PresetStore(config_dir)
    with pytest.raises(ValueError, match="not registered"):
        SystemConfig.model_validate(data).resolve(store)


def test_radius_guard_is_rejected(config_dir: Path) -> None:
    """A radius of curvature below 5*w0 violates the paraxial guard."""
    data = _base_system_dict()
    # w0 = 5.2 um -> 5 w0 = 26 um; 10 um is too small.
    data["reflector"] = {"preset": "cyl_rc31", "overrides": {"curvature_radius_m": 10.0e-6}}
    store = PresetStore(config_dir)
    with pytest.raises(ValueError, match="paraxial guard"):
        SystemConfig.model_validate(data).resolve(store)


def test_spot_guard_is_rejected(config_dir: Path) -> None:
    """A gap large enough that the spot exceeds R_c/3 is rejected."""
    data = _base_system_dict()
    # Huge gap blows up w(A) well past R_c/3 for R_c = 31 um.
    data["reflector"] = {"preset": "cyl_rc31", "overrides": {"gap_m": 5.0e-3}}
    store = PresetStore(config_dir)
    with pytest.raises(ValueError, match="R_c/3"):
        SystemConfig.model_validate(data).resolve(store)


def test_resonance_requires_line_freq() -> None:
    """A resonant composition without a line frequency fails validation."""
    data = _base_system_dict()
    data["mode"] = "resonance"
    data["line_freq_hz"] = None
    with pytest.raises(ValueError, match="line_freq_hz"):
        SystemConfig.model_validate(data)


def test_band_order_is_validated() -> None:
    """An inverted band (f_max <= f_min) fails validation."""
    data = _base_system_dict()
    data["band"] = {"f_min_hz": 100.0, "f_max_hz": 10.0}
    with pytest.raises(ValueError, match="f_max_hz"):
        SystemConfig.model_validate(data)
