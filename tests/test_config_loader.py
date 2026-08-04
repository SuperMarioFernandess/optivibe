"""Configuration loader: path resolution and error branches (task S-25, gap 18 G8).

``core/config/loader.py`` is the first thing every run touches and was the least
covered module of the core (64 %). Nothing here changes behaviour: the loader
already fails loudly, as coding convention 10 §7 requires. What was missing was
the proof that it fails *legibly* -- with a message naming the file, the
variable or the variant at fault instead of a traceback -- and that its
three-step search for ``configs/`` (environment variable, repository layout,
upward walk) really behaves as its docstring claims.

The flat (pre-composition) variant path is exercised here too: every variant
shipped in ``configs/`` is a composition, so the legacy branch had no test at
all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from optivibe.core.config import loader
from optivibe.core.config.loader import (
    default_config_dir,
    load_constants,
    load_scenario,
    load_variant,
    load_variant_file,
)
from optivibe.core.config.models import VariantConfig

_ENV_VAR = "OPTIVIBE_CONFIG_DIR"


# --------------------------------------------------------------------------- #
# Reading YAML: the file must exist and must be a mapping.
# --------------------------------------------------------------------------- #
def test_missing_configuration_file_names_the_path(tmp_path: Path) -> None:
    """A missing file is reported by path, not by traceback (10 §7).

    The path is escaped before it becomes a regex: on Windows it contains
    backslashes, and a drive-letter path is then not a valid pattern (the
    backslash before ``Users`` reads as an incomplete unicode escape). What is
    under test is the message, not the matching.
    """
    missing = tmp_path / "constants.yaml"
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        load_constants(missing)


def test_non_mapping_yaml_names_the_file_and_the_type(tmp_path: Path) -> None:
    """A YAML list where a mapping is expected fails with both facts in the message.

    Configuration documents are mappings by construction (09 §7); a sequence at
    the top level is a user error, and the message has to say which file and
    what was found there for the user to fix it without reading the source.
    """
    path = tmp_path / "scenario.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a YAML mapping at the top level, got list"):
        load_scenario(path)


# --------------------------------------------------------------------------- #
# default_config_dir: environment variable, repository layout, upward walk.
# --------------------------------------------------------------------------- #
def test_env_var_overrides_the_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``OPTIVIBE_CONFIG_DIR`` wins over the repository layout (module docstring)."""
    monkeypatch.setenv(_ENV_VAR, str(tmp_path))
    assert default_config_dir() == tmp_path


def test_env_var_pointing_at_a_non_directory_is_rejected_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad override fails loudly and quotes the variable and its value.

    Silently falling back to the repository layout would be the worst outcome:
    the run would succeed against configurations the user did not select.
    """
    not_a_dir = tmp_path / "configs.yaml"
    not_a_dir.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(_ENV_VAR, str(not_a_dir))
    with pytest.raises(FileNotFoundError, match=f"{_ENV_VAR}=.*is not a directory"):
        default_config_dir()


def test_config_directory_is_found_by_walking_up_from_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside the repository layout the loader walks up from the working directory.

    Simulated by moving the module's apparent location (so the repository
    candidate misses) and the working directory (so the walk has somewhere to
    go), which is the situation of an installed wheel run from a project tree.
    """
    monkeypatch.delenv(_ENV_VAR, raising=False)
    fake_module = tmp_path / "elsewhere" / "a" / "b" / "c" / "loader.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(loader, "__file__", str(fake_module))

    project = tmp_path / "project"
    expected = project / "configs"
    expected.mkdir(parents=True)
    deep = project / "work" / "deeper"
    deep.mkdir(parents=True)
    monkeypatch.setattr(loader.Path, "cwd", staticmethod(lambda: deep))

    assert default_config_dir() == expected


def test_no_config_directory_anywhere_points_at_the_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When nothing is found the error names the way out (set the env var)."""
    monkeypatch.delenv(_ENV_VAR, raising=False)
    fake_module = tmp_path / "elsewhere" / "a" / "b" / "c" / "loader.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(loader, "__file__", str(fake_module))

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(loader.Path, "cwd", staticmethod(lambda: empty))

    with pytest.raises(FileNotFoundError, match=f"set {_ENV_VAR}"):
        default_config_dir()


# --------------------------------------------------------------------------- #
# Variant lookup by name.
# --------------------------------------------------------------------------- #
def test_unknown_variant_lists_the_known_ones(config_dir: Path) -> None:
    """An unknown name fails with the admissible set in the message (10 §7)."""
    with pytest.raises(ValueError, match="unknown variant 'Z'; expected one of"):
        load_variant("Z", config_dir)


def test_load_variant_without_config_dir_uses_the_repository_layout() -> None:
    """Omitting ``config_dir`` resolves it, and resolves to the same variant."""
    assert load_variant("B") == load_variant("B", default_config_dir())


# --------------------------------------------------------------------------- #
# Flat (legacy) variant files vs compositions.
# --------------------------------------------------------------------------- #
def test_flat_variant_file_loads_without_a_preset_tree(tmp_path: Path, config_dir: Path) -> None:
    """A flat ``VariantConfig`` document loads as itself, with no presets involved.

    The delivered ``configs/variants/*.yaml`` are all compositions (S9-A), so
    this branch of :func:`load_variant_file` had no coverage. The check is a
    round trip: the resolved variant B, written out flat, must read back equal
    -- the composition layer resolves *to* this document type (09 §7), so the
    two forms are required to agree.
    """
    resolved = load_variant("B", config_dir)
    flat = tmp_path / "B_flat.yaml"
    flat.write_text(
        yaml.safe_dump(json.loads(resolved.model_dump_json()), sort_keys=False),
        encoding="utf-8",
    )

    reloaded = load_variant_file(flat)

    assert isinstance(reloaded, VariantConfig)
    assert reloaded == resolved


def test_composition_infers_its_preset_root_from_the_file_location(config_dir: Path) -> None:
    """Without an explicit ``config_dir`` the preset root is inferred from the path.

    ``<config_dir>/variants/B.yaml`` implies ``<config_dir>``, which is accepted
    only because it holds a ``presets/`` tree; the inferred route must produce
    the same variant as the explicit one.
    """
    inferred = load_variant_file(config_dir / "variants" / "B.yaml")
    explicit = load_variant_file(config_dir / "variants" / "B.yaml", config_dir)
    assert inferred == explicit


def test_composition_outside_a_preset_tree_falls_back_to_the_default_root(
    tmp_path: Path, config_dir: Path
) -> None:
    """A composition copied elsewhere resolves against the default configuration root.

    The inferred parent has no ``presets/``, so the loader falls back to
    :func:`default_config_dir` rather than failing: the presets are a shared
    library, and a composition may legitimately live outside it.
    """
    variants = tmp_path / "variants"
    variants.mkdir()
    copied = variants / "B.yaml"
    copied.write_text(
        (config_dir / "variants" / "B.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )

    assert load_variant_file(copied) == load_variant("B", config_dir)
