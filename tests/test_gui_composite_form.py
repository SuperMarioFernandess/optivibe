"""GUI tests for the per-kind composite sub-forms (task S-23, doc 11 §2.1.4-§2.1.5).

Skipped automatically without the ``gui`` extra; runs head-less under
``QT_QPA_PLATFORM=offscreen`` (conftest). The subject is the *editor*: that it
builds the same config a scenario file carries, that the round trip through the
YAML view is stable, that what the forms cannot hold exactly stays on the YAML
path instead of being rounded, and that a stimulus assembled here still reaches
the expected-peak layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

pytestmark = pytest.mark.gui

from optivibe.analysis.expected_peaks import predict_expected_peaks  # noqa: E402
from optivibe.core.config.loader import load_constants, load_variant  # noqa: E402
from optivibe.core.config.models import ScenarioConfig  # noqa: E402
from optivibe.excitation.composite import CompositeExcitationSource  # noqa: E402
from optivibe.gui.controllers.scenario_builder import build_excitation_spec  # noqa: E402
from optivibe.gui.widgets.excitation_builder import ExcitationBuilder  # noqa: E402


def _builder(qtbot) -> ExcitationBuilder:
    """Build an excitation panel already switched to the composite page."""
    builder = ExcitationBuilder()
    qtbot.addWidget(builder)
    builder._kind.setCurrentText("composite")
    return builder


# --------------------------------------------------------------------------- #
# Rows: add / remove / per-kind sub-forms
# --------------------------------------------------------------------------- #
def test_composite_form_defaults_to_two_valid_components(qtbot) -> None:
    """The fresh page carries the S-21 demo stimulus as rows, and it validates."""
    builder = _builder(qtbot)
    assert builder._composite.count() == 2
    spec = build_excitation_spec(builder.excitation_payload())
    assert [component.kind for component in spec.components] == ["sine", "random"]


def test_composite_form_adds_and_removes_components(qtbot) -> None:
    """Components are added and removed row by row, never below one."""
    composite = _builder(qtbot)._composite
    composite._add_row({"kind": "shock", "peak_g": 50.0, "pulse_ms": 2.0})
    assert composite.count() == 3
    composite._remove_row(composite._rows[-1])
    assert composite.count() == 2
    composite._remove_row(composite._rows[-1])
    composite._remove_row(composite._rows[-1])
    assert composite.count() == 1


def test_composite_form_covers_every_admissible_component_kind(qtbot) -> None:
    """Each generated kind has a sub-form and produces a valid component."""
    builder = _builder(qtbot)
    composite = builder._composite
    row = composite._rows[0]
    for kind in ("sine", "multitone", "sweep", "random", "shock"):
        row._kind.setCurrentText(kind)
        assert row.payload()["kind"] == kind
    composite._remove_row(composite._rows[1])
    spec = build_excitation_spec(builder.excitation_payload())
    assert spec.components[0].kind == "shock"


# --------------------------------------------------------------------------- #
# Axis: per-component, summed per axis (doc 11 §2.1.4)
# --------------------------------------------------------------------------- #
def test_composite_form_gives_a_component_its_own_axis(qtbot) -> None:
    """A row may name its axis; left inherited it follows the composite's."""
    builder = _builder(qtbot)
    builder._axis.setCurrentText("y")
    composite = builder._composite
    composite._rows[1]._axis.setCurrentText("z")
    spec = build_excitation_spec(builder.excitation_payload())
    assert [component.axis for component in spec.components] == ["y", "z"]


# --------------------------------------------------------------------------- #
# Grid: defined once, on the composite (doc 11 §2.1.4, Q6 of S-21)
# --------------------------------------------------------------------------- #
def test_composite_form_offers_no_per_component_grid(qtbot) -> None:
    """No row emits ``fs_hz`` / ``duration_s``: the grid lives on the composite."""
    builder = _builder(qtbot)
    builder._fs.setValue(20000.0)
    builder._duration.setValue(0.5)
    payload = builder.excitation_payload()
    assert (payload["fs_hz"], payload["duration_s"]) == (20000.0, 0.5)
    for component in payload["components"]:
        assert "fs_hz" not in component
        assert "duration_s" not in component
    assert "20000" in builder._composite._grid_note.text()


def test_composite_form_leaves_a_contradicting_grid_on_the_yaml_path(qtbot) -> None:
    """A component that disagrees with the composite grid is not adopted silently.

    Such a component is a loud error at run time (doc 11 §2.1.4); the editor
    keeps it as written so the user sees their own text in the report.
    """
    composite = _builder(qtbot)._composite
    composite.set_components(
        [{"kind": "sine", "fs_hz": 999.0, "frequency_hz": 100.0, "amplitude_g": 1.0}]
    )
    assert composite._tabs.currentIndex() == 1
    assert not composite._yaml_note.isHidden()
    assert composite.components()[0]["fs_hz"] == 999.0


# --------------------------------------------------------------------------- #
# Config-first: the round trip through the YAML view
# --------------------------------------------------------------------------- #
def test_composite_form_round_trips_through_the_yaml_view(qtbot) -> None:
    """form -> YAML -> form returns the same components, and stays stable."""
    composite = _builder(qtbot)._composite
    composite._add_row({"kind": "sweep", "axis": "z", "f_start_hz": 20.0, "f_end_hz": 400.0})
    before = composite.components()

    composite._tabs.setCurrentIndex(1)
    text = composite._text.toPlainText()
    assert yaml.safe_load(text) == before

    composite._tabs.setCurrentIndex(0)
    assert composite._tabs.currentIndex() == 0
    assert composite.components() == before

    composite._tabs.setCurrentIndex(1)
    assert composite._text.toPlainText() == text


def test_composite_form_reads_the_scenario_file_verbatim(qtbot, examples_dir: Path) -> None:
    """The components of the S-21 example load into the forms unchanged.

    Config-first in both directions: the panel edits the very mapping the
    scenario file carries (doc 13, coordination 2026-07-29).
    """
    document = yaml.safe_load((examples_dir / "composite_modulated.yaml").read_text())
    components = document["excitation"]["components"]
    builder = _builder(qtbot)
    builder._fs.setValue(document["excitation"]["fs_hz"])
    builder._duration.setValue(document["excitation"]["duration_s"])
    builder._composite.set_components(components)
    assert builder._composite._tabs.currentIndex() == 0
    assert builder._composite.components() == components


def test_composite_form_keeps_an_unrepresentable_component_as_text(qtbot) -> None:
    """A PSD noise level has no field here, so the whole list stays in YAML."""
    composite = _builder(qtbot)._composite
    components = [{"kind": "random", "band_hz": [10.0, 100.0], "psd_g2_hz": 1.0e-4}]
    composite.set_components(components)
    assert composite._tabs.currentIndex() == 1
    assert composite.components() == components


def test_composite_form_passes_unparsable_text_through(qtbot) -> None:
    """Text that is not YAML reaches the validator, which reports it (10 §7)."""
    composite = _builder(qtbot)._composite
    composite.set_components("- kind: [sine\n")
    assert isinstance(composite.components(), str)


# --------------------------------------------------------------------------- #
# Modulation (doc 11 §2.1.3): the depth bound and its way out
# --------------------------------------------------------------------------- #
def test_composite_form_flags_over_modulation_and_names_the_way_out(qtbot) -> None:
    """m > 1 is shown as rejected, with the three-tone composite as the route.

    Formula source: doc 11 §2.1.3 -- above m = 1 the envelope changes sign, and
    the same waveform is exactly a carrier plus two sidebands m*a_c/2.
    """
    builder = _builder(qtbot)
    composite = builder._composite
    composite.set_components(
        [
            {
                "kind": "sine",
                "frequency_hz": 100.0,
                "amplitude_g": 1.0,
                "modulation": {"kind": "am", "f_m_hz": 10.0, "depth": 1.4},
            }
        ]
    )
    sine = composite._rows[0]._forms["sine"]
    assert sine._mod_depth.value() == pytest.approx(1.4)
    assert not sine._warning.isHidden()
    assert "three sine components" in sine._warning.text()

    with pytest.raises(ValueError, match="composite"):
        build_excitation_spec(builder.excitation_payload())

    sine._mod_depth.setValue(0.6)
    assert sine._warning.isHidden()
    spec = build_excitation_spec(builder.excitation_payload())
    assert spec.components[0].modulation is not None


# --------------------------------------------------------------------------- #
# Seeding (doc 11 §2.1.5)
# --------------------------------------------------------------------------- #
def test_composite_form_pins_a_noise_component_seed(qtbot) -> None:
    """The opt-in seed of a noise row reaches ``RandomSpec.seed`` (doc 11 §2.1.5)."""
    builder = _builder(qtbot)
    noise = builder._composite._rows[1]._forms["random"]
    assert "seed" not in noise.payload()
    noise._own_seed.setChecked(True)
    noise._seed.setValue(4242)
    spec = build_excitation_spec(builder.excitation_payload())
    assert spec.components[1].seed == 4242


# --------------------------------------------------------------------------- #
# The form changes the config, never the numbers
# --------------------------------------------------------------------------- #
def test_composite_form_stimulus_equals_the_scenario_file_bit_for_bit(
    qtbot, examples_dir: Path
) -> None:
    """A stimulus assembled in the form is the file's stimulus, to the byte.

    The editor is a config surface (09 §9): the same components on the same grid
    and seed must generate the very same array as the scenario file does.
    """
    document = yaml.safe_load((examples_dir / "composite_modulated.yaml").read_text())
    excitation = document["excitation"]
    builder = _builder(qtbot)
    builder._axis.setCurrentText(excitation["axis"])
    builder._fs.setValue(excitation["fs_hz"])
    builder._duration.setValue(excitation["duration_s"])
    builder._composite.set_components(excitation["components"])

    source = CompositeExcitationSource()
    seed = document["seed"]
    from_form = source.generate(build_excitation_spec(builder.excitation_payload()), seed=seed)
    from_file = source.generate(build_excitation_spec(excitation), seed=seed)
    assert np.asarray(from_form.a_x).tobytes() == np.asarray(from_file.a_x).tobytes()


# --------------------------------------------------------------------------- #
# The predicted-peak layer still sees a form-built stimulus (S-16/S-17, S-21)
# --------------------------------------------------------------------------- #
def test_composite_form_stimulus_is_marked_by_the_expected_peaks(qtbot) -> None:
    """form -> scenario -> ExpectedPeaks marks the AM sidebands f_c +- f_m.

    Reference: doc 11 §2.1.3 -- an AM carrier puts one sideband pair of
    amplitude m*a_c/2 at f_c +- f_m (not a snapshot of the code, 18 §5(g)).
    """
    builder = _builder(qtbot)
    builder._fs.setValue(50000.0)
    builder._duration.setValue(1.0)
    builder._composite.set_components(
        [
            {
                "kind": "sine",
                "frequency_hz": 1000.0,
                "amplitude_g": 10.0,
                "modulation": {"kind": "am", "f_m_hz": 37.0, "depth": 0.4},
            }
        ]
    )
    scenario = ScenarioConfig(
        name="gui_composite_form_probe",
        variant="B",
        excitation=builder.excitation_payload(),
        stages={"detector": "photodiode"},
    )
    peaks = predict_expected_peaks(scenario, load_variant("B"), load_constants())
    sidebands = {round(peak.freq_hz, 6) for peak in peaks.of_kind("sideband")}
    assert {963.0, 1037.0} <= sidebands
