"""GUI tests for the expected-peak overlay on the Live spectrum (S-16/S-17).

Skipped automatically without the ``gui`` extra; runs head-less under
``QT_QPA_PLATFORM=offscreen`` (conftest). Proves the overlay is opt-in, that it
draws one marker per predicted peak plus the ``f1/Q`` band, that it clears
cleanly, and that a real run attaches a prediction -- while the view itself
computes nothing (09 §9).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

pytestmark = pytest.mark.gui

from optivibe.analysis.expected_peaks import predict_expected_peaks  # noqa: E402
from optivibe.core.config.loader import load_constants, load_variant  # noqa: E402
from optivibe.core.config.models import ScenarioConfig  # noqa: E402
from optivibe.gui.main_window import MainWindow  # noqa: E402
from optivibe.gui.widgets.live_view import LiveView  # noqa: E402


def _prediction():
    """Build an ExpectedPeaks artifact off-widget (as the app does)."""
    scenario = ScenarioConfig(
        name="gui_expected_probe",
        variant="B",
        excitation={
            "kind": "sine",
            "fs_hz": 51200.0,
            "duration_s": 1.0,
            "frequency_hz": 100.0,
            "amplitude_g": 10.2,
        },
        stages={"detector": "photodiode"},
    )
    return predict_expected_peaks(scenario, load_variant("B"), load_constants())


def test_expected_overlay_is_opt_in(qtbot) -> None:
    """Attaching a prediction draws nothing until the user asks for it."""
    live = LiveView()
    qtbot.addWidget(live)
    assert not live._expected_check.isChecked()

    live.set_expected_peaks(_prediction())
    assert live._expected_items == []

    live._expected_check.setChecked(True)
    assert live._expected_items


def test_expected_overlay_draws_a_marker_per_peak_plus_the_band(qtbot) -> None:
    """One marker per predicted line, plus the shaded ``f1/Q`` region."""
    live = LiveView()
    qtbot.addWidget(live)
    expected = _prediction()
    live._expected_check.setChecked(True)
    live.set_expected_peaks(expected)

    assert expected.band_hz is not None
    assert len(live._expected_items) == len(expected.peaks) + 1


def test_expected_overlay_clears_and_does_not_accumulate(qtbot) -> None:
    """Re-attaching or clearing removes the previous items (no leak)."""
    live = LiveView()
    qtbot.addWidget(live)
    live._expected_check.setChecked(True)
    expected = _prediction()

    live.set_expected_peaks(expected)
    first = len(live._expected_items)
    live.set_expected_peaks(expected)
    assert len(live._expected_items) == first

    live.set_expected_peaks(None)
    assert live._expected_items == []

    live.set_expected_peaks(expected)
    live._expected_check.setChecked(False)
    assert live._expected_items == []


def test_expected_overlay_survives_a_panel_reflow(qtbot) -> None:
    """Hiding/showing panels does not disturb the overlay (isolated helper)."""
    live = LiveView()
    qtbot.addWidget(live)
    live._expected_check.setChecked(True)
    live.set_expected_peaks(_prediction())
    before = len(live._expected_items)

    live._checks["det"].setChecked(False)
    live._checks["det"].setChecked(True)
    assert len(live._expected_items) == before


def test_real_run_attaches_the_prediction(qtbot) -> None:
    """A finished run hands the Live view a prediction for its own scenario."""
    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.controller.finished, timeout=20000):
        window.run_button.click()

    live = window.plot
    expected = live._expected
    assert expected is not None
    assert expected.f1_hz > 0.0
    # The default GUI composition samples well below f1 (~25 kHz), so the mode
    # is correctly *not* promised; the drive harmonics are.
    assert expected.nyquist_hz is not None and expected.nyquist_hz < expected.f1_hz
    assert expected.of_kind("mode") == ()
    assert expected.of_kind("harmonic")
