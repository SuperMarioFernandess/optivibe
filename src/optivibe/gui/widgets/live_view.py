"""Live PyQtGraph displays for a run (task S7 §3; visibility toggles S7-mod §4).

Renders, off nothing but core/analysis *results* (no DSP in the view): the
cantilever bend animation; the time-domain input-vs-recovered acceleration; the
detector signal; the recovered velocity and displacement; the recovered
amplitude spectrum (``VibrationResult.spectrum``, computed by the core); and the
NEA(f) density with its shot/RIN/Johnson/thermal plateau split (from the analysis
``NeaBudget``). Long series are decimated before drawing. The richer
input-vs-recovered spectral overlay and the spectrogram live in the (matplotlib)
Report tab, so this tab stays light and fast.

An opt-in check-box overlays the *expected* peaks of the run on the spectrum
panel (tasks S-16/S-17): the cantilever resonance ``f1`` with its shaded
``f1/Q`` band and the drive-tone harmonics, so correct physics stops reading as
an artifact (doc 20 §3). Every number arrives ready-made in an
:class:`~optivibe.analysis.expected_peaks.ExpectedPeaks` artifact -- this view
computes nothing (09 §9). The overlay is confined to :meth:`LiveView.
set_expected_peaks` / :meth:`LiveView._refresh_expected` and touches no layout,
so the planned real-time oscilloscope (O-SW-03) can reuse the helper instead of
writing its own.

A row of checkboxes (task S7-mod §4) shows/hides each panel; hiding a panel
**reflows** the layout so the visible panels expand to fill the freed space (the
panels live in one :class:`pyqtgraph.GraphicsLayoutWidget`, re-added in order on
each toggle). The cantilever animation is a separate widget and is toggled with
``setVisible``. The selection is session-only (not persisted).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QSplitter, QVBoxLayout, QWidget

from optivibe.analysis import NeaBudget
from optivibe.analysis.expected_peaks import ExpectedPeaks
from optivibe.core.types import FloatArray, VibrationResult
from optivibe.gui.i18n import t
from optivibe.gui.widgets.cantilever_view import CantileverView
from optivibe.gui.widgets.ui_helpers import tab_header
from optivibe.pipeline import RunArtifacts

__all__ = ["LiveView"]

_G0 = 9.80665
_MAX_POINTS = 4000

# Marker colour per expected-peak taxonomy branch (mirrors viz.dsp so the
# pyqtgraph overlay and the matplotlib figure read the same).
_EXPECTED_COLORS: dict[str, str] = {
    "mode": "#2ca02c",
    "harmonic": "#ff7f0e",
    "intermod": "#9467bd",
}
_EXPECTED_FALLBACK_COLOR = "#7f7f7f"

# Panel key -> checkbox label (order defines the top-to-bottom layout).
_PANEL_LABELS: tuple[tuple[str, str], ...] = (
    ("accel", "acceleration"),
    ("det", "detector"),
    ("vel", "velocity"),
    ("disp", "displacement"),
    ("spec", "spectrum"),
    ("nea", "NEA(f)"),
)


def _decimate(*arrays: FloatArray, n_max: int = _MAX_POINTS) -> list[FloatArray]:
    """Stride-decimate parallel arrays to at most ``n_max`` points."""
    if not arrays:
        return []
    size = arrays[0].size
    stride = max(1, size // n_max)
    return [np.asarray(a, dtype=np.float64)[::stride] for a in arrays]


class LiveView(QWidget):
    """Composite live view: bending animation over toggleable PyQtGraph panels.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cantilever = CantileverView()
        self._plots = pg.GraphicsLayoutWidget()

        self._p_accel = pg.PlotItem(title=t("Acceleration: input vs recovered"))
        self._p_accel.addLegend(offset=(-10, 5))
        self._accel_true = self._p_accel.plot(
            [], [], pen=pg.mkPen("#1f77b4", width=2), name=t("input")
        )
        self._accel_rec = self._p_accel.plot(
            [], [], pen=pg.mkPen("#ff7f0e", width=1), name=t("recovered")
        )
        self._p_accel.setLabel("left", t("a"), units="m/s^2")

        self._p_det = pg.PlotItem(title=t("Detector signal"))
        self._det = self._p_det.plot([], [], pen=pg.mkPen("#2ca02c", width=1))
        self._p_det.setLabel("left", t("samples"))

        self._p_vel = pg.PlotItem(title=t("Recovered velocity"))
        self._vel = self._p_vel.plot([], [], pen=pg.mkPen("#9467bd", width=1))
        self._p_vel.setLabel("left", t("v"), units="m/s")

        self._p_disp = pg.PlotItem(title=t("Recovered displacement"))
        self._disp = self._p_disp.plot([], [], pen=pg.mkPen("#8c564b", width=1))
        self._p_disp.setLabel("left", t("x"), units="m")
        self._p_disp.setLabel("bottom", t("time"), units="s")

        self._p_spec = pg.PlotItem(title=t("Recovered amplitude spectrum"))
        self._spec = self._p_spec.plot([], [], pen=pg.mkPen("#1f77b4", width=1))
        self._p_spec.setLabel("bottom", t("frequency"), units="Hz")
        self._p_spec.setLabel("left", t("amplitude"))
        self._p_spec.setLogMode(x=False, y=True)

        self._p_nea = pg.PlotItem(title=t("NEA(f) - run Report for the budget"))
        self._p_nea.setLabel("bottom", t("frequency"), units="Hz")
        self._p_nea.setLabel("left", t("NEA [ug/sqrt(Hz)]"))
        self._p_nea.setLogMode(x=True, y=True)
        self._p_nea.addLegend(offset=(-10, 5))
        self._nea_total = self._p_nea.plot(
            [], [], pen=pg.mkPen("#000000", width=2), name=t("total")
        )

        for plot in (self._p_accel, self._p_det, self._p_vel, self._p_disp, self._p_spec):
            plot.showGrid(x=True, y=True, alpha=0.3)

        self._panels: dict[str, pg.PlotItem] = {
            "accel": self._p_accel,
            "det": self._p_det,
            "vel": self._p_vel,
            "disp": self._p_disp,
            "spec": self._p_spec,
            "nea": self._p_nea,
        }
        self._visible: dict[str, bool] = {key: True for key, _ in _PANEL_LABELS}
        self._expected: ExpectedPeaks | None = None
        self._expected_items: list[pg.GraphicsObject] = []
        self._relayout()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._cantilever)
        splitter.addWidget(self._plots)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self._header = tab_header(
            "Live",
            "Live view of the last run: the cantilever bend animation over "
            "PyQtGraph panels for input-vs-recovered acceleration, the detector "
            "signal, recovered velocity/displacement, the amplitude spectrum and "
            "the NEA(f) density. The check-row shows/hides each panel (session "
            "only); no controls here change the model -- edit on the left and Run.",
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._header)
        layout.addWidget(self._visibility_bar())
        layout.addWidget(splitter)

    def retranslate(self) -> None:
        """Refresh static text after a language change (legends refresh on re-run)."""
        self._header.retranslate()
        self._cantilever_check.setText(t("cantilever"))
        self._expected_check.setText(t("expected peaks"))
        for key, label in _PANEL_LABELS:
            self._checks[key].setText(t(label))
        self._p_accel.setTitle(t("Acceleration: input vs recovered"))
        self._p_accel.setLabel("left", t("a"), units="m/s^2")
        self._p_det.setTitle(t("Detector signal"))
        self._p_det.setLabel("left", t("samples"))
        self._p_vel.setTitle(t("Recovered velocity"))
        self._p_vel.setLabel("left", t("v"), units="m/s")
        self._p_disp.setTitle(t("Recovered displacement"))
        self._p_disp.setLabel("left", t("x"), units="m")
        self._p_disp.setLabel("bottom", t("time"), units="s")
        self._p_spec.setTitle(t("Recovered amplitude spectrum"))
        self._p_spec.setLabel("bottom", t("frequency"), units="Hz")
        self._p_spec.setLabel("left", t("amplitude"))
        self._p_nea.setLabel("bottom", t("frequency"), units="Hz")
        self._p_nea.setLabel("left", t("NEA [ug/sqrt(Hz)]"))

    # ------------------------------------------------------------------ #
    # Panel visibility (task S7-mod §4)
    # ------------------------------------------------------------------ #
    def _visibility_bar(self) -> QWidget:
        """Build the row of show/hide checkboxes (cantilever + each panel)."""
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(2, 2, 2, 2)
        self._checks: dict[str, QCheckBox] = {}

        self._cantilever_check = QCheckBox(t("cantilever"))
        cantilever_check = self._cantilever_check
        cantilever_check.setChecked(True)
        cantilever_check.toggled.connect(self._cantilever.setVisible)
        row.addWidget(cantilever_check)

        for key, label in _PANEL_LABELS:
            check = QCheckBox(t(label))
            check.setChecked(True)
            check.toggled.connect(lambda shown, k=key: self._set_panel_visible(k, shown))
            self._checks[key] = check
            row.addWidget(check)

        # Expected-peak overlay: opt-in, so the spectrum stays uncluttered by
        # default (tasks S-16/S-17).
        self._expected_check = QCheckBox(t("expected peaks"))
        self._expected_check.setChecked(False)
        self._expected_check.setToolTip(
            t("Overlay the peaks the twin predicts: resonance f1 (+ f1/Q band) and drive harmonics")
        )
        self._expected_check.toggled.connect(lambda _shown: self._refresh_expected())
        row.addWidget(self._expected_check)
        row.addStretch(1)
        return bar

    def _set_panel_visible(self, key: str, shown: bool) -> None:
        """Toggle a panel and reflow so the visible ones fill the space."""
        self._visible[key] = shown
        self._relayout()

    def _relayout(self) -> None:
        """Re-add the visible panels in order (true reflow, freeing space)."""
        self._plots.clear()
        row = 0
        for key, _label in _PANEL_LABELS:
            if self._visible[key]:
                self._plots.addItem(self._panels[key], row=row, col=0)
                row += 1

    # ------------------------------------------------------------------ #
    # Expected-peak overlay (tasks S-16/S-17; doc 20 §3)
    # ------------------------------------------------------------------ #
    def set_expected_peaks(self, expected: ExpectedPeaks | None) -> None:
        """Attach (or clear) the predicted peak set for the spectrum overlay.

        Parameters
        ----------
        expected : ExpectedPeaks or None
            Prediction from
            :func:`~optivibe.analysis.expected_peaks.predict_expected_peaks`;
            ``None`` clears the overlay. Nothing is computed here -- the
            artifact already carries the positions, widths and thresholds.
        """
        self._expected = expected
        self._refresh_expected()

    def _refresh_expected(self) -> None:
        """Rebuild the overlay items on the spectrum panel (no layout change)."""
        for item in self._expected_items:
            self._p_spec.removeItem(item)
        self._expected_items = []
        expected = self._expected
        if expected is None or not self._expected_check.isChecked():
            return
        band = expected.band_hz
        if band is not None:
            region = pg.LinearRegionItem(
                values=band, brush=pg.mkBrush(44, 160, 44, 40), movable=False
            )
            region.setZValue(-10)
            self._p_spec.addItem(region)
            self._expected_items.append(region)
        for peak in expected.peaks:
            color = _EXPECTED_COLORS.get(peak.kind, _EXPECTED_FALLBACK_COLOR)
            line = pg.InfiniteLine(
                pos=peak.freq_hz,
                angle=90,
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine),
                label=t(peak.label) if peak.kind == "mode" else peak.label,
                labelOpts={"position": 0.9, "color": color, "rotateAxis": (1, 0)},
            )
            line.setToolTip(peak.explanation)
            self._p_spec.addItem(line)
            self._expected_items.append(line)

    def show_artifacts(self, artifacts: RunArtifacts, beta1_l: float) -> None:
        """Render a run's intermediates and recovered signals.

        Parameters
        ----------
        artifacts : RunArtifacts
            The forward + inverse run (intermediates + result).
        beta1_l : float
            First eigenvalue ``beta_1 * L`` for the bend animation.
        """
        result = artifacts.result
        a_true = np.asarray(artifacts.forward.excitation.a_x, dtype=np.float64)
        fs = result.fs
        a_t, a_r = _decimate(a_true, np.asarray(result.a))
        t = np.arange(a_t.size) * (max(1, a_true.size // a_t.size) / fs)
        self._accel_true.setData(t, a_t)
        self._accel_rec.setData(t, a_r)

        det = np.asarray(artifacts.forward.detector.samples, dtype=np.float64)
        (det_d,) = _decimate(det)
        det_stride = max(1, det.size // det_d.size)
        t_det = np.arange(det_d.size) * (det_stride / artifacts.forward.detector.fs)
        self._det.setData(t_det, det_d)

        v_d, x_d = _decimate(np.asarray(result.v), np.asarray(result.x))
        t_vx = np.arange(v_d.size) * (max(1, result.v.size // v_d.size) / fs)
        self._vel.setData(t_vx, v_d)
        self._disp.setData(t_vx, x_d)

        if result.spectrum is not None:
            self._spec.setData(result.spectrum.freq.tolist(), result.spectrum.values.tolist())
        else:
            self._spec.setData([], [])

        self._cantilever.set_motion(artifacts.forward.tip, beta1_l, artifacts.variant.length_m)
        self._reset_nea_panel()

    def show_nea(self, nea: NeaBudget | None) -> None:
        """Render the NEA(f) density and its plateau contribution split.

        Parameters
        ----------
        nea : NeaBudget or None
            The NEA budget; ``None`` (stub detector) clears the panel.
        """
        self._reset_nea_panel()
        if nea is None:
            self._p_nea.setTitle(t("NEA(f) - not available (use the photodiode detector)"))
            return
        self._p_nea.setTitle(t("NEA(f) with shot / RIN / Johnson / thermal plateaus"))
        scale = 1.0e6 / _G0
        self._nea_total.setData(nea.freq_hz.tolist(), (nea.nea_density * scale).tolist())
        # The fourth (Brownian thermal) branch of the budget (M-12, doc 07 §2):
        # acceleration-domain, flat across the band like the referred trio.
        colors = {"shot": "#d62728", "rin": "#2ca02c", "johnson": "#9467bd", "thermal": "#ff7f0e"}
        f_lo, f_hi = float(nea.freq_hz[0]), float(nea.freq_hz[-1])
        for key, color in colors.items():
            level = nea.contributions.get(key, 0.0) * scale
            if level > 0.0:
                self._p_nea.plot(
                    [f_lo, f_hi],
                    [level, level],
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                    name=t(key),
                )

    def show_result(self, result: VibrationResult) -> None:
        """Minimal S0-compatible render (recovered acceleration + spectrum)."""
        (a_r,) = _decimate(np.asarray(result.a))
        t = np.arange(a_r.size) * (max(1, result.a.size // a_r.size) / result.fs)
        self._accel_rec.setData(t, a_r)
        self._accel_true.setData([], [])
        if result.spectrum is not None:
            self._spec.setData(result.spectrum.freq.tolist(), result.spectrum.values.tolist())

    def stop(self) -> None:
        """Stop the bend animation (e.g. on close)."""
        self._cantilever.clear_motion()

    def _reset_nea_panel(self) -> None:
        """Clear the NEA panel down to the (empty) total curve."""
        self._p_nea.clear()
        self._nea_total = self._p_nea.plot(
            [], [], pen=pg.mkPen("#000000", width=2), name=t("total")
        )
