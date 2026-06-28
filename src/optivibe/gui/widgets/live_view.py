"""Live PyQtGraph displays for a run (task S7 §3; visibility toggles S7-mod §4).

Renders, off nothing but core/analysis *results* (no DSP in the view): the
cantilever bend animation; the time-domain input-vs-recovered acceleration; the
detector signal; the recovered velocity and displacement; the recovered
amplitude spectrum (``VibrationResult.spectrum``, computed by the core); and the
NEA(f) density with its shot/RIN/Johnson plateau split (from the analysis
``NeaBudget``). Long series are decimated before drawing. The richer
input-vs-recovered spectral overlay and the spectrogram live in the (matplotlib)
Report tab, so this tab stays light and fast.

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
from optivibe.core.types import FloatArray, VibrationResult
from optivibe.gui.widgets.cantilever_view import CantileverView
from optivibe.pipeline import RunArtifacts

__all__ = ["LiveView"]

_G0 = 9.80665
_MAX_POINTS = 4000

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

        self._p_accel = pg.PlotItem(title="Acceleration: input vs recovered")
        self._p_accel.addLegend(offset=(-10, 5))
        self._accel_true = self._p_accel.plot(
            [], [], pen=pg.mkPen("#1f77b4", width=2), name="input"
        )
        self._accel_rec = self._p_accel.plot(
            [], [], pen=pg.mkPen("#ff7f0e", width=1), name="recovered"
        )
        self._p_accel.setLabel("left", "a", units="m/s^2")

        self._p_det = pg.PlotItem(title="Detector signal")
        self._det = self._p_det.plot([], [], pen=pg.mkPen("#2ca02c", width=1))
        self._p_det.setLabel("left", "samples")

        self._p_vel = pg.PlotItem(title="Recovered velocity")
        self._vel = self._p_vel.plot([], [], pen=pg.mkPen("#9467bd", width=1))
        self._p_vel.setLabel("left", "v", units="m/s")

        self._p_disp = pg.PlotItem(title="Recovered displacement")
        self._disp = self._p_disp.plot([], [], pen=pg.mkPen("#8c564b", width=1))
        self._p_disp.setLabel("left", "x", units="m")
        self._p_disp.setLabel("bottom", "time", units="s")

        self._p_spec = pg.PlotItem(title="Recovered amplitude spectrum")
        self._spec = self._p_spec.plot([], [], pen=pg.mkPen("#1f77b4", width=1))
        self._p_spec.setLabel("bottom", "frequency", units="Hz")
        self._p_spec.setLabel("left", "amplitude")
        self._p_spec.setLogMode(x=False, y=True)

        self._p_nea = pg.PlotItem(title="NEA(f) - run Report for the budget")
        self._p_nea.setLabel("bottom", "frequency", units="Hz")
        self._p_nea.setLabel("left", "NEA [ug/sqrt(Hz)]")
        self._p_nea.setLogMode(x=True, y=True)
        self._p_nea.addLegend(offset=(-10, 5))
        self._nea_total = self._p_nea.plot([], [], pen=pg.mkPen("#000000", width=2), name="total")

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
        self._relayout()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._cantilever)
        splitter.addWidget(self._plots)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._visibility_bar())
        layout.addWidget(splitter)

    # ------------------------------------------------------------------ #
    # Panel visibility (task S7-mod §4)
    # ------------------------------------------------------------------ #
    def _visibility_bar(self) -> QWidget:
        """Build the row of show/hide checkboxes (cantilever + each panel)."""
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(2, 2, 2, 2)
        self._checks: dict[str, QCheckBox] = {}

        cantilever_check = QCheckBox("cantilever")
        cantilever_check.setChecked(True)
        cantilever_check.toggled.connect(self._cantilever.setVisible)
        row.addWidget(cantilever_check)

        for key, label in _PANEL_LABELS:
            check = QCheckBox(label)
            check.setChecked(True)
            check.toggled.connect(lambda shown, k=key: self._set_panel_visible(k, shown))
            self._checks[key] = check
            row.addWidget(check)
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
            self._p_nea.setTitle("NEA(f) - not available (use the photodiode detector)")
            return
        self._p_nea.setTitle("NEA(f) with shot / RIN / Johnson plateaus")
        scale = 1.0e6 / _G0
        self._nea_total.setData(nea.freq_hz.tolist(), (nea.nea_density * scale).tolist())
        colors = {"shot": "#d62728", "rin": "#2ca02c", "johnson": "#9467bd"}
        f_lo, f_hi = float(nea.freq_hz[0]), float(nea.freq_hz[-1])
        for key, color in colors.items():
            level = nea.contributions.get(key, 0.0) * scale
            if level > 0.0:
                self._p_nea.plot(
                    [f_lo, f_hi],
                    [level, level],
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                    name=key,
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
        self._nea_total = self._p_nea.plot([], [], pen=pg.mkPen("#000000", width=2), name="total")
