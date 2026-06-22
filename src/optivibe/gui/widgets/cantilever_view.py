"""Live cantilever-bending animation (PyQtGraph; task S7 §3).

Plays back the fiber bend from a :class:`~optivibe.core.types.TipState`: the
*unit* first mode shape ``phi_1(z/L)`` (the pure helper
:func:`optivibe.mechanics.first_mode_shape`) is scaled by the tip displacement
``dx(t)`` to draw the deflected centreline; because the unit shape's tip slope is
the rigid coupling ``1.377`` (doc 04 §2), the drawn tip tangent automatically
carries ``theta_y(t) = 1.377 dx / L``. The view computes **no physics** -- it only
scales a precomputed unit shape (SW-09). Real deflections are nm--um, so the draw
is exaggerated by a clearly-labelled factor for visibility; the live readout
shows the true ``dx`` / ``theta_y``. A :class:`~PySide6.QtCore.QTimer` drives a
decimated frame index (play / pause / speed).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from optivibe.core.types import FloatArray, TipState
from optivibe.mechanics import first_mode_shape

__all__ = ["CantileverView"]

_N_NODES = 80
_MAX_FRAMES = 600
_TARGET_TIP_FRACTION = 0.18  # exaggerated tip deflection as a fraction of L
_TIMER_MS = 33  # ~30 fps


class CantileverView(QWidget):
    """Animated side view of the bending cantilever.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "position along fiber z", units="mm")
        self._plot.setLabel("left", "lateral deflection (exaggerated)", units="mm")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.addLine(x=0.0, pen=pg.mkPen("#888", width=4))  # clamp wall
        self._beam = self._plot.plot([], [], pen=pg.mkPen("#1f77b4", width=3))
        self._axis = self._plot.plot([], [], pen=pg.mkPen("#d62728", width=2))
        self._tip = pg.ScatterPlotItem(size=9, brush=pg.mkBrush("#d62728"))
        self._plot.addItem(self._tip)

        self._play_button = QPushButton("Pause")
        self._play_button.setCheckable(True)
        self._play_button.setChecked(True)
        self._play_button.clicked.connect(self._on_play_toggled)
        self._speed = QSlider()
        self._speed.setOrientation(pg.QtCore.Qt.Orientation.Horizontal)
        self._speed.setMinimum(1)
        self._speed.setMaximum(20)
        self._speed.setValue(4)
        self._readout = QLabel("tip: -")

        controls = QHBoxLayout()
        controls.addWidget(self._play_button)
        controls.addWidget(QLabel("speed"))
        controls.addWidget(self._speed, stretch=1)
        controls.addWidget(self._readout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, stretch=1)
        layout.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.setInterval(_TIMER_MS)
        self._timer.timeout.connect(self._tick)

        self._z_mm: FloatArray = np.zeros(0)
        self._shape: FloatArray = np.zeros(0)
        self._dx: FloatArray = np.zeros(0)
        self._theta: FloatArray = np.zeros(0)
        self._exaggeration = 1.0
        self._length_mm = 1.0
        self._frame = 0

    def set_motion(self, tip: TipState, beta1_l: float, length_m: float) -> None:
        """Load a tip-state trajectory and (re)start the animation.

        Parameters
        ----------
        tip : TipState
            Tip-state time series (``dx`` / ``theta_y`` drive the bend).
        beta1_l : float
            First eigenvalue ``beta_1 * L`` (``constants.universal.beta1_l``).
        length_m : float
            Cantilever length ``L``, m.
        """
        length_mm = length_m * 1.0e3
        xi = np.linspace(0.0, 1.0, _N_NODES)
        self._shape = first_mode_shape(xi, beta1_l)
        self._z_mm = xi * length_mm
        self._length_mm = length_mm

        dx = np.asarray(tip.dx, dtype=np.float64)
        theta = np.asarray(tip.theta_y, dtype=np.float64)
        stride = max(1, dx.size // _MAX_FRAMES)
        self._dx = dx[::stride]
        self._theta = theta[::stride]

        peak = float(np.max(np.abs(self._dx))) if self._dx.size else 0.0
        # Map the peak true deflection (m) to a fraction of L (mm) on screen.
        target_mm = _TARGET_TIP_FRACTION * length_mm
        self._exaggeration = (target_mm / (peak * 1.0e3)) if peak > 0.0 else 0.0

        span = max(target_mm, 1e-6)
        self._plot.setXRange(-0.05 * length_mm, 1.18 * length_mm)
        self._plot.setYRange(-1.4 * span, 1.4 * span)
        self._frame = 0
        self._play_button.setChecked(True)
        self._play_button.setText("Pause")
        self._timer.start()
        self._draw_frame(0)

    def clear_motion(self) -> None:
        """Stop playback and clear the curves."""
        self._timer.stop()
        self._beam.setData([], [])
        self._axis.setData([], [])
        self._tip.setData([], [])
        self._readout.setText("tip: -")

    def _on_play_toggled(self, playing: bool) -> None:
        """Play/pause handler for the toggle button."""
        if playing and self._dx.size:
            self._play_button.setText("Pause")
            self._timer.start()
        else:
            self._play_button.setText("Play")
            self._timer.stop()

    def _tick(self) -> None:
        """Advance the frame index by the current speed and redraw."""
        if not self._dx.size:
            return
        self._frame = (self._frame + int(self._speed.value())) % self._dx.size
        self._draw_frame(self._frame)

    def _draw_frame(self, index: int) -> None:
        """Draw the bent beam and tip tangent for sample ``index``."""
        if not self._dx.size or self._exaggeration == 0.0:
            return
        dx = float(self._dx[index])
        deflection_mm = self._exaggeration * dx * 1.0e3 * self._shape
        self._beam.setData(self._z_mm, deflection_mm)
        tip_z = self._z_mm[-1]
        tip_y = float(deflection_mm[-1])
        self._tip.setData([tip_z], [tip_y])
        # Short tangent at the tip (shows the endface tilt direction).
        slope = (deflection_mm[-1] - deflection_mm[-2]) / (self._z_mm[-1] - self._z_mm[-2])
        seg_z = np.array([tip_z, tip_z + 0.14 * self._length_mm])
        seg_y = np.array([tip_y, tip_y + slope * 0.14 * self._length_mm])
        self._axis.setData(seg_z, seg_y)
        theta_urad = float(self._theta[index]) * 1.0e6
        self._readout.setText(
            f"tip dx = {dx * 1e9:+.2f} nm, theta_y = {theta_urad:+.2f} urad  "
            f"(x{self._exaggeration:.0f} exaggerated)"
        )
