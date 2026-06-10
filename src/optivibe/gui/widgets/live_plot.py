"""A PyQtGraph widget that shows the reconstructed signal and its spectrum.

In S0 this is a placeholder: it plots the reconstructed acceleration time series
(top) and the amplitude spectrum (bottom) carried by a
:class:`~optivibe.core.types.VibrationResult`. The richer, interactive plots
(velocity/displacement, ISO bands, cross-axis) arrive with the analytics in S6;
their non-interactive figure logic will live in :mod:`optivibe.viz`.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget

from optivibe.core.types import VibrationResult

__all__ = ["LivePlotWidget"]


class LivePlotWidget(pg.GraphicsLayoutWidget):
    """Two stacked plots: reconstructed acceleration and its spectrum.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._accel_plot = self.addPlot(row=0, col=0, title="Reconstructed acceleration")
        self._accel_plot.setLabel("bottom", "time", units="s")
        self._accel_plot.setLabel("left", "a (uncalibrated)")
        self._accel_plot.showGrid(x=True, y=True, alpha=0.3)

        self._spectrum_plot = self.addPlot(row=1, col=0, title="Amplitude spectrum")
        self._spectrum_plot.setLabel("bottom", "frequency", units="Hz")
        self._spectrum_plot.setLabel("left", "amplitude")
        self._spectrum_plot.showGrid(x=True, y=True, alpha=0.3)

    def clear_plots(self) -> None:
        """Remove any plotted curves from both panels."""
        self._accel_plot.clear()
        self._spectrum_plot.clear()

    def show_result(self, result: VibrationResult) -> None:
        """Plot a reconstructed result.

        Parameters
        ----------
        result : VibrationResult
            Reconstructed vibration (its acceleration and spectrum are plotted).
        """
        self.clear_plots()
        time_axis = [i / result.fs for i in range(result.n_samples)]
        self._accel_plot.plot(time_axis, result.a.tolist(), pen=pg.mkPen(width=1))
        if result.spectrum is not None:
            self._spectrum_plot.plot(
                result.spectrum.freq.tolist(),
                result.spectrum.values.tolist(),
                pen=pg.mkPen(width=1),
            )
