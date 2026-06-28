"""Reusable GUI widgets: live PyQtGraph panels and matplotlib figure embedding."""

from optivibe.gui.widgets.analysis_tabs import MonteCarloPanel, ReportPanel, SweepPanel
from optivibe.gui.widgets.cantilever_view import CantileverView
from optivibe.gui.widgets.control_panel import ControlPanel
from optivibe.gui.widgets.excitation_builder import ExcitationBuilder
from optivibe.gui.widgets.live_view import LiveView
from optivibe.gui.widgets.mpl_canvas import MplFigureView
from optivibe.gui.widgets.physics_tab import PhysicsTab
from optivibe.gui.widgets.subsystem_forms import SystemBuilderPanel

__all__ = [
    "CantileverView",
    "ControlPanel",
    "ExcitationBuilder",
    "LiveView",
    "MonteCarloPanel",
    "MplFigureView",
    "PhysicsTab",
    "ReportPanel",
    "SweepPanel",
    "SystemBuilderPanel",
]
