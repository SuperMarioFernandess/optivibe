"""Smoke test: every module imports and registries are populated."""

from __future__ import annotations

import importlib

import pytest

CORE_MODULES = [
    "optivibe",
    "optivibe.core",
    "optivibe.core.types",
    "optivibe.core.units",
    "optivibe.core.logging",
    "optivibe.core.registry",
    "optivibe.core.stages",
    "optivibe.core.config",
    "optivibe.core.config.models",
    "optivibe.core.config.loader",
    "optivibe.excitation",
    "optivibe.mechanics",
    "optivibe.optics",
    "optivibe.detector",
    "optivibe.dsp",
    "optivibe.pipeline",
    "optivibe.pipeline.orchestrator",
    "optivibe.analysis",
    "optivibe.viz",
    "optivibe.io",
    "optivibe.cli",
    "optivibe.cli.main",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_modules_import(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_stage_registries_have_stub_keys() -> None:
    from optivibe.detector import DETECTOR_REGISTRY
    from optivibe.dsp import DSP_REGISTRY
    from optivibe.excitation import EXCITATION_REGISTRY
    from optivibe.mechanics import MECHANICS_REGISTRY
    from optivibe.optics import OPTICS_REGISTRY

    assert "sine" in EXCITATION_REGISTRY
    assert "stub" in MECHANICS_REGISTRY
    assert "stub" in OPTICS_REGISTRY
    assert "stub" in DETECTOR_REGISTRY
    assert "stub" in DSP_REGISTRY


def test_gui_imports_optional() -> None:
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    for module_name in (
        "optivibe.gui",
        "optivibe.gui.app",
        "optivibe.gui.main_window",
        "optivibe.gui.workers",
        "optivibe.gui.controllers",
        "optivibe.gui.widgets",
    ):
        assert importlib.import_module(module_name) is not None
