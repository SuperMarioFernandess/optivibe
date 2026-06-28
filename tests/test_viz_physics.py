"""Head-less tests for the reference-curve figures (task S7-mod §5; SW-09).

The physics tab's design curves are pure :class:`matplotlib.figure.Figure`
producers (no Qt, no pyplot), so they build under the Agg backend without a
display. These cover ``f1(L)``, ``|H_lat(f)|`` and the shape-agnostic
``eta(dx)`` for the cylinder variant and each demo reflector shape.
"""

from __future__ import annotations

import pytest

from optivibe.core.config.loader import load_constants, load_variant
from optivibe.mechanics.cantilever import CantileverModel
from optivibe.optics.reflector import build_reflector_model
from optivibe.viz.physics import (
    plot_first_mode_vs_length,
    plot_lateral_transfer,
    plot_reflector_eta_vs_dx,
)


def test_first_mode_vs_length_builds() -> None:
    consts = load_constants()
    variant = load_variant("B")
    fig = plot_first_mode_vs_length(consts, variant.length_m)
    assert fig.get_axes()


def test_lateral_transfer_builds() -> None:
    consts = load_constants()
    variant = load_variant("B")
    model = CantileverModel.from_config(consts, variant)
    fig = plot_lateral_transfer(
        model, f_min_hz=variant.band.f_min_hz, f_max_hz=variant.band.f_max_hz
    )
    ax = fig.get_axes()[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


@pytest.mark.parametrize("name", ["B", "sphere_demo", "plane_demo", "wedge_demo"])
def test_reflector_eta_vs_dx_builds_for_all_shapes(name: str) -> None:
    variant = load_variant(name)
    model = build_reflector_model(variant)
    sigma = float(getattr(model, "sigma_m", 0.0))
    bias = float(getattr(model, "bias_m", variant.optics.bias_offset_m))
    span = 3.0 * sigma + bias if sigma > 0.0 else 6.0 * variant.optics.bias_offset_m
    span = max(span, 8.0 * variant.optics.mode_field_radius_m)
    fig = plot_reflector_eta_vs_dx(
        model,
        span_m=span,
        eta0=model.eta_working_point(),
        bias_offset_m=variant.optics.bias_offset_m,
        shape=variant.reflector.shape,
    )
    assert fig.get_axes()
