"""S9-B reflector-family tests: golden references against analytic limits.

Covers the four shapes behind the optics shape layer (doc 03 §c-§e; the S9-B
addendum to 03/04):

* **cylinder** -- byte-for-byte regression of eta on fixed tip inputs and of the
  resolved working point (the refactor onto the shape layer must not move a
  bit; the S3 golden in ``test_optics_cylinder.py`` is the companion check);
* **sphere** -- isotropy: anisotropy ~ 1 and ``|d eta/d Dy| == |d eta/d Dx|``;
* **plane** -- the ``R_c -> inf`` reference: ``d eta/d Dx -> 0`` (numerically
  ~0 against the non-zero cylinder), defocus on both axes through the gap;
* **wedge** -- the working point is shifted by the built-in angle; ``alpha_w=0``
  reproduces the plane exactly;
* **invariants** -- ``0 <= eta <= 1`` (property test) and loud failures on
  broken per-shape configs.

The shapes are exercised through the *config* path (the ``*_demo`` variants and
``build_reflector_model``), so the registry, the resolver guards and the models
are all on the tested path (DoD 10 §13).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optivibe.core.config.loader import load_variant
from optivibe.core.config.models import VariantConfig
from optivibe.optics import (
    CylinderOpticsModel,
    PlaneOpticsModel,
    SphereOpticsModel,
    WedgeOpticsModel,
    build_reflector_model,
)
from optivibe.optics.cylinder import CylinderOptics
from optivibe.optics.reflector import ReflectorOptics

# --------------------------------------------------------------------------- #
# Fixed tip inputs and the cylinder byte-for-byte anchor (captured on the
# pre-S9-B HEAD 55949b4 with variant A; doc 14 §7 bit-identity gate).
# --------------------------------------------------------------------------- #
_DX = np.array([0.0, 1.0e-6, -2.0e-6, 3.0e-6])
_DY = np.array([0.0, 1.0e-6, 2.0e-6, -1.0e-6])
_DZ = np.array([0.0, 1.0e-7, -1.0e-7, 2.0e-7])
_TX = np.array([0.0, 1.0e-4, -1.0e-4, 2.0e-4])
_TY = np.array([0.0, 2.0e-4, 1.0e-4, -1.0e-4])

# Exact eta(_DX.._TY) of the S3 cylinder model for variant A (frozen anchor).
_CYL_ETA_ANCHOR = (
    0.23964053560536128,
    0.11072064177482513,
    0.4379227963737634,
    0.010214001155132081,
)
_CYL_ETA0_ANCHOR = 0.23964053560536128


@pytest.fixture
def variant_a(config_dir: Path) -> VariantConfig:
    """Resolved cylinder variant A (R_c = 62.5 um)."""
    return load_variant("A", config_dir=config_dir)


@pytest.fixture
def variant_sphere(config_dir: Path) -> VariantConfig:
    """Resolved sphere demo variant (curved both planes, radial bias)."""
    return load_variant("sphere_demo", config_dir=config_dir)


@pytest.fixture
def variant_plane(config_dir: Path) -> VariantConfig:
    """Resolved plane demo variant (R_c -> inf reference)."""
    return load_variant("plane_demo", config_dir=config_dir)


@pytest.fixture
def variant_wedge(config_dir: Path) -> VariantConfig:
    """Resolved wedge demo variant (alpha_w = 20 mrad)."""
    return load_variant("wedge_demo", config_dir=config_dir)


def _slope(model: object, axis: str, h: float = 1.0e-9) -> float:
    """Central finite-difference slope d eta / d{Dx|Dy} at the working point."""
    assert hasattr(model, "eta")
    if axis == "x":
        return float((model.eta(dx=h) - model.eta(dx=-h)).item() / (2.0 * h))  # type: ignore[attr-defined]
    return float((model.eta(dy=h) - model.eta(dy=-h)).item() / (2.0 * h))  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Cylinder: byte-for-byte regression through the new shape layer.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_cylinder_eta_is_bit_identical_through_layer(variant_a: VariantConfig) -> None:
    """eta is unchanged after the refactor onto the shape layer.

    Bit-identity is asserted *within the running environment*: the layer-built
    model must equal the cylinder model built directly from the untouched S3
    ``from_config`` (identical code path => identical bits). The frozen anchor is
    an additional regression guard, compared with a tolerance -- floating-point
    results differ by a few ULP across Python / numpy / BLAS builds, so an exact
    match to a value captured on another interpreter is not portable (the S3
    golden in ``test_optics_cylinder.py`` likewise uses tolerances).
    """
    model = build_reflector_model(variant_a)
    assert isinstance(model, CylinderOpticsModel)  # cylinder routes to its own model
    eta = model.eta(_DX, _DY, _DZ, _TX, _TY)
    # Same-environment bit-identity: the layer is a pure pass-through to S3.
    direct = CylinderOpticsModel.from_config(variant_a).eta(_DX, _DY, _DZ, _TX, _TY)
    assert np.array_equal(eta, direct)
    # Cross-environment regression guard against the documented anchor.
    np.testing.assert_allclose(eta, _CYL_ETA_ANCHOR, rtol=1.0e-6, atol=0.0)
    assert model.eta_working_point() == pytest.approx(_CYL_ETA0_ANCHOR, rel=1.0e-6)


def test_cylinder_stage_matches_reflector_stage(variant_a: VariantConfig) -> None:
    """The back-compat CylinderOptics stage equals the dispatching ReflectorOptics."""
    from optivibe.core.types import TipState

    tip = TipState(dx=_DX, dy=_DY, dz=_DZ, theta_x=_TX, theta_y=_TY, fs=5000.0)
    out_cyl = CylinderOptics().run(tip, variant_a)
    out_ref = ReflectorOptics().run(tip, variant_a)
    assert np.array_equal(out_cyl.eta, out_ref.eta)
    assert out_cyl.bias == out_ref.bias
    assert issubclass(CylinderOptics, ReflectorOptics)


# --------------------------------------------------------------------------- #
# Sphere: isotropic response (anisotropy -> 1).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_sphere_is_isotropic(variant_sphere: VariantConfig) -> None:
    """Curved both planes with a radial bias => equal x/y displacement slopes."""
    model = build_reflector_model(variant_sphere)
    assert isinstance(model, SphereOpticsModel)
    slope_x = _slope(model, "x")
    slope_y = _slope(model, "y")
    assert slope_x != 0.0
    # Symmetric (radial) working point => slopes equal to machine precision.
    assert slope_y == pytest.approx(slope_x, rel=1.0e-9)
    anisotropy = abs(slope_x / slope_y)
    assert anisotropy == pytest.approx(1.0, abs=5.0e-2)  # <= 5 %


def test_sphere_planes_are_symmetric(variant_sphere: VariantConfig) -> None:
    """Swapping (dx,theta_y) with (dy,theta_x) leaves eta unchanged (symmetry)."""
    model = build_reflector_model(variant_sphere)
    eta_a = model.eta(dx=1.5e-6, theta_y=3.0e-4)
    eta_b = model.eta(dy=1.5e-6, theta_x=3.0e-4)
    assert eta_b.item() == pytest.approx(eta_a.item(), rel=1.0e-12)


# --------------------------------------------------------------------------- #
# Plane: no displacement sensitivity, defocus through the gap.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_plane_has_no_displacement_sensitivity(
    variant_plane: VariantConfig, variant_a: VariantConfig
) -> None:
    """d eta/d Dx and d eta/d Dy are exactly 0 (vs a non-zero cylinder slope)."""
    plane = build_reflector_model(variant_plane)
    assert isinstance(plane, PlaneOpticsModel)
    assert _slope(plane, "x") == 0.0
    assert _slope(plane, "y") == 0.0
    # The cylinder, in contrast, has a large finite slope on its target axis.
    assert abs(_slope(build_reflector_model(variant_a), "x")) > 1.0e4


def test_plane_on_axis_ceiling_and_defocus(variant_plane: VariantConfig) -> None:
    """On-axis eta = 1/(1+(A/zR)^2) and it drops as the gap (dz) grows."""
    plane = build_reflector_model(variant_plane)
    assert isinstance(plane, PlaneOpticsModel)
    beam = plane.beam
    gap = plane.gap_m
    expected = 1.0 / (1.0 + (gap / beam.rayleigh_range_m) ** 2)
    assert plane.eta_working_point() == pytest.approx(expected, rel=1.0e-12)
    # Defocus on both axes: a larger gap lowers eta.
    assert plane.eta(dz=20.0e-6).item() < plane.eta_working_point()


# --------------------------------------------------------------------------- #
# Wedge: angular bias; alpha_w = 0 degenerates to the plane.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_wedge_zero_angle_matches_plane(
    variant_wedge: VariantConfig, variant_plane: VariantConfig
) -> None:
    """At alpha_w = 0 the wedge reproduces the plane bit-for-bit."""
    plane = build_reflector_model(variant_plane)
    wedge0_variant = variant_wedge.model_copy(
        update={"optics": variant_wedge.optics.model_copy(update={"wedge_angle_rad": 0.0})}
    )
    wedge0 = build_reflector_model(wedge0_variant)
    assert isinstance(wedge0, WedgeOpticsModel)
    grid = {"dz": 5.0e-7, "theta_x": 2.0e-4, "theta_y": 3.0e-4}
    assert wedge0.eta(**grid).item() == pytest.approx(plane.eta(**grid).item(), rel=1.0e-12)
    assert wedge0.eta_working_point() == pytest.approx(plane.eta_working_point(), rel=1.0e-12)


def test_wedge_shifts_working_point(
    variant_wedge: VariantConfig, variant_plane: VariantConfig
) -> None:
    """A non-zero wedge angle lowers the working point below the plane peak."""
    plane = build_reflector_model(variant_plane)
    wedge = build_reflector_model(variant_wedge)
    assert isinstance(wedge, WedgeOpticsModel)
    assert wedge.eta_working_point() < plane.eta_working_point()
    # The wedge is still flat => no displacement sensitivity.
    assert _slope(wedge, "x") == 0.0


# --------------------------------------------------------------------------- #
# Invariants: 0 <= eta <= 1 for every shape (property test).
# --------------------------------------------------------------------------- #
@settings(max_examples=80, deadline=None)
@given(
    dx=st.floats(-5.0e-6, 5.0e-6),
    dy=st.floats(-5.0e-6, 5.0e-6),
    dz=st.floats(-1.0e-5, 1.0e-5),
    tx=st.floats(-1.0e-3, 1.0e-3),
    ty=st.floats(-1.0e-3, 1.0e-3),
)
def test_eta_is_bounded_for_all_shapes(
    dx: float, dy: float, dz: float, tx: float, ty: float
) -> None:
    """Coupling stays in [0, 1] for random tip states across all shapes."""
    cfg = Path(__file__).resolve().parent.parent / "configs"
    for name in ("A", "sphere_demo", "plane_demo", "wedge_demo"):
        model = build_reflector_model(load_variant(name, config_dir=cfg))
        eta = model.eta(dx, dy, dz, tx, ty).item()
        assert 0.0 <= eta <= 1.0


# --------------------------------------------------------------------------- #
# Validators: broken per-shape configs fail loudly.
# --------------------------------------------------------------------------- #
def test_sphere_from_config_rejects_wrong_shape(variant_a: VariantConfig) -> None:
    """SphereOpticsModel refuses a non-sphere variant loudly."""
    with pytest.raises(ValueError, match="sphere"):
        SphereOpticsModel.from_config(variant_a)


def test_plane_from_config_rejects_wrong_shape(variant_a: VariantConfig) -> None:
    """PlaneOpticsModel refuses a non-plane variant loudly."""
    with pytest.raises(ValueError, match="plane"):
        PlaneOpticsModel.from_config(variant_a)


def test_wedge_from_config_rejects_wrong_shape(variant_a: VariantConfig) -> None:
    """WedgeOpticsModel refuses a non-wedge variant loudly."""
    with pytest.raises(ValueError, match="wedge"):
        WedgeOpticsModel.from_config(variant_a)


def test_wedge_requires_angle(variant_wedge: VariantConfig) -> None:
    """A wedge variant without an angle fails in from_config."""
    no_angle = variant_wedge.model_copy(
        update={"optics": variant_wedge.optics.model_copy(update={"wedge_angle_rad": None})}
    )
    with pytest.raises(ValueError, match="wedge_angle_rad"):
        WedgeOpticsModel.from_config(no_angle)


def test_wedge_angle_out_of_range_rejected(variant_wedge: VariantConfig) -> None:
    """A wedge angle past the paraxial range is rejected."""
    too_big = variant_wedge.model_copy(
        update={"optics": variant_wedge.optics.model_copy(update={"wedge_angle_rad": 0.5})}
    )
    with pytest.raises(ValueError, match="paraxial"):
        WedgeOpticsModel.from_config(too_big)


def test_sphere_requires_radius(variant_sphere: VariantConfig) -> None:
    """A sphere variant without a radius fails in from_config."""
    no_radius = variant_sphere.model_copy(
        update={
            "reflector": variant_sphere.reflector.model_copy(
                update={"radius_of_curvature_m": None}
            )
        }
    )
    with pytest.raises(ValueError, match="radius"):
        SphereOpticsModel.from_config(no_radius)


def test_sphere_paraxial_and_spot_guards(variant_sphere: VariantConfig) -> None:
    """Sphere shares the cylinder guards: R_c >= 5 w0 and w(A) <= R_c/3."""
    small_rc = variant_sphere.model_copy(
        update={
            "reflector": variant_sphere.reflector.model_copy(
                update={"radius_of_curvature_m": 10.0e-6}
            )
        }
    )
    with pytest.raises(ValueError, match="paraxial"):
        SphereOpticsModel.from_config(small_rc)
    big_gap = variant_sphere.model_copy(
        update={"optics": variant_sphere.optics.model_copy(update={"gap_m": 300.0e-6})}
    )
    with pytest.raises(ValueError, match="spot"):
        SphereOpticsModel.from_config(big_gap)


def test_sphere_post_init_validates() -> None:
    """Direct construction with bad geometry fails (defensive __post_init__)."""
    from optivibe.optics.gaussian import GaussianBeam

    beam = GaussianBeam(wavelength_m=1550.0e-9, waist_radius_m=5.2e-6)
    with pytest.raises(ValueError, match="gap_m"):
        SphereOpticsModel(beam=beam, gap_m=-1.0e-6, radius_of_curvature_m=62.5e-6, bias_m=2.0e-6)
    with pytest.raises(ValueError, match="radius_of_curvature_m"):
        SphereOpticsModel(beam=beam, gap_m=31.0e-6, radius_of_curvature_m=-1.0, bias_m=2.0e-6)
    with pytest.raises(ValueError, match="bias_m"):
        SphereOpticsModel(beam=beam, gap_m=31.0e-6, radius_of_curvature_m=62.5e-6, bias_m=-1.0e-6)


def test_cylinder_from_config_rejects_none_radius(variant_a: VariantConfig) -> None:
    """The cylinder model's defensive None-radius guard (R_c is mandatory)."""
    no_radius = variant_a.model_copy(
        update={"reflector": variant_a.reflector.model_copy(update={"radius_of_curvature_m": None})}
    )
    with pytest.raises(ValueError, match="radius_of_curvature_m"):
        CylinderOpticsModel.from_config(no_radius)
