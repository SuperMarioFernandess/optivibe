"""R-59 golden: full overlap of the mismatched returned beam (doc 03 §d, §d', §d'').

The returned beam of a convex (or flat) mirror is *not* matched to the fiber
mode (``q_out != i zR``); its overlap with the mode carries a cross term
``C_da d alpha`` that the former multiplier ``exp[-(d/w0)^2 - (alpha/theta0)^2]``
lacked (R-59, chat K5-PHYS, S-30). Mandatory set of backlog card S-30:

1. the matched limit ``q_out -> q_in`` reproduces the former multiplier identically;
2. ``sigma_eff`` of the code equals the closed form and ``sigma_base / eta_par^x``;
3. sign of the cross term: convex / concave mirror against ABCD ``f = -/+ R_c/2``;
4. flat plane, plane and wedge: the tilt factor does not depend on the gap;
5. one point against an independent numerical overlap integral over the
   parabolic sag (identity to ~1e-12).

Plus the closed form of the curved plane as an exact special case of the general
overlap (cylinder and sphere, with a gap change).

References (plan 18 §5(g)) are never the model output: the closed formulas of
doc 03 §d evaluated here from the model *inputs* (lambda, w0, A, R_c); the
mirror-plane form of doc 03 §d', ``eta = |∫ u^2 exp(-2ikh) dx|^2 / (∫ |u|^2 dx)^2``
(reciprocity), evaluated by direct quadrature; and the values tabulated in doc 03
§d' / §d''. The statement is about the model; L3 is not passed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from optivibe.core.config.loader import load_variant
from optivibe.core.types import FloatArray
from optivibe.optics import (
    CylinderOpticsModel,
    GaussianBeam,
    PlaneOpticsModel,
    SphereOpticsModel,
    WedgeOpticsModel,
    eta_parallel_curved,
    misalignment_factor,
    overlap_coefficients,
    round_trip_q,
)

WAVELENGTH_M = 1550.0e-9  # doc 03 §a (variants A-D)
WAIST_M = 5.2e-6  # w0 of SMF-28e+ at 1550 nm, doc 03 §a
UM = 1.0e-6


@pytest.fixture(scope="module")
def beam() -> GaussianBeam:
    """Fiber mode of doc 03 §a (1550 nm, w0 = 5.2 um)."""
    return GaussianBeam(wavelength_m=WAVELENGTH_M, waist_radius_m=WAIST_M)


def _rayleigh(beam: GaussianBeam) -> float:
    """``zR = pi w0^2 / lambda`` (doc 03 §a), from the beam inputs."""
    return math.pi * beam.waist_radius_m**2 / beam.wavelength_m


def _theta0(beam: GaussianBeam) -> float:
    """``theta0 = lambda / (pi w0)`` (doc 03 §a), from the beam inputs."""
    return beam.wavelength_m / (math.pi * beam.waist_radius_m)


def _eta_par_curved_closed(beam: GaussianBeam, gap: float, radius: float) -> float:
    """``eta_par^x = R_c zR / sqrt((g^2 + zR^2)((R_c + g)^2 + zR^2))`` (doc 03 §d)."""
    z_r = _rayleigh(beam)
    return radius * z_r / math.sqrt((gap**2 + z_r**2) * ((radius + gap) ** 2 + z_r**2))


def _eta_par_flat_closed(beam: GaussianBeam, gap: float) -> float:
    """``eta_par^y = 1/sqrt(1 + (g/zR)^2)`` (doc 03 §d)."""
    return 1.0 / math.sqrt(1.0 + (gap / _rayleigh(beam)) ** 2)


def _sigma_eff_closed(beam: GaussianBeam, gap: float, radius: float) -> float:
    """``sigma_eff = (w0/2) sqrt(1 + ((R_c + g)/zR)^2) = w(R_c + g)/2`` (doc 03 §d, R-59)."""
    return 0.5 * beam.waist_radius_m * math.sqrt(1.0 + ((radius + gap) / _rayleigh(beam)) ** 2)


def _mirror_plane_overlap(
    beam: GaussianBeam, gap: float, sag: Callable[[FloatArray], FloatArray]
) -> float:
    """1-D overlap in the mirror plane by direct quadrature (doc 03 §d').

    ``u(x) = exp(-i k x^2 / (2 q1))`` with ``q1 = g + i zR`` is the emitted field at
    the mirror; ``h(x)`` is the surface sag (convex: ``(x - D)^2/(2 R_c) >= 0``) and
    the reflection adds the phase ``exp(-2ikh)``. No ABCD matrix and no
    misalignment formula enters this reference.
    """
    k = 2.0 * math.pi / beam.wavelength_m
    q1 = gap + 1j * _rayleigh(beam)
    w1 = beam.waist_radius_m * math.sqrt(1.0 + (gap / _rayleigh(beam)) ** 2)
    x = np.linspace(-12.0 * w1, 12.0 * w1, 400_001)
    u = np.exp(-1j * k * x**2 / (2.0 * q1))
    numerator = abs(np.trapezoid(u**2 * np.exp(-2j * k * sag(x)), x)) ** 2
    denominator = float(np.trapezoid(np.abs(u) ** 2, x)) ** 2
    return float(numerator / denominator)


# --------------------------------------------------------------------------- #
# (1) Matched limit.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_matched_limit_reproduces_former_multiplier(beam: GaussianBeam) -> None:
    """``q_out -> q_in``: ``C_dd = 1/w0^2``, ``C_da = 0``, ``C_aa = 1/theta0^2`` (doc 03 §d).

    The former multiplier ``exp[-(d/w0)^2 - (alpha/theta0)^2]`` is exactly the
    matched special case of the general overlap -- identically, not approximately.
    """
    q_in = 1j * _rayleigh(beam)
    c_dd, c_da, c_aa = overlap_coefficients(beam, q_in)
    w0, theta0 = beam.waist_radius_m, _theta0(beam)
    assert beam.rayleigh_range_m == pytest.approx(_rayleigh(beam), rel=1.0e-15)
    assert beam.divergence_rad == pytest.approx(theta0, rel=1.0e-15)
    assert float(c_dd) == pytest.approx(1.0 / w0**2, rel=1.0e-13)
    assert float(c_aa) == pytest.approx(1.0 / theta0**2, rel=1.0e-13)
    assert abs(float(c_da)) <= 1.0e-12 * math.sqrt(float(c_dd) * float(c_aa))
    offset = np.linspace(-3.0 * UM, 3.0 * UM, 13)[:, None]
    angle = np.linspace(-0.05, 0.05, 11)[None, :]
    np.testing.assert_allclose(
        misalignment_factor(beam, offset, angle, q_out=q_in),
        np.exp(-((offset / w0) ** 2) - (angle / theta0) ** 2),
        rtol=1.0e-13,
    )


def test_misalignment_factor_requires_q_out(beam: GaussianBeam) -> None:
    """The returned-beam parameter is mandatory and keyword-only (S-30 API decision).

    A call in the former three-argument form must fail loudly rather than
    silently evaluate the matched limit -- exactly the error class R-59 fixed
    (doc 10 §7).
    """
    with pytest.raises(TypeError):
        misalignment_factor(beam, 1.0 * UM, 0.01)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        misalignment_factor(beam, 1.0 * UM, 0.01, 1j)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# (2) sigma_eff: closed form and sigma_base / eta_par^x.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_sigma_eff_closed_form_and_identity(beam: GaussianBeam, config_dir: Path) -> None:
    """``sigma_m`` = ``(w0/2) sqrt(1 + ((R_c + A)/zR)^2)`` = ``sigma_base / eta_par^x`` (03 §d).

    ``sigma_base = (w0/2) R_c / sqrt(A^2 + zR^2)`` is the width of the revoked
    matched-beam multiplier. Tabulated references: the formula column of the
    doc 03 §d' check table (4.738 / 5.142 / 5.556 / 3.926 / 7.844 um at
    (A, R_c) = (21, 62.5) / (31, 62.5) / (41, 62.5) / (31, 31) / (31, 125) um),
    the variants A and B/C/D of doc 03 §d'' (5.1415 / 3.9257 um) and the
    "anatomy" of doc 03 §d' (contributions to 1/sigma^2 per unit Delta_eff^2 at
    A = 31, R_c = 62.5 um: 0.0369 / -0.1113 / 0.1122 um^-2 mismatched,
    0.0364 / 0 / 0.1138 um^-2 matched).
    """
    table = (
        (21.0 * UM, 62.5 * UM, 4.738 * UM),
        (31.0 * UM, 62.5 * UM, 5.142 * UM),
        (41.0 * UM, 62.5 * UM, 5.556 * UM),
        (31.0 * UM, 31.0 * UM, 3.926 * UM),
        (31.0 * UM, 125.0 * UM, 7.844 * UM),
    )
    z_r = _rayleigh(beam)
    for gap, radius, reference in table:
        model = CylinderOpticsModel(beam=beam, gap_m=gap, radius_of_curvature_m=radius, bias_m=0.0)
        closed = _sigma_eff_closed(beam, gap, radius)
        sigma_base = 0.5 * beam.waist_radius_m * radius / math.sqrt(gap**2 + z_r**2)
        assert model.sigma_m == pytest.approx(closed, rel=1.0e-12)
        assert model.sigma_m == pytest.approx(
            sigma_base / _eta_par_curved_closed(beam, gap, radius), rel=1.0e-12
        )
        assert model.sigma_m == pytest.approx(reference, abs=0.5e-3 * UM)
    for name, reference in (("A", 5.1415 * UM), ("B", 3.9257 * UM)):
        variant_model = CylinderOpticsModel.from_config(load_variant(name, config_dir=config_dir))
        assert variant_model.sigma_m == pytest.approx(reference, abs=0.5e-4 * UM)

    # Anatomy of doc 03 §d' (4 decimals; tolerance one unit of the last digit,
    # since e.g. 0.113748 is tabulated as 0.1138).
    gap, radius = 31.0 * UM, 62.5 * UM
    per_um2 = (2.0 / radius) ** 2 * 1.0e-12  # (2/R_c)^2, expressed per um^2 of Delta_eff
    q_convex = round_trip_q(beam, gap, -radius / 2.0)
    c_dd, c_da, c_aa = (float(c) for c in overlap_coefficients(beam, q_convex))
    assert c_dd * gap**2 * per_um2 == pytest.approx(0.0369, abs=1.0e-4)
    assert c_da * gap * per_um2 == pytest.approx(-0.1113, abs=1.0e-4)
    assert c_aa * per_um2 == pytest.approx(0.1122, abs=1.0e-4)
    m_dd, m_da, m_aa = (float(c) for c in overlap_coefficients(beam, 1j * z_r))
    assert m_dd * gap**2 * per_um2 == pytest.approx(0.0364, abs=1.0e-4)
    assert abs(m_da) * gap * per_um2 < 1.0e-12
    assert m_aa * per_um2 == pytest.approx(0.1138, abs=1.0e-4)


# --------------------------------------------------------------------------- #
# (3) Sign of the cross term: convex vs concave.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_cross_term_sign_convex_vs_concave(beam: GaussianBeam) -> None:
    """Convex sag <-> ABCD ``f = -R_c/2``, concave <-> ``f = +R_c/2``; ``C_da < 0`` convex.

    Doc 03 §d' sign test: the aligned overlap from the mirror-plane integral
    with the convex (``h >= 0``) and concave (``h <= 0``) parabolic sag matches the
    ABCD form with ``f = -R_c/2`` and ``f = +R_c/2`` respectively, while the
    crossed assignment misses by more than 5 %. The diverging return of the
    convex mirror has ``Im a > 0``, hence ``C_da < 0`` (and ``> 0`` for the
    concave one); a decentred convex mirror follows the overlap *with* that
    cross term, and the sign-flipped term misses by far.
    """
    gap, radius = 31.0 * UM, 62.5 * UM
    z_r = _rayleigh(beam)

    def abcd(focal: float) -> float:
        q = complex(round_trip_q(beam, gap, focal))
        return 2.0 * math.sqrt(z_r * q.imag) / abs(-1j * z_r - q)

    convex = _mirror_plane_overlap(beam, gap, lambda x: x**2 / (2.0 * radius))
    concave = _mirror_plane_overlap(beam, gap, lambda x: -(x**2) / (2.0 * radius))
    assert float(eta_parallel_curved(beam, gap, radius)) == pytest.approx(convex, rel=1.0e-10)
    assert abcd(-radius / 2.0) == pytest.approx(convex, rel=1.0e-10)
    assert abcd(+radius / 2.0) == pytest.approx(concave, rel=1.0e-10)
    assert abs(abcd(+radius / 2.0) / convex - 1.0) > 0.05
    assert abs(abcd(-radius / 2.0) / concave - 1.0) > 0.05

    q_convex = round_trip_q(beam, gap, -radius / 2.0)
    _, c_da_convex, _ = overlap_coefficients(beam, q_convex)
    _, c_da_concave, _ = overlap_coefficients(beam, round_trip_q(beam, gap, +radius / 2.0))
    assert float(c_da_convex) < 0.0 < float(c_da_concave)

    shift = 2.0 * UM
    measured = (
        _mirror_plane_overlap(beam, gap, lambda x: (x - shift) ** 2 / (2.0 * radius)) / convex
    )
    c_dd, c_da, c_aa = (float(c) for c in overlap_coefficients(beam, q_convex))
    scale = (2.0 / radius) ** 2 * shift**2  # curved-plane maps d = (2g/R_c) D, alpha = (2/R_c) D
    with_term = math.exp(-scale * (c_dd * gap**2 + c_da * gap + c_aa))
    flipped = math.exp(-scale * (c_dd * gap**2 - c_da * gap + c_aa))
    assert measured == pytest.approx(with_term, rel=1.0e-10)
    assert abs(flipped / measured - 1.0) > 0.5


# --------------------------------------------------------------------------- #
# (4) Flat plane, plane, wedge: tilt factor independent of the gap.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_flat_tilt_factor_independent_of_gap(beam: GaussianBeam) -> None:
    """``eta = eta_par(g) exp[-(2 theta/theta0)^2]``; angular width theta0/2 = 47.44 mrad.

    Doc 03 §d (R-59): on the flat-plane maps (``d = 2 g theta``,
    ``alpha = 2 theta``) the full overlap is independent of the gap -- the former
    ``exp[-(2 g theta/w0)^2 - (2 theta/theta0)^2]`` narrowed from 47.2 to 22.8 mrad
    over 5-100 um (doc 03 §d'). Checked on the plane, on the cylinder's flat axis,
    on the wedge working point (``theta -> alpha_w``) and against the mirror-plane
    integral of a tilted flat mirror (``h = theta x``).
    """
    theta0 = _theta0(beam)
    assert theta0 / 2.0 == pytest.approx(47.44e-3, abs=5.0e-6)
    tilt = 20.0e-3
    factor = math.exp(-((2.0 * tilt / theta0) ** 2))
    for gap in (5.0 * UM, 31.0 * UM, 100.0 * UM):
        flat_par = _eta_par_flat_closed(beam, gap)
        plane = PlaneOpticsModel(beam=beam, gap_m=gap)
        assert float(plane.eta(theta_y=tilt).item()) == pytest.approx(
            flat_par**2 * factor, rel=1.0e-12
        )
        cylinder = CylinderOpticsModel(
            beam=beam, gap_m=gap, radius_of_curvature_m=62.5 * UM, bias_m=2.0 * UM
        )
        _, eta_y = cylinder.eta_components(0.0, 0.0, 0.0, tilt, 0.0)
        assert float(eta_y.item()) == pytest.approx(flat_par * factor, rel=1.0e-12)
        wedge = WedgeOpticsModel(beam=beam, gap_m=gap, wedge_angle_rad=tilt)
        assert wedge.eta_working_point() == pytest.approx(flat_par**2 * factor, rel=1.0e-12)
        tilted = _mirror_plane_overlap(beam, gap, lambda x: tilt * x)
        assert tilted == pytest.approx(flat_par * factor, rel=1.0e-10)


# --------------------------------------------------------------------------- #
# (5) One point against the numerical overlap integral.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_overlap_matches_numerical_parabolic_integral(config_dir: Path) -> None:
    """Model eta_x at one tip state == mirror-plane quadrature over the parabolic sag.

    Variant A (cylinder) at ``dx = 1 um``, ``theta_y = 1e-4``: the curved-plane
    factor of the model must equal the doc 03 §d' integral with the mirror
    centre missed by ``Delta_eff = Delta x0 + dx + (R_c + A) theta_y`` (R-43), to
    ~1e-12. The same for both factors of the sphere demo (isotropic, two curved
    planes). Independent of the ABCD round trip and of the overlap coefficients.
    """
    dx, dy, theta_x, theta_y = 1.0 * UM, -0.5 * UM, 2.0e-4, 1.0e-4

    cylinder = CylinderOpticsModel.from_config(load_variant("A", config_dir=config_dir))
    gap, radius = cylinder.gap_m, cylinder.radius_of_curvature_m
    eta_x, _ = cylinder.eta_components(dx, 0.0, 0.0, 0.0, theta_y)
    d_eff = cylinder.bias_m + dx + (radius + gap) * theta_y
    reference = _mirror_plane_overlap(
        cylinder.beam, gap, lambda x: (x - d_eff) ** 2 / (2.0 * radius)
    )
    assert float(eta_x.item()) == pytest.approx(reference, rel=1.0e-12)

    sphere = _sphere_demo(config_dir)
    gap, radius = sphere.gap_m, sphere.radius_of_curvature_m
    sx, sy = sphere.eta_components(dx, dy, 0.0, theta_x, theta_y)
    dx_eff = sphere.bias_m + dx + (radius + gap) * theta_y
    dy_eff = sphere.bias_m + dy + (radius + gap) * theta_x
    ref_x = _mirror_plane_overlap(sphere.beam, gap, lambda x: (x - dx_eff) ** 2 / (2.0 * radius))
    ref_y = _mirror_plane_overlap(sphere.beam, gap, lambda x: (x - dy_eff) ** 2 / (2.0 * radius))
    assert float(sx.item()) == pytest.approx(ref_x, rel=1.0e-12)
    assert float(sy.item()) == pytest.approx(ref_y, rel=1.0e-12)


def _sphere_demo(config_dir: Path) -> SphereOpticsModel:
    """Sphere demo model (curved in both planes, radial bias)."""
    return SphereOpticsModel.from_config(load_variant("sphere_demo", config_dir=config_dir))


# --------------------------------------------------------------------------- #
# Closed form of the curved plane as an exact special case.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_curved_plane_closed_form_is_exact_special_case(config_dir: Path) -> None:
    """On the curved-plane maps the general overlap is ``eta_par^x(g) exp(-D^2/sigma_eff(g)^2)``.

    Doc 03 §d (R-59): the full overlap on the maps ``d = (2g/R_c) D``,
    ``alpha = (2/R_c) D`` folds into an exact Gaussian in ``D = Delta_eff`` with
    ``sigma_eff(g)``, ``g = A + dz``; the flat axis folds into
    ``eta_par^y(g) exp[-(2 theta_x/theta0)^2]``. Checked on a grid of tip states
    including gap changes, for the cylinder (A) and the sphere demo.
    """
    rng = np.random.default_rng(59)
    n = 64
    dx = rng.uniform(-3.0, 3.0, n) * UM
    dy = rng.uniform(-3.0, 3.0, n) * UM
    dz = rng.uniform(-5.0, 5.0, n) * UM
    theta_x = rng.uniform(-2.0e-3, 2.0e-3, n)
    theta_y = rng.uniform(-2.0e-3, 2.0e-3, n)

    cylinder = CylinderOpticsModel.from_config(load_variant("A", config_dir=config_dir))
    sphere = _sphere_demo(config_dir)
    for model in (cylinder, sphere):
        beam, radius = model.beam, model.radius_of_curvature_m
        gaps = model.gap_m + dz
        par_x = np.array([_eta_par_curved_closed(beam, g, radius) for g in gaps])
        sigma = np.array([_sigma_eff_closed(beam, g, radius) for g in gaps])
        d_x = model.bias_m + dx + (radius + gaps) * theta_y
        eta_x, eta_y = model.eta_components(dx, dy, dz, theta_x, theta_y)
        np.testing.assert_allclose(eta_x, par_x * np.exp(-((d_x / sigma) ** 2)), rtol=1.0e-12)
        if isinstance(model, SphereOpticsModel):
            d_y = model.bias_m + dy + (radius + gaps) * theta_x
            expected_y = par_x * np.exp(-((d_y / sigma) ** 2))
        else:
            par_y = np.array([_eta_par_flat_closed(beam, g) for g in gaps])
            expected_y = par_y * np.exp(-((2.0 * theta_x / _theta0(beam)) ** 2))
        np.testing.assert_allclose(eta_y, expected_y, rtol=1.0e-12)
