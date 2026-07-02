"""Model-level (L2/L3) cross-checks of plan 19 (docs 17 §2, 19).

L2 = "model vs model" (doc 17 §3): consistency between subsystem models,
cross-model identities inside the reflector family and the derived relations of
doc 17 §2 that no single-subsystem L1 golden pins. L3 = "model vs bench": the
falsifiable predictions of the prototype twin configuration (doc 16 §3a;
plan 19 §L3, experiment E1) that the bench must reproduce qualitatively.

Check IDs (plan 19 §2/§3): V19-L2-04 (NEA ~ f_max^2 master law), V19-L2-05
(sphere R_c -> inf reproduces the plane), V19-L2-06 (cylinder curved plane ==
sphere plane; cylinder flat plane == plane model), V19-L2-07 (d eta/d dz
three-term decomposition, revised doc 04 §4), V19-L3-01 (2f response at
bias ~ 0), V19-L3-02 (single-ended floor is RIN-limited), V19-L3-03 (pedestal
multiplier 1 + R1/(rho eta0) ~ 5 for the bare prototype, doc 05 R-20).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optivibe.analysis import AxisGrid, SweepSpec, run_sweep
from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import Constants, VariantConfig
from optivibe.detector.photodiode import noise_psd
from optivibe.optics import CylinderOpticsModel, PlaneOpticsModel, SphereOpticsModel
from optivibe.optics.gaussian import eta_parallel_curved

# Shared tip trajectory exercising all channels (same grid family as the S9-B
# anchors in test_optics_reflector_family.py).
_DX = np.array([0.0, 1.0e-6, -2.0e-6, 3.0e-6])
_DY = np.array([0.0, 1.0e-6, 2.0e-6, -1.0e-6])
_DZ = np.array([0.0, 1.0e-7, -1.0e-7, 2.0e-7])
_TX = np.array([0.0, 1.0e-4, -1.0e-4, 2.0e-4])
_TY = np.array([0.0, 2.0e-4, 1.0e-4, -1.0e-4])


@pytest.fixture(scope="module")
def constants() -> Constants:
    return load_constants(Path(__file__).resolve().parent.parent / "configs/constants.yaml")


@pytest.fixture(scope="module")
def variant_a() -> VariantConfig:
    return load_variant("A")


@pytest.fixture(scope="module")
def variant_proto() -> VariantConfig:
    return load_variant("proto_poc")


# --------------------------------------------------------------------------- #
# L2 -- reflector-family cross-model identities and limits (doc 03 §c-§e).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_sphere_infinite_radius_matches_plane(variant_a: VariantConfig) -> None:
    """V19-L2-05: sphere with R_c -> inf reproduces the plane model (doc 03 §c').

    At huge R_c the curved maps collapse to the plane's tilt-only maps
    (``d -> 2 g theta``, ``alpha -> 2 theta``; the (R_c + g) lever cancels the
    2/R_c curvature factor) and ``eta_par_curved -> eta_par_flat``, so the FULL
    eta of the sphere must converge to the plane model on any trajectory --
    including non-zero dx/dy, to which the plane is insensitive.
    """
    beam_kwargs = {
        "wavelength_m": variant_a.source.wavelength_m,
        "waist_radius_m": variant_a.optics.mode_field_radius_m,
    }
    from optivibe.optics.gaussian import GaussianBeam

    beam = GaussianBeam(**beam_kwargs)
    gap = variant_a.optics.gap_m
    sphere = SphereOpticsModel(
        beam=beam, gap_m=gap, radius_of_curvature_m=1.0, bias_m=0.0
    )  # R_c = 1 m >> zR
    plane = PlaneOpticsModel(beam=beam, gap_m=gap)
    eta_sphere = sphere.eta(dx=_DX, dy=_DY, dz=_DZ, theta_x=_TX, theta_y=_TY)
    eta_plane = plane.eta(dx=_DX, dy=_DY, dz=_DZ, theta_x=_TX, theta_y=_TY)
    np.testing.assert_allclose(eta_sphere, eta_plane, rtol=1.0e-3)


@pytest.mark.golden
def test_golden_cylinder_planes_match_sphere_and_plane(variant_a: VariantConfig) -> None:
    """V19-L2-06: per-plane cross-model identity of the reflector family.

    The cylinder is the composite of the family (doc 03 §c): its curved x-z
    plane must equal the sphere's x-z plane (same R_c, same bias, same maps and
    the same ABCD parallel factor), and its flat y-z plane must equal the
    plane model's per-plane factor for the same tilt. Any drift between the
    three implementations is a model-consistency error (doc 17 §2).
    """
    cyl = CylinderOpticsModel.from_config(variant_a)
    sphere = SphereOpticsModel(
        beam=cyl.beam,
        gap_m=cyl.gap_m,
        radius_of_curvature_m=cyl.radius_of_curvature_m,
        bias_m=cyl.bias_m,
    )
    plane = PlaneOpticsModel(beam=cyl.beam, gap_m=cyl.gap_m)
    cyl_x, cyl_y = cyl.eta_components(_DX, _DY, _DZ, _TX, _TY)
    sph_x, _ = sphere.eta_components(_DX, np.zeros_like(_DY), _DZ, np.zeros_like(_TX), _TY)
    _, pln_y = plane.eta_components(np.zeros_like(_DX), _DY, _DZ, _TX, np.zeros_like(_TY))
    np.testing.assert_allclose(cyl_x, sph_x, rtol=1.0e-12)
    np.testing.assert_allclose(cyl_y, pln_y, rtol=1.0e-12)


@pytest.mark.golden
def test_golden_dz_slope_three_term_decomposition(variant_a: VariantConfig) -> None:
    """V19-L2-07: d eta/d dz = eta0 * (flat + curved + map) (revised doc 04 §4).

    The dz channel of the cylinder collects three independent log-derivative
    contributions at the working point (T-19 revision of the doc 04 §4
    reference; doc 19 §4): the flat-plane envelope ``-A/(zR^2 + A^2)``, the
    curved-plane ABCD envelope ``d ln eta_par^x/d g`` and the gap dependence of
    the offset map ``-8 Dx0^2 A/(R_c^2 w0^2)``. The full central-difference
    derivative of eta must equal eta0 times their sum (product/factorization
    rule), and the total must match the documented ~ -7.9e3 1/m of the revised
    doc 04 §4 (was: illustrative -2e4, not reproduced by any derivation).
    """
    m = CylinderOpticsModel.from_config(variant_a)
    gap, radius, bias = m.gap_m, m.radius_of_curvature_m, m.bias_m
    z_r, w0 = m.beam.rayleigh_range_m, m.beam.waist_radius_m
    h = 1.0e-10
    eta0 = float(m.eta().item())
    slope_full = float(((m.eta(dz=h) - m.eta(dz=-h)) / (2.0 * h)).item())
    term_flat = -gap / (z_r**2 + gap**2)
    epc = lambda g: float(eta_parallel_curved(m.beam, g, radius))  # noqa: E731
    term_curved = (np.log(epc(gap + h)) - np.log(epc(gap - h))) / (2.0 * h)
    term_map = -8.0 * bias**2 * gap / (radius**2 * w0**2)
    assert slope_full == pytest.approx(eta0 * (term_flat + term_curved + term_map), rel=1.0e-6)
    assert slope_full == pytest.approx(-7.9e3, rel=0.05)  # revised doc 04 §4 reference
    # The doc's old closed form (flat envelope only) is just one of three terms:
    assert abs(eta0 * term_flat) < 0.3 * abs(slope_full)


@pytest.mark.golden
def test_golden_nea_master_law_fmax_squared(constants: Constants) -> None:
    """V19-L2-04: master law NEA ~ f_max^2 (doc 07 R-31; doc 17 §2).

    Ties two doc 17 §2 relations together on one design sweep: with
    ``NEA ~ L^-4`` and ``f1 ~ 1/L^2`` the plateau NEA of length-scaled designs
    must follow the square of their first mode, ``NEA_i/NEA_j = (f1_i/f1_j)^2``
    (the supported band edge scales with f1). A drift between the two scalings
    is a cross-metric inconsistency (doc 17 §2).
    """
    spec = SweepSpec(
        kind="sweep",
        name="L",
        mode="design",
        variant="B",
        parameter="length_m",
        grid=AxisGrid(start=1.0e-3, stop=2.0e-3, num=5),
    )
    result = run_sweep(spec, constants)
    nea = result.metrics["nea_plateau_ug"]
    f1 = result.metrics["f1_hz"]
    np.testing.assert_allclose(nea / nea[-1], (f1 / f1[-1]) ** 2, rtol=0.10)


# --------------------------------------------------------------------------- #
# L3 -- falsifiable predictions of the prototype twin (doc 16 §3a; plan 19 E1).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_proto_poc_two_f_response(variant_proto: VariantConfig) -> None:
    """V19-L3-01: at bias ~ 0 the transverse response is quadratic (2f).

    Prototype prediction E1-P1 (doc 16 §3a R-4; doc 05 §2 limit iv): with the
    working point at the eta peak the 1f line vanishes, the dominant line is at
    2f, and the 2f amplitude scales as the drive amplitude SQUARED (x4 for x2).
    Introducing a bias (doc 16 D-05) must bring the 1f line back and make it
    dominate. The bench replication of this test is the first cheap validation
    of the twin against the real prototype (scenario examples/proto_poc_2f.yaml,
    dominant 240 Hz).
    """
    model = SphereOpticsModel.from_config(variant_proto)
    assert model.bias_m == 0.0
    fs, f0 = 50_000.0, 120.0
    t = np.arange(int(fs * 0.5)) / fs
    freqs = np.fft.rfftfreq(t.size, 1.0 / fs)
    i1, i2 = (int(np.argmin(np.abs(freqs - f))) for f in (f0, 2.0 * f0))

    def line_amps(drive_m: float, m: SphereOpticsModel) -> tuple[float, float]:
        eta = m.eta(dx=drive_m * np.sin(2.0 * np.pi * f0 * t))
        spec = np.abs(np.fft.rfft(eta - eta.mean())) / eta.size * 2.0
        return float(spec[i1]), float(spec[i2])

    a1_small, a2_small = line_amps(1.0e-7, model)
    _, a2_big = line_amps(2.0e-7, model)
    assert a1_small < 1.0e-9 * a2_small  # 1f suppressed to numerical zero
    assert a2_big / a2_small == pytest.approx(4.0, rel=5.0e-3)  # 2f ~ d^2
    # With a bias the linear channel returns and dominates (doc 05 §2):
    biased = SphereOpticsModel(
        beam=model.beam,
        gap_m=model.gap_m,
        radius_of_curvature_m=model.radius_of_curvature_m,
        bias_m=2.0e-6,
    )
    a1_bias, a2_bias = line_amps(1.0e-7, biased)
    assert a1_bias > 10.0 * a2_bias


@pytest.mark.golden
def test_golden_proto_poc_rin_dominates_single_ended(
    variant_proto: VariantConfig, constants: Constants
) -> None:
    """V19-L3-02: the single-ended prototype floor is RIN-limited (doc 07 §1.2).

    Prototype prediction E1-P2 (doc 16 §3a R-3): with balanced=false the SLD
    RIN enters unsuppressed and dominates the shot noise by orders of magnitude
    in PSD; re-enabling the balanced channel (CMRR 40 dB) must push RIN back
    below shot. The bench counterpart: the measured noise floor of the POC must
    sit far above the shot estimate from its DC photocurrent.
    """
    rho = variant_proto.reflector.reflectivity
    r1 = variant_proto.endface_reflectivity
    model = SphereOpticsModel.from_config(variant_proto)
    eta0 = float(model.eta().item())
    i_dc = variant_proto.responsivity_a_w * variant_proto.source.power_w * (r1 + rho * eta0)
    single = noise_psd(i_dc, variant_proto, constants, balanced=False)
    assert single["rin"] > 100.0 * single["shot"]  # RIN-limited floor
    balanced = noise_psd(i_dc, variant_proto, constants, balanced=True)
    assert balanced["rin"] < balanced["shot"]  # reference channel restores shot limit


@pytest.mark.golden
def test_golden_proto_poc_pedestal_multiplier(variant_proto: VariantConfig) -> None:
    """V19-L3-03: bare-prototype pedestal multiplier 1 + R1/(rho eta0) ~ 5.

    Doc 05 R-20: for the bare reflector the endface pedestal R1 inflates the DC
    (and hence the shot noise) without carrying signal; the multiplier
    ``1 + R1/(rho eta0)`` evaluates to ~ 5 for R1 = 0.036, rho ~ 0.035,
    eta0 ~ 0.25 (doc 16 §3a R-2 'пьедестал/сигнал ~ 5'). The bench counterpart:
    the POC DC level must exceed the signal-carrying part ~ 5x.
    """
    model = SphereOpticsModel.from_config(variant_proto)
    eta0 = float(model.eta().item())
    rho = variant_proto.reflector.reflectivity
    r1 = variant_proto.endface_reflectivity
    multiplier = 1.0 + r1 / (rho * eta0)
    assert multiplier == pytest.approx(5.0, rel=0.05)
