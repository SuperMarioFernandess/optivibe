"""Composite / modulated excitation: goldens against the closed forms of the base.

Task S-21; plan 18 §5(g) -- every golden here pins a formula of doc 11 §2.1
(AM sidebands ``m a_c / 2``, the FM Bessel family ``a_c |J_k(beta)|``, Parseval
and the level convention), never the current output of the code. The Bessel
goldens are deliberately anchored where the analytic answer is *sharp*: the
carrier vanishes at the first zero of ``J_0`` and the mean square is exactly
``a_c^2 / 2`` for every ``beta`` because ``sum_k J_k^2 = 1``.

The regression side (18 §5) checks the other direction: the S-21 fields are
opt-in, so a pre-S-21 spec must still produce the pre-S-21 waveform *to the
bit*, which is what protects the shared ``_common`` helpers and the 19
acceptance dominants.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

from optivibe.core.config.models import (
    CompositeSpec,
    MultitoneSpec,
    RandomSpec,
    ShockSpec,
    SineSpec,
    SweepSpec,
)
from optivibe.core.types import Excitation
from optivibe.dsp.spectra import amplitude_spectrum
from optivibe.excitation import EXCITATION_REGISTRY
from optivibe.excitation.composite import component_seed

G0 = 9.80665

#: Grid of the goldens: an integer number of periods of every frequency used
#: below fits the record, so the rectangular-window lines are exact.
FS_HZ = 20_000.0
DURATION_S = 2.0

#: First zero of ``J_0``: the FM carrier line vanishes there (Abramowitz &
#: Stegun 9.5.2, and reproduced by ``scipy.special.jn_zeros(0, 1)``).
J0_FIRST_ZERO = 2.404_825_557_695_773


def _generate(spec: object, seed: int | None = 20260730) -> Excitation:
    """Run the registered generator for ``spec`` (kind taken from the spec)."""
    return EXCITATION_REGISTRY.create(spec.kind).generate(spec, seed=seed)  # type: ignore[attr-defined]


def _line(signal: np.ndarray, freq_hz: float) -> float:
    """Amplitude of the spectral line nearest ``freq_hz``."""
    spectrum = amplitude_spectrum(signal, FS_HZ)
    return float(spectrum.values[int(np.argmin(np.abs(spectrum.freq - freq_hz)))])


def _mean_square(signal: np.ndarray) -> float:
    """Mean square of a time series, in its own units squared."""
    return float(np.mean(signal**2))


# --------------------------------------------------------------------------- #
# AM: doc 11 §2.1.3
# --------------------------------------------------------------------------- #
def test_golden_am_sidebands_are_half_depth() -> None:
    """AM puts ``m a_c / 2`` at ``f_c +- f_m`` and leaves the carrier at ``a_c``."""
    amplitude_g, depth, f_c, f_m = 3.0, 0.4, 500.0, 37.0
    spec = SineSpec(
        fs_hz=FS_HZ,
        duration_s=DURATION_S,
        frequency_hz=f_c,
        amplitude_g=amplitude_g,
        modulation={"kind": "am", "f_m_hz": f_m, "depth": depth},
    )
    signal = _generate(spec).a_x
    a_c = amplitude_g * G0
    assert _line(signal, f_c) == pytest.approx(a_c, rel=1.0e-9)
    for sideband in (f_c - f_m, f_c + f_m):
        assert _line(signal, sideband) == pytest.approx(0.5 * depth * a_c, rel=1.0e-9)


def test_golden_am_mean_square_matches_parseval() -> None:
    """``<a^2> = (a_c^2 / 2)(1 + m^2 / 2)``: the time and line sums agree."""
    amplitude_g, depth, f_c, f_m = 2.0, 0.7, 400.0, 31.0
    spec = SineSpec(
        fs_hz=FS_HZ,
        duration_s=DURATION_S,
        frequency_hz=f_c,
        amplitude_g=amplitude_g,
        modulation={"kind": "am", "f_m_hz": f_m, "depth": depth},
    )
    signal = _generate(spec).a_x
    a_c = amplitude_g * G0
    from_formula = 0.5 * a_c**2 * (1.0 + 0.5 * depth**2)
    from_lines = 0.5 * (a_c**2 + 2.0 * (0.5 * depth * a_c) ** 2)
    assert from_lines == pytest.approx(from_formula, rel=1.0e-12)
    assert _mean_square(signal) == pytest.approx(from_formula, rel=1.0e-9)


def test_am_depth_above_one_is_rejected_with_the_composite_route() -> None:
    """Over-modulation is refused, and the message names the way around it."""
    with pytest.raises(ValueError, match="composite"):
        SineSpec(
            fs_hz=FS_HZ,
            duration_s=DURATION_S,
            frequency_hz=500.0,
            amplitude_g=1.0,
            modulation={"kind": "am", "f_m_hz": 37.0, "depth": 1.5},
        )


# --------------------------------------------------------------------------- #
# FM: doc 11 §2.1.3
# --------------------------------------------------------------------------- #
def test_golden_fm_sidebands_follow_bessel() -> None:
    """Line ``k`` of an FM carrier holds ``a_c |J_k(beta)|`` at ``f_c +- k f_m``."""
    amplitude_g, f_c, f_m, beta = 3.0, 2000.0, 50.0, 3.0
    spec = SineSpec(
        fs_hz=FS_HZ,
        duration_s=DURATION_S,
        frequency_hz=f_c,
        amplitude_g=amplitude_g,
        modulation={"kind": "fm", "f_m_hz": f_m, "deviation_hz": beta * f_m},
    )
    signal = _generate(spec).a_x
    a_c = amplitude_g * G0
    for order in range(0, 6):
        predicted = a_c * abs(float(jv(order, beta)))
        for sign in (1.0, -1.0):
            measured = _line(signal, f_c + sign * order * f_m)
            assert measured == pytest.approx(predicted, rel=1.0e-6, abs=1.0e-9)


def test_golden_fm_carrier_vanishes_at_the_first_bessel_zero() -> None:
    """At ``beta = 2.40483`` the carrier line disappears; the sidebands hold the power.

    A formula golden in the strongest sense: the answer (zero) is fixed by
    ``J_0``, so no output of the code can be mistaken for it.
    """
    amplitude_g, f_c, f_m = 1.0, 2000.0, 50.0
    spec = SineSpec(
        fs_hz=FS_HZ,
        duration_s=DURATION_S,
        frequency_hz=f_c,
        amplitude_g=amplitude_g,
        modulation={"kind": "fm", "f_m_hz": f_m, "deviation_hz": J0_FIRST_ZERO * f_m},
    )
    signal = _generate(spec).a_x
    a_c = amplitude_g * G0
    assert _line(signal, f_c) < 1.0e-6 * a_c
    assert _line(signal, f_c + f_m) == pytest.approx(
        a_c * abs(float(jv(1, J0_FIRST_ZERO))), rel=1.0e-6
    )


@pytest.mark.parametrize("beta", [0.5, 2.404_825_557_695_773, 8.0])
def test_golden_fm_conserves_mean_square(beta: float) -> None:
    """``sum_k J_k^2 = 1``, so FM never changes the level -- only its distribution."""
    amplitude_g, f_c, f_m = 2.0, 3000.0, 100.0
    spec = SineSpec(
        fs_hz=FS_HZ,
        duration_s=DURATION_S,
        frequency_hz=f_c,
        amplitude_g=amplitude_g,
        modulation={"kind": "fm", "f_m_hz": f_m, "deviation_hz": beta * f_m},
    )
    signal = _generate(spec).a_x
    assert _mean_square(signal) == pytest.approx(0.5 * (amplitude_g * G0) ** 2, rel=1.0e-6)


# --------------------------------------------------------------------------- #
# Degenerate cases and the opt-in regression (18 §5)
# --------------------------------------------------------------------------- #
def _plain_sine(**overrides: object) -> SineSpec:
    """A pre-S-21 sine spec (no phase, no modulation)."""
    fields: dict[str, object] = {
        "fs_hz": FS_HZ,
        "duration_s": DURATION_S,
        "frequency_hz": 500.0,
        "amplitude_g": 1.0,
    }
    fields.update(overrides)
    return SineSpec(**fields)  # type: ignore[arg-type]


def test_golden_sine_matches_its_closed_form() -> None:
    """``a(t) = a_c sin(2 pi f_c t + phi_c)`` to the bit, phase included (11 §2.1.1)."""
    phase = 0.7
    spec = _plain_sine(phase_rad=phase)
    signal = _generate(spec).a_x
    t = np.arange(round(FS_HZ * DURATION_S)) / FS_HZ
    reference = (1.0 * G0) * np.sin(2.0 * np.pi * 500.0 * t + phase)
    assert signal.tobytes() == reference.tobytes()


@pytest.mark.parametrize(
    "modulation",
    [
        None,
        {"kind": "am", "f_m_hz": 37.0, "depth": 0.0},
        {"kind": "fm", "f_m_hz": 37.0, "deviation_hz": 0.0},
    ],
)
def test_degenerate_modulation_is_bit_identical_to_the_plain_carrier(
    modulation: dict[str, object] | None,
) -> None:
    """``m = 0`` and ``beta = 0`` reduce to the unmodulated tone exactly, not nearly.

    IEEE-754 makes this exact rather than approximate (``1 + 0*cos = 1`` and
    ``x + 0 = x``), which is why the acceptance is stated bit-for-bit.
    """
    plain = _generate(_plain_sine()).a_x
    degenerate = _generate(_plain_sine(modulation=modulation)).a_x
    assert degenerate.tobytes() == plain.tobytes()


def test_pre_s21_specs_are_untouched_by_the_new_optional_fields() -> None:
    """Every generated kind keeps its waveform when built the pre-S-21 way.

    The five kinds share ``_common`` (time grid, axis packing, RMS scaling), so
    this is the regression that would catch a change leaking into the shared
    helpers -- and with it into the 19 acceptance dominants of 18 §5.
    """
    specs = (
        _plain_sine(),
        MultitoneSpec(
            fs_hz=FS_HZ,
            duration_s=DURATION_S,
            tones=[{"frequency_hz": 120.0, "amplitude_g": 1.0}],
        ),
        SweepSpec(
            fs_hz=FS_HZ,
            duration_s=DURATION_S,
            f_start_hz=20.0,
            f_end_hz=2000.0,
            amplitude_g=1.0,
        ),
        RandomSpec(fs_hz=FS_HZ, duration_s=DURATION_S, band_hz=(20.0, 2000.0), g_rms=1.0),
        ShockSpec(fs_hz=FS_HZ, duration_s=DURATION_S, peak_g=10.0, pulse_ms=2.0, delay_s=0.1),
    )
    for spec in specs:
        dump = spec.model_dump()
        # The S-21 additions must be inert by default wherever they exist.
        assert dump.get("modulation") is None
        assert dump.get("phase_rad", 0.0) == 0.0
        assert dump.get("seed") is None
        # ... and generation must be reproducible for the same seed.
        assert _generate(spec).a_x.tobytes() == _generate(spec).a_x.tobytes()

    # The tone of the hello acceptance keeps its closed form exactly.
    t = np.arange(round(FS_HZ * DURATION_S)) / FS_HZ
    reference = (1.0 * G0) * np.sin(2.0 * np.pi * 120.0 * t)
    assert _generate(specs[1]).a_x.tobytes() == reference.tobytes()


# --------------------------------------------------------------------------- #
# Composite: doc 11 §2.1.4-§2.1.5
# --------------------------------------------------------------------------- #
def _composite(components: list[dict[str, object]], **overrides: object) -> CompositeSpec:
    """Build a composite on the golden grid."""
    payload: dict[str, object] = {
        "kind": "composite",
        "fs_hz": FS_HZ,
        "duration_s": DURATION_S,
        "components": components,
    }
    payload.update(overrides)
    return CompositeSpec.model_validate(payload)


@pytest.mark.parametrize(
    "component",
    [
        {"kind": "sine", "frequency_hz": 500.0, "amplitude_g": 1.0},
        {"kind": "random", "band_hz": [20.0, 2000.0], "g_rms": 1.0},
    ],
)
def test_single_component_composite_is_bit_identical_to_the_component(
    component: dict[str, object],
) -> None:
    """One component in, that component out -- including the noise realization.

    Holds by construction: the composite calls the same registered generator,
    and component 0 inherits the scenario seed unchanged (doc 11 §2.1.5).
    """
    seed = 20260730
    standalone = EXCITATION_REGISTRY.create(str(component["kind"])).generate(
        _composite([component]).components[0], seed=seed
    )
    composite = _generate(_composite([component]), seed=seed)
    assert composite.a_x.tobytes() == standalone.a_x.tobytes()


def test_golden_composite_power_adds_over_disjoint_supports() -> None:
    """``<a^2> = sum_i <a_i^2>`` when the components do not share frequencies.

    The level convention of doc 11 §2.1.4 (no renormalization) is exactly what
    makes this predictable before the run; the cross terms vanish because the
    supports are disjoint.
    """
    parts = [
        {"kind": "sine", "frequency_hz": 300.0, "amplitude_g": 1.0},
        {"kind": "sine", "frequency_hz": 700.0, "amplitude_g": 0.5},
        {
            "kind": "sine",
            "frequency_hz": 1500.0,
            "amplitude_g": 2.0,
            "modulation": {"kind": "am", "f_m_hz": 41.0, "depth": 0.5},
        },
    ]
    spec = _composite(parts)
    total = _generate(spec).a_x
    from_parts = (
        0.5 * (1.0 * G0) ** 2 + 0.5 * (0.5 * G0) ** 2 + 0.5 * (2.0 * G0) ** 2 * (1.0 + 0.5 * 0.5**2)
    )
    assert _mean_square(total) == pytest.approx(from_parts, rel=1.0e-9)


def test_composite_sums_its_components_sample_by_sample() -> None:
    """The composite is the plain sum -- no renormalization is applied."""
    parts = [
        {"kind": "sine", "frequency_hz": 300.0, "amplitude_g": 1.0},
        {"kind": "sine", "frequency_hz": 700.0, "amplitude_g": 0.5},
    ]
    spec = _composite(parts)
    total = _generate(spec).a_x
    pieces = [
        EXCITATION_REGISTRY.create("sine").generate(component, seed=component_seed(20260730, index))
        for index, component in enumerate(spec.components)
    ]
    assert np.array_equal(total, pieces[0].a_x + pieces[1].a_x)


def test_composite_places_components_on_their_own_axes() -> None:
    """A component may name its own axis; the parts are summed per axis."""
    spec = _composite(
        [
            {"kind": "sine", "frequency_hz": 300.0, "amplitude_g": 1.0},
            {"kind": "sine", "frequency_hz": 700.0, "amplitude_g": 0.5, "axis": "z"},
        ],
        axis="x",
    )
    out = _generate(spec)
    assert np.any(out.a_x != 0.0)
    assert np.any(out.a_z != 0.0)
    assert not np.any(out.a_y)
    assert _mean_square(out.a_z) == pytest.approx(0.5 * (0.5 * G0) ** 2, rel=1.0e-9)


def test_composite_inherits_the_grid_and_refuses_a_conflicting_one() -> None:
    """fs / duration / axis come from the composite; a contradiction fails loudly."""
    spec = _composite([{"kind": "sine", "frequency_hz": 300.0, "amplitude_g": 1.0}], axis="y")
    component = spec.components[0]
    assert (component.fs_hz, component.duration_s, component.axis) == (FS_HZ, DURATION_S, "y")
    with pytest.raises(ValueError, match="contradicts the composite grid"):
        _composite([{"kind": "sine", "frequency_hz": 300.0, "amplitude_g": 1.0, "fs_hz": 8000.0}])


def test_composite_admits_generated_kinds_only() -> None:
    """File replay and nested composites are not components (doc 11 §2.1.4)."""
    for rejected in (
        {"kind": "csv", "path": "x.csv", "column": 1, "fs_hz": FS_HZ, "units": "g"},
        {"kind": "composite", "components": [{"kind": "sine", "frequency_hz": 1.0}]},
    ):
        with pytest.raises(ValueError):
            _composite([rejected])


def test_noise_components_draw_independent_streams() -> None:
    """Two ``random`` components never share a stream, and appending is stable.

    Sharing would silently turn two independent noises into one doubled noise
    (doc 11 §2.1.5); the position-derived sub-seed prevents it.
    """
    noise = {"kind": "random", "band_hz": [20.0, 2000.0], "g_rms": 1.0}
    two = _generate(_composite([dict(noise), dict(noise)])).a_x
    one = _generate(_composite([dict(noise)])).a_x
    assert not np.allclose(two, 2.0 * one)

    # Appending a component leaves the streams of the earlier ones untouched:
    # compare the component's own realization, not a difference of sums (a + b - b
    # is not a in floating point).
    extended = _composite(
        [dict(noise), {"kind": "sine", "frequency_hz": 300.0, "amplitude_g": 1.0}]
    )
    kept = EXCITATION_REGISTRY.create("random").generate(
        extended.components[0], seed=component_seed(20260730, 0)
    )
    assert np.array_equal(kept.a_x, one)


def test_an_explicit_component_seed_pins_the_realization() -> None:
    """``seed:`` on a component makes its noise independent of its position."""
    pinned = {"kind": "random", "band_hz": [20.0, 2000.0], "g_rms": 1.0, "seed": 4242}
    tone = {"kind": "sine", "frequency_hz": 300.0, "amplitude_g": 1.0}
    first = _generate(_composite([dict(pinned), dict(tone)])).a_x
    second = _generate(_composite([dict(tone), dict(pinned)])).a_x
    assert np.array_equal(first, second)


def test_component_seed_derivation_is_deterministic_and_position_dependent() -> None:
    """Component 0 inherits the seed; later components get stable sub-seeds."""
    assert component_seed(7, 0) == 7
    assert component_seed(None, 3) is None
    assert component_seed(7, 1) == component_seed(7, 1)
    assert component_seed(7, 1) != component_seed(7, 2)
    assert component_seed(7, 1) != component_seed(8, 1)


def test_composite_meta_reports_the_level_it_did_not_renormalize() -> None:
    """The un-renormalized level is reported, per component and in total."""
    spec = _composite(
        [
            {"kind": "sine", "frequency_hz": 300.0, "amplitude_g": 1.0},
            {"kind": "sine", "frequency_hz": 700.0, "amplitude_g": 0.5},
        ]
    )
    meta = _generate(spec).meta
    assert meta["n_components"] == 2
    assert [component["kind"] for component in meta["components"]] == ["sine", "sine"]
    assert meta["rms_m_s2"] == pytest.approx(
        np.sqrt(0.5 * (1.0 * G0) ** 2 + 0.5 * (0.5 * G0) ** 2), rel=1.0e-9
    )
    assert meta["peak_m_s2"] <= (1.0 + 0.5) * G0 + 1.0e-9
