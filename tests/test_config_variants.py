"""Variant presets load and match the documented parameters (doc 08 §6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from optivibe.core.config import load_variant

# Documented per-variant parameters (doc 08 §6.2/§6.3). SI units.
EXPECTED = {
    "A": {
        "length_m": 5.0e-3,
        "full_scale_g": 5.0,
        "rc_m": 62.5e-6,
        "power_w": 0.020,
        "source_kind": "SLD",
        "route": 2,
        "mode": "offresonance",
        "band": (0.1, 100.0),
    },
    "B": {
        "length_m": 2.0e-3,
        "full_scale_g": 50.0,
        "rc_m": 31.0e-6,
        "power_w": 0.016,
        "source_kind": "SLD",
        "route": 2,
        "mode": "offresonance",
        "band": (1.0, 10000.0),
    },
    "C": {
        "length_m": 1.41e-3,
        "full_scale_g": 50.0,
        "rc_m": 31.0e-6,
        "power_w": 0.016,
        "source_kind": "SLD",
        "route": 2,
        "mode": "offresonance",
        "band": (10.0, 20000.0),
    },
    "D": {
        "length_m": 4.47e-3,
        "full_scale_g": 0.022,
        "rc_m": 31.0e-6,
        "power_w": 0.100,
        "source_kind": "DFB",
        "route": 1,
        "mode": "resonance",
        "band": None,  # placeholder band in S0; not asserted
    },
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_variant_matches_doc08(name: str, config_dir: Path) -> None:
    expected = EXPECTED[name]
    v = load_variant(name, config_dir=config_dir)

    assert v.name == name
    assert v.length_m == pytest.approx(expected["length_m"])
    assert v.full_scale_g == pytest.approx(expected["full_scale_g"])
    assert v.reflector.radius_of_curvature_m == pytest.approx(expected["rc_m"])
    assert v.source.power_w == pytest.approx(expected["power_w"])
    assert v.source.kind == expected["source_kind"]
    assert v.route == expected["route"]
    assert v.mode == expected["mode"]

    # Common optical platform R-40 (doc 08): lambda, rho, R, R1, eta0.
    assert v.source.wavelength_m == pytest.approx(1550.0e-9)
    assert v.reflector.reflectivity == pytest.approx(0.98)
    assert v.responsivity_a_w == pytest.approx(1.0)
    assert v.endface_reflectivity == pytest.approx(0.036)
    assert v.eta_bias == pytest.approx(0.25)

    band = expected["band"]
    if band is not None:
        assert v.band.f_min_hz == pytest.approx(band[0])
        assert v.band.f_max_hz == pytest.approx(band[1])


def test_resonant_variant_has_line_frequency(config_dir: Path) -> None:
    d = load_variant("D", config_dir=config_dir)
    assert d.mode == "resonance"
    assert d.line_freq_hz == pytest.approx(5000.0)
    assert d.vacuum is True
