"""Validation behaviour of the array-carrying data contracts."""

from __future__ import annotations

import numpy as np
import pytest

from optivibe.core.types import (
    DetectorOutput,
    Excitation,
    OpticalResponse,
    TipState,
    VibrationResult,
)


def _ones(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.float64)


def test_excitation_valid_and_properties() -> None:
    exc = Excitation(a_x=_ones(10), a_y=_ones(10), a_z=_ones(10), fs=1000.0)
    assert exc.n_samples == 10
    assert exc.duration_s == pytest.approx(0.01)
    assert exc.a_x.dtype == np.float64


def test_excitation_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        Excitation(a_x=_ones(10), a_y=_ones(9), a_z=_ones(10), fs=1000.0)


def test_non_positive_fs_raises() -> None:
    with pytest.raises(ValueError, match="fs"):
        Excitation(a_x=_ones(4), a_y=_ones(4), a_z=_ones(4), fs=0.0)


def test_two_dimensional_array_raises() -> None:
    bad = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="1-D"):
        Excitation(a_x=bad, a_y=_ones(4), a_z=_ones(4), fs=10.0)


def test_empty_array_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Excitation(a_x=np.array([]), a_y=np.array([]), a_z=np.array([]), fs=10.0)


def test_non_finite_raises() -> None:
    bad = np.array([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="non-finite"):
        Excitation(a_x=bad, a_y=_ones(3), a_z=_ones(3), fs=10.0)


def test_tipstate_component_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        TipState(
            dx=_ones(5),
            dy=_ones(5),
            dz=_ones(5),
            theta_x=_ones(4),
            theta_y=_ones(5),
            fs=100.0,
        )


def test_optical_factor_length_checked() -> None:
    with pytest.raises(ValueError, match="eta_x"):
        OpticalResponse(eta=_ones(5), bias=0.25, fs=100.0, eta_x=_ones(4))


def test_detector_and_result_construct() -> None:
    det = DetectorOutput(samples=_ones(8), fs=200.0, dc_level=0.5)
    assert det.units == "A"
    assert det.n_samples == 8
    res = VibrationResult(a=_ones(8), v=_ones(8), x=_ones(8), fs=200.0)
    assert res.n_samples == 8
    assert res.spectrum is None
