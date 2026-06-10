"""SI unit policy: named constants and boundary conversions.

All internal computation uses SI base units (m, s, kg, A, W, Hz, rad). Conversions
live only at input/output boundaries and in tests, per coding convention 10 §6.
The numeric value of the standard gravity is taken from knowledge-base document 01
§4.3 and is the single source of truth in code for the ``g`` ↔ ``m/s^2`` mapping.

References
----------
See 01 §4.3 (universal constants) and 10 §6 (unit policy).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Standard gravity, 1 g, in m/s^2 (knowledge base 01 §4.3).
G0_M_S2: float = 9.80665

# Scalar SI prefixes used at boundaries.
_MM_PER_M = 1.0e3
_UM_PER_M = 1.0e6
_NM_PER_M = 1.0e9
_PM_PER_M = 1.0e12

FloatArray = npt.NDArray[np.float64]


def g_to_ms2(value_g: float) -> float:
    """Convert an acceleration from g units to m/s^2.

    Parameters
    ----------
    value_g : float
        Acceleration in multiples of standard gravity (g).

    Returns
    -------
    float
        Acceleration in m/s^2.
    """
    return value_g * G0_M_S2


def ms2_to_g(value_ms2: float) -> float:
    """Convert an acceleration from m/s^2 to g units.

    Parameters
    ----------
    value_ms2 : float
        Acceleration in m/s^2.

    Returns
    -------
    float
        Acceleration in multiples of standard gravity (g).
    """
    return value_ms2 / G0_M_S2


def mm_to_m(value_mm: float) -> float:
    """Convert millimetres to metres."""
    return value_mm / _MM_PER_M


def um_to_m(value_um: float) -> float:
    """Convert micrometres to metres."""
    return value_um / _UM_PER_M


def nm_to_m(value_nm: float) -> float:
    """Convert nanometres to metres."""
    return value_nm / _NM_PER_M


def m_to_mm(value_m: float) -> float:
    """Convert metres to millimetres."""
    return value_m * _MM_PER_M


def m_to_um(value_m: float) -> float:
    """Convert metres to micrometres."""
    return value_m * _UM_PER_M


def m_to_nm(value_m: float) -> float:
    """Convert metres to nanometres."""
    return value_m * _NM_PER_M


def m_to_pm(value_m: float) -> float:
    """Convert metres to picometres."""
    return value_m * _PM_PER_M
