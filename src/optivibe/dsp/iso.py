"""ISO 10816-3 / 20816-3 vibration-severity reference data and classifier (S5 §4).

The evaluation-zone boundaries are *reference thresholds from the standard*, kept
as data here (a typed table, not a physical formula -- task S5 §4): a measured
broadband RMS velocity in the band is compared against the published zone limits.
ISO 20816-3 supersedes ISO 10816-3 with the same continuous zone boundaries.

Zones (ISO 10816-3 §5): **A** newly commissioned machines; **B** acceptable for
unrestricted long-term operation; **C** unsatisfactory for long-term operation
(limited time only); **D** vibration severe enough to cause damage.

Boundaries are velocity RMS in mm/s for the machine groups and support classes of
the standard (Group 1: large machines 300 kW-50 MW; Group 2: medium machines
15-300 kW; rigid vs flexible support). The default class is Group 2 / rigid; the
table is overridable for other classes. This is reference data only -- the
numbers are not derived by the model.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ISO_10816_3_ZONES", "IsoZoneLimits", "classify_velocity_rms", "iso_assessment"]


@dataclass(frozen=True)
class IsoZoneLimits:
    """Velocity-RMS zone boundaries of one ISO 10816-3 machine class, mm/s.

    Parameters
    ----------
    machine_class : str
        Human-readable class label (group + support condition).
    a_b_mm_s, b_c_mm_s, c_d_mm_s : float
        Upper RMS-velocity limits of zones A, B and C (the A/B, B/C and C/D
        boundaries), mm/s. Zone D is everything above ``c_d_mm_s``.
    """

    machine_class: str
    a_b_mm_s: float
    b_c_mm_s: float
    c_d_mm_s: float


# Published ISO 10816-3 evaluation-zone boundaries (RMS velocity, mm/s). Keys are
# "<group>_<support>". Reference data; not computed by the model.
ISO_10816_3_ZONES: dict[str, IsoZoneLimits] = {
    "group1_rigid": IsoZoneLimits("Group 1 (large), rigid support", 2.3, 4.5, 7.1),
    "group1_flexible": IsoZoneLimits("Group 1 (large), flexible support", 3.5, 7.1, 11.0),
    "group2_rigid": IsoZoneLimits("Group 2 (medium), rigid support", 1.4, 2.8, 4.5),
    "group2_flexible": IsoZoneLimits("Group 2 (medium), flexible support", 2.3, 4.5, 7.1),
}

_DEFAULT_CLASS = "group2_rigid"


def classify_velocity_rms(v_rms_mm_s: float, limits: IsoZoneLimits) -> str:
    """Return the ISO 10816-3 evaluation zone (``"A"``..``"D"``) for an RMS velocity.

    Parameters
    ----------
    v_rms_mm_s : float
        Broadband RMS velocity in the assessment band, mm/s.
    limits : IsoZoneLimits
        Zone boundaries of the machine class.

    Returns
    -------
    str
        Zone label ``"A"``, ``"B"``, ``"C"`` or ``"D"``.
    """
    if v_rms_mm_s <= limits.a_b_mm_s:
        return "A"
    if v_rms_mm_s <= limits.b_c_mm_s:
        return "B"
    if v_rms_mm_s <= limits.c_d_mm_s:
        return "C"
    return "D"


def iso_assessment(
    v_rms_m_s: float,
    *,
    machine_class: str = _DEFAULT_CLASS,
    band_hz: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Assemble an ISO 10816-3 severity assessment from a band RMS velocity (S5 §4).

    Parameters
    ----------
    v_rms_m_s : float
        Broadband RMS velocity in the assessment band, m/s (SI).
    machine_class : str, optional
        Key into :data:`ISO_10816_3_ZONES` (default ``"group2_rigid"``).
    band_hz : tuple of float or None, optional
        The assessment band ``(f_lo, f_hi)``, Hz, recorded for traceability.

    Returns
    -------
    dict
        Assessment with the standard, machine class, RMS velocity (mm/s and SI),
        zone label and the zone boundaries.

    Raises
    ------
    KeyError
        If ``machine_class`` is not a known class.
    """
    limits = ISO_10816_3_ZONES[machine_class]
    v_mm_s = v_rms_m_s * 1.0e3
    zone = classify_velocity_rms(v_mm_s, limits)
    return {
        "standard": "ISO 10816-3 / 20816-3",
        "machine_class": limits.machine_class,
        "v_rms_mm_s": v_mm_s,
        "v_rms_m_s": v_rms_m_s,
        "zone": zone,
        "zone_boundaries_mm_s": {
            "A/B": limits.a_b_mm_s,
            "B/C": limits.b_c_mm_s,
            "C/D": limits.c_d_mm_s,
        },
        "band_hz": band_hz,
    }
