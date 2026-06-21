"""Calibration: detector samples -> target-axis acceleration (docs 05/07).

The forward read-out of route 2 is ``I(t) = R P (R1 + rho eta(t))`` with the DC
pedestal ``I_DC = R P (R1 + rho eta0)`` (docs 04 §4, 05 §0). Only ``rho eta``
carries signal, so the canonical AC photocurrent is

``I_AC(t) = R P rho * Delta eta(t)``  [A]   (doc 05 §5, R-20).

Linearizing the optical coupling at the working point and substituting the
mechanical transfer (doc 05 §2),

``Delta eta = (d eta / d dx)_eff * dx_eff``,  ``dx_eff = H_lat(f) * a_x``,

gives the **through (end-to-end) sensitivity**

``s_target(f) = R P rho (d eta / d dx)_eff H_lat(f)``  [A/(m/s^2)]   (doc 05 §2).

On the off-resonance plateau (``f << f1``) ``|D(f)| -> 1`` and ``H_lat -> H_lat^QS``
so ``s_target -> s_target^QS = const`` and the calibration is a single signed
scalar (the off-resonance operating mode is the documented choice R-21). The
scalar is *signed*: ``(d eta / d dx)_eff < 0`` (doc 04 §4), so a positive
acceleration produces a negative photocurrent swing; dividing the AC current by
the signed ``s_target`` recovers the acceleration with the correct sign (the
slope inversion is handled, not discarded).

Two calibration sources (task S5 §1):

* **ideal** (the v1 default) -- ``s_target`` is computed from the variant config
  and the *same* S3/S2 models that generated the signal
  (:class:`~optivibe.optics.cylinder.CylinderOpticsModel`,
  :class:`~optivibe.mechanics.cantilever.CantileverModel`), so the recovery is a
  faithful test oracle (known exactly);
* **bench** -- ``s_target`` is *estimated* from a known calibration excitation
  (RMS of the AC current over RMS of the applied acceleration). Exposed as a
  helper :func:`bench_sensitivity` for the stand workflow (S6); not used by the
  default path.

The endface-pedestal normalization ``m = 1 + R1/(rho eta0)`` of doc 05 §5.3 is
the ratio between the *relative geometric* modulation ``Delta eta / eta0`` and
the *detected* modulation depth ``I_AC / I_DC``. It is **not** applied to the
canonical ``I_AC`` (the pedestal ``R1`` lives in ``I_DC``, not in ``I_AC``);
:func:`target_sensitivity_via_multiplier` re-derives the same ``s_target`` through
``m`` as an explicit consistency cross-check (doc 05 §5.3 is "accounted for").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from optivibe.core.config.loader import load_constants
from optivibe.core.config.models import Constants, VariantConfig
from optivibe.core.types import DetectorOutput, FloatArray
from optivibe.detector.photodiode import signal_multiplier
from optivibe.mechanics.cantilever import CantileverModel
from optivibe.optics.cylinder import CylinderOpticsModel

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids the sensitivity import cycle)
    from optivibe.dsp.sensitivity import SensitivityModel

ComplexArray = npt.NDArray[np.complex128]

__all__ = [
    "bench_sensitivity",
    "calibrate_acceleration",
    "detector_ac_current",
    "dynamic_sensitivity",
    "target_sensitivity",
    "target_sensitivity_via_multiplier",
]


def _resolve_constants(constants: Constants | None) -> Constants:
    """Return ``constants`` or load the default ``configs/constants.yaml``."""
    return load_constants() if constants is None else constants


def target_sensitivity(variant: VariantConfig, constants: Constants | None = None) -> float:
    """Signed plateau sensitivity ``s_target^QS``, A/(m/s^2) (doc 05 §2a).

    ``s_target^QS = R P rho (d eta / d dx)_eff H_lat^QS`` built from the same S3
    optics and S2 mechanics models that generate the forward signal (test-oracle
    consistency). The sign is preserved: with ``(d eta / d dx)_eff < 0`` the
    value is negative, so dividing the AC photocurrent by it recovers the
    acceleration with the correct sign.

    Parameters
    ----------
    variant : VariantConfig
        Sensor variant (R, P, rho, geometry, length).
    constants : Constants or None, optional
        Physical constants; the default ``configs/constants.yaml`` is loaded when
        ``None``.

    Returns
    -------
    float
        Signed through sensitivity on the plateau, A/(m/s^2).
    """
    consts = _resolve_constants(constants)
    optics = CylinderOpticsModel.from_config(variant)
    tcl = consts.tilt_displacement_coupling_per_l
    slope_eff = optics.effective_slope_dx(variant.length_m, tcl)  # 1/m, < 0
    cantilever = CantileverModel.from_config(consts, variant)
    h_qs = cantilever.h_lat_qs  # m/(m/s^2)
    gain = variant.responsivity_a_w * variant.source.power_w  # A
    return float(gain * variant.reflector.reflectivity * slope_eff * h_qs)


def target_sensitivity_via_multiplier(
    variant: VariantConfig, constants: Constants | None = None
) -> float:
    """Re-derive ``s_target^QS`` through the pedestal factor ``m`` (doc 05 §5.3).

    Cross-check that the endface normalization is accounted for. Writing the
    detected modulation depth ``I_AC/I_DC = (Delta eta / eta0) / m`` with
    ``m = 1 + R1/(rho eta0)`` and ``I_DC = R P (R1 + rho eta0)`` gives the
    identity ``I_DC / m = R P rho eta0``; hence

    ``s_target^QS = (I_DC / m) * ((d eta / d dx)_eff H_lat^QS / eta0)``

    which must equal :func:`target_sensitivity`. The sign is preserved.

    Parameters
    ----------
    variant : VariantConfig
        Sensor variant.
    constants : Constants or None, optional
        Physical constants (default loaded when ``None``).

    Returns
    -------
    float
        Signed through sensitivity, A/(m/s^2), via the modulation-depth form.
    """
    consts = _resolve_constants(constants)
    optics = CylinderOpticsModel.from_config(variant)
    tcl = consts.tilt_displacement_coupling_per_l
    slope_eff = optics.effective_slope_dx(variant.length_m, tcl)
    eta0 = optics.eta_working_point()
    cantilever = CantileverModel.from_config(consts, variant)
    h_qs = cantilever.h_lat_qs
    gain = variant.responsivity_a_w * variant.source.power_w
    r1 = variant.endface_reflectivity
    rho = variant.reflector.reflectivity
    i_dc = gain * (r1 + rho * eta0)
    multiplier = signal_multiplier(r1, rho, eta0)  # 1 + R1/(rho eta0)
    return float((i_dc / multiplier) * (slope_eff * h_qs / eta0))


def dynamic_sensitivity(
    variant: VariantConfig,
    freq_hz: FloatArray,
    constants: Constants | None = None,
) -> ComplexArray:
    """Complex through sensitivity ``s_target(f) = s_target^QS D(f)`` (doc 05 §2b).

    The single-mode dynamic factor ``D(f)`` rolls the plateau sensitivity up to
    ``~Q`` at ``f1`` (and rotates its phase). Used to refer the noise PSD to the
    input across the band (NEA(f)) and for the optional ``|H_lat(f)|``
    deconvolution near ``f1``.

    Parameters
    ----------
    variant : VariantConfig
        Sensor variant.
    freq_hz : numpy.ndarray
        Frequencies, Hz (non-negative).
    constants : Constants or None, optional
        Physical constants (default loaded when ``None``).

    Returns
    -------
    numpy.ndarray
        Complex sensitivity at each frequency, A/(m/s^2).
    """
    consts = _resolve_constants(constants)
    s_qs = target_sensitivity(variant, consts)
    cantilever = CantileverModel.from_config(consts, variant)
    factor = cantilever.dynamic_factor(freq_hz)
    return np.asarray(s_qs * factor, dtype=np.complex128)


def detector_ac_current(detector: DetectorOutput, variant: VariantConfig) -> FloatArray:
    """Recover the AC photocurrent ``I_AC(t)`` from the detector samples, A.

    The AC-coupled front end already removes the pedestal: the samples are
    ``dc_level + ADC(I - I_DC + noise)`` (doc 07 §1.4), so ``samples - dc_level``
    is the digitized AC modulation. When the detector reports a transimpedance
    *voltage* the modulation is divided by ``Rf`` to return amperes (so the
    calibration is independent of the read-out units).

    Parameters
    ----------
    detector : DetectorOutput
        Digitized detector signal (current or voltage).
    variant : VariantConfig
        Sensor variant (provides ``Rf`` for the voltage branch).

    Returns
    -------
    numpy.ndarray
        AC photocurrent ``I_AC(t)``, A.

    Raises
    ------
    ValueError
        If the output is in volts but no transimpedance ``Rf`` is configured.
    """
    ac = np.asarray(detector.samples, dtype=np.float64) - detector.dc_level
    if detector.units == "V":
        rf = variant.detector.transimpedance_ohm
        if rf is None:
            msg = "voltage detector output requires a transimpedance_ohm to calibrate"
            raise ValueError(msg)
        ac = ac / rf
    return np.ascontiguousarray(ac, dtype=np.float64)


def calibrate_acceleration(
    detector: DetectorOutput,
    variant: VariantConfig,
    constants: Constants | None = None,
    *,
    model: SensitivityModel | None = None,
) -> tuple[FloatArray, float]:
    """Calibrate detector samples to target-axis acceleration, m/s^2 (doc 05 §2).

    Recovers the AC photocurrent and turns it into acceleration through a
    sensitivity *model* (task S6 §A, decision SW-33). When ``model`` is ``None``
    the v1 path is used unchanged -- divide by the signed plateau scalar
    ``s_target^QS`` :func:`target_sensitivity` -- so the default is bit-identical
    to S5. A model lets the caller switch the operating-point binding (static /
    operating-point / non-linear-curve) without changing this signature.

    Valid on the off-resonance plateau (``f << f1``, the documented operating mode
    R-21); near ``f1`` use the ``|H_lat(f)|`` deconvolution (axis C).

    Parameters
    ----------
    detector : DetectorOutput
        Digitized detector signal.
    variant : VariantConfig
        Sensor variant.
    constants : Constants or None, optional
        Physical constants (default loaded when ``None``).
    model : SensitivityModel or None, optional
        Operating-point binding strategy (axis B). ``None`` selects the v1
        static-plateau behaviour.

    Returns
    -------
    accel : numpy.ndarray
        Reconstructed acceleration on the target axis, m/s^2.
    s_target : float
        The signed plateau sensitivity used, A/(m/s^2).
    """
    consts = _resolve_constants(constants)
    i_ac = detector_ac_current(detector, variant)
    if model is None:
        s_target = target_sensitivity(variant, consts)
        accel: FloatArray = i_ac / s_target
        return np.ascontiguousarray(accel, dtype=np.float64), s_target
    recovered = model.recover_acceleration(i_ac, detector.fs)
    return np.ascontiguousarray(recovered, dtype=np.float64), model.plateau_value


def bench_sensitivity(
    detector: DetectorOutput,
    reference_accel: FloatArray,
    variant: VariantConfig,
) -> float:
    """Estimate ``s_target`` from a known calibration excitation, A/(m/s^2).

    Stand-calibration helper (task S5 §1b; for S6): the RMS of the AC current
    over the RMS of the applied reference acceleration, signed by their
    correlation so the slope inversion is preserved. Not used by the default
    ideal path.

    Parameters
    ----------
    detector : DetectorOutput
        Digitized detector signal of the calibration run.
    reference_accel : numpy.ndarray
        The known applied acceleration on the target axis, m/s^2.
    variant : VariantConfig
        Sensor variant (for the voltage branch ``Rf``).

    Returns
    -------
    float
        Estimated signed sensitivity, A/(m/s^2).

    Raises
    ------
    ValueError
        If the reference acceleration is identically zero (no excitation).
    """
    i_ac = detector_ac_current(detector, variant)
    ref = np.asarray(reference_accel, dtype=np.float64)
    ref_rms = float(np.sqrt(np.mean(ref**2)))
    if ref_rms == 0.0:
        msg = "bench_sensitivity needs a non-zero reference acceleration"
        raise ValueError(msg)
    ac_rms = float(np.sqrt(np.mean(i_ac**2)))
    sign = float(np.sign(np.mean(i_ac * ref))) or 1.0
    return sign * ac_rms / ref_rms
