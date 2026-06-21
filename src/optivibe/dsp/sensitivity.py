"""Switchable sensitivity model ``s_target`` (task S6 §A; decision SW-33).

``s_target`` has *four orthogonal axes of choice*; SW-33 makes them composable
switches rather than one 24-way enum. This module owns axis **B** (the
*algorithms*) as a registry and provides the vector-ready seam for axis **D**.

Axes (SW-33, doc 13)
--------------------
* **A. Source** -- flag ``DspOptions.calibration in {ideal, bench}`` (S5). The
  model is computed from the config (``ideal``, the v1 default) or estimated on a
  stand (``bench``, helper :func:`~optivibe.dsp.calibration.bench_sensitivity`).
  Live-pipeline ``bench`` needs a reference excitation and is *deferred* (14 §8):
  :func:`build_sensitivity_model` rejects it loudly.
* **B. Operating-point binding** -- this registry ``SENSITIVITY_REGISTRY``
  (family ``dsp.sensitivity``). Keys: ``static`` (scalar at the nominal bias, the
  v1 default), ``operating_point`` (recompute ``s_target`` at the SNR-optimum bias
  via the 0.37 rule, doc 08 R-40), ``nonlinear_curve`` (point-wise inversion of
  the monotonic branch of ``eta(dx)`` for the >50 g / non-linearity study, doc 00).
* **C. Frequency** -- flag ``DspOptions.sensitivity_freq in {plateau, dynamic}``;
  ``plateau`` uses the QS scalar ``s_target^QS`` and ``dynamic`` the complex
  ``s_target(f) = s_target^QS D(f)`` (applied by the ``deconvolve_hlat``
  mechanism in :class:`~optivibe.dsp.standard.StandardDsp`). A *parameter* of the
  strategy, not a registry key.
* **D. Scalar vs vector (full 3-D inverse)** -- a *seam* only. The protocol
  :class:`SensitivityModel` returns a :class:`Sensitivity` whose ``value`` is a
  scalar in v1 but whose type already admits a row of the Jacobian
  ``d I_AC / d a`` (vector by axis) and an array by frequency, so the 3-D
  pseudo-inverse reuses the same signature later (:func:`recover_acceleration_3d`
  raises until then).

Invariants (S6 golden, doc 13 SW-33)
------------------------------------
``static + ideal + plateau`` reproduces v1 *exactly* (``s_target`` and the
recovered acceleration are bit-identical to the S5 path); invalid combinations
(``nonlinear_curve + bench`` without a measured curve; ``vector`` before axis D)
fail loudly (10 §7); each axis is tested independently (the 24-way cross product
is never enumerated).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from optivibe.core.config.loader import load_constants
from optivibe.core.config.models import Constants, DspOptions, VariantConfig
from optivibe.core.registry import Registry
from optivibe.core.types import FloatArray
from optivibe.dsp.calibration import target_sensitivity
from optivibe.mechanics.cantilever import CantileverModel
from optivibe.optics.cylinder import CylinderOpticsModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

ComplexArray = npt.NDArray[np.complex128]

# The SNR-optimum working-point ratio eta0/eta_peak (doc 08 R-40/O-05).
DEFAULT_ETA_RATIO = 0.37

__all__ = [
    "DEFAULT_ETA_RATIO",
    "SENSITIVITY_REGISTRY",
    "NonlinearCurveSensitivity",
    "OperatingPointSensitivity",
    "Sensitivity",
    "SensitivityModel",
    "StaticSensitivity",
    "TipPoint",
    "build_sensitivity_model",
    "recover_acceleration_3d",
]


@dataclass(frozen=True)
class TipPoint:
    """A single tip-state point ``(dx, dy, dz, theta_x, theta_y)`` (doc 04 §2).

    The state at which a :class:`SensitivityModel` is evaluated. All-zero is the
    *working point* (the nominal bias only), used for the QS plateau scalar.

    Parameters
    ----------
    dx, dy, dz : float
        Transverse displacements and gap change, m.
    theta_x, theta_y : float
        Tip tilts, rad.
    """

    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    theta_x: float = 0.0
    theta_y: float = 0.0


#: The working point (zeros): the model is evaluated at its nominal bias.
WORKING_POINT = TipPoint()


@dataclass(frozen=True)
class Sensitivity:
    """Through-sensitivity sample(s) at a tip state (the axis-D seam).

    Parameters
    ----------
    value : numpy.ndarray (complex)
        The sensitivity ``d I_AC / d a``, A/(m/s^2). Shape ``()`` for the QS
        plateau scalar (real value carried as complex with zero imaginary part)
        and ``(n_freq,)`` for the dynamic case (complex, carrying the ``D(f)``
        phase). The type also admits a *vector by axis* (a row of the Jacobian
        over the five state inputs) and an array by frequency, reserved for the
        3-D inverse (axis D).
    target_axis : str
        Which axis this sensitivity refers to (``"x"`` in v1).
    freq_hz : numpy.ndarray or None
        Frequencies of a dynamic sample (``None`` for the QS plateau scalar).
    """

    value: ComplexArray
    target_axis: str = "x"
    freq_hz: FloatArray | None = field(default=None)

    @property
    def is_dynamic(self) -> bool:
        """Whether this is a dynamic (per-frequency) sample."""
        return self.freq_hz is not None

    def plateau_scalar(self) -> float:
        """Return the real plateau scalar, A/(m/s^2).

        Returns
        -------
        float
            The (real) target-axis sensitivity.

        Raises
        ------
        ValueError
            If the value is not a scalar (a dynamic or vector sample): the
            plateau scalar is only defined for the QS working point.
        """
        arr = np.asarray(self.value)
        if arr.shape != ():
            msg = (
                "plateau_scalar is only defined for the QS scalar sample; "
                f"got shape {arr.shape} (dynamic/vector — use .value)"
            )
            raise ValueError(msg)
        return float(arr.real)


@runtime_checkable
class SensitivityModel(Protocol):
    """Protocol of an operating-point binding strategy (SW-33 axis B/D).

    A model knows the sensor variant and resolves the through-sensitivity at any
    tip state and frequency, and how to turn an AC photocurrent into acceleration
    (a division for the linear strategies, a curve inversion for the non-linear
    one). The signature is vector-ready (axis D): ``at`` returns a
    :class:`Sensitivity` whose type already admits a vector by axis.
    """

    @property
    def plateau_value(self) -> float:
        """Signed QS plateau sensitivity ``s_target^QS``, A/(m/s^2)."""
        ...

    def at(self, state: TipPoint, freq_hz: FloatArray | None = None) -> Sensitivity:
        """Through-sensitivity at ``state`` (and ``freq_hz`` if dynamic)."""

    def recover_acceleration(self, i_ac: FloatArray, fs: float) -> FloatArray:
        """Reconstruct target-axis acceleration from the AC photocurrent, m/s^2."""


class _LinearSensitivity:
    """Shared base of the linear strategies: divide by the plateau scalar.

    Holds the signed plateau sensitivity ``s_target^QS`` and the variant's
    cantilever (for the dynamic ``D(f)`` of axis C). Subclasses only differ in
    *how* the plateau scalar is computed (nominal bias vs SNR-optimum bias).
    """

    def __init__(self, variant: VariantConfig, constants: Constants, plateau: float) -> None:
        self._variant = variant
        self._constants = constants
        self._plateau = plateau
        self._cantilever = CantileverModel.from_config(constants, variant)

    @property
    def plateau_value(self) -> float:
        """Signed QS plateau sensitivity used by this model, A/(m/s^2)."""
        return self._plateau

    def at(self, state: TipPoint, freq_hz: FloatArray | None = None) -> Sensitivity:
        """Through-sensitivity at the working point (scalar) or per-frequency.

        The linear strategies are working-point quantities, so ``state`` does not
        change the scalar; ``freq_hz`` rolls the plateau up by ``D(f)`` (axis C).

        Parameters
        ----------
        state : TipPoint
            Tip state (unused by the linear scalar; present for the protocol).
        freq_hz : numpy.ndarray or None, optional
            Frequencies for a dynamic sample; ``None`` returns the QS scalar.

        Returns
        -------
        Sensitivity
            The QS scalar (``freq_hz`` None) or the complex ``s_target(f)``.
        """
        del state  # the linear scalar is a working-point quantity
        if freq_hz is None:
            return Sensitivity(value=np.asarray(self._plateau, dtype=np.complex128))
        freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
        s_f: ComplexArray = self._plateau * self._cantilever.dynamic_factor(freq)
        return Sensitivity(value=np.asarray(s_f, dtype=np.complex128), freq_hz=freq)

    def recover_acceleration(self, i_ac: FloatArray, fs: float) -> FloatArray:
        """Divide the AC photocurrent by the signed plateau scalar, m/s^2.

        Parameters
        ----------
        i_ac : numpy.ndarray
            AC photocurrent ``I_AC(t)``, A.
        fs : float
            Sampling frequency, Hz (unused; present for the protocol).

        Returns
        -------
        numpy.ndarray
            Reconstructed target-axis acceleration, m/s^2.
        """
        del fs
        accel: FloatArray = i_ac / self._plateau
        return np.ascontiguousarray(accel, dtype=np.float64)


class StaticSensitivity(_LinearSensitivity):
    """Plateau scalar at the *nominal* config bias (v1 default, key ``static``).

    The plateau scalar is exactly :func:`~optivibe.dsp.calibration.target_sensitivity`,
    so ``static + ideal + plateau`` reproduces the S5 calibration bit-for-bit.

    Parameters
    ----------
    variant : VariantConfig
        Sensor variant.
    constants : Constants
        Physical constants.
    """

    def __init__(self, variant: VariantConfig, constants: Constants) -> None:
        super().__init__(variant, constants, target_sensitivity(variant, constants))


class OperatingPointSensitivity(_LinearSensitivity):
    """Plateau scalar at the SNR-optimum bias (key ``operating_point``).

    Recomputes ``s_target`` at the bias that places the working point at
    ``eta0 = eta_ratio * eta_peak`` (the 0.37 rule, doc 08 R-40/O-05), using the
    closed-form ``Delta x0 = sigma sqrt(-ln eta_ratio)``
    (:meth:`~optivibe.optics.cylinder.CylinderOpticsModel.bias_for_eta_ratio`).
    Distinct from ``static`` whenever the configured bias differs from the
    0.37-bias; supports bias studies (sweep ``eta_ratio``). Note the |slope|
    maximum is at ``Delta x0 = sigma / sqrt(2)`` while 0.37 gives
    ``Delta x0 ~ sigma`` -- which to use is the subject of the S6 study (SW-33).

    Parameters
    ----------
    variant : VariantConfig
        Sensor variant.
    constants : Constants
        Physical constants.
    eta_ratio : float, optional
        Target ``eta0 / eta_peak`` in (0, 1]; defaults to 0.37.
    """

    def __init__(
        self,
        variant: VariantConfig,
        constants: Constants,
        *,
        eta_ratio: float = DEFAULT_ETA_RATIO,
    ) -> None:
        self._eta_ratio = eta_ratio
        optics = CylinderOpticsModel.from_config(variant)
        bias = optics.bias_for_eta_ratio(eta_ratio)
        rebiased = CylinderOpticsModel(
            beam=optics.beam,
            gap_m=optics.gap_m,
            radius_of_curvature_m=optics.radius_of_curvature_m,
            bias_m=bias,
        )
        tcl = constants.tilt_displacement_coupling_per_l
        slope_eff = rebiased.effective_slope_dx(variant.length_m, tcl)
        cantilever = CantileverModel.from_config(constants, variant)
        gain = variant.responsivity_a_w * variant.source.power_w
        plateau = float(gain * variant.reflector.reflectivity * slope_eff * cantilever.h_lat_qs)
        super().__init__(variant, constants, plateau)

    @property
    def eta_ratio(self) -> float:
        """Target ``eta0 / eta_peak`` of this operating point."""
        return self._eta_ratio

    @property
    def bias_m(self) -> float:
        """Bias ``Delta x0`` placing the working point at ``eta_ratio`` (m)."""
        optics = CylinderOpticsModel.from_config(self._variant)
        return optics.bias_for_eta_ratio(self._eta_ratio)


class NonlinearCurveSensitivity:
    """Point-wise inversion of the monotonic ``eta(dx)`` branch (key ``nonlinear_curve``).

    For large excursions (>50 g, the non-linearity / break study of doc 00) the
    scalar slope systematically lies (``THD ~ 2.5 %`` at 0.5 um, doc 03 §5). This
    strategy reconstructs the *instantaneous* coupling
    ``eta(t) = eta0 + I_AC(t) / (R P rho)`` and inverts the closed-form Gaussian
    ``eta = eta_peak exp(-(dx_eff/sigma)^2)`` on the branch containing the working
    point (``dx_eff > 0``, the falling side), then removes the bias and the tilt
    lever and the QS compliance:

    ``dx_eff = sigma sqrt(-ln(eta/eta_peak))``  (working-point branch),
    ``dx = (dx_eff - Delta x0) / [1 + 1.377 (R_c + A)/L]``,  ``a = dx / H_lat^QS``.

    Exact for pure target-axis motion (the z-channel is negligible, SW-26), it
    reduces to the linear ``static`` recovery for small signals and removes the
    harmonic distortion the scalar leaves at large amplitude. The *full* 3-D /
    breakage inverse stays deferred (14 §8).

    Parameters
    ----------
    variant : VariantConfig
        Sensor variant.
    constants : Constants
        Physical constants.
    """

    def __init__(self, variant: VariantConfig, constants: Constants) -> None:
        self._variant = variant
        self._constants = constants
        optics = CylinderOpticsModel.from_config(variant)
        cantilever = CantileverModel.from_config(constants, variant)
        tcl = constants.tilt_displacement_coupling_per_l
        self._gain = variant.responsivity_a_w * variant.source.power_w  # R P, A/W * W = A
        self._rho = variant.reflector.reflectivity
        self._eta_peak = optics.eta_peak()
        self._eta0 = optics.eta_working_point()
        self._sigma = optics.sigma_m
        self._bias = optics.bias_m
        self._tilt_mult = optics.tilt_multiplier(variant.length_m, tcl)
        self._h_qs = cantilever.h_lat_qs
        self._optics = optics
        self._plateau = float(
            self._gain * self._rho * optics.effective_slope_dx(variant.length_m, tcl) * self._h_qs
        )

    @property
    def plateau_value(self) -> float:
        """Working-point effective slope ``s_target^QS`` (for NEA/reporting)."""
        return self._plateau

    def at(self, state: TipPoint, freq_hz: FloatArray | None = None) -> Sensitivity:
        """Local (per-state) through-sensitivity, the seam for the non-linear slope.

        Returns the *local* effective slope at ``state.dx`` -- the tangent of the
        ``eta(dx)`` curve -- which equals the plateau at the working point and
        diverges from it for large ``dx`` (the reason the scalar lies).

        Parameters
        ----------
        state : TipPoint
            Tip state; ``dx`` (and the coupled ``theta_y``) set ``dx_eff``.
        freq_hz : numpy.ndarray or None, optional
            Dynamic frequencies (not supported by the non-linear local slope yet).

        Returns
        -------
        Sensitivity
            The local effective-slope scalar.

        Raises
        ------
        ValueError
            If ``freq_hz`` is given: dynamic non-linear sensitivity is deferred.
        """
        if freq_hz is not None:
            msg = (
                "dynamic nonlinear_curve sensitivity is deferred (14 §8); "
                "use sensitivity_freq=plateau"
            )
            raise ValueError(msg)
        dx_eff = self._bias + state.dx * self._tilt_mult
        eta = self._eta_peak * math.exp(-((dx_eff / self._sigma) ** 2))
        local_bare = -2.0 * eta * dx_eff / self._sigma**2
        local_eff = local_bare * self._tilt_mult
        s_local = self._gain * self._rho * local_eff * self._h_qs
        return Sensitivity(value=np.asarray(s_local, dtype=np.complex128))

    def recover_acceleration(self, i_ac: FloatArray, fs: float) -> FloatArray:
        """Invert ``eta(dx)`` point-wise to reconstruct acceleration, m/s^2.

        Parameters
        ----------
        i_ac : numpy.ndarray
            AC photocurrent ``I_AC(t)``, A.
        fs : float
            Sampling frequency, Hz (unused; present for the protocol).

        Returns
        -------
        numpy.ndarray
            Reconstructed target-axis acceleration, m/s^2.
        """
        del fs
        delta_eta = i_ac / (self._gain * self._rho)
        eta = self._eta0 + delta_eta
        # Stay on the invertible branch: 0 < eta <= eta_peak.
        eta_clipped = np.clip(eta, 1.0e-12, self._eta_peak)
        arg = -np.log(eta_clipped / self._eta_peak)
        dx_eff = self._sigma * np.sqrt(np.maximum(arg, 0.0))  # working-point (dx_eff > 0) branch
        dx = (dx_eff - self._bias) / self._tilt_mult
        accel: FloatArray = dx / self._h_qs
        return np.ascontiguousarray(accel, dtype=np.float64)


SENSITIVITY_REGISTRY: Registry[SensitivityModel] = Registry("dsp.sensitivity")
SENSITIVITY_REGISTRY.register("static")(StaticSensitivity)
SENSITIVITY_REGISTRY.register("operating_point")(OperatingPointSensitivity)
SENSITIVITY_REGISTRY.register("nonlinear_curve")(NonlinearCurveSensitivity)


def build_sensitivity_model(
    variant: VariantConfig,
    options: DspOptions,
    constants: Constants | None = None,
) -> SensitivityModel:
    """Resolve, validate and construct the sensitivity model from the options.

    Composes axes A (``calibration``) and B (``sensitivity_model``); axis C
    (``sensitivity_freq``) is applied downstream by the DSP stage. Rejects the
    invalid combinations loudly (10 §7, SW-33):

    * ``nonlinear_curve + bench`` -- a non-linear inversion needs a *measured*
      ``eta(dx)`` curve, which the scalar bench RMS does not provide;
    * any ``bench`` in a live run -- the bench source needs a reference
      excitation; live-pipeline bench is *deferred* (14 §8), use
      :func:`~optivibe.dsp.calibration.bench_sensitivity` in the stand workflow.

    Parameters
    ----------
    variant : VariantConfig
        Sensor variant.
    options : DspOptions
        Inverse/DSP options (selects ``sensitivity_model`` and ``calibration``).
    constants : Constants or None, optional
        Physical constants; the default ``configs/constants.yaml`` is loaded when
        ``None``.

    Returns
    -------
    SensitivityModel
        The constructed strategy.

    Raises
    ------
    ValueError
        On an invalid axis combination (see above).
    """
    consts = load_constants() if constants is None else constants
    if options.sensitivity_model == "nonlinear_curve" and options.calibration == "bench":
        msg = (
            "invalid sensitivity combination: 'nonlinear_curve' needs a measured "
            "eta(dx) curve, but the 'bench' source provides only a scalar RMS "
            "sensitivity (SW-33; doc 10 §7)"
        )
        raise ValueError(msg)
    if options.calibration == "bench":
        msg = (
            "live-pipeline 'bench' calibration needs a reference excitation and is "
            "deferred (14 §8 loop); use dsp.bench_sensitivity in the stand workflow"
        )
        raise ValueError(msg)
    factory: Callable[..., SensitivityModel] = SENSITIVITY_REGISTRY.get(options.sensitivity_model)
    return factory(variant, consts)


def recover_acceleration_3d(
    i_ac: FloatArray,
    variant: VariantConfig,
    constants: Constants | None = None,
) -> FloatArray:
    """Full 3-D (vector) inverse -- *deferred* seam (axis D, SW-33).

    The protocol :class:`SensitivityModel` already returns a vector-capable
    :class:`Sensitivity`; the pseudo-inverse over the five state inputs reuses the
    same signature but is not implemented in v1.

    Raises
    ------
    NotImplementedError
        Always: the 3-D vector inverse is reserved for axis D (S6+).
    """
    del i_ac, variant, constants
    msg = (
        "3-D vector inverse (full eta Jacobian pseudo-inverse) is the axis-D seam "
        "and is not implemented in v1 (SW-33; reserved for a later chat)"
    )
    raise NotImplementedError(msg)
