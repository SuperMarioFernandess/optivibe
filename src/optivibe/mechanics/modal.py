"""Modal mechanics solvers: Excitation -> TipState via H_lat(f) (docs 02/05).

Two implementations of the :class:`~optivibe.core.stages.MechanicsStage`
protocol share the same :class:`~optivibe.mechanics.cantilever.CantileverModel`:

``ModalFrequencyMechanics`` ("modal", the S2 default)
    FFT of each axis, multiplication by the per-axis transfer function, inverse
    FFT. Exact for (quasi-)stationary signals; uses the periodic-extension
    semantics of the DFT, so for one-shot transients (shock) the lightly damped
    resonant tail (time constant ``Q / (pi f1)``) wraps around unless the
    record is long enough — for transient-exact results use "modal_time".

``ModalTimeMechanics`` ("modal_time", optional solver)
    State-space mode-1 oscillator integrated with :func:`scipy.signal.lsim` —
    the seam for future nonlinearities and shock studies (doc 11 §2.2). Starts
    from rest (zero initial conditions).

Both apply, per doc 02 §7:

* ``dx = H_lat(f) * a_x``; ``dy = H_lat(f) * a_y`` (axisymmetric cantilever —
  axis separation is the optics' job, docs 02 §1, 03/04);
* ``theta_y = 1.377/L * dx``, ``theta_x = 1.377/L * dy`` (frequency-independent
  modal coupling, docs 02 §7.1, 05 §3.3);
* ``dz = rho L^2/(2E) * a_z`` (quasi-static axial channel, ~1.4 pm/g at 3 mm,
  doc 02 §7.2; the second-order geometric gap term is a recorded loop).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import StateSpace, lsim

from optivibe.core.config.loader import default_config_dir, load_constants
from optivibe.core.config.models import Constants, VariantConfig
from optivibe.core.types import Excitation, FloatArray, TipState
from optivibe.mechanics.cantilever import CantileverModel

__all__ = ["ModalFrequencyMechanics", "ModalTimeMechanics"]


class _ModalBase:
    """Shared construction of the two modal solvers.

    Parameters
    ----------
    q_total : float or None, optional
        Scenario-level override of the variant quality factor (docs 07/08);
        the variant's ``q_total`` is used when None. Injected by the
        orchestrator from ``scenario.mechanics``.
    constants : Constants or None, optional
        Physical constants; loaded once from ``configs/constants.yaml`` when
        None (the only I/O, performed at construction).
    """

    def __init__(self, *, q_total: float | None = None, constants: Constants | None = None) -> None:
        if constants is None:
            constants = load_constants(default_config_dir() / "constants.yaml")
        self._constants = constants
        self._q_total = q_total

    def model(self, variant: VariantConfig) -> CantileverModel:
        """Return the cantilever model for ``variant`` (with any Q override).

        Parameters
        ----------
        variant : VariantConfig
            Sensor variant.

        Returns
        -------
        CantileverModel
            Derived mechanical quantities.
        """
        return CantileverModel.from_config(self._constants, variant, q_total=self._q_total)

    @staticmethod
    def _pack(
        model: CantileverModel,
        dx: FloatArray,
        dy: FloatArray,
        a_z: FloatArray,
        fs: float,
    ) -> TipState:
        """Assemble the tip state from the lateral responses and a_z.

        Applies the tilt coupling (doc 02 §7.1) and the quasi-static axial
        channel (doc 02 §7.2).
        """
        return TipState(
            dx=dx,
            dy=dy,
            dz=model.axial_compliance * a_z,
            theta_x=model.tilt_per_m * dy,
            theta_y=model.tilt_per_m * dx,
            fs=fs,
        )


class ModalFrequencyMechanics(_ModalBase):
    """Frequency-domain modal solver (registry key "modal"; doc 11 §2.2).

    ``a(t) -> rfft -> * H_lat(f) -> irfft`` per lateral axis; the transverse
    FRF is single-mode ``H_lat(f) = H_lat^QS * D(f)`` (docs 02 §6, 05 §1).
    """

    def run(self, excitation: Excitation, variant: VariantConfig) -> TipState:
        """Compute the tip-state time series.

        Parameters
        ----------
        excitation : Excitation
            Base acceleration on x/y/z, m/s^2.
        variant : VariantConfig
            Sensor variant (L, q_total, ...).

        Returns
        -------
        TipState
            Tip displacements (m) and tilts (rad).
        """
        model = self.model(variant)
        n = excitation.n_samples
        freq = np.fft.rfftfreq(n, d=1.0 / excitation.fs).astype(np.float64)
        h = model.h_lat(freq)
        dx = np.fft.irfft(np.fft.rfft(excitation.a_x) * h, n=n).astype(np.float64)
        dy = np.fft.irfft(np.fft.rfft(excitation.a_y) * h, n=n).astype(np.float64)
        return self._pack(model, dx, dy, excitation.a_z, excitation.fs)


class ModalTimeMechanics(_ModalBase):
    """Time-domain state-space solver (registry key "modal_time"; doc 11 §2.2).

    Mode-1 oscillator ``x'' + (w1/Q) x' + w1^2 x = H_QS w1^2 a(t)`` whose
    transfer function equals ``H_lat^QS * D(f)`` exactly (doc 05 §1), so the
    two solvers agree in steady state (golden cross-check). Integration is
    :func:`scipy.signal.lsim` from rest; sampling must comfortably resolve the
    signal band of interest.
    """

    def run(self, excitation: Excitation, variant: VariantConfig) -> TipState:
        """Compute the tip-state time series by time integration.

        Parameters
        ----------
        excitation : Excitation
            Base acceleration on x/y/z, m/s^2.
        variant : VariantConfig
            Sensor variant (L, q_total, ...).

        Returns
        -------
        TipState
            Tip displacements (m) and tilts (rad).
        """
        model = self.model(variant)
        w1 = 2.0 * np.pi * model.f1_hz
        system = StateSpace(
            [[0.0, 1.0], [-(w1**2), -w1 / model.q_total]],
            [[0.0], [model.h_lat_qs * w1**2]],
            [[1.0, 0.0]],
            [[0.0]],
        )
        t = np.arange(excitation.n_samples, dtype=np.float64) / excitation.fs
        dx = self._simulate(system, t, excitation.a_x)
        dy = self._simulate(system, t, excitation.a_y)
        return self._pack(model, dx, dy, excitation.a_z, excitation.fs)

    @staticmethod
    def _simulate(system: StateSpace, t: FloatArray, drive: FloatArray) -> FloatArray:
        """Integrate one lateral axis (zero response for a zero drive)."""
        if not np.any(drive):
            return np.zeros_like(t)
        _, response, _ = lsim(system, U=drive, T=t)
        return np.asarray(response, dtype=np.float64)
