"""Stage protocols: the typed input/output contracts of each pipeline stage.

Per architecture 09 §3/§6, every stage is a typed unit ``Stage[In] -> Out``.
These :class:`typing.Protocol` definitions live in ``core`` (they depend only on
the data contracts and config models, never on a concrete implementation), so the
domain packages can register implementations and the pipeline can wire them
without any import cycles. Implementations are *structural*: any class with the
right method satisfies the protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from optivibe.core.config.models import DspOptions, ExcitationSpec, VariantConfig
from optivibe.core.types import (
    DetectorOutput,
    Excitation,
    OpticalResponse,
    TipState,
    VibrationResult,
)


@runtime_checkable
class ExcitationSource(Protocol):
    """Produces a 3-axis :class:`~optivibe.core.types.Excitation` from a spec."""

    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Generate base acceleration a(t).

        Parameters
        ----------
        spec : ExcitationSpec
            Generator parameters (kind, fs, duration, amplitude, axis).
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        Excitation
            Generated input signal.
        """
        ...


@runtime_checkable
class MechanicsStage(Protocol):
    """Maps base acceleration to the fiber tip state q_tip(t)."""

    def run(self, excitation: Excitation, variant: VariantConfig) -> TipState:
        """Compute the tip-state time series.

        Parameters
        ----------
        excitation : Excitation
            Base acceleration on x/y/z.
        variant : VariantConfig
            Sensor variant (provides L, mode, ...).

        Returns
        -------
        TipState
            Tip displacements and tilts over time.
        """
        ...


@runtime_checkable
class OpticsStage(Protocol):
    """Maps the tip state to the optical coupling response eta(t)."""

    def run(self, tip: TipState, variant: VariantConfig) -> OpticalResponse:
        """Compute the optical response.

        Parameters
        ----------
        tip : TipState
            Tip-state time series.
        variant : VariantConfig
            Sensor variant (provides R_c, eta0, ...).

        Returns
        -------
        OpticalResponse
            Coupling efficiency around the bias working point.
        """
        ...


@runtime_checkable
class DetectorStage(Protocol):
    """Maps the optical response to a digitized detector signal."""

    def run(self, optical: OpticalResponse, variant: VariantConfig) -> DetectorOutput:
        """Compute the digitized photodetector output.

        Parameters
        ----------
        optical : OpticalResponse
            Optical coupling response.
        variant : VariantConfig
            Sensor variant (provides P, R, rho, R1, ...).

        Returns
        -------
        DetectorOutput
            Digitized samples with DC level and metadata.
        """
        ...


@runtime_checkable
class DspStage(Protocol):
    """Inverse stage: detector samples -> reconstructed vibration on target axis."""

    def run(
        self, detector: DetectorOutput, variant: VariantConfig, options: DspOptions
    ) -> VibrationResult:
        """Reconstruct acceleration/velocity/displacement and metrics.

        Parameters
        ----------
        detector : DetectorOutput
            Digitized detector signal.
        variant : VariantConfig
            Sensor variant (for calibration).
        options : DspOptions
            Integrator/spectrum options.

        Returns
        -------
        VibrationResult
            Reconstructed signals, spectrum and metrics.
        """
        ...
