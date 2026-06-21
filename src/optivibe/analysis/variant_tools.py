"""Shared helpers: variant overrides and analytic design-point metrics (S6 §B).

Both the parameter sweep and the Monte-Carlo build *perturbed* variants and read
out analytic figures (``s_target``, NEA, modulation, dynamic range) without a
time-domain run, so the design maps are cheap. The mutation goes through
pydantic ``model_copy`` so the frozen contracts stay intact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from optivibe.core.config.loader import load_constants
from optivibe.core.config.models import Constants, VariantConfig
from optivibe.detector.photodiode import noise_psd
from optivibe.dsp.calibration import target_sensitivity
from optivibe.mechanics.cantilever import CantileverModel
from optivibe.optics.cylinder import CylinderOpticsModel

__all__ = ["AnalyticPoint", "analytic_point", "with_overrides"]

G0 = 9.80665


def with_overrides(
    variant: VariantConfig,
    *,
    length_m: float | None = None,
    radius_of_curvature_m: float | None = None,
    power_w: float | None = None,
    bias_offset_m: float | None = None,
    full_scale_g: float | None = None,
    gap_m: float | None = None,
    q_total: float | None = None,
) -> VariantConfig:
    """Return a copy of ``variant`` with the given fields overridden.

    Nested blocks (reflector / source / optics) are copied through
    ``model_copy``; the geometry guards of the optics model still apply when the
    variant is later used (an invalid ``R_c`` fails loudly).

    Parameters
    ----------
    variant : VariantConfig
        Base variant.
    length_m, radius_of_curvature_m, power_w : float or None
        Top-level and reflector/source overrides; ``None`` keeps the base value.
    bias_offset_m, full_scale_g, gap_m, q_total : float or None
        Optics / full-scale / gap / quality-factor overrides; ``None`` keeps the
        base value.

    Returns
    -------
    VariantConfig
        The perturbed variant.
    """
    updates: dict[str, object] = {}
    if length_m is not None:
        updates["length_m"] = length_m
    if full_scale_g is not None:
        updates["full_scale_g"] = full_scale_g
    if q_total is not None:
        updates["q_total"] = q_total
    if radius_of_curvature_m is not None:
        updates["reflector"] = variant.reflector.model_copy(
            update={"radius_of_curvature_m": radius_of_curvature_m}
        )
    if power_w is not None:
        updates["source"] = variant.source.model_copy(update={"power_w": power_w})
    optics_update: dict[str, object] = {}
    if bias_offset_m is not None:
        optics_update["bias_offset_m"] = bias_offset_m
    if gap_m is not None:
        optics_update["gap_m"] = gap_m
    if optics_update:
        updates["optics"] = variant.optics.model_copy(update=optics_update)
    return variant.model_copy(update=updates)


@dataclass(frozen=True)
class AnalyticPoint:
    """Analytic design-point metrics of one variant (no time-domain run).

    Attributes
    ----------
    s_target : float
        Signed plateau sensitivity, A/(m/s^2).
    eta0, eta_peak, eta_ratio : float
        Working point, aligned peak and their ratio (dimensionless).
    f1_hz : float
        First mechanical mode, Hz.
    i_dc_a : float
        DC photocurrent, A.
    nea_plateau : float
        Plateau NEA density, (m/s^2)/sqrt(Hz).
    nea_plateau_ug : float
        Plateau NEA density in ug/sqrt(Hz) (the reporting unit of doc 07/08).
    nea_full_band : float
        Full-band NEA over the band, m/s^2 (RMS).
    bandwidth_hz : float
        Band used for the full-band figure, Hz.
    modulation_at_fs : float
        AC/DC modulation depth at full scale (dimensionless).
    dynamic_range_db : float
        Full-band dynamic range, dB (full-scale signal over the noise floor).
    """

    s_target: float
    eta0: float
    eta_peak: float
    eta_ratio: float
    f1_hz: float
    i_dc_a: float
    nea_plateau: float
    nea_plateau_ug: float
    nea_full_band: float
    bandwidth_hz: float
    modulation_at_fs: float
    dynamic_range_db: float


def analytic_point(variant: VariantConfig, constants: Constants | None = None) -> AnalyticPoint:
    """Compute analytic sensitivity/NEA/modulation for one variant (S6 §B7).

    Parameters
    ----------
    variant : VariantConfig
        Sensor variant (possibly perturbed by :func:`with_overrides`).
    constants : Constants or None, optional
        Physical constants (default loaded when ``None``).

    Returns
    -------
    AnalyticPoint
        The analytic design-point metrics.
    """
    consts = load_constants() if constants is None else constants
    optics = CylinderOpticsModel.from_config(variant)
    cantilever = CantileverModel.from_config(consts, variant)
    s_target = target_sensitivity(variant, consts)
    abs_s = abs(s_target)
    eta0 = optics.eta_working_point()
    eta_peak = optics.eta_peak()
    gain = variant.responsivity_a_w * variant.source.power_w
    i_dc = gain * (variant.endface_reflectivity + variant.reflector.reflectivity * eta0)
    psd = noise_psd(
        i_dc,
        variant,
        consts,
        balanced=variant.detector.balanced,
        reference_arm=variant.detector.reference_arm,
    )
    bandwidth = variant.band.f_max_hz - variant.band.f_min_hz
    fs_signal = abs_s * variant.full_scale_g * G0  # |I_AC| at full scale, A
    noise_full_band = math.sqrt(psd["total"] * bandwidth)
    if abs_s > 0.0:
        nea_plateau = math.sqrt(psd["total"]) / abs_s
        modulation_at_fs = fs_signal / i_dc
        dynamic_range_db = 20.0 * math.log10(fs_signal / noise_full_band)
    else:
        # Degenerate working point exactly at the eta peak (slope = 0): the
        # sensitivity vanishes and the input-referred NEA diverges. Report inf so
        # the percentile summaries (which drop non-finite draws) stay well-defined.
        nea_plateau = math.inf
        modulation_at_fs = 0.0
        dynamic_range_db = -math.inf
    nea_full_band = nea_plateau * math.sqrt(bandwidth)
    return AnalyticPoint(
        s_target=s_target,
        eta0=eta0,
        eta_peak=eta_peak,
        eta_ratio=eta0 / eta_peak,
        f1_hz=cantilever.f1_hz,
        i_dc_a=i_dc,
        nea_plateau=nea_plateau,
        nea_plateau_ug=nea_plateau / G0 * 1.0e6,
        nea_full_band=nea_full_band,
        bandwidth_hz=bandwidth,
        modulation_at_fs=modulation_at_fs,
        dynamic_range_db=dynamic_range_db,
    )
