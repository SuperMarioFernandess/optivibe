"""Measured parameters -> digital-twin configuration (task S-13; O-SW-14).

Takes the :class:`~optivibe.io.characterization.CharacterizationResult` records
produced by the input layer and applies them to a composed
:class:`~optivibe.core.config.subsystems.SystemConfig` (typically the
``proto_poc`` twin, doc 16 §3a) as subsystem *overrides* -- the same mechanism
the GUI composition editor uses, so nothing new enters the resolve path and the
A-D contract is untouched by construction (user compositions only).

Rules encoded here (they are physics discipline, not IO plumbing):

Anti-double-count (R-57(v), R-54 lineage)
    A measured quantity **replaces** its modelled counterpart, it is never
    added to it. A measured RIN trace already contains the ASE floor and the
    driver excess, so it *replaces* the derived floor; a measured spectrum is
    the single source of truth of its row, so next to it the scalar
    ``linewidth_fwhm_m`` is forced to ``None`` (R-57(a)) and, absent a measured
    RIN, the nameplate ``rin_db_hz`` of the preset is forced to ``None`` so the
    floor derives from the *measured table* (``RIN = 2 tau_c``) rather than
    from a datasheet number.
Override semantics for computed fields (M-02/M-18)
    A measured ``q_total`` is written as the explicit system-level value --
    exactly the documented override of the computable ``Q(L)`` model; the
    model's own prediction is recorded next to it in the report (the L3
    cross-check of doc 19).
GUM gate (17 §4)
    Parameters with ``u = None`` are informational and are **refused** as
    config inputs; they appear in the report only.
No config slot
    ``f1_hz`` (validation metric, P20-1) and ``dop`` (R-58: no DOP parameter
    in the model) are accepted as measurements but deliberately have no
    configuration target -- the report marks them as such.

Provenance is persisted as a **separate artifact** next to the produced
composition (:func:`save_provenance`), never inside the variant file: this is
the doc 16 §2a contract decision keeping the bit-compared A-D golden
independent of measurement files.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from optivibe.core.config.models import Constants
from optivibe.core.config.subsystems import SystemConfig
from optivibe.core.logging import get_logger
from optivibe.io.characterization import CharacterizationResult, MeasuredParameter

logger = get_logger(__name__)

__all__ = [
    "PARAMETER_TARGETS",
    "FieldChange",
    "IngestReport",
    "apply_measurements",
    "save_provenance",
]

#: Measured-parameter name -> (config block, field). Blocks are the subsystem
#: override dicts of :class:`SystemConfig`; ``"system"`` is the composition
#: scalar level. Names match the config fields one-to-one (doc 04 ICD), so no
#: renaming layer exists to drift.
PARAMETER_TARGETS: dict[str, tuple[str, str]] = {
    "length_m": ("cantilever", "length_m"),
    "gap_m": ("reflector", "gap_m"),
    "bias_offset_m": ("reflector", "bias_offset_m"),
    "curvature_radius_m": ("reflector", "curvature_radius_m"),
    "metallization_rho": ("reflector", "metallization_rho"),
    "wavelength_m": ("source", "wavelength_m"),
    "power_w": ("source", "power_w"),
    "rin_db_hz": ("source", "rin_db_hz"),
    "linewidth_fwhm_m": ("source", "linewidth_fwhm_m"),
    "mode_field_radius_m": ("fiber", "mode_field_radius_m"),
    "fresnel_R1": ("fiber", "fresnel_R1"),
    "responsivity": ("detector", "responsivity"),
    "cmrr_db": ("detector", "cmrr_db"),
    "q_total": ("system", "q_total"),
}

_SUBSYSTEM_BLOCKS = ("source", "fiber", "cantilever", "reflector", "detector")


@dataclass(frozen=True)
class FieldChange:
    """One configuration field replaced by a measurement.

    Attributes
    ----------
    target : str
        Dotted config location, e.g. ``"cantilever.length_m"``.
    old : object
        The value the base composition would have used (preset + override view;
        ``None`` when the base leaves it to a derivation).
    new : object
        The measured value written (SI).
    u : float
        Standard uncertainty of the measured value (SI; GUM 17 §4).
    provenance_index : int
        Index into :attr:`IngestReport.provenances` of the source artifact.
    note : str
        Rule note (override semantics, anti-double-count, ...).
    """

    target: str
    old: object
    new: object
    u: float
    provenance_index: int
    note: str = ""


@dataclass(frozen=True)
class IngestReport:
    """Result of applying measurements to a composition.

    Attributes
    ----------
    system : SystemConfig
        The updated composition (validated; the caller resolves it to check the
        composition-time guards, e.g. the route-2 wash-out).
    base_name : str
        Name of the base composition the measurements were applied to.
    changes : tuple of FieldChange
        Config fields now backed by measurements.
    informational : tuple of MeasuredParameter
        Measured values that were *not* written: report-only derivations
        (``linewidth_eff_hz``, ``rin_db_hz_floor``), parameters without a
        config slot (``f1_hz``, ``dop``) and parameters without ``u``.
    model_defaults : tuple of str
        Config-bound targets that remain model/default-valued (the explicit
        "what is still NOT measured" list of doc 16 §2a).
    provenances : tuple
        Provenance records of all consumed artifacts, in input order.
    q_total_model : float or None
        The ``Q(L)`` model prediction for the *measured* geometry, recorded
        next to a measured ``q_total`` for the L3 cross-check (M-18); ``None``
        when no measured Q was applied or constants were not supplied.
    """

    system: SystemConfig
    base_name: str
    changes: tuple[FieldChange, ...]
    informational: tuple[MeasuredParameter, ...]
    model_defaults: tuple[str, ...]
    provenances: tuple[Any, ...]
    q_total_model: float | None = None

    def summary_text(self) -> str:
        """Render a human-readable ingest report."""
        lines = [f"base composition : {self.base_name}"]
        lines.append("measured -> config:")
        if not self.changes:
            lines.append("  (none)")
        for change in self.changes:
            prov = self.provenances[change.provenance_index]
            note = f"  [{change.note}]" if change.note else ""
            lines.append(
                f"  {change.target:32s} {_fmt(change.old):>12s} -> {_fmt(change.new):>12s}"
                f"  (u = {change.u:.3g}; {prov.sidecar}){note}"
            )
        if self.q_total_model is not None:
            lines.append(
                f"  Q(L) model at the measured geometry: {self.q_total_model:.6g} "
                "(cross-check against the measured q_total; M-18)"
            )
        lines.append("informational (not written):")
        if not self.informational:
            lines.append("  (none)")
        for param in self.informational:
            u_text = f"u = {param.u:.3g}" if param.u is not None else "u not evaluated"
            lines.append(f"  {param.name:32s} = {param.value:.6g}  ({u_text}; {param.method})")
        lines.append("still model/default (not measured):")
        lines.append("  " + (", ".join(self.model_defaults) if self.model_defaults else "(none)"))
        return "\n".join(lines)


def _fmt(value: object) -> str:
    """Compact numeric formatting for the report."""
    if value is None:
        return "<derived>"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


@dataclass
class _Pending:
    """Internal accumulator of one config-bound write."""

    param: MeasuredParameter
    provenance_index: int
    note: str = ""


def apply_measurements(
    system: SystemConfig,
    results: list[CharacterizationResult] | tuple[CharacterizationResult, ...],
    *,
    constants: Constants | None = None,
) -> IngestReport:
    """Apply characterization results to a composition as overrides.

    Parameters
    ----------
    system : SystemConfig
        Base composition (e.g. the loaded ``proto_poc``).
    results : sequence of CharacterizationResult
        Artifacts from :func:`optivibe.io.characterization.load_characterization`.
    constants : Constants or None, optional
        Physical constants; when given and a measured ``q_total`` is applied,
        the ``Q(L)`` model value at the measured geometry is recorded in the
        report for the M-18 cross-check.

    Returns
    -------
    IngestReport
        Updated composition + the measured/informational/still-default split.

    Raises
    ------
    ValueError
        On two measurements of the same config field (an explicit conflict:
        the operator must pick, e.g. two profile azimuths, M-17/M-03), or on a
        scalar linewidth measured next to a measured spectrum (R-57(a)).
    """
    pending: dict[str, _Pending] = {}
    informational: list[MeasuredParameter] = []
    provenances = [result.provenance for result in results]
    spectrum: tuple[int, CharacterizationResult] | None = None

    for index, result in enumerate(results):
        if result.spectrum_wavelength_m is not None:
            if spectrum is not None:
                msg = (
                    "two measured spectra supplied; the table is the single source of "
                    "truth of its row (R-57) -- pick one artifact"
                )
                raise ValueError(msg)
            spectrum = (index, result)
        for param in result.params:
            if param.name not in PARAMETER_TARGETS:
                informational.append(param)
                continue
            if param.u is None:
                logger.warning(
                    "%s has no uncertainty; refusing it as a config input (GUM 17 §4)",
                    param.name,
                )
                informational.append(param)
                continue
            if param.name in pending:
                msg = (
                    f"parameter {param.name!r} measured twice "
                    f"({provenances[pending[param.name].provenance_index].sidecar} and "
                    f"{provenances[index].sidecar}); two measurements of one field is an "
                    "explicit conflict -- pick one (for two profile azimuths the "
                    "toroidal combination is backlog M-03/M-17)"
                )
                raise ValueError(msg)
            pending[param.name] = _Pending(param=param, provenance_index=index)

    if spectrum is not None and "linewidth_fwhm_m" in pending:
        msg = (
            "a scalar linewidth measurement cannot be applied next to a measured "
            "spectrum: with lineshape='measured' the table is the single source of "
            "truth (R-57(a)) -- drop one artifact"
        )
        raise ValueError(msg)

    payload = system.model_dump(mode="python")
    changes: list[FieldChange] = []

    for name, entry in pending.items():
        block, fld = PARAMETER_TARGETS[name]
        note = entry.note
        if name == "q_total":
            note = "override of the computable Q(L) model (M-02 semantics; measured wins)"
        if name == "rin_db_hz":
            note = "replaces the derived ASE floor (anti-double-count R-57(v))"
        old = _current_value(payload, block, fld)
        _write(payload, block, fld, entry.param.value)
        changes.append(
            FieldChange(
                target=f"{block}.{fld}" if block != "system" else fld,
                old=old,
                new=entry.param.value,
                u=float(entry.param.u or 0.0),
                provenance_index=entry.provenance_index,
                note=note,
            )
        )

    if spectrum is not None:
        index, result = spectrum
        overrides = payload["source"]["overrides"]
        old_shape = _current_value(payload, "source", "lineshape")
        overrides["lineshape"] = "measured"
        overrides["spectrum_wavelength_m"] = list(result.spectrum_wavelength_m or ())
        overrides["spectrum_psd"] = list(result.spectrum_psd or ())
        # R-57(a): the table is the single source of truth -- kill any preset
        # scalar linewidth explicitly.
        overrides["linewidth_fwhm_m"] = None
        wavelength = result.get("wavelength_m")
        u_spectrum = float(wavelength.u) if wavelength is not None and wavelength.u else 0.0
        changes.append(
            FieldChange(
                target="source.lineshape",
                old=old_shape,
                new="measured",
                u=u_spectrum,
                provenance_index=index,
                note="measured spectrum table attached; scalar linewidth_fwhm_m "
                "forced to None (R-57(a))",
            )
        )
        if "rin_db_hz" not in pending:
            # The nameplate/preset RIN (if any) must not silently outrank the
            # measured table: an explicit rin_db_hz has priority at resolve
            # time (R-57(v)), so it is force-dropped here and the ASE floor
            # derives from the measured spectrum (RIN = 2 tau_c). A *measured*
            # RIN trace, when supplied, is written instead and wins.
            old_rin = _current_value(payload, "source", "rin_db_hz")
            overrides["rin_db_hz"] = None
            changes.append(
                FieldChange(
                    target="source.rin_db_hz",
                    old=old_rin if old_rin is not None else "<preset value>",
                    new=None,
                    u=0.0,
                    provenance_index=index,
                    note="nameplate RIN dropped: the floor now derives from the "
                    "measured table (RIN = 2 tau_c; R-57); a measured RIN trace, "
                    "when supplied, replaces it instead (R-57(v))",
                )
            )

    updated = SystemConfig.model_validate(payload)

    q_model: float | None = None
    if "q_total" in pending and constants is not None:
        from optivibe.mechanics.damping import q_total_model

        length = _current_value(payload, "cantilever", "length_m")
        if isinstance(length, (int, float)):
            q_model = q_total_model(constants, float(length), vacuum=updated.vacuum)

    measured_targets = {change.target for change in changes}
    if spectrum is not None:
        # With lineshape='measured' the scalar linewidth row is superseded by
        # the table (R-57(a)): listing it as "still model/default" would be
        # misleading.
        measured_targets.add("source.linewidth_fwhm_m")
    model_defaults = tuple(
        (f"{block}.{fld}" if block != "system" else fld)
        for name, (block, fld) in sorted(PARAMETER_TARGETS.items())
        if (f"{block}.{fld}" if block != "system" else fld) not in measured_targets
    )
    return IngestReport(
        system=updated,
        base_name=system.name,
        changes=tuple(changes),
        informational=tuple(informational),
        model_defaults=model_defaults,
        provenances=tuple(provenances),
        q_total_model=q_model,
    )


def _current_value(payload: dict[str, Any], block: str, fld: str) -> object:
    """Read the effective base value of ``block.fld`` (override view only).

    For subsystem blocks the *preset* body is not loaded here (the config layer
    resolves presets); the report therefore shows the override if present and
    ``<derived>``/``None`` otherwise -- honest about what this layer knows.
    """
    if block == "system":
        return payload.get(fld)
    return payload[block]["overrides"].get(fld)


def _write(payload: dict[str, Any], block: str, fld: str, value: object) -> None:
    """Write ``value`` into the composition payload as an override."""
    if block == "system":
        payload[fld] = value
        return
    if block not in _SUBSYSTEM_BLOCKS:  # pragma: no cover - table integrity
        msg = f"unknown config block {block!r}"
        raise ValueError(msg)
    payload[block]["overrides"][fld] = value


def save_provenance(report: IngestReport, composition_path: Path) -> Path:
    """Persist the measurement provenance next to the produced composition.

    A **separate artifact** by design (doc 16 §2a): the variant YAML stays a
    pure :class:`SystemConfig` document (round-trippable, preset-compatible)
    and the bit-compared A-D golden can never grow a dependency on measurement
    files.

    Parameters
    ----------
    report : IngestReport
        The ingest result.
    composition_path : pathlib.Path
        Path of the composition YAML that was (or will be) written; the
        provenance lands at ``<same stem>.provenance.yaml``.

    Returns
    -------
    pathlib.Path
        The provenance file written.
    """
    path = composition_path.with_suffix(".provenance.yaml")
    body = {
        "composition": report.system.name,
        "base": report.base_name,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "measured_fields": [
            {
                "target": change.target,
                "value": change.new,
                "u": change.u,
                "note": change.note,
                "kind": report.provenances[change.provenance_index].kind,
                "sidecar": report.provenances[change.provenance_index].sidecar,
                "data_file": report.provenances[change.provenance_index].data_file,
                "sha256": report.provenances[change.provenance_index].sha256,
                "instrument": report.provenances[change.provenance_index].instrument,
                "timestamp": report.provenances[change.provenance_index].timestamp,
            }
            for change in report.changes
        ],
        "informational": [
            {"name": p.name, "value": p.value, "u": p.u, "method": p.method}
            for p in report.informational
        ],
        "model_or_default_fields": list(report.model_defaults),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(body, handle, sort_keys=False, default_flow_style=False)
    logger.info("provenance written to %s", path)
    return path
