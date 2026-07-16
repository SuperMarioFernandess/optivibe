"""Characterization-artifact input layer (task S-13; O-SW-14; doc 16 §2a).

Phase 0 of the bench plan (doc 20 §1) produces *measured* system parameters --
geometry (L, A), the reflector (R_c, rho), the source (spectrum, RIN), the
mechanics (ring-down Q, f1) -- as instrument files. This module turns one such
artifact (a CSV data file plus a mandatory YAML *sidecar* declaring its units,
instrument and uncertainties) into the standard **measured-parameter contract**:

``MeasuredParameter``
    value + uncertainty ``u`` + method, in SI -- the record that feeds the GUM
    uncertainty budget (doc 17 §4, doc 19 §4).
``Provenance``
    what file, when, which instrument, sha256 -- travels with the parameters
    but never enters the bit-compared A-D contract (it is written as a
    *separate* artifact next to user compositions; doc 16 §2a).

Readers are registered in :data:`CHARACTERIZATION_REGISTRY` by *kind* -- the
same pattern as :data:`~optivibe.io.records.RECORD_REGISTRY` (SW-52) and
:data:`~optivibe.io.loaders.LOADER_REGISTRY` (SW-08): adding an artifact kind
or a file format is a new registration, never a core change. Each reader
implements ``load -> validate -> reduce -> (params, u, provenance)``.

Kinds (backlog cards, doc 16 §1/§2):

``scalar``
    A hand-recorded single value (L, A, rho, CMRR, DOP, ...); the sidecar *is*
    the measurement, no data file.
``spectrum`` (M-15)
    An OSA trace ``(lambda, S_lambda)``. Reduced to the centroid ``lambda0``,
    the FWHM ``dlam``, the noise-equivalent width ``dnu_eff`` and the ASE
    floor ``RIN = 2 tau_c`` through the *existing* physics of
    :mod:`optivibe.optics.source` (R-57) -- no metric is re-implemented here.
    The table itself is carried along for ``lineshape = "measured"``.
``rin_psd`` (M-16, reduce level)
    A RIN(f) trace from the PD+TIA bench (doc 20 F0-7/F0-8), *already*
    floor-corrected (shot/dark/Johnson subtracted -- the subtraction itself is
    the M-16 remainder). Reduced to a scalar ``rin_db_hz`` over a declared
    band; per the anti-double-count rule R-57(v) this value **replaces** the
    derived ASE floor, it is never added to it.
``ringdown`` (M-18)
    A free-decay record ``(t, y)``. Reduced to ``f1`` (FFT peak) and ``Q``
    from the logarithmic decrement: for
    ``y = A exp(-pi f1 t / Q) cos(2 pi f1 t)`` the envelope decays at
    ``sigma = pi f1 / Q``, so ``Q = pi f1 / sigma`` (dimensions: [sigma] =
    1/s, [f1] = Hz => Q dimensionless; per-period decrement
    ``delta = sigma / f1 = pi / Q``). Limits: an undamped tone gives
    ``sigma -> 0`` and is rejected loudly (no significant decay).
``profile`` (M-17, single azimuth)
    A tip-contour trace ``(x, z)``. Reduced to the curvature radius ``R_c``
    by an algebraic (Kasa) circle fit with a covariance-propagated ``u(R_c)``.
    Astigmatism needs two azimuths and the toroidal optics M-03 -- one file =
    one azimuth here; combining them stays in the backlog.

Uncertainty discipline (doc 17 §4): a parameter without ``u`` cannot feed the
budget, so ``u`` is either *computed* by the reducer (fit covariance) or
*declared* in the sidecar; parameters that end up with ``u = None`` are
**informational only** -- the ingest layer (:mod:`optivibe.io.ingest`) refuses
to write them into the twin configuration.

Units are **mandatory and declared** in the sidecar, never guessed (doc 16
§2a; the nm-vs-m trap of R-57(b)).
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.signal import hilbert

from optivibe.core.logging import get_logger
from optivibe.core.registry import Registry
from optivibe.core.types import FloatArray
from optivibe.optics.source import (
    effective_linewidth_measured_hz,
    rin_ase_measured_db_hz,
)

logger = get_logger(__name__)

__all__ = [
    "CHARACTERIZATION_REGISTRY",
    "CharacterizationReader",
    "CharacterizationResult",
    "CharacterizationSpec",
    "MeasuredParameter",
    "ProfileReader",
    "Provenance",
    "RinPsdReader",
    "RingdownReader",
    "ScalarReader",
    "SpectrumReader",
    "load_characterization",
    "resolve_sidecar_path",
]

# --------------------------------------------------------------------------- #
# Unit tables (SI conversion factors). Declared, never guessed (doc 16 §2a).
# --------------------------------------------------------------------------- #
_LENGTH_UNITS: dict[str, float] = {"m": 1.0, "mm": 1.0e-3, "um": 1.0e-6, "nm": 1.0e-9}
_TIME_UNITS: dict[str, float] = {"s": 1.0, "ms": 1.0e-3, "us": 1.0e-6}
_FREQ_UNITS: dict[str, float] = {"hz": 1.0, "khz": 1.0e3, "mhz": 1.0e6}
_POWER_UNITS: dict[str, float] = {"w": 1.0, "mw": 1.0e-3, "uw": 1.0e-6}
_DIMENSIONLESS: dict[str, float] = {"-": 1.0, "1": 1.0}
_IDENTITY_DB: dict[str, float] = {"db": 1.0}
_IDENTITY_DB_HZ: dict[str, float] = {"db/hz": 1.0, "db_hz": 1.0}
_ARBITRARY: dict[str, float] = {"arb": 1.0}
_RESPONSIVITY_UNITS: dict[str, float] = {"a/w": 1.0, "a_w": 1.0}

#: Scalar parameters accepted by the ``scalar`` kind, with their unit family.
#: Names are the *configuration field names* (doc 04 ICD / config models), so
#: the ingest mapping is one-to-one; ``dop`` and ``f1_hz`` are measured but
#: deliberately have no config slot (R-58: no DOP parameter in the model;
#: P20-1: f1 is a validation metric, not an input).
SCALAR_UNIT_FAMILIES: dict[str, dict[str, float]] = {
    "length_m": _LENGTH_UNITS,
    "gap_m": _LENGTH_UNITS,
    "bias_offset_m": _LENGTH_UNITS,
    "curvature_radius_m": _LENGTH_UNITS,
    "wavelength_m": _LENGTH_UNITS,
    "linewidth_fwhm_m": _LENGTH_UNITS,
    "mode_field_radius_m": _LENGTH_UNITS,
    "power_w": _POWER_UNITS,
    "rin_db_hz": _IDENTITY_DB_HZ,
    "cmrr_db": _IDENTITY_DB,
    "responsivity": _RESPONSIVITY_UNITS,
    "metallization_rho": _DIMENSIONLESS,
    "fresnel_R1": _DIMENSIONLESS,
    "q_total": _DIMENSIONLESS,
    "dop": _DIMENSIONLESS,
    "f1_hz": _FREQ_UNITS,
}


def _to_si(value: float, unit: str, family: dict[str, float], *, what: str) -> float:
    """Convert ``value`` in ``unit`` to SI using ``family``; fail loudly.

    Parameters
    ----------
    value : float
        Value in the declared unit.
    unit : str
        Declared unit (case-insensitive).
    family : dict
        Allowed units and their SI factors.
    what : str
        Human-readable name for the error message.

    Returns
    -------
    float
        Value in SI.

    Raises
    ------
    ValueError
        If the unit is not in the family (units are declared, never guessed).
    """
    key = unit.strip().lower()
    if key not in family:
        allowed = ", ".join(sorted(family))
        msg = f"unknown unit {unit!r} for {what}; declared units must be one of: {allowed}"
        raise ValueError(msg)
    return value * family[key]


# --------------------------------------------------------------------------- #
# The measured-parameter contract (doc 17 §4; doc 19 §4).
# --------------------------------------------------------------------------- #
class _Frozen(BaseModel):
    """Frozen strict base (10 §7): unknown keys and mutation fail loudly."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class MeasuredParameter(_Frozen):
    """One measured parameter in SI: value + uncertainty + method (GUM, 17 §4).

    Attributes
    ----------
    name : str
        Parameter name -- a configuration field name (``length_m``,
        ``rin_db_hz``, ...) or a named informational quantity (``f1_hz``,
        ``dop``, ``linewidth_eff_hz``).
    value : float
        Value in the SI unit implied by the name suffix (``_m``, ``_hz``,
        ``_w``; dB-named fields stay in dB).
    u : float or None
        Standard uncertainty ``u`` in the same unit, ``>= 0``. ``None`` means
        *not evaluated*: the parameter is informational and MUST NOT feed the
        twin configuration or the uncertainty budget (doc 17 §4).
    method : str
        Free-text method note ("circle fit, 34 points", "sidecar declaration").
    """

    name: str = Field(min_length=1)
    value: float
    u: float | None = Field(default=None, ge=0.0)
    method: str = ""


class Provenance(_Frozen):
    """Where a measurement came from (doc 16 §2a).

    Kept *outside* the bit-compared A-D contract by construction: provenance is
    persisted as a separate artifact next to user compositions
    (:func:`optivibe.io.ingest.save_provenance`), never inside a variant file.

    Attributes
    ----------
    kind : str
        Artifact kind (``scalar`` / ``spectrum`` / ``rin_psd`` / ``ringdown``
        / ``profile``).
    sidecar : str
        Sidecar file name.
    data_file : str or None
        Data file name (``None`` for the sidecar-only ``scalar`` kind).
    sha256 : str or None
        SHA-256 of the data file (``None`` when there is no data file).
    instrument : str
        Instrument description from the sidecar.
    timestamp : str
        ISO-8601 measurement timestamp from the sidecar.
    """

    kind: str
    sidecar: str
    data_file: str | None = None
    sha256: str | None = None
    instrument: str
    timestamp: str


@dataclass(frozen=True)
class CharacterizationResult:
    """Output of one artifact: parameters + provenance (+ the spectrum table).

    Attributes
    ----------
    params : tuple of MeasuredParameter
        Reduced parameters in SI. Parameters with ``u = None`` are
        informational only (see :class:`MeasuredParameter`).
    provenance : Provenance
        Origin record for the whole artifact.
    spectrum_wavelength_m, spectrum_psd : tuple of float or None
        The measured source spectrum (``kind = "spectrum"`` only), in SI, ready
        for ``SourceConfig(lineshape="measured", ...)`` -- the table is the
        single source of truth of its row (R-57), so it travels whole rather
        than being collapsed to a scalar.
    """

    params: tuple[MeasuredParameter, ...]
    provenance: Provenance
    spectrum_wavelength_m: tuple[float, ...] | None = None
    spectrum_psd: tuple[float, ...] | None = None

    def get(self, name: str) -> MeasuredParameter | None:
        """Return the parameter called ``name``, or ``None``."""
        for param in self.params:
            if param.name == name:
                return param
        return None


# --------------------------------------------------------------------------- #
# Sidecar spec.
# --------------------------------------------------------------------------- #
class ColumnSpec(_Frozen):
    """One CSV column declaration: physical name + mandatory unit."""

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)


class BandSpec(_Frozen):
    """Frequency band ``[f_min, f_max]`` for the ``rin_psd`` reduction, Hz."""

    f_min_hz: float = Field(gt=0.0)
    f_max_hz: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_order(self) -> BandSpec:
        if self.f_max_hz <= self.f_min_hz:
            msg = f"f_max_hz ({self.f_max_hz}) must exceed f_min_hz ({self.f_min_hz})"
            raise ValueError(msg)
        return self


class CharacterizationSpec(_Frozen):
    """Validated sidecar of one characterization artifact (doc 16 §2a).

    Attributes
    ----------
    kind : str
        Artifact kind; selects the reader in
        :data:`CHARACTERIZATION_REGISTRY`.
    instrument : str
        Instrument description (mandatory provenance, doc 16 §2a).
    timestamp : str
        ISO-8601 measurement timestamp (mandatory provenance).
    method : str
        Optional free-text method note.
    data_file : str or None
        Data file, relative to the sidecar (all kinds except ``scalar``).
    columns : dict of ColumnSpec or None
        ``{"x": ..., "y": ...}`` column declarations with mandatory units
        (XY kinds only).
    uncertainties : dict of float
        Declared standard uncertainties, keyed by SI parameter name, values in
        that parameter's SI unit (e.g. ``wavelength_m: 0.5e-9``). Declared
        values override computed ones (the operator knows the instrument).
    parameter, value, unit, u : scalar-kind fields
        The hand-recorded measurement itself: parameter name (a key of
        :data:`SCALAR_UNIT_FAMILIES`), value and uncertainty in the declared
        unit.
    band : BandSpec or None
        Reduction band for ``rin_psd``.
    """

    kind: Literal["scalar", "spectrum", "rin_psd", "ringdown", "profile"]
    instrument: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    method: str = ""
    data_file: str | None = None
    columns: dict[str, ColumnSpec] | None = None
    uncertainties: dict[str, float] = Field(default_factory=dict)
    parameter: str | None = None
    value: float | None = None
    unit: str | None = None
    u: float | None = Field(default=None, ge=0.0)
    band: BandSpec | None = None

    @model_validator(mode="after")
    def _check_kind_fields(self) -> CharacterizationSpec:
        """Require exactly the fields the kind consumes (10 §7: loud errors)."""
        if self.kind == "scalar":
            if self.parameter is None or self.value is None or self.unit is None:
                msg = "kind='scalar' requires parameter, value and unit in the sidecar"
                raise ValueError(msg)
            if self.u is None:
                msg = (
                    "kind='scalar' requires the standard uncertainty u (same unit as "
                    "value): a parameter without u cannot feed the GUM budget (17 §4)"
                )
                raise ValueError(msg)
            if self.data_file is not None:
                msg = "kind='scalar' is sidecar-only; drop data_file"
                raise ValueError(msg)
        else:
            if self.data_file is None:
                msg = f"kind={self.kind!r} requires data_file (the CSV trace)"
                raise ValueError(msg)
            if self.columns is None or set(self.columns) != {"x", "y"}:
                msg = f"kind={self.kind!r} requires columns: {{x: ..., y: ...}} with units"
                raise ValueError(msg)
        if self.kind == "rin_psd" and self.band is None:
            msg = "kind='rin_psd' requires band: {f_min_hz, f_max_hz} for the reduction"
            raise ValueError(msg)
        for key, value in self.uncertainties.items():
            if value < 0.0:
                msg = f"uncertainties[{key!r}] must be >= 0, got {value!r}"
                raise ValueError(msg)
        return self


# --------------------------------------------------------------------------- #
# CSV reading (two numeric columns; comments/headers skipped loudly-but-once).
# --------------------------------------------------------------------------- #
def _read_xy_csv(path: Path, *, min_rows: int) -> tuple[FloatArray, FloatArray]:
    """Read a two-column numeric CSV trace.

    Accepts ``,``, ``;`` or whitespace separation; skips ``#`` comments and a
    non-numeric header line. Fails loudly on fewer than ``min_rows`` numeric
    rows or on rows with fewer than two numeric fields.

    Parameters
    ----------
    path : pathlib.Path
        CSV file.
    min_rows : int
        Minimum number of numeric rows required.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)`` as float64 arrays, in file units.
    """
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for lineno, raw in enumerate(csv.reader(_normalized_lines(handle)), start=1):
            if not raw:
                continue
            try:
                xs.append(float(raw[0]))
                ys.append(float(raw[1]))
            except (ValueError, IndexError):
                if not xs:  # tolerate a single leading header line
                    continue
                msg = f"{path}: line {lineno} is not a two-column numeric row: {raw!r}"
                raise ValueError(msg) from None
    if len(xs) < min_rows:
        msg = f"{path}: expected at least {min_rows} numeric rows, got {len(xs)}"
        raise ValueError(msg)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _normalized_lines(handle: Any) -> list[str]:
    """Normalise separators so ``csv.reader`` sees comma-separated fields."""
    lines: list[str] = []
    for raw in handle:
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        text = text.replace(";", ",").replace("\t", ",")
        if "," not in text:
            text = ",".join(text.split())
        lines.append(text)
    return lines


def _declared_u(
    spec: CharacterizationSpec, name: str, computed: float | None = None
) -> float | None:
    """Resolve a parameter's uncertainty: sidecar declaration wins over the fit."""
    if name in spec.uncertainties:
        return spec.uncertainties[name]
    return computed


def _provenance(spec: CharacterizationSpec, sidecar: Path, data: Path | None) -> Provenance:
    """Assemble the provenance record (sha256 of the data file when present)."""
    sha = hashlib.sha256(data.read_bytes()).hexdigest() if data is not None else None
    return Provenance(
        kind=spec.kind,
        sidecar=sidecar.name,
        data_file=data.name if data is not None else None,
        sha256=sha,
        instrument=spec.instrument,
        timestamp=spec.timestamp,
    )


# --------------------------------------------------------------------------- #
# Reader protocol + registry (pattern of SW-52 / SW-08).
# --------------------------------------------------------------------------- #
@runtime_checkable
class CharacterizationReader(Protocol):
    """Loads + reduces one characterization artifact kind."""

    def load(self, spec: CharacterizationSpec, sidecar_path: Path) -> CharacterizationResult:
        """Load the artifact described by ``spec`` (sidecar at ``sidecar_path``).

        Parameters
        ----------
        spec : CharacterizationSpec
            Validated sidecar.
        sidecar_path : pathlib.Path
            Sidecar location; ``spec.data_file`` is resolved relative to it.

        Returns
        -------
        CharacterizationResult
            Reduced parameters + provenance.
        """
        ...


CHARACTERIZATION_REGISTRY: Registry[CharacterizationReader] = Registry("io.characterization")


def _data_path(spec: CharacterizationSpec, sidecar_path: Path) -> Path:
    """Resolve the data file relative to the sidecar; fail loudly if absent."""
    assert spec.data_file is not None  # guaranteed by the sidecar validator
    path = (sidecar_path.parent / spec.data_file).resolve()
    if not path.is_file():
        msg = f"data file {spec.data_file!r} (from {sidecar_path.name}) not found at {path}"
        raise FileNotFoundError(msg)
    return path


@CHARACTERIZATION_REGISTRY.register("scalar")
class ScalarReader:
    """A hand-recorded scalar: the sidecar is the measurement (doc 20 F0-*)."""

    def load(self, spec: CharacterizationSpec, sidecar_path: Path) -> CharacterizationResult:
        """Convert the declared value + u to SI and emit one parameter."""
        assert spec.parameter is not None and spec.value is not None  # sidecar validator
        assert spec.unit is not None and spec.u is not None
        if spec.parameter not in SCALAR_UNIT_FAMILIES:
            allowed = ", ".join(sorted(SCALAR_UNIT_FAMILIES))
            msg = f"unknown scalar parameter {spec.parameter!r}; known: {allowed}"
            raise ValueError(msg)
        family = SCALAR_UNIT_FAMILIES[spec.parameter]
        value = _to_si(spec.value, spec.unit, family, what=spec.parameter)
        u = _to_si(spec.u, spec.unit, family, what=f"u({spec.parameter})")
        param = MeasuredParameter(
            name=spec.parameter,
            value=value,
            u=u,
            method=spec.method or "sidecar declaration",
        )
        return CharacterizationResult(
            params=(param,), provenance=_provenance(spec, sidecar_path, None)
        )


@CHARACTERIZATION_REGISTRY.register("spectrum")
class SpectrumReader:
    """OSA trace ``(lambda, S_lambda)`` -> lambda0, dlam, dnu_eff, RIN (M-15).

    All physics is delegated to :mod:`optivibe.optics.source` (R-57): the
    noise-equivalent width and the ASE floor are the *same* quadratures the
    resolve-time ``lineshape = "measured"`` path evaluates -- no metric is
    re-implemented. The centroid and FWHM are cheap lambda-domain reductions
    for the report and for ``wavelength_m``.

    Emitted parameters: ``wavelength_m`` (centroid; config-bound, needs a
    declared ``u``), ``spectrum_fwhm_m`` / ``linewidth_eff_hz`` /
    ``rin_db_hz_floor`` (informational: with ``lineshape = "measured"`` the
    table is the single source of truth of its row (R-57a), so the scalar
    linewidth must NOT be written to the config, and the derived floor must
    not be double-counted next to a measured RIN trace, R-57v).
    """

    #: Sanity window for the centroid: catches the nm-vs-m unit slip (R-57(b))
    #: before the table ever reaches a composition.
    _LAMBDA_SANITY_M = (0.1e-6, 20.0e-6)

    def load(self, spec: CharacterizationSpec, sidecar_path: Path) -> CharacterizationResult:
        """Load the trace, convert to SI, reduce through optics.source."""
        assert spec.columns is not None  # sidecar validator
        data = _data_path(spec, sidecar_path)
        lam_raw, psd = _read_xy_csv(data, min_rows=4)
        lam = np.asarray(
            [_to_si(v, spec.columns["x"].unit, _LENGTH_UNITS, what="wavelength") for v in lam_raw]
        )
        _to_si(1.0, spec.columns["y"].unit, _ARBITRARY, what="spectral density")
        order = np.argsort(lam)
        lam = lam[order]
        psd = psd[order]

        lam_list = lam.tolist()
        psd_list = psd.tolist()
        # Reduction through the existing physics (validates the table itself).
        dnu_eff = effective_linewidth_measured_hz(lam_list, psd_list)
        rin_db = rin_ase_measured_db_hz(lam_list, psd_list)
        centroid = float(np.trapezoid(lam * psd, lam) / np.trapezoid(psd, lam))
        lo, hi = self._LAMBDA_SANITY_M
        if not lo <= centroid <= hi:
            msg = (
                f"spectrum centroid {centroid:.3e} m is outside the sanity window "
                f"[{lo:.1e}, {hi:.1e}] m -- check the declared wavelength unit "
                "(OSA exports are often in nm; R-57(b))"
            )
            raise ValueError(msg)
        fwhm = _fwhm_lambda_m(lam, psd)

        u_lambda = _declared_u(spec, "wavelength_m")
        if u_lambda is None:
            msg = (
                "spectrum sidecar must declare uncertainties: {wavelength_m: ...} "
                "(SI metres): the centroid feeds the twin configuration and a "
                "config-bound parameter needs u (17 §4)"
            )
            raise ValueError(msg)
        params = (
            MeasuredParameter(
                name="wavelength_m",
                value=centroid,
                u=u_lambda,
                method="spectrum centroid (M-15)",
            ),
            MeasuredParameter(
                name="spectrum_fwhm_m",
                value=fwhm,
                u=_declared_u(spec, "spectrum_fwhm_m", math.sqrt(2.0) * u_lambda),
                method="numeric FWHM of the trace (informational report figure only: "
                "R-57(a) forbids writing a scalar linewidth next to lineshape='measured'; "
                "the table is the config input)",
            ),
            MeasuredParameter(
                name="linewidth_eff_hz",
                value=dnu_eff,
                u=_declared_u(spec, "linewidth_eff_hz"),
                method="noise-equivalent width 1/tau_c by Parseval (optics.source; R-56)",
            ),
            MeasuredParameter(
                name="rin_db_hz_floor",
                value=rin_db,
                u=_declared_u(spec, "rin_db_hz_floor"),
                method="ASE floor 2 tau_c of the measured spectrum (optics.source; R-57)",
            ),
        )
        return CharacterizationResult(
            params=params,
            provenance=_provenance(spec, sidecar_path, data),
            spectrum_wavelength_m=tuple(lam_list),
            spectrum_psd=tuple(psd_list),
        )


def _fwhm_lambda_m(lam: FloatArray, psd: FloatArray) -> float:
    """Numeric FWHM in the lambda domain: outermost half-maximum crossings.

    For a structured (multi-lobe) spectrum this is the *envelope* width between
    the outermost crossings -- a report-level figure only (the model consumes
    the whole table, R-57).
    """
    half = float(np.max(psd)) / 2.0
    above = psd >= half
    idx = np.flatnonzero(above)
    if idx.size == 0:  # pragma: no cover - max(psd)/2 always has one point above
        msg = "cannot locate the half-maximum level of the spectrum"
        raise ValueError(msg)
    lo = _crossing(lam, psd, int(idx[0]), half, left=True)
    hi = _crossing(lam, psd, int(idx[-1]), half, left=False)
    return hi - lo


def _crossing(lam: FloatArray, psd: FloatArray, index: int, half: float, *, left: bool) -> float:
    """Linear-interpolated half-maximum crossing next to ``index``."""
    j = index - 1 if left else index + 1
    if j < 0 or j >= lam.size:
        return float(lam[index])  # spectrum clipped at the table edge
    y0, y1 = float(psd[index]), float(psd[j])
    if y0 == y1:  # pragma: no cover - degenerate plateau
        return float(lam[index])
    frac = (y0 - half) / (y0 - y1)
    return float(lam[index] + frac * (lam[j] - lam[index]))


@CHARACTERIZATION_REGISTRY.register("rin_psd")
class RinPsdReader:
    """RIN(f) trace -> scalar ``rin_db_hz`` over a declared band (M-16 reduce).

    The trace must already be floor-corrected (shot/dark/Johnson subtracted;
    doc 20 F0-7/F0-8) -- the subtraction and the full spectral-shape use are
    the M-16 remainder. Per R-57(v) the reduced value **replaces** the derived
    ASE floor downstream; it is never summed with it.
    """

    def load(self, spec: CharacterizationSpec, sidecar_path: Path) -> CharacterizationResult:
        """Load the trace and take the median RIN over the declared band."""
        assert spec.columns is not None and spec.band is not None  # sidecar validator
        data = _data_path(spec, sidecar_path)
        freq_raw, rin_db = _read_xy_csv(data, min_rows=4)
        freq = np.asarray(
            [_to_si(v, spec.columns["x"].unit, _FREQ_UNITS, what="frequency") for v in freq_raw]
        )
        _to_si(1.0, spec.columns["y"].unit, _IDENTITY_DB_HZ, what="RIN")
        mask = (freq >= spec.band.f_min_hz) & (freq <= spec.band.f_max_hz)
        if int(np.count_nonzero(mask)) < 4:
            msg = (
                f"fewer than 4 trace points inside the reduction band "
                f"[{spec.band.f_min_hz}, {spec.band.f_max_hz}] Hz"
            )
            raise ValueError(msg)
        value = float(np.median(rin_db[mask]))
        u = _declared_u(spec, "rin_db_hz")
        if u is None:
            msg = (
                "rin_psd sidecar must declare uncertainties: {rin_db_hz: ...} (dB): "
                "plan 20 targets u ~ 0.5-1 dB and a config-bound parameter needs u (17 §4)"
            )
            raise ValueError(msg)
        param = MeasuredParameter(
            name="rin_db_hz",
            value=value,
            u=u,
            method=(
                f"median of the floor-corrected RIN trace over "
                f"[{spec.band.f_min_hz:g}, {spec.band.f_max_hz:g}] Hz (M-16 reduce; "
                "replaces the ASE floor per R-57(v))"
            ),
        )
        return CharacterizationResult(
            params=(param,), provenance=_provenance(spec, sidecar_path, data)
        )


@CHARACTERIZATION_REGISTRY.register("ringdown")
class RingdownReader:
    """Free-decay record ``(t, y)`` -> ``Q`` and ``f1`` (M-18).

    Model ``y = A exp(-pi f1 t / Q) cos(2 pi f1 t + phi)``: the Hilbert
    envelope decays at ``sigma = pi f1 / Q``, so ``Q = pi f1 / sigma`` with
    ``u(Q)/Q = sqrt((u_sigma/sigma)^2 + (u_f1/f1)^2)``. Guards: a uniform time
    grid, at least ~10 cycles in the fit window, and a *significant* decay
    (``sigma > 3 u_sigma``) -- an undamped tone is rejected loudly rather than
    reported as ``Q -> inf``.
    """

    _MIN_CYCLES = 10.0
    _EDGE_FRACTION = 0.05  # Hilbert edge effects: trim 5 % on both ends

    def load(self, spec: CharacterizationSpec, sidecar_path: Path) -> CharacterizationResult:
        """Load the record, fit the decrement and the frequency."""
        assert spec.columns is not None  # sidecar validator
        data = _data_path(spec, sidecar_path)
        t_raw, y = _read_xy_csv(data, min_rows=64)
        t = np.asarray([_to_si(v, spec.columns["x"].unit, _TIME_UNITS, what="time") for v in t_raw])
        _to_si(1.0, spec.columns["y"].unit, _ARBITRARY, what="ring-down signal")
        dt = np.diff(t)
        if np.any(dt <= 0.0) or (float(np.max(dt)) - float(np.min(dt))) > 1.0e-3 * float(
            np.mean(dt)
        ):
            msg = "ring-down time axis must be strictly increasing and uniform"
            raise ValueError(msg)
        fs = 1.0 / float(np.mean(dt))
        y = y - float(np.mean(y))

        f1, u_f1 = _fft_peak_hz(y, fs)
        sigma, u_sigma, n_cycles = self._fit_decay(t, y, f1)
        if sigma <= 3.0 * u_sigma:
            msg = (
                f"no significant decay in the record (sigma = {sigma:.3g} 1/s, "
                f"u = {u_sigma:.3g}): a ring-down needs a decaying envelope; "
                "an undamped tone cannot yield Q (M-18)"
            )
            raise ValueError(msg)
        if n_cycles < self._MIN_CYCLES:
            msg = (
                f"only {n_cycles:.1f} oscillation cycles inside the fit window; "
                f"at least {self._MIN_CYCLES:.0f} are needed for a trustworthy decrement"
            )
            raise ValueError(msg)
        q = math.pi * f1 / sigma
        u_q = q * math.hypot(u_sigma / sigma, u_f1 / f1)

        params = (
            MeasuredParameter(
                name="q_total",
                value=q,
                u=_declared_u(spec, "q_total", u_q),
                method=f"log-decrement of the Hilbert envelope, {n_cycles:.0f} cycles "
                "(M-18; Q = pi f1 / sigma)",
            ),
            MeasuredParameter(
                name="f1_hz",
                value=f1,
                u=_declared_u(spec, "f1_hz", u_f1),
                method="FFT peak with parabolic refinement (informational: f1 is a "
                "validation metric, not a config input; P20-1)",
            ),
        )
        return CharacterizationResult(
            params=params, provenance=_provenance(spec, sidecar_path, data)
        )

    def _fit_decay(self, t: FloatArray, y: FloatArray, f1: float) -> tuple[float, float, float]:
        """Linear fit of ``ln |envelope|`` -> decay rate sigma with its u."""
        env = np.abs(hilbert(y))
        n = env.size
        edge = max(1, int(self._EDGE_FRACTION * n))
        core = slice(edge, n - edge)
        env_core = env[core]
        t_core = t[core]
        # A ring-down starts at its maximum: fit from the beginning of the
        # (edge-trimmed) record down to 5 % of the initial envelope (~26 dB of
        # decay) or to the end. A *non*-decaying tone never drops to this
        # floor, so the whole record becomes the window and the near-zero
        # slope is caught by the sigma > 3 u_sigma significance guard below (a
        # clearer failure than a truncated window).
        peak = float(env_core[0])
        floor = 0.05 * peak
        below = np.flatnonzero(env_core <= floor)
        stop = int(below[0]) if below.size else env_core.size
        window = slice(0, stop)
        tw = t_core[window]
        ew = env_core[window]
        if tw.size < 16:
            msg = "decay window too short to fit the decrement (fewer than 16 samples)"
            raise ValueError(msg)
        design = np.column_stack([tw, np.ones_like(tw)])
        log_env = np.log(ew)
        coef, residuals, _, _ = np.linalg.lstsq(design, log_env, rcond=None)
        sigma = -float(coef[0])
        dof = tw.size - 2
        if residuals.size:
            rss = float(residuals[0])
        else:  # pragma: no cover - lstsq returns residuals for tall systems
            rss = float(np.sum((design @ coef - log_env) ** 2))
        s2 = rss / max(dof, 1)
        cov = s2 * np.linalg.inv(design.T @ design)
        u_sigma = math.sqrt(float(cov[0, 0]))
        n_cycles = f1 * float(tw[-1] - tw[0])
        return sigma, u_sigma, n_cycles


def _fft_peak_hz(y: FloatArray, fs: float) -> tuple[float, float]:
    """Dominant frequency by rFFT peak with parabolic interpolation.

    Returns the refined peak and a conservative bin-width uncertainty
    ``u = df / 2``.
    """
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, d=1.0 / fs)
    k = int(np.argmax(spectrum[1:])) + 1
    df = freqs[1] - freqs[0]
    if 1 <= k < spectrum.size - 1:
        alpha, beta, gamma = (float(spectrum[k - 1]), float(spectrum[k]), float(spectrum[k + 1]))
        denom = alpha - 2.0 * beta + gamma
        delta = 0.5 * (alpha - gamma) / denom if denom != 0.0 else 0.0
    else:  # pragma: no cover - peak at the spectrum edge
        delta = 0.0
    return float(freqs[k] + delta * df), float(df) / 2.0


@CHARACTERIZATION_REGISTRY.register("profile")
class ProfileReader:
    """Tip-contour trace ``(x, z)`` -> curvature radius ``R_c`` (M-17, 1 azimuth).

    Algebraic (Kasa) circle fit: ``x^2 + z^2 + D x + E z + F = 0`` is linear in
    ``(D, E, F)``; the centre is ``(-D/2, -E/2)`` and
    ``R = sqrt(D^2/4 + E^2/4 - F)``. ``u(R_c)`` is propagated from the linear
    least-squares covariance through the gradient of ``R(D, E, F)``. Guards:
    at least 20 contour points (doc 20 F0-3) and an RMS radial residual below
    5 % of ``R_c`` (a grossly non-circular contour fails loudly). Astigmatism
    (two azimuths -> ``R_cx != R_cy``) needs the toroidal optics M-03 and stays
    in the backlog: one file = one azimuth.
    """

    _MIN_POINTS = 20
    _MAX_RESIDUAL_REL = 0.05

    def load(self, spec: CharacterizationSpec, sidecar_path: Path) -> CharacterizationResult:
        """Load the contour and fit the circle."""
        assert spec.columns is not None  # sidecar validator
        data = _data_path(spec, sidecar_path)
        x_raw, z_raw = _read_xy_csv(data, min_rows=self._MIN_POINTS)
        x = np.asarray(
            [_to_si(v, spec.columns["x"].unit, _LENGTH_UNITS, what="profile x") for v in x_raw]
        )
        z = np.asarray(
            [_to_si(v, spec.columns["y"].unit, _LENGTH_UNITS, what="profile z") for v in z_raw]
        )
        radius, u_radius, residual_rel = _fit_circle(x, z)
        if residual_rel > self._MAX_RESIDUAL_REL:
            msg = (
                f"contour is not circular: RMS radial residual is "
                f"{100.0 * residual_rel:.1f} % of R_c (limit "
                f"{100.0 * self._MAX_RESIDUAL_REL:.0f} %); a melted tip should fit a "
                "circle over the crown (doc 20 F0-3)"
            )
            raise ValueError(msg)
        param = MeasuredParameter(
            name="curvature_radius_m",
            value=radius,
            u=_declared_u(spec, "curvature_radius_m", u_radius),
            method=f"Kasa circle fit, {x.size} points, residual "
            f"{100.0 * residual_rel:.2f} % of R_c (M-17, one azimuth)",
        )
        return CharacterizationResult(
            params=(param,), provenance=_provenance(spec, sidecar_path, data)
        )


def _fit_circle(x: FloatArray, z: FloatArray) -> tuple[float, float, float]:
    """Kasa circle fit with covariance-propagated ``u(R)``.

    Returns
    -------
    tuple of float
        ``(radius, u_radius, rms_radial_residual / radius)``.

    Raises
    ------
    ValueError
        On a degenerate (collinear) contour.
    """
    design = np.column_stack([x, z, np.ones_like(x)])
    target = -(x**2 + z**2)
    coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        msg = "profile points are collinear: a circle cannot be fitted (R_c -> inf?)"
        raise ValueError(msg)
    d_c, e_c, f_c = (float(coef[0]), float(coef[1]), float(coef[2]))
    r2 = d_c**2 / 4.0 + e_c**2 / 4.0 - f_c
    if r2 <= 0.0:
        msg = "circle fit produced a non-positive radius: contour is degenerate"
        raise ValueError(msg)
    radius = math.sqrt(r2)
    cx, cz = -d_c / 2.0, -e_c / 2.0
    radial = np.hypot(x - cx, z - cz) - radius
    dof = max(x.size - 3, 1)
    s2 = float(np.sum((design @ coef - target) ** 2)) / dof
    cov = s2 * np.linalg.inv(design.T @ design)
    grad = np.array([d_c / (4.0 * radius), e_c / (4.0 * radius), -1.0 / (2.0 * radius)])
    u_radius = math.sqrt(float(grad @ cov @ grad))
    residual_rel = float(np.sqrt(np.mean(radial**2))) / radius
    return radius, u_radius, residual_rel


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def resolve_sidecar_path(path: Path | str) -> Path:
    """Resolve a user-picked artifact path to its sidecar YAML.

    The convention of doc 16 §2a is ``<id>.csv`` + ``<id>.yaml`` side by side,
    so picking *either* file identifies the artifact: a ``.csv`` path hops to
    the same-stem sidecar next to it; a ``.yaml``/``.yml`` path is returned as
    is. Anything else fails loudly.

    Parameters
    ----------
    path : pathlib.Path or str
        Sidecar YAML or the CSV data file of one artifact.

    Returns
    -------
    pathlib.Path
        The sidecar path.

    Raises
    ------
    FileNotFoundError
        If a CSV was given and no same-stem sidecar exists next to it.
    ValueError
        On an unrecognised file extension.
    """
    given = Path(path)
    suffix = given.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return given
    if suffix == ".csv":
        for candidate in (given.with_suffix(".yaml"), given.with_suffix(".yml")):
            if candidate.is_file():
                return candidate
        msg = (
            f"no sidecar found next to {given.name}: a characterization artifact is "
            f"the CSV plus a same-stem YAML sidecar declaring its units, instrument "
            f"and uncertainties (doc 16 §2a) -- expected {given.with_suffix('.yaml').name}"
        )
        raise FileNotFoundError(msg)
    msg = f"unrecognised artifact file {given.name!r}: pick the sidecar YAML or the CSV"
    raise ValueError(msg)


def load_characterization(sidecar_path: Path | str) -> CharacterizationResult:
    """Load one characterization artifact from its sidecar YAML (or its CSV).

    Parameters
    ----------
    sidecar_path : pathlib.Path or str
        Path to the sidecar (``<id>.yaml``) or to the CSV data file (the
        same-stem sidecar next to it is used, doc 16 §2a); ``data_file``
        inside the sidecar is resolved relative to the sidecar directory.

    Returns
    -------
    CharacterizationResult
        Reduced parameters + provenance (+ the spectrum table for
        ``kind = "spectrum"``).

    Raises
    ------
    FileNotFoundError
        If the sidecar or the data file is missing.
    ValueError
        On a malformed sidecar, undeclared units, a malformed trace or a
        failed physical guard.
    """
    path = resolve_sidecar_path(sidecar_path)
    if not path.is_file():
        msg = f"characterization sidecar not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        msg = f"sidecar {path} must be a YAML mapping"
        raise ValueError(msg)
    spec = CharacterizationSpec.model_validate(raw)
    reader = CHARACTERIZATION_REGISTRY.create(spec.kind)
    result = reader.load(spec, path)
    logger.info(
        "characterization %s (%s): %s",
        path.name,
        spec.kind,
        ", ".join(f"{p.name}={p.value:.6g}" for p in result.params),
    )
    return result
