"""Excitation source that replays a measured record from disk (seam SW-08).

This is the bridge between the excitation-generator registry and the
data-loader registry: the source looks the loader up by the spec ``kind``
(``"csv"`` or ``"wav"``) in :data:`optivibe.io.loaders.LOADER_REGISTRY` and
returns the loader's :class:`Excitation` unchanged. Instrument formats
(TDMS / UFF / MAT / HDF5) plug in later by registering new loaders (S8);
this source needs no change for that.
"""

from __future__ import annotations

from optivibe.core.config.models import CsvSpec, ExcitationSpec, WavSpec
from optivibe.core.types import Excitation
from optivibe.io.loaders import LOADER_REGISTRY

__all__ = ["FileExcitationSource"]


class FileExcitationSource:
    """Replay source: delegates to the file loader registered under ``kind``."""

    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Load the record described by a file spec (CSV or WAV)."""
        if not isinstance(spec, (CsvSpec, WavSpec)):
            msg = f"file source expects CsvSpec or WavSpec, got kind={spec.kind!r}"
            raise TypeError(msg)
        loader = LOADER_REGISTRY.create(spec.kind)
        return loader.load(spec, seed=seed)
