"""Band-limited random excitation with a prescribed one-sided PSD or band RMS.

Synthesis is done directly in the frequency domain (doc 11 §2.1, "spectral
shaping"): independent complex-Gaussian amplitudes are drawn on the rFFT bins
inside the band and inverse-transformed.

Scaling
-------
For a one-sided target PSD ``S0`` [(m/s^2)^2/Hz] and ``X = rfft(x)`` of an
``n``-sample record at rate ``fs``, the periodogram estimate of the one-sided
PSD at an interior bin is ``S_k = 2 |X_k|^2 / (fs * n)``. Drawing
``X_k = (u + i v) * sigma`` with ``u, v ~ N(0, 1)`` gives
``E|X_k|^2 = 2 sigma^2``, so the target is met by

    sigma = sqrt(S0 * fs * n) / 2.

By Parseval the expected mean square is ``S0 * (f_hi - f_lo)``, i.e. the band
RMS is ``sqrt(S0 * BW)`` — the flat-shape link between the two level options
of :class:`RandomSpec`. When the level is given as ``g_rms`` the realization
is additionally normalized to hit the target RMS exactly (deterministic given
the seed); when given as ``psd_g2_hz`` the statistics of the PSD are kept and
only the expectation matches the target.

DC and (for even ``n``) the Nyquist bin are always zero, so the signal is
zero-mean by construction.
"""

from __future__ import annotations

import numpy as np

from optivibe.core.config.models import ExcitationSpec, RandomSpec
from optivibe.core.types import Excitation
from optivibe.core.units import G0_M_S2
from optivibe.excitation._common import pack_on_axis

__all__ = ["RandomExcitationSource"]


class RandomExcitationSource:
    """Gaussian band-limited noise on one axis, reproducible by seed (10 §8)."""

    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Generate a random time series (see :class:`RandomSpec`).

        Raises
        ------
        ValueError
            If the requested band contains no FFT bins for the given duration.
        """
        if not isinstance(spec, RandomSpec):
            msg = f"'random' source expects RandomSpec, got kind={spec.kind!r}"
            raise TypeError(msg)
        n = round(spec.duration_s * spec.fs_hz)
        if n < 2:
            msg = f"random excitation needs >= 2 samples, got {n}"
            raise ValueError(msg)
        f_lo, f_hi = spec.band_hz
        freqs = np.fft.rfftfreq(n, d=1.0 / spec.fs_hz)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        mask[0] = False  # no DC
        if n % 2 == 0:
            mask[-1] = False  # keep the Nyquist bin real-zero
        n_bins = int(np.count_nonzero(mask))
        if n_bins == 0:
            msg = (
                f"band {spec.band_hz} Hz contains no FFT bins at df = "
                f"{spec.fs_hz / n:.6g} Hz; increase duration_s or widen the band"
            )
            raise ValueError(msg)

        bandwidth = f_hi - f_lo
        if spec.psd_g2_hz is not None:
            psd_si = spec.psd_g2_hz * G0_M_S2**2  # (m/s^2)^2 / Hz
        else:
            assert spec.g_rms is not None
            psd_si = (spec.g_rms * G0_M_S2) ** 2 / bandwidth

        rng = np.random.default_rng(seed)
        sigma = np.sqrt(psd_si * spec.fs_hz * n) / 2.0
        spectrum = np.zeros(freqs.size, dtype=np.complex128)
        spectrum[mask] = sigma * (rng.standard_normal(n_bins) + 1j * rng.standard_normal(n_bins))
        wave = np.fft.irfft(spectrum, n=n).astype(np.float64)

        if spec.g_rms is not None:
            actual_rms = float(np.sqrt(np.mean(wave**2)))
            if actual_rms > 0.0:
                wave *= (spec.g_rms * G0_M_S2) / actual_rms

        meta: dict[str, object] = {
            "generator": "random",
            "axis": spec.axis,
            "band_hz": spec.band_hz,
            "g_rms": spec.g_rms,
            "psd_g2_hz": spec.psd_g2_hz,
            "shape": spec.shape,
            "n_band_bins": n_bins,
        }
        return pack_on_axis(wave, spec.axis, spec.fs_hz, seed, meta)
