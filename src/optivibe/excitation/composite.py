"""Composite excitation: the sum of several components on one sampling grid.

Task S-21 (doc 16); specification: doc 11 §2.1.4-§2.1.5. The source owns no
signal physics of its own -- it calls the *same* registered generators the
scenario would have called for each component and adds the results per axis, so
a one-component composite reproduces that component bit-for-bit by construction
rather than by numerical coincidence.

Level convention (doc 11 §2.1.4)
--------------------------------
Every component keeps the level it declares and the sum is **not**
renormalized. Consequences, both intended:

* the composite mean square is known in closed form before the run --
  ``<a^2> = sum_i <a_i^2>`` whenever the components occupy disjoint spectral
  supports (the cross terms vanish), which is what the Parseval golden checks;
* the total may exceed the variant full scale. That is a legitimate study
  (behaviour beyond FS is itself a subject, doc 00), so the level is reported,
  not clipped: ``meta`` carries the composite ``rms``/``peak`` and the per
  component RMS, and the orchestrator warns when the peak passes FS.

Seeding (doc 11 §2.1.5)
-----------------------
Component 0 inherits the scenario seed unchanged -- that is what makes a
one-component composite bit-identical to the standalone kind, including
``random``. Components 1.. draw a deterministic sub-seed from
``SeedSequence([scenario_seed, tag, index])``, so two ``random`` components in
one composite never share a stream (which would silently make one doubled noise
out of two independent ones). Appending a component therefore leaves the
existing streams untouched; inserting one in the middle re-rolls the later
streams -- deterministic, but position-dependent, which is why ``RandomSpec``
also takes an explicit ``seed`` that pins a realization irrespective of
position.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optivibe.core.config.models import CompositeSpec, ExcitationSpec
from optivibe.core.stages import ExcitationSource
from optivibe.core.types import Excitation, FloatArray
from optivibe.excitation.random_noise import RandomExcitationSource
from optivibe.excitation.shock import ShockExcitationSource
from optivibe.excitation.sweep import SweepExcitationSource
from optivibe.excitation.tonal import MultitoneExcitationSource, SineExcitationSource

__all__ = ["CompositeExcitationSource", "component_seed"]

# Fixed tag mixed into the scenario seed so a component's noise stream is
# reproducible yet independent of the other components and of the detector
# stream (10 §8; same pattern as ``detector_seed_sequence``). Spells "CMP0".
_COMPOSITE_SEED_TAG = 0x434D5030

# Component kind -> generator class. Bound directly (not through
# EXCITATION_REGISTRY) so this module stays importable from the package
# ``__init__`` that builds the registry, without an import cycle.
_COMPONENT_SOURCES: dict[str, Callable[[], ExcitationSource]] = {
    "sine": SineExcitationSource,
    "multitone": MultitoneExcitationSource,
    "sweep": SweepExcitationSource,
    "random": RandomExcitationSource,
    "shock": ShockExcitationSource,
}


def component_seed(scenario_seed: int | None, index: int) -> int | None:
    """Return the seed of composite component ``index`` (doc 11 §2.1.5).

    Parameters
    ----------
    scenario_seed : int or None
        The scenario-level seed; ``None`` propagates (non-reproducible run).
    index : int
        0-based position of the component in ``CompositeSpec.components``.

    Returns
    -------
    int or None
        ``scenario_seed`` itself for ``index == 0`` (so a one-component
        composite reproduces the standalone kind bit-for-bit), a deterministic
        sub-seed derived from :class:`numpy.random.SeedSequence` otherwise.
    """
    if scenario_seed is None:
        return None
    if index == 0:
        return int(scenario_seed)
    sequence = np.random.SeedSequence(entropy=[int(scenario_seed), _COMPOSITE_SEED_TAG, int(index)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


class CompositeExcitationSource:
    """Sum of registered component generators on the composite's grid."""

    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Generate the composite time series (see :class:`CompositeSpec`).

        Parameters
        ----------
        spec : ExcitationSpec
            Must be a :class:`CompositeSpec`.
        seed : int or None, optional
            Scenario seed; distributed over the components by
            :func:`component_seed`.

        Returns
        -------
        Excitation
            The summed 3-axis input, with the composite level and the per
            component RMS in ``meta``.
        """
        if not isinstance(spec, CompositeSpec):
            msg = f"'composite' source expects CompositeSpec, got kind={spec.kind!r}"
            raise TypeError(msg)
        channels: dict[str, FloatArray] | None = None
        components: list[dict[str, object]] = []
        for index, component in enumerate(spec.components):
            source: ExcitationSource = _COMPONENT_SOURCES[component.kind]()
            part = source.generate(component, seed=component_seed(seed, index))
            axes = {"x": part.a_x, "y": part.a_y, "z": part.a_z}
            if channels is None:
                channels = {axis: values.copy() for axis, values in axes.items()}
            else:
                for axis, values in axes.items():
                    channels[axis] += values
            components.append(
                {
                    "kind": component.kind,
                    "axis": component.axis,
                    "rms_m_s2": _rms(axes[component.axis]),
                }
            )
        assert channels is not None  # components has min_length=1
        # Level of the composite as a vector quantity: the magnitude series
        # |a(t)|, whose RMS reduces to the axis RMS for a single-axis composite
        # and whose peak is what the full-scale warning compares against.
        magnitude = np.sqrt(channels["x"] ** 2 + channels["y"] ** 2 + channels["z"] ** 2)
        meta: dict[str, object] = {
            "generator": "composite",
            "axis": spec.axis,
            "n_components": len(spec.components),
            "components": components,
            "rms_m_s2": _rms(magnitude),
            "peak_m_s2": float(np.max(magnitude)),
        }
        return Excitation(
            a_x=channels["x"],
            a_y=channels["y"],
            a_z=channels["z"],
            fs=spec.fs_hz,
            seed=seed,
            meta=meta,
        )


def _rms(values: FloatArray) -> float:
    """Root mean square of a time series, in its own units."""
    return float(np.sqrt(np.mean(values**2)))
