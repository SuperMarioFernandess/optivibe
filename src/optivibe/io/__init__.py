"""I/O package: real-data import behind the two data seams of decision SW-08.

The seam distinguishes two roles (SW-08): (a) the replay *input* --
:mod:`optivibe.io.loaders`, measured acceleration mapped onto the
:class:`~optivibe.core.types.Excitation` contract (CSV/WAV in S1; TDMS/UFF/
MAT/HDF5 in S8, ``LOADER_REGISTRY``); and (b) the instrument *output* --
:mod:`optivibe.io.records` (role S-02, doc 20 §5), recorded photocurrent mapped
onto :class:`~optivibe.core.types.DetectorOutput` (``RECORD_REGISTRY``).
Persistence of run artifacts (HDF5/Parquet, 11 §5) remains a future extension.
"""
