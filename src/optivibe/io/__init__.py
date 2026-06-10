"""I/O package: run persistence and real-data import (lands in S1/S8).

This package will hold the loaders/savers behind the data seam of decision SW-08:
CSV/WAV import of measured acceleration registered in an excitation-source
registry, and HDF5/Parquet persistence of run artifacts (09 §4, 11 §5). In S0 it
is an empty namespace so the package layout and contracts are fixed first.
"""
