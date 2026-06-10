"""Command-line interface package.

Exposes the headless entry point ``optivibe`` (see :mod:`optivibe.cli.main`).
The CLI is a thin application layer over :mod:`optivibe.pipeline`; it owns no
physics and runs the exact same pipeline the GUI does, preserving headless
parity (architecture 09 §9).
"""
