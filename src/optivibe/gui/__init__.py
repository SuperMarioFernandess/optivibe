"""Desktop GUI shell (PySide6 + PyQtGraph).

This package is the *only* place Qt is imported (architecture 09 §9). It runs the
exact same :mod:`optivibe.pipeline` the CLI runs, off the UI thread, and renders
results with PyQtGraph. It is shipped behind the optional ``gui`` extra so the
physics core installs and runs without Qt. Plot *content* that is not interactive
lives in :mod:`optivibe.viz` (Qt-free); this package only embeds it.

Sub-modules
-----------
``app``
    ``optivibe-gui`` entry point (creates the ``QApplication``).
``main_window``
    The top-level window (controls + live plot).
``controllers``
    Thread/worker lifecycle management (keeps the window thin).
``workers``
    The ``QObject`` worker that runs a scenario off the UI thread.
``widgets``
    Reusable widgets (the PyQtGraph live-plot placeholder).
"""
