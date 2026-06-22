"""GUI workers: the off-UI-thread job runner and the Qt-free job definitions.

Sub-modules
-----------
``jobs``
    Qt-free :class:`Job` definitions calling the core/analysis (no Qt --
    unit-testable without a display).
``job_worker``
    :class:`JobWorker` -- the ``QObject`` that runs a job off the UI thread
    (imports Qt).

Submodules are imported directly (e.g.
``from optivibe.gui.workers.jobs import ScenarioJob``) so importing the Qt-free
jobs does not pull in Qt.
"""
