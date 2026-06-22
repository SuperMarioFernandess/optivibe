"""GUI controllers: thread/worker lifecycle and Qt-free config assembly.

Sub-modules
-----------
``job_controller``
    :class:`JobController` -- owns the worker thread, canonical teardown,
    cancellation and progress (imports Qt).
``run_controller``
    :class:`RunController` -- the back-compatible S0 scenario controller (a thin
    adapter over :class:`JobController`; imports Qt).
``scenario_builder``
    Qt-free assembly of validated ``ScenarioConfig`` / sweep / Monte-Carlo specs
    from GUI payloads (no Qt -- unit-testable without a display).

Submodules are imported directly (e.g.
``from optivibe.gui.controllers.scenario_builder import build_scenario_config``)
so importing the Qt-free builder does not pull in Qt.
"""
