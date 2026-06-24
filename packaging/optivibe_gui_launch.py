"""Frozen-app entry point for PyInstaller (task S8).

PyInstaller bundles a *script*, not a console-script entry point, so this thin
launcher just calls the same ``main`` that the ``optivibe-gui`` entry point uses
(see ``[project.gui-scripts]`` in ``pyproject.toml``). Keep it dependency-free
beyond the package itself.
"""

from __future__ import annotations

from optivibe.gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
