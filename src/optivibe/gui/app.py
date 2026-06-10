"""``optivibe-gui`` entry point: create the application and show the window.

This is the thinnest possible application bootstrap (architecture 09 §9): it
installs logging, creates the single :class:`~PySide6.QtWidgets.QApplication`,
shows the :class:`~optivibe.gui.main_window.MainWindow`, and enters the Qt event
loop. All real work happens off this thread, in the pipeline worker.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from optivibe.core.logging import configure_logging
from optivibe.gui.main_window import MainWindow

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the OptiVibe desktop GUI.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Argument vector (defaults to ``sys.argv``).

    Returns
    -------
    int
        Qt application exit code.
    """
    configure_logging()
    app = QApplication(list(argv) if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover - module run shim
    raise SystemExit(main())
