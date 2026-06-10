"""Logging configuration helpers.

The library uses the standard :mod:`logging` module with one module-level logger
per module (``logging.getLogger(__name__)``); ``print`` is never used in library
code (coding convention 10 §7). Applications (CLI/GUI) call
:func:`configure_logging` once at start-up to install a console handler.
"""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single console handler on the root logger (idempotent).

    Parameters
    ----------
    level : int, optional
        Logging level for the root logger (default :data:`logging.INFO`).

    Notes
    -----
    Safe to call multiple times: if a handler is already present the function
    only updates the level, so repeated CLI/GUI start-ups do not duplicate output.
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    return logging.getLogger(name)
