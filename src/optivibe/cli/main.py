"""``optivibe`` command-line entry point.

Subcommands
-----------
``run <scenario.yaml>``
    Load a scenario, run the forward + inverse pipeline head-less, and print a
    short summary (variant, samples, dominant frequency, RMS metrics). This is
    the acceptance command of S0 (doc 10 §13).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from optivibe.core.logging import configure_logging, get_logger
from optivibe.pipeline import RunArtifacts, run_scenario

logger = get_logger(__name__)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser exposing the ``run`` subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="optivibe",
        description="OptiVibe: fiber-optic vibration sensor digital twin (head-less runner).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a scenario through the pipeline")
    run_p.add_argument("scenario", type=Path, help="path to a scenario YAML file")
    run_p.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="override the configs/ directory (variant presets)",
    )
    run_p.set_defaults(func=_cmd_run)
    return parser


def _format_summary(artifacts: RunArtifacts) -> str:
    """Render a human-readable one-block summary of a run.

    Parameters
    ----------
    artifacts : RunArtifacts
        Completed run.

    Returns
    -------
    str
        Multi-line summary text.
    """
    result = artifacts.result
    dominant = ", ".join(f"{f:.3f}" for f in result.dominant_freqs_hz) or "-"
    rms = result.rms
    rms_text = ", ".join(f"{key}={value:.4g}" for key, value in sorted(rms.items())) if rms else "-"
    duration_s = result.n_samples / result.fs
    lines = [
        f"scenario : {artifacts.scenario.name}",
        f"variant  : {artifacts.variant.name} ({artifacts.variant.description})",
        f"samples  : {result.n_samples}  @ fs = {result.fs:.1f} Hz  ({duration_s:.4f} s)",
        f"dominant : {dominant} Hz",
        f"rms      : {rms_text}",
    ]
    return "\n".join(lines)


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (``scenario``, ``config_dir``).

    Returns
    -------
    int
        Process exit code (0 on success, non-zero on failure).
    """
    try:
        artifacts = run_scenario(args.scenario, config_dir=args.config_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("run failed: %s", exc)
        return 2
    sys.stdout.write(_format_summary(artifacts) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``optivibe`` console script.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - module run shim
    raise SystemExit(main())
