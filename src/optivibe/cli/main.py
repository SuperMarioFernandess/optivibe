"""``optivibe`` command-line entry point.

Subcommands
-----------
``run <scenario.yaml>``
    Load a scenario, run the forward + inverse pipeline head-less, and print a
    short summary (variant, samples, dominant frequency, RMS metrics). The
    acceptance command of S0 (doc 10 §13).
``report <scenario.yaml>``
    Run a scenario and print the end-to-end ``truth vs recovery`` error budget
    and the NEA budget (task S6 §B); optionally save the analysis figures.
``sweep <spec.yaml>``
    Run a parameter sweep or a tolerance Monte-Carlo from its spec, print a
    summary, and persist the result (``.npz``) and figures (task S6 §B7/§B8/§B9).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure

from optivibe.analysis import (
    MonteCarloSpec,
    SweepSpec,
    load_analysis_spec,
    nea_budget,
    run_monte_carlo,
    run_sweep,
    save_monte_carlo_npz,
    save_sweep_npz,
    truth_vs_recovery,
)
from optivibe.core.logging import configure_logging, get_logger
from optivibe.pipeline import RunArtifacts, run_scenario

logger = get_logger(__name__)

__all__ = ["build_parser", "main"]

G0 = 9.80665


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser exposing the ``run``, ``report`` and ``sweep`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="optivibe",
        description="OptiVibe: fiber-optic vibration sensor digital twin (head-less runner).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a scenario through the pipeline")
    run_p.add_argument("scenario", type=Path, help="path to a scenario YAML file")
    run_p.add_argument("--config-dir", type=Path, default=None, help="override the configs/ dir")
    run_p.set_defaults(func=_cmd_run)

    report_p = sub.add_parser("report", help="run a scenario and print the analysis budgets")
    report_p.add_argument("scenario", type=Path, help="path to a scenario YAML file")
    report_p.add_argument("--config-dir", type=Path, default=None, help="override the configs/ dir")
    report_p.add_argument(
        "--figures", type=Path, default=None, help="directory to save analysis figures (PNG)"
    )
    report_p.set_defaults(func=_cmd_report)

    sweep_p = sub.add_parser("sweep", help="run a parameter sweep or Monte-Carlo spec")
    sweep_p.add_argument("spec", type=Path, help="path to a sweep/montecarlo spec YAML file")
    sweep_p.add_argument(
        "--out", type=Path, default=None, help="output directory for results (npz) and figures"
    )
    sweep_p.set_defaults(func=_cmd_sweep)
    return parser


def _format_summary(artifacts: RunArtifacts) -> str:
    """Render a human-readable one-block summary of a run."""
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
    """Execute the ``run`` subcommand."""
    try:
        artifacts = run_scenario(args.scenario, config_dir=args.config_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("run failed: %s", exc)
        return 2
    sys.stdout.write(_format_summary(artifacts) + "\n")
    return 0


def _save_figure(fig: Figure, directory: Path, name: str) -> Path:
    """Save a figure as PNG under ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{name}.png"
    fig.savefig(out, dpi=120)
    return out


def _cmd_report(args: argparse.Namespace) -> int:
    """Execute the ``report`` subcommand: truth-vs-recovery + NEA budgets."""
    try:
        artifacts = run_scenario(args.scenario, config_dir=args.config_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("report failed: %s", exc)
        return 2
    budget = truth_vs_recovery(
        artifacts.forward.excitation,
        artifacts.result,
        artifacts.forward.detector,
        variant=artifacts.variant,
    )
    out = [_format_summary(artifacts), "", budget.summary_text()]
    nb = nea_budget(artifacts.forward.detector, artifacts.variant)
    if nb is not None:
        out += [
            "",
            "NEA budget (plateau):",
            f"  plateau           : {nb.nea_plateau / G0 * 1e6:.3g} ug/sqrt(Hz)",
            f"  full band         : {nb.nea_full_band / G0 * 1e6:.4g} ug (RMS)",
            f"  shot/rin/johnson  : "
            f"{nb.contributions['shot'] / G0 * 1e6:.3g} / "
            f"{nb.contributions['rin'] / G0 * 1e6:.3g} / "
            f"{nb.contributions['johnson'] / G0 * 1e6:.3g} ug/sqrt(Hz)",
            f"  analytic rel err  : {nb.psd_rel_error:.2e} (ref arm {nb.reference_arm})",
        ]
    sys.stdout.write("\n".join(out) + "\n")
    if args.figures is not None:
        from optivibe.viz.analysis import plot_nea_budget, plot_truth_vs_recovery_avx

        a_true = artifacts.forward.excitation.a_x
        _save_figure(plot_truth_vs_recovery_avx(a_true, artifacts.result), args.figures, "truth")
        if nb is not None:
            _save_figure(plot_nea_budget(nb), args.figures, "nea_budget")
        sys.stdout.write(f"figures saved to {args.figures}\n")
    return 0


def _report_sweep(spec: SweepSpec, out_dir: Path | None) -> int:
    """Run and report a parameter sweep."""
    result = run_sweep(spec)
    lines = [
        f"sweep '{result.name}' ({result.mode}) over {result.parameter} ["
        f"{result.variant}]: {len(result.axis_labels)} points"
    ]
    for key, values in sorted(result.metrics.items()):
        head = ", ".join(f"{v:.4g}" for v in values[:6])
        more = " ..." if values.size > 6 else ""
        lines.append(f"  {key:20s}: {head}{more}")
    sys.stdout.write("\n".join(lines) + "\n")
    if out_dir is not None:
        path = save_sweep_npz(result, out_dir / result.name)
        from optivibe.viz.analysis import plot_sweep

        _save_figure(plot_sweep(result), out_dir, result.name)
        sys.stdout.write(f"saved {path} and figure to {out_dir}\n")
    return 0


def _report_monte_carlo(spec: MonteCarloSpec, out_dir: Path | None) -> int:
    """Run and report a tolerance Monte-Carlo."""
    result = run_monte_carlo(spec)
    lines = [
        f"monte-carlo '{result.name}' [{result.variant}]: {result.n_draws} draws; "
        f"tolerances: {', '.join(result.tolerances) or '-'}"
    ]
    for key, stats in sorted(result.stats.items()):
        lines.append(
            f"  {key:24s}: p05={stats['p05']:.4g}  p50={stats['p50']:.4g}  p95={stats['p95']:.4g}"
        )
    sys.stdout.write("\n".join(lines) + "\n")
    if out_dir is not None:
        path = save_monte_carlo_npz(result, out_dir / result.name)
        from optivibe.viz.analysis import plot_monte_carlo

        _save_figure(plot_monte_carlo(result), out_dir, result.name)
        sys.stdout.write(f"saved {path} and figure to {out_dir}\n")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Execute the ``sweep`` subcommand (sweep or Monte-Carlo, by spec kind)."""
    try:
        spec = load_analysis_spec(args.spec)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("sweep failed: %s", exc)
        return 2
    if isinstance(spec, SweepSpec):
        return _report_sweep(spec, args.out)
    return _report_monte_carlo(spec, args.out)


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
