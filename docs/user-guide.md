# User guide

OptiVibe runs from the command line (head-less) or through an optional desktop
app. Both drive the same pipeline and the same scenario files.

## Install

```bash
uv sync                      # core only
uv sync --extra gui          # + desktop app
uv sync --extra io-formats   # + TDMS/UFF/MAT/HDF5 replay
```

(Equivalently, `pip install -e '.[gui,io-formats]'`.)

## Command line

The `optivibe` command has three subcommands:

```bash
optivibe run    examples/hello.yaml          # run a scenario, print a summary
optivibe report examples/hello.yaml          # run + print the analysis budgets
optivibe sweep  examples/sweep_amplitude.yaml # run a parameter sweep / Monte-Carlo
```

Useful options:

- `optivibe run --config-dir <dir>` — override the `configs/` directory.
- `optivibe report --figures <dir>` — also save analysis figures as PNG.
- `optivibe sweep --out <dir>` — write sweep results (`.npz`) and figures.
- `optivibe -v …` — enable debug logging.

A **scenario** is a small YAML file: a name, a sensor `variant` (`A`–`D`), an
`excitation` block, the `stages` to run, and a `seed`. Variant `B`, for example,
is the general-purpose wideband configuration (1 Hz–10 kHz). The `examples/`
directory has a scenario for every excitation kind and several studies
(linearity ramp, resonance sweep, noise floor, cross-axis, Monte-Carlo).

## Desktop app

With the `gui` extra installed:

```bash
optivibe-gui
```

The app builds a scenario interactively — pick the variant and excitation kind,
fill the form, run the pipeline, and inspect the recovered signal and budgets in
PyQtGraph plots. The excitation builder exposes every kind, including the file
formats: choosing `csv`, `wav`, `tdms`, `uff`, `mat`, or `hdf5` reveals an import
form with a file picker and the format's channel/rate/units fields (the `tdms`,
`uff`, `mat`, and `hdf5` choices need the `io-formats` extra). See
[Importing measured data](data-import.md) for what those fields mean.

## Replaying measured records

Any scenario can replay a real capture instead of a synthetic signal by setting
the excitation `kind` to a file format and pointing it at a file. This is the
data seam described in [Importing measured data](data-import.md).
