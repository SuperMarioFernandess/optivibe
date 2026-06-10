# OptiVibe

Digital twin of a **fiber-optic vibration sensor** — a clean, contract-driven
simulation of the full signal chain (base acceleration → fiber-cantilever tip
motion → optical back-coupling → photodetector signal) and its inverse
(reconstruction of the target-axis vibration), with analytics and an optional
desktop GUI.

This repository is the **software** of the project; the physics and interface
contracts live in the knowledge-base documents (00–13). The architecture is
specified in document 09 and the coding conventions in document 10.

> **Status: S0 — architectural skeleton.** The package installs, the pipeline
> runs end to end head-less, and the GUI launches; the *physics* stages
> (mechanics, optics, detector noise, calibrated DSP) are deliberate stubs that
> later stages (S1–S8) replace behind the same contracts.

## Install

The project uses [uv](https://docs.astral.sh/uv/) and a `src/` layout.

```bash
uv venv
uv pip install -e ".[dev]"      # core + developer tooling
uv pip install -e ".[dev,gui]"  # also install the PySide6 desktop shell
uv lock                         # generate the pinned uv.lock (not committed yet)
```

(Plain `pip install -e ".[dev]"` works too.)

## Run

Head-less scenario runner (the S0 acceptance command):

```bash
optivibe run examples/hello.yaml
```

Desktop GUI (requires the `gui` extra):

```bash
optivibe-gui
```

## Layout

```
src/optivibe/
  core/        contracts (types), config (pydantic + YAML), units, registry, stage protocols
  excitation/  base-acceleration generators (S0: sine)
  mechanics/   acceleration -> tip state q_tip (S0: stub identity; S2: modal H_lat)
  optics/      tip state -> coupling efficiency eta (S0: bounded stub; S3: Gaussian-beam)
  detector/    eta -> digitized photocurrent S = R·P·(R1 + rho·eta) (S0: noiseless)
  dsp/         inverse: detector -> reconstructed a/v/x + spectrum (S0: minimal real)
  pipeline/    forward+inverse orchestrator (no physics of its own)
  analysis/    cross-sensitivity, sweeps, metrics (S6)
  viz/         Qt-free figures (matplotlib/plotly)
  io/          real-data import + run persistence (S1/S8)
  cli/         `optivibe` head-less entry point
  gui/         `optivibe-gui` desktop shell (PySide6 + PyQtGraph), runs off-UI-thread
configs/       constants.yaml + variants/{A,B,C,D}.yaml + scenarios/
examples/      runnable scenarios (hello.yaml)
tests/         smoke, config-consistency, golden (vs analytical references), pipeline, GUI
docs/          mkdocs-material site
```

## Single source of truth

Numbers are **not** hard-coded in Python. Physical constants live in
`configs/constants.yaml` (mirror of document 01) and the sensor-family presets in
`configs/variants/*.yaml` (mirror of document 08). A consistency test
(`tests/test_constants_golden.py`) checks them against the documented references
and against the analytical first-mode law `f₁ ≈ 100 / L²[mm] kHz`.

## Quality gates (Definition of Done, doc 10 §13)

```bash
ruff check . && ruff format --check .   # lint + format
mypy                                    # strict typing (src/)
pytest                                  # tests; core coverage >= 85%
optivibe run examples/hello.yaml        # head-less run works
```

CI runs the same gates on every push (`.github/workflows/ci.yml`).
