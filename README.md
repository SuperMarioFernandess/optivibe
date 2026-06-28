# OptiVibe

Digital twin of a **fiber-optic vibration sensor** — a clean, contract-driven
simulation of the full signal chain (base acceleration → fiber-cantilever tip
motion → optical back-coupling → photodetector signal) and its inverse
(reconstruction of the target-axis vibration), with analytics and an optional
desktop GUI.

This repository is the **software** of the project; the physics and interface
contracts live in the knowledge-base documents (00–13). The architecture is
specified in document 09 and the coding conventions in document 10.

> **Status: S6 published; S7 desktop app.** The package installs, the forward +
> inverse pipeline and the analytics (truth-vs-recovery, NEA budget, parameter
> sweeps, tolerance Monte-Carlo) run end to end head-less, and the **desktop
> application** drives the same core off the UI thread with live plots and
> embedded report figures. The physics stages (modal mechanics, Gaussian-beam
> cylinder optics, photodiode noise, calibrated DSP + switchable sensitivity)
> are implemented behind frozen contracts; the stubs remain registered for
> regression.

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

Analysis report and parameter sweeps (head-less):

```bash
optivibe report examples/recover_sine.yaml          # truth-vs-recovery + NEA budget
optivibe sweep  examples/nea_vs_L.yaml --out out/L  # design sweep, npz + figure
```

## Desktop application

The desktop app (`optivibe-gui`, requires the `gui` extra) is a **thin shell over
the same core** — it builds a scenario, runs it *off the UI thread*, and renders
the result. It contains no physics or DSP of its own.

```bash
optivibe-gui
```

Workflow:

1. **Control panel (left).** Describe the sensor as an **editable composition**:
   start from one of the A/B/C/D *starting compositions*, then edit any subsystem
   (source / fiber / cantilever / reflector / detector) by choosing a preset and
   overriding its labelled, unit-carrying fields. The **reflector** form selects
   the profile (`cylinder` / `sphere` / `plane` / `wedge`) with the matching
   parameters (`R_c` for the curved shapes, `α_w` for the wedge). Save a
   composition to `configs/user/systems/` and load it back. Then build the
   excitation (`sine` / `multitone` / `sweep` / `random` / `shock`, or replay a
   CSV/WAV file) — the **multitone** editor adds/removes components dynamically
   (default two) with an optional per-tone phase — and flip the physics layers
   (optics `cylinder` ↔ `stub`, detector `photodiode` ↔ `stub`, DSP `standard` ↔
   `stub`, plus the seed). The defaults reproduce variant B / `recover_sine`, so
   the first **Run** shows a faithful recovery. (The calibrated `standard` inverse
   is cylinder-only; the sphere/plane/wedge family runs with the `stub` inverse.)
2. **Run / Report.** *Run* executes the forward + inverse pipeline on a worker
   thread — resolving the edited composition off the UI thread — and fills the
   **Live** tab (a bending-cantilever animation, the input-vs-recovered
   acceleration, the detector signal, the recovered velocity/displacement and
   spectrum). Each Live panel has a **show/hide checkbox** that reflows the layout
   so the visible panels expand. *Report* adds the analysis: the **Report** tab
   shows the truth-vs-recovery a/v/x overlay, the NEA budget with its
   shot/RIN/Johnson split, the spectrogram and the error budget; the Live NEA(f)
   panel fills in.
3. **Sweeps / Monte-Carlo.** The **Sweeps** tab runs a design or response
   parameter sweep; the **Monte-Carlo** tab runs a tolerance Monte-Carlo. Both
   render the corresponding `viz` figure.
4. **Physics.** The **Physics** tab is a reference surface for the current
   composition: light, auto-recomputed design curves (`f1(L)`, `|H_lat(f)|`,
   `η(Δx)`) built by `optivibe.viz.physics`, a *Compute NEA(f)* button that runs
   the measured budget through the worker, and reference notes (mechanics,
   reference-arm options, inverse/DSP, sensitivity, integrator). The sensor
   *family* sweep is on the Sweeps tab.
5. **Cancel / Export.** A running job can be cancelled (its result is dropped);
   **Export** writes the current figures (PNG) and result (`.npz`) to a folder.

The heavy work always runs on a background `QThread` (the GUI thread never
computes); the same runs are reachable head-less through `optivibe` (parity).

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
