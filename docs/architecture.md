# Architecture

This page summarizes how the code is organized; the authoritative specification
is knowledge-base document 09.

## Layers (dependencies point inward)

```
cli / gui  ->  pipeline  ->  stage implementations  ->  core
                                (excitation,            (contracts, config,
                                 mechanics, optics,      units, registry,
                                 detector, dsp)          stage protocols)
```

- **core** has no knowledge of the GUI, file formats or any concrete physics.
  It defines the data contracts (`core.types`), the configuration models and
  loaders (`core.config`), the unit policy (`core.units`), the
  `Registry` (`core.registry`) and the stage **protocols** (`core.stages`).
- Each **stage family** lives in its own package, owns a `Registry`, and
  registers one or more implementations under string keys. S0 ships stub/identity
  implementations under the key `stub` (and `sine` for excitation).
- **pipeline** wires the stages in order (forward then inverse). It contains no
  physics: it only resolves implementations from the registries by the keys in
  the scenario and calls them.

## Why a registry?

Adding a new reflector, light source, DSP method or data loader is a *new adapter
with a registration*, never a change to the core (decision SW-02). The scenario
selects implementations by key:

```yaml
stages:
  mechanics: stub      # later: "modal"
  optics: stub         # later: "gaussian_beam"
```

## Data contracts

Stages communicate **only** through the typed, immutable contracts in
`core.types` (a mirror of interface document 04): `Excitation`, `TipState`,
`OpticalResponse`, `DetectorOutput`, `Spectrum`, `VibrationResult`. Array-carrying
contracts are frozen dataclasses validated once on construction; pydantic is
reserved for configuration and metadata.

## Configuration

Three levels (document 09 §7): physical **constants** (`configs/constants.yaml`,
mirror of document 01), sensor-family **variants** (`configs/variants/*.yaml`,
mirror of document 08) and a per-run **scenario**. Numbers are never duplicated
in code; a consistency test checks the YAML against the documented references.

## GUI threading

The desktop shell (`gui`) is the only place Qt is imported, and it is optional.
The long-running pipeline runs **off the UI thread**: a `PipelineWorker`
(`QObject`) is moved onto a `QThread` by a `RunController`, which re-emits the
result back on the UI thread via signals. Non-interactive plot content lives in
the Qt-free `viz` package so the same outputs are reachable head-less.
