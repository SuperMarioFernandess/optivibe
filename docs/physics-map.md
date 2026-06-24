# Physics → code map

OptiVibe's physics and interface contracts live in the knowledge-base documents
(00–08). This page maps each document to the code that implements it, so a reader
can move from a modelling decision to the module that realises it. The mapping is
the authority in document 09 §10; the contracts themselves stay in the documents.

| Knowledge-base document | Code |
|-------------------------|------|
| **00** — Project specification & assumptions (band 0.1 Hz–20 kHz, 0.1 g–50 g, target-axis recovery) | `configs/`, scenario validation |
| **01** — Notation, coordinate system, constants | `core/config`, `configs/constants.yaml`, `core/units` |
| **02** — Mechanical model (fiber cantilever, tip motion) | `mechanics` |
| **03** — Optical model (gap, reflector, back-coupling) | `optics` |
| **04** — ICD: mechanics ↔ optics interface | `core/types` (the in-code ICD: `TipState`, `Coupling`, …) |
| **05** — End-to-end transfer function | `pipeline`, plus the `mechanics` + `optics` tests |
| **07** — Noise budget & limit of detection (NEA) | `detector`, `analysis` |
| **08** — Optimisation & configuration family (variants A–D) | `configs/variants/*`, `core/config` presets |

Document **06** is the project decision log (physics); the software has its own
decision log (document 13).

## How the contracts show up in code

- **Units to SI at the edges.** Constants and unit conversions (document 01) are
  centralised in `core/units` and the constants config; every loader and stage
  converts to SI at its boundary rather than threading mixed units through the
  pipeline.
- **The tip-state ICD** (document 04) is a frozen data contract in
  `core/types` — the exact handoff `(dx, dy, dz, theta_x, theta_y)` between the
  mechanical and optical stages.
- **Variants A–D** (document 08) are YAML presets under `configs/variants/`,
  loaded and validated by `core/config`. A scenario names one with `variant: B`.

For the generated, docstring-level API of these modules, see the
[API reference](reference.md).
