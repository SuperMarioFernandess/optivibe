# OptiVibe

Digital twin of a **fiber-optic vibration sensor**: a contract-driven simulation
of the full forward chain (base acceleration → fiber-cantilever tip motion →
optical back-coupling → photodetector signal) and the inverse chain
(reconstruction of the target-axis vibration), with analytics and an optional
PySide6 desktop shell.

The physics and interface contracts are specified in the knowledge-base
documents (00–13). This site documents the **software**: its architecture
(document 09), its conventions (document 10) and its public API.

## Quick start

```bash
uv sync                       # core
uv sync --extra gui           # + desktop app  (optivibe-gui)
uv sync --extra io-formats    # + TDMS/UFF/MAT/HDF5 replay
optivibe run examples/hello.yaml
```

## Where to look

- **User guide** — the CLI, the desktop app, and replaying measured records. See
  [User guide](user-guide.md).
- **Importing measured data** — the instrument-format loaders and how units map
  to SI. See [Importing measured data](data-import.md).
- **Physics → code map** — which module implements each knowledge-base document.
  See [Physics → code map](physics-map.md).
- **Architecture** — the layered design, the registry/ports pattern, the data
  contracts and the GUI threading model. See [Architecture](architecture.md).
- **Packaging** — building the desktop app bundle. See [Packaging](packaging.md).
- **API reference** — the modules, contracts and stages, generated from the
  numpy-style docstrings. See [API reference](reference.md).

!!! note "S0 status"
    The package is at the **architectural-skeleton** stage: it installs, runs the
    pipeline end to end, and launches the GUI, but the physics stages are stubs
    that later stages replace behind the same contracts.
