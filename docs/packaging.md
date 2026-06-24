# Packaging the desktop app

The GUI can be bundled into a self-contained desktop application with
[PyInstaller](https://pyinstaller.org/) (decision SW-38). The build is
**Windows-first** — the CI build job runs on `windows-latest` and uploads the
result — but the spec is host-portable and also builds a Linux/macOS bundle
locally.

## Build

```bash
uv sync --extra gui --extra io-formats --extra packaging
uv run pyinstaller packaging/optivibe-gui.spec --noconfirm
```

The result is a one-dir bundle in `dist/OptiVibe/`; launch
`dist/OptiVibe/OptiVibe` (`OptiVibe.exe` on Windows). The spec ships the
`configs/` and `examples/` trees next to the binary, so the bundled examples run
without a source checkout, and it pulls in the lazily-imported backends
(`scipy`, `pyqtgraph`, and the `io-formats` readers) as hidden imports.

## CI artifact

On every run, the `package-windows` job builds the bundle and uploads it as a
GitHub Actions artifact (`optivibe-gui-windows-<sha>`). That job first runs the
core and loader tests on Windows (`pytest -m "not gui"`) so the build is gated on
a green non-GUI suite.

## Manual GUI smoke

A headless CI runner cannot exercise a real window, so the automated job only
proves the bundle **assembles**. After downloading the artifact (or building
locally), do a quick manual smoke:

1. Launch the bundled app.
2. Build a simple scenario (e.g. variant **B**, a `sine` excitation) and run it.
3. Confirm the recovered signal and the analysis plots render.
4. Optionally, import one of the bundled `examples/` records to check the file
   pickers and the `io-formats` path.

If the launch fails on a clean machine, it is almost always a missing hidden
import or data file — add it to `packaging/optivibe-gui.spec` and rebuild.
