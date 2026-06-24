# PyInstaller spec for the OptiVibe desktop app (task S8, decision SW-38).
#
# One-dir bundle of the PySide6/PyQtGraph GUI. Windows-first: the CI build job
# runs on windows-latest, but this spec is host-portable (PyInstaller always
# targets the OS it runs on), so it also builds a Linux/macOS bundle locally.
#
# Build from the repo root (after `pip install -e '.[gui,io-formats,packaging]'`):
#     pyinstaller packaging/optivibe-gui.spec --noconfirm
# The result is dist/OptiVibe/ (run dist/OptiVibe/OptiVibe[.exe]).
#
# GUI smoke must be MANUAL: a headless CI runner cannot exercise the window, so
# the build job only proves the bundle assembles. See docs/packaging.md.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# `SPECPATH` is injected by PyInstaller; the repo root is its parent.
ROOT = Path(SPECPATH).resolve().parent

# Ship the runnable scenarios and constants next to the binary so a user can run
# the bundled examples without a source checkout. The physics core reads these
# at runtime (configs/constants.yaml, configs/variants/*, examples/*).
datas = [
    (str(ROOT / "configs"), "configs"),
    (str(ROOT / "examples"), "examples"),
]
# matplotlib (used by the viz layer) ships data files it loads at runtime.
datas += collect_data_files("matplotlib", includes=["mpl-data/**"])

# The stage registries register implementations by importing their modules, so
# pull the whole package in rather than relying on static import discovery.
hiddenimports = collect_submodules("optivibe")
# Backends that are imported lazily inside loaders/plots (PyInstaller cannot see
# a deferred ``import`` in a function body).
hiddenimports += [
    "scipy.signal",
    "scipy.special",
    "scipy.io",
    "pyqtgraph",
    "nptdms",
    "pyuff",
    "h5py",
]

block_cipher = None

a = Analysis(
    [str(ROOT / "packaging" / "optivibe_gui_launch.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Keep the test/dev surface out of the shipped app.
    excludes=["pytest", "hypothesis", "mypy", "ruff", "tests", "IPython", "tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OptiVibe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed GUI app (no console on Windows)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OptiVibe",
)
