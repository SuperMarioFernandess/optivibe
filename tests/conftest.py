"""Shared pytest fixtures and headless-Qt setup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Make any Qt that gets imported run head-less (no display needed in CI).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def config_dir(repo_root: Path) -> Path:
    """Absolute path to the ``configs/`` directory."""
    return repo_root / "configs"


@pytest.fixture(scope="session")
def examples_dir(repo_root: Path) -> Path:
    """Absolute path to the ``examples/`` directory."""
    return repo_root / "examples"


@pytest.fixture(scope="session")
def hello_scenario(examples_dir: Path) -> Path:
    """Absolute path to the hello scenario."""
    return examples_dir / "hello.yaml"
