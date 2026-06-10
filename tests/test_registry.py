"""Unit tests for the generic Registry."""

from __future__ import annotations

import pytest

from optivibe.core.registry import Registry, RegistryError


def test_register_get_create() -> None:
    reg: Registry[type] = Registry("demo")

    @reg.register("thing")
    class Thing:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    assert "thing" in reg
    assert len(reg) == 1
    assert reg.get("thing") is Thing
    instance = reg.create("thing", value=7)
    assert isinstance(instance, Thing)
    assert instance.value == 7
    assert list(reg.keys()) == ["thing"]


def test_duplicate_key_raises() -> None:
    reg: Registry[type] = Registry("demo")

    @reg.register("dup")
    class _A:
        pass

    with pytest.raises(RegistryError):

        @reg.register("dup")
        class _B:
            pass


def test_unknown_key_raises_with_available() -> None:
    reg: Registry[type] = Registry("demo")

    @reg.register("known")
    class _K:
        pass

    with pytest.raises(RegistryError, match="known"):
        reg.get("missing")
