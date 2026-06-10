"""Registry of swappable implementations selected by a config key.

Architecture decision SW-02 (doc 09 §6): each pipeline stage is a typed
``Stage`` with a contract on input and output; concrete implementations
register themselves in a registry and are selected by a string key coming from
the scenario/variant configuration. Adding a new reflector, light source, DSP
method or data loader is therefore a *new adapter with a registration*, never a
change to the core.

A registry is generic over the registered type ``T`` so the same machinery
serves every stage family (excitation, mechanics, optics, detector, dsp) and the
data-loader registry (doc 09 §4, SW-08).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised when a key is registered twice or requested but absent."""


class Registry(Generic[T]):
    """A name-to-factory registry for one family of swappable implementations.

    Parameters
    ----------
    name : str
        Human-readable family name (e.g. ``"optics.reflector"``), used in error
        messages and logs.

    Examples
    --------
    >>> reflectors: Registry[type] = Registry("optics.reflector")
    >>> @reflectors.register("cylinder")
    ... class CylinderReflector:
    ...     pass
    >>> reflectors.get("cylinder") is CylinderReflector
    True
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._factories: dict[str, Callable[..., T]] = {}

    @property
    def name(self) -> str:
        """Family name of this registry."""
        return self._name

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Return a decorator that registers a factory under ``key``.

        Parameters
        ----------
        key : str
            Selection key used in configuration files.

        Returns
        -------
        Callable
            Decorator returning its argument unchanged (so the decorated class or
            function keeps its identity).

        Raises
        ------
        RegistryError
            If ``key`` is already registered in this family.
        """

        def _decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if key in self._factories:
                msg = f"{self._name}: key {key!r} already registered"
                raise RegistryError(msg)
            self._factories[key] = factory
            return factory

        return _decorator

    def get(self, key: str) -> Callable[..., T]:
        """Return the factory registered under ``key``.

        Parameters
        ----------
        key : str
            Selection key.

        Returns
        -------
        Callable
            The registered factory (class or function).

        Raises
        ------
        RegistryError
            If ``key`` is not registered. The message lists the available keys.
        """
        try:
            return self._factories[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            msg = f"{self._name}: unknown key {key!r}; available: {available}"
            raise RegistryError(msg) from exc

    def create(self, key: str, /, *args: object, **kwargs: object) -> T:
        """Instantiate the implementation registered under ``key``.

        Parameters
        ----------
        key : str
            Selection key.
        *args, **kwargs
            Forwarded to the registered factory.

        Returns
        -------
        T
            The constructed implementation.
        """
        return self.get(key)(*args, **kwargs)

    def keys(self) -> Iterable[str]:
        """Return the registered keys (sorted)."""
        return sorted(self._factories)

    def __contains__(self, key: object) -> bool:
        return key in self._factories

    def __len__(self) -> int:
        return len(self._factories)
