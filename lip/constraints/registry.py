"""Constraint registry with @register decorator and auto-discovery."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Type

from lip.constraints.base import BaseConstraint

log = logging.getLogger(__name__)

_REGISTRY: dict[str, Type[BaseConstraint]] = {}


def register(name: str):
    """Decorator to register a constraint class.

    Usage:
        @register("property_range")
        class PropertyRangeConstraint(BaseConstraint):
            ...
    """
    def decorator(cls: Type[BaseConstraint]):
        if name in _REGISTRY:
            log.warning(f"Overwriting constraint '{name}' with {cls.__name__}")
        _REGISTRY[name] = cls
        cls._registered_name = name
        return cls
    return decorator


def get_constraint(name: str) -> Type[BaseConstraint] | None:
    """Look up a constraint class by name."""
    _auto_discover()
    return _REGISTRY.get(name)


def list_constraints() -> list[str]:
    """Return list of registered constraint names."""
    _auto_discover()
    return sorted(_REGISTRY.keys())


def create_constraint(
    type_name: str, weight: float = 1.0, params: dict | None = None
) -> BaseConstraint:
    """Create a constraint instance by type name."""
    cls = get_constraint(type_name)
    if cls is None:
        raise ValueError(
            f"Unknown constraint type '{type_name}'. "
            f"Available: {list_constraints()}"
        )
    return cls(weight=weight, params=params)


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

_discovered = False


def _auto_discover():
    """Import all modules in lip.constraints to trigger @register decorators."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    package = importlib.import_module("lip.constraints")
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if modname in ("base", "registry"):
            continue
        try:
            importlib.import_module(f"lip.constraints.{modname}")
        except Exception as e:
            log.debug(f"Failed to import lip.constraints.{modname}: {e}")
