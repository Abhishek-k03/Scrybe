"""Maps a config's `kind` to the factory that builds that stage.

Implementations register themselves on import; the registry imports none of them, so adding
a stage variant never means editing this file.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeVar

Stage = Literal["chunk", "embed", "index", "retrieve", "rerank"]

STAGES: tuple[Stage, ...] = ("chunk", "embed", "index", "retrieve", "rerank")

Factory = Callable[[Any], Any]
F = TypeVar("F", bound=Factory)

_REGISTRY: dict[Stage, dict[str, Factory]] = {stage: {} for stage in STAGES}


class UnknownStageError(KeyError):
    """Raised for a stage name that is not part of the pipeline."""


class UnregisteredKindError(KeyError):
    """Raised when a config names a variant that nothing has registered."""


def _stage_table(stage: str) -> dict[str, Factory]:
    if stage not in _REGISTRY:
        raise UnknownStageError(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    return _REGISTRY[stage]  # type: ignore[index]


def register(stage: Stage, kind: str) -> Callable[[F], F]:
    """Register a factory for one variant of a stage."""

    def decorator(factory: F) -> F:
        table = _stage_table(stage)
        if kind in table:
            raise ValueError(f"{stage} kind {kind!r} is already registered")
        table[kind] = factory
        return factory

    return decorator


def build(stage: Stage, config: Any, **kwargs: Any) -> Any:
    """Build a stage from its config, dispatching on `config.kind`.

    Extra keyword arguments are forwarded to the factory. Secrets travel this way rather
    than through the config, which gets serialised into eval artifacts.
    """
    table = _stage_table(stage)
    kind = getattr(config, "kind", None)
    if kind is None:
        raise UnregisteredKindError(f"{stage} config has no 'kind' field: {config!r}")
    if kind not in table:
        known = sorted(table) or ["<none registered>"]
        raise UnregisteredKindError(f"no {stage} registered for kind {kind!r}; known: {known}")
    return table[kind](config, **kwargs)


def registered_kinds(stage: Stage) -> tuple[str, ...]:
    return tuple(sorted(_stage_table(stage)))


def clear(stage: Stage | None = None) -> None:
    """Drop registrations. Intended for tests."""
    if stage is None:
        for table in _REGISTRY.values():
            table.clear()
        return
    _stage_table(stage).clear()
