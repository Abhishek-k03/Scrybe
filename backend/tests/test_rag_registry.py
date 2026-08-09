"""Registry dispatch and its failure messages."""

from __future__ import annotations

import pytest

from app.rag import registry
from app.rag.config import FixedCharChunkConfig, MemoryIndexConfig


@pytest.fixture(autouse=True)
def _blank_registry():
    """Run each test against an empty registry, then restore the real registrations."""
    saved = {stage: dict(table) for stage, table in registry._REGISTRY.items()}
    registry.clear()
    yield
    for stage, table in saved.items():
        registry._REGISTRY[stage] = table


def test_registered_factory_is_called_with_the_config() -> None:
    seen: list[object] = []

    @registry.register("chunk", "fixed_char")
    def _build(config):
        seen.append(config)
        return "built"

    config = FixedCharChunkConfig(chunk_size=400, overlap=50)
    assert registry.build("chunk", config) == "built"
    assert seen == [config]


def test_dispatch_picks_the_factory_matching_the_kind() -> None:
    registry.register("index", "memory")(lambda config: "memory-index")
    registry.register("index", "chroma")(lambda config: "chroma-index")

    assert registry.build("index", MemoryIndexConfig()) == "memory-index"


def test_unregistered_kind_names_what_is_available() -> None:
    registry.clear("chunk")
    registry.register("chunk", "sentence")(lambda config: None)

    with pytest.raises(registry.UnregisteredKindError, match="sentence"):
        registry.build("chunk", FixedCharChunkConfig())


def test_unregistered_kind_on_empty_stage_is_explicit() -> None:
    registry.clear("chunk")
    with pytest.raises(registry.UnregisteredKindError, match="none registered"):
        registry.build("chunk", FixedCharChunkConfig())


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(registry.UnknownStageError, match="unknown stage"):
        registry.build("summarise", FixedCharChunkConfig())  # type: ignore[arg-type]


def test_config_without_a_kind_is_rejected() -> None:
    class Bare:
        pass

    with pytest.raises(registry.UnregisteredKindError, match="no 'kind'"):
        registry.build("chunk", Bare())


def test_duplicate_registration_is_rejected() -> None:
    registry.clear("rerank")
    registry.register("rerank", "noop")(lambda config: None)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("rerank", "noop")(lambda config: None)


def test_registered_kinds_is_sorted() -> None:
    registry.clear("embed")
    registry.register("embed", "jina")(lambda config: None)
    registry.register("embed", "fake")(lambda config: None)

    assert registry.registered_kinds("embed") == ("fake", "jina")


def test_every_pipeline_stage_has_a_table() -> None:
    for stage in registry.STAGES:
        assert registry.registered_kinds(stage) is not None
