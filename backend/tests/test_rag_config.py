"""Config must round-trip exactly and reject anything ambiguous."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.rag.config import (
    ChromaIndexConfig,
    DenseRetrieveConfig,
    FakeEmbedConfig,
    FixedCharChunkConfig,
    JinaEmbedConfig,
    MemoryIndexConfig,
    MmrRerankConfig,
    PipelineConfig,
    TokenChunkConfig,
)

# --------------------------------------------------------------------------------------
# round-trip
# --------------------------------------------------------------------------------------


def test_default_config_round_trips_through_json() -> None:
    config = PipelineConfig()
    assert PipelineConfig.from_json(config.to_json()) == config


def test_customised_config_round_trips_through_json() -> None:
    config = PipelineConfig(
        chunk=TokenChunkConfig(max_tokens=256, overlap_tokens=32),
        embed=JinaEmbedConfig(batch_size=8),
        index=ChromaIndexConfig(path="/tmp/idx", read_only=False),
        retrieve=DenseRetrieveConfig(top_k=10, fetch_k=50),
        rerank=MmrRerankConfig(lambda_mult=0.7),
    )
    assert PipelineConfig.from_json(config.to_json()) == config


def test_to_json_is_stable_across_calls() -> None:
    """Eval artifacts embed this string; it must not reorder between runs."""
    config = PipelineConfig()
    assert config.to_json() == config.to_json()


def test_serialised_config_records_every_stage_kind() -> None:
    payload = json.loads(PipelineConfig().to_json())
    assert {stage: payload[stage]["kind"] for stage in payload} == {
        "chunk": "fixed_char",
        "embed": "fake",
        "index": "memory",
        "retrieve": "dense",
        "rerank": "noop",
    }


def test_from_file_reads_a_config(tmp_path) -> None:
    config = PipelineConfig(retrieve=DenseRetrieveConfig(top_k=3))
    path = tmp_path / "sweep.json"
    path.write_text(config.to_json(), encoding="utf-8")
    assert PipelineConfig.from_file(path) == config


# --------------------------------------------------------------------------------------
# rejection
# --------------------------------------------------------------------------------------


def test_unknown_top_level_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PipelineConfig.from_json('{"chunker": {"kind": "fixed_char"}}')


def test_unknown_stage_key_is_rejected() -> None:
    """A typo in a sweep file must fail, not silently run the default."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PipelineConfig.from_json('{"chunk": {"kind": "fixed_char", "chunk_sise": 400}}')


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.from_json('{"chunk": {"kind": "semantic"}}')


def test_missing_discriminator_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.from_json('{"chunk": {"chunk_size": 400}}')


def test_configs_are_frozen() -> None:
    config = PipelineConfig()
    with pytest.raises(ValidationError):
        config.chunk = FixedCharChunkConfig()  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# stage validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("chunk_size", "overlap"), [(100, 100), (100, 150), (10, 10)])
def test_fixed_char_rejects_overlap_at_or_above_size(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValidationError, match="chunk_size must be greater than overlap"):
        FixedCharChunkConfig(chunk_size=chunk_size, overlap=overlap)


def test_fixed_char_defaults_match_production() -> None:
    config = FixedCharChunkConfig()
    assert (config.chunk_size, config.overlap, config.stride) == (800, 150, 650)


def test_token_chunk_rejects_overlap_at_or_above_budget() -> None:
    with pytest.raises(ValidationError, match="max_tokens must be greater than overlap_tokens"):
        TokenChunkConfig(max_tokens=64, overlap_tokens=64)


def test_chroma_requires_an_explicit_path() -> None:
    """No default path, so a config can never accidentally address the live index."""
    with pytest.raises(ValidationError):
        ChromaIndexConfig()


def test_chroma_is_read_only_by_default() -> None:
    assert ChromaIndexConfig(path="/tmp/idx").read_only is True


def test_default_pipeline_touches_no_network_or_disk() -> None:
    config = PipelineConfig()
    assert isinstance(config.embed, FakeEmbedConfig)
    assert isinstance(config.index, MemoryIndexConfig)


def test_fetch_k_must_cover_top_k() -> None:
    with pytest.raises(ValidationError, match="fetch_k must be at least top_k"):
        DenseRetrieveConfig(top_k=10, fetch_k=5)


def test_mmr_lambda_is_bounded() -> None:
    with pytest.raises(ValidationError):
        MmrRerankConfig(lambda_mult=1.5)
