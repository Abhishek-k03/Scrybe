"""Deterministic offline embedder.

Feature-hashes word tokens into a fixed-width vector. Unlike a pure hash of the whole
string this preserves lexical similarity, so retrieval tests exercise real ranking rather
than comparing unrelated random vectors.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from app.rag.config import FakeEmbedConfig
from app.rag.protocols import Embedder, EmbedKind
from app.rag.registry import register

_TOKEN = re.compile(r"[a-z0-9]+")


def _token_bucket(token: str, seed: int, dimensions: int) -> tuple[int, float]:
    digest = hashlib.sha256(f"{seed}\0{token}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % dimensions
    sign = 1.0 if digest[4] & 1 else -1.0
    return bucket, sign


def embed_one(text: str, config: FakeEmbedConfig) -> list[float]:
    vector = [0.0] * config.dimensions
    for token in _TOKEN.findall(text.lower()):
        bucket, sign = _token_bucket(token, config.seed, config.dimensions)
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # Empty or symbol-only text: a fixed unit vector keeps cosine well defined.
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: Sequence[str], config: FakeEmbedConfig) -> list[list[float]]:
    return [embed_one(text, config) for text in texts]


@register("embed", "fake")
def build(config: FakeEmbedConfig) -> Embedder:
    async def embedder(texts: Sequence[str], kind: EmbedKind = "passage") -> list[list[float]]:
        return embed_texts(texts, config)

    return embedder
