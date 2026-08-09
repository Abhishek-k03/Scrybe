"""Token-budget chunking.

The tokenizer is injectable so the packing logic can be tested without tiktoken, which
downloads its encoding files on first use.
"""

from __future__ import annotations

from typing import Protocol

from app.rag.config import TokenChunkConfig
from app.rag.protocols import Chunker
from app.rag.registry import register
from app.rag.types import Chunk, Document


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


def load_tiktoken(encoding: str) -> Tokenizer:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"token chunking needs tiktoken (encoding {encoding!r}); pip install tiktoken"
        ) from exc
    return tiktoken.get_encoding(encoding)


def chunk_tokens(doc: Document, config: TokenChunkConfig, tokenizer: Tokenizer) -> list[Chunk]:
    text = doc.text
    if not text:
        return []

    tokens = tokenizer.encode(text)
    if not tokens:
        return []

    stride = config.max_tokens - config.overlap_tokens
    prefix_len: dict[int, int] = {0: 0, len(tokens): len(text)}

    def char_offset(token_index: int) -> int:
        if token_index not in prefix_len:
            prefix_len[token_index] = len(tokenizer.decode(tokens[:token_index]))
        return prefix_len[token_index]

    chunks: list[Chunk] = []
    cursor = 0

    while cursor < len(tokens):
        stop = min(cursor + config.max_tokens, len(tokens))
        raw = text[char_offset(cursor) : char_offset(stop)]
        stripped = raw.strip()
        if stripped:
            start = char_offset(cursor) + (len(raw) - len(raw.lstrip()))
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    doc_label=doc.label,
                    source_type=doc.source_type,
                    chunk_index=len(chunks),
                    text=stripped,
                    start_char=start,
                    end_char=start + len(stripped),
                )
            )
        if stop >= len(tokens):
            break
        cursor += stride

    return chunks


@register("chunk", "token")
def build(config: TokenChunkConfig) -> Chunker:
    cached: list[Tokenizer] = []

    def chunker(doc: Document) -> list[Chunk]:
        if not cached:
            cached.append(load_tiktoken(config.encoding))
        return chunk_tokens(doc, config, cached[0])

    return chunker
