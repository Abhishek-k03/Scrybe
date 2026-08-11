import logging

from groq import AsyncGroq

from app.core.config import settings
from app.rag.cache import DiskCache, content_key

log = logging.getLogger("scrybe.llm")

MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0
SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "provided context. If the answer is not in the context, say "
    "\"I couldn't find information about this in your sources.\" "
    "Always be concise and factual. When you use information from a source, "
    "you may reference it by its source label in parentheses."
)


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        label = c.get("source_label", "unknown")
        text = c.get("text", "")
        parts.append(f"[Source {i} — {label}]\n{text}")
    return "\n\n---\n\n".join(parts)


async def generate_answer(
    question: str,
    context_chunks: list[dict],
    cache: DiskCache | None = None,
) -> str:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    if not context_chunks:
        return "I couldn't find information about this in your sources."

    context = _format_context(context_chunks)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )

    key = content_key(MODEL, str(TEMPERATURE), SYSTEM_PROMPT, user_message)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return str(cached)

    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    resp = await client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = (resp.choices[0].message.content or "").strip()

    if cache is not None:
        cache.set(key, answer)

    log.info("Generated answer (%d chars) using %d context chunks", len(answer), len(context_chunks))
    return answer
