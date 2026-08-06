import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse, SourceRef
from app.services import chats as chats_svc
from app.services.llm import generate_answer
from app.services.retriever import retrieve
from app.services.store import count_total_chunks

log = logging.getLogger("scrybe.query")
router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(req: QueryRequest):
    if count_total_chunks() == 0:
        raise HTTPException(
            status_code=400,
            detail="No sources indexed yet. Add a URL or file first.",
        )

    import time
    t0 = time.perf_counter()

    chunks = await retrieve(req.question, top_k=req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No matching chunks found")

    try:
        answer = await generate_answer(req.question, chunks)
    except Exception as e:
        log.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}") from e

    seen: set[str] = set()
    sources: list[SourceRef] = []
    for c in chunks:
        sid = c["source_id"]
        if sid in seen:
            continue
        seen.add(sid)
        sources.append(
            SourceRef(source_id=sid, label=c["source_label"], type=c["source_type"])
        )

    latency_ms = int((time.perf_counter() - t0) * 1000)

    if req.chat_id and chats_svc.is_configured():
        try:
            chats_svc.add_message(
                req.chat_id, "user", req.question
            )
            chats_svc.add_message(
                req.chat_id,
                "assistant",
                answer,
                sources=[s.model_dump() for s in sources],
                latency_ms=latency_ms,
                chunks_used=len(chunks),
            )
            # If this looks like a brand-new chat (default title), auto-name it.
            chat = chats_svc.get_chat(req.chat_id)
            if chat and chat.get("title") in (None, "", "New chat") and len(chat.get("messages", [])) <= 2:
                title = req.question.strip().splitlines()[0][:80]
                chats_svc.set_title(req.chat_id, title)
        except Exception:
            log.exception("Failed to persist chat messages — continuing")

    return QueryResponse(
        answer=answer,
        sources=sources,
        chunks_used=len(chunks),
        chat_id=req.chat_id,
    )
