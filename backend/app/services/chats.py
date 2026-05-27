import logging
from typing import Any, Optional

from supabase import Client, create_client

from app.core.config import settings

log = logging.getLogger("scrybe.chats")

_client: Optional[Client] = None


def supabase() -> Client:
    global _client
    if _client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError(
                "Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in backend/.env"
            )
        _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        log.info("Supabase client ready (%s)", settings.SUPABASE_URL)
    return _client


def is_configured() -> bool:
    return bool(settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY)


def create_chat(title: str = "New chat") -> dict[str, Any]:
    res = supabase().table("chats").insert({"title": title}).execute()
    return res.data[0]


def list_chats() -> list[dict[str, Any]]:
    chats = (
        supabase()
        .table("chats")
        .select("id, title, created_at, updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    rows = chats.data or []
    if not rows:
        return []

    ids = [r["id"] for r in rows]
    msgs = (
        supabase()
        .table("messages")
        .select("chat_id")
        .in_("chat_id", ids)
        .execute()
    )
    counts: dict[str, int] = {}
    for m in msgs.data or []:
        counts[m["chat_id"]] = counts.get(m["chat_id"], 0) + 1

    for r in rows:
        r["message_count"] = counts.get(r["id"], 0)
    return rows


def get_chat(chat_id: str) -> Optional[dict[str, Any]]:
    chat = (
        supabase()
        .table("chats")
        .select("id, title, created_at, updated_at")
        .eq("id", chat_id)
        .maybe_single()
        .execute()
    )
    if not chat or not chat.data:
        return None
    messages = (
        supabase()
        .table("messages")
        .select("id, role, content, sources, latency_ms, chunks_used, error, created_at")
        .eq("chat_id", chat_id)
        .order("created_at", desc=False)
        .execute()
    )
    return {**chat.data, "messages": messages.data or []}


def add_message(
    chat_id: str,
    role: str,
    content: str,
    sources: Optional[list[dict[str, Any]]] = None,
    latency_ms: Optional[int] = None,
    chunks_used: Optional[int] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    payload = {
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "sources": sources or [],
    }
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    if chunks_used is not None:
        payload["chunks_used"] = chunks_used
    if error is not None:
        payload["error"] = error

    res = supabase().table("messages").insert(payload).execute()
    supabase().table("chats").update({"updated_at": "now()"}).eq("id", chat_id).execute()
    return res.data[0]


def set_title(chat_id: str, title: str) -> None:
    supabase().table("chats").update({"title": title[:120]}).eq("id", chat_id).execute()


def delete_chat(chat_id: str) -> None:
    supabase().table("chats").delete().eq("id", chat_id).execute()
