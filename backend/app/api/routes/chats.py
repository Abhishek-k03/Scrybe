import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatCreateRequest, ChatRenameRequest, ChatStatus
from app.services import chats as chats_svc

log = logging.getLogger("weboracle.chats")
router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("/status", response_model=ChatStatus)
async def status():
    return ChatStatus(configured=chats_svc.is_configured())


def _require_configured():
    if not chats_svc.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in backend/.env.",
        )


@router.post("")
async def create(req: ChatCreateRequest | None = None):
    _require_configured()
    title = (req.title if req and req.title else "New chat").strip() or "New chat"
    try:
        return chats_svc.create_chat(title=title)
    except Exception as e:
        log.exception("create_chat failed")
        raise HTTPException(status_code=502, detail=str(e))


@router.get("")
async def list_all():
    _require_configured()
    try:
        return chats_svc.list_chats()
    except Exception as e:
        log.exception("list_chats failed")
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/{chat_id}")
async def get_one(chat_id: str):
    _require_configured()
    try:
        chat = chats_svc.get_chat(chat_id)
    except Exception as e:
        log.exception("get_chat failed")
        raise HTTPException(status_code=502, detail=str(e))
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.patch("/{chat_id}")
async def rename(chat_id: str, req: ChatRenameRequest):
    _require_configured()
    try:
        chats_svc.set_title(chat_id, req.title)
        return {"ok": True}
    except Exception as e:
        log.exception("rename failed")
        raise HTTPException(status_code=502, detail=str(e))


@router.delete("/{chat_id}")
async def remove(chat_id: str):
    _require_configured()
    try:
        chats_svc.delete_chat(chat_id)
        return {"ok": True, "chat_id": chat_id}
    except Exception as e:
        log.exception("delete failed")
        raise HTTPException(status_code=502, detail=str(e))
