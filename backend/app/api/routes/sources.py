from fastapi import APIRouter

from app.api.schemas import DeleteResponse, SourceItem
from app.services import vector_map
from app.services.store import delete_source, get_all_sources

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("", response_model=list[SourceItem])
async def list_sources():
    return get_all_sources()


@router.delete("/{source_id}", response_model=DeleteResponse)
async def remove_source(source_id: str):
    n = delete_source(source_id)
    vector_map.invalidate()
    return DeleteResponse(source_id=source_id, deleted_chunks=n)
