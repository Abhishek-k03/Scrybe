import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import VectorMapResponse, VectorQueryRequest, VectorQueryResponse
from app.services import vector_map

log = logging.getLogger("weboracle.vector_map")
router = APIRouter(prefix="/vector_map", tags=["vector_map"])


@router.get("", response_model=VectorMapResponse)
async def get_map():
    try:
        return vector_map.get_map()
    except Exception as e:
        log.exception("vector_map build failed")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/query", response_model=VectorQueryResponse)
async def query_map(req: VectorQueryRequest):
    try:
        return await vector_map.project_query(req.question, top_k=req.top_k)
    except Exception as e:
        log.exception("vector_map query failed")
        raise HTTPException(status_code=502, detail=str(e))
