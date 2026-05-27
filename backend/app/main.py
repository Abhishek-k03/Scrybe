import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import chats, ingest, query, sources, vector_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scrybe")

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    log.info("%s v%s starting", settings.PROJECT_NAME, settings.VERSION)
    log.info("Groq key configured: %s", bool(settings.GROQ_API_KEY))
    log.info("Jina key configured: %s", bool(settings.JINA_API_KEY))
    log.info("Supabase configured: %s", bool(settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY))


@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.VERSION}


app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(sources.router, prefix="/api")
app.include_router(chats.router, prefix="/api")
app.include_router(vector_map.router, prefix="/api")
