import logging
from threading import Lock
from typing import Any, Optional

import numpy as np

from app.services.embedder import embed_texts
from app.services.store import _get_collection

log = logging.getLogger("weboracle.vector_map")

# Fixed palette cycled through per-source.
PALETTE = [
    "#67e8f9",  # cyan
    "#34d399",  # emerald
    "#a78bfa",  # violet
    "#fbbf24",  # amber
    "#f472b6",  # pink
    "#60a5fa",  # blue
    "#f87171",  # rose
    "#facc15",  # yellow
    "#4ade80",  # green
    "#c084fc",  # purple
]

VIEW_W = 900
VIEW_H = 520

_cache: dict[str, Any] = {
    "n_chunks": -1,
    "points": None,
    "sources": None,
    "mean": None,
    "components": None,  # shape (2, dim)
    "scale": None,       # (x_min, x_range, y_min, y_range)
    "dim": 0,
}
_lock = Lock()


def _fit_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, components_2xD) by SVD on centered matrix."""
    mean = X.mean(axis=0)
    Xc = X - mean
    # full_matrices=False so Vt is (min(n,d), d)
    _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:2]  # (2, dim)
    return mean, components


def _project(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (X - mean) @ components.T  # (n, 2)


def _scale_to_view(coords: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Scale 2D coords into the SVG viewBox with padding. Returns (scaled, (x_min, x_range, y_min, y_range))."""
    if coords.size == 0:
        return coords, (0.0, 1.0, 0.0, 1.0)
    pad = 0.08
    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)
    sx = (x - x_min) / x_range
    sy = (y - y_min) / y_range
    sx = sx * (1 - 2 * pad) + pad
    sy = sy * (1 - 2 * pad) + pad
    scaled = np.stack([sx * VIEW_W, sy * VIEW_H], axis=1)
    return scaled, (x_min, x_range, y_min, y_range)


def _apply_view_scale(coords: np.ndarray, scale: tuple[float, float, float, float]) -> np.ndarray:
    pad = 0.08
    x_min, x_range, y_min, y_range = scale
    sx = (coords[:, 0] - x_min) / x_range
    sy = (coords[:, 1] - y_min) / y_range
    sx = sx * (1 - 2 * pad) + pad
    sy = sy * (1 - 2 * pad) + pad
    return np.stack([sx * VIEW_W, sy * VIEW_H], axis=1)


def _build(force: bool = False) -> dict[str, Any]:
    coll = _get_collection()
    n = coll.count()

    if not force and _cache["n_chunks"] == n and _cache["points"] is not None:
        return _cache

    with _lock:
        if not force and _cache["n_chunks"] == n and _cache["points"] is not None:
            return _cache

        if n == 0:
            _cache.update({
                "n_chunks": 0,
                "points": [],
                "sources": [],
                "mean": None,
                "components": None,
                "scale": None,
                "dim": 0,
            })
            return _cache

        result = coll.get(include=["embeddings", "documents", "metadatas"])
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = []
        documents = result.get("documents")
        if documents is None:
            documents = []
        metadatas = result.get("metadatas")
        if metadatas is None:
            metadatas = []
        ids = result.get("ids")
        if ids is None:
            ids = []

        if len(embeddings) == 0:
            _cache.update({
                "n_chunks": 0, "points": [], "sources": [],
                "mean": None, "components": None, "scale": None, "dim": 0,
            })
            return _cache

        X = np.asarray(embeddings, dtype=np.float64)
        dim = X.shape[1]

        if X.shape[0] >= 2:
            mean, components = _fit_pca(X)
            coords = _project(X, mean, components)
        else:
            # Single point: skip PCA, place at center.
            mean = X[0].copy()
            components = np.zeros((2, dim))
            coords = np.zeros((1, 2))

        scaled, scale = _scale_to_view(coords)

        # Per-source color assignment (deterministic).
        seen_ids: list[str] = []
        for m in metadatas:
            sid = (m or {}).get("source_id", "unknown")
            if sid not in seen_ids:
                seen_ids.append(sid)
        color_for: dict[str, str] = {sid: PALETTE[i % len(PALETTE)] for i, sid in enumerate(seen_ids)}

        source_counts: dict[str, int] = {}
        source_labels: dict[str, str] = {}
        source_types: dict[str, str] = {}

        points: list[dict[str, Any]] = []
        for chunk_id, meta, doc, xy in zip(ids, metadatas, documents, scaled):
            m = meta or {}
            sid = m.get("source_id", "unknown")
            source_counts[sid] = source_counts.get(sid, 0) + 1
            source_labels[sid] = m.get("source_label", sid)
            source_types[sid] = m.get("source_type", "unknown")
            preview = (doc or "").strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "…"
            points.append({
                "id": chunk_id,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "source_id": sid,
                "source_label": m.get("source_label", sid),
                "source_type": m.get("source_type", "unknown"),
                "chunk_index": int(m.get("chunk_index", 0)),
                "text_preview": preview,
            })

        sources = [
            {
                "source_id": sid,
                "source_label": source_labels[sid],
                "source_type": source_types[sid],
                "color": color_for[sid],
                "count": source_counts[sid],
            }
            for sid in seen_ids
        ]

        _cache.update({
            "n_chunks": n,
            "points": points,
            "sources": sources,
            "mean": mean,
            "components": components,
            "scale": scale,
            "dim": dim,
            "color_for": color_for,
        })
        log.info("Built vector map: %d points, %d sources, dim=%d", n, len(sources), dim)
        return _cache


def get_map() -> dict[str, Any]:
    state = _build()
    return {
        "points": state["points"],
        "sources": state["sources"],
        "dim": state["dim"],
        "point_count": len(state["points"]),
    }


async def project_query(question: str, top_k: int = 6) -> dict[str, Any]:
    state = _build()
    if not state["points"]:
        return {"query_point": None, "hits": []}

    embeddings = await embed_texts([question], task="retrieval.query")
    if not embeddings:
        return {"query_point": None, "hits": []}

    q_vec = np.asarray(embeddings[0], dtype=np.float64)
    mean = state["mean"]
    components = state["components"]
    scale = state["scale"]

    q_proj = ((q_vec - mean) @ components.T).reshape(1, 2)
    q_scaled = _apply_view_scale(q_proj, scale)[0]

    # Cosine distance against every cached chunk: re-fetch raw embeddings cheaply by re-querying ChromaDB.
    coll = _get_collection()
    res = coll.query(
        query_embeddings=[embeddings[0]],
        n_results=min(top_k, state["n_chunks"]),
        include=["documents", "metadatas", "distances"],
    )
    hit_ids = (res.get("ids") or [[]])[0]
    hit_docs = (res.get("documents") or [[]])[0]
    hit_metas = (res.get("metadatas") or [[]])[0]
    hit_dists = (res.get("distances") or [[]])[0]

    point_by_id = {p["id"]: p for p in state["points"]}

    hits: list[dict[str, Any]] = []
    for hid, doc, meta, dist in zip(hit_ids, hit_docs, hit_metas, hit_dists):
        cached = point_by_id.get(hid)
        if not cached:
            continue
        preview = (doc or "").strip().replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "…"
        hits.append({
            "id": hid,
            "x": cached["x"],
            "y": cached["y"],
            "source_id": cached["source_id"],
            "source_label": cached["source_label"],
            "source_type": cached["source_type"],
            "chunk_index": cached["chunk_index"],
            "distance": float(dist),
            "text_preview": preview,
        })

    return {
        "query_point": {"x": float(q_scaled[0]), "y": float(q_scaled[1])},
        "hits": hits,
    }


def invalidate() -> None:
    _cache["n_chunks"] = -1
