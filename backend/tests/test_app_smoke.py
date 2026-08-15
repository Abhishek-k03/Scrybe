"""The FastAPI app end to end over the refactored pipeline.

Runs against a temp index with the offline embedder and a stubbed LLM, so a full
ingest-then-answer round trip is exercised without a network or the live index.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.rag.config import (
    DenseRetrieveConfig,
    FakeEmbedConfig,
    FixedCharChunkConfig,
    NoopRerankConfig,
    PipelineConfig,
)

DOC = (
    "Python uses reference counting to free memory. When an object's count reaches zero it "
    "is deallocated immediately. A separate cycle detector reclaims groups of objects that "
    "refer to each other. The global interpreter lock serialises bytecode execution, so "
    "threads do not run Python code in parallel."
)


@pytest.fixture
def client(monkeypatch, _isolated_chroma) -> Iterator[TestClient]:
    """An app whose pipeline embeds offline and whose index is the per-test temp dir."""
    from app.main import app
    from app.services import pipeline

    real_config = pipeline.default_config

    def offline_config() -> PipelineConfig:
        return PipelineConfig(
            chunk=FixedCharChunkConfig(chunk_size=120, overlap=20),
            embed=FakeEmbedConfig(dimensions=128),
            # The real index config, so this still points at the isolated temp directory.
            index=real_config().index,
            retrieve=DenseRetrieveConfig(top_k=5),
            rerank=NoopRerankConfig(),
        )

    monkeypatch.setattr(pipeline, "default_config", offline_config)
    pipeline.reset()

    async def fake_answer(question, chunks, cache=None):
        return f"Answered {question!r} from {len(chunks)} chunks."

    monkeypatch.setattr("app.api.routes.query.generate_answer", fake_answer)

    with TestClient(app) as test_client:
        yield test_client

    pipeline.reset()


def upload(client: TestClient, name: str = "python.txt", body: str = DOC):
    return client.post(
        "/api/ingest/file",
        files={"file": (name, body.encode("utf-8"), "text/plain")},
    )


# --------------------------------------------------------------------------------------
# boot
# --------------------------------------------------------------------------------------


def test_health_is_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_every_router_is_mounted(client: TestClient) -> None:
    paths = {route.path for route in client.app.routes}
    for expected in ("/api/query", "/api/ingest/url", "/api/sources", "/api/vector_map"):
        assert expected in paths


# --------------------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------------------


def test_uploading_a_text_file_indexes_it(client: TestClient) -> None:
    response = upload(client)
    assert response.status_code == 200

    body = response.json()
    assert body["source_label"] == "python.txt"
    assert body["source_type"] == "file"
    assert body["chunks_stored"] > 1


def test_the_source_id_is_a_content_hash_not_a_uuid(client: TestClient) -> None:
    first = upload(client).json()["source_id"]

    from app.rag.types import DOC_ID_LENGTH, make_doc_id

    assert len(first) == DOC_ID_LENGTH
    assert first == make_doc_id("python.txt", DOC)


def test_uploading_the_same_file_twice_stores_nothing_the_second_time(
    client: TestClient,
) -> None:
    first = upload(client).json()
    second = upload(client).json()

    assert second["source_id"] == first["source_id"]
    assert second["chunks_stored"] == 0
    assert client.get("/api/sources").json()[0]["chunk_count"] == first["chunks_stored"]


def test_an_edited_file_is_indexed_separately(client: TestClient) -> None:
    first = upload(client).json()
    second = upload(client, body=DOC + " Edited.").json()

    assert second["source_id"] != first["source_id"]
    assert second["chunks_stored"] > 0


def test_an_unsupported_extension_is_rejected(client: TestClient) -> None:
    response = upload(client, name="archive.zip", body="x")
    assert response.status_code == 400
    assert "pdf" in response.json()["detail"]


def test_an_empty_upload_is_rejected(client: TestClient) -> None:
    assert upload(client, body="").status_code == 400


def test_a_whitespace_only_upload_is_rejected(client: TestClient) -> None:
    response = upload(client, body="   \n\t  ")
    assert response.status_code == 400
    assert "No extractable text" in response.json()["detail"]


# --------------------------------------------------------------------------------------
# query
# --------------------------------------------------------------------------------------


def test_querying_an_empty_index_asks_for_a_source_first(client: TestClient) -> None:
    response = client.post("/api/query", json={"question": "anything"})
    assert response.status_code == 400
    assert "No sources indexed" in response.json()["detail"]


def test_query_returns_an_answer_and_its_sources(client: TestClient) -> None:
    upload(client)
    response = client.post("/api/query", json={"question": "how does python free memory"})

    assert response.status_code == 200
    body = response.json()
    assert body["chunks_used"] > 0
    assert [source["label"] for source in body["sources"]] == ["python.txt"]


def test_query_respects_top_k(client: TestClient) -> None:
    upload(client)
    response = client.post("/api/query", json={"question": "python", "top_k": 2})
    assert response.json()["chunks_used"] == 2


def test_sources_are_deduplicated_across_chunks(client: TestClient) -> None:
    """Several chunks of one document must be cited once, not once per chunk."""
    stored = upload(client).json()["chunks_stored"]
    body = client.post("/api/query", json={"question": "python", "top_k": stored}).json()

    assert body["chunks_used"] == stored > 1
    assert len(body["sources"]) == 1


def test_an_empty_question_is_rejected_by_the_schema(client: TestClient) -> None:
    upload(client)
    assert client.post("/api/query", json={"question": ""}).status_code == 422


# --------------------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------------------


def test_sources_lists_what_was_ingested(client: TestClient) -> None:
    stored = upload(client).json()
    listing = client.get("/api/sources").json()

    assert listing == [
        {
            "source_id": stored["source_id"],
            "source_label": "python.txt",
            "source_type": "file",
            "chunk_count": stored["chunks_stored"],
        }
    ]


def test_deleting_a_source_removes_its_chunks(client: TestClient) -> None:
    stored = upload(client).json()
    response = client.delete(f"/api/sources/{stored['source_id']}")

    assert response.json()["deleted_chunks"] == stored["chunks_stored"]
    assert client.get("/api/sources").json() == []


def test_deleting_an_unknown_source_removes_nothing(client: TestClient) -> None:
    upload(client)
    assert client.delete("/api/sources/nope").json()["deleted_chunks"] == 0


# --------------------------------------------------------------------------------------
# vector map
# --------------------------------------------------------------------------------------


def test_vector_map_has_a_point_per_chunk(client: TestClient) -> None:
    stored = upload(client).json()
    body = client.get("/api/vector_map").json()

    assert body["point_count"] == stored["chunks_stored"]
    assert body["dim"] == 128


def test_vector_map_query_marks_the_same_chunks_retrieval_returned(client: TestClient) -> None:
    """The map used to run its own search and could disagree with the answer path."""
    upload(client)
    question = "how does python free memory"

    mapped = client.post("/api/vector_map/query", json={"question": question, "top_k": 3}).json()
    answered = client.post("/api/query", json={"question": question, "top_k": 3}).json()

    assert len(mapped["hits"]) == answered["chunks_used"] == 3
    assert mapped["query_point"] is not None


def test_vector_map_query_on_an_empty_index_returns_no_point(client: TestClient) -> None:
    body = client.post("/api/vector_map/query", json={"question": "anything"}).json()
    assert body["hits"] == []
