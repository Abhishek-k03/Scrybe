"""Turning URLs, uploads and corpus directories into documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.ingest.file import (
    ALLOWED_EXTENSIONS,
    FileParseError,
    UnsupportedFileError,
    document_from_file,
    parse_bytes,
    parse_pdf,
    parse_txt,
)
from app.rag.ingest.local import load_directory, load_file
from app.rag.ingest.url import document_from_url, html_to_text
from app.rag.types import make_doc_id


def minimal_pdf(text: str) -> bytes:
    """A one-page PDF containing `text`, built by hand to avoid a writer dependency."""
    content = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


# --------------------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------------------


def test_pdf_text_is_extracted() -> None:
    assert "Reference counting" in parse_pdf(minimal_pdf("Reference counting frees memory"))


def test_unreadable_pdf_raises_rather_than_returning_empty_text() -> None:
    """Silently indexing an empty document would look like a page with no content."""
    with pytest.raises(FileParseError, match="Failed to parse PDF"):
        parse_pdf(b"this is not a pdf")


def test_utf8_text_round_trips() -> None:
    assert parse_txt("héllo wörld".encode()) == "héllo wörld"


def test_undecodable_bytes_are_dropped_not_fatal() -> None:
    assert parse_txt(b"ok \xff\xfe tail") == "ok  tail"


def test_extension_dispatch_is_case_insensitive() -> None:
    assert parse_bytes("NOTES.TXT", b"hello") == "hello"
    assert "Hi" in parse_bytes("SCAN.PDF", minimal_pdf("Hi"))


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(UnsupportedFileError, match="Only"):
        parse_bytes("archive.zip", b"PK\x03\x04")


def test_allowed_extensions_are_the_two_the_route_advertises() -> None:
    assert set(ALLOWED_EXTENSIONS) == {".pdf", ".txt"}


def test_document_from_file_labels_with_the_filename() -> None:
    doc = document_from_file("notes.txt", b"some text")
    assert (doc.label, doc.source_type, doc.text) == ("notes.txt", "file", "some text")


def test_identical_uploads_get_the_same_id() -> None:
    """Content-hashed ids are what make a repeat ingest a no-op."""
    first = document_from_file("notes.txt", b"same bytes")
    second = document_from_file("notes.txt", b"same bytes")
    assert first.doc_id == second.doc_id


def test_changed_content_gets_a_new_id() -> None:
    first = document_from_file("notes.txt", b"version one")
    second = document_from_file("notes.txt", b"version two")
    assert first.doc_id != second.doc_id


def test_same_content_under_a_different_name_is_a_different_document() -> None:
    a = document_from_file("a.txt", b"shared")
    b = document_from_file("b.txt", b"shared")
    assert a.doc_id != b.doc_id


# --------------------------------------------------------------------------------------
# urls
# --------------------------------------------------------------------------------------


HTML = """
<html>
  <head><title>T</title><style>.a { color: red }</style></head>
  <body>
    <nav>Home About Contact</nav>
    <h1>Reference counting</h1>
    <p>CPython    frees   an object when its count reaches zero.</p>
    <script>console.log("tracking")</script>
    <footer>© 2026</footer>
  </body>
</html>
"""


def test_chrome_and_scripts_are_stripped() -> None:
    text = html_to_text(HTML)
    for noise in ("Home About Contact", "console.log", "color: red", "© 2026"):
        assert noise not in text


def test_body_content_survives() -> None:
    text = html_to_text(HTML)
    assert "Reference counting" in text
    assert "frees an object when its count reaches zero" in text


def test_runs_of_spaces_are_collapsed() -> None:
    assert "  " not in html_to_text(HTML)


def test_blank_lines_are_dropped() -> None:
    assert "" not in html_to_text(HTML).splitlines()


def test_empty_html_yields_empty_text() -> None:
    assert html_to_text("") == ""


async def test_document_from_url_uses_the_injected_fetcher() -> None:
    """Extraction is testable without launching a browser."""

    async def fake_fetch(url: str, timeout_ms: int) -> str:
        return f"<html><body><p>Fetched {url}</p></body></html>"

    doc = await document_from_url("https://example.com/page", fetch=fake_fetch)

    assert doc.source_type == "url"
    assert doc.label == "https://example.com/page"
    assert "Fetched https://example.com/page" in doc.text


async def test_url_document_id_is_the_hash_of_label_and_text() -> None:
    async def fake_fetch(url: str, timeout_ms: int) -> str:
        return "<html><body>body text</body></html>"

    doc = await document_from_url("https://example.com", fetch=fake_fetch)
    assert doc.doc_id == make_doc_id("https://example.com", doc.text)


# --------------------------------------------------------------------------------------
# local corpus
# --------------------------------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for name, body in [("b.txt", "beta"), ("a.txt", "alpha"), ("c.md", "gamma")]:
        (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_directory_loads_only_the_matching_pattern(corpus: Path) -> None:
    assert [doc.label for doc in load_directory(corpus)] == ["a.txt", "b.txt"]


def test_directory_order_is_sorted_not_filesystem_order(corpus: Path) -> None:
    """An eval run must see the same corpus in the same order on every machine."""
    labels = [doc.label for doc in load_directory(corpus)]
    assert labels == sorted(labels)


def test_directory_pattern_is_configurable(corpus: Path) -> None:
    assert [doc.label for doc in load_directory(corpus, "*.md")] == ["c.md"]


def test_directory_documents_carry_the_corpus_source_type(corpus: Path) -> None:
    assert {doc.source_type for doc in load_directory(corpus)} == {"corpus"}


def test_loading_twice_produces_identical_ids(corpus: Path) -> None:
    first = [doc.doc_id for doc in load_directory(corpus)]
    second = [doc.doc_id for doc in load_directory(corpus)]
    assert first == second


def test_missing_directory_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError, match="corpus directory not found"):
        load_directory(tmp_path / "absent")


def test_single_file_loads(corpus: Path) -> None:
    assert load_file(corpus / "a.txt").text == "alpha"
