"""PDF and plain-text parsing."""

from __future__ import annotations

from app.rag.types import Document

PDF_EXTENSIONS = (".pdf",)
TEXT_EXTENSIONS = (".txt",)
ALLOWED_EXTENSIONS = PDF_EXTENSIONS + TEXT_EXTENSIONS


class UnsupportedFileError(ValueError):
    """Raised for an extension this module does not parse."""


class FileParseError(ValueError):
    """Raised when a supported file cannot be read."""


def parse_pdf(raw: bytes) -> str:
    # Imported here so `import app.rag` does not pull in pypdf.
    import io

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise FileParseError(f"Failed to parse PDF: {exc}") from exc


def parse_txt(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # A few undecodable bytes should cost those bytes, not the whole document.
        return raw.decode("utf-8", errors="ignore")


def parse_bytes(filename: str, raw: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(PDF_EXTENSIONS):
        return parse_pdf(raw)
    if lower.endswith(TEXT_EXTENSIONS):
        return parse_txt(raw)
    raise UnsupportedFileError(f"Only {' and '.join(ALLOWED_EXTENSIONS)} are supported")


def document_from_file(filename: str, raw: bytes) -> Document:
    return Document.create(
        label=filename,
        text=parse_bytes(filename, raw),
        source_type="file",
    )
