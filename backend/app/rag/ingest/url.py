"""URL ingestion: render the page, strip chrome, flatten to text.

The fetch and the extraction are separate so `html_to_text` can be tested on a fixed HTML
string without launching a browser.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from app.rag.types import Document

STRIP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "iframe", "svg")
DEFAULT_TIMEOUT_MS = 30000

Fetcher = Callable[[str, int], Awaitable[str]]


def html_to_text(html: str) -> str:
    # Imported here so `import app.rag` does not pull in bs4/lxml.
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())


async def fetch_html(url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            return await page.content()
        finally:
            await browser.close()


async def document_from_url(
    url: str,
    *,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    fetch: Fetcher | None = None,
) -> Document:
    html = await (fetch or fetch_html)(url, timeout_ms)
    return Document.create(label=url, text=html_to_text(html), source_type="url")
