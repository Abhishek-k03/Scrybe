"""Scraping, as the routes see it. The implementation lives in `app.rag.ingest.url`."""

from __future__ import annotations

import logging

from app.rag.ingest.url import DEFAULT_TIMEOUT_MS, fetch_html, html_to_text

log = logging.getLogger("scrybe.scraper")


async def scrape_url(url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    cleaned = html_to_text(await fetch_html(url, timeout_ms))
    log.info("Scraped %d chars from %s", len(cleaned), url)
    return cleaned
