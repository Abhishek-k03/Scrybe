import logging
import re

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

log = logging.getLogger("weboracle.scraper")

STRIP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "iframe", "svg")


async def scrape_url(url: str, timeout_ms: int = 30000) -> str:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            html = await page.content()
        finally:
            await browser.close()

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())

    log.info("Scraped %d chars from %s", len(cleaned), url)
    return cleaned
