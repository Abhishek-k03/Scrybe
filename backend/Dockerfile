# Playwright's own image, so Chromium and its system libraries are already present.
# This base image supplies the browser dependencies needed for URL ingestion.
#
# Pin must track playwright in requirements.txt: image v1.60.0 <-> playwright==1.60.0.
FROM mcr.microsoft.com/playwright/python:v1.60.0-noble

# Ubuntu 24.04's system Python is PEP 668 externally-managed; a venv keeps pip usable
# without --break-system-packages and isolates us from the image's own packages.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The base image ships Python without the venv module, so install it first.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "$VIRTUAL_ENV"

WORKDIR /app

# Copied first so the dependency layer caches across code-only deploys.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# CHROMA_PATH and EMBED_CACHE_PATH are CWD-relative by default, which resolves under
# /app here. Cloud Run's container filesystem is ephemeral — see README.
EXPOSE 10000

# Shell form so $PORT (injected by the host) expands at runtime.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
