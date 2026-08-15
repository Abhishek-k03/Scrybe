"""Ingest stage: raw sources in, `Document` out.

Not registry-dispatched. Which source a document came from is a property of the run, not a
knob a sweep varies, so it sits outside `PipelineConfig`.
"""

from app.rag.ingest import file, local, url

__all__ = ["file", "local", "url"]
