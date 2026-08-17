"""Parser plugin interfaces and built-in implementations."""

from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.registry import ParserRegistry
from app.ingestion.parsers.remote import DoclingClient, MinerUClient, RemoteParseResult

__all__ = [
    "DoclingClient",
    "DocxParser",
    "MinerUClient",
    "ParserRegistry",
    "RemoteParseResult",
]
