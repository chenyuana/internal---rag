from __future__ import annotations

from pathlib import Path

from pypdf import __version__ as pypdf_version

from app.ingestion.pipeline import ParsedDocument, parse_pdf


class NativePdfParser:
    name = "native-pdf"
    version = pypdf_version
    supported_extensions = frozenset({".pdf"})

    def parse(self, path: Path) -> ParsedDocument:
        return parse_pdf(path)
