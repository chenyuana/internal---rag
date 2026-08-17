from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.ingestion.pipeline import ParsedDocument


class ParserPlugin(Protocol):
    name: str
    version: str
    supported_extensions: frozenset[str]

    def parse(self, path: Path) -> ParsedDocument: ...
