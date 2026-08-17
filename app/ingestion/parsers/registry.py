from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from app.core.config import IngestionSettings
from app.core.exceptions import AppError
from app.ingestion.parsers.base import ParserPlugin
from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.figure_vision import FigureVisionClient
from app.ingestion.parsers.hybrid_pdf import HybridPdfParser
from app.ingestion.parsers.native_pdf import NativePdfParser
from app.ingestion.parsers.remote import DoclingClient, MinerUClient
from app.ingestion.pipeline import ParsedDocument


class ParserRegistry:
    def __init__(self, plugins: list[ParserPlugin]) -> None:
        self._plugins = plugins

    @classmethod
    def with_builtins(
        cls,
        settings: IngestionSettings | None = None,
        *,
        mineru_transport: httpx.BaseTransport | None = None,
        docling_transport: httpx.BaseTransport | None = None,
        figure_vlm_transport: httpx.BaseTransport | None = None,
    ) -> ParserRegistry:
        resolved = settings or IngestionSettings()
        native = NativePdfParser()
        hybrid = HybridPdfParser(
            native,
            MinerUClient(resolved.mineru, transport=mineru_transport),
            DoclingClient(resolved.docling, transport=docling_transport),
            FigureVisionClient(
                resolved.figure_vlm,
                transport=figure_vlm_transport,
            ),
        )
        return cls([hybrid, DocxParser()])

    @property
    def plugins(self) -> tuple[ParserPlugin, ...]:
        return tuple(self._plugins)

    def select(self, path: Path) -> ParserPlugin:
        extension = path.suffix.casefold()
        for plugin in self._plugins:
            if extension in plugin.supported_extensions:
                return plugin
        raise AppError(
            code="PARSER_NOT_AVAILABLE",
            message="No parser plugin is available for this document.",
            status_code=422,
            details={
                "extension": extension,
                "registered_parsers": [plugin.name for plugin in self._plugins],
            },
        )

    def parse(
        self,
        path: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ParsedDocument:
        plugin = self.select(path)
        if isinstance(plugin, HybridPdfParser):
            return plugin.parse(path, progress_callback=progress_callback)
        return plugin.parse(path)
