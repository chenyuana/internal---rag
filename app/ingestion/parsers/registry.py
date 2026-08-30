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
from app.ingestion.parsers.scan_regulatory import (
    ScannedRegulatoryPdfParser,
    is_scan_regulatory_pdf,
)
from app.ingestion.pipeline import ParsedDocument


class ParserRegistry:
    def __init__(
        self,
        plugins: list[ParserPlugin],
        *,
        scan_parser: ParserPlugin | None = None,
    ) -> None:
        self._plugins = plugins
        self._scan = scan_parser

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
        mineru = MinerUClient(resolved.mineru, transport=mineru_transport)
        hybrid = HybridPdfParser(
            native,
            mineru,
            DoclingClient(resolved.docling, transport=docling_transport),
            FigureVisionClient(
                resolved.figure_vlm,
                transport=figure_vlm_transport,
            ),
        )
        scan = ScannedRegulatoryPdfParser(mineru)
        return cls([hybrid, DocxParser()], scan_parser=scan)

    @property
    def plugins(self) -> tuple[ParserPlugin, ...]:
        return tuple(self._plugins)

    def select(self, path: Path) -> ParserPlugin:
        extension = path.suffix.casefold()
        # 内容路由：扫描型英文法规/案卷走独立解析器，其余继续走扩展名匹配。
        if extension == ".pdf" and self._scan is not None:
            try:
                if is_scan_regulatory_pdf(path):
                    return self._scan
            except Exception:
                # 判定失败(如读取异常)时保守回落，不影响解析。
                pass
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
        if isinstance(plugin, (HybridPdfParser, ScannedRegulatoryPdfParser)):
            return plugin.parse(path, progress_callback=progress_callback)
        return plugin.parse(path)
