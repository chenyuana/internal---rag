from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pypdf import PdfWriter

from app.core.config import IngestionSettings, RemoteParserSettings
from app.core.exceptions import AppError
from app.ingestion.jobs import IngestionJobService
from app.ingestion.parsers.hybrid_pdf import HybridPdfParser
from app.ingestion.parsers.native_pdf import NativePdfParser
from app.ingestion.parsers.remote import MinerUClient, RemoteContentBlock, RemoteParseResult
from app.ingestion.parsers.scan_regulatory import ScannedRegulatoryPdfParser
from app.ingestion.pipeline import write_document
from app.schemas.ingestion import IngestionJob, IngestionJobStatus, PublicationStatus


@pytest.fixture
def cached(tmp_path):
    source = tmp_path / "sample.pdf"
    writer = PdfWriter()
    for _ in range(2):
        writer.add_blank_page(width=600, height=800)
    writer.write(source)
    raw = {"type": "text", "text": r"Speed $\pmb{V}_{s_{0}}$", "bbox": [100, 200, 300, 220]}
    client = MinerUClient(RemoteParserSettings(enabled=True, base_url="http://unused.local"))
    client.parse_pages = Mock(
        return_value=RemoteParseResult(
            "mineru",
            page_blocks={
                1: [
                    RemoteContentBlock(
                        1,
                        "paragraph",
                        "§ 23.1 Original heading",
                        raw_item={"type": "text", "text": "§ 23.1 Original heading"},
                    )
                ],
                2: [RemoteContentBlock(2, "paragraph", "Speed ±bV_s0", raw_item=raw)],
            },
        )
    )
    parser = ScannedRegulatoryPdfParser(client)
    document = parser.parse(source)
    output = Path(write_document(tmp_path / "output", document)["output_folder"])
    ir = json.loads((output / "document_ir.json").read_text(encoding="utf-8"))
    client.parse_pages.reset_mock()
    client.ocr_table_lines = Mock(side_effect=AssertionError("Unexpected table OCR"))
    return source, parser, client, document, output, ir


def test_clean_page_does_not_call_ocr_and_keeps_other_page(cached):
    source, parser, client, _, output, ir = cached
    before = copy.deepcopy(ir)
    result = parser.reprocess_page(source, ir, output, 2, "clean")
    client.parse_pages.assert_not_called()
    client.ocr_table_lines.assert_not_called()
    assert "Speed V_s0" in result.pages[1].cleaned_text
    assert result.pages[0].cleaned_text == ir["pages"][0]["cleaned_text"]
    assert result.blocks[-1].section_path == ["§ 23.1 Original heading"]
    assert all("±b" not in c.text for c in result.chunks)
    assert result.qa["page_reprocess"]["effective_mode"] == "clean"
    assert ir == before


@pytest.mark.parametrize("mode", ["clean", "ocr"])
def test_legacy_fallback_or_explicit_ocr_requests_only_one_page(cached, mode):
    source, parser, client, _, output, ir = cached
    ir["pages"][1]["layout_features"]["ocr_blocks"][0].pop("raw_ocr_item")
    client.parse_pages.return_value = RemoteParseResult(
        "mineru", page_blocks={2: [RemoteContentBlock(2, "paragraph", "New page two")]}
    )
    result = parser.reprocess_page(source, ir, output, 2, mode)
    assert client.parse_pages.call_args.args[1] == [2]
    assert len(result.pages) == 2
    assert "New page two" in result.pages[1].cleaned_text
    assert result.qa["page_reprocess"]["effective_mode"] == "ocr"


def test_heading_change_rebuilds_following_page_chunks(cached):
    source, parser, client, _, output, ir = cached
    client.parse_pages.return_value = RemoteParseResult(
        "mineru", page_blocks={1: [RemoteContentBlock(1, "paragraph", "§ 23.2 New heading")]}
    )
    result = parser.reprocess_page(source, ir, output, 1, "ocr")
    assert result.blocks[-1].section_path == ["§ 23.2 New heading"]
    assert all("Original heading" not in c.section_path for c in result.chunks)


def test_empty_ocr_or_hash_mismatch_rejected(cached):
    source, parser, client, _, output, ir = cached
    client.parse_pages.return_value = RemoteParseResult("mineru")
    with pytest.raises(ValueError, match="no structured"):
        parser.reprocess_page(source, ir, output, 2, "ocr")
    ir["source"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        parser.reprocess_page(source, ir, output, 2, "clean")


def test_hybrid_page_ocr_retries_fragmented_table_and_rebuilds_chunks(cached):
    source, _, client, _, output, ir = cached
    partial = (
        "<table><tr><td colspan=3>Table RF3</td></tr>"
        "<tr><td>Part 23 Section</td><td>Costs</td><td>Turbojet</td></tr>"
        "<tr><td>23.831</td><td>300,000</td><td></td></tr>"
        "<tr><td>X</td><td></td><td>X</td></tr></table>"
    )
    complete = (
        "<table><tr><td colspan=3>Table RF3</td></tr>"
        "<tr><td>Part 23 Section</td><td>Costs</td><td>Turbojet</td></tr>"
        "<tr><td>23.831</td><td>$300,000 and 7#</td><td>X</td></tr>"
        "<tr><td>23.1309(a)(3)</td><td>550</td><td>X</td></tr>"
        "<tr><td>23.1353</td><td>$1,000 and 30#</td><td>X</td></tr></table>"
    )
    client.parse_pages.side_effect = [
        RemoteParseResult(
            "mineru",
            page_blocks={2: [RemoteContentBlock(2, "table", partial, table_html=partial)]},
        ),
        RemoteParseResult(
            "mineru",
            page_blocks={2: [RemoteContentBlock(2, "table", complete, table_html=complete)]},
        ),
    ]
    parser = HybridPdfParser(
        NativePdfParser(),
        client,
        SimpleNamespace(enabled=False),
        SimpleNamespace(enabled=False),
    )
    result = parser.reprocess_page(source, ir, output, 2, "ocr")
    assert client.parse_pages.call_count == 2
    assert all(call.args[1] == [2] for call in client.parse_pages.call_args_list)
    table = next(block for block in result.blocks if block.block_type == "table")
    assert any(row[0] == "23.1309(a)(3)" for row in table.table_rows)
    assert any(row[0] == "23.1353" for row in table.table_rows)
    assert "$300,000 and 7#" in table.text
    assert "§ 23.1 Original heading" in result.pages[0].cleaned_text
    assert result.qa["page_reprocess"]["retried_suspect_table"] is True


def test_hybrid_page26_fragment_and_following_heading_are_detected():
    markup = (
        "<table><tr><td colspan=4>Final Rule Benefits and Costs ($M)</td></tr>"
        "<tr><td>89.1</td><td></td><td></td><td></td></tr>"
        "<tr><td>High Case</td><td>188.6</td><td>48.1</td><td></td></tr>"
        "<tr><td>38.3</td><td></td><td></td><td></td></tr>"
        "<tr><td>High Case</td><td>72.9</td><td>$26.7</td><td></td></tr></table>"
    )
    result = RemoteParseResult(
        "mineru",
        page_blocks={26: [RemoteContentBlock(26, "table", markup, table_html=markup)]},
    )
    assert HybridPdfParser._table_page_needs_retry(result, 26)
    rows = [["Final Rule Benefits and Costs ($M)", "", "", ""]]
    assert HybridPdfParser._separate_table_caption(
        "Who is Potentially Affected by This Rule", rows
    ) == ("", "Who is Potentially Affected by This Rule")
    assert HybridPdfParser._separate_table_caption("Table RF3", rows) == (
        "Table RF3",
        "",
    )


@pytest.fixture
def job_service(cached, tmp_path):
    source, _, _, document, output, _ = cached
    service = IngestionJobService(IngestionSettings(root_dir=tmp_path / "jobs-root"))
    now = datetime.now(UTC)
    job = IngestionJob(
        job_id="page-test",
        status="completed",
        source_name=source.name,
        source_sha256=document.source_hash,
        media_type="application/pdf",
        size_bytes=source.stat().st_size,
        created_by="tester",
        original_path=str(source),
        output_path=str(output),
        created_at=now,
        updated_at=now,
    )
    service._write_job(job)
    return service, job


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_page_job_atomic_output_switch_or_rollback(job_service, cached, failure):
    service, job = job_service
    document = cached[3]
    old_bytes = (Path(job.output_path) / "document_ir.json").read_bytes()
    await service.reprocess_page(job.job_id, 2, "clean")
    with patch.object(
        ScannedRegulatoryPdfParser,
        "reprocess_page",
        side_effect=RuntimeError("OCR down") if failure else None,
        return_value=document,
    ):
        result = await service.process_job(job.job_id)
    assert result.page_reprocess is None
    assert (Path(job.output_path) / "document_ir.json").read_bytes() == old_bytes
    if failure:
        assert result.output_path == job.output_path
        assert result.status == job.status
        assert "旧结果已保留" in result.error_message
    else:
        assert result.output_path != job.output_path
        assert "revisions" in result.output_path
        assert result.review.status == "unreviewed"
        assert result.publication.status == "not_requested"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["published", "partial", "manual", "busy", "page"])
async def test_page_job_guards(job_service, reason):
    service, job = job_service
    if reason == "published":
        job.publication.status = PublicationStatus.PUBLISHED
    elif reason == "partial":
        job.publication.document_id = "remote-document"
    elif reason == "busy":
        job.status = IngestionJobStatus.RUNNING
    elif reason == "manual":
        path = Path(job.output_path) / "document_ir.json"
        ir = json.loads(path.read_text(encoding="utf-8"))
        ir["chunks"][0]["manual_revision"] = {"revision": 1}
        path.write_text(json.dumps(ir), encoding="utf-8")
    service._write_job(job)
    with pytest.raises(AppError):
        await service.reprocess_page(job.job_id, 99 if reason == "page" else 2, "clean")


def test_page_reprocess_api_queues_once_and_validates_mode(client, job_service):
    service, job = job_service
    with (
        patch.object(client.app.state, "ingestion_service", service),
        patch.object(service, "enqueue", new_callable=AsyncMock) as enqueue,
    ):
        url = f"/api/v1/ingestion/jobs/{job.job_id}/pages/2/reprocess"
        assert client.post(url, json={"mode": "invalid"}).status_code == 422
        response = client.post(url, json={"mode": "clean"})
        assert response.status_code == 202
        assert response.json()["page_reprocess"]["page_number"] == 2
        assert client.post(url, json={"mode": "ocr"}).status_code == 409
        enqueue.assert_awaited_once_with(job.job_id)


@pytest.mark.asyncio
async def test_page_reprocess_service_accepts_hybrid_pdf(job_service):
    service, job = job_service
    path = Path(job.output_path) / "document_ir.json"
    ir = json.loads(path.read_text(encoding="utf-8"))
    ir["parser"]["name"] = "hybrid-pdf"
    path.write_text(json.dumps(ir), encoding="utf-8")
    queued = await service.reprocess_page(job.job_id, 2, "clean")
    assert queued.page_reprocess is not None
    assert queued.page_reprocess["page_number"] == 2
    assert queued.page_reprocess["mode"] == "clean"
