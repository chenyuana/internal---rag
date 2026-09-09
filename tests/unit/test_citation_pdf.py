from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.retrieval import ChunkMetadata, RetrievedChunk
from app.services.citation_service import CitationService


def test_citation_preserves_dataset_and_page():
    _, citations = CitationService().build(
        [
            RetrievedChunk(
                chunk_id="chunk",
                document_id="doc",
                dataset_id="original-kb",
                text="evidence",
                metadata=ChunkMetadata(page_number=4),
            )
        ]
    )
    assert citations[0].dataset_id == "original-kb"
    assert citations[0].page_number == 4


def test_file_prefers_exact_local_publication(client, tmp_path, monkeypatch):
    path = tmp_path / "source.pdf"
    path.write_bytes(b"%PDF-1.4\nlocal-original")

    def lookup(dataset, doc):
        if (dataset, doc) == ("kb", "doc"):
            return path, "application/pdf", "source.pdf"
        return None

    monkeypatch.setattr(client.app.state.ingestion_service, "published_source_file", lookup)
    response = client.get("/api/v1/documents/document/kb/doc/file")
    assert response.status_code == 200 and response.content == path.read_bytes()
    assert response.headers["content-disposition"] == "inline"
    assert client.get("/api/v1/documents/document/other/doc/file").status_code == 503


def test_pdf_dialog_is_served(client):
    html = client.get("/chat").text
    assert 'id="citationPdfDialog"' in html and 'id="citationPdfFrame"' in html
    js = client.get("/assets/citation_pdf.js").text
    assert "encodeURIComponent(dataset)" in js and "#page=" in js
    assert "AbortController" in js and "revokeObjectURL" in js


def test_remote_fallback_accepts_pdf_and_rejects_cleaned_text(client, monkeypatch):
    remote = SimpleNamespace(
        download_document=AsyncMock(return_value=(b"%PDF-1.4\nremote", "application/octet-stream"))
    )
    monkeypatch.setattr(client.app.state.services, "_ragflow_client", remote)
    result = client.get("/api/v1/documents/document/kb/doc/file")
    assert result.status_code == 200 and result.headers["content-type"] == "application/pdf"
    remote.download_document.return_value = (b"# cleaned markdown", "text/plain")
    assert client.get("/api/v1/documents/document/kb/doc/file").status_code == 415
    # Restore before the application lifespan closes registered clients.
    monkeypatch.setattr(client.app.state.services, "_ragflow_client", None)
