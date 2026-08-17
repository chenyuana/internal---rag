from __future__ import annotations

import json
from pathlib import Path

import httpx

from app.core.config import RagflowSettings
from app.ingestion.publishers import RagflowPublisher


def _document_ir() -> dict[str, object]:
    return {
        "chunks": [
            {
                "chunk_id": "internal-1",
                "title": "第一条",
                "text": "设备应当满足测试要求。",
                "page_start": 3,
                "page_end": 3,
                "section_path": ["第一章", "第一条"],
                "article_id_normalized": "21.1(1)",
                "article_aliases": ["第21.1条1", "第21.1条第1款"],
                "keywords": ["适用范围"],
                "question_aliases": ["第21.1条1的内容是什么？"],
            },
            {
                "chunk_id": "internal-2",
                "title": "第二条",
                "text": "记录应当长期保存。",
                "page_start": 4,
                "page_end": 5,
                "section_path": ["第一章", "第二条"],
            },
        ]
    }


def test_publish_plan_is_deterministic() -> None:
    first = RagflowPublisher.build_plan(
        dataset_id="kb-1",
        source_name="规章.pdf",
        source_sha256="abc",
        document_ir=_document_ir(),
    )
    second = RagflowPublisher.build_plan(
        dataset_id="kb-1",
        source_name="规章.pdf",
        source_sha256="abc",
        document_ir=_document_ir(),
    )

    assert first.plan_sha256 == second.plan_sha256
    assert len(first.chunks) == 2
    assert "页码：3" in first.chunks[0].payload["content"]
    assert "internal_chunk_id:internal-1" in first.chunks[0].payload["tag_kwd"]
    assert first.chunks[0].payload["chunk_order"] == 0
    assert first.chunks[0].payload["page_numbers"] == [3]
    assert "条号：21.1(1)" in first.chunks[0].payload["content"]
    assert "article_id:21.1(1)" in first.chunks[0].payload["tag_kwd"]
    assert "第21.1条1" in first.chunks[0].payload["important_keywords"]
    assert first.chunks[0].payload["questions"] == ["第21.1条1的内容是什么？"]
    assert first.chunks[1].payload["chunk_order"] == 1
    assert first.chunks[1].payload["page_numbers"] == [4, 5]


def test_publish_plan_preserves_structured_table_and_asset(
    tmp_path: Path,
) -> None:
    asset_folder = tmp_path / "assets"
    asset_folder.mkdir()
    table_image = b"\x89PNG\r\n\x1a\nTABLE"
    (asset_folder / "table.png").write_bytes(table_image)
    document_ir = {
        "assets": [
            {
                "asset_id": "p0162-table-01",
                "asset_type": "table",
                "page_number": 162,
                "bbox": [90.0, 521.0, 505.0, 651.0],
                "relative_path": "assets/table.png",
            }
        ],
        "chunks": [
            {
                "chunk_id": "table-chunk",
                "title": "表1 限制飞行载荷系数",
                "text": "飞行载荷系数 | 正常类 | 实用类 | 特技类",
                "page_start": 162,
                "page_end": 162,
                "section_path": ["第A23.13条"],
                "asset_ids": ["p0162-table-01"],
                "table_html": [
                    "<table><tr><th>飞行载荷系数</th>"
                    "<th>正常类</th><th>实用类</th><th>特技类</th></tr>"
                    "<tr><td>n1</td><td>3.8</td><td>4.4</td><td>6.0</td></tr>"
                    "</table>"
                ],
            }
        ],
    }

    plan = RagflowPublisher.build_plan(
        dataset_id="kb-1",
        source_name="CCAR-23.pdf",
        source_sha256="abc",
        document_ir=document_ir,
        asset_root=tmp_path,
    )

    payload = plan.chunks[0].payload
    assert "飞行载荷系数 | 正常类 | 实用类 | 特技类" in payload["content"]
    assert "<table>" not in payload["content"]
    assert "content_type:table" in payload["tag_kwd"]
    assert payload["positions"] == [[162, 90, 505, 521, 651]]
    assert payload["image_base64"]


async def test_upload_document_uses_docx_media_type(tmp_path: Path) -> None:
    source = tmp_path / "测试要求.docx"
    source.write_bytes(b"PK-test")
    request_content = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_content
        request_content = request.content
        return httpx.Response(
            200,
            json={"code": 0, "data": [{"id": "doc-word"}]},
        )

    publisher = RagflowPublisher(
        RagflowSettings(
            enabled=True,
            base_url="http://ragflow.local",
            api_key="secret",
        ),
        transport=httpx.MockTransport(handler),
    )
    plan = publisher.build_plan(
        dataset_id="kb-1",
        source_name=source.name,
        source_sha256="word-hash",
        document_ir=_document_ir(),
    )
    try:
        document_id = await publisher.upload_document(plan, source)
    finally:
        await publisher.close()

    assert document_id == "doc-word"
    assert (
        b"application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        in request_content
    )


async def test_live_publish_uploads_original_and_chunks(tmp_path: Path) -> None:
    source = tmp_path / "规章.pdf"
    source.write_bytes(b"%PDF-test")
    calls: list[tuple[str, str]] = []
    progress_updates: list[tuple[str, int, int]] = []
    published_batches: list[list[dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/documents") and request.method == "POST":
            assert request.headers["authorization"] == "Bearer secret"
            return httpx.Response(200, json={"code": 0, "data": [{"id": "doc-1"}]})
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"code": 0, "data": {"total": 0, "chunks": []}},
            )
        payload = json.loads(request.content)
        published_batches.append(payload["chunks"])
        return httpx.Response(
            200,
            json={"code": 0, "data": {"chunks": [{"id": "chunk"}]}},
        )

    publisher = RagflowPublisher(
        RagflowSettings(
            enabled=True,
            base_url="http://ragflow.local",
            api_key="secret",
        ),
        transport=httpx.MockTransport(handler),
    )
    plan = publisher.build_plan(
        dataset_id="kb-1",
        source_name=source.name,
        source_sha256="abc",
        document_ir=_document_ir(),
    )
    try:
        document_id, published, skipped = await publisher.publish(
            plan,
            source,
            progress=lambda document_id, published, skipped: progress_updates.append(
                (document_id, published, skipped)
            ),
        )
    finally:
        await publisher.close()

    assert document_id == "doc-1"
    assert published == 2
    assert skipped == 0
    assert calls.count(
        ("POST", "/api/v1/datasets/kb-1/documents/doc-1/chunks/batch")
    ) == 1
    assert len(published_batches) == 1
    assert len(published_batches[0]) == 2
    assert progress_updates == [
        ("doc-1", 0, 0),
        ("doc-1", 1, 0),
        ("doc-1", 2, 0),
    ]


async def test_publish_splits_chunks_into_configured_batches(tmp_path: Path) -> None:
    source = tmp_path / "规章.pdf"
    source.write_bytes(b"%PDF-test")
    batch_sizes: list[int] = []
    document_ir = {
        "chunks": [
            {
                "chunk_id": f"internal-{index}",
                "text": f"第 {index} 个不重复分块。",
                "page_start": index + 1,
                "page_end": index + 1,
            }
            for index in range(5)
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"code": 0, "data": {"total": 0, "chunks": []}},
            )
        assert request.url.path.endswith("/chunks/batch")
        batch_sizes.append(len(json.loads(request.content)["chunks"]))
        return httpx.Response(200, json={"code": 0, "data": {"chunks": []}})

    publisher = RagflowPublisher(
        RagflowSettings(
            enabled=True,
            base_url="http://ragflow.local",
            api_key="secret",
            publish_batch_size=2,
        ),
        transport=httpx.MockTransport(handler),
    )
    plan = publisher.build_plan(
        dataset_id="kb-1",
        source_name=source.name,
        source_sha256="abc",
        document_ir=document_ir,
    )
    try:
        _, published, skipped = await publisher.publish(
            plan,
            source,
            document_id="doc-1",
        )
    finally:
        await publisher.close()

    assert published == 5
    assert skipped == 0
    assert batch_sizes == [2, 2, 1]


async def test_retry_skips_content_already_in_ragflow(tmp_path: Path) -> None:
    source = tmp_path / "规章.pdf"
    source.write_bytes(b"%PDF-test")
    added = 0
    plan = RagflowPublisher.build_plan(
        dataset_id="kb-1",
        source_name=source.name,
        source_sha256="abc",
        document_ir=_document_ir(),
    )
    existing_content = plan.chunks[0].payload["content"]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal added
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "total": 1,
                        "chunks": [{"id": "old", "content": existing_content}],
                    },
                },
            )
        added += 1
        return httpx.Response(
            200,
            content=json.dumps({"code": 0, "data": {"chunk": {"id": "new"}}}),
        )

    publisher = RagflowPublisher(
        RagflowSettings(
            enabled=True,
            base_url="http://ragflow.local",
            api_key="secret",
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        _, published, skipped = await publisher.publish(
            plan,
            source,
            document_id="doc-1",
        )
    finally:
        await publisher.close()

    assert published == 1
    assert skipped == 1
    assert added == 1


async def test_retry_removes_stale_managed_chunks(tmp_path: Path) -> None:
    source = tmp_path / "规章.pdf"
    source.write_bytes(b"%PDF-test")
    plan = RagflowPublisher.build_plan(
        dataset_id="kb-1",
        source_name=source.name,
        source_sha256="abc",
        document_ir=_document_ir(),
    )
    deleted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "total": 1,
                        "chunks": [
                            {
                                "id": "stale-chunk",
                                "content": "old flattened table",
                                "tag_kwd": [
                                    "source_sha256:abc",
                                    "internal_chunk_id:old-table",
                                ],
                            }
                        ],
                    },
                },
            )
        if request.method == "DELETE":
            deleted.extend(request.read().decode())
            return httpx.Response(200, json={"code": 0, "data": None})
        return httpx.Response(
            200,
            json={"code": 0, "data": {"chunk": {"id": "new"}}},
        )

    publisher = RagflowPublisher(
        RagflowSettings(
            enabled=True,
            base_url="http://ragflow.local",
            api_key="secret",
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        _, published, skipped = await publisher.publish(
            plan,
            source,
            document_id="doc-1",
        )
    finally:
        await publisher.close()

    assert published == 2
    assert skipped == 0
    assert "stale-chunk" in "".join(deleted)


async def test_retry_recovers_document_created_before_upload_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "规章.pdf"
    source.write_bytes(b"%PDF-test")
    chunk_posts = 0
    posted_chunks = 0
    upload_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chunk_posts, posted_chunks, upload_attempts
        if request.method == "POST" and request.url.path.endswith("/documents"):
            upload_attempts += 1
            return httpx.Response(
                409,
                json={
                    "code": 102,
                    "message": "Duplicated document name in the same dataset.",
                },
            )
        if request.method == "GET" and request.url.path.endswith("/documents"):
            assert request.url.params["keywords"] == source.name
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "total": 1,
                        "docs": [
                            {
                                "id": "recovered-doc",
                                "name": source.name,
                                "chunk_count": 0,
                            }
                        ],
                    },
                },
            )
        if request.method == "GET" and request.url.path.endswith("/chunks"):
            return httpx.Response(
                200,
                json={"code": 0, "data": {"total": 0, "chunks": []}},
            )
        if request.method == "POST" and request.url.path.endswith("/chunks/batch"):
            chunk_posts += 1
            posted_chunks += len(json.loads(request.content)["chunks"])
            return httpx.Response(
                200,
                json={"code": 0, "data": {"chunks": [{"id": f"chunk-{chunk_posts}"}]}},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    publisher = RagflowPublisher(
        RagflowSettings(
            enabled=True,
            base_url="http://ragflow.local",
            api_key="secret",
        ),
        transport=httpx.MockTransport(handler),
    )
    plan = publisher.build_plan(
        dataset_id="kb-1",
        source_name=source.name,
        source_sha256="abc",
        document_ir=_document_ir(),
    )
    try:
        document_id, published, skipped = await publisher.publish(plan, source)
    finally:
        await publisher.close()

    assert document_id == "recovered-doc"
    assert upload_attempts == 1
    assert published == 2
    assert skipped == 0
    assert chunk_posts == 1
    assert posted_chunks == 2


async def test_retry_rejects_same_name_with_different_source(tmp_path: Path) -> None:
    source = tmp_path / "规章.pdf"
    source.write_bytes(b"%PDF-test")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/documents"):
            return httpx.Response(
                409,
                json={"code": 102, "message": "Duplicated document name."},
            )
        if request.method == "GET" and request.url.path.endswith("/documents"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "total": 1,
                        "docs": [{"id": "other-doc", "name": source.name}],
                    },
                },
            )
        if request.method == "GET" and request.url.path.endswith("/chunks"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "total": 1,
                        "chunks": [
                            {
                                "content": "other",
                                "tag_kwd": ["source_sha256:different"],
                            }
                        ],
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    publisher = RagflowPublisher(
        RagflowSettings(
            enabled=True,
            base_url="http://ragflow.local",
            api_key="secret",
        ),
        transport=httpx.MockTransport(handler),
    )
    plan = publisher.build_plan(
        dataset_id="kb-1",
        source_name=source.name,
        source_sha256="abc",
        document_ir=_document_ir(),
    )
    try:
        try:
            await publisher.publish(plan, source)
        except Exception as exc:
            assert getattr(exc, "code", "") == "RAGFLOW_DOCUMENT_NAME_CONFLICT"
        else:
            raise AssertionError("Expected a document name conflict")
    finally:
        await publisher.close()
