import io
import json
import zipfile

from app.ingestion.exporters.zip_export import (
    build_batch_export_archive,
    build_export_archive,
    render_markdown,
)


def _figure_block(page, y, asset_id):
    return {
        "block_type": "figure",
        "text": f"Figure — caption on page {page}",
        "page_number": page,
        "bbox": [0, y, 400, y + 200],
        "asset_id": asset_id,
        "asset_ids": [asset_id],
    }


def _paragraph(page, y, text):
    return {"block_type": "paragraph", "text": text, "page_number": page, "bbox": [0, y, 400, y + 20]}


def _asset(tmp_path, asset_id, relative_path="assets/a-fig.png"):
    (tmp_path / "assets").mkdir(exist_ok=True)
    (tmp_path / relative_path).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / relative_path).write_bytes(b"\x89PNG")
    return {
        "asset_id": asset_id,
        "relative_path": relative_path,
        "mime_type": "image/png",
        "page_number": 1,
        "bbox": [0, 0, 400, 200],
    }


def test_figure_becomes_image_reference_and_table_becomes_grid(tmp_path):
    document_ir = {
        "blocks": [
            {"block_type": "table", "text": "", "page_number": 1, "bbox": [0, 0, 400, 100],
             "table_title": "表1 参数", "table_rows": [["项目", "值"], ["A", "1"]]},
            _figure_block(1, 120, "a-fig"),
        ],
        "assets": [_asset(tmp_path, "a-fig")],
    }
    markdown = render_markdown(document_ir, include_images=True, asset_root=tmp_path)
    assert "| 项目 | 值 |" in markdown
    assert "![Figure — caption on page 1](assets/a-fig.png)" in markdown


def test_md_mode_drops_image_reference_but_keeps_caption():
    document_ir = {"blocks": [_figure_block(1, 0, "a-fig")], "assets": []}
    markdown = render_markdown(document_ir, include_images=False)
    assert "assets/a-fig.png" not in markdown
    assert "Figure — caption on page 1" in markdown


def test_blocks_sort_into_reading_order_by_bbox(tmp_path):
    document_ir = {
        "blocks": [
            _paragraph(1, 60, "第二段"),
            _figure_block(1, 30, "a-fig"),
            _paragraph(1, 0, "第一段"),
        ],
        "assets": [_asset(tmp_path, "a-fig")],
    }
    markdown = render_markdown(document_ir, include_images=True, asset_root=tmp_path)
    assert markdown.index("第一段") < markdown.index("assets/a-fig.png") < markdown.index("第二段")


def test_archive_contains_markdown_assets_and_manifest(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "a-fig.png").write_bytes(b"\x89PNG")
    document_ir = {
        "source": {"name": "sample.pdf", "sha256": "abc"},
        "document_id": "doc-1",
        "blocks": [_figure_block(2, 0, "a-fig")],
        "assets": [
            {"asset_id": "a-fig", "relative_path": "assets/a-fig.png", "mime_type": "image/png",
             "page_number": 2, "bbox": [0, 0, 400, 200], "caption": "Figure — caption"},
        ],
    }
    archive = build_export_archive(document_ir, tmp_path, "sample.pdf")
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = zf.namelist()
        assert "sample.md" in names
        assert "assets/a-fig.png" in names
        assert "manifest.json" in names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["asset_count"] == 1
        assert manifest["assets"][0]["asset_id"] == "a-fig"
        assert manifest["assets"][0]["page_number"] == 2
        assert manifest["assets"][0]["relative_path"] == "assets/a-fig.png"


def test_escaping_asset_path_is_not_written(tmp_path):
    document_ir = {
        "blocks": [_figure_block(1, 0, "a-fig")],
        "assets": [
            {"asset_id": "a-fig", "relative_path": "../secret.png", "mime_type": "image/png",
             "page_number": 1, "bbox": [0, 0, 400, 200]},
        ],
    }
    archive = build_export_archive(document_ir, tmp_path, "sample.pdf")
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        assert "../secret.png" not in zf.namelist()
        # The reference is dropped from the markdown because the path is unsafe.
        assert "secret.png" not in zf.read("sample.md").decode("utf-8")


def _single_figure_doc(name, sha, asset_rel):
    return (
        {
            "source": {"name": name, "sha256": sha},
            "document_id": f"doc-{sha}",
            "blocks": [_figure_block(1, 0, "fig")],
            "assets": [
                {"asset_id": "fig", "relative_path": asset_rel, "mime_type": "image/png",
                 "page_number": 1, "bbox": [0, 0, 400, 200]},
            ],
        },
        name,
        asset_rel,
    )


def test_batch_archive_groups_each_document_into_its_own_folder(tmp_path):
    (tmp_path / "assets").mkdir(exist_ok=True)
    (tmp_path / "assets" / "a.png").write_bytes(b"\x89PNG")
    (tmp_path / "assets" / "b.png").write_bytes(b"\x89PNG")
    doc_a, name_a, rel_a = _single_figure_doc("a.pdf", "sha-a", "assets/a.png")
    doc_b, name_b, rel_b = _single_figure_doc("b.pdf", "sha-b", "assets/b.png")
    archive = build_batch_export_archive(
        [(doc_a, tmp_path, name_a), (doc_b, tmp_path, name_b)],
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = zf.namelist()
        assert "a/a.md" in names and "a/assets/a.png" in names and "a/manifest.json" in names
        assert "b/b.md" in names and "b/assets/b.png" in names and "b/manifest.json" in names
        batch = json.loads(zf.read("manifest.json"))
        assert batch["document_count"] == 2
        assert {entry["folder"] for entry in batch["documents"]} == {"a", "b"}


def test_batch_archive_deduplicates_duplicate_stems(tmp_path):
    (tmp_path / "assets").mkdir(exist_ok=True)
    (tmp_path / "assets" / "x.png").write_bytes(b"\x89PNG")
    doc, name, rel = _single_figure_doc("same.pdf", "sha", "assets/x.png")
    archive = build_batch_export_archive(
        [(doc, tmp_path, name), (doc, tmp_path, name)],
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        batch = json.loads(zf.read("manifest.json"))
        folders = [entry["folder"] for entry in batch["documents"]]
        assert len(folders) == 2 and len(set(folders)) == 2
        assert "same" in folders
        assert any(folder.startswith("same-") for folder in folders)
