"""Render a cleaned document IR back into portable Markdown (+ ZIP).

The ingestion pipeline already persists a ``cleaned.md`` per document, but that
file is text-only: figure images live as separate asset files whose only link to
the text is the metadata in ``document_ir.json`` (``asset_id`` join key plus the
per-block ``page_number`` / ``bbox`` used for reading order).  These helpers
rebuild that link into an exportable artifact:

* ``render_markdown`` emits page-ordered Markdown where figure blocks become
  ``![caption](assets/xxx.png)`` references and table blocks become real
  Markdown grids (reusing ``table_to_markdown``).  A figure reference is only
  emitted when its asset file actually resolves under ``asset_root``, so the
  Markdown never points at a file that is not part of the export.
* ``build_export_archive`` packages the Markdown, the referenced asset files and
  a ``manifest.json`` (asset_id -> page -> bbox) into one ZIP for verification.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ingestion.pipeline import table_to_markdown


def _bbox(block: dict[str, Any]) -> tuple[float, float] | None:
    raw = block.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def _ordered_page_blocks(
    page_blocks: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    """Sort a page's blocks into reading order.

    When every block carries a bbox, sort by (top, left) exactly like the
    pipeline's page assembly does.  Otherwise keep the original relative order
    so bbox-less native text is not arbitrarily reshuffled.
    """
    if page_blocks and all(_bbox(block) is not None for _, block in page_blocks):
        return sorted(
            page_blocks,
            key=lambda pair: (_bbox(pair[1])[1], _bbox(pair[1])[0]),  # type: ignore[index]
        )
    return page_blocks


def _resolve_asset(asset_root: Path | None, relative_path: str) -> Path | None:
    if asset_root is None or not relative_path.strip():
        return None
    root = asset_root.resolve()
    candidate = (root / relative_path.replace("\\", "/")).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate


def _asset_by_id(document_ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(asset.get("asset_id")): asset
        for asset in document_ir.get("assets", [])
        if isinstance(asset, dict) and asset.get("asset_id")
    }


def _figure_reference(
    block: dict[str, Any],
    asset: dict[str, Any] | None,
    asset_root: Path | None,
) -> str | None:
    if asset is None:
        return None
    relative_path = str(asset.get("relative_path") or "").strip()
    if not relative_path or _resolve_asset(asset_root, relative_path) is None:
        return None
    text = str(block.get("text") or "").strip()
    alt = (text.split("\n", 1)[0] if text else "").strip()
    if not alt:
        alt = str(asset.get("caption") or "").strip() or "图片"
    return f"![{alt}]({relative_path.replace(chr(92), '/')})"


def _render_block(
    block: dict[str, Any],
    asset_by_id: dict[str, dict[str, Any]],
    *,
    include_images: bool,
    asset_root: Path | None,
) -> str:
    block_type = str(block.get("block_type") or "").casefold()
    text = str(block.get("text") or "").strip()

    if block_type == "table":
        rows = block.get("table_rows") or []
        if isinstance(rows, list) and rows:
            rendered = table_to_markdown(
                [[str(cell) for cell in row] for row in rows],
                title=block.get("table_title"),
            )
            if rendered.strip():
                return rendered
        return text

    if block_type == "figure":
        lines: list[str] = []
        if text:
            lines.append(text)
        if include_images:
            asset_id = block.get("asset_id") or (
                (block.get("asset_ids") or [None])[0]
            )
            asset = asset_by_id.get(str(asset_id)) if asset_id is not None else None
            reference = _figure_reference(block, asset, asset_root)
            if reference:
                lines.append(reference)
        return "\n\n".join(lines)

    return text


def render_markdown(
    document_ir: dict[str, Any],
    *,
    include_images: bool = True,
    asset_root: Path | None = None,
) -> str:
    blocks = document_ir.get("blocks") or []
    asset_by_id = _asset_by_id(document_ir) if include_images else {}
    pages: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        page_number = int(block.get("page_number") or 0)
        pages.setdefault(page_number, []).append((index, block))

    sections: list[str] = []
    for page_number in sorted(pages):
        ordered = _ordered_page_blocks(pages[page_number])
        parts = [
            rendered
            for _, block in ordered
            if (
                rendered := _render_block(
                    block,
                    asset_by_id,
                    include_images=include_images,
                    asset_root=asset_root,
                )
            ).strip()
        ]
        if parts:
            sections.append(f"<!-- page: {page_number} -->\n\n" + "\n\n".join(parts))
    return "\n\n".join(sections)


def _safe_stem(source_name: str) -> str:
    stem = Path(str(source_name)).stem or "document"
    return "".join(
        "_" if char in '<>:"/\\|?*\x00-\x1f' else char for char in stem
    ).strip(" .") or "document"


def _write_export(
    archive: zipfile.ZipFile,
    document_ir: dict[str, Any],
    asset_root: Path | None,
    source_name: str,
    *,
    folder_prefix: str = "",
) -> dict[str, Any]:
    """Write one document's Markdown, asset files and manifest into an open ZIP.

    All entries are written under ``folder_prefix`` so a batch archive can keep
    each document in its own folder without path collisions.
    """
    markdown = render_markdown(
        document_ir,
        include_images=True,
        asset_root=asset_root,
    )
    safe_stem = _safe_stem(source_name)
    archive.writestr(folder_prefix + f"{safe_stem}.md", markdown)

    manifest_assets: list[dict[str, Any]] = []
    for asset in document_ir.get("assets", []):
        if not isinstance(asset, dict) or not asset.get("asset_id"):
            continue
        relative_path = str(asset.get("relative_path") or "").strip()
        asset_path = _resolve_asset(asset_root, relative_path)
        if asset_path is not None:
            archive.write(asset_path, folder_prefix + relative_path.replace("\\", "/"))
        manifest_assets.append(
            {
                "asset_id": str(asset.get("asset_id")),
                "page_number": asset.get("page_number"),
                "bbox": asset.get("bbox"),
                "asset_type": asset.get("asset_type"),
                "mime_type": asset.get("mime_type"),
                "filename": asset.get("filename"),
                "relative_path": relative_path.replace("\\", "/") or None,
                "caption": asset.get("caption") or None,
                "description": asset.get("description") or None,
            }
        )
    source = document_ir.get("source") or {}
    manifest = {
        "format": "cleaned-markdown+assets/1",
        "source_name": source_name,
        "source_sha256": source.get("sha256"),
        "document_id": document_ir.get("document_id"),
        "exported_at": datetime.now(UTC).isoformat(),
        "markdown_file": f"{safe_stem}.md",
        "asset_count": len(manifest_assets),
        "assets": manifest_assets,
    }
    archive.writestr(
        folder_prefix + "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    return manifest


def build_export_archive(
    document_ir: dict[str, Any],
    asset_root: Path | None,
    source_name: str,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        _write_export(archive, document_ir, asset_root, source_name)
    return buffer.getvalue()


def build_batch_export_archive(
    documents: list[tuple[dict[str, Any], Path | None, str]],
) -> bytes:
    """Package multiple documents into one ZIP, one folder per document.

    ``documents`` is a list of ``(document_ir, asset_root, source_name)``.  The
    top-level ``manifest.json`` lists every document with its folder, so the
    caller can map each folder back to its source without guessing.
    """
    buffer = io.BytesIO()
    used_folders: set[str] = set()
    entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for document_ir, asset_root, source_name in documents:
            base = _safe_stem(source_name)
            folder = base
            counter = 2
            while folder.casefold() in used_folders:
                folder = f"{base}-{counter}"
                counter += 1
            used_folders.add(folder.casefold())
            manifest = _write_export(
                archive,
                document_ir,
                asset_root,
                source_name,
                folder_prefix=folder + "/",
            )
            entries.append(
                {
                    "folder": folder,
                    "source_name": source_name,
                    "source_sha256": (document_ir.get("source") or {}).get("sha256"),
                    "document_id": document_ir.get("document_id"),
                    "asset_count": manifest["asset_count"],
                    "markdown_file": manifest["markdown_file"],
                }
            )
        batch_manifest = {
            "format": "cleaned-markdown-batch/1",
            "exported_at": datetime.now(UTC).isoformat(),
            "document_count": len(entries),
            "documents": entries,
        }
        archive.writestr(
            "manifest.json",
            json.dumps(batch_manifest, ensure_ascii=False, indent=2),
        )
    return buffer.getvalue()
