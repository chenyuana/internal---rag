"""Figure names and explicit cross references; no OCR or vision calls."""

import hashlib
import re

from app.ingestion.parsers.verified_tables import DOCKET4080_SHA256
from app.ingestion.pipeline import AssetRecord, ChunkRecord, PageRecord, stable_id

# Names checked against the source. Both document and image bytes must match.
# These are metadata corrections, not inferred curve contents or numerical values.
KNOWN_TITLES = {
    "d30cc03b7aacc5986f2b804d9eb7172c6b5c1db87b15d55a306b5779f667e3a5":
        (38, "Figure 1 — Chart for Finding n3 Factor at Speed Vc"),
    "48ac728438239f0266c5feb36dbb72a6c2d6463938426da52d98d0327fe7f6a9":
        (38, "Figure 2 — Chart for Finding n4 Factor at Speed Vc"),
    "dde377414ae1cb7275b17e53bc6fad5e9571e6f3fdbcbb222a0f6f2c97714229":
        (39, "Figure 3 — Minimum Design Airspeeds"),
    "790d7a3d23936aa91b9b28bdcb81d43702a6f53c1acdccfd702dddd0c4d2ffd8":
        (39, "Figure 4 — Flight Envelope"),
    "998abb3ce96fad885056408f6d21ba50298b34ff7a2516648a21953c63c4c19b":
        (40, "Figure 5 — Average Limit Control Surface Loading"),
    "bf52e85e79a335de42fa453f01364e73f58e7e4b1d2925ca3ac18250c3a7fd59":
        (40, "Figure 6 — Average Limit Control Surface Loading"),
}


def restore_figure_names(source_hash: str, pages: list[PageRecord],
                         assets: list[AssetRecord]) -> None:
    by_id = {a.asset_id: a for a in assets}
    for page in pages:
        for item in page.rich_blocks:
            if item.get("block_type") != "figure" or item.get("exclusion_reason"):
                continue
            asset = by_id.get(item.get("asset_id"))
            if asset is None:
                continue
            title = item.get("caption") or asset.caption
            if not title:
                first_line = str(item.get("text") or "").split("\n", 1)[0].strip()
                if re.match(r"(?:Figure|Fig\.)\s*\d+\b", first_line, re.I):
                    title = first_line
            if not title and source_hash == DOCKET4080_SHA256:
                known = KNOWN_TITLES.get(hashlib.sha256(asset.content).hexdigest())
                if known and known[0] == page.page_number:
                    title = known[1]
                    item["figure_title_source"] = "source_checked_metadata_2026-08-31"
            if title:
                item["caption"] = asset.caption = title
                if not item.get("text"):
                    item["text"] = title


def _scope(chunk: ChunkRecord) -> tuple[str, ...]:
    for i, title in enumerate(chunk.section_path):
        if re.match(r"APPENDIX\b", title, re.I):
            return tuple(chunk.section_path[:i + 1])
    return tuple(chunk.section_path)


def _references(text: str) -> set[str]:
    refs: set[str] = set()
    for match in re.finditer(
        r"\b(?:figures?|figs?\.)\s*(\d+)((?:\s*(?:,|and|&)\s*\d+)*)", text, re.I
    ):
        refs.update(re.findall(r"\d+", match[0]))
    return refs


def link_figures(document_id: str, chunks: list[ChunkRecord]) -> None:
    targets: dict[tuple[tuple[str, ...], str], list[ChunkRecord]] = {}
    for chunk in chunks:
        if chunk.content_type == "figure":
            match = re.match(r"(?:Figure|Fig\.)\s*(\d+)\b", chunk.title, re.I)
            if match:
                targets.setdefault((_scope(chunk), match[1]), []).append(chunk)
    edges: list[tuple[ChunkRecord, ChunkRecord, str]] = []
    for source in chunks:
        if source.content_type == "figure":
            continue
        # Explicit references to another appendix need a separate resolver.
        # Do not silently bind them to a local figure with the same number.
        mentioned = set(re.findall(r"\bAppendix\s+([A-Z])\b", source.text, re.I))
        scope = _scope(source)
        own = re.match(r"APPENDIX\s+([A-Z])\b", scope[-1], re.I) if scope else None
        if mentioned and (own is None or any(a.upper() != own[1].upper() for a in mentioned)):
            continue
        for number in sorted(_references(source.text), key=int):
            candidates = targets.get((_scope(source), number), [])
            # Ambiguous/missing figure names must never be resolved by page proximity.
            if len(candidates) == 1:
                edges.append((source, candidates[0], number))
    additions: dict[str, list[str]] = {}
    for source, target, number in edges:
        additions.setdefault(source.chunk_id, []).append(
            f"Related figure: {target.title}; source page {target.page_start}."
        )
        # Carry the exact referencing row/line for retrieval, explicitly attributed.
        lines = source.text.splitlines()
        evidence = [line for line in lines if number in _references(line)]
        additions.setdefault(target.chunk_id, []).append(
            f"Referenced from page {source.page_start}, {source.title}:\n"
            + "\n".join(dict.fromkeys(evidence))
        )
    for chunk in chunks:
        if chunk.chunk_id in additions:
            chunk.text += "\n\n[Cross-page references; curve values not transcribed]\n" + "\n".join(
                dict.fromkeys(additions[chunk.chunk_id])
            )
            chunk.chunk_id = stable_id(document_id, "figure-links-v1", chunk.chunk_id, chunk.text)
    for source, target, number in edges:
        source.related_chunks.append({
            "relation": "references_figure", "figure_number": number,
            "chunk_id": target.chunk_id, "page_number": target.page_start,
            "title": target.title, "asset_ids": list(target.asset_ids),
        })
        target.related_chunks.append({
            "relation": "referenced_by", "figure_number": number,
            "chunk_id": source.chunk_id, "page_number": source.page_start,
            "title": source.title, "asset_ids": list(source.asset_ids),
        })
