from app.ingestion.parsers.figure_links import link_figures, restore_figure_names
from app.ingestion.pipeline import AssetRecord, ChunkRecord, PageRecord
from app.ingestion.publishers.ragflow import RagflowPublisher


def chunk(cid, kind, text, page, appendix="A"):
    return ChunkRecord(cid, "parent", text if kind == "figure" else "Table 1", text,
                       page, page, ["PART 23", f"APPENDIX {appendix}—LOADS"], [cid],
                       content_type=kind, asset_ids=[cid] if kind == "figure" else [])


def test_both_directions_and_exact_row_context_survive_without_merging_pages():
    table = chunk("table", "table", "n3: Find n3 from Fig. 1\nn4: Find n4 from Fig. 2", 37)
    one = chunk("one", "figure", "Figure 1 — n3 factor", 38)
    two = chunk("two", "figure", "Figure 2 — n4 factor", 38)
    link_figures("doc", [table, one, two])
    assert {r["chunk_id"] for r in table.related_chunks} == {one.chunk_id, two.chunk_id}
    assert one.related_chunks[0]["chunk_id"] == table.chunk_id
    assert "n3: Find n3 from Fig. 1" in one.text
    assert "n4: Find n4" not in one.text
    assert one.asset_ids == ["one"] and table.page_end == 37 and one.page_start == 38


def test_appendix_scoping_duplicate_targets_and_unknown_numbers():
    table = chunk("t", "table", "Figures 1 and 2; Fig. 3", 37)
    wrong = chunk("b", "figure", "Figure 1 — other appendix", 42, "B")
    dup1 = chunk("d1", "figure", "Figure 2 — first", 38)
    dup2 = chunk("d2", "figure", "Figure 2 — second", 39)
    link_figures("doc", [table, wrong, dup1, dup2])
    assert not table.related_chunks
    assert table.text == "Figures 1 and 2; Fig. 3"


def test_plural_references_resolve_in_same_appendix():
    table = chunk("t", "table", "Figures 5 and 6 of this Appendix", 37)
    figures = [chunk(str(n), "figure", f"Figure {n} — Loads", 40) for n in (5, 6)]
    link_figures("doc", [table, *figures])
    assert len(table.related_chunks) == 2


def test_explicit_other_appendix_reference_does_not_bind_local_figure():
    table = chunk("t", "table", "See Figure 1 of Appendix B", 37)
    figure = chunk("f", "figure", "Figure 1 — Local", 38)
    link_figures("doc", [table, figure])
    assert not table.related_chunks


def test_publication_carries_related_ids_and_attributed_context():
    from dataclasses import asdict

    table = chunk("t", "table", "Find n3 from Fig. 1", 37)
    figure = chunk("f", "figure", "Figure 1 — n3", 38)
    link_figures("doc", [table, figure])
    plan = RagflowPublisher.build_plan(
        dataset_id="test", source_name="sample.pdf", source_sha256="hash",
        document_ir={"chunks": [asdict(table), asdict(figure)]},
    )
    assert f"related_chunk_id:{figure.chunk_id}" in plan.chunks[0].payload["tag_kwd"]
    assert "Find n3 from Fig. 1" in plan.chunks[1].payload["content"]
    assert plan.chunks[1].payload["page_numbers"] == [38]


def test_unknown_image_is_not_named_by_page_order():
    asset = AssetRecord("image", 38, "figure", [0, 0, 1, 1], "x.jpg", "image/jpeg", b"other")
    item = {"block_type": "figure", "text": "", "asset_id": "image"}
    page = PageRecord(38, "", rich_blocks=[item])
    restore_figure_names("other-document", [page], [asset])
    assert not asset.caption and item["text"] == ""
    asset.caption = "Figure 7 — User supplied title"
    restore_figure_names("other-document", [page], [asset])
    assert item["text"] == asset.caption
