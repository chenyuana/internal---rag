from app.ingestion.parsers.scan_regulatory import ScannedRegulatoryPdfParser
from app.ingestion.pipeline import PageRecord


def test_six_figures_isolated_from_next_appendix():
    parser = ScannedRegulatoryPdfParser(None)
    pages = [PageRecord(37, "", rich_blocks=[
        {"block_type": "heading", "text": "APPENDIX A—DESIGN LOADS"}
    ])]
    for number in (38, 39, 40):
        pages.append(PageRecord(number, "", rich_blocks=[
            {"block_type": "figure", "text": "", "asset_id": f"{number}-{i}",
             "asset_ids": [f"{number}-{i}"]} for i in (1, 2)
        ]))
    pages.append(PageRecord(41, "", rich_blocks=[
        {"block_type": "paragraph", "text": "APPENDIX BCONTROL SURFACE LÒADINGS"},
        {"block_type": "paragraph", "text": "B23.1 General."},
        {"block_type": "paragraph", "text": "The control surface loads."},
    ]))
    blocks = parser._build_blocks("doc", pages)
    chunks = parser._build_chunks("doc", blocks)
    figures = [c for c in chunks if c.content_type == "figure"]
    assert len(figures) == 6
    assert all(c.page_start == c.page_end and len(c.asset_ids) == 1 for c in figures)
    assert all("APPENDIX A" in c.section_path[-1] for c in figures)
    assert all("B23.1" not in c.text for c in figures)
    body = [c for c in chunks if "B23.1" in c.text]
    assert len(body) == 1
    assert body[0].page_start == 41 and not body[0].asset_ids
    assert body[0].section_path[-1] == "APPENDIX B—CONTROL SURFACE LÒADINGS"


def test_joined_appendix_requires_matching_following_section():
    parser = ScannedRegulatoryPdfParser(None)
    page = PageRecord(1, "", rich_blocks=[
        {"block_type": "paragraph", "text": "APPENDIX BCONTROL SURFACE LOADINGS"},
        {"block_type": "paragraph", "text": "A23.1 General."},
    ])
    assert parser._build_blocks("doc", [page])[0].block_type == "paragraph"
