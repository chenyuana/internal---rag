"""Conservative human review gate; successful parsing is not verification."""
from typing import Any


def table_review_gate(pages: list[Any], diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    reasons: dict[int, set[str]] = {}

    def add(number: int, reason: str) -> None:
        reasons.setdefault(number, set()).add(reason)

    for number in diagnostics.get('merged_cell_review_pages', []):
        add(int(number), '结构或单元格内容存在风险')
    for page in pages:
        # Retained OCR evidence survives canonical rebuilding and sample repairs.
        for item in page.layout_features.get('ocr_blocks', []):
            if item.get('block_type') == 'table':
                add(page.page_number, 'OCR 表格需对照原图核对数字和行列对应')
        for item in page.rich_blocks:
            if item.get('block_type') != 'table':
                continue
            if page.route in {'remote_ocr', 'scan_regulatory_ocr'}:
                add(page.page_number, 'OCR 表格需对照原图核对数字和行列对应')
            start = int(item.get('source_page_start') or page.page_number)
            end = int(item.get('source_page_end') or start)
            if end > start:
                for number in range(start, end + 1):
                    add(number, '跨页表格需核对续行、重复表头和内容完整性')
            if item.get('table_merged_cell_count', 0):
                add(page.page_number, '合并单元格需核对内容归属')
    if not reasons:
        return None
    return {
        'status': 'fail',
        'gate': 'table_manual_review',
        'pages': sorted(reasons),
        'review_reasons': {str(n): sorted(reasons[n]) for n in sorted(reasons)},
        'message': (
            '表格尚未经人工核验，已阻止发布。请对照原始 PDF/表格截图检查数字、'
            '行列对应、空白格及跨页衔接；确认或修正后使用“人工确认无误”放行。'
            f'需复核页：{sorted(reasons)}'
        ),
    }
