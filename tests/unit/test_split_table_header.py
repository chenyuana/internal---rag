from types import SimpleNamespace

import pytest

from app.ingestion.parsers.vector_pdf import VectorPdfParser
from app.ingestion.pipeline import PageRecord, merge_cross_page_tables


def grid(rows, box, divider):
    return SimpleNamespace(
        bbox=box,
        extract=lambda: rows,
        rows=[SimpleNamespace(cells=[
            (box[0], box[1], divider, box[3]),
            (divider, box[1], box[2], box[3]),
        ])],
    )


@pytest.mark.parametrize(
    ('header_top', 'body_top', 'divider', 'body', 'expected'),
    [
        (750, 30, 300, [['0 to 10', '20 minimum'], ['10 to 30', '40']], True),
        (350, 30, 300, [['0 to 10', '20 minimum'], ['10 to 30', '40']], False),
        (750, 300, 300, [['0 to 10', '20 minimum'], ['10 to 30', '40']], False),
        (750, 30, 380, [['0 to 10', '20 minimum'], ['10 to 30', '40']], False),
        (750, 30, 300, [['Area', 'Weight'], ['10 to 30', '40']], False),
    ],
)
def test_header_requires_adjacent_body_and_matching_columns(
    header_top, body_top, divider, body, expected,
):
    header = grid([['Area (m2)', 'Weight (kg)']], [30, header_top, 570, 790], 300)
    following = SimpleNamespace(
        width=600, height=820,
        find_tables=lambda: [grid(body, [30, body_top, 570, body_top + 100], divider)],
    )
    page = SimpleNamespace(width=600, height=820, page_number=1)
    page.pdf = SimpleNamespace(pages=[page, following])
    assert VectorPdfParser._is_split_header(page, header) is expected


def test_two_column_header_keeps_first_data_row_and_page_provenance():
    header = {
        'block_type': 'table', 'table_rows': [['Area', 'Weight']],
        'table_header_only': True, 'bbox': [30, 750, 570, 790],
    }
    body = {
        'block_type': 'table',
        'table_rows': [['0 to 10', '20 minimum'], ['10 to 30', '40'], ['30+', '60']],
        'bbox': [30, 30, 570, 150],
    }
    pages = [PageRecord(1, '', rich_blocks=[header]), PageRecord(2, '', rich_blocks=[body])]
    result = merge_cross_page_tables(pages, 'fixture')
    assert result['cross_page_table_count'] == 1
    table = pages[0].rich_blocks[0]
    assert table['table_rows'] == [['Area', 'Weight'], *body['table_rows']]
    assert table['table_row_pages'] == [1, 2, 2, 2]
    assert table['source_page_end'] == 2
    assert '<td>0 to 10</td>' in table['table_html']


def test_sparse_continuation_completes_open_row_and_keeps_page_provenance():
    header = {
        'block_type': 'table',
        'table_rows': [[
            'Foreign object', 'Test quantity', 'Speed', 'Operation', 'Ingestion',
        ]],
        'table_header_only': True,
        'bbox': [30, 750, 570, 790],
    }
    body = {
        'block_type': 'table',
        'table_rows': [
            ['Birds: 3-ounce size', 'One for each 50', 'Liftoff speed', 'Takeoff...', ''],
            ['Ice:', 'Maximum 2-minute delay', 'Sucked in', '', 'Maximum cruise...'],
        ],
        'bbox': [30, 30, 570, 700],
    }
    sparse_continuation = {
        'block_type': 'table',
        'table_rows': [['', 'anti-icing system', '', '', '']],
        'table_header_only': True,
        'bbox': [30, 30, 570, 150],
    }
    pages = [
        PageRecord(1, '', rich_blocks=[header]),
        PageRecord(2, '', rich_blocks=[body]),
        PageRecord(3, '', rich_blocks=[sparse_continuation]),
    ]

    result = merge_cross_page_tables(pages, 'fixture-open-row')

    assert result['structured_table_count'] == 1
    table = pages[0].rich_blocks[0]
    assert table['source_page_end'] == 3
    assert table['table_rows'][-1][1] == 'Maximum 2-minute delay anti-icing system'
    assert table['table_row_pages'] == [1, 2, 3]


@pytest.mark.parametrize(('mark_width', 'expected'), [(9, '≤'), (200, '<')])
def test_short_equality_stroke_is_preserved_but_table_rule_is_not(mark_width, expected):
    char = {'text': '<', 'x0': 100, 'x1': 107, 'top': 90, 'bottom': 102, 'size': 12}
    page = SimpleNamespace(
        chars=[char], rects=[{'x0': 100, 'x1': 100 + mark_width, 'top': 103, 'bottom': 103.7}],
    )
    assert VectorPdfParser._smart_cell_text(page, [0, 0, 600, 800]) == expected
    assert char['text'] == '<'
