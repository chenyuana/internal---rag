"""A remote "table" with no grid is a heading the model boxed as a table.

Docket 21-44 p. 10 printed

    § 23.397 Limit control forces and torques.

and the remote table-structure model returned it as a three-row, five-column
table (plus one OCR noise glyph), which published as a table chunk reading
"记录1：第1列=torques.".

The test is the grid itself: a table has at least one row carrying two or more
cells side by side.  It carries no row-count and no height threshold, because a
one-row table is still a table (Docket 75-19 p. 31 is exactly that).
"""

from app.ingestion.pipeline import (
    degenerate_table_reason,
    degenerate_table_text,
    table_rows_from_html,
)

HEADING_HTML = (
    "<table>"
    "<tr><td>§ 23.397 Limit control forces and</td><td></td><td></td><td></td><td></td></tr>"
    "<tr><td>torques.</td><td></td><td></td><td></td><td></td></tr>"
    "<tr><td></td><td></td><td>电</td><td></td><td></td></tr>"
    "</table>"
)
# The genuine table printed immediately below it on the same page.
REAL_HTML = (
    "<table>"
    "<tr><td>Control</td><td>Maximum forces</td><td>Minimum forces</td></tr>"
    "<tr><td>Aileron: Stick.</td><td>67 lbs.</td><td>40 lbs.</td></tr>"
    "<tr><td>Rudder.</td><td>200 lbs.</td><td>130 lbs.</td></tr>"
    "</table>"
)


def test_boxed_heading_has_no_grid() -> None:
    rows = table_rows_from_html(HEADING_HTML)

    assert degenerate_table_reason(rows) == "no_grid"
    assert degenerate_table_text(rows) == "§ 23.397 Limit control forces and torques. 电"


def test_a_one_row_table_is_still_a_table() -> None:
    # Docket 75-19 p. 31: a single row of column labels whose data rows continue
    # on the next page.  It has a grid, so it stays a table at any row count.
    header_only = table_rows_from_html(
        "<table><tr><td>14 CFR (FAR Sec.)</td><td>Proposal No.</td>"
        "<td>Agenda Item</td><td>Proponent</td></tr></table>"
    )
    assert degenerate_table_reason(header_only) is None

    # A single data row is also a grid.
    one_data_row = table_rows_from_html(
        "<table><tr><td>Stick</td><td>40 lbs.</td></tr></table>"
    )
    assert degenerate_table_reason(one_data_row) is None


def test_real_tables_are_never_demoted() -> None:
    assert degenerate_table_reason(table_rows_from_html(REAL_HTML)) is None
    two_by_two = table_rows_from_html(
        "<table><tr><td>Control</td><td>Force</td></tr>"
        "<tr><td>Stick</td><td>40 lbs.</td></tr></table>"
    )
    assert degenerate_table_reason(two_by_two) is None


def test_a_long_block_with_one_cell_per_row_is_not_a_table() -> None:
    # The grid test needs no row or height cap: however tall it is, a block whose
    # every row holds one cell is a column of text, not a table.  The demotion is
    # lossless -- the cells become the block's text.
    column = table_rows_from_html(
        "<table>"
        "<tr><td>Proposal No. 18</td></tr><tr><td>The proponent suggested</td></tr>"
        "<tr><td>an environmental impact</td></tr><tr><td>statement be required</td></tr>"
        "</table>"
    )
    assert degenerate_table_reason(column) == "no_grid"
    assert degenerate_table_text(column).startswith("Proposal No. 18 The proponent suggested")
