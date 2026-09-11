"""Rebuilding a table whose row separators were lost.

Docket 21-44 p. 12 printed a 14-row vapour-pressure correction table.  The remote
table model returned it as a header row plus ONE data row whose four cells were
the four columns run together, and the pipeline published records like
``AltitudeH(i.)=01,0002,0003,0004,...``.

The rebuild is decided by the columns agreeing with each other -- never by a
document hash, a document name, or a word list.
"""

from collections import Counter

from app.ingestion.pipeline import (
    collapsed_table_rows,
    split_collapsed_table_rows,
)

HEADER = [
    "AltitudeH(i.)",
    "Vaporpressuree (In. Hg.)",
    "Specific hu-midity w(Lb. mois-ture per Ib.dry air)",
    "Densityratiopσ=0.0023769",
]
# The four cells exactly as the remote model returned them.  Note the vapour
# column: the OCR also lost the decimal point between 0.1172 and 0.1010
# ("…1356.11721010.0463…"), which the reader has to put back.
COLLAPSED = [
    "01,0002,0003,0004,0005,0006,0007,0008.0009,00010,00015,00020.00025,000",
    "0.403.354.311.272.238.207.1805.1568.1356.11721010.0463.01978.00778",
    "0.00849.00773.00703.00638.00578.00523.00472.00425.00382.00343.00307.001710.000896.000436",
    "0.99508.96672.93895.91178.88514.85910.83361.80870.78134.76053.73722.62868.53263.44806",
]


def test_the_collapsed_shape_is_recognised() -> None:
    assert collapsed_table_rows([HEADER, COLLAPSED]) is True


def test_an_ordinary_two_row_table_is_not_flagged() -> None:
    # Several numbers in a cell are normal for a small real table, so the
    # predicate that gates a quarantine stays narrow.
    ordinary = [
        ["Control", "Maximum force", "Minimum force"],
        ["Aileron: Stick.", "100 lbs. 67 lbs.", "40 lbs. 40 lbs."],
    ]
    assert collapsed_table_rows(ordinary) is False
    assert collapsed_table_rows([HEADER]) is False
    assert collapsed_table_rows([HEADER, COLLAPSED, COLLAPSED]) is False


def test_the_printed_table_is_rebuilt_from_column_agreement() -> None:
    rebuilt = split_collapsed_table_rows([HEADER, COLLAPSED])

    assert rebuilt is not None
    assert rebuilt[0] == HEADER
    assert len(rebuilt) == 15  # header + 14 printed rows
    altitudes = [row[0] for row in rebuilt[1:]]
    assert altitudes == [
        "0", "1,000", "2,000", "3,000", "4,000", "5,000", "6,000", "7,000",
        "8,000", "9,000", "10,000", "15,000", "20,000", "25,000",
    ]
    # Standard-atmosphere density ratio at 25,000 ft is 0.4481, which is what the
    # fourth column ends on: an independent check that the split is the real one.
    assert rebuilt[-1][3] == "0.44806"
    assert [row[1] for row in rebuilt[1:]] == [
        "0.403", "0.354", "0.311", "0.272", "0.238", "0.207", "0.1805",
        "0.1568", "0.1356", "0.1172", "0.101", "0.0463", "0.01978", "0.00778",
    ]
    assert rebuilt[13][0] == "20,000"  # the altitude column keeps its grouping
    assert [row[2] for row in rebuilt[1:]] == [
        "0.00849", "0.00773", "0.00703", "0.00638", "0.00578", "0.00523",
        "0.00472", "0.00425", "0.00382", "0.00343", "0.00307", "0.00171",
        "0.000896", "0.000436",
    ]
    assert [row[3] for row in rebuilt[1:]] == [
        "0.99508", "0.96672", "0.93895", "0.91178", "0.88514", "0.85910",
        "0.83361", "0.80870", "0.78134", "0.76053", "0.73722", "0.62868",
        "0.53263", "0.44806",
    ]
    # No digit is lost: the rebuilt columns carry at least every digit of their
    # source cell (the extra zeros are the ones the OCR had dropped after the
    # first value of a run, and the grouped column only normalises its separator).
    for index, cell in enumerate(COLLAPSED):
        printed = Counter("".join(row[index] for row in rebuilt[1:]))
        source = Counter(cell.replace(".", ",") if "," in cell else cell)
        assert all(printed[ch] >= count for ch, count in source.items()), f"column {index}"


def test_an_ambiguous_run_is_refused() -> None:
    # No column here can be read monotonically into agreeing counts, so the
    # caller must quarantine rather than guess.
    ambiguous = [
        ["Total", "Change"],
        ["12 34 56 78", "9 87 65 43 21"],
    ]
    assert split_collapsed_table_rows(ambiguous) is None
