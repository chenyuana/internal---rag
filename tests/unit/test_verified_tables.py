from types import SimpleNamespace
from unittest.mock import Mock

from app.ingestion.parsers.scan_regulatory import ScannedRegulatoryPdfParser
from app.ingestion.parsers.verified_tables import (
    DOCKET4080_SHA256,
    DRS_88_1_SHA256,
    DRS_98_19_SHA256,
    FINAL_RULE_2005_20245_SHA256,
    verified_hybrid_rows,
    verified_table,
)
from app.ingestion.pipeline import PageRecord, table_rows_from_html

RAW = (
    "<table><tr><td>Values in pounds of force asapplied to the control wheelor rudder pedals"
    "</td><td>Pitch</td><td>Roll</td><td>Yaw</td></tr><tr><td>"
    "(a) For temporary applica-tion:StickWheel (applied to rim).Rudder pedal."
    "(b) For prolonged application.</td><td>607510</td><td>30605</td><td>15020</td></tr></table>"
)


def test_visual_correction_preserves_empty_cells_and_prolonged_row():
    fixed = verified_table(DOCKET4080_SHA256, 6, RAW)
    assert fixed
    assert table_rows_from_html(fixed)[1:] == [
        ["(a) For temporary application:", "", "", ""],
        ["Stick", "60", "30", ""],
        ["Wheel (applied to rim)", "75", "60", ""],
        ["Rudder pedal", "", "", "150"],
        ["(b) For prolonged application.", "10", "5", "20"],
    ]


def test_different_document_page_or_numeric_evidence_never_uses_correction():
    assert verified_table("different", 6, RAW) is None
    assert verified_table(DOCKET4080_SHA256, 7, RAW) is None
    assert verified_table(DOCKET4080_SHA256, 6, RAW.replace("607510", "607511")) is None
    assert verified_table(DOCKET4080_SHA256, 6, RAW.replace("Pitch", "Other")) is None


def test_clean_mode_uses_correction_without_remote_ocr_and_retains_raw(tmp_path):
    client = SimpleNamespace(ocr_table_lines=Mock(side_effect=AssertionError("No OCR")))
    parser = ScannedRegulatoryPdfParser(client)
    item = {"text": RAW, "table_html": RAW, "block_type": "table"}
    page = PageRecord(6, RAW, rich_blocks=[item])
    parser._recover_tables(tmp_path / "unused.pdf", [page], allow_ocr=False,
                           source_hash=DOCKET4080_SHA256)
    assert item["table_review_status"] == "recovered"
    assert item["raw_table_html"] == RAW
    assert item["raw_table_text"] == RAW
    assert "withheld" not in item["text"]
    assert item["table_recovery_method"] == "source_visual_review"
    client.ocr_table_lines.assert_not_called()


def test_visually_verified_final_rule_tables_restore_missing_rows_and_columns():
    page_28 = [
        ["Assumptior/parameter", "Final rule", "Proposal"],
        ["Part121Airplanes:NumberofMagneticTapeCVRstobereplaced", "2,941", "5,904"],
        ["PriceofAviationFuel", "1.60", "$0.75"],
    ]
    page_37_appendix_e = [
        ["Parameters", "Range", "Installed system 1 minimum accuracy", "Sampling", "Resolution4"],
        ["Stabilizer Trim Position or Pitch Con- Full Range", "", "±3%", "1", "31"],
    ]
    page_37_appendix_f = [
        ["Parameters", "Range", "Resolution3"],
        ["Collective 4", "Full Range", "21"],
        ["Controllable Stabilator Position 4", "Full Range", "21"],
    ]
    page_39 = [
        ["Parameters", "Range", "Accuracy", "Seconds", "Resolution", "Remarks"],
        ["Time or relative times counts.1", "", "24 Hrs, 0 to 4095", "", "1 sec", "operation."],
        ["12a. Pitch control(s) position", "Full Range", "±2°", "0.5", "0.5%", ""],
        ["13b. Lateral control position(s)", "Full Range", "±2°", "0.5", "0.2%", ""],
    ]

    repaired_28 = verified_hybrid_rows(FINAL_RULE_2005_20245_SHA256, 28, page_28)
    repaired_e = verified_hybrid_rows(FINAL_RULE_2005_20245_SHA256, 37, page_37_appendix_e)
    repaired_f = verified_hybrid_rows(FINAL_RULE_2005_20245_SHA256, 37, page_37_appendix_f)
    repaired_39 = verified_hybrid_rows(FINAL_RULE_2005_20245_SHA256, 39, page_39)

    assert len(repaired_28) == 23
    assert repaired_28[5] == [
        "Number of 30-Minute Memory Solid State CVRs to be replaced", "4,634", "3,741"
    ]
    assert repaired_e == [
        ["Parameters", "Range", "Installed system¹ minimum accuracy (to recovered data)",
         "Sampling interval (per second)", "Resolution⁴ read out (percent)"],
        ["Stabilizer Trim Position or Pitch Control Position⁵", "Full Range",
         "±3% unless higher uniquely required", "1", "³1"],
    ]
    assert repaired_f[3] == ["Lat. Cyclic⁴", "Full Range", "±3", "2", "²1"]
    assert repaired_39[1][1:5] == ["24 Hrs, 0 to 4095", "±0.125% per hour", "4", "1 sec"]
    assert repaired_39[-1][0] == "13b. Lateral control position(s) (fly-by-wire).⁴ ¹⁸"


def test_final_rule_visual_correction_never_applies_without_matching_source_page_and_evidence():
    rows = [["Parameters", "Range"], ["Collective", "Full Range"]]

    assert verified_hybrid_rows("different", 37, rows) == rows
    assert verified_hybrid_rows(FINAL_RULE_2005_20245_SHA256, 36, rows) == rows
    assert verified_hybrid_rows(FINAL_RULE_2005_20245_SHA256, 37, rows) == rows


def test_drs_verified_rows_restore_ambiguous_or_subrows_without_filling_blanks():
    rows = [
        ["Parameters", "Range", "Accuracy", "Sampling", "Resolution"],
        ["Stabilizer Trim PositionORPitch Control Position", "Full RangeFull Range",
         "±3% unless higher uniquely required±3% unless higher uniquely required", "11", "1%.1%."],
        ["ENGINE POWER, EACH ENGINE"] * 5,
        ["Fan or N1 Speed or EPRor Cockpit indications Used for Aircraft Certification."
         "ORProp. speed and Torque", "MaximumRange", "+5%",
         "11 (prop Speed)1 (torque)", "1%"],
        ["TE Flaps (Discrete or Analog)",
         "Each discrete position OR Analog 0-100%", "+3", "11", "1%"],
    ]
    repaired = verified_hybrid_rows(DRS_88_1_SHA256, 37, rows)
    assert repaired[1:3] == [
        ["Stabilizer Trim Position", "Full Range",
         "±3% unless higher uniquely required", "1", "1%."],
        ["Pitch Control Position", "Full Range",
         "±3% unless higher uniquely required", "1", "1%."],
    ]
    prop = next(row for row in repaired if row[0].startswith("Prop speed"))
    assert prop[1:] == ["", "", "1 (prop speed); 1 (torque)", ""]
    analog = next(row for row in repaired if row[0] == "TE Flaps (Analog)")
    assert analog == ["TE Flaps (Analog)", "Analog 0-100% range", "+3", "1", ""]


def test_drs_verified_rows_are_hash_page_and_evidence_scoped():
    rows = [["Stabilizer Trim PositionORPitch Control Position", "Full RangeFull Range",
             "±3%±3%", "11", "1%1%"]]
    assert verified_hybrid_rows("different", 37, rows) == rows
    assert verified_hybrid_rows(DRS_88_1_SHA256, 36, rows) == rows
    assert verified_hybrid_rows(DRS_88_1_SHA256, 37, rows) == rows


def test_drs_page_41_verified_rows_preserve_table_columns():
    rows = [
        ["AFCS Mode andEngagement Status", "Discrete (5 bitsNecessary)", "", "1", ""],
        ["DME 1 and 2 Distance", "0-200 NM;", "As installed", "0.25", "1Mi"],
        ["Outside Air Temperature", "-90 C to +50 C", "±2C", "0.5", "0.3 C"],
    ]
    repaired = verified_hybrid_rows(DRS_88_1_SHA256, 41, rows)
    assert repaired == [
        ["AFCS Mode and Engagement Status", "Discrete (5 bits Necessary)", "", "1", ""],
        ["DME 1 and 2 Distance", "0-200 NM", "As installed", "0.25", "1 Mi"],
        ["Outside Air Temperature", "-90 C to +50 C", "±2 C", "0.5", "0.3 C"],
    ]


def test_drs_cost_table_restores_totals_to_their_numeric_rows():
    rows = [
        ["", "Undiscounted", "Discounted present value"],
        ["Part 91-Airplanes", "", ""],
        ["Fuel consumption", "13,080,000", "9,918,000"],
        ["Total", "", ""],
        ["Total cost for all operating rules", "314,658,000", "222,339,000"],
    ]
    repaired = verified_hybrid_rows(DRS_88_1_SHA256, 12, rows)

    assert repaired[4] == ["Fuel consumption", "12,124,000", "7,212,000"]
    assert repaired[5] == [
        "Certification, start up, and support equipment (10 percent of equipment/installation)",
        "13,080,000",
        "9,918,000",
    ]
    assert repaired[6] == ["Total", "160,915,000", "119,228,000"]
    assert repaired[-1] == [
        "Total cost for all operating rules", "314,658,000", "222,339,000"
    ]


def test_drs_98_19_page7_not_hash_corrected_general_crop_handles_it():
    # Page 7's tiny table is fixed by the general high-resolution crop re-OCR,
    # NOT by a source-hash correction. verified_hybrid_rows must leave it
    # untouched so the general mechanism is the one that applies.
    rows = [
        ["Weight of bird", "Number of birds"],
        ["1.0-1.5", ""],
        ["1.5–2.5", "3"],
        ["2.5+", "3 2"],
    ]
    assert verified_hybrid_rows(DRS_98_19_SHA256, 7, rows) == rows


def test_drs_98_19_page25_table2_rows_are_realigned():
    # MinerU drops the first row's "3", fuses quantities/weights, and detaches
    # the "Plus N" continuation rows.
    header = ["Engine inlet area (A) square-meters (square-inches)",
              "Bird quantity", "Bird weight kg. (lb.)"]
    rows = [
        header,
        ["0.05 (77.5) A", "0.05 (77.5) A", "0.05 (77.5) A"],
        [".05 (77.5)≤ A < 0.10 (155)", "None 1", "0.35 (0.77). 0.45 (0.99)."],
        ["0.10 (155)≤ A < 0.20 (310)", "1", ""],
        ["0.20 (310)≤ A < 0.40 (620)", "", "2"],
        ["4.50 (6975)≤ A", "4", "1.15 (2.53)."],
    ]
    repaired = verified_hybrid_rows(DRS_98_19_SHA256, 25, rows)
    assert repaired[0] == header
    assert repaired[1] == ["0.05 (77.5)> A", "None", ""]
    assert repaired[2] == [".05 (77.5)≤ A < 0.10 (155)", "1", "0.35 (0.77)"]
    assert repaired[3] == ["0.10 (155)≤ A < 0.20 (310)", "1", "0.45 (0.99)"]
    assert repaired[4] == ["0.20 (310)≤ A < 0.40 (620)", "2", "0.45 (0.99)"]
    assert repaired[-1] == ["4.50 (6975)≤ A", "4", "1.15 (2.53)"]
    assert len(repaired) == 18


def test_drs_98_19_page25_table3_and_foreign_object_are_recovered():
    header = ["Engine inlet area (A) square-meters (square-inches)",
              "Bird quantity", "Bird weight kg. (lb.)"]
    table3 = [
        header,
        ["1.35 (2,092)> A", "None", ""],
        ["1.35 (2,092)≤ A 2.90 (4,495)", "", "1.15 (2.53)."],
        ["3.90 (6,045)≤ A ....", "2", "1.15 (2.53)."],
    ]
    repaired3 = verified_hybrid_rows(DRS_98_19_SHA256, 25, table3)
    assert repaired3 == [
        header,
        ["1.35 (2,092)> A", "None", ""],
        ["1.35 (2,092)≤ A < 2.90 (4,495)", "1", "1.15 (2.53)"],
        ["2.90 (4,495)≤ A < 3.90 (6,045)", "2", "1.15 (2.53)"],
        ["3.90 (6,045)≤ A", "1", "1.15 (2.53)"],
        ["", "Plus 6", "0.70 (1.54)"],
    ]

    foreign_object = [
        ["Foreignobject", "Test quantity", "Speed of foreignobject",
         "Engine operation", "Ingestion"],
        ["Ice …", "Maximum accumulation on a typical inletcowl", "Sucked in ..",
         "Maximum cruise .....", "icingencounter"],
    ]
    repaired_foreign = verified_hybrid_rows(DRS_98_19_SHA256, 25, foreign_object)
    assert repaired_foreign[0] == [
        "Foreign object", "Test quantity", "Speed of foreign object",
        "Engine operation", "Ingestion",
    ]
    assert repaired_foreign[1][0] == "Ice"
    assert repaired_foreign[1][2] == "Sucked in"
    assert repaired_foreign[1][3] == "Maximum cruise"
    assert repaired_foreign[1][4] == (
        "To simulate a continuous maximum icing encounter at 25 degrees Fahrenheit."
    )


def test_drs_appendix_tables_replace_cross_page_flattening():
    appendix_d = [
        ["Parameters", "Range", "Accuracy", "Sampling", "Resolution"],
        ["ENGINE POWER, EACH ENGINE"] * 5,
        ["Autopilot engaged (discrete)", "Engaged or disengaged", "", "1", ""],
    ]
    appendix_e_continuation = [
        ["", "±10% Resolution 250 1 250 fpm ENGINE POWER"] * 5,
        ["Main roto speed", "Maximum range", "±5%", "1", "1%"],
        ["FLIGHT CONTROLS"] * 5,
        ["Controllable stabilator position", "Full range", "±3%", "2", "1%"],
    ]

    repaired_d = verified_hybrid_rows(DRS_88_1_SHA256, 23, appendix_d)
    repaired_e = verified_hybrid_rows(DRS_88_1_SHA256, 26, appendix_e_continuation)

    assert len(repaired_d) == 24
    assert repaired_d[2][2].endswith("below 175 KIAS.")
    assert repaired_d[-1][0] == "Autopilot engaged (discrete)"
    assert repaired_e[0][0] == "Altitude rate"
    assert repaired_e[1] == ["ENGINE POWER, EACH ENGINE", "", "", "", ""]
    assert repaired_e[-1][0] == "Controllable stabilator position"
