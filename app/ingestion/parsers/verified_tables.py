"""Source-specific, visually checked corrections; never general OCR heuristics.

Each correction is restricted to immutable PDF bytes, page, and original cells.
The source evidence remains in the IR. New documents must not inherit values.
"""

import re
from html import escape

from app.ingestion.pipeline import table_rows_from_html

DOCKET4080_SHA256 = "584f6dda15b8abe37bc5d3388b2e26c311c786a531afd939f4c23b6de658cd16"
DRS_88_1_SHA256 = "bf3738181740c26505f625e19a63f29c68b12a450463314559cbe59963fce3d0"
FINAL_RULE_2005_20245_SHA256 = (
    "43418b5c19a3341816c3c8df4a5d32eb35eb6b55c2e2a24a71fa5b47bd598ebc"
)
DRS_98_19_SHA256 = "848359e52d7e39cee37f107f586c0f2517ecd7ebfed05ae3037e08334f046f95"
FINAL_RULE_26324_SHA256 = (
    "a4f7ecc693790d20f36ee84911d3c3d52b8d08c12d914c85240bb7e91e7dbef5"
)
AISLE_ORIGINAL_CELLS = [
    ["Number of pas-senger seats", "Minimum main passengeraisle width (inches)",
     "Minimum main passengeraisle width (inches)"],
    ["Number of pas-senger seats", "Less than25 inchesfrom floor",
     "25 inchesand morefrom floor"],
    ["10 or fewer ....11 through 19 ...", "11212", "1520"],
]
# Visually checked against page 10. The first 1 is a superscript footnote,
# not part of the width. Flatten the spanning header without losing units.
AISLE_VERIFIED_CELLS = [
    ["Number of passenger seats",
     "Minimum main passenger aisle width (inches) / Less than 25 inches from floor",
     "Minimum main passenger aisle width (inches) / 25 inches and more from floor"],
    ["10 or fewer", "¹12", "15"],
    ["11 through 19", "12", "20"],
]

# Page 25 of 23-54-DRS_98-19 renders three full-width raster tables whose rows
# and columns MinerU misaligns (the "Bird quantity" / "Bird weight kg. (lb.)"
# values land in the wrong cells, and "Plus N" continuation rows detach from
# their inlet-area rows). Transcribed cell-by-cell from the source pixels and
# confirmed against the Federal Register text (63 FR 68645-68646).
#
# Note: page 7's tiny bird-quantity table is NOT corrected here — it is fixed
# generally by the higher-resolution crop re-OCR in HybridPdfParser.

# Table 2.--Medium Flocking Bird Weight and Quantity Requirements.
_DRS_98_19_INLET_AREA_HEADER = [
    "Engine inlet area (A) square-meters (square-inches)",
    "Bird quantity",
    "Bird weight kg. (lb.)",
]
DRS_98_19_PAGE_25_TABLE_2_ROWS = [
    _DRS_98_19_INLET_AREA_HEADER,
    ["0.05 (77.5)> A", "None", ""],
    [".05 (77.5)≤ A < 0.10 (155)", "1", "0.35 (0.77)"],
    ["0.10 (155)≤ A < 0.20 (310)", "1", "0.45 (0.99)"],
    ["0.20 (310)≤ A < 0.40 (620)", "2", "0.45 (0.99)"],
    ["0.40 (620)≤ A < 0.60 (930)", "2", "0.70 (1.54)"],
    ["0.60 (930)≤ A < 1.00 (1,550)", "3", "0.70 (1.54)"],
    ["1.00 (1,550)≤ A < 1.35 (2,092)", "4", "0.70 (1.54)"],
    ["1.35 (2,092)≤ A < 1.70 (2,635)", "1", "1.15 (2.53)"],
    ["", "Plus 3", "0.70 (1.54)"],
    ["1.70 (2,635)≤ A < 2.10 (3,255)", "1", "1.15 (2.53)"],
    ["", "Plus 4", "0.70 (1.54)"],
    ["2.10 (3,255)≤ A < 2.50 (3,875)", "1", "1.15 (2.53)"],
    ["", "Plus 5", "0.70 (1.54)"],
    ["2.50 (3,875)≤ A < 3.90 (6045)", "1", "1.15 (2.53)"],
    ["", "Plus 6", "0.70 (1.54)"],
    ["3.90 (6045)≤ A < 4.50 (6975)", "3", "1.15 (2.53)"],
    ["4.50 (6975)≤ A", "4", "1.15 (2.53)"],
]

# Table 3.--Additional Integrity Assessment.
DRS_98_19_PAGE_25_TABLE_3_ROWS = [
    _DRS_98_19_INLET_AREA_HEADER,
    ["1.35 (2,092)> A", "None", ""],
    ["1.35 (2,092)≤ A < 2.90 (4,495)", "1", "1.15 (2.53)"],
    ["2.90 (4,495)≤ A < 3.90 (6,045)", "2", "1.15 (2.53)"],
    ["3.90 (6,045)≤ A", "1", "1.15 (2.53)"],
    ["", "Plus 6", "0.70 (1.54)"],
]

# The 33.77(e) foreign-object ingestion grid: MinerU fuses word boundaries
# ("Foreignobject", "Sucked in ..", "Maximum cruise .....", "icingencounter").
DRS_98_19_PAGE_25_FOREIGN_OBJECT_ROWS = [
    ["Foreign object", "Test quantity", "Speed of foreign object", "Engine operation", "Ingestion"],
    [
        "Ice",
        "Maximum accumulation on a typical inlet cowl and engine face resulting "
        "from a 2-minute delay in actuating anti-icing system, or a slab of ice "
        "which is comparable in weight or thickness for that size engine.",
        "Sucked in",
        "Maximum cruise",
        "To simulate a continuous maximum icing encounter at 25 degrees Fahrenheit.",
    ],
]
ORIGINAL_CELLS = [
    ["Values in pounds of force as applied to the control wheel or rudder pedals",
     "Pitch", "Roll", "Yaw"],
    ["(a) For temporary applica-tion:Stick Wheel (applied to rim).Rudder pedal."
     "(b) For prolonged application.", "607510", "30605", "15020"],
]
# Transcribed from the source pixels, including intentionally empty cells.
# In particular 10 / 5 / 20 belong to prolonged application, not Rudder pedal.
VERIFIED_CELLS = [
    ORIGINAL_CELLS[0],
    ["(a) For temporary application:", "", "", ""],
    ["Stick", "60", "30", ""],
    ["Wheel (applied to rim)", "75", "60", ""],
    ["Rudder pedal", "", "", "150"],
    ["(b) For prolonged application.", "10", "5", "20"],
]

# The following tables were transcribed from the source PDF pixels after the
# upstream table model collapsed rows/columns. They are deliberately limited
# by source hash, page number, and identifiable OCR evidence below.
FINAL_RULE_PAGE_28_ROWS = [
    ["Assumption/parameter", "Final rule", "Proposal"],
    ["Present Value (7%) of Total Costs", "$169", "$256"],
    ["Time Frame for Analysis", "11 Years (2007-2017)", "20 Years (2003-2022)"],
    ["Part 121 Airplanes:", "", ""],
    ["Number of Magnetic Tape CVRs to be replaced", "2,941", "5,904"],
    ["Number of 30-Minute Memory Solid State CVRs to be replaced", "4,634", "3,741"],
    ["Number of Production Airplanes with 30-Minute Memory Recorders", "394", "13,658"],
    ["Percent of All Production Airplanes with 30-Minute Memory Recorders", "10%", "100%"],
    ["Cost of Increased Memory/2 hours", "$1,500", "$3,500"],
    ["Need RIPS (number of aircraft)", "3,935", "13,658"],
    ["Cost of RIPS", "$4,180", "$2,820"],
    ["Record CPDLC (number of aircraft)", "1,181", "13,658"],
    ["Percent that will Record CPDLC", "20%", "100%"],
    ["Increased FDR and DFDAU Capacity", "3,935", "13,658"],
    ["Large Production Helicopters:", "", ""],
    ["Number of Production Helicopters with 30-Minute Memory CVRs", "0", "1,337"],
    ["Need RIPS (number of aircraft)", "259", "1,337"],
    ["Record CPDLC (number of aircraft)", "0", "1,337"],
    ["Business Jets:", "", ""],
    ["Number of Production Business Jets for which costs were estimated", "3,575", "0"],
    ["Miscellaneous:", "", ""],
    ["Assumption/parameter", "Final rule", "Proposal"],
    ["Price of Aviation Fuel", "$1.60", "$0.75"],
]

FINAL_RULE_PAGE_37_APPENDIX_E_ROWS = [
    [
        "Parameters",
        "Range",
        "Installed system¹ minimum accuracy (to recovered data)",
        "Sampling interval (per second)",
        "Resolution⁴ read out (percent)",
    ],
    [
        "Stabilizer Trim Position or Pitch Control Position⁵",
        "Full Range",
        "±3% unless higher uniquely required",
        "1",
        "³1",
    ],
]

FINAL_RULE_PAGE_37_APPENDIX_F_ROWS = [
    [
        "Parameters",
        "Range",
        "Installed system¹ minimum accuracy (to recovered data) (in percent)",
        "Sampling interval (per second)",
        "Resolution³ read out (in percent)",
    ],
    ["Collective⁴", "Full Range", "±3", "2", "²1"],
    ["Pedal Position⁴", "Full Range", "±3", "2", "²1"],
    ["Lat. Cyclic⁴", "Full Range", "±3", "2", "²1"],
    ["Long. Cyclic⁴", "Full Range", "±3", "2", "²1"],
    ["Controllable Stabilator Position⁴", "Full Range", "±3", "2", "²1"],
]

_BREAKAWAY_REMARK = (
    "For airplanes that have a flight control breakaway capability that allows "
    "either pilot to operate the controls independently, record both control "
    "inputs. The control inputs may be sampled alternately once per second to "
    "produce the sampling interval of 0.5 or 0.25, as applicable."
)

FINAL_RULE_PAGE_39_ROWS = [
    [
        "Parameters",
        "Range",
        "Accuracy (sensor input)",
        "Seconds per sampling interval",
        "Resolution",
        "Remarks",
    ],
    [
        "1. Time or relative times counts.¹",
        "24 Hrs, 0 to 4095",
        "±0.125% per hour",
        "4",
        "1 sec",
        "UTC time preferred when available. Count increments each 4 seconds of system operation.",
    ],
    [
        "12a. Pitch control(s) position (nonfly-by-wire systems).¹⁸",
        "Full Range",
        "±2° unless higher accuracy uniquely required.",
        "0.5 or 0.25 for airplanes operated under § 121.344(f).",
        "0.5% of full range",
        _BREAKAWAY_REMARK,
    ],
    [
        "12b. Pitch control(s) position (fly-by-wire systems).³ ¹⁸",
        "Full Range",
        "±2° unless higher accuracy uniquely required.",
        "0.5 or 0.25 for airplanes operated under § 121.344(f).",
        "0.2% of full range",
        "",
    ],
    [
        "13a. Lateral control position(s) (nonfly-by-wire).¹⁸",
        "Full Range",
        "±2° unless higher accuracy uniquely required.",
        "0.5 or 0.25 for airplanes operated under § 121.344(f).",
        "0.2% of full range",
        _BREAKAWAY_REMARK,
    ],
    [
        "13b. Lateral control position(s) (fly-by-wire).⁴ ¹⁸",
        "Full Range",
        "±2° unless higher accuracy uniquely required.",
        "0.5 or 0.25 for airplanes operated under § 121.344(f).",
        "0.2% of full range",
        "",
    ],
]

# DRS_88-1 tables checked cell-by-cell against the rendered source pages.
# Keep source typography (including apparent source typos such as ``Main roto``
# and ``Latitude cyclic``) rather than silently editorialising the regulation.
DRS_PAGE_12_13_COST_ROWS = [
    ["", "Undiscounted", "Discounted present value"],
    ["Part 91-Airplanes:", "", ""],
    ["Equipment/Installation", "130,805,000", "99,184,000"],
    ["Maintenance", "4,906,000", "2,914,000"],
    ["Fuel consumption", "12,124,000", "7,212,000"],
    ["Certification, start up, and support equipment (10 percent of equipment/installation)",
     "13,080,000", "9,918,000"],
    ["Total", "160,915,000", "119,228,000"],
    ["Part 91-Rotorcraft:", "", ""],
    ["Equipment/Installation", "291,000", "151,000"],
    ["Maintenance", "785,000", "410,000"],
    ["Fuel consumption", "1,880,000", "1,096,000"],
    ["Certification, start up, and support equipment (10 percent of equipment/installation)",
     "18,802,000", "10,955,000"],
    ["Total", "21,758,000", "12,612,000"],
    ["Part 135-Airplanes:", "", ""],
    ["Equipment/Installation", "2,645,000", "1,539,000"],
    ["Maintenance", "7,593,000", "4,480,000"],
    ["Fuel consumption", "5,363,000", "3,862,000"],
    ["Certification, start up, and support equipment (10 percent of equipment/installation)",
     "53,633,000", "38,622,000"],
    ["Total", "69,234,000", "48,503,000"],
    ["Part 135-Rotorcraft:", "", ""],
    ["Equipment/Installation", "1,114,000", "626,000"],
    ["Maintenance", "3,767,000", "2,099,000"],
    ["Fuel consumption", "3,261,000", "2,052,000"],
    ["Certification, start up, and support equipment (10 percent of equipment/installation)",
     "32,609,000", "20,522,000"],
    ["Total", "40,751,000", "25,299,000"],
    ["Part 121-Airplanes:", "", ""],
    ["Equipment/Installation (other cost categories are minimal)",
     "8,090,000", "4,708,000"],
    ["Total", "8,090,000", "4,708,000"],
    ["Part 125-Airlines:", "", ""],
    ["Equipment/Installation", "1,080,000", "659,000"],
    ["Maintenance", "1,170,000", "714,000"],
    ["Fuel consumption", "1,060,000", "965,000"],
    ["Certification, start up, and support equipment (10 percent of equipment/installation)",
     "10,600,000", "9,651,000"],
    ["Total", "13,910,000", "11,989,000"],
    ["Total cost for all operating rules", "314,658,000", "222,339,000"],
]

DRS_PAGE_23_24_APPENDIX_D_ROWS = [
    ["Parameters", "Range", "Installed System¹ minimum accuracy (to recovered data)",
     "Sampling interval (per second)", "Resolution readout"],
    ["Relative time (from recorded on prior to takeoff)", "8 hr minimum",
     "±0.125% per hour", "1", "1 sec."],
    ["Indicated airspeed", "VSO to VD (KIAS)",
     "±5% or ±10 kts., whichever is greater. Resolution 2 kts. below 175 KIAS.",
     "1", "1%."],
    ["Altitude", "-1,000 ft. to max. cert. alt. of A/C",
     "±100 to ±700 ft. (see Table 1 TSO-C51-a)", "1", "25 to 150."],
    ["Magnetic heading", "360°", "±5°", "1", "1°."],
    ["Vertical acceleration", "-3g to +6g",
     "±0.2g in addition to ±0.3g maximum datum.",
     "4 (or 1 per second where peaks, ref. to 1g are recorded).", "0.03g."],
    ["Longitudinal acceleration", "±1.0g",
     "±0.5g in addition to max. datum error of ±0.1g.", "2", "0.01g."],
    ["Pitch altitude", "100% of usable", "±2°", "1", "0.8°."],
    ["Roll altitude", "±60° or 100% of usable range, whichever is greater",
     "+2°", "1", "0.8°."],
    ["Stabilizer trim position", "Full range",
     "±3% unless higher uniquely required", "1", "1%."],
    ["Pitch control position", "Full range",
     "±3% unless higher uniquely required", "1", "1%."],
    ["ENGINE POWER, EACH ENGINE", "", "", "", ""],
    ["Fan or N₁ Speed or EPR or cockpit indications used for aircraft certification.",
     "Maximum range", "+5%", "1", "1%."],
    ["Prop speed and torque (sample once/sec as close together as practicable).",
     "", "", "1 (prop speed); 1 (torque)", ""],
    ["Altitude rate² (need depends on altitude resolution).", "+8,000 fpm",
     "+10%. Resolution 250 fpm below 12,000 ft. indicated.", "1",
     "250 fpm below 12,000."],
    ["Angle of attack² (need depends on altitude resolution).",
     "-20 to 40 or of usable range", "+2", "1", "0.8°."],
    ["Radio transmitter keying (discrete).", "On/off", "", "1", ""],
    ["TE flaps (discrete)", "Each discrete position (U, D, T/O, AAP)", "", "1", "1%."],
    ["TE flaps (analog)", "Analog 0-100% range", "+3", "1", ""],
    ["LE flaps (discrete)", "Each discrete position (U, D, T/O, AAP)", "", "1", "1%."],
    ["LE flaps (analog)", "Analog 0-100% range", "+3", "1", ""],
    ["Thrust reverse, each engine (discrete)", "Stowed or full reverse", "", "1", ""],
    ["Spoiler/speedbrake (discrete)", "Stowed or out", "", "1", ""],
    ["Autopilot engaged (discrete)", "Engaged or disengaged", "", "1", ""],
]

DRS_PAGE_26_APPENDIX_E_CONTINUATION_ROWS = [
    ["Altitude rate", "±8,000 fpm",
     "±10% Resolution 250 fpm below 12,000 ft. indicated.", "1",
     "250 fpm below 12,000."],
    ["ENGINE POWER, EACH ENGINE", "", "", "", ""],
    ["Main roto speed", "Maximum range", "±5%", "1", "1%"],
    ["Free or power turbine", "Maximum range", "±5%", "1", "1%"],
    ["Engine torque", "Maximum range", "±5%", "1", "1%"],
    ["FLIGHT CONTROL HYDRAULIC PRESSURE", "", "", "", ""],
    ["Primary (discrete)", "High/low", "", "1", ""],
    ["Secondary-if applicable (discrete).", "High/low", "", "1", ""],
    ["Radio transmitter keying (discrete).", "On/off", "", "1", ""],
    ["Autopilot engaged (discrete)", "Engaged or disengaged", "", "1", ""],
    ["SAS status-engaged (discrete)", "Engaged/disengaged", "", "1", ""],
    ["SAS fault status (discrete)", "Fault/OK", "", "1", ""],
    ["FLIGHT CONTROLS", "", "", "", ""],
    ["Collective", "Full range", "±3%", "2", "1%"],
    ["Pedal position", "Full range", "±3%", "2", "1%"],
    ["Latitude cyclic", "Full range", "±3%", "2", "1%"],
    ["Longitude cyclic", "Full range", "±3%", "2", "1%"],
    ["Controllable stabilator position", "Full range", "±3%", "2", "1%"],
]

DRS_PAGE_25_26_APPENDIX_E_ROWS = [
    ["Parameters", "Range", "Installed System¹ minimum accuracy (to recovered data)",
     "Sampling interval (per second)", "Resolution readout"],
    ["Relative time (from recorded on prior to takeoff)", "4 hr minimum",
     "±125% per hour", "1", "1 sec."],
    ["Indicated airspeed",
     "VM in to VD (KIAS) (minimum airspeed signal attainable with installed pilot static system).",
     "±5% pt ±10 kts., whichever is greater", "1", "1 kt."],
    ["Altitude", "-1,000 ft. to 20,000 ft. pressure altitude.",
     "±700 ft. (see Table 1, TSO C51-a)", "1", "25 to 150 ft."],
    ["Magnetic heading", "360°", "±5°", "1", "1°."],
    ["Vertical acceleration", "-3g to +6g",
     "±0.2g in addition to ±0.3g maximum datum",
     "4 (or 1 per second where peaks, ref. to 1g are recorded).", "0.5g."],
    ["Longitudinal acceleration", "±1.0g",
     "±0.5g in addition to max. datum error of ±0.1g.", "2", "0.03g."],
    ["Pitch altitude", "100% of usable range", "±2°", "1", "0.8°."],
    ["Roll altitude", "±60° or 100% of usable range, whichever is greater.",
     "±2°", "1", "0.8°."],
    *DRS_PAGE_26_APPENDIX_E_CONTINUATION_ROWS,
]


def _compact_table_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _has_table_evidence(rows: list[list[str]], *markers: str) -> bool:
    text = " ".join(cell for row in rows for cell in row)
    compact = _compact_table_text(text)
    return all(_compact_table_text(marker) in compact for marker in markers)


def _copy_rows(rows: list[list[str]]) -> list[list[str]]:
    return [list(row) for row in rows]


def _repair_drs_recorder_spec_rows(
    page: int,
    rows: list[list[str]],
) -> list[list[str]]:
    """Restore visually verified sub-rows lost by OCR in DRS_88-1.

    These replacements are called only after the immutable source hash matches.
    The generic OR normalizer remains conservative; this source-scoped layer is
    where intentionally blank cells can be restored without inventing values in
    unrelated documents.
    """

    if page not in {24, 37} or not _has_table_evidence(
        rows, "Stabilizer", "ENGINE POWER"
    ):
        return rows

    repaired: list[list[str]] = []
    index = 0
    seen_radio_keying = False
    while index < len(rows):
        row = list(rows[index])
        label = re.sub(r"\s+", " ", row[0]).strip().casefold() if row else ""
        compact_label = _compact_table_text(row[0] if row else "")

        if "stabilizertrimposition" in compact_label and "pitchcontrolposition" in compact_label:
            repaired.extend([
                ["Stabilizer Trim Position", "Full Range",
                 "±3% unless higher uniquely required", "1", "1%."],
                ["Pitch Control Position", "Full Range",
                 "±3% unless higher uniquely required", "1", "1%."],
            ])
            index += 1
            continue

        if (
            "fan" in label
            and ("n1" in compact_label or "n₁" in row[0].casefold())
            and "prop" in label
            and "torque" in label
        ):
            repaired.extend([
                [
                    "Fan or N₁ Speed or EPR or cockpit indications used for "
                    "aircraft certification.",
                    "Maximum Range", "+5%", "1", "1%.",
                ],
                [
                    "Prop speed and torque (sample once/sec as close together as practicable).",
                    "", "", "1 (prop speed); 1 (torque)", "",
                ],
            ])
            index += 1
            continue

        if page == 37 and label.startswith("altitude rate"):
            repaired.append([
                "Altitude Rate² (need depends on altitude resolution).",
                "+8,000 fpm",
                "+10%. Resolution 250 fpm below 12,000 ft. indicated.",
                "1",
                "250 fpm below 12,000",
            ])
            index += 1
            while index < len(rows):
                residue = re.sub(r"\s+", " ", str(rows[index][0])).strip().casefold()
                if residue != "altitude resolution).":
                    break
                index += 1
            continue

        if page == 37 and label.startswith("radio transmitter keying"):
            if seen_radio_keying:
                index += 1
                continue
            seen_radio_keying = True

        # Some post-merge cleanup has already lost the TE/LE prefix and leaves
        # three consecutive generic "Flaps" rows.  In this hash-scoped source
        # they are the duplicated TE fragments followed by LE; restore both
        # visually checked parameter pairs together.
        if page == 37 and label.startswith("flaps"):
            flap_run = 0
            while index + flap_run < len(rows):
                candidate_label = re.sub(
                    r"\s+", " ", str(rows[index + flap_run][0])
                ).strip().casefold()
                if not candidate_label.startswith("flaps"):
                    break
                flap_run += 1
            if flap_run >= 3:
                for kind in ("TE", "LE"):
                    repaired.extend([
                        [f"{kind} Flaps (Discrete)",
                         "Each discrete position (U, D, T/O, AAP)", "", "1", "1%."],
                        [f"{kind} Flaps (Analog)",
                         "Analog 0-100% range", "+3", "1", ""],
                    ])
                index += flap_run
                continue

        flap_kind = "TE" if label.startswith("te flaps") else (
            "LE" if label.startswith("le flaps") else ""
        )
        if flap_kind:
            repaired.extend([
                [f"{flap_kind} Flaps (Discrete)",
                 "Each discrete position (U, D, T/O, AAP)", "", "1", "1%."],
                [f"{flap_kind} Flaps (Analog)",
                 "Analog 0-100% range", "+3", "1", ""],
            ])
            index += 1
            while index < len(rows):
                next_row = rows[index]
                next_label = re.sub(r"\s+", " ", str(next_row[0])).strip().casefold()
                next_range = _compact_table_text(str(next_row[1]) if len(next_row) > 1 else "")
                if next_label.startswith(flap_kind.casefold() + " flaps") or (
                    not next_label and "oranalog0100" in next_range
                ):
                    index += 1
                    continue
                break
            continue

        repaired.append(row)
        index += 1
    return repaired


def verified_table(source_hash: str | None, page: int, markup: str) -> str | None:
    rows = table_rows_from_html(markup)

    def normalized(cells: list[list[str]]) -> list[list[str]]:
        return [[re.sub(r"[\s.:-]", "", cell).casefold() for cell in row] for row in cells]

    if source_hash == FINAL_RULE_26324_SHA256 and page == 10:
        original, verified = AISLE_ORIGINAL_CELLS, AISLE_VERIFIED_CELLS
    elif source_hash == DOCKET4080_SHA256 and page == 6:
        original, verified = ORIGINAL_CELLS, VERIFIED_CELLS
    else:
        return None
    if normalized(rows) != normalized(original):
        return None
    return "<table>" + "".join(
        "<tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in row) + "</tr>"
        for i, row in enumerate(verified)
        for tag in ["th" if i == 0 else "td"]
    ) + "</table>"


def verified_hybrid_rows(
    source_hash: str | None,
    page: int,
    rows: list[list[str]],
) -> list[list[str]]:
    """Apply narrowly scoped cell-boundary repairs checked against source pixels."""
    if source_hash == FINAL_RULE_2005_20245_SHA256:
        if page == 28 and _has_table_evidence(
            rows,
            "Assumptior/parameter",
            "Part 121 Airplanes",
            "Price of Aviation Fuel",
        ):
            return _copy_rows(FINAL_RULE_PAGE_28_ROWS)
        if page == 37 and _has_table_evidence(
            rows,
            "Stabilizer Trim Position",
            "Resolution",
        ):
            return _copy_rows(FINAL_RULE_PAGE_37_APPENDIX_E_ROWS)
        if page == 37 and _has_table_evidence(
            rows,
            "Collective",
            "Controllable Stabilator Position",
        ):
            return _copy_rows(FINAL_RULE_PAGE_37_APPENDIX_F_ROWS)
        if page == 39 and _has_table_evidence(
            rows,
            "Time or relative times counts",
            "Pitch control",
            "Lateral control",
        ):
            return _copy_rows(FINAL_RULE_PAGE_39_ROWS)
    if source_hash == DRS_98_19_SHA256:
        if page == 25 and _has_table_evidence(
            rows,
            "Engine inlet area (A) square-meters (square-inches)",
            "Bird quantity",
            "0.05 (77.5)",
            "4.50 (6975)",
        ):
            return _copy_rows(DRS_98_19_PAGE_25_TABLE_2_ROWS)
        if page == 25 and _has_table_evidence(
            rows,
            "Engine inlet area (A) square-meters (square-inches)",
            "Bird quantity",
            "1.35 (2,092)> A",
            "2.90 (4,495)",
        ):
            return _copy_rows(DRS_98_19_PAGE_25_TABLE_3_ROWS)
        if page == 25 and _has_table_evidence(
            rows,
            "Foreignobject",
            "Test quantity",
            "Engine operation",
            "Ice",
        ):
            return _copy_rows(DRS_98_19_PAGE_25_FOREIGN_OBJECT_ROWS)
        return rows
    if source_hash != DRS_88_1_SHA256:
        return rows
    if page == 12 and _has_table_evidence(
        rows, "Part 91-Airplanes", "Total cost for all operating rules"
    ):
        return _copy_rows(DRS_PAGE_12_13_COST_ROWS)
    if page == 23 and _has_table_evidence(
        rows, "Parameters", "Autopilot engaged", "ENGINE POWER"
    ):
        return _copy_rows(DRS_PAGE_23_24_APPENDIX_D_ROWS)
    if page == 25 and _has_table_evidence(
        rows, "Parameters", "Controllable stabilator", "ENGINE POWER"
    ):
        return _copy_rows(DRS_PAGE_25_26_APPENDIX_E_ROWS)
    if page == 26 and _has_table_evidence(
        rows, "Main roto speed", "Controllable stabilator", "FLIGHT CONTROLS"
    ):
        return _copy_rows(DRS_PAGE_26_APPENDIX_E_CONTINUATION_ROWS)
    repaired = _repair_drs_recorder_spec_rows(page, _copy_rows(rows))
    for row in repaired:
        label = re.sub(r"\s+", " ", row[0]).strip().casefold() if row else ""
        if page == 27 and label == "altitude" and len(row) == 5:
            row[3], row[4] = "1", "5 to 35'."
        elif page in {27, 28} and label.startswith("normal acceleration") and len(row) == 5:
            row[3], row[4] = "8", "0.01 g."
        elif page in {27, 28, 29} and "ground proximity" in label and len(row) == 5:
            row[0] = "GPWS (ground proximity warning system)."
        elif page == 41 and label.startswith("time (gmt)") and len(row) == 5:
            row[:] = [
                "Time (GMT)",
                "24 Hrs",
                "±0.125% Per Hour",
                "0.25 (1 per 4 seconds)",
                "1 sec.",
            ]
        elif page == 41 and label == "altitude" and len(row) == 5:
            row[3], row[4] = "1", "5' to 30'."
        elif page == 41 and label == "airspeed" and len(row) == 5:
            row[1] = "As the installed measuring system"
        elif page == 41 and label.startswith("pilot input and surface") and len(row) == 5:
            row[:] = [
                "Pilot Input and Surface Position - Primary Controls (Pitch, Roll, Yaw)",
                "Full Range",
                "±2 Unless Higher Accuracy Uniquely Required",
                "1",
                "0.01g",
            ]
        elif page == 41 and (
            label.startswith("afcs mode") or label.startswith("mode andengagement")
        ) and len(row) == 5:
            row[0], row[1], row[3] = (
                "AFCS Mode and Engagement Status",
                "Discrete (5 bits Necessary)",
                "1",
            )
        elif page == 41 and label.startswith("nav 1 and 2 frequency") and len(row) == 5:
            row[:] = [
                "Nav 1 and 2 Frequency Selection", "Full Range", "As installed", "0.25", ""
            ]
        elif page == 41 and (
            label.startswith("dme 1 and 2 distance") or label.startswith("1 and 2 distance")
        ) and len(row) == 5:
            row[:] = ["DME 1 and 2 Distance", "0-200 NM", "As installed", "0.25", "1 Mi"]
        elif page == 41 and label.startswith("main gear squat") and len(row) == 5:
            row[0] = "Main Gear Squat Switch Status"
        elif page == 41 and label.startswith("outside air temperature") and len(row) == 5:
            row[:] = ["Outside Air Temperature", "-90 C to +50 C", "±2 C", "0.5", "0.3 C"]
        elif page == 41 and label.startswith("hydraulic, each system") and len(row) == 5:
            row[0] = "Hydraulic, Each System Low Pressure"
        elif page == 41 and label.startswith("landing gear or gear") and len(row) == 5:
            row[0] = "Landing Gear or Gear Selector Position"
    return repaired
