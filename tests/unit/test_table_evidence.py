from __future__ import annotations

from app.schemas.retrieval import ChunkMetadata, RetrievedChunk
from app.services.cell_evidence import dimension_keywords, substantive_lines
from app.services.citation_service import CitationService
from app.services.table_evidence import (
    html_table_blocks,
    html_table_records,
    markdown_table_records,
)


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text=text,
        metadata=ChunkMetadata(document_name="规程"),
        hybrid_score=0.8,
    )


def test_html_table_is_converted_to_key_value_records() -> None:
    text = (
        "<table><caption>表A.3 平面相对位置精度检查记录表</caption>"
        "<tr><th>检查项目</th><th>限差</th></tr>"
        "<tr><td>平面位置中误差</td><td>0.20 m</td></tr></table>"
    )

    assert html_table_records(text) == [
        "表A.3 平面相对位置精度检查记录表；检查项目: 平面位置中误差；限差: 0.20 m"
    ]
    lines = substantive_lines(_chunk(text), {"平面", "精度"})
    assert lines == [
        "表A.3 平面相对位置精度检查记录表；检查项目: 平面位置中误差；限差: 0.20 m"
    ]
    assert "<table>" not in lines[0]


def test_caption_without_data_is_not_substantive_evidence() -> None:
    text = "<table><caption>表A.3 平面相对位置精度检查记录表</caption></table>"
    assert html_table_records(text) == []
    assert substantive_lines(_chunk(text), {"平面", "精度"}) == []


def test_html_table_blocks_preserve_blank_form_cells_and_escaped_tags() -> None:
    text = (
        r"A.3 平面相对位置精度检查记录表样式见表A.3。\<table>"
        r"\<caption>表A.3 平面相对位置精度检查记录表\</caption>"
        r"\<tr>\<td>项目名称：\</td>\<td>\</td>\</tr>\</table>"
    )
    blocks = html_table_blocks(text)
    assert len(blocks) == 1
    assert "<caption>表A.3 平面相对位置精度检查记录表</caption>" in blocks[0]
    assert "<td>项目名称：</td><td></td>" in blocks[0]
    assert html_table_records(text) == []


def test_html_table_blocks_keep_multiple_tables_in_one_chunk() -> None:
    text = (
        "<table><caption>表A.3</caption><tr><td>甲</td></tr></table>"
        "<table><caption>表A.4</caption><tr><td>乙</td></tr></table>"
    )
    assert ["表A.3", "表A.4"] == [
        block.split("<caption>", 1)[1].split("</caption>", 1)[0]
        for block in html_table_blocks(text)
    ]


def test_citation_service_exposes_all_table_blocks() -> None:
    text = (
        "<table><caption>表A.3</caption><tr><td>甲</td></tr></table>"
        "<table><caption>表A.4</caption><tr><td>乙</td></tr></table>"
    )
    _, citations = CitationService().build([_chunk(text)])
    assert len(citations[0].table_htmls) == 2
    assert citations[0].table_title == "表A.3"


def test_markdown_table_is_converted_to_key_value_records() -> None:
    text = "| 项目 | 要求 |\n| --- | --- |\n| 影像分辨率 | 不低于 2 cm |"
    assert markdown_table_records(text) == ["项目: 影像分辨率；要求: 不低于 2 cm"]


def test_colon_guide_line_is_not_substantive_evidence() -> None:
    text = "4.3.4 云台相机应满足以下要求:\n相机分辨率不应低于2000万像素。"
    assert substantive_lines(_chunk(text), {"相机", "分辨"}) == [
        "相机分辨率不应低于2000万像素。"
    ]


def test_comparison_aspect_rejects_cross_stage_lines() -> None:
    text = (
        "3.2 自动巡检 automatic patrol由飞行控制系统控制无人机开展桥梁巡检作业。\n"
        "巡检方案应包括设备选型、航迹规划和应急预案。\n"
        "宜选择多旋翼无人机进行巡检作业。\n"
        "无人机不应在梁底净空高度 <5 m 的桥下巡检。\n"
        "成果资料应包括有效原始病害图片和巡检报告。"
    )

    assert substantive_lines(
        _chunk(text),
        dimension_keywords("设备选型"),
        "设备选型",
    ) == [
        "巡检方案应包括设备选型、航迹规划和应急预案。",
        "宜选择多旋翼无人机进行巡检作业。",
    ]
    assert substantive_lines(
        _chunk(text),
        dimension_keywords("成果验收"),
        "成果验收",
    ) == []


def test_acceptance_rejects_deliverables_and_reports() -> None:
    text = (
        "a) 成果清单（见附录A.5）；\n"
        "b) 无人机航测成果精度检查记录表；\n"
        "c) 测量技术成果报告，格式按照CH/T 1001执行。"
    )
    assert substantive_lines(
        _chunk(text),
        dimension_keywords("成果验收"),
        "成果验收",
    ) == []


def test_acceptance_decodes_html_entities_in_normative_text() -> None:
    line = "表3；检查项目: 平面位置精度；限差: 0.20 m；应适&#x5F53;放宽"
    assert substantive_lines(
        _chunk(line),
        dimension_keywords("成果验收"),
        "成果验收",
    ) == ["表3；检查项目: 平面位置精度；限差: 0.20 m；应适当放宽"]


def test_model_training_accuracy_is_not_deliverable_acceptance() -> None:
    line = "若模型测试平均精度MAP ≥0.5，可认为该模型有效，否则重新训练。"
    assert substantive_lines(
        _chunk(line),
        dimension_keywords("成果验收"),
        "成果验收",
    ) == []


def test_normative_reference_catalog_is_not_acceptance_evidence() -> None:
    text = (
        "GB/T 24356 测绘成果质量检查与验收\n"
        "CH/T 1029.1 航空摄影成果质量检验技术规程"
    )
    assert substantive_lines(
        _chunk(text),
        dimension_keywords("成果验收"),
        "成果验收",
    ) == []


def test_equipment_inspection_is_not_equipment_selection() -> None:
    text = (
        "作业适用的无人机、搭载传感器应进行检查校正。\n"
        "相机与POS系统应牢固固定于飞行平台上。"
    )
    assert substantive_lines(
        _chunk(text),
        dimension_keywords("设备选型"),
        "设备选型",
    ) == []


def test_control_point_input_list_is_not_layout_requirement() -> None:
    line = "收集航测影像资料、控制点、地形图、交通图和行政区划图。"
    assert substantive_lines(
        _chunk(line),
        dimension_keywords("像控布设"),
        "像控布设",
    ) == []


def test_equipment_choice_must_target_equipment_not_takeoff_site() -> None:
    line = "为选择合适的起降点，无人机飞行前应开展测区实地踏勘。"
    assert substantive_lines(
        _chunk(line),
        dimension_keywords("设备选型"),
        "设备选型",
    ) == []


def test_table_values_rank_before_prose_table_reference() -> None:
    text = (
        "点云数据密度及航带接边精度参照表4执行。"
        "<table><caption>表4 不同调查比例尺的点云密度</caption>"
        "<tr><th>比例尺</th><th>点云密度</th></tr>"
        "<tr><td>1:500</td><td>16点/m2</td></tr></table>"
    )
    lines = substantive_lines(
        _chunk(text),
        dimension_keywords("成果验收"),
        "成果验收",
    )
    assert lines[0].startswith("表4 不同调查比例尺的点云密度")


def test_acceptance_uses_quality_semantics_not_fixed_table_numbers() -> None:
    generic_table = "表7 数字正射影像成果质量要求；地面分辨率: 0.05 m"
    assert substantive_lines(
        _chunk(generic_table),
        dimension_keywords("成果验收"),
        "成果验收",
    ) == [generic_table]


def test_acquisition_ground_resolution_table_is_not_acceptance_by_itself() -> None:
    line = "表1 无人机航摄地面分辨率；调查比例尺: 1:500；分辨率: 5 cm"
    assert substantive_lines(
        _chunk(line),
        dimension_keywords("成果验收"),
        "成果验收",
    ) == []


def test_product_name_without_quality_requirement_is_not_acceptance_evidence() -> None:
    line = "数字正射影像图、数字高程模型和数字线划图应按生产方法制作。"
    assert substantive_lines(
        _chunk(line),
        dimension_keywords("成果验收"),
        "成果验收",
    ) == []


def test_blank_form_reference_is_not_acceptance_evidence() -> None:
    line = "平面相对位置精度检查记录表样式见表A.3。"
    assert substantive_lines(
        _chunk(line),
        dimension_keywords("成果验收"),
        "成果验收",
    ) == []


def test_scope_sentence_is_filtered_from_evidence() -> None:
    """文档范围句（"本文件规定了…/本文件适用于…"）不是实质条款。"""
    line = (
        "本文件规定了无人机应用服务的基本要求、机构要求、人员要求、设施设备要求、"
        "服务要求、信息管理等内容。"
    )
    assert substantive_lines(
        _chunk(line),
        dimension_keywords("运营管理"),
        "运营管理",
    ) == []


def test_wrapped_pdf_line_is_merged_back() -> None:
    """PDF 硬换行切断的句子（"（无人驾驶飞行\n器）作为载体"）应接回。"""
    text = (
        "在沙区，利用搭载了高精度导航定位系统、智能控制系统和专用播撒系统的无人机（无人驾驶飞行\n"
        "器）作为载体，根据预设的飞行路线和播种要求，进行自动化、定量、定点、均匀地播撒种子造林种草\n"
        "的一种现代化治沙技术。"
    )
    lines = substantive_lines(
        _chunk(text),
        dimension_keywords("技术参数"),
        "技术参数",
    )
    assert lines and "飞行器）作为载体" in lines[0]
    assert not any(line.startswith("器）作为载体") for line in lines)


def test_numbered_clause_with_verb_is_not_toc_line() -> None:
    """"8.2.2 应依据作业要求匹配作业现场条件出具现场勘察报告。"是条款不是目录。"""
    text = (
        "8.2 现场勘察\n"
        "8.2.1 应按照MH/T 1069 的要求，派遣专业人员实地勘察。\n"
        "8.2.2 应依据作业要求匹配作业现场条件出具现场勘察报告。"
    )
    lines = substantive_lines(
        _chunk(text),
        dimension_keywords("运营管理"),
        "运营管理",
    )
    assert any("8.2.2 应依据作业要求匹配作业现场条件出具现场勘察报告" in line for line in lines)
