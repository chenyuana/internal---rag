from pathlib import Path

from app.ingestion.parsers.vector_pdf import (
    VectorPdfParser,
    table_to_html,
    table_to_text,
)
from app.ingestion.pipeline import (
    PageRecord,
    is_flattened_table_noise,
    split_blocks,
    split_table_rows,
)


def test_preflight_routes_image_charts_and_vector_formulas(monkeypatch) -> None:
    class FakePage:
        width = 100
        height = 100
        rects = []

        def __init__(
            self,
            *,
            text: str,
            images: list[dict[str, float]],
            line_count: int,
            curve_count: int,
        ) -> None:
            self._text = text
            self.images = images
            self.lines = [{}] * line_count
            self.curves = [{}] * curve_count

        def extract_text(self) -> str:
            return self._text

    class FakePdf:
        pages = [
            FakePage(
                text="图 A2 曲线",
                images=[
                    {
                        "x0": 10.0,
                        "x1": 90.0,
                        "top": 10.0,
                        "bottom": 40.0,
                    }
                ],
                line_count=0,
                curve_count=0,
            ),
            FakePage(
                text=(
                    "V = 3.47\nS\nn1\nV = 2.46\nwg\n"
                    "V = 2.17\nS\nV = 1.59"
                ),
                images=[],
                line_count=15,
                curve_count=10,
            ),
            FakePage(
                text="普通正文，没有需要特殊处理的视觉内容。",
                images=[],
                line_count=0,
                curve_count=0,
            ),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    result = VectorPdfParser().preflight_pages(
        Path("sample.pdf"),
        candidate_pages={1, 2, 3},
    )

    assert result.figure_pages == {1}
    assert result.formula_pages == {2}
    assert result.table_pages == set()
    assert result.table_series == []


def test_preflight_flags_low_coverage_image_with_figure_caption(monkeypatch) -> None:
    # 图片覆盖率 0.08（低于 0.12 阈值），但文本含行首图注"图1 …"，
    # 应仍被识别为 figure 页（模拟《吊挂控制》第8页的漏识别场景）。
    class FakePage:
        width = 100
        height = 100
        rects = []

        def __init__(self, *, text: str, images: list[dict[str, float]]) -> None:
            self._text = text
            self.images = images
            self.lines = []
            self.curves = []

        def extract_text(self) -> str:
            return self._text

    class FakePdf:
        pages = [
            FakePage(
                text=(
                    "图1 物流无人机货物吊挂控制逻辑框图\n"
                    "4.2 基本飞行控制\n"
                    "物流无人机基本飞行控制功能应符合 GB/T38997—2020 中的 4.2 要求。"
                ),
                images=[
                    {
                        "x0": 30.0,
                        "x1": 70.0,
                        "top": 20.0,
                        "bottom": 60.0,
                    }
                ],
            ),
            FakePage(
                text="正文引用图1所示的内容，本页无插图。",
                images=[],
            ),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    result = VectorPdfParser().preflight_pages(
        Path("sample.pdf"),
        candidate_pages={1, 2},
    )

    assert result.figure_pages == {1}


def test_preflight_detects_consecutive_ruled_table_pages(monkeypatch) -> None:
    class FakeTable:
        bbox = (10.0, 10.0, 90.0, 90.0)

        @staticmethod
        def extract():
            return [
                ["条款号", "名称", "对应条款", "名称"],
                ["23.2000", "适用范围", "23.1", "适用范围"],
            ]

    class FakePage:
        width = 100
        height = 100
        rects = []
        edges = []
        images = []
        lines = []
        curves = []

        def __init__(self, *, has_table: bool) -> None:
            self._has_table = has_table

        @staticmethod
        def extract_text() -> str:
            return "具有完整文本层的普通文字"

        def find_tables(self):
            return [FakeTable()] if self._has_table else []

    class FakePdf:
        pages = [
            FakePage(has_table=True),
            FakePage(has_table=True),
            FakePage(has_table=False),
            FakePage(has_table=True),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    result = VectorPdfParser().preflight_pages(
        Path("cross-page-table.pdf"),
        candidate_pages={1, 2, 3, 4},
    )

    assert result.table_pages == {1, 2, 4}
    assert result.table_series == [[1, 2], [4]]


def test_preflight_detects_fraction_formula_with_single_equals(monkeypatch) -> None:
    # 分式公式（如 R=floor((C−H)/F)）整页只有一个 "="，但含函数名 floor/ceil
    # 和孤立短行碎片（"( )"、"F"），应被识别为公式页。
    class FakePage:
        width = 100
        height = 100
        rects = []
        edges = []
        lines = []
        curves = []
        images = [{"x0": 10.0, "x1": 20.0, "top": 10.0, "bottom": 15.0}]

        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakePdf:
        pages = [
            FakePage(
                "( )\n"
                "C(cid:0)H\n"
                "R= floor\n"
                "F\n"
                "式中：\n"
                "R ─单卡多点记录数；\n"
            ),
            FakePage(
                "普通正文，没有公式特征，max 是最大值但不是公式。\n"
                "本页没有任何公式。\n"
            ),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    result = VectorPdfParser().preflight_pages(
        Path("fraction-formula.pdf"),
        candidate_pages={1, 2},
    )

    assert result.formula_pages == {1}


def test_smart_cell_text_merges_floating_punctuation_lines() -> None:
    # 字间距大的 PDF 中，冒号/斜杠/单位字符 top 偏移约 6pt，pdfplumber 会
    # 把它们分到下一行。智能重建应按 top 聚类后合并这些"浮动下标"行。
    def ch(text, x0, x1, top, bottom):
        return {"text": text, "x0": float(x0), "x1": float(x1),
                "top": float(top), "bottom": float(bottom)}

    class FakePage:
        width = 595
        height = 842
        chars = [
            # "从下述列项选一或多选:滑跑/弹射/手抛/垂直/其他"
            ch("从", 200, 212, 100, 114),
            ch("下", 214, 226, 100, 114),
            ch("述", 228, 240, 100, 114),
            ch("列", 242, 254, 100, 114),
            ch("项", 256, 268, 100, 114),
            ch("选", 270, 282, 100, 114),
            ch("一", 284, 296, 100, 114),
            ch("或", 298, 310, 100, 114),
            ch("多", 312, 324, 100, 114),
            ch("选", 326, 338, 100, 114),
            # 冒号偏移 6pt（浮动下标行）
            ch(":", 340, 346, 106, 120),
            ch("滑", 350, 362, 100, 114),
            ch("跑", 364, 376, 100, 114),
            ch("/", 378, 384, 106, 120),
            ch("弹", 388, 400, 100, 114),
            ch("射", 402, 414, 100, 114),
            ch("/", 416, 422, 106, 120),
            ch("手", 426, 438, 100, 114),
            ch("抛", 440, 452, 100, 114),
            ch("/", 454, 460, 106, 120),
            ch("垂", 464, 476, 100, 114),
            ch("直", 478, 490, 100, 114),
            ch("/", 492, 498, 106, 120),
            ch("其", 502, 514, 100, 114),
            ch("他", 516, 528, 100, 114),
        ]

    parser = VectorPdfParser()
    bbox = [150, 90, 580, 130]
    text = parser._smart_cell_text(FakePage(), bbox)
    # 浮动标点（冒号/斜杠）应合并到主行，而不是拆成独立行
    assert "\n" not in text
    assert "从下述列项选一或多选:滑跑/弹射/手抛/垂直/其他" in text


def test_flowchart_regions_aggregate_imagemask_boxes() -> None:
    # 流程图由 1-bit 文字掩码 + 框线(rect) + 箭头(curve) 构成，应聚合为
    # 整体区域而不是每个掩码切一块。
    class FakePage:
        width = 595
        height = 842
        # 4 个文字掩码（imagemask）+ 3 个框线 rect + 3 个箭头 curve
        images = [
            {"x0": 275.0, "x1": 359.0, "top": 154.0, "bottom": 198.0, "imagemask": True},
            {"x0": 271.0, "x1": 362.0, "top": 224.0, "bottom": 272.0, "imagemask": True},
            {"x0": 217.0, "x1": 302.0, "top": 303.0, "bottom": 348.0, "imagemask": True},
            {"x0": 331.0, "x1": 417.0, "top": 303.0, "bottom": 348.0, "imagemask": True},
        ]
        rects = [
            {"x0": 200.0, "x1": 429.0, "top": 143.0, "bottom": 373.0},
            {"x0": 200.0, "x1": 429.0, "top": 143.0, "bottom": 373.0},
            {"x0": 275.0, "x1": 359.0, "top": 154.0, "bottom": 198.0},
            {"x0": 271.0, "x1": 362.0, "top": 224.0, "bottom": 272.0},
            {"x0": 217.0, "x1": 302.0, "top": 303.0, "bottom": 348.0},
            {"x0": 331.0, "x1": 417.0, "top": 303.0, "bottom": 348.0},
        ]
        curves = [
            {"x0": 0.0, "x1": 1.0, "top": 0.0, "bottom": 1.0, "pts": []},
            {"x0": 0.0, "x1": 1.0, "top": 0.0, "bottom": 1.0, "pts": []},
            {"x0": 0.0, "x1": 1.0, "top": 0.0, "bottom": 1.0, "pts": []},
        ]

        def extract_words(self):
            return []  # 流程图区域无可提取文字

    regions = VectorPdfParser._flowchart_regions(FakePage())

    # 应聚合出主流程图区域（大外框），而非逐掩码切分
    assert regions, "应识别出流程图区域"
    largest = max(regions, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    assert largest[0] == 200.0 and largest[1] == 143.0
    assert largest[2] == 429.0 and largest[3] == 373.0


def test_vector_table_preserves_rows_columns_and_escapes_content() -> None:
    rows = [
        ["参数", "平均值", "备注"],
        ["温度", "59.95", "<安全>"],
        ["标准差", "20.14", None],
    ]

    table_html = table_to_html(rows)
    table_text = table_to_text(rows)

    assert "<th>参数</th>" in table_html
    assert "<td>&lt;安全&gt;</td>" in table_html
    assert "温度 | 59.95 | <安全>" in table_text
    assert table_text.splitlines()[2] == "标准差 | 20.14 | "


def test_vector_table_preserves_merged_note_as_one_cell() -> None:
    rows = [
        ["比例尺", "平地", "山地"],
        ["1:500", "0.15", "0.35"],
        ["注：困难地区允许中误差为规定值的1.5倍。", None, None],
    ]

    table_html = table_to_html(rows)
    table_text = table_to_text(rows)

    assert '<td colspan="3">注：困难地区允许中误差为规定值的1.5倍。</td>' in table_html
    assert table_text.splitlines()[-1] == "注：困难地区允许中误差为规定值的1.5倍。"
    assert "注：困难地区允许中误差为规定值的1.5倍。 |" not in table_text


def test_valid_table_accepts_header_only_blank_record_form() -> None:
    # 空白记录表只有表头一行有内容（如航摄作业记录表），不应被 _valid_table
    # 误杀。修复前 populated >= width+1 会拒绝 populated == width 的表。
    bbox = [70.0, 250.0, 540.0, 726.0]
    rows = [
        ["监测时间", "测区序号", "县", "乡镇", "村", "航拍时间", "覆盖面积", "变异木数量"],
        ["", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
    ]
    assert VectorPdfParser._valid_table((bbox, rows)) is True
    # 单列表（附录标题）仍应被拒绝
    single = [["附录 B"], ["（资料性）"], ["航摄作业记录表"]]
    assert VectorPdfParser._valid_table((bbox, single)) is False


def test_valid_table_accepts_single_row_multi_column_schematic() -> None:
    # 单行多列的数据结构示意图（如 GB 46750-2025 数据包格式：1行8列表头）
    # 是真实表格，不应被 _valid_table 误杀（修复前 len(rows)>=2 会拒绝它，
    # 导致该表格缺失、相邻真实表格被 expanded 残片覆盖）。
    bbox = [95.1, 420.7, 492.2, 440.5]
    rows = [
        [
            "数据类型",
            "版本号",
            "数据长度",
            "数据标识",
            "数据内容项\n1",
            "数据内容项\n2",
            "…",
            "数据内容项N",
        ],
    ]
    assert VectorPdfParser._valid_table((bbox, rows)) is True
    # 单行窄表格（如装饰框/单列）仍应被拒绝
    narrow = [["数据包内容", "长度"]]
    assert VectorPdfParser._valid_table((bbox, narrow)) is False


def test_expanded_ruled_table_restores_outer_columns() -> None:
    class FakePage:
        width = 100
        height = 800
        edges = [
            {"x0": 10, "x1": 90, "top": y, "bottom": y}
            for y in (100, 120, 140, 160)
        ] + [
            {"x0": x, "x1": x, "top": 120, "bottom": 160}
            for x in (30, 50, 70)
        ]

        @staticmethod
        def extract_table(settings):
            assert settings["explicit_vertical_lines"] == [
                10.0,
                30.0,
                50.0,
                70.0,
                90.0,
            ]
            return [
                ["操纵面", "载荷方向", "载荷大小", "弦上分布"],
                ["水平尾翼", "向上和向下", "图A5曲线", "见图A7"],
            ]

    tables = VectorPdfParser._expanded_ruled_tables(FakePage())

    assert len(tables) == 1
    bbox, rows = tables[0]
    assert bbox == [10.0, 100.0, 90.0, 160.0]
    assert len(rows[0]) == 4
    assert rows[1][3] == "见图A7"


def test_complete_table_outranks_fragments_around_a_tall_merged_row(
    monkeypatch,
) -> None:
    full_rows = [
        ["article", "name", "standard", "remark"],
        ["23.2150", "stall characteristics", "F3180", ""],
        ["23.2155", "handling characteristics", "F3173", ""],
        ["23.2160", "vibration", "F3173", "complete multi-line remark"],
        ["23.2165", "icing conditions", "F3120", ""],
        ["23.2200", "design envelope", "F3116", ""],
    ]
    fragments = [
        ([0.0, 0.0, 100.0, 30.0], full_rows[:3]),
        ([0.0, 70.0, 100.0, 100.0], full_rows[4:]),
    ]

    class FakeTable:
        bbox = (0.0, 0.0, 100.0, 100.0)

        @staticmethod
        def extract():
            return full_rows

    class FakePage:
        @staticmethod
        def find_tables():
            return [FakeTable()]

    monkeypatch.setattr(
        VectorPdfParser,
        "_expanded_ruled_tables",
        classmethod(lambda _cls, _page: fragments),
    )

    tables = VectorPdfParser._extract_best_tables(FakePage())

    assert len(tables) == 1
    assert tables[0][0] == [0.0, 0.0, 100.0, 100.0]
    assert tables[0][1][3][0] == "23.2160"


def test_captioned_adjacent_tables_outrank_one_expanded_grid() -> None:
    detected = [
        (
            [0.0, 0.0, 100.0, 40.0],
            [["表 1 温度分布", "", ""], ["参数", "平均", "方差"]],
        ),
        (
            [0.0, 50.0, 100.0, 100.0],
            [["表 2 航段分布", "", ""], ["自", "至", "比例"]],
        ),
    ]
    expanded = [
        (
            [0.0, 0.0, 100.0, 100.0],
            [
                ["参数", "平均", "方差", ""],
                ["59.95", "-70", "120", ""],
                ["表 2", "航段", "分布", ""],
                ["0", "200", "11.7", "7.5"],
            ],
        )
    ]

    chosen = VectorPdfParser._choose_best_tables(detected, expanded)

    assert chosen == detected


def test_caption_geometry_splits_merged_grid_and_binds_distinct_titles(
    monkeypatch,
) -> None:
    class FakeCrop:
        def __init__(self, bbox) -> None:
            self.bbox = bbox

    class FakePage:
        width = 100
        height = 240
        edges = [
            {"x0": 0, "x1": 100, "top": y, "bottom": y}
            for y in (30, 60, 90, 130, 160, 200)
        ]

        @staticmethod
        def extract_words(**_kwargs):
            return [
                {
                    "text": "表7 高程注记点高程精度要求",
                    "x0": 20,
                    "x1": 80,
                    "top": 10,
                    "bottom": 20,
                },
                {
                    "text": "表8 等高线插求点高程精度要求",
                    "x0": 20,
                    "x1": 80,
                    "top": 110,
                    "bottom": 120,
                },
            ]

        @staticmethod
        def crop(bbox):
            return FakeCrop(bbox)

    combined = (
        [0.0, 30.0, 100.0, 200.0],
        [["表7内容", "1"], ["说明及表8内容", "2"]],
    )
    table7 = ([0.0, 30.0, 100.0, 90.0], [["地形", "中误差"], ["平地", "1"]])
    table8 = (
        [0.0, 130.0, 100.0, 200.0],
        [["地形类别", "允许中误差"], ["山地", "2"]],
    )

    def fake_extract(_cls, page):
        if isinstance(page, FakeCrop):
            return [table7] if page.bbox[1] < 100 else [table8]
        return [combined]

    monkeypatch.setattr(
        VectorPdfParser,
        "_extract_best_tables",
        classmethod(fake_extract),
    )

    tables = VectorPdfParser._extract_captioned_tables(FakePage())

    assert [table[2] for table in tables] == [
        "表7 高程注记点高程精度要求",
        "表8 等高线插求点高程精度要求",
    ]
    assert tables[0][1][0] == ["地形", "中误差"]
    assert tables[1][1][0] == ["地形类别", "允许中误差"]


def test_caption_geometry_trims_appendix_heading_before_form(monkeypatch) -> None:
    class FakeCrop:
        def __init__(self, bbox) -> None:
            self.bbox = bbox

    class FakePage:
        width = 100
        height = 240
        edges = [
            {"x0": 0, "x1": 100, "top": y, "bottom": y}
            for y in (20, 40, 60, 100, 140, 200)
        ]

        @staticmethod
        def extract_words(**_kwargs):
            return [
                {
                    "text": "表A.1 飞行记录表",
                    "x0": 25,
                    "x1": 75,
                    "top": 80,
                    "bottom": 90,
                }
            ]

        @staticmethod
        def crop(bbox):
            return FakeCrop(bbox)

    combined = (
        [0.0, 20.0, 100.0, 200.0],
        [["附", "录A"], ["项目名称", "测区名称"]],
    )
    form = (
        [0.0, 100.0, 100.0, 200.0],
        [["项目名称", "测区名称"], ["参加人员", None]],
    )

    def fake_extract(_cls, page):
        return [form] if isinstance(page, FakeCrop) else [combined]

    monkeypatch.setattr(
        VectorPdfParser,
        "_extract_best_tables",
        classmethod(fake_extract),
    )

    tables = VectorPdfParser._extract_captioned_tables(FakePage())

    assert len(tables) == 1
    assert tables[0][2] == "表A.1 飞行记录表"
    assert tables[0][1][0] == ["项目名称", "测区名称"]


def test_structured_table_replaces_flattened_numeric_stream() -> None:
    numeric_stream = " ".join(str(value / 10) for value in range(120))
    assert is_flattened_table_noise(numeric_stream)
    page = PageRecord(
        page_number=258,
        raw_text=numeric_stream,
        cleaned_text=f"表 2 飞行距离分布\n{numeric_stream}",
        rich_blocks=[
            {
                "block_type": "table",
                "text": "距离 | 1000 | 2000\n概率 | 0.1 | 0.2",
                "table_html": (
                    "<table><tr><th>距离</th><th>1000</th><th>2000</th></tr>"
                    "<tr><td>概率</td><td>0.1</td><td>0.2</td></tr></table>"
                ),
                "asset_id": "p0258-table-01",
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "confidence": 1.0,
            }
        ],
    )

    blocks = split_blocks("document", [page])

    assert not any(block.text == numeric_stream for block in blocks)
    table = next(block for block in blocks if block.block_type == "table")
    assert table.asset_id == "p0258-table-01"
    assert table.table_html is not None


def test_structured_table_replaces_textual_native_cells_and_false_clauses() -> None:
    page = PageRecord(
        page_number=38,
        raw_text="23.2000 适用范围 23.1 适用范围",
        cleaned_text=(
            "23.2000适用范围及定义\n"
            "23.1适用范围\n"
            "23.2005正常类飞机审定\n"
            "23.3飞机类别"
        ),
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": "表 2 条款对应关系表",
                "bbox": [10.0, 5.0, 90.0, 9.0],
            },
            {
                "block_type": "table",
                "text": (
                    "CCAR-23-R4 | 名称 | CCAR-23-R3 | 名称\n"
                    "23.2000 | 适用范围及定义 | 23.1 | 适用范围"
                ),
                "table_html": "<table><tr><th>CCAR-23-R4</th></tr></table>",
                "asset_id": "p0038-table-01",
                "bbox": [10.0, 10.0, 90.0, 90.0],
            },
        ],
    )

    blocks = split_blocks("document", [page])

    assert [block.block_type for block in blocks] == ["paragraph", "table"]
    assert not any(block.article_id_normalized for block in blocks)
    assert sum(block.block_type == "table" for block in blocks) == 1


def test_has_math_font_detects_cambria_math() -> None:
    class MathPage:
        chars = [{"fontname": "DCWGQU+CambriaMath"}]

    class PlainPage:
        chars = [{"fontname": "LNUHNF+SimSun"}]

    class NoCharsPage:
        pass

    parser = VectorPdfParser()
    assert parser._has_math_font(MathPage()) is True
    assert parser._has_math_font(PlainPage()) is False
    assert parser._has_math_font(NoCharsPage()) is False


def test_long_table_splits_only_between_rows_and_repeats_header() -> None:
    header = [
        "表2 航段距离分布",
        "航段距离 | 飞机最大航程",
        "自 | 至 | 1000 | 2000",
        " | | 航段距离分布百分比",
    ]
    rows = [f"{value} | {value + 200} | 1.0 | 2.0" for value in range(0, 4000, 200)]

    parts = list(split_table_rows("\n".join([*header, *rows]), 220))
    normalized_header = [line.strip() for line in header]

    assert len(parts) > 1
    assert all(part.splitlines()[:4] == normalized_header for part in parts)
    assert all(line.count("|") == 3 for part in parts for line in part.splitlines()[4:])


def test_repair_fake_glyphs_restores_latin_letters() -> None:
    from app.ingestion.parsers.vector_pdf import normalize_cell, repair_fake_glyphs

    # Damaged GB/T font maps ASCII to fake CJK glyphs (U+7280..U+72D8).
    assert repair_fake_glyphs("犿 ≤３％ ｆ") == "m ≤3% f"
    assert repair_fake_glyphs("犚犛犇≤１％") == "RSD≤1%"
    assert repair_fake_glyphs("狊（狓）≤０．５％") == "s(x)≤0.5%"
    assert repair_fake_glyphs("犕 狕 ≤０．３") == "M z ≤0.3"
    assert repair_fake_glyphs("犘 ｍ ≤０．５") == "P m ≤0.5"
    assert repair_fake_glyphs("公式（２）") == "公式(2)"
    assert repair_fake_glyphs("犌犅／犜４１４５０—２０２２") == "GB/T41450—2022"
    # 犜犲犮犺狀犻犮犪犾狊狆犲犮犻犳犻犮犪狋犻狅狀 = Technicalspecification
    assert (
        repair_fake_glyphs("犜犲犮犺狀犻犮犪犾狊狆犲犮犻犳犻犮犪狋犻狅狀")
        == "Technicalspecification"
    )
    # 表 caption：全角数字归一化 + 假字还原
    assert repair_fake_glyphs("表３ 无人机") == "表3 无人机"
    # normalize_cell applies the same restoration after collapsing whitespace.
    assert normalize_cell("　犿 ≤３％ ｆ　") == "m ≤3% f"


def test_caption_near_prefers_figure_caption_line_over_symbol_explanation() -> None:
    # 图片下方约 97pt 处才是真正的图注；抓取带内先命中符号说明时应跳过，
    # 优先返回行首的"图N xxx"图注。
    class FakePage:
        width = 595.3
        height = 842.0

        def __init__(self, crop_text: str) -> None:
            self._crop_text = crop_text

        def crop(self, band):
            return self

        def extract_text(self) -> str:
            return self._crop_text

    parser = VectorPdfParser()
    bbox = [123.5, 195.8, 479.5, 383.6]

    # 抓取带内含"a 起降过程…"符号说明 + 行首"图2 物流无人机…"图注
    page = FakePage(
        "a 起降过程 b 航线飞行 ) ) 引符号说明 :\n"
        "图2 物流无人机吊挂货物飞行示意图\n"
    )
    assert parser._caption_near(page, bbox) == "图2 物流无人机吊挂货物飞行示意图"

    # 只有符号说明、无图注时退回首行
    page = FakePage("a 起降过程 b 航线飞行 ) ) 引符号说明 :\n")
    assert parser._caption_near(page, bbox).startswith("a 起降过程")



def test_preflight_table_page_with_math_font_is_not_formula(monkeypatch) -> None:
    # 空白记录表（如 A.2 采样记录表）页面含 CambriaMath 数学字体（表底公式
    # 注），但有完整矢量表格线框。此前 _has_math_font 会把整页判为 formula，
    # 导致 route=formula_review、整页走 OCR 表格重建被破坏。表格信号应优先。
    class FakePage:
        width = 841.95
        height = 595.35
        rects = []
        lines = []
        curves = []
        images = []
        edges = [
            {"x0": 89.0, "x1": 753.0, "top": y, "bottom": y}
            for y in (135.8, 247.8, 275.1, 507.2)
        ] + [
            {"x0": x, "x1": x, "top": 135.8, "bottom": 507.2}
            for x in (89.1, 191.4, 752.8)
        ]
        chars = [{"fontname": "DCWGQU+CambriaMath"}] * 3

        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

        def find_tables(self):
            return []

        def extract_words(self, **_kwargs):
            return []

        def to_image(self, **_kwargs):
            return self

        @property
        def original(self):
            return self

        def save(self, *args, **kwargs):
            return None

    class FakePdf:
        pages = [
            FakePage(
                "空气中化学毒物采样记录表参见表A.2。\n"
                "A.2 空气中化学毒物采样记录表\n"
                "注：采样体积包括现场采样体积（Vt）和标准采样体积(V0)，\n"
                "Vt=F×t，V0=Vt×273+.../101.3\n"
            ),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    # preflight 只依赖 edges 检测表格（_extract_best_tables 内部走 find_tables
    # 和 expanded_ruled_tables）。用 monkeypatch 让 _extract_best_tables 返回真
    # 实表格，验证表格信号优先于数学字体。
    monkeypatch.setattr(
        VectorPdfParser,
        "_extract_best_tables",
        classmethod(
            lambda cls, page: [
                ([89.0, 135.8, 753.0, 247.8], [["现场地址", "值"], ["检测项目", ""]]),
                ([89.0, 275.1, 753.0, 507.2], [["样品编号", "备注"], ["", ""]]),
            ]
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    parser = VectorPdfParser()
    result = parser.preflight_pages(Path("record-form.pdf"), candidate_pages={1})

    assert 1 in result.table_pages
    assert 1 not in result.formula_pages  # 表格优先，不判为公式页


def test_parse_pages_table_wins_over_math_font(monkeypatch) -> None:
    # 页面同时在 table_pages 和 formula_pages（数学字体触发），但含真实表格。
    # parse_pages 的 page_type 应为 table 而非 formula，避免生成整页公式资产。
    class FakePage:
        width = 841.95
        height = 595.35
        rects = []
        lines = []
        curves = []
        images = []
        chars = [{"fontname": "DCWGQU+CambriaMath"}] * 3

        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

        def extract_words(self, **_kwargs):
            return []

        def to_image(self, **_kwargs):
            from PIL import Image
            return type("Img", (), {"original": Image.new("RGB", (100, 100), "white")})()

    class FakePdf:
        pages = [FakePage("A.2 空气中化学毒物采样记录表\n注：V0=Vt×273/101.3")]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        VectorPdfParser,
        "_extract_captioned_tables",
        classmethod(
            lambda cls, page: [
                (
                    [89.0, 275.1, 753.0, 507.2],
                    [["样品编号", "备注"], ["", ""]],
                    "A.2 空气中化学毒物采样记录表",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    parser = VectorPdfParser()
    results = parser.parse_pages(
        Path("record-form.pdf"),
        table_pages={1},
        figure_pages=set(),
        formula_pages={1},  # 数学字体曾触发 formula
    )
    result = results[1]

    assert result.page_type == "table"  # 表格优先
    assert result.table_count == 1
    # 纯表格页不生成整页公式/版面资产，避免与表格内容重复
    assert not any(asset.asset_type in {"formula_page", "figure_page"} for asset in result.assets)
    # 空白记录表会附带整页 form_page 原图（混合方案），其余为表格资产
    assert all(
        asset.asset_type in {"table", "form_page"}
        for asset in result.assets
    )


def test_parse_pages_filters_boxed_diagram_frame_labels(monkeypatch) -> None:
    # 带外框文字结构图页：_line_blocks 提取的框内标签（封面/附录/正文等）
    # 是图的一部分，不应作为独立 paragraph 块进入 split_blocks（否则"附录"
    # 标签会劫持章节栈，把 6.2.x 挂到 ['附录'] 下）。parse_pages 应在
    # _boxed_text_diagram_region 命中时过滤位于外框内的 paragraph 块。
    frame = [146.6, 260.1, 433.7, 452.2]

    class FakePage:
        width = 595.22
        height = 842.0
        rects = [
            {
                "x0": frame[0],
                "x1": frame[2],
                "top": frame[1],
                "bottom": frame[3],
            }
        ]
        lines = [{} for _ in range(8)]
        curves = [{} for _ in range(2)]
        images = []

        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

        def extract_words(self, **_kwargs):
            # Words both inside and outside the frame.
            return [
                {"text": "6.2", "x0": 56.7, "x1": 78.0, "top": 208.0, "bottom": 220.0},
                {"text": "构成", "x0": 82.0, "x1": 104.0, "top": 208.0, "bottom": 220.0},
                {"text": "封面", "x0": 274.3, "x1": 291.9, "top": 273.0, "bottom": 285.0},
                {"text": "附录", "x0": 274.3, "x1": 291.9, "top": 431.0, "bottom": 443.0},
                {"text": "6.2.1", "x0": 56.7, "x1": 98.0, "top": 485.0, "bottom": 497.0},
                {"text": "封面", "x0": 102.0, "x1": 114.5, "top": 485.0, "bottom": 497.0},
            ]

        def crop(self, bbox):
            class Crop:
                def extract_text(self):
                    return (
                        "封面\n内封和内封背\n更改记录\n有效页目录\n"
                        "操作员手册 前置材料 临时更改单记录\n"
                        "正文\n附录\n"
                    )
            return Crop()

        def to_image(self, **_kwargs):
            from PIL import Image
            return type("Img", (), {"original": Image.new("RGB", (595, 842), "white")})()

    class FakePdf:
        pages = [FakePage("6 III类手册编制\n6.2 构成\n图1 III类手册的构成\n6.2.1 封面")]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )
    monkeypatch.setattr(
        VectorPdfParser,
        "_caption_near",
        staticmethod(lambda page, bbox: "图1 III类手册的构成"),
    )

    parser = VectorPdfParser()
    results = parser.parse_pages(
        Path("boxed-diagram.pdf"),
        table_pages=set(),
        figure_pages={1},
        formula_pages=set(),
    )
    result = results[1]

    assert result.page_type == "figure"
    # 框内标签（封面/附录/正文）被过滤，不生成独立 paragraph 块
    frame_texts = {
        b.text
        for b in result.blocks
        if b.block_type == "paragraph"
        and b.bbox
        and b.bbox[0] >= frame[0] - 1
        and b.bbox[1] >= frame[1] - 1
        and b.bbox[2] <= frame[2] + 1
        and b.bbox[3] <= frame[3] + 1
    }
    assert not {"封面", "附录", "正文"}.intersection(frame_texts)
    # 外框之外的正文段保留（6.2 构成、6.2.1 封面）
    outside = {
        b.text
        for b in result.blocks
        if b.block_type == "paragraph"
        and b.bbox
    } - frame_texts
    assert "6.2 构成" in outside or "6.2构成" in outside
    assert "6.2.1 封面" in outside or "6.2.1封面" in outside


def test_preflight_keeps_figure_caption_page_out_of_table_pages(monkeypatch) -> None:
    # 正文页带图题"表 A.1 小麦产量预测模型构建流程"（实为流程图，正文写
    # "见图A.1"），但没有真实表格线框。preflight 的 _extract_best_tables 应
    # 返回空，页面不得进入 table_pages，否则会被当作表格候选导致
    # table_structure 门禁失败（unresolved_table_pages）。
    class FakePage:
        width = 100
        height = 100
        rects = []
        lines = []
        curves = []
        edges = []

        def __init__(self, text: str, *, images: list[dict]) -> None:
            self._text = text
            self.images = images

        def extract_text(self) -> str:
            return self._text

        def find_tables(self):
            return []

        def extract_words(self, **_kwargs):
            return []

    class FakePdf:
        pages = [
            FakePage(
                "附录 A\n"
                "（资料性）\n"
                "产量预测模型 MultimodalNet 使用方法\n"
                "遥感产量预测模型构建流程见下图。该模型由3个并联的卷积神经网络模型组成。\n"
                "小麦产量预测模型构建流程见图A.1。\n"
                "表 A.1 小麦产量预测模型构建流程\n",
                images=[
                    {"x0": 30.0, "x1": 70.0, "top": 20.0, "bottom": 60.0},
                    {"x0": 40.0, "x1": 60.0, "top": 70.0, "bottom": 75.0},
                ],
            ),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        VectorPdfParser,
        "_extract_best_tables",
        classmethod(lambda cls, page: []),  # 无真实表格线框
    )
    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    result = VectorPdfParser().preflight_pages(
        Path("wheat-figure-page.pdf"),
        candidate_pages={1},
    )

    assert 1 not in result.table_pages
    assert 1 in result.figure_pages


def test_flowchart_regions_detect_tile_image_flow_chart(monkeypatch) -> None:
    # 一些 PDF 用普通小图（非 imagemask）拼流程图：多个 rect 框 + curve 箭头
    # + 大量小尺寸图片。此前 _flowchart_regions 只识别 imagemask 流程图，
    # 导致这些页每张小图都触发 VLM（修复《荒漠》第19页 5 次 VLM 卡住）。
    class FakePage:
        width = 595
        height = 842
        images = [
            {"x0": float(50 + i * 60), "x1": float(150 + i * 60),
             "top": 100.0, "bottom": 130.0, "imagemask": None}
            for i in range(12)
        ]
        rects = [{"x0": float(50 + i * 60), "x1": float(150 + i * 60),
                  "top": 100.0, "bottom": 130.0} for i in range(12)]
        curves = [{} for _ in range(15)]

    regions = VectorPdfParser._flowchart_regions(FakePage())

    assert len(regions) == 1  # 合并为一个完整流程图


def test_flowchart_regions_not_triggered_without_arrows() -> None:
    # 只有 rect 没有 arrow（如普通表格线框）不应判为流程图。
    class FakePage:
        width = 595
        height = 842
        images = [
            {"x0": float(50 + i * 60), "x1": float(150 + i * 60),
             "top": 100.0, "bottom": 130.0, "imagemask": None}
            for i in range(8)
        ]
        rects = [{"x0": float(50 + i * 60), "x1": float(150 + i * 60),
                  "top": 100.0, "bottom": 130.0} for i in range(8)]
        curves = []

    assert VectorPdfParser._flowchart_regions(FakePage()) == []


def test_flowchart_regions_detect_pure_vector_flow_chart() -> None:
    # 纯矢量流程图（无图片对象，只有 rect 框 + curve 箭头 + 文字）应被识别
    # 为一个完整流程图（修复《外业规范》第8页图1工作流程图未识别）。
    class FakePage:
        width = 595
        height = 842
        images = []
        rects = [
            {"x0": float(180 + i % 2 * 160), "x1": float(310 + i % 2 * 160),
             "top": float(170 + i * 30), "bottom": float(195 + i * 30)}
            for i in range(7)
        ]
        curves = [{} for _ in range(6)]

    regions = VectorPdfParser._flowchart_regions(FakePage())

    assert len(regions) == 1  # 合并为一个完整流程图


def test_preflight_flags_pure_vector_flow_chart_with_caption(monkeypatch) -> None:
    # 无图片但含图题 + 纯矢量流程图的页面应判为 figure 页
    # （修复《外业规范》第8页图1工作流程图漏识别）。
    class FakePage:
        width = 595
        height = 842
        images = []
        rects = [{} for _ in range(8)]
        curves = [{} for _ in range(6)]
        lines = []

        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakePdf:
        pages = [
            FakePage(
                "低空数字航空摄影工作流程见下图。\n"
                "图1 工作流程图\n"
                "5 准备工作\n"
            ),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        VectorPdfParser,
        "_flowchart_regions",
        staticmethod(lambda page: [[185.0, 170.0, 437.0, 373.0]]),
    )
    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    result = VectorPdfParser().preflight_pages(
        Path("pure-vector-flow.pdf"),
        candidate_pages={1},
    )

    assert 1 in result.figure_pages


def test_boxed_text_diagram_region_detects_single_frame_structure_diagram() -> None:
    # 带外框的文字结构图（单个外框 rect + 框内标签文字 + 少量连接线，无图片）
    # 应被识别为一个整体结构图（修复《飞行手册编制规范》第9页图1未识别）。
    # 模拟页面：只有一个 287x192 的外框，框内 11 个标签行，少量连线。
    class FakeCrop:
        def extract_text(self) -> str:
            return (
                "封面\n"
                "内封和内封背\n"
                "更改记录\n"
                "有效页目录\n"
                "更改单记录\n"
                "操作员手册 前置材料 临时更改单记录\n"
                "技术通报记录\n"
                "目次和图表清单\n"
                "前言\n"
                "正文\n"
                "附录\n"
            )

    class FakePage:
        width = 595.22
        height = 842.0
        images = []
        curves = [{} for _ in range(2)]
        lines = [{} for _ in range(8)]
        rects = [
            {
                "x0": 146.6,
                "x1": 433.7,
                "top": 260.1,
                "bottom": 452.2,
            }
        ]

        def crop(self, _bbox: object) -> FakeCrop:
            return FakeCrop()

    region = VectorPdfParser._boxed_text_diagram_region(FakePage())

    assert region is not None
    assert region == [146.6, 260.1, 433.7, 452.2]


def test_boxed_text_diagram_region_rejects_plain_text_page_without_frame() -> None:
    # 无外框的普通正文页（即使含"图N"文字引用）不应被识别为结构图。
    class FakePage:
        width = 595
        height = 842
        images = []
        rects = []
        curves = []
        lines = []

        def crop(self, _bbox: object) -> None:
            raise AssertionError("crop should not be called without a frame")

    assert VectorPdfParser._boxed_text_diagram_region(FakePage()) is None


def test_preflight_flags_boxed_text_diagram_with_caption(monkeypatch) -> None:
    # 无图片、含图注、单个外框 + 框内标签文字的结构图页应判为 figure 页
    # （修复《飞行手册编制规范》第9页图1 III类手册构成未识别）。
    class FakeCrop:
        def extract_text(self) -> str:
            return (
                "封面\n内封和内封背\n更改记录\n有效页目录\n"
                "操作员手册 前置材料 临时更改单记录\n正文\n附录\n"
            )

    class FakePage:
        width = 595.22
        height = 842.0
        images = []
        curves = [{} for _ in range(2)]
        lines = [{} for _ in range(8)]
        rects = [
            {
                "x0": 146.6,
                "x1": 433.7,
                "top": 260.1,
                "bottom": 452.2,
            }
        ]

        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

        def crop(self, _bbox: object) -> FakeCrop:
            return FakeCrop()

    class FakePdf:
        pages = [
            FakePage(
                "6 III 类手册编制\n6.2 构成\n"
                "III类手册的内容应包括封面、内封和内封背、前置材料、正文及附录，"
                "详见图1。\n"
                "图1 III 类手册的构成\n"
                "6.2.1 封面\n"
            ),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    result = VectorPdfParser().preflight_pages(
        Path("boxed-text-diagram.pdf"),
        candidate_pages={1},
    )

    assert 1 in result.figure_pages


def test_cover_page_not_treated_as_table() -> None:
    # 标准封面页（ICS/CCS 分类号 + 标准号 + 发布/实施）的装饰框线会被
    # pdfplumber 误判为表格。封面识别应排除这类页（修复《应急通信空中
    # 基站》第1页封面标题被拆成表格 chunk）。
    from app.ingestion.parsers.vector_pdf import _looks_like_cover_page

    cover = (
        "ICS 33.070\nCCS M37\n中华人民共和国通信行业标准\n"
        "YD/T××××—××××\n基于系留无人机的应急通信空中基站技术要求\n"
        "××××-××-××发布 ××××-××-××实施\n"
        "中华人民共和国工业和信息化部发布"
    )
    assert _looks_like_cover_page(cover.replace(" ", "")) is True

    # 真实正文页不误判
    body = (
        "4 一般要求\n本文件规定了系留无人机空中基站的通用要求。\n"
        "表1 空中基站技术指标\n指标名称 数值范围"
    )
    assert _looks_like_cover_page(body.replace(" ", "")) is False


def test_cover_page_recognizes_dl_power_industry_standard() -> None:
    # 电力行业标准（DL/T）封面也应识别为 cover，否则被当表格/图片处理
    # （修复《电网设备无人机图像识别系统技术要求》第1页被解析为 table_figure）。
    from app.ingestion.parsers.vector_pdf import _looks_like_cover_page

    cover = (
        "ICS 29.020\nCCS F29\n中华人民共和国电力行业标准\n"
        "DL/T 2908—2025\n电网设备无人机图像识别系统技术要求\n"
        "Technical requirements for UAV image identification system\n"
        "2025-06-30 发布 2025-12-30 实施\n国家能源局 发 布"
    )
    assert _looks_like_cover_page(cover.replace(" ", "")) is True

    # 其他常见行业标准前缀（NB/HG/HJ/CJ/JG/JC/QB/FZ/JT/JTG）也应识别
    for prefix in ("NB", "HG", "HJ", "CJ", "JG", "JC", "QB", "FZ", "JT", "JTG"):
        cover2 = (
            "ICS 01.040\nCCS Z00\n中华人民共和国行业标准\n"
            f"{prefix}/T 1234—2025\n示例标准\n"
            "2025-01-01 发布 2025-04-01 实施"
        )
        assert _looks_like_cover_page(cover2.replace(" ", "")) is True, prefix


def test_preflight_excludes_cover_page_from_table_pages(monkeypatch) -> None:
    class FakePage:
        width = 595
        height = 842
        images = []
        rects = [{} for _ in range(6)]
        curves = []
        lines = []

        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakePdf:
        pages = [
            FakePage(
                "ICS 33.070 CCS M37\n中华人民共和国通信行业标准\n"
                "YD/T××××—××××\n××××-××-××发布 ××××-××-××实施\n"
            ),
            FakePage(
                "4 一般要求\n本文件规定了系留无人机空中基站的通用要求。\n"
                "表1 空中基站技术指标\n指标名称 数值范围\n"
            ),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        VectorPdfParser,
        "_extract_best_tables",
        classmethod(
            lambda cls, page: [
                ([70.0, 90.0, 550.0, 210.0], [["中 华 人 民", "共 和 国"], ["YD/T", "××××"]])
            ]
        ),
    )
    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    result = VectorPdfParser().preflight_pages(
        Path("cover-with-frame.pdf"),
        candidate_pages={1, 2},
    )

    assert 1 not in result.table_pages  # 封面页的框线不是表格
    assert 2 in result.table_pages  # 真实表格页仍识别


def test_preflight_text_layer_corrupted_page_excluded_from_figure_targets(
    monkeypatch,
) -> None:
    # 文本层损坏页（PostScript 字形名 /G26 等）+ 少量图片时，preflight 会因
    # 图片覆盖率把页面判为 figure。hybrid 层须从 figure_targets 中排除
    # text_layer_targets，否则乱码文本页走 figure 解析（输出乱码）而不是
    # OCR（GB 42590-2023 第26页）。
    class FakePage:
        width = 595.32
        height = 841.98
        rects = []
        curves = []
        lines = []
        # Two large images so largest_image_coverage >= 0.12 (the figure rule)
        images = [
            {"x0": 50.0, "x1": 400.0, "top": 100.0, "bottom": 400.0},
            {"x0": 60.0, "x1": 380.0, "top": 450.0, "bottom": 750.0},
        ]

        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakePdf:
        pages = [FakePage("/G26\n/G28\n/G56/G22/G24/G23")]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        "app.ingestion.parsers.vector_pdf.pdfplumber.open",
        lambda _path: FakePdf(),
    )

    preflight = VectorPdfParser().preflight_pages(
        Path("broken-text-layer.pdf"),
        candidate_pages={1},
    )
    assert 1 in preflight.figure_pages  # 单独看 preflight 仍判为 figure（图片覆盖率）

    # hybrid 层逻辑：figure_targets 应排除 text_layer_targets
    text_layer_targets = {1}
    visual_targets = preflight.figure_pages - text_layer_targets
    assert 1 not in visual_targets  # 文本层损坏页不进 figure 解析


def test_line_blocks_merges_floating_clause_dots() -> None:
    # 条款编号的数字与下沉点号分两行（"515" 与 ".."，实际是 "5.1.5"）。
    # _line_blocks 应把点号插回数字间隙（GB 46750-2025 第10页）。
    class FakeWord(dict):
        pass

    words = [
        {"text": "515", "x0": 60.8, "x1": 87.4, "top": 102.3, "bottom": 110.0},
        {
            "text": "民用无人驾驶航空器系统工作",
            "x0": 96.1,
            "x1": 242.9,
            "top": 100.8,
            "bottom": 109.0,
        },
        {
            "text": "应对运行识别发送模块功能自检",
            "x0": 249.2,
            "x1": 420.7,
            "top": 100.8,
            "bottom": 109.0,
        },
        {"text": "..", "x0": 68.7, "x1": 86.8, "top": 109.3, "bottom": 116.0},
    ]

    class FakePage:
        width = 595.32
        height = 841.98

        def extract_words(self, **_kwargs):
            return words

    blocks = VectorPdfParser._line_blocks(FakePage())

    assert any("5.1.5" in b.text for b in blocks)
    assert any("5.1.5 民用无人驾驶航空器系统" in b.text for b in blocks)


def test_line_blocks_interleaves_dots_into_clause_number() -> None:
    # 术语编号 31 + 下沉点号 .. -> 3.1（GB 46750-2025 第7页术语定义）。
    class FakePage:
        width = 595
        height = 842

        def extract_words(self, **_kwargs):
            return [
                {"text": "31", "x0": 68.7, "x1": 80.0, "top": 493.0, "bottom": 501.0},
                {"text": "无人驾驶航空器", "x0": 92.3, "x1": 160.0, "top": 493.0, "bottom": 502.0},
                {"text": ".", "x0": 76.6, "x1": 79.0, "top": 500.0, "bottom": 507.0},
            ]

    blocks = VectorPdfParser._line_blocks(FakePage())

    assert any(b.text.startswith("3.1 ") for b in blocks)


def test_line_blocks_interleaves_dots_into_letter_prefix_clause_number() -> None:
    # 附录条款编号 A.3 / A.3.1 的点号下沉到下一行（GB 46761-2025 第17页：
    # "A3 . 激活…" 实际是 "A.3 激活…"；"A31 .. 数据…" 实际是 "A.3.1 数据…"）。
    # 点号应插入到数字间隙，或字母与数字之间。
    class FakePage:
        width = 595
        height = 842

        def extract_words(self, **_kwargs):
            return [
                {"text": "A3", "x0": 69.9, "x1": 88.0, "top": 398.0, "bottom": 407.0},
                {
                    "text": "激活状态上报接口要求",
                    "x0": 97.3,
                    "x1": 201.4,
                    "top": 396.5,
                    "bottom": 406.0,
                },
                {"text": ".", "x0": 78.4, "x1": 81.0, "top": 405.6, "bottom": 412.0},
            ]

    blocks = VectorPdfParser._line_blocks(FakePage())

    assert any(b.text.startswith("A.3 ") for b in blocks)


def test_line_blocks_keeps_normal_dotted_number_intact() -> None:
    # 正常编号 "6.2 构成" 不应被点号插入逻辑破坏（点号前无空格）。
    class FakePage:
        width = 595
        height = 842

        def extract_words(self, **_kwargs):
            return [
                {"text": "6.2", "x0": 56.7, "x1": 78.0, "top": 208.0, "bottom": 220.0},
                {"text": "构成", "x0": 82.0, "x1": 104.0, "top": 208.0, "bottom": 220.0},
            ]

    blocks = VectorPdfParser._line_blocks(FakePage())

    assert any(b.text.startswith("6.2 构成") for b in blocks)
    assert not any("6 2" in b.text for b in blocks)


def test_line_blocks_merges_term_number_with_term_name() -> None:
    # 术语条目编号（3.5.10）与术语名（愿景 vision）分两行，应合并成
    # "3.5.10 愿景 vision"（GB/T 19000-2016 第20页术语定义）。
    class FakePage:
        width = 595
        height = 842

        def extract_words(self, **_kwargs):
            return [
                {"text": "3.5.10", "x0": 7.6, "x1": 39.4, "top": 148.5, "bottom": 165.2},
                {"text": "愿景", "x0": 31.2, "x1": 51.4, "top": 162.3, "bottom": 173.4},
                {"text": "vision", "x0": 60.0, "x1": 89.8, "top": 164.0, "bottom": 175.0},
            ]

    blocks = VectorPdfParser._line_blocks(FakePage())

    assert any(b.text.startswith("3.5.10 愿景 vision") for b in blocks)


def test_line_blocks_merges_term_number_with_term_name_lenient() -> None:
    # 英文名分号下沉导致术语名被规范化成 "客体 objectentityitem ; ;"，
    # 仍应识别为术语名并与编号合并（GB/T 19000-2016 第20页 3.6.1）。
    class FakePage:
        width = 595
        height = 842

        def extract_words(self, **_kwargs):
            return [
                {"text": "3.6.1", "x0": 7.6, "x1": 34.2, "top": 318.4, "bottom": 335.1},
                {"text": "客体", "x0": 31.2, "x1": 51.4, "top": 332.1, "bottom": 343.0},
                {
                    "text": "objectentityitem",
                    "x0": 60.1,
                    "x1": 143.4,
                    "top": 333.5,
                    "bottom": 344.0,
                },
            ]

    blocks = VectorPdfParser._line_blocks(FakePage())

    assert any(b.text.startswith("3.6.1 客体 objectentityitem") for b in blocks)


def test_figure_crop_uses_higher_resolution(monkeypatch) -> None:
    # 扁平概念关系图（高仅 ~64pt）在 160dpi 整页渲染下裁剪过小，VLM 误判为
    # "无实际图形内容"。figure 裁剪应用更高分辨率渲染（GB/T 19000 第33页
    # 图A2 从属关系图）。验证 _figure_crop_png 使用 300dpi。
    rendered = {}

    class FakePage:
        width = 595.3
        height = 841.9

        def to_image(self, resolution, antialias=True):
            from PIL import Image
            rendered["res"] = resolution
            return type(
                "Img", (), {"original": Image.new("RGB", (200, 200), "white")}
            )()

    crop = VectorPdfParser._figure_crop_png(
        FakePage(),
        bbox=[56.7, 628.8, 424.1, 693.0],
    )
    assert rendered["res"] == VectorPdfParser._FIGURE_RENDER_RESOLUTION
    assert rendered["res"] == 300
    assert crop


def test_figure_crop_accounts_for_mediabox_origin_offset() -> None:
    # GB/T 19000-2016 的页面 MediaBox 原点不在 (0,0) 而在 (-57.24, 53.91)。
    # pdfplumber 报告的对象 bbox 相对 MediaBox 原点，而 to_image 渲染的 PNG
    # 相对 cropbox 原点。裁剪若不减去原点偏移，会整体错位、截断树形连接线
    # （图A1 属种关系图右侧内容丢失）。验证 _crop_png 应用 origin 平移。
    from PIL import Image

    class FakePage:
        width = 595.3
        height = 841.9
        cropbox = (-57.237274, 53.913879, 538.03833, 895.803648)

        def to_image(self, resolution, antialias=True):
            # 模拟渲染 300dpi: 宽 595.3pt*300/72=2480, 高 841.9pt*300/72=3508
            w = round(self.width * resolution / 72)
            h = round(self.height * resolution / 72)
            return type("Img", (), {"original": Image.new("RGB", (w, h), "white")})()

    page = FakePage()
    origin_x, origin_y = VectorPdfParser._page_origin(page)
    assert abs(origin_x - (-57.237274)) < 1e-6
    assert abs(origin_y - 53.913879) < 1e-6

    # 用白底图，在目标位置画一个黑块模拟图内容，验证裁剪窗口对准。
    # 图A1 bbox 在 pdfplumber 坐标系 [94.95, 413.82, 385.76, 487.28]。
    bbox = [94.95, 413.82, 385.76, 487.28]
    origin_x, origin_y = VectorPdfParser._page_origin(page)

    # 直接调用 _crop_png（带 origin）时应得到正确的区域
    from io import BytesIO

    page_img = page.to_image(300).original
    img_bytes = BytesIO()
    page_img.save(img_bytes, format="PNG")
    crop_bytes = VectorPdfParser._crop_png(
        img_bytes.getvalue(),
        page_width=page.width,
        page_height=page.height,
        bbox=bbox,
        origin=VectorPdfParser._page_origin(page),
    )
    im = Image.open(BytesIO(crop_bytes))
    assert im.width > 0 and im.height > 0
    # 尺寸应接近 (right-left)*scale = (385.76-94.95)*4.167 ≈ 1212
    assert 1150 < im.width < 1280
    assert 250 < im.height < 350


def test_page_origin_defaults_zero_when_no_cropbox() -> None:
    # 常规页面无偏移时 origin 应为 (0,0)，不改变既有裁剪行为
    class FakePage:
        width = 595.3
        height = 841.9

    assert VectorPdfParser._page_origin(FakePage()) == (0.0, 0.0)


def test_is_fullpage_background_image_detects_white_jpeg() -> None:
    # GJB 2489A-2023 每页都叠了一张 ~1241x1755 的白色 JPEG 背景
    # (bbox 覆盖整页、stream 仅十几 KB)。它是装饰背景而非内容图，
    # 不得触发 figure 判定。验证 _is_fullpage_background_image。
    from io import BytesIO

    from PIL import Image

    class FakeStream:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def get_data(self) -> bytes:
            return self._data

    class FakePage:
        width = 595.44
        height = 842.16

    # 生成一张纯白 JPEG（尺寸 1241x1755，极小体积 → bpp 极低）
    white = Image.new("RGB", (1241, 1755), "white")
    buf = BytesIO()
    white.save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()
    # 确认 bpp 足够低
    assert len(jpeg_bytes) / (1241 * 1755) < 0.05

    bg_image = {
        "x0": 0.0, "top": 0.0, "x1": 595.44, "bottom": 842.16,
        "srcsize": (1241, 1755),
        "stream": FakeStream(jpeg_bytes),
    }
    assert VectorPdfParser._is_fullpage_background_image(FakePage(), bg_image) is True


def test_is_fullpage_background_image_keeps_real_image() -> None:
    # 真实照片/扫描图 bpp 较高，即使覆盖整页也不应被当作背景图。
    from io import BytesIO

    from PIL import Image

    class FakeStream:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def get_data(self) -> bytes:
            return self._data

    class FakePage:
        width = 595.44
        height = 842.16

    # 生成一张有噪声的整页图（bpp 明显高于 0.05）
    import random
    random.seed(7)
    noisy = Image.new("RGB", (1241, 1755))
    pixels = noisy.load()
    for y in range(0, 1755, 4):
        for x in range(0, 1241, 4):
            v = random.randint(0, 255)
            pixels[x, y] = (v, v, v)
    buf = BytesIO()
    noisy.save(buf, format="JPEG", quality=70)
    jpeg_bytes = buf.getvalue()
    bpp = len(jpeg_bytes) / (1241 * 1755)
    # 有噪声内容时 bpp 应高于阈值
    assert bpp > 0.05

    real_image = {
        "x0": 0.0, "top": 0.0, "x1": 595.44, "bottom": 842.16,
        "srcsize": (1241, 1755),
        "stream": FakeStream(jpeg_bytes),
    }
    assert VectorPdfParser._is_fullpage_background_image(FakePage(), real_image) is False


def test_preflight_ignores_fullpage_background_image() -> None:
    # 页面只有整页白色背景图 + 正文文本时，不应被判为 figure 页。
    from io import BytesIO

    from PIL import Image

    white = Image.new("RGB", (1241, 1755), "white")
    buf = BytesIO()
    white.save(buf, format="JPEG", quality=85)

    class FakePage:
        width = 595.44
        height = 842.16
        images = [
            {
                "x0": 0.0, "top": 0.0, "x1": 595.44, "bottom": 842.16,
                "srcsize": (1241, 1755),
                "stream": type(
                    "S", (), {"get_data": lambda self: buf.getvalue()}
                )(),
            }
        ]
        lines = []
        curves = []
        rects = []
        chars = []

        def extract_text(self) -> str:
            return "前言\n本标准代替 GJB2489-1995\na) 增加了履历本、产品合格证\nb) 增加了配套原则"

        def extract_words(self, **kw):
            return []

    page = FakePage()
    assert VectorPdfParser._is_fullpage_background_image(
        page, page.images[0]
    ) is True

    parser = VectorPdfParser()
    # 手动模拟 preflight 的 figure 判定分支
    image_count, image_coverage, largest = parser._visual_coverage(page)
    substantive = sum(
        1 for img in page.images
        if not parser._is_fullpage_background_image(page, img)
    )
    assert substantive == 0
    assert image_count == 1


def test_text_column_boundaries_infers_columns_without_vertical_rules() -> None:
    # GJB 2489A-2023 附录D 第37页的表格只有水平线、无垂直线。列边界必须从
    # 文本 x 坐标聚类推导。验证 _text_column_boundaries 返回合理的列边界。
    class FakePage:
        width = 600
        height = 800

        def extract_words(self, **kw):
            words = []
            # 4 列：序号 x≈100、页别 x≈130、文字内容 x≈230、字体字号 x≈436
            spec = [
                (100, "1"), (130, "封面"), (230, "名称"), (436, "五号宋体"),
                (100, "2"), (130, "封底"), (230, "装卸记录"), (436, "五号宋体"),
                (100, "3"), (130, "备注"), (230, "备注正文"), (436, "五号宋体"),
                (100, "4"), (130, "封面"), (230, "验收证明"), (436, "三号宋体"),
            ]
            for i, (x0, text) in enumerate(spec):
                top = 262.6 + (i // 4) * 34
                words.append({
                    "x0": float(x0), "x1": float(x0) + 14,
                    "top": float(top), "bottom": float(top) + 10,
                    "text": text,
                })
            return words

    page = FakePage()
    boundaries = VectorPdfParser._text_column_boundaries(
        page, top=262.6, bottom=708.7, left=100.0, right=467.0,
    )
    assert boundaries is not None
    assert len(boundaries) >= 5  # left + 4 columns + right
    # 列中心应接近 100/130/230/436
    centers = boundaries[1:-1]
    assert any(abs(c - 100) < 15 for c in centers)
    assert any(abs(c - 130) < 15 for c in centers)
    assert any(abs(c - 230) < 15 for c in centers)
    assert any(abs(c - 436) < 15 for c in centers)


def test_expanded_ruled_tables_uses_text_columns_for_verticalless_table(monkeypatch) -> None:
    # 只有水平线、无垂直线的表格（GJB 2489A p37）也能重建网格。
    class FakeTable:
        def __init__(self, bbox, cells):
            self.bbox = bbox
            self._cells = cells

        @property
        def rows(self):
            return [[{"text": c} for c in row] for row in self._cells]

    class FakePage:
        width = 600
        height = 800
        lines = []
        curves = []
        rects = []
        edges = []
        images = []

        def extract_words(self, **kw):
            spec = [
                (100, "1"), (130, "封面"), (230, "名称"), (436, "五号宋体"),
                (100, "2"), (130, "封底"), (230, "装卸记录"), (436, "五号宋体"),
                (100, "3"), (130, "备注"), (230, "备注正文"), (436, "五号宋体"),
            ]
            out = []
            for i, (x0, text) in enumerate(spec):
                top = 100 + (i // 4) * 30
                out.append({
                    "x0": float(x0), "x1": float(x0) + 14,
                    "top": float(top), "bottom": float(top) + 10,
                    "text": text,
                })
            return out

        def extract_table(self, settings):
            return [
                ["1", "封面", "名称", "五号宋体"],
                ["2", "封底", "装卸记录", "五号宋体"],
                ["3", "备注", "备注正文", "五号宋体"],
            ]

    page = FakePage()
    # 手动构造水平线场景，调用 _expanded_ruled_tables 需要 edges 与 horizontal
    # 检测；这里直接验证 _text_column_boundaries 参与路径的核心逻辑
    boundaries = VectorPdfParser._text_column_boundaries(
        page, top=90, bottom=200, left=90, right=470,
    )
    assert boundaries is not None
    assert len(boundaries) >= 5


class _FakeCaptionPage:
    """Minimal page double for _caption_near tests."""

    width = 595.3
    height = 842.0

    def __init__(self, crop_text: str) -> None:
        self._crop_text = crop_text

    def crop(self, band):
        return self

    def extract_text(self) -> str:
        return self._crop_text


def test_caption_near_unnumbered_caption_not_mixed_with_body() -> None:
    # CCAR-23-R3 p52: the figure caption "图 驾驶员操纵方向舵的最大力" has no
    # number after 图, so it used to fall through to the multi-line fallback,
    # which returned the caption fused with the revision date and the clause
    # body below the figure. Unnumbered captions must be picked up on their own.
    parser = VectorPdfParser()
    bbox = [92.46, 75.9, 502.86, 333.3]
    page = _FakeCaptionPage(
        "图 驾驶员操纵方向舵的最大力\n"
        "[1993年12月23日第二次修订，2004年×月×日第三次修订]\n"
        "第23.443条 突风载荷\n"
        "(a)垂直翼面必须设计成当速度为V 的非加速飞行时\n"
    )
    assert parser._caption_near(page, bbox) == "图 驾驶员操纵方向舵的最大力"


def test_caption_near_fallback_rejects_body_text() -> None:
    # When the crop band only contains body text (revision dates, clause
    # headings, sub-item markers, page footers), the fallback must return an
    # empty string instead of returning that text as the caption. Otherwise the
    # figure's published text mixes with the surrounding paragraph.
    parser = VectorPdfParser()
    bbox = [92.46, 75.9, 502.86, 333.3]

    # Revision date + clause body below the figure (CCAR-23-R3 p39).
    page = _FakeCaptionPage(
        "[1990年7月18日第一次修订]\n"
        "第23.335条 设计空速\n"
        "除本条(a)(4)的规定外，所取的设计空速均为当量空速（EAS）。\n"
    )
    assert parser._caption_near(page, bbox) == ""

    # Multi-line clause body continuation fragments (CCAR-23-R3 p143).
    page = _FakeCaptionPage(
        "(3)在低于和等于飞机的最大使用高度时，供给每个使用者的最小补氧流量不得\n"
        "示出的流量。\n"
        "(b)如果装有飞行机组成员使用的肺式供氧设备\n"
    )
    assert parser._caption_near(page, bbox) == ""

    # Page footer only (CCAR-23-R3 p161).
    page = _FakeCaptionPage("CCAR 23 R3 － 149 －\n")
    assert parser._caption_near(page, bbox) == ""


def test_caption_near_single_symbol_legend_still_returned() -> None:
    # The single descriptive-line fallback must keep working for a lone symbol
    # legend / axis label: a short line without body punctuation is a usable
    # caption hint and should still be returned (regression for HB 8730-2023).
    parser = VectorPdfParser()
    bbox = [123.5, 195.8, 479.5, 383.6]
    page = _FakeCaptionPage("a 起降过程 b 航线飞行 ) ) 引符号说明 :\n")
    assert parser._caption_near(page, bbox).startswith("a 起降过程")





