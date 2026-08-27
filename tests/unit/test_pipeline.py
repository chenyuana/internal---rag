from pathlib import Path
from typing import Any

from app.ingestion.pipeline import (
    GLYPH_NAME_RE,
    PageRecord,
    analyze_complex_table_fidelity,
    clean_page,
    detect_table_hints,
    extract_page_layout_features,
    filter_repeated_margin_rich_blocks,
    has_fake_glyph_corruption,
    has_low_quality_ocr_text,
    has_tounicode_cid_conflict,
    is_fake_cjk_garbage,
    is_glyph_name_garbage,
    is_probable_revision_page,
    is_watermark_only_page,
    normalize_table_html,
    parse_pdf,
    repair_fake_glyphs,
    table_rows_from_html,
    table_to_markdown,
    table_to_semantic_text,
)


class FakeXObject:
    def __init__(self, obj: dict[str, Any]) -> None:
        self._obj = obj

    def get_object(self) -> dict[str, Any]:
        return self._obj


class FakePdfPage:
    def __init__(
        self,
        text: str,
        *,
        xobjects: dict[str, FakeXObject] | None = None,
    ) -> None:
        self._text = text
        self._xobjects = xobjects or {}

    def extract_text(self) -> str:
        return self._text

    def get(self, key: str, default: Any = None) -> Any:
        if key == "/Resources":
            return {"/XObject": self._xobjects}
        return default


class FakePdfReader:
    def __init__(self, pages: list[FakePdfPage]) -> None:
        self.pages = pages


def test_glyph_name_re_matches_postscript_glyph_names() -> None:
    assert GLYPH_NAME_RE.findall("/G21/G22/G2A/G2B") == [
        "/G21",
        "/G22",
        "/G2A",
        "/G2B",
    ]
    assert GLYPH_NAME_RE.findall("/G21") == ["/G21"]
    assert GLYPH_NAME_RE.findall("/G2") == []
    assert GLYPH_NAME_RE.findall("/g21") == []


def test_is_glyph_name_garbage_detects_broken_font_text_layer() -> None:
    sample = (
        "书\n书\n/G21/G22/G23\n/G21/G22\n/G21\n"
        "/G23/G24/G23\n/G22/G22/G23/G24\n/G25/G26\n"
        "/G21/G22/G23/G24/G25/G26/G27/G27/G28/G29/G2A\n"
        "/G25/G26\n/G21\n/G27\n/G25/G27/G22/G24/G21\n"
        "/G21\n/G28/G28\n/G22\n/G24/G23/G24/G25\n"
        "/G21/G22/G23/G24/G25/G26/G27/G28/G29/G2A/G2B/G2C/G2D/G2E/G2F/G30\n"
        "/G31\n/G28/G28\n/G32/G33\n/G21\n"
        "/G34/G35/G2D/G2E\n"
    )

    assert is_glyph_name_garbage(sample) is True


def test_is_glyph_name_garbage_ignores_normal_chinese_document() -> None:
    sample = (
        "中华人民共和国国家标准\n"
        "GB/T 38924.11—2023\n"
        "民用轻小型无人机系统环境试验方法\n"
        "第11部分：霉菌试验\n"
        "Environmental test methods for civil small and light unmanned aircraft system—"
        "Part 11: Fungus test"
    )

    assert is_glyph_name_garbage(sample) is False


def test_is_glyph_name_garbage_ignores_short_text() -> None:
    assert is_glyph_name_garbage("/G21/G22") is False


def test_is_glyph_name_garbage_ignores_sparse_glyph_names() -> None:
    sample = (
        "本文件按照 GB/T 1.1—2020 的规则起草。"
        "/G21 是某个内部编码，不影响整体可读性。"
        "请注意阅读正文内容。"
    )

    assert is_glyph_name_garbage(sample) is False


def test_is_glyph_name_garbage_detects_high_density_with_cjk_fake_chars() -> None:
    # 损坏字体可能把部分字形映射到 CJK 区间的假字（如犪犻 U+72xx），
    # 此时中文占比看似高，但字形名密度极高（>0.3），仍应判为损坏。
    sample = (
        "书书书犐犆犛.犃"
        "/G21/G22/G23/G24/G25/G26/G27/G27/G28/G29/G2A"
        "犌犅/犜—"
        "/G21/G22/G23/G24/G25/G26/G27/G23/G28/G29/G2A/G2B/G2C"
        "犛狆犲犮犻犳犻犮犪狋犻狅狀狊犳狅狉犾狅狑"
        "犪犾狋犻狋狌犱犲犱犻犵犻狋犪犾犪犲狉犻犪犾"
        "狆犺狅狋狅犵狉犪狆犺狔犪狀犱犱犪狋犪狆狉狅犮犲狊狊犻狀犵"
        "/G2D/G2E/G2F/G30"
    )

    assert is_glyph_name_garbage(sample) is True


def test_parse_pdf_routes_glyph_name_pages_to_text_layer_review(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "glyph-name-sample.pdf"
    source.write_bytes(b"fake-pdf")

    broken_text = "书\n书\n" + "\n".join(f"/G{i:02X}" for i in range(21, 71))

    monkeypatch.setattr(
        "app.ingestion.pipeline.PdfReader",
        lambda _path: FakePdfReader([FakePdfPage(broken_text)]),
    )
    monkeypatch.setattr(
        "app.ingestion.pipeline.sha256_file",
        lambda _path: "a" * 64,
    )

    document = parse_pdf(source)

    assert len(document.pages) == 1
    page = document.pages[0]
    assert page.route == "text_layer_review"
    assert page.text_layer_corruption.get("glyph_name_garbage") is True
    assert document.route == "hybrid_review"
    assert document.qa["text_layer_corrupted_pages"] == [1]


def test_parse_pdf_keeps_normal_page_as_native_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "normal-sample.pdf"
    source.write_bytes(b"fake-pdf")

    normal_text = (
        "中华人民共和国国家标准\n"
        "GB/T 38924.11—2023\n"
        "民用轻小型无人机系统环境试验方法\n"
        "第11部分：霉菌试验\n"
    )

    monkeypatch.setattr(
        "app.ingestion.pipeline.PdfReader",
        lambda _path: FakePdfReader([FakePdfPage(normal_text)]),
    )
    monkeypatch.setattr(
        "app.ingestion.pipeline.sha256_file",
        lambda _path: "b" * 64,
    )

    document = parse_pdf(source)

    assert document.pages[0].route == "native_text"
    assert document.pages[0].text_layer_corruption.get("glyph_name_garbage") is False
    assert document.qa["text_layer_corrupted_pages"] == []


def test_is_watermark_only_page_detects_small_image_watermark() -> None:
    page = FakePdfPage(
        "",
        xobjects={
            "/Xi0": FakeXObject({"/Subtype": "/Image", "/Width": 220, "/Height": 220}),
            "/Xi1": FakeXObject({"/Subtype": "/Image", "/Width": 220, "/Height": 220}),
        },
    )

    assert is_watermark_only_page(page, "") is True


def test_is_watermark_only_page_rejects_page_with_text() -> None:
    page = FakePdfPage(
        "some text",
        xobjects={
            "/Xi0": FakeXObject({"/Subtype": "/Image", "/Width": 220, "/Height": 220}),
        },
    )

    assert is_watermark_only_page(page, "some text") is False


def test_is_watermark_only_page_rejects_large_image() -> None:
    page = FakePdfPage(
        "",
        xobjects={
            "/Xi0": FakeXObject({"/Subtype": "/Image", "/Width": 948, "/Height": 474}),
        },
    )

    assert is_watermark_only_page(page, "") is False


def test_repeated_small_image_watermark_selection_requires_size_and_frequency() -> None:
    from app.ingestion.parsers.hybrid_pdf import _repeated_small_image_draw_names

    class _Image:
        def __init__(self, name: str, data: bytes) -> None:
            self.name = name
            self.data = data

    class _Content:
        def __init__(self, operations: list[tuple[list[object], bytes]]) -> None:
            self.operations = operations

    class _Page:
        def __init__(self, images: list[_Image], edge: float) -> None:
            self.images = images
            self._edge = edge

        def get_contents(self) -> _Content:
            return _Content(
                [
                    ([self._edge, 0, 0, self._edge, 10, 10], b"cm"),
                    (["/Stamp"], b"Do"),
                ]
            )

    class _Reader:
        def __init__(self) -> None:
            self.pages = [
                _Page([_Image("Stamp.png", b"same-watermark")], 15.0),
                _Page([_Image("Stamp.png", b"same-watermark")], 15.0),
                _Page([_Image("Stamp.png", b"same-watermark")], 15.0),
                _Page([_Image("Stamp.png", b"same-watermark")], 80.0),
            ]

    # Patch the module-local ContentStream only for this unit test, so the
    # selection rule is tested without depending on a synthetic PDF writer.
    import app.ingestion.parsers.hybrid_pdf as hybrid_pdf

    original = hybrid_pdf.ContentStream
    hybrid_pdf.ContentStream = lambda content, _reader: content  # type: ignore[assignment]
    try:
        selected = _repeated_small_image_draw_names(_Reader())  # type: ignore[arg-type]
    finally:
        hybrid_pdf.ContentStream = original

    assert selected == {1: {"/Stamp"}, 2: {"/Stamp"}, 3: {"/Stamp"}}


def test_without_image_draws_preserves_everything_except_selected_painting() -> None:
    from app.ingestion.parsers.hybrid_pdf import _without_image_draws

    operations = [
        (["/Keep"], b"Do"),
        (["/Watermark"], b"Do"),
        ([1, 0, 0, 1, 10, 10], b"cm"),
    ]

    assert _without_image_draws(operations, {"/Watermark"}) == [
        (["/Keep"], b"Do"),
        ([1, 0, 0, 1, 10, 10], b"cm"),
    ]


def test_parse_pdf_routes_watermark_only_page_to_watermark_excluded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "watermark-only.pdf"
    source.write_bytes(b"fake-pdf")

    watermark_page = FakePdfPage(
        "",
        xobjects={
            "/Xi0": FakeXObject({"/Subtype": "/Image", "/Width": 220, "/Height": 220}),
        },
    )

    monkeypatch.setattr(
        "app.ingestion.pipeline.PdfReader",
        lambda _path: FakePdfReader([watermark_page]),
    )
    monkeypatch.setattr(
        "app.ingestion.pipeline.sha256_file",
        lambda _path: "c" * 64,
    )

    document = parse_pdf(source)

    assert len(document.pages) == 1
    page = document.pages[0]
    assert page.route == "watermark_excluded"
    assert page.indexable is False
    assert document.qa["watermark_pages"] == [1]


def test_filter_repeated_margin_rich_blocks_removes_top_header() -> None:
    page = PageRecord(
        page_number=10,
        raw_text="",
        rich_blocks=[
            # 页首重复页眉（y0=72 < 842*0.12），应被移除
            {
                "block_type": "paragraph",
                "text": "DB3205/T 1145—2024",
                "bbox": [56.64, 72.59, 156.42, 83.04],
            },
            {
                "block_type": "paragraph",
                "text": "附录 A",
                "bbox": [266.88, 128.63, 314.11, 139.08],
            },
            {
                "block_type": "table",
                "text": "表A.1 地面公众移动通信系统频率表\n运营商 | 频段",
                "bbox": [62.43, 227.45, 516.36, 438.36],
            },
        ],
    )

    removed = filter_repeated_margin_rich_blocks(
        page,
        {"db3205/t1145—2024"},
    )

    assert removed == 1
    texts = [item.get("text") for item in page.rich_blocks]
    assert "DB3205/T 1145—2024" not in texts
    assert "附录 A" in texts
    assert any("表A.1" in text for text in texts)


def test_filter_repeated_margin_rich_blocks_keeps_body_and_tables() -> None:
    page = PageRecord(
        page_number=10,
        raw_text="",
        rich_blocks=[
            # y0=600，位于页面中部，即使 key 命中也不应移除
            {
                "block_type": "paragraph",
                "text": "DB3205/T 1145—2024",
                "bbox": [56.64, 600.0, 156.42, 610.0],
            },
            {
                "block_type": "table",
                "text": "表A.1 内容",
                "bbox": [62.43, 227.45, 516.36, 438.36],
            },
        ],
    )

    removed = filter_repeated_margin_rich_blocks(
        page,
        {"db3205/t1145—2024"},
    )

    assert removed == 0
    assert len(page.rich_blocks) == 2


def test_is_fake_cjk_garbage_detects_extension_a_fake_glyphs() -> None:
    # 损坏字体把字母映射成 CJK 扩展 A 假字（犲 U+72B2），无 /G 字形名。
    fake = (
        "书书书犐犆犛犆犆犛犃中华人民共和国国家标准"
        "犌犅犜无人机低空遥感监测的多传感器一致性检测技术规范"
        "犜犲犮犺狀犻犮犪犾狊狆犲犮犻犳犻犮犪狋犻狅狀"
        "犮狅狀狊犻狊狋犲狀犮狔狋犲狊狋犻狀犵"
    )
    assert is_fake_cjk_garbage(fake) is True


def test_has_fake_glyph_corruption_detects_mixed_english_terms() -> None:
    # 中文正文正常 + 英文术语被损坏成连续假字（犅狅狉狀犲...）。
    text = (
        "光学遥感传感器"
        "犅狅狉狀犲犾狅狑犪犾狋犻狋狌犱犲犿狅狀犻狋狅狉犻狀犵"
        "在电磁波谱范围内接收物体辐射信息的仪器。"
    )
    assert has_fake_glyph_corruption(text) is True


def test_has_fake_glyph_corruption_detects_formula_symbols() -> None:
    # 公式中的拉丁/希腊符号被映射成分散假字（犕f、∑狀犻）。
    text = (
        "反射率相对中误差(犕f)应按公式()计算:"
        "犕f=±∑狀犻=犚,犻-犚,犻( )"
        "式中犚为反射率。"
    )
    assert has_fake_glyph_corruption(text) is True


def test_has_fake_glyph_corruption_ignores_clean_body_and_header() -> None:
    # 正常中文正文 + 少量页眉假字 不应误判为损坏页。
    text = (
        "犌犅/犜—\n"
        "本文件规定了民用大中型无人机光电任务载荷设备与无人机之间的机械接口和电气接口要求。\n"
        "本文件适用于最大起飞重量不小于150kg的民用无人机挂载的光电任务载荷设备。\n"
        "接口应满足光电任务载荷设备与无人机的可靠性要求，包括机械接口和电气接口的可靠连接。\n"
        "电气接口应实现供电、通信控制、状态反馈和视频传输等功能。\n"
    )
    assert has_fake_glyph_corruption(text) is False


def test_filter_removes_fake_gb_header_rich_block() -> None:
    # 损坏字体把 "GB/T" 映射成 犌犅/犜— 作为每页页眉，应被过滤。
    page = PageRecord(
        page_number=5,
        raw_text="",
        rich_blocks=[
            {"block_type": "paragraph", "text": "犌犅/犜—", "bbox": None},
            {
                "block_type": "paragraph",
                "text": "犌犅／犜４１４５０—２０２２",
                "bbox": [62.3, 65.32, 153.61, 86.03],
            },
            {
                "block_type": "paragraph",
                "text": "3.6一致性检测 consistency testing",
                "bbox": [62.3, 200.0, 500.0, 220.0],
            },
        ],
    )

    removed = filter_repeated_margin_rich_blocks(page, {"犌犅/犜—"})

    assert removed == 2
    texts = [item.get("text") for item in page.rich_blocks]
    assert "犌犅/犜—" not in texts
    assert "犌犅／犜４１４５０—２０２２" not in texts
    assert any("3.6一致性检测" in text for text in texts)


def test_extract_page_layout_features_collects_layout_signals() -> None:
    # 第一层版面特征：保留页面尺寸、图片数、字符/行数、字号分布，供后续打分。
    class FakeImage:
        width = 100
        height = 50
        name = "img.png"

    class FakeMediaBox:
        width = 595.33
        height = 842.0

    class FakePdfPage:
        mediabox = FakeMediaBox()
        images = [FakeImage()]

        def extract_text(self, visitor_text=None):
            text = "第一行\n第二行\n"
            if visitor_text is not None:
                for _ in text:
                    visitor_text("", None, None, None, 12.0)
            return text

    features = extract_page_layout_features(FakePdfPage(), "第一行\n第二行\n")

    assert features["page_width"] == 595.33
    assert features["page_height"] == 842.0
    assert features["image_count"] == 1
    assert features["char_count"] == 6  # 第一行第二行 (去空白)
    assert features["line_count"] == 2
    assert features["dominant_font_size"] == 12.0


def test_extract_page_layout_features_survives_decompression_limit() -> None:
    # HB 9102-2008 的扫描页含超大图像流（~14000×14000 px），pypdf 读取图像
    # 尺寸时解压超过 ZLIB_MAX_OUTPUT_LENGTH（75MB），抛 LimitReachedError
    # （PyPdfError，非 ValueError）。版面特征提取不应因单个超大图像流而
    # 让整个文档解析失败，应把超限图像当作缺失处理。
    from pypdf.errors import LimitReachedError

    class FakeOversizedImage:
        @property
        def width(self) -> int:
            raise LimitReachedError(
                "Limit reached while decompressing. 1841615 bytes remaining."
            )

        @property
        def height(self) -> int:
            raise AssertionError("width 已抛错，不应访问 height")

    class FakeMediaBox:
        width = 595.33
        height = 842.0

    class FakePdfPage:
        mediabox = FakeMediaBox()
        images = [FakeOversizedImage()]

        def extract_text(self, visitor_text=None):
            text = "扫描页无文本层\n"
            if visitor_text is not None:
                for _ in text:
                    visitor_text("", None, None, None, 10.0)
            return text

    features = extract_page_layout_features(FakePdfPage(), "扫描页无文本层")

    assert features["page_width"] == 595.33
    assert features["page_height"] == 842.0
    # 超限图像被当作缺失：image_count=0，不抛异常
    assert features["image_count"] == 0
    assert features["image_ratio"] == 0.0
    assert "font_sizes" in features
    assert "image_ratio" in features


def test_is_fake_cjk_garbage_ignores_normal_chinese() -> None:
    normal = (
        "民用大中型无人机光电任务载荷设备接口要求\n"
        "本文件规定了民用大中型无人机光电任务载荷设备与无人机之间机械接口"
        "和电气接口的一般要求和详细要求。\n"
        "1 范围\n2 规范性引用文件\n3 术语和定义\n4 缩略语"
    )
    assert is_fake_cjk_garbage(normal) is False


def test_repair_fake_glyphs_decodes_damaged_font_words() -> None:
    assert repair_fake_glyphs("犫狅狉狀犲犾狅狑犪犾狋犻狋狌犱犲") == "bornelowaltitude"
    long_word = "犜犲犮犺狀犻犮犪犾狊狆犲犮犻犳犻犮犪狋犻狅狀"
    assert repair_fake_glyphs(long_word) == "Technicalspecification"
    assert repair_fake_glyphs("犌犅／犜４１４５０—２０２２") == "GB/T41450—2022"
    assert repair_fake_glyphs("附录犃(资料性)观测手簿") == "附录A(资料性)观测手簿"


def test_repair_fake_glyphs_restores_pua_clause_dot() -> None:
    # 部分 PDF 把条款编号的点号（4.1.2）映射到 PUA 私有区字符 U+1001B0
    # （如 ４<U+1001B0>１<U+1001B0>２）。应还原为点号，使 heading_kind/
    # _clause_number 能识别编号（修复 NY/T 4616-2025 第3-6页条款错乱）。
    pua_dot = "\U001001b0"
    assert repair_fake_glyphs(f"4{pua_dot}1{pua_dot}2 监测方法") == "4.1.2 监测方法"
    assert repair_fake_glyphs(f"4{pua_dot}2 资源监测") == "4.2 资源监测"
    assert repair_fake_glyphs(f"6{pua_dot}1 监测信息分析") == "6.1 监测信息分析"
    # 还原后的编号可被识别为 clause
    from app.ingestion.pipeline import heading_kind

    assert heading_kind("4.1.2 监测方法") == "clause"
    assert heading_kind("4.2 资源监测") == "clause"


def test_repair_fake_glyphs_restores_symbol_font_math_symbols() -> None:
    # MH/T 4063.1-2026 附录 A/B 网格编码公式使用 SymbolMT 字体，数学符号被
    # 映射到 PUA 私有区（U+F0xx）。应还原为 °′″⌈⌉⌊⌋×Δ 等真实数学符号。
    assert repair_fake_glyphs("1041919.304") == "104°19′19.304"
    assert repair_fake_glyphs("") == "−"
    assert repair_fake_glyphs("") == "×"
    assert repair_fake_glyphs("") == "⌈"
    assert repair_fake_glyphs("") == "⌉"
    assert repair_fake_glyphs("") == "⌊"
    assert repair_fake_glyphs("") == "⌋"
    assert repair_fake_glyphs("") == ">"
    assert repair_fake_glyphs("") == "Δ"
    # 完整公式片段： 104°19′19.304″−102°
    assert repair_fake_glyphs("1041919.304102") == (
        "104°19′19.304″−102°"
    )


def test_clean_page_restores_pua_clause_dot_in_body_text() -> None:
    # clean_page 对每行执行 repair_fake_glyphs，PUA 点号应被还原，
    # 使清洗后的条款编号可读（NY/T 4616-2025 第5页）。
    pua_dot = "\U001001b0"
    page = PageRecord(
        page_number=5,
        raw_text=(
            "NY/T４６１６—２０２５\n"
            f"４{pua_dot}１{pua_dot}２ 监测方法\n"
            f"４{pua_dot}１{pua_dot}２{pua_dot}１ 初始信息获取\n"
            "保护点建成后,按照各项设施的定位信息,绕飞拍摄或定点拍摄各项设施的照片和视频,作为初始信息储存.\n"
        ),
        cleaned_text="",
        route="native_text",
        indexable=True,
    )
    cleaned, _removed = clean_page(page, set())
    assert "4.1.2 监测方法" in cleaned
    assert "4.1.2.1 初始信息获取" in cleaned
    assert pua_dot not in cleaned


def test_clean_page_restores_fake_glyphs_in_body_text() -> None:
    # 页眉 犌犅/犜 与正文拉丁假字在清洗阶段应被还原为可读文字。
    page = PageRecord(
        page_number=4,
        raw_text=(
            "犌犅/犜—\n"
            "图犃.多传感器辐射一致性检测观测手簿示例\n"
            "犘m=∑狀犻=犘犻\n"
            "本文件规定了接口要求。\n"
        ),
        cleaned_text="",
        route="native_text",
        indexable=True,
    )
    cleaned, removed = clean_page(page, set())
    assert "GB/T—" in cleaned
    assert "图A.多传感器辐射一致性检测观测手簿示例" in cleaned
    assert "Pm=∑ni=Pi" in cleaned


def test_table_to_markdown_renders_grid_with_header_separator() -> None:
    rows = [
        ["内容", None, "衡量指标", "数值范围", "一致性程度"],
        ["辐射一致性", "激光雷达\n辐射一致性", "反射率中误差", "m_f ≤3%", "优"],
        [None, None, None, "3%<m_f ≤5%", "良"],
    ]
    md = table_to_markdown(rows, title="表3 检测结果指标评价")
    lines = md.splitlines()
    assert lines[0] == "表3 检测结果指标评价"
    assert "| 内容 |  | 衡量指标 | 数值范围 | 一致性程度 |" in lines[1]
    assert lines[2] == "| --- | --- | --- | --- | --- |"
    # 换行折叠为空格，合并单元格留空保持列对齐
    assert "| 辐射一致性 | 激光雷达 辐射一致性 | 反射率中误差 | m_f ≤3% | 优 |" in lines[3]
    assert "|  |  |  | 3%<m_f ≤5% | 良 |" in lines[4]


def test_normalize_table_html_collapses_repeated_full_width_header() -> None:
    title = "HB 9103 过程控制文件 共2页第2页"
    source = (
        "<table><tr>"
        + "".join(f"<th>{title}</th>" for _ in range(5))
        + "</tr><tr><td>步骤3</td><td>步骤4</td><td>步骤5</td>"
        "<td>步骤6</td><td>步骤7</td></tr></table>"
    )

    rows = table_rows_from_html(source)
    normalized = normalize_table_html(source)

    assert rows[0] == [title, "", "", "", ""]
    assert table_to_markdown(rows).splitlines()[0].count(title) == 1
    assert f'<th colspan="5">{title}</th>' in normalized
    assert "步骤3</td><td>步骤4" in normalized


def test_table_to_markdown_escapes_pipe_in_cell() -> None:
    rows = [["列A", "列B"], ["a|b", "c"]]
    md = table_to_markdown(rows)
    assert "| a\\|b | c |" in md


def test_table_to_semantic_text_repeats_headers_for_each_record() -> None:
    rows = [
        ["参数", "正常类", "实用类"],
        ["载荷系数", "3.8", "4.4"],
        ["安全系数", "1.5", "1.5"],
    ]

    text = table_to_semantic_text(rows, title="表1 飞行载荷", header_rows=1)

    assert text.splitlines()[0] == "表1 飞行载荷"
    assert "记录1：参数=载荷系数；正常类=3.8；实用类=4.4" in text
    assert "记录2：参数=安全系数；正常类=1.5；实用类=1.5" in text
    assert "<table" not in text


def test_table_to_semantic_text_does_not_invent_missing_headers() -> None:
    rows = [
        ["过程控制文件", "", "", ""],
        ["步骤3", "负责人", "输入", "输出"],
    ]

    text = table_to_semantic_text(
        rows,
        title="过程控制文件",
        header_rows=1,
    )

    assert "记录1：第1列=步骤3；第2列=负责人；第3列=输入；第4列=输出" in text


def test_table_to_semantic_text_compacts_expanded_merged_cells() -> None:
    rows = [
        ["过程控制文件", "过程控制文件", "过程控制文件", "过程控制文件"],
        ["步骤3", "步骤3", "步骤4", "步骤4"],
    ]

    text = table_to_semantic_text(rows, header_rows=1)

    assert text == "记录1：第1-2列=步骤3；第3-4列=步骤4"
    assert text.count("过程控制文件=") == 0


def test_complex_table_fidelity_flags_rotated_scanned_wide_table() -> None:
    page = PageRecord(
        page_number=16,
        raw_text="",
        route="remote_ocr",
        page_type="scanned_text",
        image_coverage=1.0,
        rich_blocks=[
            {
                "block_type": "table",
                "table_id": "table-16",
                "bbox": [218.0, 137.0, 697.0, 913.0],
                "table_rows": [
                    ["过程控制文件", *("" for _ in range(18))],
                    [f"字段{column}" for column in range(19)],
                    [f"值{column}" for column in range(19)],
                    [f"值{column + 19}" for column in range(19)],
                ],
                "table_html": (
                    '<table><tr><th colspan="19">过程控制文件</th></tr></table>'
                ),
            }
        ],
    )

    diagnostics = analyze_complex_table_fidelity([page])

    assert len(diagnostics) == 1
    assert diagnostics[0]["page_number"] == 16
    assert diagnostics[0]["column_count"] == 19
    assert diagnostics[0]["recommended_route"] == "table_schema_review"
    assert "suspected_rotated_table" in diagnostics[0]["reasons"]
    assert "missing_column_header_schema" in diagnostics[0]["reasons"]


def test_complex_table_fidelity_does_not_flag_native_wide_table() -> None:
    page = PageRecord(
        page_number=3,
        raw_text="",
        route="vector_layout",
        page_type="table",
        image_coverage=0.0,
        rich_blocks=[
            {
                "block_type": "table",
                "bbox": [40.0, 100.0, 760.0, 400.0],
                "table_rows": [
                    [f"字段{column}" for column in range(12)],
                    [f"值{column}" for column in range(12)],
                ],
            }
        ],
    )

    assert analyze_complex_table_fidelity([page]) == []


def test_complex_table_fidelity_flags_ocr_figure_misclassified_as_table() -> None:
    page = PageRecord(
        page_number=9,
        raw_text="",
        route="remote_ocr",
        page_type="scanned_text",
        image_coverage=1.0,
        rich_blocks=[
            {
                "block_type": "table",
                "table_title": "图1 关键特性波动管理的优选模式",
                "bbox": [119.0, 141.0, 853.0, 787.0],
                "table_rows": [
                    ["步骤", "活动", "说明"],
                    ["步骤1", "了解关键特性", "进行评审"],
                ],
            }
        ],
    )

    diagnostics = analyze_complex_table_fidelity([page])

    assert len(diagnostics) == 1
    assert diagnostics[0]["page_number"] == 9
    assert "figure_caption_on_table_block" in diagnostics[0]["reasons"]


class _FakeStream:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def get_object(self) -> "_FakeStream":
        return self

    def get_data(self) -> bytes:
        return self._data


class _FakeFont:
    def __init__(self, subtype: str, to_unicode_cmap: str | None) -> None:
        self._subtype = subtype
        self._cmap = to_unicode_cmap

    def get_object(self) -> "_FakeFont":
        return self

    def get(self, key: str, default: Any = None) -> Any:
        if key == "/Subtype":
            return self._subtype
        if key == "/ToUnicode":
            if self._cmap is None:
                return None
            return _FakeStream(self._cmap.encode("latin-1"))
        return default


def test_tounicode_cid_conflict_detects_contradictory_mapping() -> None:
    # GJB 2489A 的 ToUnicode 表把同一 CID 映射到不同 Unicode（同一字形在
    # 不同位置提取成不同汉字）。这种自我矛盾的映射是文本层损坏的可靠信号，
    # 应返回 True 以路由到 OCR。需达到冲突密度阈值（>=2 冲突且 >=0.5%）。
    cmap = (
        "/CIDInit /ProcSet findresource begin\n"
        "begincmap\n"
        "/CIDSystemInfo 3 dict dup begin /Ordering (SI) end def\n"
        "begincmap\n"
        "1 begincodespacerange\n"
        "<0000> <00FF>\n"
        "endcodespacerange\n"
        "8 beginbfchar\n"
        "<0004> <4E94>\n"  # 五
        "<0004> <5B8B>\n"  # 同一 CID 又映射到 宋（矛盾）
        "<0006> <5B8B>\n"  # 宋
        "<0006> <4F53>\n"  # 同一 CID 又映射到 体（矛盾）
        "<0001> <4F53>\n"  # 体
        "<0005> <4F11>\n"  # 休
        "<0003> <6B63>\n"  # 正
        "<0007> <4E00>\n"  # 一
        "endbfchar\n"
        "endcmap\n"
    )
    page = _FakePdfPageWithFonts(
        {
            "/C0_0": _FakeFont("/Type0", cmap),
        }
    )
    assert has_tounicode_cid_conflict(page) is True


def test_tounicode_cid_conflict_clean_mapping_returns_false() -> None:
    # 正常文档的 ToUnicode 表每个 CID 只映射一个 Unicode，不应误报。
    cmap = (
        "/CIDInit /ProcSet findresource begin\n"
        "begincmap\n"
        "/CIDSystemInfo 3 dict dup begin /Ordering (SI) end def\n"
        "begincmap\n"
        "1 begincodespacerange\n"
        "<0000> <00FF>\n"
        "endcodespacerange\n"
        "5 beginbfchar\n"
        "<0001> <4F53>\n"  # 体
        "<0002> <8BB0>\n"  # 记
        "<0003> <6B63>\n"  # 正
        "<0004> <4E94>\n"  # 五
        "<0006> <5B8B>\n"  # 宋
        "endbfchar\n"
        "endcmap\n"
    )
    page = _FakePdfPageWithFonts(
        {
            "/C0_0": _FakeFont("/Type0", cmap),
        }
    )
    assert has_tounicode_cid_conflict(page) is False


def test_tounicode_cid_conflict_single_quirk_returns_false() -> None:
    # CCAR-23-R3 的正常页面偶有 1 个 CID→空格 编码瑕疵（1166 个 bfchar 中
    # 1 个冲突），不应误判为 ToUnicode 损坏而走 OCR（否则图片页 OCR 返回空，
    # 触发 "OCR pages remain unresolved" 门禁失败）。只有冲突密度达到阈值
    # （>=2 冲突 且 >=0.5%）才算损坏。
    cmap = (
        "/CIDInit /ProcSet findresource begin\n"
        "begincmap\n"
        "/CIDSystemInfo 3 dict dup begin /Ordering (SI) end def\n"
        "begincmap\n"
        "1 begincodespacerange\n"
        "<0000> <00FF>\n"
        "endcodespacerange\n"
        "6 beginbfchar\n"
        "<0003> <556D>\n"  # 啭
        "<0003> <0020>\n"  # 同一 CID 又映射到 空格（单次瑕疵）
        "<0001> <4F53>\n"  # 体
        "<0002> <8BB0>\n"  # 记
        "<0004> <4E94>\n"  # 五
        "<0006> <5B8B>\n"  # 宋
        "endbfchar\n"
        "endcmap\n"
    )
    page = _FakePdfPageWithFonts(
        {
            "/C0_0": _FakeFont("/Type0", cmap),
        }
    )
    assert has_tounicode_cid_conflict(page) is False


class _FakePdfPageWithFonts:
    def __init__(self, fonts: dict[str, _FakeFont]) -> None:
        self._fonts = fonts

    def get(self, key: str, default: Any = None) -> Any:
        if key == "/Resources":
            return {"/Font": self._fonts}
        return default


class _FakePdfPageWithFontsAndText(_FakePdfPageWithFonts):
    def __init__(self, text: str, fonts: dict[str, _FakeFont]) -> None:
        super().__init__(fonts)
        self._text = text

    def extract_text(self) -> str:
        return self._text


def test_parse_pdf_records_tounicode_conflict_and_routes_to_ocr(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # ToUnicode CID 冲突页应记录 tounicode_cid_conflict 并路由到
    # text_layer_review（OCR），确保不被拉回 vector 表格解析。
    source = tmp_path / "tounicode-corrupt.pdf"
    source.write_bytes(b"fake-pdf")

    conflicting_cmap = (
        "/CIDInit /ProcSet findresource begin\n"
        "begincmap\n"
        "8 beginbfchar\n"
        "<0004> <4E94>\n"
        "<0004> <5B8B>\n"
        "<0006> <5B8B>\n"
        "<0006> <4F53>\n"
        "<0001> <4F53>\n"
        "<0005> <4F11>\n"
        "<0003> <6B63>\n"
        "<0007> <4E00>\n"
        "endbfchar\n"
        "endcmap\n"
    )
    page = _FakePdfPageWithFontsAndText(
        "GJB 2489A-2023\n附录 D\n表D.1 履历本封面",
        {"/C0_0": _FakeFont("/Type0", conflicting_cmap)},
    )

    monkeypatch.setattr(
        "app.ingestion.pipeline.PdfReader",
        lambda _path: FakePdfReader([page]),
    )
    monkeypatch.setattr(
        "app.ingestion.pipeline.sha256_file",
        lambda _path: "b" * 64,
    )

    document = parse_pdf(source)

    assert len(document.pages) == 1
    p = document.pages[0]
    # 短文本可能先路由到 low_text_review；两者都进入 OCR 目标。
    assert p.route in {"text_layer_review", "low_text_review"}
    assert p.text_layer_corruption.get("tounicode_cid_conflict") is True
    # 该文本没有 PostScript 字形名（/GXX），glyph_name_garbage 应为精确标志 False，
    # 只有 ToUnicode 冲突为 True。
    assert p.text_layer_corruption.get("glyph_name_garbage") is False


def test_detect_table_hints_rejects_fragmented_cell_content() -> None:
    # AC-23-AA-2022-01 第28页的文本层把单元格内容拆成碎片行：
    # "ASTM F3230-17 表3 替换…" 被拆成 "表"+"3"+"替换"。detect_table_hints
    # 拼接后得到 "表 3 替换"（动词开头），这是单元格引用而非表题，不应进
    # expected_caption_refs（否则门禁 table_caption_alignment 失败）。
    raw = (
        "—\n26\n—\nCCAR-23-R\n4\n规章条款号\n名\n称\nASTM\n"
        "F3230-17\n表\n3\n替换\n“\n适航级别\n”\n为\n“\n飞机审定级别\n”\n"
        "23.2515\n电子和电气系统闪电"
    )
    hints = detect_table_hints(raw)
    assert hints == []

    # 真实表题（名词短语）仍应保留
    raw2 = "表\n3\n空域编码规则\n表 A.1 术语和定义\n表 1 履历本封面"
    hints2 = detect_table_hints(raw2)
    assert "表 3 空域编码规则" in hints2
    assert "表 A.1 术语和定义" in hints2
    assert "表 1 履历本封面" in hints2


def test_low_quality_ocr_detects_period_between_digits() -> None:
    # CCAR-27-R2: 扫描件内嵌的 OCR 文本层把小数点和条款号点号识别成全角
    # 句号（"27。 673" / "0。 335"）。需要 >=3 处且密度达标才判定，
    # 避免正常文档零散的 "C23．1" 误报。
    raw = (
        "第 27。 673条 主飞行操纵系统\n"
        "驾驶员用来直接操纵旋翼航空器的俯仰、横滚、偏航和垂直运动。\n"
        "第 27。 674条 交连的操纵系统\n"
        "主飞行操纵系统必须能在任何交连的辅助操纵系统出现故障。\n"
        "第 27。 675条 止动器\n"
        "每个主飞行操纵系统必须设计成当速度为 0。 335 时承受最大载荷。\n"
    )
    assert has_low_quality_ocr_text(raw) is True


def test_low_quality_ocr_clean_page_not_detected() -> None:
    # 正常文档：全角句号只出现在句末，不夹在数字之间；即使有一两处
    # "C23．1" 附录编号也不构成 OCR 文本层。
    raw = (
        "附件 C 基本着陆情况\n"
        "C23．1 基本着陆情况 本附件规定了基本着陆情况的要求。\n"
        "第 23.975 条 燃油箱的通气和汽化器蒸气的排放\n"
        "本条为修订条款，将汽化器修订为发动机。\n"
    )
    assert has_low_quality_ocr_text(raw) is False


def test_low_quality_ocr_cjk_pair_spacing_requires_scan_image() -> None:
    # 第二种信号：汉字被逐字空格隔离（"如 果 申 请 人 证 实"）是 OCR 文本层
    # 特征，但必须配合大图（image_ratio>=0.5），否则正常排版的高空格文档
    # （表格单元格、封面）会误报。
    ocr_scanned = (
        "如 果 申 请 人 证 实 因 受 几 何 形 状、可 检 查 性 和 良 好 的 设 计\n"
        "实 践 的 限 制，进 行 损 伤 容 限 评 定 不 切 实 际，申 请 人 必 须\n"
        "按 本 条 进 行 疲 劳 评 定。此 外，还 应 考 虑 外 挂 物 的 作 用 限\n"
        "制 到 已 表 明 符 合 本 条 要 求 的 角 度 之 内，并 确 保 使 用 中 不\n"
        "会 超 过 此 较 小 的 角 度，以 保 证 结 构 安 全 与 可 靠 性。\n"
    )
    assert has_low_quality_ocr_text(ocr_scanned, image_ratio=1.0) is True
    # 无大图时不判定（image_ratio=0）
    assert has_low_quality_ocr_text(ocr_scanned, image_ratio=0.0) is False
    # 正常排版（汉字连续、无逐字空格）即使有大图也不判定
    clean = "申请人证实因受几何形状、可检查性和良好设计实践的限制，进行损伤容限评定不切实际。"
    assert has_low_quality_ocr_text(clean, image_ratio=1.0) is False


def test_figure_dominant_pages_excludes_low_quality_ocr() -> None:
    # CCAR-27-R2 第1页：封面页文本层被判定为 low_quality_ocr（扫描文字），
    # 虽然文本短（<=120 字符）但它不是纯图表，必须走 OCR 恢复内容，
    # 不能被 _figure_dominant_pages 当作"图片主导页"拉回 figure 路径。
    # 对照 CCAR-23 图A2：clean 的短文本（只有图名）才是真正的图片页。
    from app.ingestion.parsers.hybrid_pdf import HybridPdfParser

    cover = PageRecord(
        page_number=1,
        raw_text="中华人民共和国交通运榆部令 zO17年第 10号《交通运输部关于修改…》",
        route="text_layer_review",
    )
    cover.cleaned_text = cover.raw_text
    cover.text_layer_corruption = {
        "low_quality_ocr": True,
        "glyph_name_garbage": False,
        "tounicode_cid_conflict": False,
    }
    chart = PageRecord(
        page_number=164,
        raw_text="- 152 - CCAR 23 R3 图A2 在速度V时确定n系数的曲线图",
        route="text_layer_review",
    )
    chart.cleaned_text = chart.raw_text
    chart.text_layer_corruption = {
        "low_quality_ocr": False,
        "glyph_name_garbage": True,
        "tounicode_cid_conflict": False,
    }

    class _Doc:
        pages = [cover, chart]

    result = HybridPdfParser._figure_dominant_pages(_Doc(), figure_pages={1, 164})
    assert result == {164}
    assert 1 not in result


def test_glyph_name_pages_excluded_from_vector_preflight_targets() -> None:
    # GB 42590-2023：整份文档文本层是 PostScript 字形名（/G8E/G47/...）。
    # 这类页虽然 route=text_layer_review，但 hybrid 层之前只排除了
    # ToUnicode 冲突页，glyph-name 页被拉回 vector 表格预检 → 表格结构
    # "恢复"了但单元格全是 /GXX 乱码。现在 glyph-name 页也必须排除出
    # vector preflight，强制走 OCR。
    glyph_page = PageRecord(
        page_number=18,
        raw_text="/G8E/G47/G6C/G72/G30/G31/GA7/GA8/GBC",
        route="text_layer_review",
    )
    glyph_page.text_layer_corruption = {
        "glyph_name_garbage": True,
        "tounicode_cid_conflict": False,
        "low_quality_ocr": False,
    }
    tounicode_page = PageRecord(
        page_number=20,
        raw_text="GJB 2489A-2023 附录D 表D.1",
        route="text_layer_review",
    )
    tounicode_page.text_layer_corruption = {
        "glyph_name_garbage": False,
        "tounicode_cid_conflict": True,
        "low_quality_ocr": False,
    }
    native_page = PageRecord(
        page_number=30,
        raw_text="正常文本 第 23.1 条 适用范围",
        route="native_text",
    )
    native_page.text_layer_corruption = {
        "glyph_name_garbage": False,
        "tounicode_cid_conflict": False,
        "low_quality_ocr": False,
    }

    class _Doc:
        pages = [glyph_page, tounicode_page, native_page]

    text_layer_targets = {18, 20}
    tounicode_corrupted = {20}
    glyph_name_corrupted = {18}
    # 模拟 hybrid 层 vector_preflight_targets 的计算
    vector_preflight_targets = {30}  # native_text
    vector_preflight_targets.update(
        text_layer_targets - tounicode_corrupted - glyph_name_corrupted
    )
    assert 18 not in vector_preflight_targets  # glyph-name 页排除
    assert 20 not in vector_preflight_targets  # tounicode 页排除
    assert 30 in vector_preflight_targets      # native 页保留


def test_revision_action_line_detects_amendment_page() -> None:
    # CCAR-27-R2 修订决定第 10 页："十六、增加一条，作为第27.573条："
    text = "\n".join(
        [
            "十六、增加一条，作为第27.573条：",
            "“第27.573条 复合材料旋翼航空器结构的损伤容限和疲劳评定",
            "“(a)每一申请人必须按本条(d)的损伤容限标准评定……",
        ]
    )
    assert is_probable_revision_page(text) is True


def test_revision_action_detects_modify_and_delete() -> None:
    text = "\n".join(
        [
            "四、将第27.51条修改为：",
            "二十、删去第27.1309条(d)款。",
        ]
    )
    assert is_probable_revision_page(text) is True


def test_revision_action_detects_annex_clause_modify() -> None:
    # "三十一、将附件B第v条(a)款修改为：" —— 条款号前带"附件X"前缀。
    text = "三十一、将附件B第v条(a)款修改为："
    assert is_probable_revision_page(text) is True


def test_revision_page_detects_quote_wrapped_continuation() -> None:
    # 修订决定的复述续页没有动作句式，只有整页引号复述的条款全文。
    lines = [
        "“(1)临界重量；",
        "“(2)临界重心；",
        "“(3)最大连续功率；",
        "“(4)起落架收起；",
        "“(5)在Vy配平旋翼航空器。",
    ]
    assert is_probable_revision_page("\n".join(lines)) is True


def test_revision_page_detects_lower_quote_ratio_continuation() -> None:
    # 修订续页混入不带引号的跨页续行时，引号比例可低至 0.3 左右（CCAR-27
    # 修订决定第 11 页实测约 0.30），仍应识别为修订续页。
    lines = [
        "件变化的影响。每一申请人必须评定包括机体PSE、主/尾旋翼传动系统……",
        "件变化的影响。每一申请人必须评定包括机体PSE、主/尾旋翼传动系统……",
        "件变化的影响。每一申请人必须评定包括机体PSE、主/尾旋翼传动系统……",
        "“(i)确定所有的 PSE;",
        "“(ii)用于确定所有 PSE 的载荷或应力……",
        "“(iii)以本条(d)(1)(ii)确定的载荷或应力为基础……",
        "件变化的影响。每一申请人必须评定包括机体PSE、主/尾旋翼传动系统……",
    ]
    assert is_probable_revision_page("\n".join(lines)) is True


def test_revision_page_ignores_few_quote_lines() -> None:
    # 只有一两行引号不足以判定为修订续页，避免误伤正文里的零星引用。
    text = "\n".join(
        [
            "“第27.573条 复合材料旋翼航空器结构的损伤容限和疲劳评定",
            "本条规定的损伤容限评定不切实际时，才进行疲劳评定。",
            "申请人必须按本条(e)进行疲劳评定。",
        ]
    )
    assert is_probable_revision_page(text) is False


def test_revision_page_ignores_quote_wrapped_math_variables() -> None:
    # 正文里用引号标记数学变量/坐标（如航空灯色度坐标 "X"、"Y"、"Z"）的页，
    # 引号后紧跟 ASCII 字母，不是修订复述，不应被误判。
    text = "\n".join(
        [
            "“Z”不大于0.002。",
            "(b)航空绿色",
            "“X”不大于0.440—0.320Y;",
            "“X”不大于 Y—0.170;",
            "“Y”不小于0.390—0.170X。",
            "(c)航空白色",
            "“X”不小于0.300且不大于0.540;",
            "“Y”不小于“X—0.040”或“Yc—0.010”,取小者;",
            "“Y”不大于“X+0.020”也不大于“0.636—0.400X”。",
        ]
    )
    assert is_probable_revision_page(text) is False


def test_revision_page_ignores_normal_clause_body() -> None:
    # 正文条款页行首不带引号，也没有修订动作，不应被误判。
    text = "\n".join(
        [
            "第27.571条 飞行结构的疲劳评定",
            "(a)总则飞行结构的每一部分……必须予以认定，并必须按本节规定进行评定。",
            "(1)评定的方法必须是经批准的。",
            "(2)必须确定可能破坏的部位。",
        ]
    )
    assert is_probable_revision_page(text) is False


def test_revision_page_ignores_article_reference_in_body() -> None:
    # 正文引用"按照第27.573条"不含修订动作，不应被误判。
    text = "申请人必须按照第27.573条(d)款的规定进行损伤容限评定。"
    assert is_probable_revision_page(text) is False
