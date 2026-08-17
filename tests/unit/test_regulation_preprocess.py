from app.ingestion import pipeline
from app.ingestion.regulations import (
    REGULATION_MAX_CHARS,
    extract_article_references,
)


def test_normalize_line_repairs_full_width_and_chinese_spacing() -> None:
    assert pipeline.normalize_line("第 ２１．１３７ 条  质 量 系 统") == "第 21.137 条 质量系统"


def test_repair_invalid_unicode_replaces_lone_surrogates() -> None:
    repaired, count = pipeline.repair_invalid_unicode(
        "R² = \ud835 / \ud835 and valid pair \ud835\udc00"
    )

    assert repaired == "R² = � / � and valid pair 𝐀"
    assert count == 2
    assert repaired.encode("utf-8")


def test_stable_id_accepts_pdf_text_with_lone_surrogate() -> None:
    identifier = pipeline.stable_id("document", "公式 \ud835")

    assert len(identifier) == 20


def test_heading_kind_detects_regulation_levels() -> None:
    assert pipeline.heading_kind("第十七章 附则") == "chapter"
    assert pipeline.heading_kind("第 21.137 条 质量系统") == "article"
    assert pipeline.heading_kind("UHS.993 燃油系统导管和接头") == "clause"
    assert pipeline.heading_kind("23.21") is None


def test_heading_kind_detects_top_level_numbered_clause() -> None:
    assert pipeline.heading_kind("6 追溯方法") == "clause"
    assert pipeline.heading_kind("2 规范性引用文件") == "clause"
    assert pipeline.heading_kind("5 高水效小麦品种鉴定") == "clause"
    assert pipeline.heading_kind("3 个并联的卷积神经网络") is None
    assert pipeline.heading_kind("3 次重复") is None
    assert pipeline.heading_kind("80±5%") is None


def test_clean_page_removes_annex_decor_letter() -> None:
    page = pipeline.PageRecord(
        page_number=8,
        raw_text=(
            "DB 1311/T 087—2025\n"
            "A\n"
            "A\n"
            "附录 A\n"
            "（资料性）产量预测模型使用方法\n"
            "5\n"
        ),
    )
    cleaned, removed = pipeline.clean_page(page, set())
    assert "A A" not in cleaned
    assert "附录 A" in cleaned
    # 移除 2 个附录装饰字母（页首 A、A）+ 1 个页码（页尾 5）
    assert removed == 3


def test_route_rejects_scanned_document() -> None:
    assert pipeline.determine_document_route(0.0, False) == "ocr_required"
    assert pipeline.determine_document_route(0.95, True) == "hybrid_review"
    assert pipeline.determine_document_route(1.0, False, True) == "hybrid_review"
    assert pipeline.determine_document_route(1.0, True) == "layout_review"
    assert pipeline.determine_document_route(1.0, False) == "native_text"


def test_detects_table_title_split_across_pdf_text_lines() -> None:
    text = "— 36 —\n表\n2.\nCCAR-23-R4 与 CCAR-23-R3 条款对应关系表"

    assert pipeline.detect_table_hints(text) == [
        "表 2. CCAR-23-R4 与 CCAR-23-R3 条款对应关系表"
    ]


def test_detect_table_hints_filters_prose_reference() -> None:
    # "表 A.6 的要求" 是正文引用（"应符合表A.5和表A.6的要求"），不是表题。
    # 若被当作表题 hint，会 phantom 化 expected caption，导致 table_caption_
    # alignment 门禁失败（GB 46761-2025 第17页）。
    text = (
        "生产者系统民用无人机系统请求实名登记系统激活状态上报接口请求参数应符合表 A.5\n"
        "表 A.6 的要求\n"
        "表A5 上报民用无人机激活状态加密参数\n"
    )
    hints = pipeline.detect_table_hints(text)
    assert not any("表 A.6 的要求" in h for h in hints)
    assert any("表A5" in h for h in hints)

    # _semantic_table_hints 也过滤
    semantic = pipeline._semantic_table_hints(["表 A.3 查询…", "表 A.6 的要求。"])
    assert not any("表 A.6 的要求" in h for h in semantic)
    assert len(semantic) == 1


def test_clean_page_joins_wrapped_list_item() -> None:
    page = pipeline.PageRecord(
        page_number=1,
        raw_text="（一）设计资料和后续更\n改，必须准确。\n",
    )
    cleaned, removed = pipeline.clean_page(page, set())
    assert cleaned == "(一)设计资料和后续更改,必须准确。"
    assert removed == 0


def test_detects_table_of_contents_for_exclusion() -> None:
    text = "目录\n第一章 总则…………1\n第二章 适用范围…………3\n第三章 责任…………8"
    assert pipeline.is_probable_toc_page(text) is True


def test_detects_noisy_ocr_table_of_contents() -> None:
    text = (
        "A27.2 格式 … ● ¨ ● ¨ ● 216\n"
        "A27.3 内容 ¨ ¨ ¨ … … 216\n"
        "附件B适航准则 … … ¨ ● 220\n"
    )
    assert pipeline.is_probable_toc_page(text) is True


def test_table_leader_dots_are_not_mistaken_for_toc() -> None:
    # 表格单元格中的省略号填充符（后跟括号而非页码数字）不应被判为目录。
    text = (
        "7 技术指标要求。\n"
        "…………………………（\n"
        "…………………………（\n"
        "…………………………（\n"
        "1∶1000 0.15 0.15 0.2\n"
    )
    assert pipeline.is_probable_toc_page(text) is False


def test_ocr_annex_toc_entries_are_detected_as_toc() -> None:
    # OCR 修复后附录目录条目（点线 + 页码）应被识别为目录，
    # 否则会被当成 annex 标题污染后续所有章节的 section_path。
    text = (
        "附录C(资料性附录)航摄飞行记录表……18\n"
        "附录D(资料性附录)旋角计算示意图……19\n"
        "附录E(资料性附录)航摄分区示意图和航线示意图……20\n"
        "附录F(资料性附录)摄区完成情况图……22\n"
        "附录G(资料性附录)像片控制点成果表与点之记样例……24"
    )
    assert pipeline.is_probable_toc_page(text) is True


def test_toc_with_page_number_before_leader_is_detected() -> None:
    # 部分标准目录格式是"标题 页码 ………"（页码在前、点线在后），
    # 而不是标准的"标题 ……… 页码"。此类目录也应被识别，避免目录条目
    # 被当作 annex 标题污染后续章节。
    text = (
        "目  次\n"
        "前言 Ⅲ………………………………\n"
        "1 范围 1……………………………\n"
        "2 规范性引用文件 1………………………\n"
        "5 一般要求 2……………………………\n"
        "5.1 功能 2……………………………\n"
        "5.2 设备等级 2……………………………\n"
    )
    assert pipeline.is_probable_toc_page(text) is True


def test_toc_without_page_numbers_is_detected() -> None:
    # 部分目录把页码去掉，仅保留"编号 标题 ……"或"标题 ……"。
    # 此时点线在行尾、后面没有页码数字，仍应识别为目录。
    text = (
        "目次前言I………………\n"
        "范围………………\n"
        "规范性引用文件………………\n"
        "5.1检测流程………………\n"
        "5.2检测内容………………\n"
        "5.3检测仪器………………\n"
        "6.1一般规定………………\n"
        "6.2室外检测条件………………\n"
        "7.1飞行计划书………………\n"
    )
    assert pipeline.is_probable_toc_page(text) is True


def test_compute_toc_score_ranks_toc_over_body() -> None:
    # 第二层特征打分：目录页应显著高于正文页。
    toc_text = (
        "目  次\n"
        "前言 Ⅲ………………………………\n"
        "1 范围 1……………………………\n"
        "2 规范性引用文件 1………………………\n"
        "5 一般要求 2……………………………\n"
        "5.1 功能 2……………………………\n"
        "6 详细要求 3……………………………\n"
        "6.1 机械接口 3……………………………\n"
    )
    body_text = (
        "本文件规定了民用大中型无人机光电任务载荷设备与无人机之间机械接口和电气接口的一般要求。\n"
        "本文件适用于最大起飞重量不小于150kg的民用无人机挂载的光电任务载荷设备。\n"
    )
    toc_score = pipeline.compute_toc_score(
        toc_text, page_number=3, total_pages=15
    )
    body_score = pipeline.compute_toc_score(
        body_text, page_number=7, total_pages=15
    )
    assert toc_score > body_score
    assert body_score < 0.5


def test_compute_toc_score_recognizes_title_heading() -> None:
    # 页首"目 次"标题（含空格）应被识别，分数高于纯正文。
    text = (
        "目 次\n"
        "1 范围 1\n"
        "2 规范性引用文件 1\n"
        "3 术语和定义 1\n"
        "4 缩略语 1\n"
    )
    score = pipeline.compute_toc_score(text, page_number=2, total_pages=26)
    assert score > 0.3
    # TOC_TITLE_RE 应匹配含空格的 "目 次"
    assert pipeline.TOC_TITLE_RE.fullmatch("目 次") is not None


def test_normalizes_lettered_and_hierarchical_article_numbers() -> None:
    samples = {
        "第21.1条第1款的内容是什么？": "21.1(1)",
        "第21.1条1的内容是什么？": "21.1(1)",
        "第25.3条 ETOPS批准": "25.3",
        "第25.3条（ETOPS）批准": "25.3",
        "25.1309(a)(1)有什么要求？": "25.1309(A)(1)",
        "25.1309A规定了什么？": "25.1309A",
        "A25.1 内容": "A25.1",
        "UHS.993 燃油系统": "UHS.993",
    }

    for text, expected in samples.items():
        references = extract_article_references(text)
        assert references[0].normalized_id == expected


def test_regulation_chunks_use_article_hard_boundaries_and_search_aliases() -> None:
    page = pipeline.PageRecord(
        page_number=1,
        raw_text="",
        cleaned_text=(
            "第一章 总则\n"
            "第21.1条 适用范围\n"
            "本条规定适用范围。\n"
            "第21.2A条 特殊要求\n"
            "本条规定特殊要求。"
        ),
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-1", [page])
    chunks = pipeline.build_chunks("doc-1", blocks)

    assert [chunk.article_id_normalized for chunk in chunks] == ["21.1", "21.2A"]
    assert "第21.2A条" not in chunks[0].text
    assert "第21.1条" not in chunks[1].text
    assert "第21.1条" in chunks[0].article_aliases
    assert chunks[0].question_aliases
    assert all(len(chunk.text) <= REGULATION_MAX_CHARS for chunk in chunks)


def test_standalone_inline_clauses_are_not_dropped_from_chunks() -> None:
    page = pipeline.PageRecord(
        page_number=9,
        raw_text="",
        cleaned_text=(
            "6 交互协议\n"
            "6.1 监管平台交互协议\n"
            "6.1.1 无人驾驶航空器与监管平台的交互协议应符合有关规定。\n"
            "6.1.2 有人驾驶航空器与监管平台的交互协议应符合 MH/T 4052-2021 中第 6 章的规定。\n"
            "7 专用网络\n"
            "7.1 低空航空器可依托运营商通信专用网络接入监管平台。\n"
            "第21.1条 本条规定的通信设备必须持续符合接入要求。\n"
        ),
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-inline-clauses", [page])
    chunks = pipeline.build_chunks("doc-inline-clauses", blocks)
    chunk_texts = [chunk.text for chunk in chunks]

    # 条款正文现在带父章节 context 前缀（如 "6 交互协议 | 6.1 监管平台交互协议 | 6.1.1 …"）
    assert any("\n6.1.1 " in "\n" + text for text in chunk_texts)
    assert any("\n6.1.2 " in "\n" + text for text in chunk_texts)
    assert any("\n7.1 " in "\n" + text for text in chunk_texts)
    assert any(text.startswith("第21.1条 ") for text in chunk_texts)
    assert all(text != "6 交互协议" for text in chunk_texts)
    assert all(text != "6.1 监管平台交互协议" for text in chunk_texts)
    assert all(text != "7 专用网络" for text in chunk_texts)
    assert all(text.count(text.splitlines()[0]) == 1 for text in chunk_texts)
    assert pipeline.find_unpublished_clause_blocks(blocks, chunks) == []


def test_top_level_clause_is_parent_of_sub_clauses_in_section_path() -> None:
    # 顶层条款 "4 一般要求"（编号无点号）应作为 4.1~4.7 的父章节出现在
    # section_path 中，而不是只有当前条款一级。
    page = pipeline.PageRecord(
        page_number=7,
        raw_text="",
        cleaned_text=(
            "4 一般要求\n"
            "4.1 功能\n"
            "地面站一般具有以下功能:监控、规划与管理。\n"
            "4.2 可靠性\n"
            "地面站可靠性指标应满足用户要求。\n"
            "4.3 维修性\n"
            "地面站应按用户要求的维修体制进行设计。\n"
        ),
        indexable=True,
    )
    page8 = pipeline.PageRecord(
        page_number=8,
        raw_text="",
        cleaned_text=(
            "4.8 环境适应性\n"
            "4.8.1 低气压\n"
            "地面站应满足低气压要求。\n"
        ),
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-top-clause", [page, page8])
    chunks = pipeline.build_chunks("doc-top-clause", blocks)

    by_title = {chunk.title: chunk for chunk in chunks}
    assert list(by_title["4.1 功能"].section_path) == ["4 一般要求", "4.1 功能"]
    assert list(by_title["4.3 维修性"].section_path) == ["4 一般要求", "4.3 维修性"]
    assert list(by_title["4.8.1 低气压"].section_path) == [
        "4 一般要求",
        "4.8 环境适应性",
        "4.8.1 低气压",
    ]
    # 父章节作为 context 前缀出现在 chunk 文本中
    assert "4 一般要求" in by_title["4.1 功能"].text


def test_chunk_title_not_duplicated_when_heading_block_matches_section() -> None:
    # 当第一个 content block 是标题、且与 section_path 最后一个元素相同时，
    # chunk 正文不应再次输出该标题（避免 "B.4 像片重叠度\nB.4 像片重叠度" 重复）。
    page = pipeline.PageRecord(
        page_number=20,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": "B.4 像片重叠度",
                "bbox": [100.0, 191.0, 238.0, 206.0],
            },
            {
                "block_type": "paragraph",
                "text": "像片重叠度计算见式(B.5)：",
                "bbox": [136.0, 228.0, 352.0, 243.0],
            },
            {
                "block_type": "formula",
                "text": "p_X=p^'_X+(1-p^'_X)Δh/H",
                "bbox": [378.0, 247.0, 606.0, 265.0],
            },
        ],
    )

    blocks = pipeline.split_blocks("doc-b4", [page])
    chunks = pipeline.build_chunks("doc-b4", blocks)

    b4_chunks = [c for c in chunks if "像片重叠度" in c.text]
    assert b4_chunks
    chunk_text = b4_chunks[0].text
    assert chunk_text.count("B.4 像片重叠度") == 1
    # 标题之后紧跟正文，不重复
    lines = chunk_text.splitlines()
    assert lines.count("B.4 像片重叠度") == 1


def test_clause_chunk_coverage_detects_a_missing_inline_clause() -> None:
    page = pipeline.PageRecord(
        page_number=1,
        raw_text="",
        cleaned_text=(
            "6.1.1 第一项要求应持续满足。\n"
            "6.1.2 第二项要求不得遗漏。\n"
        ),
        indexable=True,
    )
    blocks = pipeline.split_blocks("doc-missing-clause", [page])
    chunks = pipeline.build_chunks("doc-missing-clause", blocks)

    missing = pipeline.find_unpublished_clause_blocks(blocks, chunks[:1])

    assert [block.article_id_normalized for block in missing] == ["6.1.2"]


def test_figure_blocks_become_standalone_chunks_with_own_asset() -> None:
    # 一页多图时每张图应独立成 chunk（各带自己的 asset_id），而不是合并进
    # 正文 chunk 导致发布时只发第一张图。
    page = pipeline.PageRecord(
        page_number=5,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": "5.1 本章正文要求。",
                "bbox": [50, 50, 500, 60],
            },
            {
                "block_type": "figure",
                "text": "图1 吊挂控制逻辑框图",
                "bbox": [100, 100, 400, 300],
                "asset_id": "p0005-figure-01",
            },
            {
                "block_type": "figure",
                "text": "图2 吊挂货物飞行示意图",
                "bbox": [100, 320, 400, 500],
                "asset_id": "p0005-figure-02",
            },
        ],
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-fig-chunk", [page])
    chunks = pipeline.build_chunks("doc-fig-chunk", blocks)

    figure_chunks = [chunk for chunk in chunks if chunk.content_type == "figure"]
    assert len(figure_chunks) == 2
    assert figure_chunks[0].asset_ids == ["p0005-figure-01"]
    assert figure_chunks[1].asset_ids == ["p0005-figure-02"]
    assert any("图1 吊挂" in chunk.text for chunk in figure_chunks)
    assert any("图2 吊挂" in chunk.text for chunk in figure_chunks)
    # 正文段落仍进独立 text chunk
    text_chunks = [chunk for chunk in chunks if chunk.content_type == "text"]
    assert any("本章正文要求" in chunk.text for chunk in text_chunks)


def test_formula_chunk_binds_preceding_text_and_shi_zhong_vars() -> None:
    # 公式块应与前置说明 + 后置"式中"变量段落绑在同一个 chunk，作为不可切分单元。
    page = pipeline.PageRecord(
        page_number=10,
        raw_text="",
        cleaned_text=(
            "4.1 覆盖距离\n"
            "按式（1）计算传播损耗：\n"
            "PL=32.45+20lg(d)+20lg(f)+M\n"
            "式中：\n"
            "d——距离，单位为千米（km）；\n"
            "f——频率，单位为兆赫兹（MHz）；\n"
            "M——综合修正因子，单位为分贝（dB）。\n"
        ),
        rich_blocks=[
            {
                "block_type": "formula",
                "text": "PL=32.45+20lg(d)+20lg(f)+M",
                "latex": r"PL=32.45+20\lg(d)+20\lg(f)+M",
                "asset_id": "p0010-remote-formula-01",
            }
        ],
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-formula-bind", [page])
    chunks = pipeline.build_chunks("doc-formula-bind", blocks)

    formula_chunks = [chunk for chunk in chunks if chunk.formula_latex]
    assert len(formula_chunks) >= 1
    bound = formula_chunks[0]
    # 公式 + 前置"按式（1）" + 后置"式中" 同 chunk
    assert "按式（1）计算传播损耗" in bound.text
    assert "PL=32.45+20lg(d)" in bound.text
    assert "式中" in bound.text
    assert "d——距离，单位为千米" in bound.text


def test_merges_cross_page_table_and_removes_repeated_rows() -> None:
    pages = [
        pipeline.PageRecord(
            page_number=7,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_title": "表 1 不合格过程代码（续）",
                    "table_html": (
                        "<table><tr><th>分类代码</th><th>过程代码</th>"
                        "<th>定义/描述</th></tr>"
                        "<tr><td>P2</td><td>P229</td><td>铣</td></tr>"
                        "<tr><td>P2</td><td>P230</td><td>模制</td></tr></table>"
                    ),
                    "text": "legacy html",
                    "asset_id": "p0007-table-01",
                }
            ],
        ),
        pipeline.PageRecord(
            page_number=8,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_title": "表 1 不合格过程代码（续）",
                    "table_html": (
                        "<table><tr><th>分类代码</th><th>过程代码</th>"
                        "<th>定义/描述</th></tr>"
                        "<tr><td>P2</td><td>P231</td><td>涂装</td></tr>"
                        "<tr><td>P2</td><td>P232</td><td>喷丸</td></tr></table>"
                    ),
                    "text": "legacy html",
                    "asset_id": "p0008-table-01",
                }
            ],
        ),
        pipeline.PageRecord(
            page_number=9,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_title": "表 1 不合格过程代码（续）",
                    "table_html": (
                        "<table><tr><th>分类代码</th><th>过程代码</th>"
                        "<th>定义/描述</th></tr>"
                        "<tr><td>P2</td><td>P231</td><td>涂装</td></tr>"
                        "<tr><td>P2</td><td>P232</td><td>喷丸</td></tr></table>"
                    ),
                    "text": "legacy html",
                    "asset_id": "p0009-table-01",
                }
            ],
        ),
    ]

    diagnostics = pipeline.merge_cross_page_tables(pages, "document-1")
    blocks = pipeline.split_blocks("document-1", pages)
    table_blocks = [block for block in blocks if block.block_type == "table"]

    assert diagnostics["cross_page_table_count"] == 1
    assert diagnostics["structured_table_pages"] == [7, 8, 9]
    assert diagnostics["duplicate_table_rows_removed"] == 4
    assert len(table_blocks) == 1
    assert table_blocks[0].source_page_start == 7
    assert table_blocks[0].source_page_end == 9
    assert [row[1] for row in table_blocks[0].table_rows[1:]] == [
        "P229",
        "P230",
        "P231",
        "P232",
    ]
    assert "<table" not in table_blocks[0].text.casefold()
    assert table_blocks[0].asset_ids == [
        "p0007-table-01",
        "p0008-table-01",
        "p0009-table-01",
    ]


def test_split_blocks_filters_continuation_title_residue() -> None:
    # 跨页续表合并后，续表标题"表N 续"和残留"( )"不携带表格数据，
    # 不应发布为空的噪声 chunk（GB 46750-2025 第11页"表1 数据包内容续"）。
    page = pipeline.PageRecord(
        page_number=11,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": "表1 数据包内容续",
                "bbox": [248.6, 108.1, 349.1, 119.9],
            },
            {
                "block_type": "paragraph",
                "text": "( )",
                "bbox": [334.0, 115.6, 359.4, 125.3],
            },
            {
                "block_type": "paragraph",
                "text": "5.2.2 运行识别信息数据包扩展内容应使用民航行业主管部门统一发布的协议。",
                "bbox": [68.7, 189.6, 459.5, 207.8],
            },
            {
                "block_type": "table",
                "text": "表2 数据类型与标识\n| 字节位 | 数据标识位 | ...",
                "table_title": "表2 数据类型与标识",
                "bbox": [70.9, 252.2, 532.1, 744.9],
            },
        ],
    )

    blocks = pipeline.split_blocks("doc-continuation-residue", [page])

    texts = [block.text for block in blocks]
    assert not any("表1 数据包内容续" in text for text in texts)
    assert not any(text.strip() == "( )" for text in texts)
    assert any("5.2.2 运行识别信息数据包扩展内容" in text for text in texts)
    assert any("表2 数据类型与标识" in text for text in texts)


def test_merge_accepts_single_row_multi_column_schematic() -> None:
    # 单行多列表格（数据结构示意图）不应被 merge_cross_page_tables 标记为
    # invalid（修复 GB 46750-2025 第10页 table_structure 门禁失败：
    # "Table blocks could not be converted to complete rows: [10]"）。
    page = pipeline.PageRecord(
        page_number=10,
        raw_text="",
        cleaned_text="",
        table_hints=["表1 数据包内容"],
        rich_blocks=[
            {
                # 数据结构示意图（无明确标题，1行8列），不应被绑定"表1"标题
                "block_type": "table",
                "text": "数据类型 版本号 数据长度 数据标识 …",
                "bbox": [95.1, 420.7, 492.2, 440.5],
                "table_title": "",
                "table_rows": [
                    [
                        "数据类型",
                        "版本号",
                        "数据长度",
                        "数据标识",
                        "数据内容项\n1",
                        "数据内容项\n2",
                        "…",
                        "数据内容项N",
                    ]
                ],
                "asset_id": "p0010-t01",
            },
            {
                "block_type": "table",
                "text": "表1 数据包内容\n| 数据包内容 | 长度 | 说明 | 取值 |",
                "bbox": [63.1, 522.0, 524.2, 769.9],
                "table_title": "表1 数据包内容",
                "table_rows": [
                    ["数据包内容", "长度", "说明", "取值"],
                    ["数据类型", "1byte", "数据类型定义", "255"],
                    ["版本号", "1byte", "当前发送的运行识别数据包版本号", "1~3"],
                    ["数据长度", "1byte", "数据内容项的字节数", "1~200"],
                    ["数据标识", "N 3+ byte", "是否发送该位代表的数据", "1~7"],
                ],
                "asset_id": "p0010-t02",
            },
        ],
    )

    diagnostics = pipeline.merge_cross_page_tables([page], "doc-single-row")

    assert 10 not in diagnostics["invalid_table_pages"]
    assert 10 in diagnostics["structured_table_pages"]
    # 示意图不绑定"表1"标题；真正的表1保留标题
    table_titles = [
        str(item.get("table_title") or "")
        for item in page.rich_blocks
        if item.get("block_type") == "table"
    ]
    assert table_titles.count("表1 数据包内容") == 1
    rows = [
        ["条号", "要求"],
        ["23.1", "A" * 70],
        ["23.2", "B" * 70],
        ["23.3", "C" * 70],
    ]
    block = pipeline.BlockRecord(
        block_id="table-block",
        block_type="table",
        text=pipeline.table_to_text(rows, title="表 2 要求"),
        page_number=10,
        section_path=[],
        source_page_start=10,
        source_page_end=12,
        table_id="table-2",
        table_title="表 2 要求",
        table_rows=rows,
        table_header_rows=1,
        table_row_pages=[10, 10, 11, 12],
        table_html=pipeline.table_to_html(rows),
    )

    parts = list(pipeline.split_structured_table_block(block, 110))

    assert len(parts) == 3
    assert all(part.table_rows[0] == ["条号", "要求"] for part in parts)
    assert [part.table_rows[1][0] for part in parts] == ["23.1", "23.2", "23.3"]
    assert [(part.source_page_start, part.source_page_end) for part in parts] == [
        (10, 10),
        (11, 11),
        (12, 12),
    ]
    assert all("<table>" in (part.table_html or "") for part in parts)
    assert all("<td" not in part.text.casefold() for part in parts)


def test_splits_multiple_page_tables_and_merges_named_continuation() -> None:
    pages = [
        pipeline.PageRecord(
            page_number=258,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_title": "矢量网格检测到结构化表格",
                    "table_rows": [
                        ["表 1 温度分布", "", "", "", ""],
                        ["参数", "地面", "巡航", "闪点", ""],
                        ["平均温度", "59.95", "-70", "120", ""],
                        ["(b) 在分析中必须使用表 2 定义的航段距离分布。", "", "", "", ""],
                        ["表 2 航段距离分布", "", "", "", ""],
                        ["自", "至", "1000", "2000", ""],
                        ["0", "200", "11.7", "7.5", ""],
                        ["200", "400", "27.3", "19.9", ""],
                    ],
                    "asset_id": "p0258-table-01",
                }
            ],
        ),
        pipeline.PageRecord(
            page_number=259,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_rows": [
                        ["表 2 航段距离分布", "", "", ""],
                        ["自", "至", "1000", "2000"],
                        ["5000", "5200", "0.0", "0.8"],
                    ],
                    "asset_id": "p0259-table-01",
                }
            ],
        ),
    ]

    diagnostics = pipeline.merge_cross_page_tables(pages, "document-2")
    blocks = pipeline.split_blocks("document-2", pages)
    tables = [block for block in blocks if block.block_type == "table"]
    notes = [block for block in blocks if block.block_type == "paragraph"]

    assert diagnostics["structured_table_count"] == 2
    assert diagnostics["cross_page_table_count"] == 1
    assert [table.table_title for table in tables] == [
        "表 1 温度分布",
        "表 2 航段距离分布",
    ]
    assert tables[1].source_page_start == 258
    assert tables[1].source_page_end == 259
    assert tables[1].table_rows[0] == ["自", "至", "1000", "2000"]
    assert ["0", "200", "11.7", "7.5"] in tables[1].table_rows
    assert ["5000", "5200", "0.0", "0.8"] in tables[1].table_rows
    assert any("必须使用表 2" in block.text for block in notes)


def test_keeps_adjacent_tables_with_different_titles_and_same_header_separate() -> None:
    # 表5（第12页）与表6（第13页）标题不同、但表头相同（序号|数据项|评定标准）。
    # 它们不是续表，不应因 same_header 而合并。
    pages = [
        pipeline.PageRecord(
            page_number=12,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_title": "表5 数据结构测试结果评定",
                    "table_rows": [
                        ["序号", "数据项", "评定标准"],
                        ["1", "唯一识别码", "完全符合4.1为合格"],
                        ["2", "地理围栏空域类型", "完全符合4.2为合格"],
                        ["3", "几何表达规则", "完全符合4.3为合格"],
                        ["4", "时空基准", "完全符合4.4为合格"],
                        ["5", "数据格式", "完全符合4.5为合格"],
                        ["6", "数据结构说明", "完全符合4.6为合格"],
                    ],
                    "asset_id": "p12-table-05",
                }
            ],
        ),
        pipeline.PageRecord(
            page_number=13,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_title": "表6 数据录入数据更新及数据校验测试结果评定",
                    "table_rows": [
                        ["序号", "数据项", "评定标准"],
                        ["1", "数据录入", "完全符合5.1为合格"],
                        ["2", "数据更新", "完全符合5.4为合格"],
                        ["3", "数据校验", "MD5一致为合格"],
                    ],
                    "asset_id": "p13-table-06",
                }
            ],
        ),
    ]

    diagnostics = pipeline.merge_cross_page_tables(pages, "document-3")
    blocks = pipeline.split_blocks("document-3", pages)
    tables = [block for block in blocks if block.block_type == "table"]

    assert diagnostics["cross_page_table_count"] == 0
    assert [table.table_title for table in tables] == [
        "表5 数据结构测试结果评定",
        "表6 数据录入数据更新及数据校验测试结果评定",
    ]
    assert len(tables) == 2
    assert tables[0].source_page_end == 12
    assert tables[1].source_page_start == 13


def test_page_table_hints_are_consumed_once_in_visual_order() -> None:
    page = pipeline.PageRecord(
        page_number=7,
        raw_text="",
        cleaned_text="",
        table_hints=[
            "表1 点云密度要求",
            "表2 点云高程精度要求",
            "表3 数字正射影像图地面分辨率要求",
        ],
        rich_blocks=[
            {
                "block_type": "table",
                "table_rows": [["比例尺", "密度"], ["1:500", "16"]],
                "bbox": [10, 100, 90, 150],
            },
            {
                "block_type": "table",
                "table_rows": [["比例尺", "平地"], ["1:500", "0.15"]],
                "bbox": [10, 200, 90, 250],
            },
            {
                "block_type": "table",
                "table_rows": [["比例尺", "分辨率"], ["1:500", "0.05"]],
                "bbox": [10, 300, 90, 350],
            },
        ],
    )

    diagnostics = pipeline.merge_cross_page_tables([page], "lidar-guide")
    tables = [
        item for item in page.rich_blocks if item.get("block_type") == "table"
    ]

    assert [item["table_title"] for item in tables] == page.table_hints
    assert len({item["table_id"] for item in tables}) == 3
    assert diagnostics["table_caption_mismatch_pages"] == []
    assert diagnostics["duplicate_table_id_pages"] == []


def test_cross_page_table_continuation_satisfies_caption_reference() -> None:
    # 跨页续表：表 A.1 的表格体在上一页，当前页只有 table_coverage 标记 +
    # 表 A.2 的表格体。期望引用 {表a1, 表a2} 应全部满足，不判为 mismatch。
    pages = [
        pipeline.PageRecord(
            page_number=11,
            raw_text="",
            cleaned_text="",
            table_hints=["表 A.1 典型机械接口参数"],
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_rows": [["序号", "名称"], ["1", "重量"]],
                    "bbox": [10, 100, 90, 200],
                }
            ],
        ),
        pipeline.PageRecord(
            page_number=12,
            raw_text="",
            cleaned_text="",
            table_hints=["表 A.1 典型机械接口参数 (续)", "表 A.2 典型连接器安装要求"],
            rich_blocks=[
                # 跨页续表标记（表 A.1 表格体在上一页）
                {
                    "block_type": "table_coverage",
                    "text": "",
                    "source_page_start": 11,
                    "source_page_end": 12,
                    "table_id": "table-a1",
                    "table_title": "表A1 典型机械接口参数",
                },
                {
                    "block_type": "table",
                    "table_rows": [["序号", "安装要求"], ["1", "主键位"]],
                    "bbox": [10, 300, 90, 400],
                },
            ],
        ),
    ]

    diagnostics = pipeline.merge_cross_page_tables(pages, "photo-load")
    assert diagnostics["table_caption_mismatch_pages"] == []


def test_table_caption_reference_normalizes_fullwidth_digits() -> None:
    # 表题可能用全角数字（表１）而表格标题用半角（表1），引用应归一化一致，
    # 否则 table_caption_mismatch 误报。
    page = pipeline.PageRecord(
        page_number=7,
        raw_text="",
        cleaned_text="",
        table_hints=[
            "表 1 无人机低空遥感监测的多传感器一致性检测指标内容",
            "表 2 主要检测仪器及技术要求",
        ],
        rich_blocks=[
            {
                "block_type": "table",
                "table_title": "表１ 无人机低空遥感监测的多传感器一致性检测指标内容",
                "table_rows": [
                    ["一致性类型", "指标内容"],
                    ["激光雷达", "反射率中误差"],
                ],
                "bbox": [10, 100, 500, 300],
            },
            {
                "block_type": "table",
                "table_title": "表２ 主要检测仪器及技术要求",
                "table_rows": [
                    ["检测仪器名称", "技术要求"],
                    ["积分球", "不低于三级溯源"],
                ],
                "bbox": [10, 320, 500, 500],
            },
        ],
    )

    diagnostics = pipeline.merge_cross_page_tables([page], "multi-sensor")
    assert diagnostics["table_caption_mismatch_pages"] == []


def test_merged_cells_are_reported_for_visual_review() -> None:
    page = pipeline.PageRecord(
        page_number=22,
        raw_text="",
        cleaned_text="",
        table_hints=["表A.1 飞行记录表"],
        rich_blocks=[
            {
                "block_type": "table",
                "table_title": "表A.1 飞行记录表",
                "table_rows": [
                    ["项目名称", "", "测区名称", ""],
                    ["参加人员", "", "", ""],
                ],
                "table_merged_cell_count": 3,
                "bbox": [10, 100, 90, 250],
            }
        ],
    )

    diagnostics = pipeline.merge_cross_page_tables([page], "lidar-guide")

    assert diagnostics["merged_cell_review_pages"] == [22]


def test_merge_cross_page_tables_reorders_rich_blocks_by_bbox() -> None:
    page = pipeline.PageRecord(
        page_number=9,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            # 修复前: merge_cross_page_tables 会把 table block 追加到末尾,
            # 导致后面的 4.3 标题先于表 2 body 出现, 章节归属错乱。
            {
                "block_type": "table",
                "table_title": "表 3 重量限制",
                "table_rows": [
                    ["项目", "重量(kg)"],
                    ["最大起飞重量(MTOW)", "43,500"],
                ],
                "bbox": [148.92, 494.94, 486.12, 575.4],
                "asset_id": "p0009-table-03",
            },
            {
                "block_type": "paragraph",
                "text": "4.3 ARJ21-700 飞机的性能设计要求",
                "bbox": [70.92, 617.01, 500.0, 630.0],
            },
            {
                "block_type": "table",
                "table_title": "表 2 高度速度限制",
                "table_rows": [
                    ["项目", "要求"],
                    ["最大使用高度", "11,900"],
                ],
                "bbox": [148.92, 113.1, 486.12, 332.82],
                "asset_id": "p0009-table-02",
            },
        ],
    )

    pipeline.merge_cross_page_tables([page], "document-3")
    block_texts = [
        f'{item.get("block_type")}:{item.get("table_title") or item.get("text")}'
        for item in page.rich_blocks
    ]

    # 表 2 body (y=113) 必须先于 4.3 标题 (y=617)
    assert "table:表 2 高度速度限制" in block_texts
    table2_index = next(
        i for i, t in enumerate(block_texts) if "表 2 高度速度限制" in t
    )
    clause_index = next(
        i for i, t in enumerate(block_texts) if "4.3 ARJ21" in t
    )
    assert table2_index < clause_index
    # 表 3 body (y=494) 也在 4.3 标题之前
    table3_index = next(
        i for i, t in enumerate(block_texts) if "表 3 重量限制" in t
    )
    assert table3_index < clause_index


def test_split_blocks_skips_cleaned_text_when_rich_blocks_exist() -> None:
    # OCR 页面同时有 cleaned_text（半角标点）和 rich_blocks（全角标点），
    # 内容重复但标点不同，无法文本去重。split_blocks 必须只保留 rich_blocks，
    # 否则同一节内容会生成两套 blocks，发布到 RAGFlow 时重复。
    page = pipeline.PageRecord(
        page_number=5,
        raw_text="",
        cleaned_text=(
            "1 范围\n"
            "本标准规定了低空数字航摄的基本要求,包含低空数字航空摄影、像片控制测量。\n"
            "2 规范性引用文件\n"
            "下列文件对于本文件的应用是必不可少的。"
        ),
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": "1 范围",
                "bbox": [10.0, 5.0, 90.0, 9.0],
            },
            {
                "block_type": "paragraph",
                "text": "本标准规定了低空数字航摄的基本要求，包含低空数字航空摄影、像片控制测量。",
                "bbox": [10.0, 10.0, 90.0, 14.0],
            },
            {
                "block_type": "paragraph",
                "text": "2 规范性引用文件",
                "bbox": [10.0, 15.0, 90.0, 19.0],
            },
            {
                "block_type": "paragraph",
                "text": "下列文件对于本文件的应用是必不可少的。",
                "bbox": [10.0, 20.0, 90.0, 24.0],
            },
        ],
    )

    blocks = pipeline.split_blocks("doc-ocr", [page])

    # 只输出 rich_blocks 的内容，不重复 cleaned_text
    texts = [block.text for block in blocks]
    assert texts.count("1 范围") == 1
    assert texts.count("2 规范性引用文件") == 1
    assert "本标准规定了低空数字航摄的基本要求，包含低空数字航空摄影、像片控制测量。" in texts
    # cleaned_text 的半角标点版本不应出现
    assert not any("基本要求,包含" in text for text in texts)


def test_split_blocks_keeps_cleaned_text_when_no_rich_blocks() -> None:
    # 纯 native_text 页没有 rich_blocks，cleaned_text 是唯一来源，必须保留。
    page = pipeline.PageRecord(
        page_number=1,
        raw_text="",
        cleaned_text=(
            "1 范围\n"
            "本标准规定了低空数字航摄的基本要求。"
        ),
        rich_blocks=[],
    )

    blocks = pipeline.split_blocks("doc-native", [page])

    assert any("1 范围" in block.text for block in blocks)
    assert any("本标准规定了低空数字航摄的基本要求。" in block.text for block in blocks)


def test_split_blocks_filters_fullpage_figure_and_duplicate_table_title() -> None:
    # 有结构化表格的页面不应再输出整页 figure（VLM 描述）和重复的表标题。
    page = pipeline.PageRecord(
        page_number=7,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {
                "block_type": "table",
                "table_title": "表1 检测指标内容",
                "text": "表1 检测指标内容\n一致性类型 | 指标内容\n激光雷达 | 反射率中误差",
                "table_rows": [
                    ["一致性类型", "指标内容"],
                    ["激光雷达", "反射率中误差"],
                ],
                "bbox": [10, 100, 500, 300],
                "asset_id": "p0007-table-01",
            },
            {
                "block_type": "figure",
                "text": "原 PDF 第 7 页版面\n图表辅助说明（本地 VLM 生成）",
                "bbox": [0, 0, 595, 842],
                "asset_id": "p0007-page-01",
            },
            {
                "block_type": "paragraph",
                "text": "表1 检测指标内容",
                "bbox": [10, 50, 300, 60],
            },
        ],
    )

    blocks = pipeline.split_blocks("doc-figure-dedup", [page])

    texts = [block.text for block in blocks]
    assert not any(block.block_type == "figure" for block in blocks)
    assert not any("原 PDF 第 7 页版面" in block.text for block in blocks)
    assert texts.count("表1 检测指标内容") == 1  # 只有 table 里的
    assert any(block.block_type == "table" for block in blocks)


def test_merges_same_page_adjacent_blank_title_tables_into_form() -> None:
    # 空白记录表（如 A.2 采样记录表）被 pdfplumber 拆成上下两个网格：
    # 上部字段区（2 列）+ 下部数据区（10 列），两者共享空表题且 bbox 上下
    # 相邻（间距 <40pt、左右边界一致）。它们应合并为一个完整表单表格块，
    # 而不是拆成两个割裂的 table chunk。
    pages = [
        pipeline.PageRecord(
            page_number=11,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_title": "",
                    "text": "现场地址 | 省市区（县）街（乡）",
                    "table_rows": [
                        ["现场地址", "省市区（县）街（乡）"],
                        ["检测项目", ""],
                        ["空气收集器", "□采气袋□活性炭管□硅胶管□其他"],
                    ],
                    "bbox": [89.07, 135.82, 752.8, 247.82],
                    "asset_id": "p0011-table-01",
                    "asset_ids": ["p0011-table-01"],
                    "table_merged_cell_count": 0,
                },
                {
                    "block_type": "table",
                    "table_title": "",
                    "text": "样品编号 | 仪器编号 | 备注",
                    "table_rows": [
                        ["样品编号", "仪器编号", "采样内容", "备注"],
                        ["", "", "", ""],
                        ["注：采样体积包括现场采样体积。", "", "", ""],
                    ],
                    "bbox": [89.07, 275.07, 752.8, 507.17],
                    "asset_id": "p0011-table-02",
                    "asset_ids": ["p0011-table-02"],
                    "table_merged_cell_count": 0,
                },
            ],
        ),
    ]

    result = pipeline.merge_cross_page_tables(pages, "doc-form-merge")

    assert result["structured_table_count"] == 1
    assert result["structured_table_pages"] == [11]
    assert result["cross_page_table_count"] == 0

    tables = [
        block
        for block in pages[0].rich_blocks
        if block.get("block_type") == "table"
    ]
    assert len(tables) == 1
    merged_rows = tables[0]["table_rows"]
    # 字段区 3 行 + 数据区 2 行（全空行被 _rectangular_rows 过滤）
    assert len(merged_rows) == 5
    assert merged_rows[0][0] == "现场地址"
    assert merged_rows[3][0] == "样品编号"
    # 窄表行已右填充到宽表列宽，保持矩形
    assert all(len(row) == 4 for row in merged_rows)


def test_same_page_merge_requires_adjacent_bbox() -> None:
    # 同页两个表格 bbox 不相邻（间距 >40pt）时不应合并，即使表题都为空。
    pages = [
        pipeline.PageRecord(
            page_number=11,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_title": "",
                    "text": "表A | 内容",
                    "table_rows": [
                        ["表A", "内容"],
                        ["a", "b"],
                    ],
                    "bbox": [89.07, 135.82, 752.8, 247.82],
                    "asset_id": "p0011-table-01",
                    "asset_ids": ["p0011-table-01"],
                    "table_merged_cell_count": 0,
                },
                {
                    "block_type": "table",
                    "table_title": "",
                    "text": "表B | 内容",
                    "table_rows": [
                        ["表B", "内容"],
                        ["c", "d"],
                    ],
                    "bbox": [89.07, 400.0, 752.8, 500.0],  # 间距 >40pt
                    "asset_id": "p0011-table-02",
                    "asset_ids": ["p0011-table-02"],
                    "table_merged_cell_count": 0,
                },
            ],
        ),
    ]

    result = pipeline.merge_cross_page_tables(pages, "doc-form-separate")

    assert result["structured_table_count"] == 2


def test_same_page_merged_form_not_cross_merged_with_next_page() -> None:
    # A.2（第11页）与 A.3（第12页）是两个独立的空白记录表，各自有"字段区
    # + 数据区"两个网格。同页合并把 A.2 的字段区(2列)+数据区(10列)合并为
    # 10列表，随后若把 A.3 的字段区(2列)误当 A.2 的跨页续表合并，会产生
    # 页码 11-12 的错误 chunk。同页合并后必须矩形化行宽，使跨页宽度判断
    # 使用真实合并宽度（10 vs 2 → 不合并）。
    pages = [
        pipeline.PageRecord(
            page_number=11,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_title": "",
                    "text": "现场地址 | 省市区（县）街（乡）",
                    "table_rows": [
                        ["现场地址", "省市区（县）街（乡）"],
                        ["检测项目", ""],
                        ["空气收集器", "□采气袋□活性炭管□硅胶管□其他"],
                    ],
                    "bbox": [89.07, 135.82, 752.8, 247.82],
                    "asset_id": "p0011-table-01",
                    "asset_ids": ["p0011-table-01"],
                    "table_merged_cell_count": 0,
                },
                {
                    "block_type": "table",
                    "table_title": "",
                    "text": "样品编号 | 仪器编号 | 采样内容 | 备注",
                    "table_rows": [
                        ["样品编号", "仪器编号", "采样内容", "采样流量", "备注"],
                        ["", "", "", "", ""],
                    ],
                    "bbox": [89.07, 275.07, 752.8, 507.17],
                    "asset_id": "p0011-table-02",
                    "asset_ids": ["p0011-table-02"],
                    "table_merged_cell_count": 0,
                },
            ],
        ),
        pipeline.PageRecord(
            page_number=12,
            raw_text="",
            cleaned_text="",
            rich_blocks=[
                {
                    "block_type": "table",
                    "table_title": "",
                    "text": "现场地址 | 省市区（县）街（乡）",
                    "table_rows": [
                        ["现场地址", "省市区（县）街（乡）"],
                        ["检测项目", ""],
                        ["检测仪器，编号", ""],
                    ],
                    "bbox": [76.55, 189.82, 534.45, 259.07],
                    "asset_id": "p0012-table-01",
                    "asset_ids": ["p0012-table-01"],
                    "table_merged_cell_count": 0,
                },
                {
                    "block_type": "table",
                    "table_title": "",
                    "text": "样品编号 | 仪器编号 | 检测地点 | 备注",
                    "table_rows": [
                        ["样品编号", "仪器编号", "检测地点", "检测结果", "备注"],
                        ["", "", "", "", ""],
                    ],
                    "bbox": [76.55, 276.17, 534.45, 658.27],
                    "asset_id": "p0012-table-02",
                    "asset_ids": ["p0012-table-02"],
                    "table_merged_cell_count": 0,
                },
            ],
        ),
    ]

    result = pipeline.merge_cross_page_tables(pages, "doc-form-cross")

    # A.2 和 A.3 各自独立成表，不跨页合并
    assert result["cross_page_table_count"] == 0
    page11_tables = [
        block
        for block in pages[0].rich_blocks
        if block.get("block_type") == "table"
    ]
    page12_tables = [
        block
        for block in pages[1].rich_blocks
        if block.get("block_type") == "table"
    ]
    assert len(page11_tables) == 1
    assert len(page12_tables) == 1
    assert page11_tables[0]["source_page_start"] == 11
    assert page11_tables[0]["source_page_end"] == 11
    assert page12_tables[0]["source_page_start"] == 12
    assert page12_tables[0]["source_page_end"] == 12
    # A.2 合并后行宽矩形化为数据区宽度
    widths = {len(row) for row in page11_tables[0]["table_rows"]}
    assert len(widths) == 1


def test_split_blocks_filters_form_placeholder_noise() -> None:
    # 空白记录表的占位符噪声（标准编号残渣 DB4401/T、孤立破折号、
    # "第页，共页"）应被过滤，不进入最终 blocks/chunks。
    page = pipeline.PageRecord(
        page_number=11,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": "空气中化学毒物采样记录表参见表A.2。",
                "bbox": [103.08, 72.46, 289.33, 82.91],
            },
            {
                "block_type": "paragraph",
                "text": "DB4401/T",
                "bbox": [790.08, 80.04, 800.53, 122.1],
            },
            {
                "block_type": "paragraph",
                "text": "A.2 空气中化学毒物采样记录表",
                "bbox": [344.88, 92.27, 496.93, 102.72],
            },
            {
                "block_type": "paragraph",
                "text": "采样任务编号： 第页，共页",
                "bbox": [95.04, 123.75, 746.76, 132.75],
            },
            {
                "block_type": "paragraph",
                "text": "—",
                "bbox": [790.31, 141.84, 800.76, 152.29],
            },
            {
                "block_type": "table",
                "table_title": "",
                "text": "现场地址 | 省市区（县）街（乡）",
                "table_rows": [
                    ["现场地址", "省市区（县）街（乡）"],
                    ["检测项目", ""],
                ],
                "bbox": [89.07, 135.82, 752.8, 247.82],
                "asset_id": "p0011-table-01",
                "asset_ids": ["p0011-table-01"],
                "table_merged_cell_count": 0,
            },
        ],
    )

    blocks = pipeline.split_blocks("doc-form-noise", [page])

    texts = [block.text for block in blocks]
    assert any(block.block_type == "table" for block in blocks)
    assert not any("DB4401/T" in text for text in texts)
    assert not any("第页，共页" in text for text in texts)
    assert not any(text.strip() == "—" for text in texts)
    # 导语和表题保留
    assert any("参见表A.2" in text for text in texts)
    assert any("A.2 空气中化学毒物采样记录表" in text for text in texts)


def test_clean_page_keeps_annex_marker_and_title_on_own_lines() -> None:
    # 附录页 "(资料性)" 和附录标题行应各自独立成段，避免与正文粘连
    # （修复《小麦》第8页 "使用方法遥感产量预测模型…" 粘连）。
    raw = (
        "附 录 A\n"
        "（资料性）\n"
        "产量预测模型MultimodalNet使用方法\n"
        "遥感产量预测模型（MultimodalNet）构建流程见下图。\n"
    )
    page = pipeline.PageRecord(page_number=8, raw_text=raw)
    cleaned, _ = pipeline.clean_page(page, set())

    assert "附录 A" in cleaned
    assert "(资料性)" in cleaned
    assert "产量预测模型MultimodalNet使用方法" in cleaned
    # 标题行与正文不再粘连（标题行独占一行，正文另起一行）
    assert "\n产量预测模型MultimodalNet使用方法\n" in cleaned
    assert "使用方法遥感" not in cleaned  # 无换行粘连


def test_latex_to_text_strips_trailing_equation_number() -> None:
    # 公式编号 "(1)" 被 OCR 成 "·1" 并粘在公式末尾，应剥离
    # （修复《小麦》第6页 "WERI=Y_T/Y_CK·1"）。
    from app.ingestion.parsers.remote import latex_to_text

    assert latex_to_text(r"$\mathrm { WERI } = Y _ { T } / Y _ { CK } \cdot 1$") == (
        "WERI=Y_T/Y_CK"
    )
    # 化学式点乘不受影响（"·7" 在中间，且文本末尾不是 "·数字"）
    assert (
        latex_to_text(
            r"$\mathrm { ( M g S O _ { 4 } \bullet 7 H _ { 2 } O ) _ { : } 0 . 7 \ g } ;$"
        )
        == "(MgSO4·7H2O):0.7g;"
    )


def test_figure_page_paragraphs_carry_bboxes_for_reading_order() -> None:
    # figure 页正文应带真实 bbox，使 split_blocks 能按坐标排序，避免图
    # 块被归到页面末尾的章节（修复《小麦》第5页图1归属错误）。
    page = pipeline.PageRecord(
        page_number=5,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {
                "block_type": "figure",
                "text": "图1 无人机遥感法鉴定高水效小麦品种技术流程图",
                "bbox": [91.9, 100.6, 489.4, 154.0],
                "asset_id": "p0005-figure-01",
                "asset_ids": ["p0005-figure-01"],
            },
            {
                "block_type": "paragraph",
                "text": "5 高水效小麦品种鉴定",
                "bbox": [70.9, 208.3, 175.1, 218.8],
            },
            {
                "block_type": "paragraph",
                "text": "基本要求高水效小麦品种（系）鉴定在中高等地力以上水平种植。",
                "bbox": [91.9, 262.7, 538.4, 273.2],
            },
        ],
    )

    blocks = pipeline.split_blocks("doc-fig-order", [page])

    texts = [(block.block_type, block.text) for block in blocks]
    assert texts[0] == ("figure", "图1 无人机遥感法鉴定高水效小麦品种技术流程图")
    # 图块之后才是正文段落，图不归到章节末尾
    assert texts[1][0] == "paragraph"
    assert "5 高水效小麦品种鉴定" in texts[1][1]


def test_figure_page_caption_not_duplicated_in_paragraph() -> None:
    # figure 块承载图题，_line_blocks 重新提取的图题 paragraph 应去重。
    page = pipeline.PageRecord(
        page_number=5,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {
                "block_type": "figure",
                "text": "图1 无人机遥感法鉴定高水效小麦品种技术流程图",
                "bbox": [91.9, 100.6, 489.4, 154.0],
                "asset_id": "p0005-figure-01",
                "asset_ids": ["p0005-figure-01"],
            },
            {
                "block_type": "paragraph",
                "text": "图1 无人机遥感法鉴定高水效小麦品种技术流程图",  # 重复图题
                "bbox": [186.0, 177.1, 423.2, 187.6],
            },
            {
                "block_type": "paragraph",
                "text": "5 高水效小麦品种鉴定",
                "bbox": [70.9, 208.3, 175.1, 218.8],
            },
        ],
    )

    blocks = pipeline.split_blocks("doc-fig-dedup-caption", [page])

    caption_count = sum(
        1
        for block in blocks
        if "图1 无人机遥感法鉴定高水效小麦品种技术流程图" in block.text
    )
    assert caption_count == 1  # 只有 figure 块承载图题


def test_heading_kind_detects_top_level_clause_with_roman_numeral() -> None:
    # 顶层条款标题含罗马数字/括号（"6 III类手册编制"）也应识别为 clause，
    # 否则上一页的 section_path 会残留、6.2.x 无法嵌套在正确父章节下
    # （修复《飞行手册编制规范》第9页 6.2 章节归属错误）。
    assert pipeline.heading_kind("6 III类手册编制") == "clause"
    assert pipeline.heading_kind("6 操作员手册(III－1)编制") == "clause"
    assert pipeline.heading_kind("6.3 操作员手册(III-1)编制") == "clause"
    # 量词开头仍不误判
    assert pipeline.heading_kind("3 个并联的卷积神经网络") is None
    assert pipeline.heading_kind("3 次重复") is None


def test_roman_numeral_clause_is_parent_of_sub_clauses() -> None:
    # "6 III类手册编制" 识别为顶层条款后，6.2/6.2.1 应嵌套在其下，
    # 而不是挂在错误父章节（修复《飞行手册编制规范》第9页章节归属）。
    page = pipeline.PageRecord(
        page_number=9,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": "6 III类手册编制",
                "bbox": [56.7, 97.65, 139.86, 108.95],
            },
            {
                "block_type": "paragraph",
                "text": "6.2 构成",
                "bbox": [56.7, 208, 104, 218],
            },
            {
                "block_type": "figure",
                "text": "图1 III类手册的构成",
                "bbox": [146.6, 260.1, 433.7, 452.2],
                "asset_id": "p0009-figure-01",
            },
            {
                "block_type": "paragraph",
                "text": "6.2.1 封面",
                "bbox": [56.7, 485, 114.5, 495],
            },
            {
                "block_type": "paragraph",
                "text": "封面的格式与内容参照GJB 3968执行。",
                "bbox": [77.7, 508, 261.8, 518],
            },
        ],
    )

    blocks = pipeline.split_blocks("doc-roman-clause", [page])

    by_text = {block.text: block for block in blocks}
    assert by_text["6 III类手册编制"].section_path == ["6 III类手册编制"]
    assert by_text["6.2 构成"].section_path == ["6 III类手册编制", "6.2 构成"]
    assert by_text["6.2.1 封面"].section_path == [
        "6 III类手册编制",
        "6.2 构成",
        "6.2.1 封面",
    ]
    assert by_text["封面的格式与内容参照GJB 3968执行。"].section_path == [
        "6 III类手册编制",
        "6.2 构成",
        "6.2.1 封面",
    ]


def test_frame_label_does_not_hijack_clause_stack() -> None:
    # 带外框文字结构图框内的标签（"附录"/"封面"/"正文"）是图的一部分，
    # 不是正文段落。它们被 vector 层过滤后不再参与 split_blocks；本测试
    # 直接验证"若标签未过滤则劫持章节栈"的对照场景已被 vector 层拦截——
    # 这里验证过滤后的章节归属正确（修复《飞行手册编制规范》第9页 6.2.x
    # 被挂到 ['附录'] 下）。
    page = pipeline.PageRecord(
        page_number=9,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": "6 III类手册编制",
                "bbox": [56.7, 97.65, 139.86, 108.95],
            },
            {
                "block_type": "paragraph",
                "text": "6.2 构成",
                "bbox": [56.7, 208, 104, 218],
            },
            {
                "block_type": "figure",
                "text": "图1 III类手册的构成",
                "bbox": [146.6, 260.1, 433.7, 452.2],
                "asset_id": "p0009-figure-01",
            },
            # 注意：图框内标签（封面/附录/正文）已被 vector 层过滤，不在此处
            {
                "block_type": "paragraph",
                "text": "6.2.1 封面",
                "bbox": [56.7, 485, 114.5, 495],
            },
            {
                "block_type": "paragraph",
                "text": "封面的格式与内容参照GJB 3968执行。",
                "bbox": [77.7, 508, 261.8, 518],
            },
        ],
    )

    blocks = pipeline.split_blocks("doc-frame-filter", [page])

    annex_blocks = [b for b in blocks if "附录" in b.section_path]
    assert not annex_blocks  # 没有块挂在 ['附录'] 下
    cover = next(b for b in blocks if b.text == "6.2.1 封面")
    assert cover.section_path == [
        "6 III类手册编制",
        "6.2 构成",
        "6.2.1 封面",
    ]


def test_unnumbered_subheadings_get_inferred_numbers() -> None:
    # PDF 中 "5 高水效小麦品种鉴定" 下的子节标题 "基本要求/种子要求/试验要求"
    # 未渲染编号（实际为 5.1/5.2/5.3）。split_blocks 应推断编号，使
    # "5.3.1 田间布置" 的 section_path 包含 "5.3 试验要求" 父级，而不是从
    # "5" 直接跳到 "5.3.1"。
    page = pipeline.PageRecord(
        page_number=5,
        raw_text="",
        cleaned_text=(
            "5 高水效小麦品种鉴定\n"
            "基本要求\n"
            "高水效小麦品种（系）鉴定在中高等地力以上水平种植。\n"
            "种子要求\n"
            "参试品种（系）种子质量符合GB 4404.1一级标准要求。\n"
            "试验要求\n"
            "5.3.1 田间布置\n"
            "在田间自然环境下，设置相同的节水灌溉处理。\n"
            "5.3.2 播种\n"
            "统一播量、适期播种。\n"
        ),
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-unnumbered", [page])

    by_text = {block.text: block for block in blocks if block.text}
    assert "5.1 基本要求" in by_text
    assert "5.2 种子要求" in by_text
    assert "5.3 试验要求" in by_text
    assert "5.3.1 田间布置" in by_text
    assert list(by_text["5.3.1 田间布置"].section_path) == [
        "5 高水效小麦品种鉴定",
        "5.3 试验要求",
        "5.3.1 田间布置",
    ]
    assert list(by_text["5.2 种子要求"].section_path) == [
        "5 高水效小麦品种鉴定",
        "5.2 种子要求",
    ]


def test_unnumbered_subheading_inference_does_not_promote_body_text() -> None:
    # 长段落、带标点结尾、列表项、编号行等不应被推断为无编号子节标题。
    page = pipeline.PageRecord(
        page_number=6,
        raw_text="",
        cleaned_text=(
            "5 高水效小麦品种鉴定\n"
            "5.3.5 遥感测定\n"
            "天气晴朗，农田现场的风速要求不高于4 m/s。\n"
            "5.3.6 数据处理\n"
            "5.3.6.1 参数读取\n"
            "按照生成图像的时间顺序，读取分幅图像的温度信息。\n"
            "式中：\n"
            "WERI——测试品种（系）的高水效遥感鉴定指数。\n"
        ),
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-unnumbered-safe", [page])

    texts = [block.text for block in blocks]
    # 普通长段落保持原样，不被提升
    assert "天气晴朗，农田现场的风速要求不高于4 m/s。" in texts
    # "式中：" 不被推断为子节标题（带冒号）
    assert "式中：" in texts
    # 带编号的 clause 不重复推断
    assert "5.3.6 数据处理" in texts
    # 不产生伪造的 "5.4 式中："
    assert not any(text.startswith("5.4 ") for text in texts)


def test_caption_mismatch_ignores_figure_page_with_mislabeled_table_caption() -> None:
    # 图题被误标为"表 A.1 …"（正文写"见图A.1"）的页面不是真实表格候选。
    # 传入 table_candidate_pages 后，该页的表题提示不应计入 expected，
    # 从而不触发 table_caption_alignment 门禁失败（修复《小麦》第8页）。
    page7 = pipeline.PageRecord(
        page_number=7,
        raw_text="",
        cleaned_text="",
        table_hints=["表 1 高水效小麦品种(系)评价标准"],
        rich_blocks=[
            {
                "block_type": "table",
                "table_title": "表 1 高水效小麦品种(系)评价标准",
                "table_rows": [
                    ["级别", "高水效遥感鉴定指数（WERI）", "水效率等级"],
                    ["1", "≥1.150", "极高"],
                ],
                "table_merged_cell_count": 0,
                "bbox": [71.2, 120.3, 538.4, 217.9],
                "asset_id": "p0007-table-01",
                "asset_ids": ["p0007-table-01"],
            }
        ],
    )
    page8 = pipeline.PageRecord(
        page_number=8,
        raw_text="",
        cleaned_text="",
        table_hints=["表 A.1 小麦产量预测模型构建流程"],  # 图题误标，非真实表格
    )

    # 旧行为（不传 candidate）：第 8 页被计入 expected，导致 mismatch
    old = pipeline.merge_cross_page_tables([page7, page8], "doc-fig")
    assert 8 in old["table_caption_mismatch_pages"]

    # 修复后：只把真实表格候选页的表题计入 expected
    new = pipeline.merge_cross_page_tables(
        [page7, page8],
        "doc-fig",
        table_candidate_pages={7},
    )
    assert new["table_caption_mismatch_pages"] == []


def test_unnumbered_subheading_inference_handles_long_headings() -> None:
    # 长子节标题（>12字，如"无人机影像数据获取及预处理"）也应推断编号，
    # 只要后面跟正文段落（修复《大豆》第6页 5.1/5.2 缺失）。
    page = pipeline.PageRecord(
        page_number=6,
        raw_text="",
        cleaned_text=(
            "5 数据获取及处理\n"
            "无人机影像数据获取及预处理\n"
            "无人机低空拍照方式获取影像数据，无人机影像的空间分辨率应优于3 cm。\n"
            "归一化差值植被指数计算\n"
            "无人机多光谱影像进行波段运算，提取适合大豆幼苗期使用的NDVI。\n"
        ),
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-long-heading", [page])

    by_text = {block.text: block for block in blocks if block.text}
    assert "5.1 无人机影像数据获取及预处理" in by_text
    assert "5.2 归一化差值植被指数计算" in by_text
    assert list(by_text["5.1 无人机影像数据获取及预处理"].section_path) == [
        "5 数据获取及处理",
        "5.1 无人机影像数据获取及预处理",
    ]


def test_unnumbered_subheading_inference_does_not_promote_table_cells() -> None:
    # 表格内容（表头"大豆苗情等级"、单元格"好(1级)…"）不应被推断为子节标题
    # （修复《大豆》第7页误判 7.2/7.3）。
    page = pipeline.PageRecord(
        page_number=7,
        raw_text="",
        cleaned_text=(
            "7 苗情遥感等级划分\n"
            "苗情遥感等级划分标准\n"
            "以大豆遥感长势统计数据，确定遥感监测大豆苗情等级划分标准（见表1）。\n"
            "表1 大豆苗情等级划分表\n"
            "大豆苗情等级\n"
            "好(1级)较好(2级)正常(3级)较差(4级)差(5级)遥感长势指数区间 0.81~1.00\n"
            "大豆苗情遥感等级空间分布图\n"
            "按照苗情等级划分标准，生成苗情等级分布图。\n"
        ),
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-table-cells", [page])

    by_text = {block.text: block for block in blocks if block.text}
    assert "7.1 苗情遥感等级划分标准" in by_text
    assert "7.2 大豆苗情遥感等级空间分布图" in by_text
    # 表头不被提升为 7.x
    assert not any(text.startswith("7.2 大豆苗情等级") for text in by_text)
    assert not any(text.startswith("7.3 好") for text in by_text)


def test_term_definitions_get_inferred_numbers_in_terminology_clause() -> None:
    # "3 术语和定义" 下的术语项（术语名 + 英文名）在 PDF 中无编号，实际应为
    # 3.1/3.2/3.3（修复《大豆》第5页缺失 3.1/3.2/3.3）。
    page = pipeline.PageRecord(
        page_number=5,
        raw_text="",
        cleaned_text=(
            "3 术语和定义\n"
            "下列术语和定义适用于本文件。\n"
            "太阳反射波段多光谱遥感 multispectral remote sensing\n"
            "在太阳反射波段范围内，将物体反射的电磁波信息分成两个以上波谱段。\n"
            "苗情长势 growth conditions\n"
            "作物生长期间的植株生长发育状况及产量指标等信息。\n"
            "归一化差值植被指数 normalized difference vegetation index;NDVI\n"
            "近红外波段反射率和可见光红波段反射率之差与二者之间和的比值。\n"
        ),
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-terms", [page])

    by_text = {block.text: block for block in blocks if block.text}
    assert "3.1 太阳反射波段多光谱遥感 multispectral remote sensing" in by_text
    assert "3.2 苗情长势 growth conditions" in by_text
    assert "3.3 归一化差值植被指数 normalized difference vegetation index;NDVI" in by_text
    assert list(
        by_text["3.1 太阳反射波段多光谱遥感 multispectral remote sensing"].section_path
    ) == [
        "3 术语和定义",
        "3.1 太阳反射波段多光谱遥感 multispectral remote sensing",
    ]


def test_two_char_unnumbered_subheadings_get_inferred_numbers() -> None:
    # 2 字术语名（如"区块""播幅"）在 PDF 中无编号，实际应为 3.3/3.5。
    # 修复前因 _UNNUMBERED_SUBHEADING_MIN=3 被并入上一术语正文（修复
    # 《飞播造林》第5页 3.3 区块被合并）。
    page = pipeline.PageRecord(
        page_number=5,
        raw_text="",
        cleaned_text=(
            "3 术语和定义\n"
            "下列术语和定义适用于本文件。\n"
            "无人机飞播造林\n"
            "根据植被自然演替规律，利用无人机把林木种子播撒在宜播地上。\n"
            "小播区群\n"
            "若干个相对集中，可以实施串联飞播作业的地块群。\n"
            "区块\n"
            "在小播区群内区划的相对独立且满足飞播造林的地块单元。\n"
            "宜播地\n"
            "适宜开展飞播造林的各种地类。\n"
            "播幅\n"
            "飞播作业时的有效落种宽度。\n"
        ),
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-2char-terms", [page])

    by_text = {block.text: block for block in blocks if block.text}
    assert "3.1 无人机飞播造林" in by_text
    assert "3.2 小播区群" in by_text
    assert "3.3 区块" in by_text
    assert "3.4 宜播地" in by_text
    assert "3.5 播幅" in by_text
    assert list(by_text["3.3 区块"].section_path) == ["3 术语和定义", "3.3 区块"]


def test_cross_page_unnumbered_subheading_is_inferred() -> None:
    # 子节标题在页尾、正文在下一页（如《飞播造林》第6页"种子质量"在页脚，
    # 正文"种子调拨按GB/T 15776…"在第7页）。页尾标题无下一个 block，但应
    # 被推断为 6.1，使后续 6.2 种子使用 / 6.3 种子处理 编号不位移。
    page6 = pipeline.PageRecord(
        page_number=6,
        raw_text="",
        cleaned_text=(
            "6 飞播种子\n"
            "种子质量\n"  # 页尾子节标题，正文在下页
        ),
        indexable=True,
    )
    page7 = pipeline.PageRecord(
        page_number=7,
        raw_text="",
        cleaned_text=(
            "种子调拨按GB/T 15776规定执行；种子质量应达到GB 7908规定的Ⅱ级以上。\n"
            "种子使用\n"
            "飞播造林用种实行凭证用种制度。\n"
            "种子处理\n"
            "根据实际情况可做以下处理。\n"
        ),
        indexable=True,
    )

    blocks = pipeline.split_blocks("doc-cross-page-sub", [page6, page7])

    by_text = {block.text: block for block in blocks if block.text}
    assert "6.1 种子质量" in by_text
    assert "6.2 种子使用" in by_text
    assert "6.3 种子处理" in by_text
    assert list(by_text["6.3 种子处理"].section_path) == ["6 飞播种子", "6.3 种子处理"]


def test_unnumbered_subheading_under_nested_clause_gets_nested_number() -> None:
    # 无编号子节出现在"含点号的子条款"下（如《通信网智能巡检》5.6 任务载荷
    # 下的"可见光相机的指标要求如下"）时应推断为 5.6.1，而不是基于顶层条款
    # 推断成 5.1（与已存在的 5.1 概述 冲突）。修复"5.6 里又生成了一个 5.1"。
    page10 = pipeline.PageRecord(
        page_number=10,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {"block_type": "paragraph", "text": "5 巡检无人机系统", "bbox": [56.7, 90, 150, 101]},
            {"block_type": "paragraph", "text": "5.1 概述", "bbox": [56.7, 110, 120, 121]},
        ],
    )
    page11 = pipeline.PageRecord(
        page_number=11,
        raw_text="",
        cleaned_text="",
        rich_blocks=[
            {"block_type": "paragraph", "text": "5.6 任务载荷", "bbox": [56.7, 100, 120, 111]},
            {
                "block_type": "paragraph",
                "text": "任务载荷是指巡检无人机搭载的各种专用设备或装置。",
                "bbox": [56.7, 120, 500, 131],
            },
            {
                "block_type": "paragraph",
                "text": "可见光相机的指标要求如下",
                "bbox": [56.7, 180, 200, 191],
            },
            {
                "block_type": "paragraph",
                "text": "——满足采集像素2400万以上单张影像；",
                "bbox": [56.7, 200, 400, 211],
            },
            {
                "block_type": "paragraph",
                "text": "——功耗不大于3瓦；",
                "bbox": [56.7, 210, 400, 221],
            },
            {"block_type": "paragraph", "text": "5.7 地面保障设备", "bbox": [56.7, 290, 150, 301]},
        ],
    )

    blocks = pipeline.split_blocks("doc-unnumbered-nested", [page10, page11])

    by_text = {block.text: block for block in blocks if block.text}
    # 不再生成错误的 5.1（已存在 5.1 概述）
    assert "5.1 可见光相机的指标要求如下" not in by_text
    # 正确推断为 5.6.1，嵌套在 5.6 任务载荷 下
    assert "5.6.1 可见光相机的指标要求如下" in by_text
    assert list(by_text["5.6.1 可见光相机的指标要求如下"].section_path) == [
        "5 巡检无人机系统",
        "5.6 任务载荷",
        "5.6.1 可见光相机的指标要求如下",
    ]
    assert list(by_text["5.6 任务载荷"].section_path) == [
        "5 巡检无人机系统",
        "5.6 任务载荷",
    ]


def test_heading_kind_bare_annex_is_not_heading() -> None:
    # 孤立"附录"是正文引用（如 pypdf 把"附录 B 给出了…"拆成"附录"+"B"+
    # "给出了"），不是附录标题。若判为 annex 会劫持章节栈，使后续 0.4 条款
    # 全部挂到 ['附录'] 下（GB/T 19001-2016 第8页）。
    assert pipeline.heading_kind("附录") is None
    assert pipeline.heading_kind("见附录") is None
    assert pipeline.heading_kind("详见附录") is None


def test_heading_kind_annex_prose_is_not_heading() -> None:
    # "附录 B 给出了…" 是正文陈述，不是附录标题。
    assert pipeline.heading_kind("附录 B 给出了 SAC/TC 151 制定的其他质量管理体系标准") is None
    assert pipeline.heading_kind("附录 B 给出了") is None
    assert pipeline.heading_kind("附录A 应符合") is None


def test_heading_kind_annex_with_label_is_heading() -> None:
    # 带编号/性质标记的才是真附录标题。
    assert pipeline.heading_kind("附录 A") == "annex"
    assert pipeline.heading_kind("附录B") == "annex"
    assert pipeline.heading_kind("附录 A（资料性附录）") == "annex"
    assert pipeline.heading_kind("附件 B") == "annex"


def test_unnumbered_subheading_rejects_body_fragments() -> None:
    # 正文碎片（pypdf 拆行产生）不得当作无编号子标题编号成 0.4.x。
    for frag in ("给出了", "应关系见", "对应关系见", "本标准采用", "本标准使组织能够"):
        assert pipeline._looks_like_unnumbered_subheading(frag) is False
    # 真实无编号子标题（名词短语）不受影响。
    for title in ("基本要求", "试验要求", "田间布置", "监测内容", "可见光相机的指标要求如下"):
        assert pipeline._looks_like_unnumbered_subheading(title) is True
