from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image, ImageDraw
from pypdf import PdfWriter

from app.core.config import IngestionSettings, RemoteParserSettings
from app.ingestion.parsers import DoclingClient, MinerUClient, ParserRegistry
from app.ingestion.parsers.remote import clean_inline_latex, formula_latex, latex_to_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\pmb { V } _ { { s } _ { 0 } }", "V_s0"),
        (r"{ { \pmb { V } } _ { { { s } _ { 0 } } } } ^ { 2 }", "V_s0^2"),
        (r"\pmb { V } _ { \pmb { s _ { 1 } } }", "V_s1"),
        (r"{\pmb V}_{s_{0}}^{2}", "V_s0^2"),
        (r"\pm \pmb{V}_{s_{0}}^{2}", "±V_s0^2"),
        (r"\mp 7\%", "∓7%"),
        (r"\pmatrix{a}", "a"),
        (r"\pmbextra{V}", "V"),
    ],
)
def test_latex_complete_commands_preserve_bold_variables_and_real_signs(source, expected):
    assert latex_to_text(source) == expected


def test_page5_inline_climb_formula_does_not_invent_plus_minus():
    source = (
        r"(1) Each airplane with a $\pmb { V } _ { { s } _ { 0 } }$ "
        r"of more than 70 miles per hour must be able to maintain a steady rate "
        r"of climb of at least 0.02 ${ { \pmb { V } } _ { { { s } _ { 0 } } } } ^ { 2 }$ "
        r"(that is, the number of feet per minute is obtained by multiplying "
        r"the square of the number of miles per hour by 0.02)."
    )
    result = clean_inline_latex(source)
    assert "with a V_s0 of more than 70" in result
    assert "at least 0.02 V_s0^2 (that is" in result
    assert "±" not in result and "pmb" not in result


def test_hic_formula_repairs_unambiguous_t1_t2_subscripts():
    raw = {
        "type": "equation",
        "text": (
            r"H I C = \left\{ ( t _ { : } - t _ { : } ) "
            r"\left[ \frac { 1 } { ( t _ { : } - t _ { : } ) } "
            r"\intop _ { t _ { : } } ^ { t _ { 2 } } a ( t ) d t "
            r"\right] ^ { 2 . 5 } \right\} _ { M a x }"
        ),
    }
    latex = formula_latex(raw)
    assert latex is not None
    assert "t_{2}-t_{1}" in latex
    assert r"\int_{t_{1}}^{t_{2}}" in latex
    text = MinerUClient._item_text(raw)
    assert "t2-t1" in text
    assert "∫_t1^t2" in text
    assert "t:" not in text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            (
                r"$$\n\mathbf { L } _ { \mathbf { \ S } \mathbf { \ S } \mathbf { \ S } } "
                r"= \frac { \mathbf { K } _ { \mathbf { \ S } \mathbf { \Phi } } "
                r"\mathbf { U } _ { \mathbf { \Phi } \mathbf { d e } } \mathbf { \nabla } "
                r"\mathbf { V } \mathbf { \varPsi } \mathbf { a } _ { \mathbf { \ S } "
                r"\mathbf { \Phi } } \mathbf { S } _ { \mathbf { \Phi } \mathbf { \Phi } } } "
                r"{ 4 9 8 }\n$$"
            ),
            r"L_{vt} = \frac{K_{gt} U_{de} V a_{vt} S_{vt}}{498}",
        ),
        (
            r"\mathbf { k } _ { \mathbf { g }  t } = { \frac { 0 . 8 8 \mu _ { \mathbf { g } t } } { 5 . 3 + \mu _ { \mathbf { g } t } } } =",
            r"k_{gt} = \frac{0.88\mu_{gt}}{5.3 + \mu_{gt}}",
        ),
        (
            r"\mu _ { { _ \mathrm { g t } } } = \frac { 2 \mathrm { W } } { \rho \mathrm { c _ { t } } { \mathrm { g } } \mathrm { a } _ { \mathrm { v t } } \mathrm { S } _ { \mathrm { v t } } } \frac { \mathrm { K } } { \mathrm { I } _ { \mathrm { v t } } } ^ { 2 }",
            r"\mu_{gt} = \frac{2W}{\rho\bar{c}_{t} g a_{vt} S_{vt}}\frac{K^{2}}{l_{vt}^{2}}",
        ),
    ],
)
def test_formula_ocr_recovers_complete_gust_load_identities(raw, expected):
    assert formula_latex({"type": "equation", "text": raw}) == expected
    assert MinerUClient._item_text({"type": "equation", "text": raw}) == latex_to_text(
        expected
    )


def test_formula_ocr_does_not_rewrite_a_nearby_valid_mass_ratio():
    source = r"\mu_{gt}=\frac{2W}{\rho c_t g a_{vt}S_{vt}}\frac{K}{l_{vt}}"
    assert formula_latex({"type": "equation", "text": source}) == source


def test_latex_to_text_keeps_a_symbol_before_an_accented_variable():
    assert latex_to_text(r"\rho\bar{c}_{t}") == "ρc_t"


def test_mineru_preserves_margins_uncaptioned_images_and_formula_metadata() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "doc/doc_content_list.json",
            json.dumps(
                [
                    {
                        "type": "aside_text",
                        "text": "17965",
                        "page_idx": 0,
                        "bbox": [941, 891, 962, 940],
                    },
                    {"type": "image", "img_path": "images/note.png", "page_idx": 0},
                    {"type": "text", "text": "Title 14", "text_level": 2, "page_idx": 0},
                    {"type": "equation", "text": "F=ma", "page_idx": 0},
                ]
            ),
        )
        archive.writestr("doc/images/note.png", b"image")
    result = MinerUClient._read_archive(buffer.getvalue())
    assert "17965" not in result.page_texts[1]
    assert result.excluded_blocks[1][0].text == "17965"
    mapped = MinerUClient._map_batch_blocks(result, start_page=11, end_page=20)
    assert mapped[11][0].image_content == b"image"
    assert mapped[11][1].text_level == 2
    assert mapped[11][2].latex == "F=ma"


def test_mineru_promotes_display_formula_mislabeled_as_text() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "doc/doc_content_list.json",
            json.dumps(
                [
                    {
                        "type": "text",
                        "text": (
                            r"$\mathbf { k } _ { g t } = \frac { 0.88 \mu _ { g t } } "
                            r"{ 5.3 + \mu _ { g t } }$ gust alleviation factor;"
                        ),
                        "page_idx": 0,
                        "bbox": [100, 200, 400, 250],
                    }
                ]
            ),
        )

    result = MinerUClient._read_archive(buffer.getvalue())

    formula, explanation = result.page_blocks[1]
    assert formula.block_type == "formula"
    assert formula.latex is not None and r"\frac" in formula.latex
    assert "k_gt=" in formula.text
    assert explanation.block_type == "paragraph"
    assert explanation.text == "gust alleviation factor;"
    assert "gust alleviation factor;" in result.page_texts[1]


def _pdf(path: Path, *, pages: int = 1) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    with path.open("wb") as stream:
        writer.write(stream)


def _scan_pdf(path: Path, *, pages: int = 1) -> None:
    images: list[Image.Image] = []
    for page_number in range(1, pages + 1):
        image = Image.new("RGB", (595, 842), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 100, 515, 742), outline="black", width=3)
        draw.text((120, 150), f"Scanned page {page_number}", fill="black")
        images.append(image)
    images[0].save(
        path,
        "PDF",
        resolution=72,
        save_all=True,
        append_images=images[1:],
    )
    for image in images:
        image.close()


def _mineru_archive() -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "sample/sample_content_list.json",
            json.dumps(
                [
                    {"type": "text", "text": "第一条 远程 OCR 文本。", "page_idx": 0},
                    {
                        "type": "table",
                        "table_body": "| 参数 | 值 |\n|---|---|\n|A|1|",
                        "page_idx": 0,
                    },
                ],
                ensure_ascii=False,
            ),
        )
        archive.writestr("sample/sample.md", "# 远程结果")
    return output.getvalue()


def test_mineru_zip_result_preserves_page_numbers(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    _pdf(source)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"openapi": "3.1.0"})
        assert request.url.path == "/file_parse"
        return httpx.Response(
            200,
            content=_mineru_archive(),
            headers={"content-type": "application/zip"},
        )

    client = MinerUClient(
        RemoteParserSettings(enabled=True, base_url="http://mineru.local"),
        transport=httpx.MockTransport(handler),
    )

    ready, detail = client.probe()
    result = client.parse(source)

    assert ready is True
    assert detail is None
    assert "远程 OCR 文本" in result.page_texts[1]
    assert "| 参数 | 值 |" in result.page_texts[1]
    assert [block.block_type for block in result.page_blocks[1]] == [
        "paragraph",
        "table",
    ]
    assert result.page_blocks[1][1].text.startswith("| 参数 | 值 |")


def test_mineru_processes_only_target_page_batches(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    _pdf(source)
    requested_ranges: list[tuple[int, int]] = []
    requested_methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        content = request.content
        if b'name="parse_method"\r\n\r\nocr' in content:
            requested_methods.append("ocr")
        for start, end in ((4, 5), (19, 19)):
            if (
                f'name="start_page_id"\r\n\r\n{start}'.encode() in content
                and f'name="end_page_id"\r\n\r\n{end}'.encode() in content
            ):
                requested_ranges.append((start, end))
                break
        return httpx.Response(
            200,
            content=_mineru_archive(),
            headers={"content-type": "application/zip"},
        )

    client = MinerUClient(
        RemoteParserSettings(enabled=True, base_url="http://mineru.local"),
        transport=httpx.MockTransport(handler),
    )

    result = client.parse_pages(source, [5, 6, 20], parse_method="ocr")

    assert requested_ranges == [(4, 5), (19, 19)]
    assert requested_methods == ["ocr", "ocr"]
    assert set(result.page_texts) == {5, 20}


def test_mineru_reports_batch_progress_callback(tmp_path: Path) -> None:
    # 批大小 2、目标页 1-5（连续）应产生 3 批 [1,2] [3,4] [5]，
    # progress_callback 每批结束后上报 (done, total)。
    source = tmp_path / "sample.pdf"
    _pdf(source)

    client = MinerUClient(
        RemoteParserSettings(
            enabled=True,
            base_url="http://mineru.local",
            page_batch_size=2,
        ),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                content=_mineru_archive(),
                headers={"content-type": "application/zip"},
            )
        ),
    )

    events: list[tuple[int, int]] = []
    client.parse_pages(
        source,
        [1, 2, 3, 4, 5],
        parse_method="ocr",
        progress_callback=lambda done, total: events.append((done, total)),
    )
    assert events == [(2, 5), (4, 5), (5, 5)]


def test_mineru_batch_size_defaults_to_settings(tmp_path: Path) -> None:
    # batch_size 未显式传入时应使用 RemoteParserSettings.page_batch_size。
    source = tmp_path / "sample.pdf"
    _pdf(source)
    requested_ranges: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        content = request.content
        for start, end in ((0, 1), (2, 2)):
            if (
                f'name="start_page_id"\r\n\r\n{start}'.encode() in content
                and f'name="end_page_id"\r\n\r\n{end}'.encode() in content
            ):
                requested_ranges.append((start, end))
                break
        return httpx.Response(
            200,
            content=_mineru_archive(),
            headers={"content-type": "application/zip"},
        )

    # 批大小 2 → 3 页分为 [1,2] [3]
    client = MinerUClient(
        RemoteParserSettings(
            enabled=True,
            base_url="http://mineru.local",
            page_batch_size=2,
        ),
        transport=httpx.MockTransport(handler),
    )
    client.parse_pages(source, [1, 2, 3], parse_method="ocr")
    assert requested_ranges == [(0, 1), (2, 2)]


def test_mineru_reports_page_range_for_corrupt_archive(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    _pdf(source)

    client = MinerUClient(
        RemoteParserSettings(enabled=True, base_url="http://mineru.local"),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                content=b"PK-corrupt",
                headers={"content-type": "application/zip"},
            )
        ),
    )

    try:
        client.parse_pages(source, [25])
    except RuntimeError as exc:
        assert "pages 25-25" in str(exc)
        assert "BadZipFile" in str(exc)
        assert "bytes=10" in str(exc)
    else:
        raise AssertionError("Expected a corrupt MinerU archive error")


def test_docling_response_with_page_markers_is_mapped(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    _pdf(source)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "document": {
                    "md_content": "<!-- page: 1 -->\n\n# 第一页\n正文",
                }
            },
        )

    client = DoclingClient(
        RemoteParserSettings(enabled=True, base_url="http://docling.local"),
        transport=httpx.MockTransport(handler),
    )
    result = client.parse(source)

    assert result.page_texts == {1: "# 第一页\n正文"}
    assert result.warnings == []


def test_hybrid_parser_excludes_true_blank_page(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    _pdf(source)

    request_bodies: list[bytes] = []

    def mineru_handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(request.content)
        return httpx.Response(
            200,
            content=_mineru_archive(),
            headers={"content-type": "application/zip"},
        )

    settings = IngestionSettings(
        root_dir=tmp_path / "ingestion",
        mineru=RemoteParserSettings(enabled=True, base_url="http://mineru.local"),
    )
    registry = ParserRegistry.with_builtins(
        settings,
        mineru_transport=httpx.MockTransport(mineru_handler),
    )

    document = registry.parse(source)

    assert document.route == "ocr_required"
    assert document.pages[0].route == "blank_excluded"
    assert document.pages[0].indexable is False
    assert document.qa["blank_pages"] == [1]
    assert document.qa["pages_requiring_ocr"] == []
    assert document.qa["quality_gates"][0]["status"] == "fail"
    assert document.qa["quality_gates"][0]["gate"] == "empty_document"
    assert document.chunks == []
    assert request_bodies == []


def test_hybrid_parser_retries_missing_ocr_page_individually(tmp_path: Path) -> None:
    source = tmp_path / "two-page-scan.pdf"
    _scan_pdf(source, pages=2)
    requested_ranges: list[tuple[int, int]] = []

    def mineru_handler(request: httpx.Request) -> httpx.Response:
        content = request.content
        for start, end in ((0, 1), (1, 1)):
            if (
                f'name="start_page_id"\r\n\r\n{start}'.encode() in content
                and f'name="end_page_id"\r\n\r\n{end}'.encode() in content
            ):
                requested_ranges.append((start, end))
                break
        return httpx.Response(
            200,
            content=_mineru_archive(),
            headers={"content-type": "application/zip"},
        )

    settings = IngestionSettings(
        root_dir=tmp_path / "ingestion",
        mineru=RemoteParserSettings(enabled=True, base_url="http://mineru.local"),
    )
    registry = ParserRegistry.with_builtins(
        settings,
        mineru_transport=httpx.MockTransport(mineru_handler),
    )

    document = registry.parse(source)

    assert requested_ranges == [(0, 1), (1, 1)]
    assert [page.route for page in document.pages] == ["remote_ocr", "remote_ocr"]
    assert document.qa["pages_requiring_ocr"] == []
    mineru_trace = document.qa["parser_trace"][-1]
    assert mineru_trace["retried_pages"] == [2]


def test_latex_to_text_converts_chemistry_formula_to_readable_text() -> None:
    source = r"$( \mathrm { N a N O _ { 3 } } ) : 3 . 0 \ \mathrm { g } ;$"
    assert latex_to_text(source) == "(NaNO3):3.0g;"


def test_latex_to_text_preserves_sqrt_sum_and_subscripts() -> None:
    # 区域网平差中误差公式：平方根 + 求和 + 上下标都应保留
    source = (
        r"$$\nm _ { \mathrm { p } } = \pm \sqrt { \frac { 1 } { n }"
        r"\sum _ { i = 1 } ^ { n } { ( \Delta _ { i X } ^ { 2 } +"
        r"\Delta _ { i Y } ^ { 2 } ) } }\tag{…………………………………(3}"
    )
    result = latex_to_text(source)
    assert "m_p=" in result
    assert "sqrt(" in result
    assert "∑" in result
    assert "Δ_iX^2" in result
    assert "±" in result
    assert r"\sqrt" not in result
    assert r"\sum" not in result


def test_clean_inline_latex_handles_display_equation_span() -> None:
    # MinerU 的 display 公式是 `$$\n...\n$$`（\n 为字面反斜杠+n）。
    # clean_inline_latex 必须能处理这种跨行公式，否则公式块原样保留。
    source = (
        r"$$\nm _ { \mathrm { h } } = \pm \sqrt { \frac { 1 } { n }"
        r"\sum _ { i = 1 } ^ { n } \Delta _ { i \mathrm { h } } ^ { 2 }"
        r"}\tag{…………………………(4}\n$$"
    )
    result = clean_inline_latex(source)
    assert "m_h=±sqrt((1)/(n)∑_i=1^nΔ_ih^2)" in result
    assert "……(4" in result
    assert r"$$" not in result
    assert r"\sqrt" not in result
    assert r"\sum" not in result


def test_latex_to_text_merges_stoichiometric_numbers_keeps_letter_subscripts() -> None:
    assert latex_to_text(r"$\mathrm { K H _ { 2 } P O _ { 4 } }$") == "KH2PO4"
    # 字母下标 (变量) 保留下划线以便区分
    assert latex_to_text(r"$\mathrm { \Phi _ { p H } }$ :自然。") == "Φ_pH:自然。"
    assert latex_to_text(r"$\Lambda ) : 6 \ \mathrm { c m } \times 1 \ \mathrm { c m }$") == (
        "Λ):6cm×1cm"
    )


def test_latex_to_text_handles_fraction_and_small_group() -> None:
    source = (
        r"$\mathrm { \small { \left[ ( N H _ { 4 } ) _ { 2 } S O _ { 4 } \right]"
        r": 1 . 0 ~ g ; } }$"
    )
    assert latex_to_text(source) == "[(NH4)2SO4]:1.0g;"


def test_clean_inline_latex_handles_boldsymbol_prime_and_subscript() -> None:
    # 像片重叠度公式：\boldsymbol 剥壳、\prime 转撇号、孤立反斜杠下标
    s1 = (
        r"$$ { \boldsymbol { p } } _ { X } =  { \boldsymbol { p } ^ { \prime } } _ { X }"
        r" + ( 1 -  { \boldsymbol { p } ^ { \prime } } _ { X } ) \Delta"
        r" { \boldsymbol { h } } / H\n$$"
    )
    s2 = (
        r"$$q _ { Y } = q _ { \ Y } ^ { \prime } + ( 1 - q _ { \ Y } ^ { \prime } )"
        r"\Delta h / H\tag{…………………………(B.5}\n$$"
    )
    r1 = clean_inline_latex(s1)
    r2 = clean_inline_latex(s2)
    assert r1 == "p_X=p^'_X+(1-p^'_X)Δh/H"
    assert r2 == "q_Y=q_Y^'+(1-q_Y^')Δh/H…………………………(B.5"
    assert r"\boldsymbol" not in r1
    assert r"\prime" not in r1 and r"\prime" not in r2


def test_clean_inline_latex_collapses_duplicate_prime_and_keeps_frac() -> None:
    # 双重上标撇号（嵌套粗体组导致）应合并为一个撇号。
    var = (
        r"$\boldsymbol { p } _ { \mathrm { ~ } \boldsymbol { X } \mathrm { ~ }"
        r"\boldsymbol { \cdot } \boldsymbol { q } _ { \mathrm { ~ } \boldsymbol { Y } }"
        r" ^ { \prime } } ^ { \prime }$ 航摄像片的航向、旁向标准重叠度，以百分比表示；"
    )
    assert clean_inline_latex(var) == "p_X·q_Y' 航摄像片的航向、旁向标准重叠度，以百分比表示；"
    # \frac 分子含嵌套花括号时除号必须保留（分子分母加括号避免歧义）。
    frac = r"$$\Delta t = \frac { B _ { X } } { W }\tag{…………………………(B.6}\n$$"
    assert clean_inline_latex(frac) == "Δt=(B_X)/(W)…………………………(B.6"


def test_latex_to_text_drops_underscore_before_punctuation_subscript() -> None:
    # MinerU mislabels punctuation as a subscript: `_ { : }`. The underscore
    # must be dropped while the punctuation is kept.
    assert (
        latex_to_text(
            r"$( \mathrm { K } _ { 2 } \mathrm { H P O } _ { 4 } ) _ { : }"
            r" 1 . 0 \\ \mathrm { g } _ { : }$"
        )
        == "(K2HPO4):1.0g:"
    )
    assert (
        latex_to_text(
            r"$\mathrm { ( M g S O _ { 4 } \bullet 7 H _ { 2 } O ) _ { : }"
            r" 0 . 7 \ g } ;$"
        )
        == "(MgSO4·7H2O):0.7g;"
    )


def test_clean_inline_latex_cleans_table_html_cells() -> None:
    html = (
        "<table><tr><th>溶液成分</th><th>质量或体积</th></tr>"
        "<tr><td>磷酸二氢钾 "
        r"$\mathrm { ( K H _ { 2 } P O _ { 4 } ) }$"
        "</td><td>"
        r"$0 . 7 \mathrm { ~ g ~ }$"
        "</td></tr></table>"
    )
    cleaned = clean_inline_latex(html)
    assert "mathrm" not in cleaned
    # 化学计量数下标（_2、_4）合并为普通数字，故为 (KH2PO4)
    assert "(KH2PO4)" in cleaned
    assert "0.7g" in cleaned


def test_clean_inline_latex_preserves_surrounding_prose() -> None:
    source = (
        "将上述培养基在1.05 "
        r"$\mathrm { k g / c m ^ { 2 } } , 1 2 1 . 3 \mathrm { ^ { \circ } C }$"
        " 条件下灭菌 "
        r"$2 0 \ \mathrm { m i n }$"
    )
    assert clean_inline_latex(source) == "将上述培养基在1.05 kg/cm^2,121.3^°C 条件下灭菌 20min"


def test_mineru_archive_keeps_equation_latex() -> None:
    # 公式块同时保留清洗版文本（检索）和原始 LaTeX（元数据）。
    from app.ingestion.parsers.remote import MinerUClient

    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "sample/sample_content_list.json",
            json.dumps(
                [
                    {
                        "type": "equation",
                        "text": (
                            r"$$\nm _ { \mathrm { p } } = \pm \sqrt { \frac { 1 } { n }"
                            r"\sum _ { i = 1 } ^ { n } { ( \Delta _ { i X } ^ { 2 } +"
                            r"\Delta _ { i Y } ^ { 2 } ) } }\tag{…(3}"
                        ),
                        "page_idx": 4,
                    }
                ],
                ensure_ascii=False,
            ),
        )

    result = MinerUClient._read_archive(output.getvalue())

    assert result.page_blocks[5][0].block_type == "formula"
    # 检索版是清洗后的文本（含求和、平方根）
    assert "∑" in result.page_blocks[5][0].text
    assert r"\sum" not in result.page_blocks[5][0].text
    # 元数据保留原始 LaTeX
    assert result.page_blocks[5][0].latex is not None
    assert r"\sum" in result.page_blocks[5][0].latex
    assert r"\sqrt" in result.page_blocks[5][0].latex


def test_latex_to_text_preserves_fraction_grouping() -> None:
    # \frac 分子分母整体加括号，避免 "a+b/c+d" 的歧义。
    assert latex_to_text(r"\frac{a+b}{c+d}") == "(a+b)/(c+d)"
    # 嵌套分数递归保护。
    assert latex_to_text(r"\frac{1}{1+\frac{x}{y}}") == "(1)/(1+(x)/(y))"


def test_latex_to_text_preserves_sqrt_index() -> None:
    assert latex_to_text(r"\sqrt[3]{x}") == "3th_root(x)"
    assert latex_to_text(r"\sqrt[2]{x}") == "sqrt(x)"
    assert latex_to_text(r"\sqrt{x+1}") == "sqrt(x+1)"


def test_latex_to_text_keeps_named_operators_and_arrows() -> None:
    assert latex_to_text(r"\lim_{x\to0} f(x)") == "lim(x→0)f(x)"
    assert latex_to_text(r"\min(a,b)") == "min(a,b)"
    assert latex_to_text(r"\log_{10} x") == "log(10)x"
    assert latex_to_text(r"\uparrow") == "↑"
    assert latex_to_text(r"\downarrow") == "↓"


def test_latex_to_text_keeps_distinct_primes() -> None:
    # f' / f'' / f''' 是不同阶导数，不能去重。
    assert latex_to_text("f''") == "f''"
    assert latex_to_text("f'''") == "f'''"


def test_latex_to_text_keeps_accent_command_arguments() -> None:
    # \vec / \hat / \overline 等只去命令、保留参数。
    assert latex_to_text(r"\vec{x}") == "x"
    assert latex_to_text(r"\hat{\theta}") == "θ"
    assert latex_to_text(r"\overline{AB}") == "AB"
    assert latex_to_text(r"\mathbf{A}") == "A"
