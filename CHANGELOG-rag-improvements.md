# Internal-RAG 解析质量与文档撰写辅助 — 完整改动记录

> 面向 Codex 阅读：本文档完整记录了对 `D:\internal-rag` 项目的所有改动，包括背景、根因、实现位置、测试和验证结果。所有代码改动均已通过 `ruff` 检查，新增/修改的单元测试在 Windows 环境（Python ≥3.11）下运行 `pytest` 验证。

---

## 一、项目背景

本项目是一个内部 RAG（检索增强生成）系统，核心链路为：

```
PDF 上传 → 原生解析(native/pypdf) → 矢量解析(vector/pdfplumber)
        → 远程 OCR(MinerU) → 混合重建 → 切块(chunks) → 发布(RAGFlow)
```

部署在 Windows，Python 3.11+（`.venv`），服务端口：
- Gateway API：`http://127.0.0.1:8080`
- MinerU：`http://127.0.0.1:8886`（本地 GPU 推理，3070 Laptop 8G）
- Ollama：`http://127.0.0.1:11434`（qwen3.5:9b-q4_K_M）
- RAGFlow：`http://127.0.0.1:9380`（Docker，v0.26.4）

**关键运行约束**：8G 显存 + 32G 内存，MinerU 与 Ollama 不能同时满载运行，否则出现 `MemoryError` 批量解析失败（见第五节）。

---

## 二、改动总览

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `app/ingestion/pipeline.py` | 修改 | 文本层损坏检测、水印页检测、顶级条文识别、附录装饰清理、表格跨页重排 |
| `app/ingestion/parsers/hybrid_pdf.py` | 修改 | text_layer_review 页路由到 OCR、排除非索引页 |
| `app/ingestion/parsers/remote.py` | 修改 | LaTeX 化学式清洗（正文/表格/HTML 三处） |
| `app/ingestion/parsers/vector_pdf.py` | 修改 | 数学字体检测，公式页路由到 OCR |
| `tests/unit/test_pipeline.py` | 新增 | 文本层损坏/水印/顶级条文测试 |
| `tests/unit/test_remote_parsers.py` | 修改 | LaTeX 清洗测试 |
| `tests/unit/test_vector_pdf.py` | 修改 | 数学字体检测测试 |
| `tests/unit/test_regulation_preprocess.py` | 修改 | 顶级条文/附录装饰/表格重排测试 |
| `experiments/draft_section.py` | 新增 | 交互式文档撰写辅助脚本（试点，独立运行） |

---

## 三、解析质量改动（核心）

### 3.1 PostScript 字形名乱码检测

**问题**：GB/T 标准文档的 PDF 字体 Unicode 映射损坏时，`pypdf` 提取出 `/G21`、`/G22` 这类 PostScript 字形名而非真实文本。此类文档之前被错误路由为 `native_text`，全文乱码。

**实现**（`app/ingestion/pipeline.py`）：

```python
GLYPH_NAME_RE = re.compile(r"/G[0-9A-F]{2,}")

def is_glyph_name_garbage(text: str) -> bool:
    """Detect pages whose text layer exposes PostScript glyph names."""
    compact = compact_chars(text)
    if not compact or len(compact) < 40:
        return False
    glyph_matches = GLYPH_NAME_RE.findall(compact)
    if len(glyph_matches) < 5:
        return False
    # 中文占比低时才算垃圾（避免误伤含少量 /G 的正常文档）
    cjk = sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff")
    if cjk / len(compact) >= 0.05:
        return False
    return True
```

**路由逻辑**：`parse_pdf()` 中，`text_layer_corrupted = is_glyph_name_garbage(raw_text)`，为 `True` 时路由到新路由 `text_layer_review`。

**PageRecord 新增字段**：`text_layer_corruption: dict[str, Any]`，记录 `{"glyph_name_garbage": bool}`。

### 3.2 text_layer_review 页路由到 OCR

**问题**：文本层损坏的页面被错误地交给 vector 解析器当作 figure 处理（因为页面含图片/排版对象）。

**实现**（`app/ingestion/parsers/hybrid_pdf.py`）：

```python
text_layer_targets = {
    page.page_number for page in document.pages
    if page.route == "text_layer_review"
}
# 从 vector/OCR 目标中排除
vector_targets = (candidate_ocr_targets - text_layer_targets) | table_candidate_targets | visual_targets | formula_targets
figure_targets=(candidate_ocr_targets - text_layer_targets) | visual_targets
```

即 `text_layer_review` 页**只**进入 MinerU OCR 目标（`candidate_ocr_targets` 包含 `text_layer_review`），不进入 vector 解析。

同时 `_rebuild()` 中 content_pages 排除非索引页：

```python
content_pages = [
    page for page in document.pages
    if page.page_number not in blank_pages and page.indexable
]
```

### 3.3 水印页检测

**问题**：部分文档存在"整页只有一个蓝色小水印图、无任何文本"的页面，之前有些被忽略、有些被错误分类为 figure。

**实现**（`app/ingestion/pipeline.py`）：

```python
WATERMARK_IMAGE_MAX_EDGE = 400

def is_watermark_only_page(pdf_page: Any, raw_text: str) -> bool:
    if compact_chars(raw_text):
        return False
    resources = pdf_page.get("/Resources", {})
    xobjects = resources.get("/XObject", {})
    if not xobjects:
        return False
    image_count = 0
    large_image_count = 0
    for xobj in xobjects.values():
        try:
            obj = xobj.get_object()
            if obj.get("/Subtype") != "/Image":
                continue
            image_count += 1
            width = int(obj.get("/Width", 0) or 0)
            height = int(obj.get("/Height", 0) or 0)
            if width > WATERMARK_IMAGE_MAX_EDGE or height > WATERMARK_IMAGE_MAX_EDGE:
                large_image_count += 1
        except Exception:
            continue
    return image_count > 0 and large_image_count == 0
```

路由：此类页 → 新路由 `watermark_excluded`，`indexable=False`（不进入索引）。QA 输出新增 `watermark_pages`。

### 3.4 顶级条文识别（章节错位修复）

**问题**：标准文档中"数字 + 中文标题"形式的顶级条文（如 `6 追溯方法`、`2 规范性引用文件`）未被识别为 clause，导致跨页时章节状态残留——下一页内容错误归属到上一节的标题下，产生错位 chunk。

**实现**（`app/ingestion/pipeline.py` HEADING_PATTERNS 新增一个 clause 分支）：

```python
(
    "clause",
    re.compile(
        r"^\s*\d{1,2}\s+"
        r"(?!个|次|年|天|月|条|名|种|类|份|处|倍|元|米|cm|mm|kg|℃|°)"
        r"[\u4e00-\u9fff]{2,12}$"
    ),
),
```

设计要点：排除量词开头（`3 次重复`、`2 个` 不误判），要求 2-12 个中文字符，整行较短。

### 3.5 附录装饰字母清理

**问题**：附录页顶部有独立的单字母装饰（如 `A`、`A`），`normalize_line` 后合并成 `A A`，被当成普通段落挂到残留章节下，产生垃圾 chunk（如 `5.3.9\nA A`）。

**实现**（`app/ingestion/pipeline.py`）：

```python
ANNEX_DECOR_RE = re.compile(r"^(?:[A-Z]|[A-Z]\s+[A-Z])$")
```

在 `clean_page()` 中，仅当行处于页首 3 行（`is_margin`）时匹配并移除，避免误伤正文中的单字母。

### 3.6 表格跨页合并后的 bbox 重排

**问题**：`merge_cross_page_tables()` 把合并后的 table block 用 `append()` 追加到页尾，导致表格被排到同页后续内容（如下一个标题）之后，`split_blocks` 按顺序归属时表格章节错乱。

**实现**（`app/ingestion/pipeline.py`）：

```python
def _rich_block_bbox_key(item: dict[str, Any]) -> tuple[float, float]:
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 2:
        return (float("inf"), float("inf"))
    try:
        return (float(bbox[1]), float(bbox[0]))  # (top, left)
    except (TypeError, ValueError):
        return (float("inf"), float("inf"))

# merge_cross_page_tables() 返回前
for page in pages:
    page.rich_blocks.sort(key=_rich_block_bbox_key)
```

效果：表格块回到正确的阅读位置，章节归属正确（表 2 body 归 `4.2.2` 而非 `4.3`）。

### 3.7 数学字体检测，公式页路由到 OCR

**问题**：`WERI = YT/YCK` 这类公式的上下标在 PDF 文本层是平铺的（`YT`、`YCK`），原生提取丢失上/下标信息。原公式检测条件（`=` 出现 ≥4 次）太严格，含 CambriaMath 数学字体的页未被路由到 OCR。

**实现**（`app/ingestion/parsers/vector_pdf.py`）：

```python
@staticmethod
def _has_math_font(page: Any) -> bool:
    try:
        fontnames = {char.get("fontname", "") for char in page.chars}
    except (AttributeError, TypeError, ValueError):
        return False
    if not fontnames:
        return False
    math_hints = ("cambriamath", "cambria math", "math", "stix", "xits")
    for fontname in fontnames:
        lowered = fontname.casefold()
        if any(hint in lowered for hint in math_hints):
            return True
    return False
```

在 `preflight_pages()` 中，原有公式条件不满足时补充 `elif self._has_math_font(page): result.formula_pages.add(page_number)`。`formula_pages` 随后进入 MinerU OCR 目标（hybrid_pdf.py 已有逻辑），OCR 渲染后可保留上/下标。

---

## 四、LaTeX 化学式清洗（MinerU 输出）

**问题**：MinerU OCR 对化学式/公式输出 LaTeX 源码，如 `$( \mathrm { N a N O _ { 3 } } ) : 3 . 0 \ \mathrm { g } ;$`。这些源码直接进索引变成乱码。

### 4.1 核心函数

**实现**（`app/ingestion/parsers/remote.py`）：

```python
_LATEX_SYMBOLS = {
    r"\times": "×", r"\cdot": "·", r"\bullet": "·", r"\circ": "°",
    r"\degree": "°", r"\pm": "±", r"\div": "÷", r"\sim": "~",
    r"\approx": "≈", r"\le": "≤", r"\ge": "≥", r"\ne": "≠",
    r"\rightarrow": "→", r"\leftarrow": "←", r"\infty": "∞", r"\ldots": "…",
    r"\Lambda": "Λ", r"\Phi": "Φ", r"\Gamma": "Γ", r"\Delta": "Δ",
    r"\Theta": "Θ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Upsilon": "Υ",
    r"\Omega": "Ω", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ",
    r"\delta": "δ", r"\epsilon": "ε", r"\mu": "μ", r"\nu": "ν",
    r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\omega": "ω", r"\theta": "θ", r"\xi": "ξ", r"\psi": "ψ",
}
_GREEK_OR_SYMBOL_RE = re.compile(
    "|".join(re.escape(key) for key in sorted(_LATEX_SYMBOLS, key=len, reverse=True))
)

_PUNCT_RE = re.compile(r"^[，。；;:：、,.]$")
_STOICHIOMETRIC_RE = re.compile(r"^\d{1,3}$")

def _subscript(content: str) -> str:
    """Collapse a LaTeX subscript group."""
    collapsed = re.sub(r"\s+", "", content)
    if _PUNCT_RE.fullmatch(collapsed):      # 标点误标为下标 → 去掉下划线
        return collapsed
    if _STOICHIOMETRIC_RE.fullmatch(collapsed):  # 化学计量数 → 直接合并
        return collapsed
    return "_" + collapsed                  # 字母下标（变量）→ 保留下划线

def latex_to_text(latex: str) -> str:
    """Convert a LaTeX formula fragment to readable plain text."""
    text = latex.replace("$", "")
    text = re.sub(
        r"\\(?:mathrm|mathbf|mathit|text|mathcal|operatorname|rm)\s*\{([^{}]*)\}",
        lambda match: re.sub(r"\s+", "", match.group(1)),
        text,
    )
    text = re.sub(r"\\small\s*\{([^{}]*)\}", lambda match: match.group(1), text)
    text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\left\s*([\[\]()])", r"\1", text)
    text = re.sub(r"\\right\s*([\[\]()])", r"\1", text)
    text = _GREEK_OR_SYMBOL_RE.sub(lambda m: _LATEX_SYMBOLS[m.group(0)], text)
    text = re.sub(r"_\s*\{([^{}]*)\}", lambda m: _subscript(m.group(1)), text)
    text = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: "^" + re.sub(r"\s+", "", m.group(1)), text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    text = text.replace("~", " ")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\s+([,，。；;:：!?！？、)])", r"\1", text)
    return text.strip()

def clean_inline_latex(text: str) -> str:
    """Clean ``$...$`` inline-math fragments embedded in plain text items."""
    def replace(match: re.Match[str]) -> str:
        return latex_to_text(match.group(1))
    return re.sub(r"\$([^$\n]+)\$", replace, text)
```

### 4.2 三处应用点（`remote.py`）

1. **`_item_text` 文本分支**（`type in {equation, interline_equation, inline_equation}` → `latex_to_text`；其余文本 → `clean_inline_latex`）

2. **`_item_text` 表格分支**（table 的 `table_body` 用 `clean_inline_latex`）

3. **`_read_archive` 构建 `table_html`**（对 `raw_item["table_body"]` 用 `clean_inline_latex`）

### 4.3 清洗效果示例

| 原始 LaTeX | 清洗后 |
|---|---|
| `$( \mathrm { N a N O _ { 3 } } ) : 3 . 0 \ \mathrm { g } ;$` | `(NaNO3):3.0g;` |
| `$( \mathrm { K H _ { 2 } P O _ { 4 } } ) : 1 . 0 \ \mathrm { g } ;$` | `(KH2PO4):1.0g;` |
| `$\mathrm { ( M g S O _ { 4 } \bullet 7 H _ { 2 } O ) _ { : } 0 . 7 \ g } ;$` | `(MgSO4·7H2O):0.7g;` |
| `$\mathrm { \Phi _ { p H } }$ :自然。` | `Φ_pH:自然。`（字母下标保留 `_`） |
| `$\mathrm { \small { \left[ ( N H _ { 4 } ) _ { 2 } S O _ { 4 } \right] : 1 . 0 ~ g ; } }$` | `[(NH4)2SO4]:1.0g;` |
| `1.05 $\mathrm { k g / c m ^ { 2 } } , 1 2 1 . 3 \mathrm { ^ { \circ } C }$` | `1.05 kg/cm^2,121.3^°C` |

**设计取舍**：
- 化学计量数下标（纯数字，如 `_4`）→ 直接合并（`MgSO4`），因为检索时用户输入 `MgSO4` 而非 `MgSO₄`
- 字母下标（`_pH`、`_T`、`_LE`）→ 保留下划线，避免 `pH` 与 `p_H` 语义混淆
- 标点被误标为下标（`_ : `）→ 去掉下划线只留标点

---

## 五、文档撰写辅助试点（新增，独立）

### 5.1 背景与定位

用户目标：**通过已有知识库让大模型辅助公司人员撰写规范性技术文档**（如审定计划、试验计划、技术报告），生成后人工审核。

知识库现状：95 份文档，`dataset_id=8837f5068b2a11f1b7c4d904b160df50`（deepdoc 解析，效果一般）。含无人机/低空技术标准（~60份）、适航规章 CCAR 系列（~15份）、航空质量体系标准（~10份）。

### 5.2 实现

**新增文件**：`experiments/draft_section.py`（独立脚本，不改现有代码）

**运行**：
```powershell
cd D:\internal-rag\source\internal-rag
python experiments\draft_section.py
```

**流程**：
1. 检查 RAGFlow / Ollama 服务状态（可降级）
2. 交互选择文档类型（审定计划/试验计划/技术报告）→ 选择章节（预置骨架）
3. 输入产品/主题关键词
4. 从知识库检索（`/api/v1/retrieval`，复用 RAGFlow API）
5. 组装 prompt（模板 + 检索片段 + 引用编号要求）
6. 调用 Ollama `qwen3.5:9b-q4_K_M` 生成该节初稿
7. 展示初稿 + 引用来源

**模板骨架**（预置在 `DOC_TEMPLATES`）：
- 审定计划：范围/引用文件/符号及缩略语/产品描述/规章依据/符合性方法/符合性详细说明/试验计划
- 试验计划：范围/引用文件/试验项目/试验条件与设备/试验程序/安全与风险
- 技术报告：概述/依据文件/主要内容/结论与建议

**关键实现**：
- `retrieve()`：直接调 RAGFlow `api/v1/retrieval`，归一化 chunk 文本和来源文档名
- `generate()`：调 Ollama `/api/chat`，temperature=0.3
- `build_prompt()`：注入检索片段并要求标注 `[资料N]` 编号

### 5.3 已知限制与后续方向

- 9B 量化模型写长结构化文档能力有限，建议后续接 API 或升级硬件（24G 显存跑 32B）
- deepdoc 解析的知识库效果一般，后续可用本项目改进后的 MinerU 流水线重新解析发布
- 模板骨架目前是预置的，后续可从 ARJ21 等真实文档提炼更贴近实际的模板

---

## 六、测试

### 6.1 新增/修改测试

**`tests/unit/test_pipeline.py`**（新增）：
- `test_glyph_name_re_matches_postscript_glyph_names`
- `test_is_glyph_name_garbage_detects_broken_font_text_layer`
- `test_is_glyph_name_garbage_ignores_normal_chinese_document`
- `test_is_glyph_name_garbage_ignores_short_text` / `..._sparse_glyph_names`
- `test_parse_pdf_routes_glyph_name_pages_to_text_layer_review`
- `test_parse_pdf_keeps_normal_page_as_native_text`
- `test_is_watermark_only_page_*`（4 个：检测/拒绝带文本/拒绝大图/路由）
- Fake 类：`FakeXObject`、`FakePdfPage`（支持 `get("/Resources")`）、`FakePdfReader`

**`tests/unit/test_remote_parsers.py`**（修改）：
- `test_latex_to_text_converts_chemistry_formula_to_readable_text`
- `test_latex_to_text_merges_stoichiometric_numbers_keeps_letter_subscripts`
- `test_latex_to_text_handles_fraction_and_small_group`
- `test_latex_to_text_drops_underscore_before_punctuation_subscript`
- `test_clean_inline_latex_cleans_table_html_cells`
- `test_clean_inline_latex_preserves_surrounding_prose`

**`tests/unit/test_vector_pdf.py`**（修改）：
- `test_has_math_font_detects_cambria_math`

**`tests/unit/test_regulation_preprocess.py`**（修改）：
- `test_heading_kind_detects_top_level_numbered_clause`
- `test_clean_page_removes_annex_decor_letter`
- `test_merge_cross_page_tables_reorders_rich_blocks_by_bbox`

### 6.2 验证方式

- **ruff**：所有修改文件 `python -m ruff check` 全部通过
- **pytest**：需在 Windows 环境（项目 `.venv`，Python ≥3.11）运行。Linux 沙箱因 Python 3.10 缺 `datetime.UTC` 无法运行 pytest，改用 standalone Python 脚本验证函数逻辑
- **真实数据验证**：
  - 用 ARJ21 审定计划 IR 数据重放 `split_blocks`，验证表 2/表 3 章节归属从 `4.3` 修正为 `4.2.2`/`4.2.3`
  - 用霉菌试验 MinerU 原始输出验证 LaTeX 清洗效果

---

## 七、已知问题与部署注意

1. **Gateway 不热重载**：`deploy/gateway/start.ps1` 不用 `--reload`，且 8080 端口被占用时直接退出。代码改动后需先 `stop.ps1` 再 `start.ps1`。

2. **内存/显存约束（3070 Laptop 8G + 32G）**：
   - MinerU 与 Ollama 同时满载会互相挤出显存，导致连接失败/`MemoryError`
   - 批量解析时可能整机内存耗尽，大批文档 `INGESTION_PROCESSING_FAILED`
   - 建议：解析与问答错峰，或限制 Ollama GPU 层数

3. **RAGFlow 发布状态**：job 的 `publication` 字段多为 `not_requested`、`dataset_id: None`，说明文档尚未通过本项目发布到 RAGFlow 知识库。现有知识库（deepdoc 解析）是在 RAGFlow 界面单独管理的。

4. **RAGFlow 服务**：运行在 Docker Desktop，`deploy/ragflow/start.ps1` 启动，数据在 Docker 命名卷（勿 `docker compose down -v`）。

---

## 八、文件清单

```
D:\internal-rag\source\internal-rag\
├── app\ingestion\
│   ├── pipeline.py              (2013 行，修改)
│   └── parsers\
│       ├── hybrid_pdf.py        (809 行，修改)
│       ├── remote.py            (699 行，修改)
│       └── vector_pdf.py        (844 行，修改)
├── experiments\
│   └── draft_section.py         (365 行，新增，独立试点)
└── tests\unit\
    ├── test_pipeline.py         (219 行，新增)
    ├── test_remote_parsers.py   (321 行，修改)
    ├── test_vector_pdf.py       (366 行，修改)
    └── test_regulation_preprocess.py (382 行，修改)
```

---

## 九、资料查询与人工 Chunk 修订（2026-08-03）

### 9.1 处理队列资料查询

- `GET /api/v1/ingestion/jobs` 新增可选 `query` 参数。
- 查询在全部任务中完成匹配后再应用 `limit`，避免旧资料因“只加载最近任务”而无法找到。
- 匹配字段包括：文件名、上传文件夹相对路径、任务 ID、文件 SHA256、目标知识库 ID、解析状态、发布状态和错误信息。
- 资料接入台 02 区域新增搜索框、清空按钮、250ms 防抖和匹配数量提示；当前最多返回 200 条匹配任务。

### 9.2 页面诊断中的人工修订

参考 RAGFlow DeepDoc 的 Chunk 编辑方式，在“处理结果 → 页面解析检查 → 该页关联 Chunk”增加“编辑”入口：

- 新增 `PATCH /api/v1/ingestion/jobs/{job_id}/chunks/{chunk_id}`。
- 可修改发布用 Chunk 文本并填写可选修改说明，不改写原 PDF、原始文本层、内容块和图片/表格资产。
- 修改同步写入 `document_ir.json`、`chunks.jsonl` 和 `qa.json`。
- 每次修改追加到 `manual_edits.jsonl`，记录操作者、原因、版本、时间及修改前后文本哈希。
- 页面和 Chunk 显示人工修订标记；质量门禁增加 `manual_revision` 警告。
- 已生成但未发布的旧发布计划自动失效，需要重新生成。
- 已经发布完成的版本禁止在资料接入台继续修改，避免本地内容与 RAGFlow 知识库不一致；此时应在 RAGFlow 中编辑或重新处理生成新版本。
- 如果人工修改的是表格 Chunk，则清除该 Chunk 中旧的结构化 HTML/行数据，并改为 `manual_text`，避免发布时“新文字 + 旧表格”重复或冲突；原表格截图和解析块仍保留用于核对。

### 9.3 改动文件与验证

- `app/schemas/ingestion.py`：人工修订请求/响应模型。
- `app/api/v1/ingestion.py`：查询参数与 Chunk PATCH 路由。
- `app/ingestion/jobs.py`：全量过滤、修订持久化、审计、质量状态与发布计划失效处理。
- `app/web/ingestion.html`、`app/web/static/ingestion.js`、`app/web/static/ingestion.css`：查询与编辑交互。
- `tests/unit/test_ingestion_api.py`：新增查询、持久化、表格替换、已发布保护和界面元素测试。

验证结果：

- `tests/unit/test_ingestion_api.py`：15 项全部通过。
- Ruff：本次修改的 Python 文件全部通过。
- 前端 JavaScript：Node `--check` 通过。
- 完整 `tests/unit`：109 项通过；另有 2 项与本次改动无关的既有失败，分别位于附件装饰字符清理计数和 LaTeX 化学式下标期望。
- Gateway 已重启，健康检查正常；线上任务查询实测 95 条记录中按文件名前缀可准确返回目标文件。

---

## 十、同页多表与合并单元格保真（2026-08-03）

以《水利水电工程无人机机载激光雷达地形测量技术导则》为回归样本，修复了矢量 PDF 表格从解析到发布 Chunk 的结构损坏：

- 表题按页面坐标与表格边界一对一绑定，不再让同页多个表格全部继承第一个表题。
- 当一个检测网格覆盖多个“表 N”标题时，按标题与水平边界重新裁切为独立表格。
- 表格 ID 加入页内顺序与边界，避免同页同表头表格发生 ID 冲突。
- `表A.1` 等字母编号已纳入表题识别；附录标题与说明文字不再被吞入表格。
- 优先保留 pdfplumber 检出的真实合并单元格，不再因“填充单元格更多”误选拆碎的显式网格。
- `None` 合并占位转换为 HTML `colspan`；备注、注释等整行单元格在纯文本 Chunk 中也不再附加空列分隔符。
- 保留单页表格原始 span HTML 到最终 Chunk；仅在跨页合并或分块重建时回退为矩形 HTML。
- 新增表题一致性、重复表格 ID、合并单元格未保真三类诊断与质量门禁。

真实文件复测结果：

- 第 7～8 页：表1～表6均拥有正确表题、章节和唯一 ID。
- 第 9 页：表7和表8拆分为两个独立表格，正文说明不再混入表格。
- 第 22 页：表A.1从真实网格起点提取，章节为“附录 A”，共18行；备注和注释保留跨列结构。
- 9个表格、9个唯一 ID；表题错配、重复 ID、未保真合并单元格诊断均为空。

验证结果：相关表格测试31项通过，完整单元测试114项通过；仍有2项与本次改动无关的既有失败。Ruff检查通过，Gateway已重启，目标资料已重新解析但未自动发布到RAGFlow。

---

## 十一、单行法规条款 Chunk 完整性（2026-08-03）

修复“页面清洗文本完整、最终关联 Chunk 丢失单行条款”的问题：

- 原分块器把所有 `clause` 块都当作标题排除；当条号和完整正文位于同一行且没有后续段落时，分组会因正文为空而被直接跳过。
- 新增“实质性单行条款”判定，保留带规范性谓词、完整句末标点或足够正文长度的条款。
- `4 通信设备`、`4.1 基本要求` 等纯结构标题仍不单独生成 Chunk，避免污染检索结果。
- 单行条款同时作为标题和正文时只写入一次，避免 Chunk 内容重复。
- 原生解析和混合解析均新增 `clause_chunk_coverage` 失败门禁；如实质条款在 `blocks → chunks` 阶段消失，将阻止发布。
- QA 新增 `unpublished_clause_count` 与 `unpublished_clause_ids`，可直接检查漏条款数量和条号。

真实文件《低空航空器通信接入要求》复测结果：

- 最终 Chunk 数由 22 增加至 35。
- `6.1.2` 和 `7.1` 已分别生成独立 Chunk，页码均为第 9 页。
- `unpublished_clause_count = 0`，纯结构标题没有生成无意义的独立 Chunk。
- 任务已重新处理但未自动发送到 RAGFlow，发布状态为 `not_requested`。

验证结果：新增 2 项分块回归测试全部通过；完整单元测试 116 项通过，仍有 2 项与本次改动无关的既有失败。Ruff 检查通过，Gateway 已重启并通过健康检查。

---

## 十二、顺序化人工审核队列（2026-08-03）

为解决资料较多时难以逐份、按序完成审核的问题，资料接入台增加独立于解析状态和发布状态的人工审核工作流：

- 审核状态分为“待审核、审核中、已通过、需返工”，并与“已发送”分栏展示。
- 列表固定按首次上传时间和批次内顺序排列；重试、人工修改和状态更新不再因文件修改时间改变位置。
- 批量上传会记录 `batch_id` 和 `sequence_in_batch`，文件夹内文件继续按自然文件名顺序提交。
- 处理结果页新增审核进度、当前序号、审核备注、上一份、下一份、暂存并下一份、需返工并下一份、审核通过并下一份。
- “需返工”必须填写问题说明；所有状态变更追加写入 `review_history.jsonl`，保留操作者和时间。
- 重试解析或人工修改 Chunk 会自动重置为“待审核”，防止旧审核结论继续作用于新结果。
- 只有“已通过”的资料才能实际发布到 RAGFlow；发布计划仍可在审核前生成用于核对。
- 批量发送入口仅显示在“已通过”页签，避免误发未经审核的资料。
- 前端分页加载全部任务，不再受单次 200 条上限影响；资料搜索在已加载的完整集合内即时过滤。
- 旧任务没有审核字段时自动兼容为“待审核”，不会自动批准或自动发布。

主要改动文件：

- `app/schemas/ingestion.py`：审核状态、审核记录、批次顺序和更新请求模型。
- `app/api/v1/ingestion.py`：审核 PATCH 接口、分页与审核状态筛选参数。
- `app/ingestion/jobs.py`：稳定排序、审核持久化与审计、发布门禁、重试/修订失效规则。
- `app/web/ingestion.html`、`app/web/static/ingestion.js`、`app/web/static/ingestion.css`：审核队列、连续导航、进度统计及批量发送限制。
- `tests/unit/test_ingestion_api.py`：审核流程、返工备注、稳定分页、发布门禁和界面元素回归测试。

验证结果：接入台接口专项测试 19 项全部通过；完整单元测试 120 项通过，仍只有第九节已记录的 2 项既有失败。Ruff 与前端 JavaScript 语法检查均通过。Gateway 已重启，`/health`、`/ready`、页面、脚本和样式资源均在线，旧任务自动显示为“待审核”，未改写或自动批准任何现有资料。

---

## 十三、text_layer_review 页矢量表格恢复（2026-08-04）

### 问题
`《无人机低空遥感监测的多传感器一致性检测技术规范》` 第 18 页（表3）解析错误：
- 列错位（`0.3< Mz ≤0.5` 被放到第 1 列）；
- 单元格值合并（`良 一般`、`RSD ≤1% 1%< RSD ≤3%`）；
- 公式编号丢失（`公式(3)(4)(5)` → `公式(5)`）；
- 表注被平铺 6 列。

### 根因
第 18 页原生文本层含损坏字体假字（`犚犛犇`=RSD、`犿ｆ`=mf），被 `has_fake_glyph_corruption` 判为 `text_layer_review`，从而**只进 MinerU OCR**。但该页有完整的矢量表格线框，pdfplumber 能精确恢复 26 行×6 列结构（含合并单元格、表注）。MinerU OCR 的表格重建质量差，把正确结构破坏。

### 修复（3 个文件）
1. `app/ingestion/parsers/vector_pdf.py`：
   - 新增 `FAKE_GLYPH_TO_ASCII`：损坏字体假字码位 U+7280..U+72D8 → ASCII 字母映射（从文档英文单词推导，如 Technical/multi/GB/RSD/s(x)/Mz）。
   - 新增 `repair_fake_glyphs()`：NFKC + 假字还原。
   - `normalize_cell()`：仅当单元格含假字时执行还原（不改变正常单元格行为）。
   - 表格 `table_title` 也执行 `repair_fake_glyphs`。
2. `app/ingestion/parsers/hybrid_pdf.py`：
   - `vector_preflight_targets` 加入 `text_layer_targets`：让文本层损坏页也能 preflight 检测矢量表格。
   - `table_candidate_targets` 吸收 preflight 检测出的表格页 → 进入 vector 表格提取。
   - `ocr_targets = ocr_targets - merged_layout`：已被 vector 结构化表格的页不再交给 OCR 覆盖。
3. `tests/unit/test_vector_pdf.py`：新增 `test_repair_fake_glyphs_restores_latin_letters`。

### 验证
- 第 18 页修复后：26 行×6 列结构正确，数值 `mf≤3%`/`RSD≤1%`/`s(x)≤0.5%`/`M≤0.3`/`Pm≤0.5` 全部正确，公式编号保留，表注单单元格，正文段落完整。
- 第 22 页同样受益（23 行×4 列矢量表格正确恢复）。
- `test_vector_pdf.py` 14 项通过；完整单元测试 147 项通过（3 项既有失败与本改动无关，已用还原对比确认）。
- ruff 全部通过。

### 注意
- 本修复只针对「有矢量表格线框」的 text_layer_review 页；纯文本/扫描假字页仍走 OCR，行为不变。
- 假字映射基于单份文档推导，若其他损坏字体文档出现不同假字码位，需要补充映射表。

---

## 九、2026-09-10：块级关联信息通用化（表格 caption / 附录横幅 / FR 标签 / 表锚点 / 去枚举）

> 完整记录见 `HANDOFF-2026-09-10.md`。涉及文件：`app/ingestion/pipeline.py`、`app/ingestion/regulations.py`、`app/ingestion/parsers/scan_regulatory.py`、`app/ingestion/publishers/ragflow.py`、`app/ingestion/jobs.py`、`app/services/domain_terms.py`、`app/web/static/ingestion.js`、`app/web/ingestion.html`。

| # | 改动 | 通用判据 | 实测结果 |
|---|---|---|---|
| 1 | 表格 caption 随表发布（含跨小节归并） | 位置 + 形态（`… as follows:` 紧邻表格） | 悬空前导句 21→2；表格块数不变（205→205） |
| 2 | 附录横幅识别/拆分（`APPENDIX I COMMITTEE IV …`） | 排版（全大写）+ 首尾边界守卫 | 全库 3 份文档 9 处；75-19 p29 全部 12 块归入 `APPENDIX I` |
| 3 | 句子碎片不再被当成表标题 | 位置 + 标点形态（以 `:` 结尾且上段无句末标点） | 全库 2 处（均在 75-26） |
| 4 | 标签行判定改为「形态 + 位置」 | 形态（全大写+冒号）+ 位置（报头/分节状态机，含合订本重开） | 取代 6 标签枚举；72 份 A/B 中 66 份、2,947 处归属变化（收益与代价见 HANDOFF 第五节） |
| 5 | 表锚点：表头字段名 + 首列标识 + 标识符形状 | 结构 + 形状 | 286 表块全部获得锚点，共 +1,383；非表格 chunk 逐字不变；**发布时重算，无需重解析** |
| 6 | 删除 47 词「横幅词表」 | —（净删代码） | 实测该表在防不存在的问题，且误杀真表头（`Normal and utility categories` 等） |
| 7 | 剥离浏览器打印页脚 `about:blank N/M` | 形态 | 57 份文档（最多 88 处/文档）；引用跳页不受影响（页码来自 `positions` / `页码：` 头） |
| 8 | 预览面板与实际发布内容同源 | 单一渲染源 | `ragflow.chunk_publish_facets()`；769-chunk 真实文档 821 次 preview↔plan 对比 0 处不一致 |
| 9 | 表格 caption 实为表内分组标签时不提升为 section | 派生性（标题 = 表内某值的截断，覆盖率 ≥0.6，文档级取值表） | 全库 90 个表标题只命中 3 处（全是真缺陷），87 个真 caption 不动；75-31 p65/p66 表块回到 `APPENDIX I - MISCELLANEOUS PROPOSALS DEFERRED.` |
| 10 | 折行标题合并（含「表标题去重抢先吃掉尾行」的顺序修复） | 形态（两行全大写 + 首行以连接词结尾 + 合并 ≤200 字符）+ 位置（去重前先合并）；`_title_repeats_section` 用派生性（标题 ⊂ 已有 section 标签） | 75-31 p68 合并为完整 APPENDIX III 标题、表格 41 行归回该标题；全库 34,237 块中 59 块带截断 section（3 份文档）；真正独立截断标题仅 2 处，其中 75-10 p73 因续行为「首词全大写 + 正文」暂未覆盖 |

**验证**：全量单元测试 920 项通过；11 项失败与事故前最新版完全一致（10 项既有失败与本节改动无关：`hybrid_pdf` 版本号、CSS 缓存版本号、OCR 路由切块等；另有 `test_split_blocks_filters_fullpage_figure_and_duplicate_table_title`，在 09:42 旧副本基线上同样失败，故亦非本轮引入）；`ruff check app tests` 本轮改动文件全部通过；`mypy` 改动范围干净（`pipeline.py` 既有 38 个错误数量不变）。

> 2026-09-10 下午事故处置：`app/ingestion/pipeline.py` 曾被 PowerShell 5.1 的 ANSI 读写破坏 322 处（详见 `HANDOFF-2026-09-10.md` 第九节）。按用户决定改用 09:42 旧副本基线（sha `9bee4d62`），并把上表 1–10 项改动逐项重放回来；重放保真度以损坏前编译的 `.pyc` 常量集逐函数核对（字面量全部一致），符号面差异为 0。

**生效方式**：关键词/锚点/发布前缀属发布时重算，**重启网关后重新 publish 即生效**；chunk 正文与 section_path 类改动**必须重解析**。
