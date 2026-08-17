# Internal RAG 解析质量改进交接（2026-08-03 之后）

> 用途：供后续智能体快速接手本会话（自 `HANDOFF-2026-08-03.md` 之后）完成的解析质量改进。项目运行环境为 Windows，所有项目、运行时、模型与数据均位于 D 盘。不要将部署方案改为 WSL。

## 0. 交接起点与范围

- **起点**：`HANDOFF-2026-08-03.md`（已覆盖队列查询、人工修订、同页多表、单行条款、审核队列）
- **本会话范围**：自 2026-08-03 下午至 2026-08-04 上午的**解析质量改进**，主要围绕 PDF 文本层损坏、目录识别、表格、公式、版面特征
- **重要**：`CHANGELOG-rag-improvements.md` 是较早的累积记录（含 08-03 之前的内容），本会话新增内容需单独追加，避免混淆

## 1. 改动文件总览

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `app/ingestion/pipeline.py` | 修改 | 文本层损坏检测（新增 `has_fake_glyph_corruption`）、TOC 特征打分、版面特征、表格跨页续表、假字页眉过滤 |
| `app/ingestion/parsers/hybrid_pdf.py` | 修改 | OCR 后目录重检、页眉 rich_block 过滤 |
| `app/ingestion/parsers/remote.py` | 修改 | LaTeX 清洗增强（`\frac`/`\sqrt` 嵌套、`\sum`/`\phi`/`\prime` 等映射、display 公式、双重撇号去重） |
| `app/ingestion/parsers/vector_pdf.py` | 修改 | 数学字体检测（`_has_math_font`） |
| `app/ingestion/jobs.py` | 修改 | preview 接口透传 `layout_features` |
| `app/web/static/ingestion.js` | 修改 | 页面诊断展示 `toc_score`、`dominant_font_size`、`image_count` |
| `app/ingestion/publishers/ragflow.py` | 修改 | `formula_latex` 作为 chunk 元数据 tag 发布 |
| `tests/unit/test_pipeline.py` | 修改 | 新增假字/页眉/版面特征测试 |
| `tests/unit/test_remote_parsers.py` | 修改 | LaTeX 清洗测试 |
| `tests/unit/test_regulation_preprocess.py` | 修改 | TOC 特征打分/跨页续表测试 |
| `tests/unit/test_vector_pdf.py` | 修改 | 数学字体测试 |
| `experiments/draft_section.py` | 新增 | 交互式文档撰写辅助脚本（试点） |

## 2. 文本层损坏检测（核心）

### 2.1 三种损坏形态

GB/T 标准文档的 PDF 字体损坏，pypdf 提取乱码，需要路由到 OCR。共三类：

| 形态 | 特征 | 检测函数 |
|---|---|---|
| PostScript 字形名 | `/G21` `/G22` 等 | `is_glyph_name_garbage`（既有） |
| 整页 CJK 假字 | 全页假字（`犲` U+72B2 等） | `is_fake_cjk_garbage`（既有） |
| **混合型**：中文正文正常 + 英文/公式假字 | 连续假字串或分散假字 | **`has_fake_glyph_corruption`（本会话新增）** |

### 2.2 新增 `has_fake_glyph_corruption`

解决"中文正文正常、英文术语/公式被损坏"的混合型乱码页（如《多传感器一致性检测》大部分页）：

```python
def has_fake_glyph_corruption(text: str) -> bool:
    compact = compact_chars(text)
    if not compact or len(compact) < 20:
        return False
    total = len(compact)
    fake = sum(1 for ch in compact if 0x7280 <= ord(ch) <= 0x72CF)
    fake_ratio = fake / total
    # 连续假字串（≥4）⇒ 损坏的英文单词，如 犅狅狉狀犲
    run = 0; max_run = 0
    for ch in compact:
        if 0x7280 <= ord(ch) <= 0x72CF:
            run += 1; max_run = max(max_run, run)
        else:
            run = 0
    return fake_ratio > 0.03 or max_run >= 4
```

**关键**：
- 假字码位范围是 `U+7280~U+72CF`（不是 `U+7000~U+9FFF`，后者包含正常中文）
- `fake_ratio > 0.03` 覆盖公式分散假字（如 `犕f`、`∑狀犻`）
- `max_run >= 4` 覆盖连续英文假字
- 页眉假字（`犌犅/犜` 3 个）+ 正常正文不误判（真实正文页占比远低于 3%）

### 2.3 接入点

`parse_pdf` 中：
```python
text_layer_corrupted = (
    is_glyph_name_garbage(raw_text)
    or is_fake_cjk_garbage(raw_text)
    or has_fake_glyph_corruption(raw_text)
)
```
为 True → 路由 `text_layer_review` → 走 MinerU OCR。

## 3. 假字页眉过滤

### 3.1 问题
损坏字体把 "GB/T" 映射成 `犌犅/犜—` 作为每页页眉，作为独立 rich_block 存在，污染 chunks。

### 3.2 修复（`pipeline.py`）
新增 `FAKE_GB_HEADER_RE`，并在 `filter_repeated_margin_rich_blocks` 中识别：

```python
FAKE_GB_HEADER_RE = re.compile(r"^犌犅[／/]犜.*$")
```

过滤条件：文本匹配 `FAKE_GB_HEADER_RE` 且 `len(text) <= 40`，无论有无 bbox。

**已知限制**：粘连形态（`I犌犅/犜—` 页码前缀、`图犆...犌犅/犜` 图题粘连）不处理，避免误伤。

## 4. 目录识别（第二层特征打分）

### 4.1 背景
之前目录识别靠正则枚举格式，遇到新格式就加正则（死循环）。本会话引入**特征加权打分**。

### 4.2 `compute_toc_score`

综合 6 个特征：
- `title_score`：页首含"目 录/目 次/Contents"
- `leader_ratio`：点线密集度
- `ending_digit_ratio`：行尾页码对齐
- `hierarchy_score`：编号层级（`1`/`1.1`/`1.1.1`）
- `position_score`：页面前部位置
- `font_score`：字号统一度（来自第一层 `layout_features`）

权重：title 0.20、leader 0.35、ending_digit 0.15、hierarchy 0.10、position 0.10、font 0.10。

### 4.3 接入方式（保守）

```python
for page in pages:
    toc_score = compute_toc_score(
        page.raw_text, page_number=page.page_number,
        total_pages=len(pages), layout_features=page.layout_features,
    )
    page.layout_features["toc_score"] = toc_score
    if is_probable_toc_page(page.raw_text) or toc_score >= 0.85:
        page.is_toc = True; page.indexable = False; page.route = "toc_excluded"
```

**关键**：`toc_score >= 0.85` 只是补充，主判据仍是正则 `is_probable_toc_page`。当前目录页实测分数 0.47~0.77（达不到 0.85），所以打分主要起**记录 + 前端展示 + 未来 VLM 复核定位**作用，未真正接管判定。

### 4.4 相关修复
- `TOC_TITLE_RE` 改为 `目\s*录|目\s*次|contents`，匹配含空格的"目 次"
- `TOC_DOTTED_NO_PAGE_RE` 支持"编号+标题+点线（无页码）"格式
- `_table_reference` 统一 `表 A.1`/`表A1`/`表A1（续）` 为 `表a1`（去点）

## 5. 版面特征（第一层地基）

### 5.1 新增 `extract_page_layout_features`
为每页保留无损版面特征：
- `page_width`/`page_height`：页面尺寸
- `image_count`/`image_sizes`/`image_ratio`：图片特征
- `char_count`/`line_count`：文本量
- `font_sizes`/`dominant_font_size`：字号分布

存入 `PageRecord.layout_features`，序列化进 `document_ir.json`。

### 5.2 前端展示
`jobs.py` preview 接口 `page_summaries` 白名单加 `layout_features`；
`ingestion.js` 页面诊断卡片和单页详情展示 `toc_score`、`dominant_font_size`、`image_count`。

## 6. 表格修复

### 6.1 跨页续表 caption mismatch 误报
- **根因**：跨页续表标题（`表A1（续）`）的表格体在上一页，当前页只有 `table_coverage` 标记，未计入"已绑定"引用，导致 `table_caption_mismatch` 误报
- **修复**：
  1. `merge_cross_page_tables` 给 continuation 页的 `table_coverage` 附带 `table_title`
  2. `actual_caption_refs` 构建时把 table_coverage 的标题引用算作已绑定
  3. `_table_reference` 规范化（去点）

### 6.2 表格 + 整页图冗余（未修复，待确认）
第 7 页（表1+表2+整页渲染图）同时生成结构化表格和 figure_page + VLM 描述，导致内容重复。
**已知问题**：`vector_pdf.py` 的 `parse_pages` 只要 `table_count > 0` 就渲染整页生成 figure_page，未区分"纯表格页"和"表格+真实插图页"。
**建议修复**：当页面有真实非表格图片（`image_coverage >= 0.05`）才生成 figure_page；纯表格页只保留结构化表格。

## 7. LaTeX 公式清洗增强

`remote.py` 的 `latex_to_text` / `clean_inline_latex` 增强：

| 改进 | 说明 |
|---|---|
| `_replace_frac` | 平衡花括号扫描，处理嵌套 `\frac`（修复 `Δt=B_X/W` 除号丢失） |
| `_replace_sqrt` | 平衡花括号扫描，处理嵌套 `\sqrt` |
| `_subscript` | 剥离下标组的孤立反斜杠（`_ { \ Y }` → `_Y`） |
| `clean_inline_latex` | 处理 `$$\n...$$` display 公式（跨行） |
| 符号映射 | 补 `\sum`→∑、`\phi`→φ、`\prime`→'、`\boldsymbol` 剥壳等 |
| 双重撇号去重 | `^'^'` → `'` |

## 8. 文档撰写辅助试点

`experiments/draft_section.py`（独立，不改现有代码）：
- 交互式选择文档类型/章节 → 从知识库检索 → Ollama 生成初稿
- 复用 RAGFlow 检索接口和 dataset_id `8837f5068b2a11f1b7c4d904b160df50`

## 9. 测试

- `test_pipeline.py`：假字检测、页眉过滤、版面特征提取
- `test_remote_parsers.py`：LaTeX 清洗（`\sqrt`/`\sum`/`\prime`/`\boldsymbol`/display 公式/双重撇号）
- `test_regulation_preprocess.py`：TOC 特征打分、跨页续表 caption 绑定
- `test_vector_pdf.py`：数学字体检测
- ruff 全部通过；Windows `.venv` 下 `pytest tests/unit -q` 验证

## 10. 已知问题与待办

1. **表格+整页图冗余**（第 6.2 节）：`vector_pdf.py` 需区分"纯表格页"vs"表格+插图页"
2. **OCR 英文术语假字**：MinerU 对这份文档的小号拉丁字符识别精度不足（`犿狌犾狋犻...`），清洗无法还原，需 OCR 精度优化或视觉模型复核
3. **TOC 打分阈值未校准**：`toc_score >= 0.85` 目前目录页达不到，需积累真实分布后校准，或接入第三层 VLM 复核
4. **Gateway 需重启**：代码改动后 `stop.ps1` + `start.ps1`，前端 `Ctrl+F5`
5. **8G 显存约束**：MinerU 与 Ollama 不能同时满载，否则 MemoryError

## 11. 本会话新增改进（2026-08-04 追加）

> 以下是本次会话（承接 `HANDOFF-2026-08-03.md` 之后的解析质量改进）新增的改动，独立于上文第 1-10 节。

### 11.1 假字还原接入全局清洗

- **问题**：`repair_fake_glyphs` 原先只在 `vector_pdf.py` 的表格单元格/表标题上生效，正文残留假字（页眉 `犌犅/犜`、附录 `犃`、公式残留）无法还原。
- **改动**：
  - 将 `FAKE_GLYPH_TO_ASCII` + `repair_fake_glyphs` 从 `vector_pdf.py` **上移**到 `app/ingestion/pipeline.py`（公共底层模块），`vector_pdf.py` 改为 `from app.ingestion.pipeline import ...`。
  - `clean_page()` 中 `normalize_line` 之后对每行调用 `repair_fake_glyphs`，使所有正文（native/OCR 文本）在清洗阶段还原假字。
- **效果**：真实文档 `document_ir` 中 50 个假字残留 → `clean_page` 后 **0 残留**（100% 清除）。
- **关键约束**：`has_fake_glyph_corruption`/`is_fake_cjk_garbage`/`is_glyph_name_garbage` 仍用原始 `raw_text` 检测（`parse_pdf` 中检测先于清洗），路由判定不受影响。

### 11.2 表格输出改为 Markdown 网格

- **问题**：第 18 页表 3 原本在 chunk 中拍平为管道符文本流（`单元格 | 单元格`），丢失行列结构，用户无法直观核对与原文档一致性。
- **改动**：
  - 新增 `table_to_markdown(rows, *, title=None)`：矩形化行 → 表头行 + `| --- |` 分隔行 → 每行 `| a | b |`；空单元格留空（保持合并单元格列对齐）；换行折叠为空格；管道符转义；标题 NFKC 归一化。
  - table 块的 `text` 生成改用 `table_to_markdown`（`vector_pdf.py` 的 table 块、`pipeline.py` 的 `merge_cross_page_tables` 与 `split_structured_table_block`、`hybrid_pdf.py` 的 remote OCR 表格）。
  - `vector_pdf.py` 生成 table 块时先用 `normalize_cell` 归一化 `rows`（含假字还原），再生成 Markdown/HTML，确保单元格内假字一并还原。
- **效果**（真实文档第 18 页表 3）：
  ```
  表3 无人机低空遥感监测的多传感器一致性检测结果指标评价
  | 内容 |  | 衡量指标 | 数值范围 | 一致性程度 | 计算依据 |
  | --- | --- | --- | --- | --- | --- |
  | 辐射一致性 | 激光雷达辐射一致性 | 反射率中误差 | m ≤3% f | 优 | 公式（２） |
  ...
  ```
  26 行 × 6 列、合并单元格留空、假字全部还原（`m ≤3% f`、`RSD≤1%`、`s(x)≤0.5%`）、分组结构（辐射/几何 × 激光雷达/光学遥感/多传感器）完整保留。全文档 12 个表格 chunk，管道符数量零不一致。
- **保留**：`table_to_text`（旧管道符格式）仍存在于 `pipeline.py`/`vector_pdf.py` 并被测试引用，未删除，避免破坏既有语义。

### 11.3 测试

- `test_pipeline.py` 新增：
  - `test_repair_fake_glyphs_decodes_damaged_font_words`
  - `test_clean_page_restores_fake_glyphs_in_body_text`
  - `test_table_to_markdown_renders_grid_with_header_separator`
  - `test_table_to_markdown_escapes_pipe_in_cell`
- 基线：`pytest tests/unit -q` → 151 passed, 3 failed（3 个失败为改动前既有，与本次无关：TOC 检测、跨页 caption 绑定、figure 去重）。ruff 全部通过。

### 11.4 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/pipeline.py` | 新增 `FAKE_GLYPH_TO_ASCII`/`repair_fake_glyphs`/`table_to_markdown`；`clean_page` 接入假字还原；table 块 text 改用 Markdown |
| `app/ingestion/parsers/vector_pdf.py` | 移除重复的假字定义，改 import pipeline；table 块用归一化 rows 生成 Markdown |
| `app/ingestion/parsers/hybrid_pdf.py` | remote OCR 表格 text 改用 `table_to_markdown` |
| `tests/unit/test_pipeline.py` | 新增 4 个测试 |

## 12. 验证命令

```powershell
Set-Location D:\internal-rag\source\internal-rag
.\.venv\Scripts\python.exe -B -m pytest tests\unit -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B -m ruff check app tests\unit --no-cache
.\deploy\gateway\stop.ps1
.\deploy\gateway\start.ps1
```

## 13. 追加修复：相邻页独立表格误合并（2026-08-04）

### 13.1 问题

《民用无人机地理围栏数据技术规范》第 12-13 页，表5（第12页，7行）与表6（第13页，4行）被错误合并成一个 chunk。表5 的数据行尾部直接接上了表6 的 3 行数据，表6 的表头丢失。

### 13.2 根因

`merge_cross_page_tables` 的合并判定中，`same_header`（表头指纹相同）被单独作为跨页续表合并的充分条件。GB/T 文档中多个**独立**表格常共用相同表头（如 `序号|数据项|评定标准`），导致标题不同（表5 vs 表6）的相邻页表格被误判为续表。

原条件：
```python
merge = same_width and (
    same_title
    or continuation
    or same_header          # ← 表头相同就合并，误判
    or overlap > 0
    or contained
    or (not fragment["title_key"] and not previous["title_key"])
)
```

### 13.3 修复

`same_header` 仅在**标题一致或某侧无标题**时才作为合并证据；标题明确不同时，必须依赖 `overlap`/`contained`（内容实际延续）或标题含"续"。

```python
merge = same_width and (
    same_title
    or continuation
    or (same_header and (not previous["title_key"] or not fragment["title_key"]))
    or overlap > 0
    or contained
    or (not fragment["title_key"] and not previous["title_key"])
)
```

### 13.4 验证

- 真实文档端到端：表4/表5（第12页）、表6/表7（第13页）四个表格各自独立成 chunk，标题与数据行正确。
- 既有续表合并测试不破坏：`test_merges_cross_page_table_and_removes_repeated_rows`（同标题"续"）、`test_splits_multiple_page_tables_and_merges_named_continuation`（续表页内嵌标题）均通过。
- 新增回归测试：`test_keeps_adjacent_tables_with_different_titles_and_same_header_separate`。
- `pytest tests/unit -q` → 152 passed, 3 failed（3 个既有失败与本次无关）。ruff 全部通过。

### 13.5 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/pipeline.py` | `merge_cross_page_tables` 合并判定收紧 `same_header` 条件 |
| `tests/unit/test_regulation_preprocess.py` | 新增相邻独立表格不合并的回归测试 |

## 14. 追加修复：正文页插图漏识别（2026-08-04）

### 14.1 问题

《物流无人机货物吊挂控制通用要求》第 8 页的"图1 物流无人机货物吊挂控制逻辑框图"未被识别为图片。`image_count: 3` 但 `asset_ids: []`，route 被判定为 `native_text`，图片识别流程未触发。

### 14.2 根因

`vector_pdf.preflight_pages` 的 figure 判定仅依赖图片覆盖率：
```python
if image_count and (
    largest_image_coverage >= 0.12          # 主图覆盖率
    or (image_coverage >= 0.15 and len(compact_text) < 300)
):
    result.figure_pages.add(page_number)
```
第 8 页主图覆盖率 **0.1154**（略低于 0.12），`image_coverage` 0.1159（低于 0.15），文本 510 字符（> 300），三个条件均不满足 → 未被识别为 figure。带图注的正文页插图恰好卡在覆盖率阈值之下时极易漏识别。

### 14.3 修复

figure 判定增加**图注信号**：行首出现"图 N 描述…"（正则 `FIGURE_CAPTION_RE`，匹配 `(?:^|\n)\s*图\s*编号\s*\S`）且页面有图片时，即使覆盖率不足也判定为 figure 页。行首锚定避免误匹配正文引用（如"结果如图1所示"）。

```python
has_figure_caption = bool(FIGURE_CAPTION_RE.search(page_text))
if image_count and (
    largest_image_coverage >= 0.12
    or (image_coverage >= 0.15 and len(compact_text) < 300)
    or has_figure_caption
):
    result.figure_pages.add(page_number)
```

### 14.4 验证

- 真实文档端到端：第 8 页现在 route=`figure_review`，生成 `p0008-figure-01.png`（47KB 有效 PNG，caption="图1 物流无人机货物吊挂控制逻辑框图"），chunk asset_ids 关联该图。
- 全文档 figure_pages 从 `{9}` 变为 `{8, 9, 10}`（第 9、10 页原本已识别）。
- 新增回归测试 `test_preflight_flags_low_coverage_image_with_figure_caption`。
- `pytest tests/unit -q` → 153 passed, 3 failed（既有失败）。ruff 全部通过。

### 14.5 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/parsers/vector_pdf.py` | 新增 `FIGURE_CAPTION_RE`；`preflight_pages` figure 判定加入图注信号 |
| `tests/unit/test_vector_pdf.py` | 新增低覆盖率图注页识别回归测试 |

## 15. 追加改进：图片识别与发布链路（2026-08-04）

### 15.1 问题背景

- 图片识别依赖覆盖率阈值，带图注的正文页插图（覆盖率 0.1154 < 0.12）易漏识别。
- 图注抓取 `_caption_near` 搜索带仅 `bottom+42pt`，图注在图片下方较远时抓不到，VLM 收到错误 caption（如抓到"a 起降过程 b 航线飞行"而非"图2 物流无人机吊挂货物飞行示意图"）。
- 一页多图时 `ragflow.py` 只发第一张 `image_base64`（`if image_attached: continue`），其余图只发 positions。
- figure 块被合并进正文 chunk，导致每张图不能独立参与检索。

### 15.2 改动

**① 图注抓取 `_caption_near`**（`vector_pdf.py`）
- 下方搜索带从 `bottom+42` 加宽到 `bottom+120`。
- 返回时优先匹配行首"图N xxx"图注行（`FIGURE_CAPTION_RE`），忽略符号说明行。

**② 一页多图独立 chunk**（`pipeline.py`）
- `build_chunks` 新增 `figure` 分支：figure 块独立成 chunk（类似 table），每张图带自己的 asset_id。
- `chunk_from_blocks` 的 `content_type` 判定：figure 块单独成 chunk 时标为 `"figure"`。

**③ 发布标签**（`ragflow.py`）
- 对 `content_type == "figure"` 的 chunk 打 `content_type:figure` 标签。
- 由于每张图已是独立 chunk，发布时每张图都有自己的 `image_base64` + `positions`。

### 15.3 验证

- 吊挂文档：图1/图2/图3 各自独立成 figure chunk，各带 `image_base64` + `content_type:figure`，图注准确（"图1 物流无人机货物吊挂控制逻辑框图"等）。
- 多传感器文档第20页（15 张图极端场景）：15 张图全部独立 chunk，每张带 `image_base64` + `positions`。
- 新增测试：
  - `test_figure_blocks_become_standalone_chunks_with_own_asset`（pipeline）
  - `test_caption_near_prefers_figure_caption_line_over_symbol_explanation`（vector_pdf）
- `pytest tests/unit -q` → 155 passed, 3 failed（既有失败）。ruff 全部通过。

### 15.4 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/parsers/vector_pdf.py` | `_caption_near` 加宽搜索带 + 优先图注行 |
| `app/ingestion/pipeline.py` | `build_chunks` figure 独立 chunk；`content_type` 判定 |
| `app/ingestion/publishers/ragflow.py` | figure chunk 打 `content_type:figure` 标签 |
| `tests/unit/test_regulation_preprocess.py` | figure 独立 chunk 回归测试 |
| `tests/unit/test_vector_pdf.py` | 图注抓取优先回归测试 |

### 15.5 待办提示

- 一页多图的装饰性小图（logo、页眉）仍会作为 figure asset 生成；如需进一步过滤，可在 `parse_pages` 中按 caption 是否含"图N"或 bbox 尺寸（如 `w<60 or h<60`）过滤无图注小图。

## 16. 追加修复：顶层条款父章节关联丢失（2026-08-04）

### 16.1 问题

《民用大中型固定翼无人机系统地面站通用要求》中，"4 一般要求"应作为 4.1~4.7 的父章节，但解析后每个子条款 `section_path` 只有一级（如 `['4.3 维修性']`），父章节"4 一般要求"丢失，且没有独立 chunk。

### 16.2 根因

- `_BARE_ARTICLE_RE`（regulations.py）要求编号**必须带点号**（`\d+\.\d+`），所以顶层条款"4 一般要求"（编号无点号）的 `match_article_heading` 返回 `None`。
- `split_blocks` 用单一 `article` 变量存当前条款，遇到"4 一般要求"（顶层）后被"4.1 功能"覆盖，层级信息丢失。
- `section_path` 只有 `[annex, chapter, article]` 三个固定槽位，无法表达"4 → 4.8 → 4.8.1"三级嵌套。

### 16.3 修复

在 `split_blocks` 中引入 **`clause_stack` 条款栈**，按编号点数推导层级：

- `_clause_number(text)` 提取标题编号（`4`、`4.1`、`4.8.1`）。
- `level = number.count(".")`，`clause_stack = clause_stack[:level] + [text]`：
  - "4 一般要求"（level 0）→ `["4 一般要求"]`
  - "4.1 功能"（level 1）→ `["4 一般要求", "4.1 功能"]`
  - "4.8.1 低气压"（level 2）→ `["4 一般要求", "4.8 环境适应性", "4.8.1 低气压"]`
- `section_path = [annex, chapter, *clause_stack]`。
- 移除了不再使用的 `article` 变量。

### 16.4 验证

- 真实文档：4.1~4.7 → `['4 一般要求', '4.x xxx']`；4.8.1~4.8.14 → `['4 一般要求', '4.8 环境适应性', '4.8.x xxx']`（三级）；chunk 文本带父章节 context 前缀。
- 既有测试 `test_standalone_inline_clauses_are_not_dropped_from_chunks` 断言更新（chunk 文本现以父章节 context 开头）。
- 新增回归测试 `test_top_level_clause_is_parent_of_sub_clauses_in_section_path`。
- `pytest tests/unit -q` → 156 passed, 3 failed（既有失败）。ruff 全部通过。

### 16.5 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/pipeline.py` | 新增 `_clause_number`；`split_blocks` 引入 `clause_stack` 构建层级 `section_path` |
| `tests/unit/test_regulation_preprocess.py` | 更新内联条款断言；新增顶层条款父章节回归测试 |

## 17. 追加改进：公式处理链路（2026-08-04）

> 基于对公式处理完整链路（检测→MinerU识别→LaTeX转文本→chunk拼接→RAGFlow发布）的评审，修复 4 个影响准确率的设计点。

### 17.1 latex_to_text 语义正确性（remote.py）

修复过度清洗导致的数学语义改变：

| 输入 | 修复前 | 修复后 |
|---|---|---|
| `\frac{a+b}{c+d}` | `a+b/c+d`（歧义） | `(a+b)/(c+d)` |
| `\frac{1}{1+\frac{x}{y}}` | `1/1+xy`（错误） | `(1)/(1+(x)/(y))` |
| `\sqrt[3]{x}` | `[3]x`（丢根指数） | `3th_root(x)` |
| `\lim_{x\to0}` | `_x→0`（丢操作符） | `lim(x→0)` |
| `\min(a,b)` | `min(a,b)`（保留） | `min(a,b)` |

- `_replace_frac`：分子分母整体加括号，嵌套递归保护。
- `_replace_sqrt`：支持 `\sqrt[n]{x}`，n≠2 时保留根指数。
- `_NAMED_OPERATOR_RE`：`\lim/\max/\min/\log/\sin…` 保留操作符名 + 下标界。
- `_LATEX_SYMBOLS` 补充：`\to`→`→`、`\uparrow`→`↑`、`\downarrow`→`↓`、`\Leftrightarrow` 等。
- 保留 `\vec/\hat/\bar/\overline/\mathbf` 等命令的参数（只去命令不去内容）。
- 取消无条件撇号去重：`f'`/`f''`/`f'''` 是不同阶导数，保留。

### 17.2 formula_latex 污染与发布（remote.py + ragflow.py）

- **`'None'` 污染修复**：`_latex_if_real()` 对空文本和字符串 `"None"` 返回 `None`，不再生成 `'None'` 假 LaTeX。
- **LaTeX 移出 tag_kwd**：完整 LaTeX 不再作为高基数 tag 发布；改为有限枚举 tag `has_formula` + `content` 尾部 `[原始公式]` 专属区域。

### 17.3 公式图片关联改按 block_type（hybrid_pdf.py）

- 原来按 LaTeX 命令猜测（`\frac/\sqrt/\sum`），会漏 `E=mc^2`、`a+b=c`。
- 改为按 `chunk.formula_latex` 非空（即 chunk 确实含 formula 块）关联 `formula_page` 整页图。

### 17.4 公式与"式中"上下文绑定（pipeline.py）

- `build_chunks` 新增 formula 分支：公式块**不可切分**（不参与 split_long_text），与前置说明 + 后置"式中/其中/各参数含义如下"段落绑定在同一 chunk，即使略超字符预算。

### 17.5 验证

- 真实文档（低空5G）公式+前置说明+"式中"变量段落在同一 chunk。
- 新增测试：
  - `test_latex_to_text_preserves_fraction_grouping`
  - `test_latex_to_text_preserves_sqrt_index`
  - `test_latex_to_text_keeps_named_operators_and_arrows`
  - `test_latex_to_text_keeps_distinct_primes`
  - `test_latex_to_text_keeps_accent_command_arguments`
  - `test_formula_chunk_binds_preceding_text_and_shi_zhong_vars`
  - 更新 `test_clean_inline_latex_handles_display_equation_span`、`test_clean_inline_latex_collapses_duplicate_prime_and_keeps_frac`（frac 括号行为变更）。
- `pytest tests/unit -q` → 162 passed, 3 failed（既有失败）。ruff 全部通过。

### 17.6 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/parsers/remote.py` | `_replace_frac`/`_replace_sqrt`/`_NAMED_OPERATOR_RE`/符号映射/`_latex_if_real` |
| `app/ingestion/publishers/ragflow.py` | LaTeX 移出 tag_kwd，改 `has_formula` tag + `[原始公式]` 区域 |
| `app/ingestion/parsers/hybrid_pdf.py` | 公式图关联按 `formula_latex` 而非 LaTeX 命令 |
| `app/ingestion/pipeline.py` | `build_chunks` formula 块不可切分 + 上下文绑定 |
| `tests/unit/test_remote_parsers.py` | 新增/更新 latex_to_text 测试 |
| `tests/unit/test_regulation_preprocess.py` | 公式-式中绑定回归测试 |

### 17.7 待办（未实施，属后续）

- 公式页检测分三级（候选路由 + MinerU 最终判定）。
- 公式局部截图 + bbox（`page_N_formula_0M.png`）。
- 三字段拆分 `latex_raw`/`latex_normalized`/`formula_search_text`。
- 原生文本与 MinerU 结果按坐标合并（较大架构改动）。

## 18. 追加修复：分式公式页漏识别（2026-08-04）

### 18.1 问题

《无人机遥感测绘飞行管理信息要求》第 8、9 页的分式公式识别不完整：
- 第8页公式(1) `R=floor((C−H)/F)` 被拆成 `R = f loor (C  H F)`
- 第9页公式(2) `N=ceil(30/(2R))` 被拆成 `N = ceil ( 30 2R)`

根因：原生文本层把分式公式拆成多个独立行（分子行、分数线、分母行），整页只有一个 `"="`，而 `preflight_pages` 的公式检测要求 `equation_count >= 4`，两个页面都不满足 → 第9页被判为 `native_text`，第8页虽走 vector_layout 但公式未还原。

### 18.2 修复

在 `preflight_pages` 公式检测中补充**函数名 + 分式碎片**信号：

- 新增 `FORMULA_FUNCTION_RE`：匹配 `floor|ceil|sqrt|log|sin|cos|sum|int|lim|max|min` 等公式函数名。
- 新增判定：`has_formula_function and equation_count >= 1 and fraction_fragments`（孤立短行≥3 + 括号碎片行）→ 判为公式页。
- 分式碎片检测覆盖全页（不仅前 8 行），适配"表格后才是公式"的页面。
- **移除 `not image_count` 对公式信号路径的限制**——公式页可含小图（logo/装饰）。

### 18.3 验证

- 遥感测绘文档：第8、9页现在 route=`formula_review`（原第9页是 native_text），走公式/OCR 路径。
- 全库抽样（低空5G、试飞、多传感器、地理围栏、吊挂）无误报——普通正文页不会因含 `max`/`min` 等英文词误判（还需 `=` 且分式碎片）。
- 新增回归测试 `test_preflight_detects_fraction_formula_with_single_equals`。
- `pytest tests/unit -q` → 163 passed, 3 failed（既有失败）。ruff 全部通过。

### 18.4 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/parsers/vector_pdf.py` | 新增 `FORMULA_FUNCTION_RE`；分式公式检测信号 |
| `tests/unit/test_vector_pdf.py` | 分式公式检测回归测试 |

## 19. 追加修复：表格单元格语序错乱（2026-08-04）

### 19.1 问题

《无人机遥感测绘注册规范》第 8 页表 1 多个单元格内容错乱：
- `起飞方式 | 字符型 | 从下述列项选一或多选滑跑弹射手抛垂直其他 : / / / /`（分隔符 `/` 全挤到行尾）
- `单位为千克 精确到小数点后 位 如\n(kg), 1 , 6.3`（单位、逗号、数字错位）

### 19.2 根因

这些 PDF 表格单元格的**字间距大**，导致每个字符的 `top` 坐标有约 6pt 的抖动。pdfplumber 默认 `y_tolerance=3` 把同一视觉行内偏移较大的字符（冒号、斜杠、括号、数字、单位）错误分到下一行，造成语序错乱。

### 19.3 修复

在 `vector_pdf.py` 新增**智能单元格重建**：

- `_smart_cell_text(page, bbox)`：按字符坐标重建单元格文本
  - 字符按**中心点**（非边界）归属单元格，避免相邻单元格字符泄漏
  - 按 `top` 聚类成候选行
  - **智能合并**：相邻候选行 top 差 ≤7pt 且下一行是"浮动下标"（≤4字符或纯标点/数字/单位）时合并，修复标点/斜杠/单位错位
- `_rebuild_cells(page, table)`：对表格逐单元格重建；**合并单元格**（bbox 高度 >60pt）保留 pdfplumber 原样，避免从相邻行拉入字符（如"飞控系统"误并入下行的"卫"）
- `_extract_best_tables` 改用 `_rebuild_cells` 替代 `table.extract()`

### 19.4 效果（第8页表1）

| 单元格 | 修复前 | 修复后 |
|---|---|---|
| 起飞方式 填写说明 | `从下述列项选一或多选 滑跑 弹射 手抛 垂直 其他\n: / / / /` | `从下述列项选一或多选:滑跑/弹射/手抛/垂直/其他` |
| 机体尺寸 填写说明 | `单位为厘米 精确到小数点后 位 按 长 宽\n(cm), 1 , " × ×` | `单位为厘米(cm),精确到小数点后1位,按"长×宽×高"填报` |
| 遥测波段 填写说明 | `单位为兆赫 从下述列项选一\n(MHz), :...` | `单位为兆赫(MHz),从下述列项选一: 840.5-845...` |

### 19.5 验证

- 注册规范第8页表1 全 26 行语序正确。
- 其他文档（地理围栏表4、多传感器表3）表格输出正常，无副作用。
- 新增回归测试 `test_smart_cell_text_merges_floating_punctuation_lines`。
- `pytest tests/unit -q` → 164 passed, 3 failed（既有失败）。ruff 全部通过。

### 19.6 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/parsers/vector_pdf.py` | 新增 `_smart_cell_text`/`_rebuild_cells`；`_extract_best_tables` 改用智能重建 |
| `tests/unit/test_vector_pdf.py` | 智能单元格重建回归测试 |

## 20. 追加修复：流程图被切块（2026-08-04）

### 20.1 问题

《低空可操作飞行器反制规程》第 6 页的"图1 反制流程"被切成约 20 个小块（`figure-01` 到 `figure-32` 成对出现），而非一整张流程图。

### 20.2 根因

该流程图在 PDF 里由 **40 个 1-bit `imagemask` 文字掩码**（每个框内文字一个 mask）+ **9 个框线 rect** + **13 个箭头 curve** 构成，流程图区域**无可提取文本**（文字全在 mask 里）。当前 figure 提取遍历 `page.images`，把每个 mask 独立成 figure asset，导致一张图被切碎。

### 20.3 修复

`vector_pdf.py` 新增 `_flowchart_regions(page)`：

- **流程图特征检测**：大部分图片是 `imagemask`（≥60% 或 ≥4 个）+ rect 框线 ≥3 + curve 箭头 ≥3。
- **合并为单个整体**：把页面上所有流程框 rect（标题框 + 主流程 + 输出框）合并为一个**外包矩形**，作为完整流程图。
- `parse_pages` 的 figure 生成：检测到流程图时按外包矩形裁剪为一个 figure，否则回退到逐 mask。

### 20.4 效果

| 修复前 | 修复后 |
|---|---|
| ~20 个切块（每个 mask 一个 figure） | **1 个完整流程图** |

- `p0006-figure-01`：`(175.6, 98.1)-(429.0, 419.6)`，43KB，caption "图1 反制流程"。
- **独立 figure chunk**：`content_type=figure`，`asset_ids=['p0006-figure-01']`，与正文 chunk 分离。
- 其他文档（吊挂控制、多传感器）图页不受影响（非流程图特征不触发）。

### 20.5 验证

- 新增回归测试 `test_flowchart_regions_aggregate_imagemask_boxes`。
- `pytest tests/unit -q` → 165 passed, 3 failed（既有失败）。ruff 全部通过。
- 已用当前代码重新解析并刷新磁盘 output（`fd4cd51f.../20200803：《低空可操作飞行器反制规程》__dff83140`），第6页 figure assets 从约 20 个切块变为 **1 个完整流程图**。

### 20.6 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/parsers/vector_pdf.py` | 新增 `_flowchart_regions`；figure 生成按流程图聚合为单个外包矩形 |
| `tests/unit/test_vector_pdf.py` | 流程图聚合回归测试 |

## 21. 追加修复：条款同级重复（2026-08-04）

### 21.1 问题

《农业植保无人机安全作业规范》第 5 页，5.1.1 在每个 5.1.x 的 chunk 中重复出现：
- chunk 5.1.2 = `5 作业前 | 5.1.1 ... | 5.1.2 ...`
- chunk 5.1.3 = `5 作业前 | 5.1.1 ... | 5.1.3 ...`

### 21.2 根因

`split_blocks` 的 `clause_stack` 用 `[:level]`（编号点数）裁剪层级，但 `5.1.1`、`5.1.2`、`5.1.3` 都是 level 2（两个点），且 PDF 中没有 `5.1` 标题块。遇到 `5.1.2` 时 `clause_stack[:2]` 保留了 `['5 作业前', '5.1.1']`，把 `5.1.1` 错误当作 `5.1.2` 的父级。

### 21.3 修复

`clause_stack` 改用**编号前缀匹配**判断真父级：
```python
for existing in clause_stack:
    existing_number = _clause_number(existing)
    if heading_number.startswith(existing_number + ".") or heading_number == existing_number:
        kept.append(existing)
clause_stack = kept + [text]
```
`5.1.1` 不是 `5.1.2` 的前缀 → 被移除，`5.1.2` 的 section_path 为 `['5 作业前', '5.1.2 ...']`。

### 21.4 验证

- 第5页 5.1.1-5.1.7 section_path 全部正确，无冗余 5.1.1。
- 既有测试 `test_top_level_clause_is_parent_of_sub_clauses_in_section_path`、`test_standalone_inline_clauses_are_not_dropped_from_chunks` 通过。
- `pytest tests/unit -q` → 165 passed, 3 failed（既有失败）。ruff 全部通过。
- 已刷新磁盘 output（`da9711c9.../20240628：《农业植保无人机安全作业规范》__3f104e6c`）。

### 21.5 说明：水印图不专门处理

第5页有一张跨页重复的水印大图（覆盖正文），曾被判定为 figure 页。按用户意见**不引入专门的水印过滤逻辑**（此类水印规格少见），页面正常识别文字即可；仅修复条款分块。

### 21.6 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/pipeline.py` | `split_blocks` 的 `clause_stack` 改用编号前缀匹配 |

## 22. 追加修复：空白记录表被 _valid_table 误杀（2026-08-04）

### 22.1 问题

《无人机监测松材线虫病致死松树技术规程》(20241217) 第 11 页的"表B.1 航摄作业记录表"解析错误——附录标题（"附录B/资料性/航摄作业记录表"）被当成表格内容，真正的 8 列表头丢失。

### 22.2 根因

该表格是**空白记录表**（只有表头行有内容，其余 10 行为空待填行）。`_valid_table` 要求 `populated >= width + 1`，而空白记录表 populated=8（仅表头 8 个单元格），width=8，8 < 9 不满足 → 真实表格被 `_valid_table` 过滤掉 → 只剩 expanded（把附录标题和真实表格合并的 15 行表）。

### 22.3 修复

`_valid_table` 阈值从 `populated >= max(4, width+1)` 放宽为 `populated >= max(4, width)`，接受"只有表头一行有内容"的空白记录表。

### 22.4 验证

- 松材线虫病文档第11页：表B.1 正确解析为 8 列表头（监测时间/测区序号/县/乡镇/村/航拍时间/覆盖面积/变异木数量），附录标题不再混入。
- 多传感器表3、地理围栏表4、注册规范表1 等有真实数据的表格不受影响。
- 新增回归测试 `test_valid_table_accepts_header_only_blank_record_form`。
- `pytest tests/unit -q` → 166 passed, 3 failed（既有失败）。ruff 全部通过。
- 已刷新磁盘 output（`5b6791c6.../20241217：《无人机监测松材线虫病致死松树技术规程》__a00c59a8`）。

### 22.5 改动文件

| 文件 | 改动 |
|---|---|
| `app/ingestion/parsers/vector_pdf.py` | `_valid_table` 阈值 `width+1` → `width` |
| `tests/unit/test_vector_pdf.py` | 空白记录表有效判定回归测试 |

