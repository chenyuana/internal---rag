# Internal RAG 文档解析方法说明

> 用途：讲解本工程当前（2026-08）PDF 自动解析的整体思路、分层架构与核心流程，
> 供讲解、评审与接手智能体参考。
> 代码入口：`app/ingestion/pipeline.py`（页面路由）、`app/ingestion/parsers/`（解析器）。

---

## 1. 核心思路：混合解析（Hybrid Parsing）

本系统**不要求用户选择解析器**。用户只上传 PDF，系统自动判断**每一页**最合适的
解析路径，再把不同路径的结果合并成一个统一结构（Document IR）。

判断的依据是**文本层质量**与**版面特征**：

- 文本层可靠 → 走 **native**（原生文本提取），保留条款结构、精确字符；
- 有表格/图片/公式 → 走 **vector**（版面分析），用矢量线框恢复表格网格、裁剪图片；
- 文本层损坏 / 扫描件 / 低文本 → 走 **remote**（OCR），用视觉模型识别图像中的文字。

一句话：**能用原生文本就不用 OCR**。OCR 是"文本层损坏时的兜底"，不是"更优方案"。

---

## 2. 总体架构

```
上传 PDF
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ HybridPdfParser.parse(path)  ← 唯一入口（app/ingestion/parsers/hybrid_pdf.py） │
│                                                               │
│  ① NativePdfParser  原生文本提取（pypdf）→ 页面级路由判定        │
│  ② VectorPdfParser  pdfplumber 版面预检 → 表格/图片/公式识别    │
│  ③ MinerU / Docling 远程 OCR（对损坏页）                        │
│  ④ FigureVision VLM 图片描述（可选）                            │
│  ⑤ 合并 → Document IR + Chunks + 质量门禁                       │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
write_document() → baseline.json / document_ir.json / chunks.jsonl /
                   cleaned.md / qa.json / assets/
   │
   ▼
质量门禁 → completed / warning / needs_review / failed
   │
   ▼
（人工抽检后）RAGFlow 幂等发布
```

---

## 3. 第一步：页面级路由（pipeline.py `parse_pdf`）

对每一页先做轻量检查，决定它"可能"需要哪种解析。这是**第一次路由**，
只基于原生文本提取结果：

| 判定条件 | 页面 route | 含义 |
|---|---|---|
| 只有水印 | `watermark_excluded` | 排除，不进索引 |
| 提取文本为空 | `ocr_required` | 扫描件/纯图，必须 OCR |
| 文本极少（低于阈值） | `low_text_review` | 可能是纯图页，交给后续判断 |
| 文本层损坏 | `text_layer_review` | 假字/ToUnicode 冲突/乱码，走 OCR |
| 正常文本 | `native_text` | 原生文本可用 |
| 正常文本 + 表格提示 | `layout_review` | 可能有表格，需版面预检 |

### 文本层损坏检测（`pipeline.py`）

这是本工程反复打磨的核心。文本层"看起来有字、实际是乱码"是法规 PDF 最常见的坑：

- **假字检测**：PostScript 字形名（`/GXX`）、CJK 扩展假字（U+7280 区）、PUA 点号
- **ToUnicode CID 冲突**：同一 CID 映射到两个不同 Unicode 码点 → 映射自相矛盾，
  无法修复，只能 OCR（判断阈值：冲突数 ≥2 且冲突率 ≥0.5%，避免正常文档误报）
- **无效 Unicode**：非法码点，可修复则修复，严重则走 OCR
- **水印页**：整页只有水印文字

同时收集**版面特征**（图片数量/覆盖率、字体、行数、方程数），供后续打分。

> 第二步：TOC 识别。对每页计算 TOC 打分（`compute_toc_score`），目录页标记为
> `toc_excluded`，不进索引。

---

## 4. 第二步：vector 版面预检（vector_pdf.py `preflight_pages`）

native 判定为"需要版面分析"或"文本层损坏但可能含矢量表格"的页，交给
pdfplumber **预检**——只查 PDF 图元（线、矩形、图像对象），不渲染页面，很快。

预检发现三类页面：

### 4.1 表格页（`table_pages`）

- 用 `pdfplumber.find_tables()` / `_extract_best_tables()` 探测**矢量网格**；
- 无竖线表格（仅水平线）用 `_text_column_boundaries` 从文本 x 坐标聚类推断列边界；
- 封面装饰框（ICS/CCS 分类、标准号）会被排除，不当成表格。

### 4.2 图片页（`figure_pages`）

判据（多条件满足其一）：

- 有实质图片且覆盖率高（≥12%）；
- 图片覆盖率低但文本极少；
- 行首有图名（`图N …`）；
- 纯矢量流程图（矩形+箭头，无位图）带图名；
- 带外框文字结构图（如"图1 III 类手册的构成"）。

关键过滤：

- **整页白色背景图**（装饰背景，编码密度极低）不计为实质图片，
  避免纯文字页被误判成图片页；
- **TextLayer 损坏但图片占主导的页**（如图 A2 纯曲线图）保留为图片，
  因为 OCR 对纯图返回空、会丢图。

### 4.3 公式页（`formula_pages`）

- 方程数 ≥4 且版面特征匹配；
- 或含数学函数名（floor/sqrt/…）+ 分式碎片；
- 或使用数学字体（CambriaMath 等）且无表格网格。

---

## 5. 第三步：分路径解析

### 5.1 native 文本清洗（pipeline.py `clean_page` / `split_blocks`）

对 `native_text` 页：

- **清理**：去重复页眉页脚、修假字（`repair_fake_glyphs`）、修无效 Unicode、
  还原符号字体 PUA 数学符号（°′″⌈⌉×Δ…）；
- **分块**（`split_blocks`）：识别条款标题（"第 23.1 条""A23.13"）、
  章节层级（分部/条/款/项）、术语定义、附录结构；
- **生成硬边界**：条款按编号切分，正文并入所属条款，生成 Chunk。

这是法规问答最关键的环节：**条款结构**（`section_path`、`article_id`）、
**条款别名**（`article_aliases`，如"第23.5条"与"23.5"互认）、**引用关联**。

### 5.2 vector 版面解析（vector_pdf.py `parse_pages`）

对表格/图片/公式页：

- **表格**：从矢量网格恢复行列，识别表题（"表 2 …"）、跨页续表合并
  （`merge_cross_page_tables`）、同页多表拆分、单元格归一化，输出
  `table_rows` + `table_html` + 表格裁剪图资产；
- **图片**：按 bbox 裁剪出 figure 资产（`p00xx-figure-01.png`），
  从图片附近提取图名（`_caption_near`，只取真实图名、不混入正文）；
  整页渲染图（`p00xx-page-01.png`）作为兜底；
- **公式**：裁剪公式区域，尽量转 LaTeX / 文本。

### 5.3 remote OCR（remote.py MinerU / Docling）

对 `ocr_required`、`low_text_review`、`text_layer_review` 页（排除了已被
vector 表格/图片恢复的页）：

- 调 MinerU（或 Docling）做整页 OCR；
- 回填页面文本、表格、公式；
- 失败页自动重试单页；仍失败的页记录为 `unresolved`，
  触发 `remote_ocr` 质量门禁失败。

> **重要边界**：ToUnicode CID 冲突页**强制走 OCR**，即使矢量表格网格完整——
> 恢复的网格内容仍是乱码，无法修复。而普通假字页如果矢量表格网格完整，
> 仍走 vector（OCR 会破坏单元格对齐）。

---

## 6. 第四步：合并（hybrid_pdf.py `_merge_vector_pages` / `_merge_pages`）

各路径结果合并到统一 Document：

| 页面结果 | 最终 route / page_type |
|---|---|
| native 文本 | `native_text` |
| vector 结构化表格 | `vector_layout` / `table` |
| vector 图片 | `figure_review` / `figure` |
| vector 公式 | `formula_review` / `formula` |
| OCR 成功 | `remote_ocr` / `scanned_text` |
| 空白页 | `blank_excluded` |

同时：

- 所有块（block）带坐标（bbox），保证**阅读顺序**正确；
- 图片资产带 caption + （可选）VLM 描述；
- 跨页表格合并、表格碎片跟踪。

### 图片描述（Figure Vision VLM，可选）

对 `figure_review` 页的 figure 资产，可调本地 VLM 生成结构化描述
（图题、类型、坐标轴、曲线、观察、不确定性）。开启后
`figure_semantics` 门禁的 `VLM described` 会列出已描述的页。

---

## 7. 质量门禁（hybrid_pdf.py `_rebuild`）

解析完成后跑一组质量门禁，决定任务状态：

| 门禁 | 失败级别 | 说明 |
|---|---|---|
| `empty_document` | fail | 无可索引内容 |
| `clause_chunk_coverage` | fail | 带正文的条款未进入 Chunk |
| `remote_ocr` | fail | OCR 页未解决（unresolved） |
| `table_structure` | fail | 表格候选页未结构化 |
| `table_html_leak` | fail | 表格 HTML 泄漏到正文 |
| `table_caption_alignment` | fail | 表题与结构化表格不一一对应 |
| `table_identity` | fail | 独立表格编号重复 |
| `table_chunk_integrity` | fail | 表格 Chunk 不完整/含原始 HTML |
| `table_merge_fidelity` | warn | 合并单元格需人工目检 |
| `figure_semantics` | warn | 图片页需人工目检（VLM 可选） |
| `text_layer_unicode` | warn | 无效 Unicode 已修复并走 OCR |
| `text_layer_glyph_names` | warn | 假字页已走 OCR |
| `remote_parser` | warn | 远程解析器告警 |

- 有 **fail** → `needs_review`（禁止自动发布）
- 只有 **warn** → `warning`（可发布，但需抽检）
- 全过 → `completed`

---

## 8. 输出产物

`write_document()` 写入：

| 文件 | 内容 |
|---|---|
| `baseline.json` | 原始页文本、解析器元数据 |
| `document_ir.json` | 统一结构：pages（route/特征）、blocks、chunks、assets、qa |
| `chunks.jsonl` | 每个 Chunk（标题、正文、条款结构、资产引用、表格） |
| `cleaned.md` | 清洗后的整篇 Markdown（带页标记） |
| `qa.json` | 质量指标 + 门禁结果 |
| `assets/` | 表格/图片/公式裁剪图、整页渲染图 |

---

## 9. 发布到 RAGFlow

- 人工抽检后通过 API 发布到 RAGFlow dataset（幂等：先查重、再上传）；
- 发布前可做**人工 Chunk 修订**（带审计记录）；
- 发布状态：`not_requested` / `queued` / `publishing` / `published` / `failed`。

---

## 10. 已处理的典型问题（踩坑记录）

| 问题 | 根因 | 处理 |
|---|---|---|
| 整页白背景图误判图片页 | 装饰背景被当内容图 | 编码密度检测，排除背景图 |
| 文本层 ToUnicode 映射自相矛盾 | CID→Unicode 不稳定 | 冲突密度阈值 → 强制 OCR |
| 图片页走 OCR 返回空 | OCR 对纯图无文字 | 图片主导的损坏页保留为 figure |
| 超大扫描图解压超限崩溃 | pypdf 读取超大图流 | `except Exception` 兜底，跳过 |
| 无竖线表格识别失败 | pdfplumber 依赖竖线网格 | 文本 x 坐标聚类推断列 |
| 表格页被误判 TOC 排除 | 表格单元格伪造目录条目 | TOC 复查跳过已结构化表格页 |
| 图名与下方正文混合 | 图名无编号 + fallback 拼接整段 | 无编号图名识别 + 正文行过滤 |
| 公式符号 PUA 乱码 | Symbol 字体映射到私有区 | 建立 PUA→Unicode 映射还原 |
| 扫描件 OCR 中文效果 | OCR 对清晰中文良好 | 正常路由到 OCR（本就是扫描件） |

---

## 11. 判断法则（给接手者）

- **能 native 就 native**：保留条款结构、精确字符、表格网格、公式；
- **OCR 是兜底不是优先**：只有文本层损坏或扫描件才走；
- **扫描件 OCR 中文好**是正常现象（无文本层损坏），但会丢条款结构/表格/公式
  精度，不要因此把好文档转成扫描件；
- 修改解析逻辑后必须重启 Gateway（无热重载），并重新解析目标文档刷新产物。
