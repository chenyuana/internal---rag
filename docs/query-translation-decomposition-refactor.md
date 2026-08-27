# 查询分解与查询转换重构实施方案

> 适用工程：`D:\internal-rag\source\internal-rag`  
> 目标执行模型：DeepSeek V4（也可替换为其他支持 JSON Schema 的模型）  
> 文档状态：2026-08-21 已实施；保留模型 planner/translator/summary 开关，本机默认走零模型调用路径

## 0. 2026-08-21 评审修订与本机决策

DeepSeek 对初版提出的三个实质性缺口均已纳入实施边界：

1. **表格输入不只支持 Markdown**：证据层必须识别 HTML `<table>/<caption>/<tr>/<th>/<td>`，按表头与数据行转换为键值记录。只有 caption、空表或原始标签不能单独成为 covered evidence；冒号结尾的引导行也不能成为证据。
   检索用的键值记录与用户核验用的完整表格必须分离保存：引用侧栏和回答区应保留空白单元格、合并前的行列和 caption，不能只展示截断后的 chunk 文本。
   对地质灾害倾斜摄影测量成果验收，当前规程正文表 2～表 5 的成果质量/精度要求优先级高于附录 A.3/A.4 空白记录表；实现通过表题/表头和质量语义识别这一关系，不把“表 2～表 5”写成跨文档硬编码。后者只能作为记录格式证据，不能单独使 cell 变为 covered。
2. **评测指标必须可复现**：Stage 1 即在 `ChatCompletionResponse` 增加安全的 `diagnostics`，公开 plan/cell 状态、主体文档 ID、证据记录数、检索/模型调用数和阶段耗时，但不包含 Chunk 正文、模型原始输出或密钥。`run_live_eval.py --output` 是本地评测产物，与生产日志分开。
3. **模型能力必须服从本机资源实测**：本机为 8GB 显存、`OLLAMA_MAX_LOADED_MODELS=1`。2026-08-21 对主对比题实测 4B planner 单次约 9.9–14.9 秒，连续输出三格而非 2×3 六格，Schema 有效率为 0%。因此默认配置采用 V2 执行器 + deterministic/legacy adapter，`planning.model_enabled=false`、`translation.model_enabled=false`、`comparison_summary_enabled=false`；模型路径完整保留，只有接入独立轻量或远程 verifier 后才开启。

现有 `comparison_matrix_builder` 原位演进为消费 `EvidenceRecord`，不新增第二套矩阵实现。迁移期扩展词表移到 `requirement_taxonomy.py`，仅作为低覆盖受控补检索；Stage 6 删除与否必须先量化有/无词表的 Recall@8 和污染率。

### 0.1 实施后基线（2026-08-21）

- 规划集：24/24 精确通过；相关规划/合同单测 50 passed。
- 合成检索 A/B：保留受控 taxonomy 时 8/8，移除后 5/8，因此暂不删除。
- 真实主题：6 个 cell，3 covered + 3 low_confidence，对比输出只保留与环节对应的实质证据，缺失格不再用其他环节条款填充；限制章节族补检索后 RAGFlow 调用稳定为 15，延迟约 44–45s。
- 无证据题：`UNANSWERABLE`，0 引用，0 模型调用，20.66s。`must_pass=true` live 用例 2/2 通过。
- 完整 pytest：453 passed / 3 failed；3 项仍是改造前已知的规程预处理基线，无新增失败。

本机当前默认是“结构化规划契约 + deterministic/legacy adapter + 受控 taxonomy 转换 + 环节类型证据过滤 + 确定性矩阵”。模型 planner 并非被删除，而是因 4B 实测 Schema 无效且与 8GB 单驻留资源冲突而默认关闭。

## 1. 目标与边界

本次重构不替换 RAGFlow，也不继续为单题增加关键词和语义型正则。RAGFlow 继续承担文档解析、索引和候选召回；Gateway 新增独立、可测试的查询规划层：

1. **查询分解**：把复杂问题拆成有限个可独立检索、验证的单元。
2. **查询转换**：仅在单元格低召回时生成一条受约束的补充查询。
3. **主体文档解析**：先通过文档元数据确定范围，再检索内容。
4. **结构化证据**：检索结果先转换为 `EvidenceRecord`，下游不反复解析 Chunk。
5. **确定性结果优先**：对比题先构建矩阵，模型只生成可失败的简短总结。
6. **评测驱动**：规划、检索和答案策略必须通过固定测试集。

### 1.1 成功指标

| 指标 | 目标 |
|---|---:|
| 规划 Schema 合法率 | 100% |
| 对比题主体召回率 | ≥98% |
| 对比题维度召回率 | ≥95% |
| 单元格相关证据 Recall@8 | ≥90% |
| 跨主体文档污染率 | 0% |
| 引用存在性与归属正确率 | 100% |
| 已覆盖单元格完整率 | 100% |
| 模型失败仍可返回确定性矩阵 | 100% |
| 对比题模型调用次数 | ≤1 |
| 对比题 P95 总耗时（本机目标） | ≤90 秒 |

### 1.2 非目标

- 不让规划模型直接回答业务问题。
- 不让查询转换生成条款号、参数值或新标准名称。
- 不使用 HyDE 作为法规/标准问答默认路径。
- 不在第一阶段直接删除全部旧规则；先双轨运行、评测，再切换。
- 不使用用户临时选择的答案模型作为规划模型；规划模型由服务端固定配置。

## 2. 当前问题

- `query_analyzer.py` 的关键词同时承担标准化和语义分类。
- `query_planner.py` 同时承担句式切分、主体抽取、维度抽取、领域扩词和查询生成。
- `_DIMENSION_EXPANSIONS` / `_SUBJECT_EXPANSIONS` 已成为隐式领域模型。
- `cell_evidence.py` 直接依赖 planner 私有词表，形成循环耦合。
- `retrieval_service.py` 的首轮、主体限定重检索、章节扩展、重排串行叠加。
- `rag_pipeline.py` 在模型初答、repair、sanitize 后才使用确定性矩阵。
- `retrieval.enable_query_rewrite` 已配置，但没有独立、受限、可观测的转换服务。

重构原则：正则只处理“文本长什么样”，不能判断“文本是什么意思”。条号、引用、页码、固定标题可继续使用正则；主体、维度、环节和要求类型必须由结构化规划、元数据或受约束分类完成。

## 3. 目标架构

```text
ChatCompletionRequest
  → QueryNormalizer（无损标准化、条号/精确 token）
  → QueryPlanningService
      ├─ DeterministicFastPath（条号、常量、简单事实）
      ├─ StructuredPlannerModel（复杂题、JSON Schema）
      └─ LegacyPlannerFallback（迁移期兜底）
  → SubjectDocumentResolver（主体 → document_ids）
  → AdaptiveRetrievalExecutor
      ├─ 每个 cell 一次 original hybrid retrieval
      ├─ CoverageProbe
      ├─ 仅低覆盖 cell 调 QueryTranslator
      └─ CandidateFusion + Reranker
  → EvidenceRecordExtractor
  → CoverageMatrixBuilder（确定性矩阵）
  → SummaryGenerator（最多一次，失败可跳过）
  → Scope / Grounding / Completeness 校验
  → ChatCompletionResponse
```

## 4. 数据契约

### 4.1 `app/schemas/planning.py`

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

PlannerSource = Literal["deterministic", "model", "legacy_fallback"]
RetrievalQueryKind = Literal["original", "lexical", "semantic"]


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: RetrievalQueryKind
    text: str = Field(min_length=2, max_length=500)
    generated_by: Literal["user", "planner", "translator"]


class PlannedCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^q[1-9][0-9]*$")
    subject: str | None = Field(default=None, max_length=200)
    aspect: str = Field(min_length=2, max_length=200)
    original_query: str = Field(min_length=2, max_length=500)
    retrieval_queries: list[RetrievalQuery] = Field(min_length=1, max_length=2)
    required: bool = True


class QueryPlanV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["query-plan-v2"] = "query-plan-v2"
    query_type: Literal[
        "fact", "summary", "comparison", "multi_hop", "procedure", "unknown"
    ]
    subjects: list[str] = Field(default_factory=list, max_length=4)
    aspects: list[str] = Field(default_factory=list, max_length=8)
    cells: list[PlannedCell] = Field(min_length=1, max_length=8)
    synthesis_mode: Literal["direct", "matrix", "sequence", "map_reduce"]
    planner_source: PlannerSource
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_matrix(self) -> "QueryPlanV2":
        if self.query_type == "comparison":
            expected = len(self.subjects) * len(self.aspects)
            if len(self.subjects) < 2 or not self.aspects or len(self.cells) != expected:
                raise ValueError("comparison plan must be a complete subject × aspect matrix")
        return self
```

迁移期保留现有 `QueryPlan`，新增 `QueryPlanAdapter.v2_to_legacy()`；下游完成迁移后再删除适配器。

### 4.2 `app/schemas/evidence.py`

```python
class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    cell_id: str
    subject: str | None
    aspect: str
    document_id: str
    document_name: str
    section_id: str | None = None
    section_title: str | None = None
    requirement_type: str | None = None
    requirement_text: str
    citation_id: str
    page_number: int | None = None
    retrieval_score: float
    confidence: float = Field(ge=0, le=1)
```

Matrix Builder、完整性校验和模型总结只消费 `EvidenceRecord`。

### 4.3 Chunk 元数据扩展

入库阶段逐步补充：

```text
document_subject
section_id
section_parent_id
section_title
requirement_type
process_stage
content_type
```

推荐 `requirement_type`：`equipment`、`control_point_layout`、`flight_operation`、`data_processing`、`quality_check`、`deliverable`、`acceptance`、`reporting`、`safety`、`management`、`unknown`。

### 4.4 安全评测诊断契约

```text
diagnostics.plan_version / planner_source / planned_cell_count
diagnostics.cells[].cell_id / subject / aspect / status
diagnostics.cells[].document_ids / citation_ids / evidence_record_count
diagnostics.subject_document_ids
diagnostics.retrieval_calls / translated_cell_ids
diagnostics.model_calls / model_call_breakdown
diagnostics.latencies_ms / stage_counts
```

该契约只暴露标识符和计数，不暴露 Chunk 正文或模型原始输出。真实评测的 chunk 级对照从本地评测文件和已有引用 ID 完成，生产日志仍遵守最小化原则。

## 5. 文件修改清单

### 5.1 新增

| 文件 | 职责 |
|---|---|
| `app/schemas/planning.py` | QueryPlanV2 数据契约 |
| `app/schemas/evidence.py` | EvidenceRecord 数据契约 |
| `app/services/query_planning_service.py` | 快速路径、模型规划和旧规划兜底 |
| `app/services/structured_planner_model.py` | 固定 planner 模型和 JSON Schema 调用 |
| `app/services/subject_document_resolver.py` | 主体到文档 ID 的解析 |
| `app/services/query_translator.py` | 低覆盖 cell 的单次转换 |
| `app/services/retrieval_fusion.py` | 候选去重和 RRF/稳定排序 |
| `app/services/evidence_record_extractor.py` | Chunk → EvidenceRecord |
| `app/services/comparison_orchestrator.py` | 确定性矩阵优先流程 |
| `prompts/query_planning.txt` | DeepSeek V4 规划提示词 |
| `prompts/query_translation.txt` | 受限转换提示词 |
| `prompts/matrix_summary.txt` | 只基于矩阵的总结提示词 |

### 5.2 修改

| 文件 | 修改 |
|---|---|
| `app/core/config.py` | 新增 PlanningSettings、TranslationSettings |
| `config/default.yaml` | 增加开关、上限、超时、fallback |
| `app/services/registry.py` | 注册 planner、translator、resolver |
| `app/services/retrieval_service.py` | 使用 QueryPlanV2 驱动自适应检索 |
| `app/pipelines/rag_pipeline.py` | 对比题矩阵前置，模型总结最多一次 |
| `app/schemas/retrieval.py` | ChunkMetadata 增加结构化字段 |
| `app/web/static/chat.js` | 可选展示 planning/retrieval/summary 阶段 |

### 5.3 最终删除或降级

双轨评测通过后删除：`_DIMENSION_EXPANSIONS`、`_SUBJECT_EXPANSIONS`、承担语义判断的长尾 suffix、`query_scope` 两字子串兜底、`cell_evidence` 对 planner 私有词表的依赖，以及对比题多轮自由生成链。

## 6. 配置

```yaml
planning:
  enabled: true
  model_enabled: false
  model_role: verifier
  timeout_seconds: 15
  max_subqueries: 8
  min_confidence: 0.75
  use_deterministic_fast_path: true
  fallback_to_legacy: true
  shadow_mode: true
  prompt_path: prompts/query_planning.txt
  prompt_version: query-planning-v1

translation:
  enabled: true
  model_enabled: false
  trigger: low_confidence_only
  timeout_seconds: 10
  max_variants_per_cell: 1
  max_total_translations: 4
  min_original_results: 1
  min_coverage_confidence: 0.55
  preserve_original_query: true
  prompt_path: prompts/query_translation.txt
  prompt_version: query-translation-v1

retrieval:
  max_complex_subqueries: 8
  enable_ragflow_keyword_extraction: false
  max_parallel_ragflow_requests: 2
  per_cell_candidate_top_k: 12
  fusion_rrf_k: 60
  translated_query_weight: 0.85

generation:
  comparison_matrix_first: true
  comparison_summary_enabled: false
  comparison_summary_timeout_seconds: 30
  comparison_summary_max_attempts: 1
```

`shadow_mode=true` 时新旧 planner 同时运行但只使用旧结果，记录差异；稳定后再切换。

`translated_query_weight` 只作用于 weighted RRF 的 translated 名次贡献：`weight / (rrf_k + rank)`；它不乘原始 cosine/hybrid score。`preserve_original_query=true` 表示 original 候选始终参加融合，两项配置不冲突。

## 7. DeepSeek V4 提示词

### 7.1 查询规划 `prompts/query_planning.txt`

```text
你是企业内部标准检索的查询规划器，不回答问题，只输出满足 JSON Schema 的检索计划。

任务：
1. 判断 query_type。
2. comparison 提取明确主体和维度，生成完整 subject × aspect 单元格。
3. multi_hop 提取可独立检索的方面。
4. fact/procedure/summary 保持最少必要子查询。

硬约束：
- 不得编造文档名、标准号、条款号、参数值或用户未提及的业务主体。
- 主体和维度尽量保留用户原词，同义词不能混入 subjects/aspects。
- 每个单元格初始只生成一条 original retrieval query。
- 查询必须包含所属主体和维度，不得只输出宽泛维度词。
- 最多 8 个单元格；超出时保留核心项并写 warnings。
- 条款号、型号、常量名必须逐字保留。
- 只输出 JSON。
```

输入只包含规范化问题、已提取条号/精确 token 和允许的 query_type，不把候选 Chunk 放入规划提示词。

### 7.2 查询转换 `prompts/query_translation.txt`

```text
你是检索查询转换器。当前单元格首次检索不足，请生成一条补充检索查询。

输入：原始问题、subject、aspect、original_query、首次命中的章节标题摘要。

硬约束：
- 只生成一条查询。
- 保留 subject，不得换成其他行业或对象。
- 保留 aspect 含义。
- 不得生成用户未提供的标准号、条款号、数值或结论。
- 不回答问题。
- 优先使用标准文档可能出现的名词短语。
- original_query 已足够明确时返回 should_translate=false。
- 只输出 JSON。
```

输出：

```json
{
  "should_translate": true,
  "translated_query": "长大桥梁无人机精细化巡检 巡检成果 巡检报告 影像资料 提交要求",
  "reason": "首次检索未覆盖成果交付类章节"
}
```

### 7.3 矩阵总结 `prompts/matrix_summary.txt`

```text
只能根据给定的已校验矩阵写 2~5 句统筹说明。
- 不增加矩阵中没有的事实、数字、条款号或引用。
- 明确指出哪些单元格资料未说明。
- 不重写矩阵，不输出 JSON。
- 超时或失败时由系统直接展示矩阵。
```

## 8. 核心算法

### 8.1 规划路由

```python
async def plan(query: NormalizedQuery) -> QueryPlanV2:
    if query.article_ids or query.exact_tokens:
        return deterministic_single_plan(query)
    if query.query_type in {"fact", "procedure", "summary"} and is_simple(query):
        return deterministic_single_plan(query)
    try:
        plan = await structured_planner.plan(query)
        validate_no_invented_tokens(plan, query)
        return plan
    except (AppError, ValidationError, TimeoutError):
        return legacy_adapter.from_legacy(legacy_planner.plan(query))
```

模型规划失败不能让问答失败。

### 8.2 主体文档解析

解析顺序：

1. 请求显式 `document_ids`。
2. `document_subject` 元数据精确匹配。
3. 规范化文档标题与主体的 token/n-gram 得分。
4. 低置信度时允许一次轻量模型分类，只能从候选文档 ID 中选择。
5. 无匹配则 cell=`no_document`，禁止用第三文档填充。

```python
class SubjectResolution(BaseModel):
    subject: str
    document_ids: list[str]
    confidence: float
    source: Literal["request", "metadata", "title", "model", "none"]
```

### 8.3 自适应 Query Translation

仅在以下条件触发：

- 主体过滤后候选数为 0。
- 最高 hybrid score 低于复杂题阈值。
- CoverageProbe 判定 `low_confidence`。
- 只有章节标题，没有实质条款/表格内容。

禁止转换：精确条款号、型号、常量题；已有 covered EvidenceRecord；达到全局转换上限。

### 8.4 候选融合

每个 cell 独立融合：

```text
original results ─┐
                  ├─ (document_id, chunk_id) 去重
translated results┘
  → RRF（只使用名次）
  → subject_document_ids 再校验
  → cell query reranker
```

禁止直接比较不同查询的原始 cosine/hybrid 分数。最终阈值使用 reranker 分数或 CoverageProbe。

### 8.5 EvidenceRecord 提取

优先级：

1. 入库元数据已有 `process_stage` / `requirement_type`：直接映射。
2. section metadata 与 aspect 一致：提取条款正文。
3. 表格转换为 `表头: 值` 键值文本，保留原始引用。
   - Markdown：识别表头行、分隔行和数据行。
   - HTML：解析 `table/caption/tr/th/td`；caption 仅作记录上下文，必须存在数据行。
   - 空表、只有 caption 的表和原始 HTML 标签不得作为实质证据。
   - `4.3.4 ……应满足以下要求:` 一类冒号结尾引导行不得作为实质证据。
4. 可选模型提取只能复制输入 Chunk 中存在的内容。
5. 无实质要求时为 `low_confidence`，标题行不能单独成为 covered evidence。

所有数字、条号和核心短语必须能在被引 Chunk 中找到。

### 8.6 确定性矩阵优先

```python
records = await evidence_extractor.extract(plan, selected_chunks)
matrix = matrix_builder.build(plan, records)
validators.validate(matrix)

try:
    summary = await asyncio.wait_for(
        summary_generator.generate(matrix),
        timeout=settings.generation.comparison_summary_timeout_seconds,
    )
except Exception:
    summary = None
    warnings.append("模型总结不可用，已直接展示确定性矩阵。")

return response(matrix=matrix, summary=summary)
```

不得先让模型自由生成整张矩阵。

## 9. 错误与降级

| 故障 | 行为 |
|---|---|
| planner 超时/坏 JSON | legacy planner，记录 fallback |
| 主体解析不到 | cell=`no_document`，不跨主体补证据 |
| 原查询低覆盖 | 最多一次 translation |
| translator 超时/坏 JSON | 保留原查询结果 |
| 单 cell RAGFlow 超时 | 该 cell 重试一次；其他 cell 继续 |
| 全局 RAGFlow 不可用 | 返回 503 |
| reranker 不可用 | 使用融合名次，标记 fallback |
| EvidenceRecord 无法抽取 | cell=`low_confidence` |
| summary 超时/坏 JSON | 直接返回矩阵，HTTP 200 |
| 引用/主体校验失败 | 删除失败 record，重算 cell；不进入模型 repair 循环 |

所有循环必须有上限。

## 10. 可观测性

每个请求记录：

```text
planner_source / planner_latency_ms / planner_fallback
planned_cell_count
subject_resolution_source
initial_retrieval_calls / translated_retrieval_calls / translated_cell_ids
ragflow_latency_ms_by_cell / reranker_latency_ms
evidence_record_count_by_cell / coverage_status_by_cell
summary_attempted / summary_latency_ms / summary_fallback
total_latency_ms
```

日志不得记录 API Key；生产默认不记录完整 Chunk 和模型原始输出。

## 11. 分阶段实施

### Stage 0：冻结基线

- 完整 pytest，记录现有 3 项 regulation preprocess 基线失败。
- 保存桥梁/地灾 debug 响应和耗时。
- 先补 HTML/Markdown 表格转换、空 caption 表与冒号引导行过滤回归。
- 记录 Gateway 监听 PID、源码/pyc mtime；重启后核对 PID，避免旧进程或缓存残留。
- 冻结 `_DIMENSION_EXPANSIONS`，除非阻断性故障。

### Stage 1：契约与 Shadow Planner

- 新增 schema、配置和 planner model。
- 新 planner 只 shadow，不驱动检索。
- 记录新旧 plan 差异。
- `ChatCompletionResponse.diagnostics` 在本阶段落地，保证后续指标可统计。
- 本机 planner shadow 必须记录延迟、Schema 有效率、GPU 模型切换和与 RAGFlow 并发时吞吐；数据不达标时模型路径保持关闭。
- 用 `planning_cases.jsonl` 验证结构。

验收：规划 Schema 合法率 100%，旧问答结果不变。

### Stage 2：主体文档解析

- 每个主体只解析一次文档集合。
- 检索前绑定 `subject_document_ids`。
- 停用两字子串兜底，但暂不删除代码。

验收：合成测试污染率 0；桥梁/地灾只使用对应两份规程。

### Stage 3：自适应 Query Translation

- 首轮每 cell 一条 original query。
- 仅低覆盖 cell 转换一次。
- subject/aspect 保真校验。
- 每请求最多四次转换。

验收：精确条款/常量题转换数为 0；低覆盖用例转换数为 1。

### Stage 4：EvidenceRecord

- 复用并原位演进现有 `comparison_matrix_builder`，数据源迁移为 EvidenceRecord。
- Markdown 与 HTML 表格转键值；空 caption 表、标题行和冒号引导行不能独立覆盖 cell。
- 数字和条号 grounding。

验收：引用与数值 grounding 100%。

### Stage 5：确定性矩阵前置

- 对比题从 EvidenceRecord 建矩阵。
- summary 最多一次、30 秒。
- summary 失败仍返回 200。
- 删除对比题初答/repair/sanitize 路由。

验收：模型持续坏 JSON 时仍成功；对比题模型调用≤1。

### Stage 6：切换与清理

- `shadow_mode=false`，启用新 planner。
- 对扩展词表做 A/B：分别统计 planning/retrieval 集的 Recall@8、污染率和 translation gain；只有无词表版不退化时才删除。
- 删除经量化确认无调用或无收益的语义词表和正则。
- `cell_evidence` 不再 import planner 私有成员。
- 更新 README 和 handoff。

验收：完整 pytest 无新增失败，真实集门禁通过。

## 12. 测试策略

数据文件：

- `tests/evaluation/planning_cases.jsonl`：无外部依赖，规划与转换策略。
- `tests/evaluation/retrieval_cases.jsonl`：合成候选集，验证主体隔离、证据选择与自适应转换。
- `tests/evaluation/live_rag_cases.jsonl`：真实知识库端到端门禁。
- `tests/evaluation/run_live_eval.py`：调用当前网关执行真实集并生成机器可读报告。
- `tests/unit/test_evaluation_dataset_contract.py`：JSONL 合同校验。

必须新增测试模块：

```text
test_query_planning_service.py
test_structured_planner_model.py
test_subject_document_resolver.py
test_query_translator.py
test_retrieval_fusion.py
test_evidence_record_extractor.py
test_comparison_orchestrator.py
```

故障注入必须覆盖：planner 401/429/500/超时/坏 JSON；translator 改主体或编条款；某 cell RAGFlow 超时；reranker 不可用；summary 持续坏 JSON；第三文档高分污染；同一 Chunk 多 cell 归属。

核心指标：query_type accuracy、subject/aspect exact match、cell count、exact token preservation、cell Recall@8、污染率、translation gain、coverage completeness、citation validity、numeric grounding、P50/P95 latency、model calls/request。

## 13. DeepSeek V4 执行主提示词

每次只执行一个 Stage：

```text
你正在修改 D:\internal-rag\source\internal-rag。

先完整阅读：
1. docs/query-translation-decomposition-refactor.md
2. HANDOFF-2026-08-21.md
3. app/services/query_analyzer.py
4. app/services/query_planner.py
5. app/services/retrieval_service.py
6. app/pipelines/rag_pipeline.py
7. 对应 tests/unit 测试

规则：
- 本轮只实施我指定的 Stage，不提前删除旧路径。
- 保留 dirty worktree 中无关改动。
- 所有模型输出必须使用 Pydantic/JSON Schema，extra=forbid。
- 不新增面向单题的关键词或语义型正则。
- query translation 仅在低覆盖 cell 触发，每 cell 最多一次。
- 对比题由 EvidenceRecord 构建确定性矩阵。
- 网络、401、429、5xx 不得伪装成 JSON 修复问题。
- 不记录或输出 API Key。

流程：
1. 说明本 Stage 的问题和最小改动面。
2. 先补失败测试，再实现。
3. 运行目标测试、ruff、mypy；既有错误单独报告。
4. 完整 pytest，与既有三项解析失败基线比较。
5. 修改检索或页面后做真实端到端验证。
6. 更新 HANDOFF-2026-08-21.md。

禁止：
- 为桥梁/地灾单题增加扩展词。
- 用两字 substring 决定主体文档。
- 无界 rewrite/retry。
- summary 失败导致矩阵请求失败。
```

## 14. 推荐命令

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_evaluation_dataset_contract.py -q
.\.venv\Scripts\python.exe -m pytest tests\unit\test_query_planning_service.py -q
.\.venv\Scripts\python.exe -m pytest tests\unit\test_query_translator.py -q
.\.venv\Scripts\python.exe -m pytest tests\unit\test_retrieval_service.py tests\unit\test_rag_pipeline.py -q
.\.venv\Scripts\python.exe tests\evaluation\run_live_eval.py --base-url http://127.0.0.1:8080
.\.venv\Scripts\python.exe -m ruff check app tests
.\.venv\Scripts\python.exe -m pytest --basetemp=tmp\pytest-query-refactor -q
```

## 15. 最终验收

1. `planning_cases.jsonl` 全通过。
2. `retrieval_cases.jsonl` 全通过，尤其是错误文档高分时仍不串主体。
3. `live_rag_cases.jsonl` 的 `must_pass=true` 全通过。
4. 桥梁/地灾题不引用第三文档，六格齐全，缺失格诚实降级。
5. 对比题模型调用最多一次，坏 JSON 仍返回 200。
6. RAGFlow `keyword=false`，除非独立评测证明必要并有专用限流。
7. 完整 pytest 无新增失败。
8. P95 延迟和质量指标写入 handoff，不以单次主观观察替代评测。

## 16. 2026-08-21 实施校正

针对桥梁/地灾六格对比题的真实回归，Stage 4/5 增加以下通用约束：

- dense retrieval 保留原始 cell query；语义扩展作为独立有界查询并做 RRF，禁止把同义词袋拼进原查询。
- “成果验收”与“成果交付”分型；质量表/合格判据可覆盖验收格，成果清单、报告格式和空白记录表不可覆盖。
- 多证据维度按信息类型设置有界证据目标，优先完整表格并按表题去重；矩阵摘要跨引用轮转采样。
- 前端完整表格按“引用编号 + 答案中明确表号/表题”绑定，不再按 Chunk 粗粒度全部展开。

当前真实门禁 `2/2 passed`，完整 pytest `473 passed, 3 failed`（均为既有 PDF 预处理基线）。详细记录见 `HANDOFF-2026-08-21.md` §2.7。
