# 查询规划与 RAG 评测集

本目录是 `docs/query-translation-decomposition-refactor.md` 的配套验收集。它描述的是改造后的目标行为，不要求当前旧 planner 立即通过全部语义用例。

## 文件

| 文件 | 用途 | 是否依赖服务 |
|---|---|---|
| `planning_cases.jsonl` | 查询类型、主体、维度、cell 数量、转换边界 | 否 |
| `retrieval_cases.jsonl` | 主体隔离、证据选择、转换触发、诚实缺失 | 否 |
| `synthetic_retrieval_eval.py` | 运行检索集，并量化有/无迁移词表的 A/B | 否 |
| `live_rag_cases.jsonl` | 真实 RAGFlow 与回答链路验收 | 是 |
| `run_live_eval.py` | 运行真实集，输出 JSON 报告 | 是 |
| `../unit/test_evaluation_dataset_contract.py` | 校验三份 JSONL 的格式和约束 | 否 |

## 使用顺序

1. 改造前先运行合同测试，确认测试数据没有损坏。
2. Stage 1 将 `planning_cases.jsonl` 接入新 planner 的 shadow 测试。
3. Stage 2 到 Stage 4 将 `retrieval_cases.jsonl` 转换为参数化单元测试。
4. 每个 Stage 只运行其目标测试；Stage 6 再执行真实门禁和完整 pytest。

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_evaluation_dataset_contract.py -q
.\.venv\Scripts\python.exe tests\evaluation\synthetic_retrieval_eval.py
.\.venv\Scripts\python.exe tests\evaluation\run_live_eval.py --base-url http://127.0.0.1:8080
```

默认只运行 `must_pass=true` 的真实用例。需要查看探索性用例时：

```powershell
.\.venv\Scripts\python.exe tests\evaluation\run_live_eval.py `
  --base-url http://127.0.0.1:8080 `
  --include-exploratory `
  --output tmp\live-rag-eval.json
```

真实测试不会写入 API Key，使用网关当前默认回答模型。如果知识库名称不存在，用例会明确失败，不会自动改用其他知识库。

## 判断规则

- `planning_cases.jsonl` 是结构规划的权威数据；每题最多 8 个 cell，每 cell 最多 1 个转换查询。
- `retrieval_cases.jsonl` 中 `selected_chunk_ids` 必须命中，`forbidden_chunk_ids` 必须排除；高分也不能绕过主体文档隔离。
- `live_rag_cases.jsonl` 中 `must_pass=true` 是发布门禁，`must_pass=false` 是待确认语料覆盖后的探索性案例。
- 真实响应已公开不含正文的安全诊断字段 `diagnostics.cells`、`diagnostics.model_calls` 和阶段耗时；runner 会直接检查 cell 数量与模型调用预算。Chunk 正文和模型原始输出仍不会进入生产诊断。
- `UNANSWERABLE` 是合法结果。语料没有依据时，编造答案比返回资料不足更严重。

## 给 DeepSeek V4 的执行要求

不要一次性让模型改完整链路。每次把主文档和一个 Stage 发给它，并要求：

1. 先把该 Stage 对应 JSONL 转成参数化失败测试；
2. 只实现使该 Stage 通过的通用机制；
3. 禁止给单个问题追加关键词、同义词或专用正则；
4. 报告新增通过数、既有失败数、P95 延迟和每请求模型调用数；
5. 旧路径保留到 shadow 对比完成后再删除。

完整提示词位于 `docs/query-translation-decomposition-refactor.md` 的“DeepSeek V4 执行主提示词”章节。
