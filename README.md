# Internal RAG Gateway

Agent 接手本项目时请优先阅读：

- [`docs/agent-handoff.md`](docs/agent-handoff.md)：项目从设计、部署、接入、清洗、
  表格优化到当前遗留问题的完整交接说明。

里程碑说明：

- [`docs/milestone-2.md`](docs/milestone-2.md)：混合解析、质量门禁与 RAGFlow 发布。
- [`docs/milestone-3.md`](docs/milestone-3.md)：本地 MinerU、串行队列与单入口界面。
- [`docs/query-translation-decomposition-refactor.md`](docs/query-translation-decomposition-refactor.md)：
  查询分解与查询转换重构实施方案（对比题确定性矩阵优先、主体文档解析、受限
  查询转换、`EvidenceRecord` 结构化证据、评测驱动；分 Stage 0–6 实施）。

企业内网技术资料 RAG 问答系统的 Gateway。目前已经具备 FastAPI 基础设施、查询
标准化、ACL 范围过滤、RAGFlow 混合检索、Reranker、Chunk 规范化、引用映射、
文档上传、自动混合解析、质量门禁、预览与 RAGFlow 发布计划。

当前尚未实现完整的用户权限后台和业务数据库表；RAGFlow 正式发布需要人工配置
API Key 并确认执行。

## D 盘约束

项目和全部可控运行数据统一放置在：

```text
D:\internal-rag
```

代码位于：

```text
D:\internal-rag\source\internal-rag
```

Docker 数据、数据库、模型、缓存、日志和临时文件必须使用 `.env` 中的
`RAG_ROOT=D:/internal-rag`，不得改用匿名 Docker 卷。

## 本机启动

当前机器已有 `D:\anaconda3\python.exe`。在 PowerShell 中执行：

```powershell
Set-Location D:\internal-rag\source\internal-rag
& D:\anaconda3\python.exe -m venv .venv
$env:PIP_CACHE_DIR = 'D:\internal-rag\cache\pip'
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

访问：

- 文档接入界面：`http://127.0.0.1:8080/ingestion`
- OpenAPI：`http://127.0.0.1:8080/docs`
- 存活检查：`http://127.0.0.1:8080/health`
- 就绪检查：`http://127.0.0.1:8080/ready`

开发配置默认禁用外部服务，因此骨架可独立达到 ready。生产配置要求 RAGFlow
和四个模型服务均可用。

日常本机运行也可以使用固定的 D 盘脚本：

```powershell
.\deploy\mineru\start.ps1
.\deploy\gateway\start.ps1
```

## 测试和检查

```powershell
Set-Location D:\internal-rag\source\internal-rag
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check app tests
.\.venv\Scripts\python.exe -m mypy app
```

## Docker 基础设施

复制并修改 `.env` 中的密码后执行：

```powershell
Set-Location D:\internal-rag\source\internal-rag
docker compose config
docker compose --profile development up --build
```

Gateway Compose 包含 Gateway、PostgreSQL、Redis 和 MinIO。RAGFlow 使用
官方 `v0.26.4` 本地部署包单独启动，详见 `deploy/ragflow/README.md`。

RAGFlow 源码和部署文件已保存在：

```text
D:\internal-rag\runtime\ragflow-0.26.4
```

运行环境是 Windows。Docker Desktop 也是 Windows 应用，默认使用 Hyper-V 承载
RAGFlow 所需的 Linux 容器；应用代码不部署到 Ubuntu/WSL。首次安装：

```powershell
.\deploy\docker\install-docker-desktop.ps1
```

安装程序、Docker Desktop 程序和虚拟磁盘均写入 `D:\internal-rag`。安装完成后在
Windows 启动 Docker Desktop，再执行：

```powershell
.\deploy\ragflow\start.ps1
.\deploy\ragflow\status.ps1
```

## 基础检索 API

```http
POST /api/v1/retrieval/search
POST /api/v1/retrieval/debug
```

请求示例：

```json
{
  "query": "HEALTH_MONITOR_PERIOD_MS 的值是多少？",
  "knowledge_base_ids": ["ragflow-dataset-id"],
  "filters": {
    "project_name": "MC任务机",
    "version": "V2.3"
  }
}
```

默认强制追加 `status=effective`。查询历史版本时必须同时指定明确版本和
`include_historical=true`。调试接口返回每个候选 Chunk 的 hybrid、vector、
keyword、rerank 分数、过滤原因、最终排名和引用编号。

RAGFlow 请求显式开启 `reference_metadata`，只回传版本与引用所需字段。默认关闭
RAGFlow 的 LLM 关键词扩展；term/BM25 与 Dense 混合检索仍由
`vector_similarity_weight` 控制。

当前 `knowledge_base_ids` 对应 RAGFlow dataset ID。业务知识库 ID 到 dataset ID
的数据库映射将在文档与元数据管理阶段实现。

## 智能问答与模型切换

浏览器访问 `http://127.0.0.1:8080/chat`。问答页可在现有本地答案模型之间切换，
也可以临时接入支持 `GET /v1/models` 和 `POST /v1/chat/completions` 的
OpenAI 兼容 API：

```http
GET    /api/v1/admin/answer-models
POST   /api/v1/admin/model-connections
DELETE /api/v1/admin/model-connections/{source_id}
```

运行时 API Key 只保存在 Gateway 进程内存中，不写入配置文件、数据库或浏览器存储；
Gateway 重启后需要重新接入。连接和模型选择按 `X-User-ID` 隔离。

## 配置加载顺序

1. `config/default.yaml`
2. `config/{APP_ENV}.yaml`
3. YAML 中引用的环境变量

生产环境：

```powershell
$env:APP_ENV = 'production'
```

敏感值只放在未提交的 `.env` 或内网密钥系统中。`/api/v1/admin/config`
返回的是脱敏配置，不包含数据库 URL、对象存储密码或模型 API Key。

## 健康检查语义

- `/health`：仅表示 Gateway 进程存活，不调用任何外部依赖。
- `/ready`：并行检查所有已启用服务；任一 `required=true` 服务不可用时返回 503。
- 模型健康检查使用 OpenAI 兼容的 `GET /v1/models`。
- RAGFlow 健康路径通过 `RAGFLOW_HEALTH_PATH` 配置，避免业务代码绑定特定版本。
