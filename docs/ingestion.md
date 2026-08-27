# 自动文档接入服务

当前版本实现下一阶段的第一里程碑：用户只上传文件，不选择解析器。系统负责保存原件、
识别内容、执行预处理、生成统一 Document IR，并根据质量门禁决定是否需要人工复核。

## 数据目录

所有运行数据默认保存在：

```text
D:\internal-rag\data\ingestion
```

目录结构：

```text
originals\   按 SHA-256 保存的不可变原件
jobs\        可恢复的任务状态
outputs\     baseline、Document IR、cleaned Markdown、chunks 和 QA
temporary\   上传过程的临时文件
```

可通过以下环境变量调整，但仍应保持在 D 盘：

```dotenv
INGESTION_ENABLED=true
INGESTION_ROOT=D:/internal-rag/data/ingestion
INGESTION_MAX_UPLOAD_BYTES=209715200
```

## API

上传文件：

```http
POST /api/v1/ingestion/jobs
Content-Type: multipart/form-data
X-User-ID: user-id

file=<document>
knowledge_base_id=<optional-ragflow-dataset-id>
```

查询：

```http
GET  /api/v1/ingestion/jobs
GET  /api/v1/ingestion/jobs/{job_id}
GET  /api/v1/ingestion/jobs/{job_id}/preview
POST /api/v1/ingestion/jobs/{job_id}/retry
PATCH /api/v1/ingestion/jobs/{job_id}/review
```

任务状态：

- `queued`：已经持久化，等待处理。
- `running`：正在解析。
- `completed`：质量门禁通过。
- `warning`：已生成结果，但需要抽检。
- `needs_review`：质量门禁失败，禁止自动发布。
- `failed`：程序或文件错误。

## 人工审核与失败门禁放行

`completed` 和 `warning` 任务可按普通流程审核通过。`needs_review` 表示自动质量
门禁失败，普通“审核通过”仍会被拒绝；审核人员完成内容编辑和原件核对后，可以在
接入台点击独立的“人工确认无误”按钮显式放行：

```http
PATCH /api/v1/ingestion/jobs/{job_id}/review
Content-Type: application/json
X-User-ID: reviewer-id

{
  "status": "approved",
  "note": "已人工修订并核对第 9、15、16 页",
  "quality_gate_override": true
}
```

人工放行不会修改原任务的 `needs_review` 状态或删除失败门禁，而是在审核记录中保存
确认人、确认时间和当时被放行的完整门禁快照。发布端只有同时看到 `approved` 和有效
override 才允许发送；后续再次编辑 Chunk 或重新解析时，override 自动失效并要求重新
审核。

## 当前处理能力

- PDF 与 Word OOXML（`.docx`）文件签名、ZIP 展开大小和上传大小检查。
- SHA-256 内容寻址与重复原件复用。
- PDF 使用原生文本、矢量版面与远程 OCR 混合解析。
- DOCX 使用原生 OOXML 解析，不经过 OCR；保留标题样式层级、显式分页符、列表、
  合并表格、脚注/尾注和图片资产。
- 页级 `native_text`、`low_text_review`、`layout_review`、`ocr_required` 路由。
- 目录页标记为 `toc_excluded`，保留基线但不生成检索块。
- 页眉页脚清理、法规标题识别、父子切块和质量门禁。
- 可审计的 baseline、Document IR、清洗文本、chunk 和 QA 输出。

## 当前边界

- 解析器注册表包含 `hybrid-pdf` 与 `native-docx`；解析状态接口同时展示本地与远程服务。
- 旧版二进制 `.doc`、启用宏的 `.docm` 暂不开放；应先在可信环境转换为 `.docx`。
- DOCX 没有稳定的跨渲染器物理页码，页面诊断使用显式分页符和 Word
  `lastRenderedPageBreak` 形成逻辑页；没有分页标记的文档按一个逻辑页处理。
- Word 图片缺少替代文字、含 SmartArt/图表对象或嵌套表格时会产生人工抽检警告。
- 当前任务存储是 D 盘文件型存储；后续再迁移到 PostgreSQL 任务表。
