# 第二里程碑：混合解析与 RAGFlow 发布

本里程碑保持“用户只上传文件，不选择解析器”的产品形态。系统自动执行以下流程：

1. 所有 PDF 先运行本地原生文本解析。
2. `ocr_required` 和 `low_text_review` 页面在 MinerU 启用时进入远程 OCR。
3. `layout_review` 页面在 Docling 启用时进入远程版面解析。
4. 只有远程结果带有可靠页码时才覆盖对应页面；无法映射页码的结果保留为质量警告。
5. `needs_review` 和 `failed` 任务禁止发布到 RAGFlow。

## 解析服务

MinerU 使用 RAGFlow 0.26.4 所采用的 `/file_parse` ZIP 协议；Docling 使用
`/v1/convert/source`，并兼容 `/v1alpha/convert/source`。

两个远程服务默认关闭：

```dotenv
MINERU_ENABLED=false
MINERU_BASE_URL=http://127.0.0.1:8886
MINERU_HEALTH_PATH=/openapi.json
MINERU_TIMEOUT_SECONDS=1800
MINERU_BACKEND=pipeline
MINERU_PARSE_METHOD=auto

DOCLING_ENABLED=false
DOCLING_BASE_URL=http://127.0.0.1:5001
DOCLING_HEALTH_PATH=/openapi.json
DOCLING_TIMEOUT_SECONDS=1800
DOCLING_PARSE_METHOD=raw
```

启用后先检查：

```http
GET /api/v1/ingestion/parsers/status
```

只有对应服务返回 `ready=true` 时，才提交需要 OCR 或复杂版面解析的文件。

## RAGFlow 发布

发布前演练不会连接或修改 RAGFlow：

```http
POST /api/v1/ingestion/jobs/{job_id}/publish
Content-Type: application/json

{
  "dataset_id": "ragflow-dataset-id",
  "dry_run": true
}
```

正式发布时将 `dry_run` 改为 `false`。系统会：

1. 上传原 PDF，但不触发 RAGFlow 内置解析。
2. 把已经清洗、带章节和页码的 Chunk 逐条写入文档。
3. 保存 RAGFlow document ID 和逐 Chunk 进度。
4. 重试时读取已有 Chunk，并跳过内容完全相同的项目。

发布计划由源文件 SHA-256、目标数据集和 Chunk 内容生成确定性哈希。相同计划已经
发布成功时再次调用不会重复写入。

正式发布必须在 `.env` 中配置：

```dotenv
RAGFLOW_API_KEY=ragflow生成的API密钥
```

默认 API Key 为空，因此当前环境只会执行 dry-run，不会误写现有知识库。

## 第二里程碑验收样例

第一里程碑生成的 CCAR-21-R5 结果已完成 dry-run：

- 文档页数：107
- 计划发布 Chunk：178
- 目标数据集：`regulations-smoke`
- 发布计划 SHA-256：
  `8e99b462115ceb0bc5ed935800f5a1bd0bab65532640d33be2908142572cf984`
- RAGFlow 写入次数：0
