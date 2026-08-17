# 规章文档预处理实验

这个目录用于验证“独立 Python 预处理框架 + 可插拔解析器 + RAGFlow 索引”的流程。
当前原型只安装轻量的 `pypdf` 原生文本解析器，不会下载 MinerU 模型，也不会修改
RAGFlow 中已有的知识库。

原型会先按文档质量路由：

- `native_text`：原生文本完整，可直接清洗和按“章/条/款”切分。
- `layout_review`：含表格，保留版面文本并标记后续交给 MinerU/Docling。
- `hybrid_review`：部分页面缺少有效文本，需要逐页混合解析。
- `ocr_required`：纯扫描件，禁止用空文本继续建索引。

每份文档会生成：

- `baseline.json`：原始页级文本，作为可回溯基线。
- `document_ir.json`：统一 Document IR，包含页、块、章节路径和块坐标占位字段。
- `cleaned.md`：便于人工抽检的清洗文本。
- `chunks.jsonl`：可进一步转换为 RAGFlow Add chunk API 的子块。
- `qa.json`：覆盖率、标题识别、表格提示、清洗保留率和质量门禁。

运行：

```powershell
Set-Location D:\internal-rag\source\internal-rag
.\.venv\Scripts\python.exe .\experiments\regulation_preprocess\pipeline.py `
  --folder "D:\基于测试报告的成电RAG企业知识库项目意见\成电RAG项目效果测试要求\标准\规章" `
  --output "D:\internal-rag\data\preprocess\experiments\regulations\run-20260728"
```

当前阶段的边界：

- 扫描件只做识别和拦截，不伪装成“解析成功”。
- 表格页保留 `layout_text` 和表格提示，但不会把表格文本冒充结构化单元格。
- MinerU/Docling 应作为后续 parser 插件接入相同 IR，不改变清洗、切分和质量门禁。
