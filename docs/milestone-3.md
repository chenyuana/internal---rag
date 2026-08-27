# 第三里程碑：本地 MinerU、串行队列与单入口界面

本里程碑把第二里程碑的解析能力变成可操作的 Windows 本地服务。用户不再选择
`Manual`、`Laws` 或 `Table` 等固定解析器，只需要上传一个 PDF。

## 已实现流程

```text
单一 PDF 上传
  -> D盘内容寻址保存
  -> 持久化串行队列
  -> 原生逐页分析
  -> 按页自动判定 native / OCR / layout
  -> MinerU 或 Docling 补充解析
  -> 统一 Document IR
  -> 清洗、切块与质量门禁
  -> 人工预览
  -> RAGFlow dry-run 或确认发布
```

当前只开放 PDF 输入，但内部的解析器注册表、页面路由和统一 IR 都保留扩展点，后续
增加 Word、Excel、图片或压缩包时不需要把解析方式重新暴露给用户。

## 本地服务

全部可控数据位于 `D:\internal-rag`：

| 服务 | 运行方式 | 地址 | 说明 |
|---|---|---|---|
| Gateway | Windows Python | `127.0.0.1:8080` | 上传、队列、预览、发布 |
| MinerU | Windows Python | `127.0.0.1:8886` | OCR、公式、表格和版面解析 |
| RAGFlow | Docker Desktop | `127.0.0.1:9380` | 知识库与检索 |
| Ollama | Windows 本机 | `127.0.0.1:11434` | 本地生成与向量模型 |

MinerU 使用独立的 uv-managed Python 3.12，避免 Anaconda 自带 Visual C++ Runtime
与 ONNX Runtime DLL 冲突。模型从 ModelScope 下载后由 `local` 模式加载；正常
解析期间不需要访问公网。

## 串行资源策略

文档处理并发固定为 1：

- Gateway 只有一个持久化任务消费者；
- MinerU API 最大并发请求数为 1；
- Gateway 重启后会把残留的 `running` 任务恢复为 `queued`；
- 同一任务不会被重复入队；
- 原文件、任务状态和解析结果都保存在 D 盘。

这是当前 8GB 显存设备的保护策略。队列状态接口：

```http
GET /api/v1/ingestion/queue
```

## 使用方式

```powershell
Set-Location D:\internal-rag\source\internal-rag
.\deploy\mineru\start.ps1
.\deploy\gateway\start.ps1
```

浏览器访问：

```text
http://127.0.0.1:8080/ingestion
```

界面支持：

- 拖放或选择 PDF；
- 查看本地服务状态；
- 查看活动任务和排队任务数量；
- 查看每页路由、质量指标和质量门禁；
- 预览前 10 个 Chunk；
- 失败任务重试；
- 生成 RAGFlow 发布计划；
- 人工确认后正式发布。

## 发布边界

默认 `RAGFLOW_API_KEY` 为空，因此可以安全执行 dry-run，但不能改写 RAGFlow。
正式发布前需要在 `.env` 中填写 RAGFlow API Key，并确认目标 Dataset ID。

`needs_review` 和 `failed` 任务仍会被质量门禁阻止，不能直接发布。

## 验收结果

使用真实扫描版法规 `CCAR-37AA 民用航空材料、零部件和机载设备技术标准规定.pdf`
完成端到端验收：

- 原文件：9 页，372,878 字节；
- 原生解析判定：9 页均需要 OCR；
- MinerU 合并：9 页均为 `remote_ocr`；
- 最终路由：`hybrid_remote`；
- 文本覆盖率：100%；
- 清洗后字符数：2,049；
- Chunk 数：17；
- 目录页：0，未把封面误判为目录；
- Unicode 替换字符：0；
- 质量门禁：`pass`；
- RAGFlow dry-run 计划 Chunk：17；
- 发布计划 SHA-256：
  `94f0fdcdacec9c0b43704823516169e39413ca7ef6344f56e40658c0388841f9`；
- RAGFlow 实际写入：0。

验收同时覆盖了失败重试：首次运行暴露 MinerU 3.4.4 Windows pipeline 缺少
`six` 兼容依赖，任务被正确标记为 `needs_review`；补齐依赖后，同一任务重试成功。
部署脚本已固定安装该依赖。
