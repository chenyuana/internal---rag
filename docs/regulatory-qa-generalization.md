# 法规问答通用能力改造

本轮针对法规文档的结构和证据处理，不在运行时代码中绑定测试题、文件名、
具体条号与主题对应关系，也不把标准答案写入知识库。

## 已接入的路径

- 原生、混合解析：在逻辑块边界拆开明确的 `Sec.` / `Section.` / `§` 标题与正文；
  仅接受行首或 `to read as follows:` 后的标题。保留原块坐标，不虚构逐行坐标。
- 扫描解析：复用标题拆分，识别 `Sec.` 标记，将条款身份传给块与切片。
- 英文附录及 Part/Subpart 标题清除上一条款的归属，避免附录继承正文条号。
- 检索：区分条款归属和正文中的交叉引用；父条款可匹配子分项，但不会匹配相似前缀。
- 单查询也可执行一次有预算限制的双语补检索；保留原问题、精确标识、知识库权限和
  document_ids。无模型时使用概念术语表，不猜测条号或答案数值。
- 证据范围：多条款问题保留所有请求条款和其子项，不从不同文档中任意选一份。
- 证据提取：明确归属的条款切片作为整体证据，不因句子配额丢掉适用条件或例外。
- 生成：中文问题面对英文证据时可进入已有受约束生成流程；保留引用、主题、数值校验。
- 确定性构建保留前置条件、后续字母分项、跨切片不同续文，避免用最长分项覆盖其他证据。
- 含省略/修订信号的来源在答案中附加范围说明；不能把局部修订冒充完整条文。
- 所问条号与所有已识别证据条号都不相符时返回待核对，不能凭主题自行换号。
- 数值识别补充英制单位、英文百分比和金额量级；数值校验支持部分中英文单位等价。

## 泛化边界

标题恢复采用结构证据。没有明确边界的 OCR 粘连正文仍可能需要重新解析，不按主题补造标题。
domain_terms.py 是可扩展的概念词表，不是完整的跨语言语义模型。未收录术语依赖已有
模型翻译和原检索。精确查询的结构化字段优先；旧切片只有正文时不保证能恢复条款身份。

范围说明依赖可观察的修订/省略信号，不能证明无标记的切片就是完整条款。
当前数值校验不等于完整的条件逻辑证明，也不自动换算不同单位或推断未注明的金额口径。
未实现从历史修订自动合成现行法规；没有旧版证据时不能生成完整前后对照。
中文生成仍需可用的答案模型，不能把单元测试通过当成实际模型准确率。

## 配置及回退

`generation.cross_language_generation` / `CROSS_LANGUAGE_GENERATION` 控制英文证据的中文生成，
默认 true。关闭后保留原 answer_mode。已有 translation.enabled、model_enabled、
max_total_translations 控制翻译及预算。补检索失败保留原结果并记录失败。

候选重建不会覆盖原产物、修改审批状态或发布外部索引。旧版本需要通过原有入库流程
发布新的切片后，结构修复才能作用于在线检索。服务需要加载本轮代码与配置。

## 离线迁移和线上回归

从项目根目录运行：

```powershell
.venv/Scripts/python.exe -m scripts.audit_regulatory_ir --input-ir <document_ir.json> --questions tests/evaluation/airworthiness_questions.jsonl --output-dir <新的候选目录>
.venv/Scripts/python.exe -m scripts.run_regulatory_questions --knowledge-base-id <知识库ID> --document-id <文档ID> --questions tests/evaluation/airworthiness_questions.jsonl --output <新的结果文件.jsonl>
```

离线脚本保留表格 ID 和单元格数据，发现改变时停止写候选。输出 candidate_chunks.jsonl 和
audit.json，仅是证据目录核查，不是回答评分。线上脚本逐题记录响应、引用、诊断和耗时，
不会默认将有响应判为通过。两个脚本均拒绝覆盖已有输出。

37 题位于 tests/evaluation/airworthiness_questions.jsonl，仅供评测，应用不会加载该文件。
gold_status 为 pending_source_review；需要逐题完成原文证据和必答要点标注后才能统计正确率。
新增泛化测试使用多个 Part 和条号、行内引用反例、中文术语、英制数值、附录边界及访问范围。
