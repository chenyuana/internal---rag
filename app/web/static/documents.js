"use strict";

const $ = (id) => document.getElementById(id);
const STORAGE_KEY = "internal-rag.document-workbench.v1";
const STATUS_LABELS = { pending: "待编制", drafting: "编制中", complete: "已完成" };
const DEFAULT_GUIDANCE = "围绕本节主题检索知识库，事实、数值、条款与判据均需保留来源引用；资料不足时明确列出待补信息。";
const AIRWORTHINESS_OUTLINE = [
  ["编制说明", "说明目的、适用对象、文档边界、术语和证据使用原则。"],
  ["适用范围与无人机系统概述", "界定航空器构型、任务场景、运行环境、重量与性能边界。"],
  ["适航审定依据与审定基础", "列出知识库中适用法规、标准、条款及其适用关系。"],
  ["系统级安全性要求", "覆盖功能危险分析、失效状态、风险等级与安全目标。"],
  ["结构、材料与载荷", "覆盖结构强度、疲劳、振动、材料、连接和极限载荷要求。"],
  ["动力、能源与推进系统", "覆盖动力装置、能源储备、热安全、推进系统及异常处置。"],
  ["飞行控制、导航与自主功能", "覆盖控制律、导航精度、模式转换、边界保护和自主功能。"],
  ["指挥控制链路与地面站", "覆盖链路性能、丢链处置、频谱、电磁兼容和地面控制站。"],
  ["软件与复杂电子硬件", "覆盖开发保证、配置管理、验证确认、变更控制和可追溯性。"],
  ["环境适应性与电磁兼容", "覆盖温度、高度、湿度、振动、冲击、降水及电磁环境验证。"],
  ["运行限制与应急处置", "覆盖运行包线、禁限条件、故障告警、返航、迫降与终止飞行。"],
  ["持续适航与维修保障", "覆盖检查周期、维修说明、寿命限制、故障报告和持续适航文件。"],
  ["符合性验证方法", "按检查、分析、试验、演示、相似性等方法逐项说明验证策略。"],
  ["符合性验证矩阵", "形成要求、来源、符合性方法、条件/输入、判据、记录和状态矩阵。"],
  ["结论与待补信息", "汇总已覆盖要求、证据缺口、待试验项目和后续关闭计划。"],
];
const GENERIC_OUTLINE = [
  ["编制说明", "说明文档目的、读者、范围、术语与证据使用原则。"],
  ["背景与适用范围", "界定业务背景、对象、边界条件和不适用内容。"],
  ["依据文件", "列出知识库中的法规、标准、规范和内部文件。"],
  ["总体要求", "归纳跨章节适用的总体原则、指标和约束。"],
  ["技术要求", "按专业主题组织详细要求、参数和适用条件。"],
  ["验证与验收", "说明验证方法、输入条件、判据、记录与结论。"],
  ["实施与维护", "说明责任、过程控制、变更、维护和持续改进。"],
  ["结论与待补信息", "汇总证据覆盖、遗留问题和后续行动。"],
];
const TEST_PLAN_OUTLINE = [
  ["编制说明", "说明试验目的、验证对象、适用范围、术语及文件使用原则。"],
  ["依据文件与验证要求", "列出适用法规、标准、规范、设计要求和需验证的符合性条款。"],
  ["试验对象与构型", "说明被试对象、软硬件版本、安装构型、接口、状态及偏离项。"],
  ["试验条件与资源", "明确场地、环境、设备、仪器、工装、人员资质及校准要求。"],
  ["试验项目与方法", "按项目说明目的、输入、步骤、测量参数、数据记录和注意事项。"],
  ["通过判据", "为各试验项目规定可验证、可量化且可追溯的通过与失败判据。"],
  ["安全与风险控制", "识别试验风险、停止条件、应急处置、隔离措施和责任分工。"],
  ["数据处理与结果记录", "规定原始数据、计算方法、误差处理、记录格式和可追溯要求。"],
  ["偏差与问题处置", "说明异常、偏差、不符合项的记录、分析、复验和关闭流程。"],
  ["试验矩阵与进度", "汇总要求、试验项目、构型、条件、判据、记录及计划状态。"],
];
const TECHNICAL_REPORT_OUTLINE = [
  ["摘要", "概述研究或工程任务的背景、目标、方法、主要结果和结论。"],
  ["背景与任务范围", "说明问题背景、应用场景、任务边界、对象及不包含内容。"],
  ["依据与输入资料", "列出标准规范、设计资料、试验数据和其他输入及其适用性。"],
  ["技术方案", "说明总体思路、系统组成、关键设计、方法选择和技术路线。"],
  ["分析与实施过程", "按专业逻辑记录分析方法、实施步骤、关键参数和过程证据。"],
  ["结果与验证", "呈现结果、数据、验证方法、判据、偏差及其可信度。"],
  ["问题与风险", "分析限制条件、遗留问题、技术风险、影响及处置建议。"],
  ["结论与建议", "形成与证据一致的结论，并给出后续行动和待补信息。"],
];
const OUTLINE_PRESETS = { airworthiness: AIRWORTHINESS_OUTLINE, "test-plan": TEST_PLAN_OUTLINE, "technical-report": TECHNICAL_REPORT_OUTLINE, generic: GENERIC_OUTLINE };

const state = { projects: [], activeId: null, activeSectionId: null, datasets: [], modelOptions: new Map(), datasetsLoaded: false, modelsLoaded: false, datasetsFailed: false, modelsFailed: false };
const uid = () => crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
const activeProject = () => state.projects.find((item) => item.id === state.activeId);
const activeSection = () => activeProject()?.sections.find((item) => item.id === state.activeSectionId);
const now = () => new Date().toISOString();
const blankProject = () => ({ id: uid(), title: "未命名文档", objective: "", audience: "", targetLength: "standard", documentType: "auto", kbId: "", model: null, template: null, evidenceFiles: [], sections: [], createdAt: now(), updatedAt: now() });

function loadState() {
  try { state.projects = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch { state.projects = []; }
  if (!Array.isArray(state.projects) || !state.projects.length) state.projects = [blankProject()];
  state.projects = state.projects.slice(0, 20);
  state.activeId = state.projects[0].id;
}

function saveState() {
  const project = activeProject(); if (project) project.updatedAt = now();
  state.projects.sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)));
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.projects)); $("saveState").textContent = "已保存"; }
  catch { $("saveState").textContent = "保存空间不足"; }
}

function escapeHtml(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }
function formatDate(value) { return new Date(value).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" }); }
function countChars(text) { return String(text || "").replace(/\s/g, "").length; }
function countCitations(text) { return new Set(String(text || "").match(/\[C\d+\]/g) || []).size; }

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const code = data?.error?.code || "";
    const localized = {
      ANSWER_MODEL_UNAVAILABLE: "本地答案模型在规定时间内未返回有效结果。",
      ANSWER_MODEL_INVALID_RESPONSE: "本地答案模型返回了空内容。",
      ANSWER_SEMANTIC_VALIDATION_FAILED: "生成内容未通过证据一致性校验。",
      ANSWER_SCHEMA_VALIDATION_FAILED: "生成内容的结构不完整，无法写入章节。",
      ANSWER_JSON_PARSE_FAILED: "生成模型未返回有效的结构化结果。",
    }[code];
    const detail = data?.error?.details?.validation_error;
    const message = localized || data?.error?.message || code || data?.detail || `HTTP ${response.status}`;
    throw new Error(detail ? `${message} ${detail}` : message);
  }
  return data;
}

async function loadDatasets() {
  try {
    state.datasets = await readJson(await fetch("/api/v1/admin/datasets", { headers: { "X-User-ID": "dev" } }));
    state.datasetsLoaded = true; state.datasetsFailed = !state.datasets.length;
    const select = $("kbSelect"); select.innerHTML = "";
    state.datasets.forEach((dataset) => { const option = document.createElement("option"); option.value = dataset.id; option.textContent = dataset.name; select.appendChild(option); });
    const project = activeProject(); if (project && state.datasets.some((item) => item.id === project.kbId)) select.value = project.kbId; else if (project && state.datasets[0]) project.kbId = state.datasets[0].id;
    updateServicePill();
  } catch (error) {
    state.datasetsLoaded = true; state.datasetsFailed = true; $("kbSelect").innerHTML = '<option value="">知识库不可用</option>'; showMessage(error.message, true); updateServicePill();
  }
  renderReview(); saveState();
}

function modelOptionKey(sourceId, modelName) { return `${sourceId}\u001f${modelName}`; }
async function loadModels(preferredModel = null) {
  try {
    const catalog = await readJson(await fetch("/api/v1/admin/answer-models", { headers: { "X-User-ID": "dev" } })); const select = $("modelSelect"); select.innerHTML = ""; state.modelOptions.clear();
    (catalog.sources || []).forEach((source) => { const group = document.createElement("optgroup"); group.label = source.name; (source.models || []).forEach((modelName) => { const key = modelOptionKey(source.source_id, modelName); state.modelOptions.set(key, { source_id: source.source_id, model_name: modelName }); const option = document.createElement("option"); option.value = key; option.textContent = modelName; option.disabled = source.available === false; group.appendChild(option); }); select.appendChild(group); });
    state.modelsLoaded = true; state.modelsFailed = !state.modelOptions.size; if (!state.modelOptions.size) select.innerHTML = '<option value="">模型不可用</option>'; const project = activeProject(); const preferredKey = preferredModel ? modelOptionKey(preferredModel.source_id, preferredModel.model_name) : ""; const savedKey = project?.model ? modelOptionKey(project.model.source_id, project.model.model_name) : ""; const defaultKey = modelOptionKey(catalog.default_source_id || "", catalog.default_model || ""); if (state.modelOptions.has(preferredKey)) select.value = preferredKey; else if (state.modelOptions.has(savedKey)) select.value = savedKey; else if (state.modelOptions.has(defaultKey)) select.value = defaultKey; else if (state.modelOptions.size) select.value = state.modelOptions.keys().next().value; if (project) project.model = state.modelOptions.get(select.value) || null; updateServicePill(); saveState();
  } catch (error) { state.modelsLoaded = true; state.modelsFailed = true; $("modelSelect").innerHTML = '<option value="">模型不可用</option>'; showMessage(error.message, true); updateServicePill(); }
}

function updateServicePill() { if (!state.datasetsLoaded || !state.modelsLoaded) { $("servicePill").textContent = "正在加载编制服务…"; $("servicePill").className = "service-pill"; return; } if (state.datasetsFailed || state.modelsFailed) { $("servicePill").textContent = "编制服务不可用"; $("servicePill").className = "service-pill bad"; return; } $("servicePill").textContent = `${state.datasets.length} 个知识库 · ${state.modelOptions.size} 个模型`; $("servicePill").className = "service-pill ok"; }

function renderProjects() {
  $("projectCount").textContent = String(state.projects.length); const list = $("projectList"); list.innerHTML = "";
  state.projects.forEach((project) => {
    const row = document.createElement("div"); row.className = `project-item${project.id === state.activeId ? " active" : ""}`;
    row.innerHTML = `<button class="project-select" data-project="${escapeHtml(project.id)}"><span class="project-title">${escapeHtml(project.title || "未命名文档")}</span><span class="project-meta">${project.sections.length} 节 · ${formatDate(project.updatedAt)}</span></button><button class="project-delete" data-delete-project="${escapeHtml(project.id)}" aria-label="删除文档">×</button>`;
    list.appendChild(row);
  });
}

function renderOutline() {
  const project = activeProject(); const list = $("outlineList"); list.innerHTML = "";
  if (!project?.sections.length) list.innerHTML = '<div class="outline-empty">填写文档任务并生成大纲后，章节将在这里显示。</div>';
  else project.sections.forEach((section, index) => {
    const row = document.createElement("div"); row.className = `outline-item${section.id === state.activeSectionId ? " active" : ""}`; row.dataset.section = section.id;
    row.innerHTML = `<button class="outline-select" type="button" data-select-section="${escapeHtml(section.id)}" aria-label="打开第 ${index + 1} 章"><span class="section-number">${String(index + 1).padStart(2, "0")}</span></button><div class="outline-copy"><input class="outline-title-input" data-outline-title="${escapeHtml(section.id)}" value="${escapeHtml(section.title)}" maxlength="140" aria-label="第 ${index + 1} 章标题" /><span class="outline-status"><i class="status-dot ${escapeHtml(section.status)}"></i>${STATUS_LABELS[section.status]}</span></div><button class="outline-delete" type="button" data-delete-section="${escapeHtml(section.id)}" aria-label="删除第 ${index + 1} 章“${escapeHtml(section.title)}”" title="删除本章">⌫</button>`; list.appendChild(row);
  });
  const complete = project?.sections.filter((item) => item.status === "complete").length || 0; const total = project?.sections.length || 0; const percent = total ? Math.round(complete / total * 100) : 0;
  $("progressText").textContent = `${complete} / ${total} 节完成`; $("progressPercent").textContent = `${percent}%`; $("progressBar").style.width = `${percent}%`;
}

function renderFiles() {
  const project = activeProject(); const templateList = $("templateList"); templateList.innerHTML = "";
  if (project?.template) templateList.innerHTML = `<div class="file-row"><span title="${escapeHtml(project.template.name)}">${escapeHtml(project.template.name)}</span><button type="button" data-remove-template aria-label="移除结构文件/格式模板">×</button></div>`;
  const evidenceList = $("evidenceList"); evidenceList.innerHTML = "";
  (project?.evidenceFiles || []).forEach((file, index) => { const row = document.createElement("div"); row.className = "file-row"; row.innerHTML = `<span title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span><button type="button" data-remove-evidence="${index}" aria-label="移除补充证据">×</button>`; evidenceList.appendChild(row); });
}

function renderEditor() {
  const project = activeProject(); const section = activeSection(); const isSection = Boolean(section);
  $("setupView").hidden = isSection; $("sectionView").hidden = !isSection; $("editorBreadcrumb").textContent = isSection ? section.title : "项目设置";
  if (!project) return;
  if (!isSection) {
    $("documentTitle").value = project.title === "未命名文档" ? "" : project.title; $("documentObjective").value = project.objective || ""; $("documentAudience").value = project.audience || ""; $("targetLength").value = project.targetLength || "standard"; $("documentType").value = project.documentType || "auto"; if (project.kbId) $("kbSelect").value = project.kbId; const modelKey = project.model ? modelOptionKey(project.model.source_id, project.model.model_name) : ""; if (state.modelOptions.has(modelKey)) $("modelSelect").value = modelKey; renderFiles();
  } else {
    $("sectionTitle").value = section.title; $("sectionStatus").value = section.status; $("sectionContent").value = section.content || ""; $("sectionGuidance").value = section.guidance || DEFAULT_GUIDANCE; $("generationState").textContent = ""; $("generationState").className = "generation-state"; updateSectionStats();
  }
}

function qualityChecks(project) {
  const content = project.sections.map((item) => item.content || "").join("\n"); const complete = project.sections.filter((item) => item.status === "complete").length;
  return [
    { pass: project.title && project.title !== "未命名文档", label: "已填写明确的文档名称" },
    { pass: Boolean(project.objective?.trim()), label: "已说明编制目标和内容边界" },
    { pass: Boolean(project.kbId), label: "已选择知识库证据范围" },
    { pass: Boolean(project.sections.length), label: "已建立分章节编制大纲" },
    { pass: Boolean(project.template), label: "已上传结构文件/格式模板（可选）", optional: true },
    { pass: countCitations(content) > 0, label: "正文包含可追溯引用" },
    { pass: project.sections.length > 0 && complete === project.sections.length, label: "所有章节均标记为已完成" },
  ];
}

function renderReview() {
  const project = activeProject(); if (!project) return; const allContent = project.sections.map((item) => item.content || "").join("\n"); const chars = countChars(allContent); const citations = countCitations(allContent); const complete = project.sections.filter((item) => item.status === "complete").length;
  $("totalStats").textContent = `${chars.toLocaleString()} 字 · ${citations} 条引用`; $("coverageStats").textContent = project.sections.length ? `${complete}/${project.sections.length} 个章节完成` : "尚未建立大纲";
  $("templateSummary").textContent = project.template?.name || "未上传"; const kbName = state.datasets.find((item) => item.id === project.kbId)?.name || (project.kbId ? "已选择知识库" : "未选择知识库"); $("evidenceSummary").textContent = `${kbName} · ${project.evidenceFiles.length} 份补充资料`;
  const checks = qualityChecks(project); const passedRequired = checks.filter((item) => item.pass && !item.optional).length; const required = checks.filter((item) => !item.optional).length; const score = Math.round(passedRequired / required * 100); $("qaScore").textContent = score;
  const list = $("checkList"); list.innerHTML = ""; checks.forEach((check) => { const item = document.createElement("li"); item.className = `check-item ${check.pass ? "pass" : (check.optional ? "warn" : "")}`; item.innerHTML = `<span class="check-mark">${check.pass ? "✓" : (check.optional ? "!" : "·")}</span><span>${escapeHtml(check.label)}</span>`; list.appendChild(item); });
  const next = !project.title || project.title === "未命名文档" ? "先填写文档名称。" : !project.objective ? "补充编制目标和内容深度要求。" : !project.sections.length ? "生成大纲并检查章节覆盖是否完整。" : complete < project.sections.length ? `在当前工作台逐节检索证据并编制剩余 ${project.sections.length - complete} 个章节。` : citations === 0 ? "章节已完成，但尚未检测到 [C1] 形式的引用。" : "草稿已通过基础检查，可以导出并进入 DOCX 套版阶段。"; $("nextStep").textContent = next; renderSectionEvidence();
}

function renderSectionEvidence() { const section = activeSection(); const panel = $("sectionEvidence"); panel.innerHTML = ""; if (!section) { panel.innerHTML = "<p>选择章节后查看本节引用。</p>"; return; } const usedIds = new Set((String(section.content || "").match(/\[C\d+\]/g) || []).map((marker) => marker.slice(1, -1))); const citations = (section.citations || []).filter((citation) => usedIds.has(citation.citation_id)); const missing = section.missingInformation || []; if (!citations.length && !missing.length) { panel.innerHTML = "<p>生成章节后在这里核对引用与缺失信息。</p>"; return; } citations.forEach((citation) => { const item = document.createElement("div"); item.className = "evidence-mini"; const quote = String(citation.quote || "").replace(/\s+/g, " ").slice(0, 180); item.innerHTML = `<strong>[${escapeHtml(citation.citation_id)}]</strong><span title="${escapeHtml(citation.document_name || "")}">${escapeHtml(citation.document_name || "未知文档")}</span>${quote ? `<small>${escapeHtml(quote)}${quote.length >= 180 ? "…" : ""}</small>` : ""}`; panel.appendChild(item); }); missing.forEach((message) => { const item = document.createElement("div"); item.className = "missing-mini"; item.textContent = `待补：${message}`; panel.appendChild(item); }); }

function renderAll() { renderProjects(); renderOutline(); renderEditor(); renderReview(); }
function showMessage(message, error = false) { $("formMessage").textContent = message || ""; $("formMessage").className = `form-message${error ? " error" : " ok"}`; }
function persistField(key, value) { const project = activeProject(); if (!project) return; project[key] = value; if (key === "title" && !value.trim()) project.title = "未命名文档"; $("saveState").textContent = "保存中…"; saveState(); renderProjects(); renderReview(); }

function createProject() { const project = blankProject(); state.projects.unshift(project); state.activeId = project.id; state.activeSectionId = null; saveState(); renderAll(); $("documentTitle").focus(); }
function switchProject(id) { state.activeId = id; state.activeSectionId = null; renderAll(); }
async function deleteProject(id) { const dialog = $("confirmDialog"); dialog.showModal(); const result = await new Promise((resolve) => dialog.addEventListener("close", () => resolve(dialog.returnValue), { once: true })); if (result !== "confirm") return; state.projects = state.projects.filter((item) => item.id !== id); if (!state.projects.length) state.projects = [blankProject()]; state.activeId = state.projects[0].id; state.activeSectionId = null; saveState(); renderAll(); }

function buildOutline() {
  const project = activeProject(); const title = $("documentTitle").value.trim(); const objective = $("documentObjective").value.trim();
  if (!title || !objective) { showMessage("请先填写文档名称和编制目标。", true); return; }
  project.title = title; project.objective = objective; project.audience = $("documentAudience").value.trim(); project.targetLength = $("targetLength").value; project.documentType = $("documentType").value; project.kbId = $("kbSelect").value;
  if (project.sections.length && !window.confirm("重新生成会替换当前大纲和章节正文，是否继续？")) return;
  const recommendedType = /试验|测试|验证计划/.test(`${title} ${objective}`) ? "test-plan" : /技术报告|分析报告|研究报告/.test(`${title} ${objective}`) ? "technical-report" : /适航|符合性|审定/.test(`${title} ${objective}`) ? "airworthiness" : "generic";
  const source = OUTLINE_PRESETS[project.documentType === "auto" ? recommendedType : project.documentType] || GENERIC_OUTLINE;
  project.sections = source.map(([sectionTitle, guidance]) => ({ id: uid(), title: sectionTitle, guidance, status: "pending", content: "" })); state.activeSectionId = project.sections[0].id; saveState(); showMessage(""); renderAll();
}

function addSection() { const project = activeProject(); if (!project) return; const section = { id: uid(), title: `新增章节 ${project.sections.length + 1}`, guidance: DEFAULT_GUIDANCE, status: "pending", content: "" }; project.sections.push(section); state.activeSectionId = section.id; saveState(); renderAll(); $("sectionTitle").select(); }
function deleteSectionById(sectionId) { const project = activeProject(); const section = project?.sections.find((item) => item.id === sectionId); if (!project || !section || !window.confirm(`确认删除“${section.title}”？该章正文也会一并删除。`)) return; const index = project.sections.findIndex((item) => item.id === section.id); project.sections = project.sections.filter((item) => item.id !== section.id); if (state.activeSectionId === section.id) state.activeSectionId = project.sections[index]?.id || project.sections[index - 1]?.id || null; saveState(); renderAll(); }
function deleteSection() { const section = activeSection(); if (section) deleteSectionById(section.id); }
function updateSectionStats() { const section = activeSection(); if (!section) return; $("sectionStats").textContent = `${countChars(section.content).toLocaleString()} 字 · ${countCitations(section.content)} 条引用`; }
async function generateSection() {
  const project = activeProject(); const section = activeSection();
  if (!project || !section) return;
  if (!project.kbId) { setGenerationState("请先选择知识库。", true); return; }
  if (!project.model) { setGenerationState("请先选择章节生成模型。", true); return; }
  const button = $("generateSection"); button.disabled = true; button.textContent = "正在检索与生成…";
  const startedAt = Date.now();
  const progressTimer = window.setInterval(() => {
    const seconds = Math.floor((Date.now() - startedAt) / 1000);
    const stage = seconds < 20 ? "检索与整理知识库证据" : seconds < 90 ? "本地模型正在编制长文" : "正在完成长文生成与证据校验";
    setGenerationState(`${stage} · 已用时 ${seconds} 秒`);
  }, 1000);
  setGenerationState("正在检索与整理知识库证据 · 已用时 0 秒");
  try {
    const response = await fetch("/api/v1/documents/sections/generate", {
      method: "POST", headers: { "Content-Type": "application/json", "X-User-ID": "dev" },
      body: JSON.stringify({ document_title: project.title, objective: project.objective, audience: project.audience || null, target_length: project.targetLength || "standard", section_title: section.title, section_guidance: section.guidance || DEFAULT_GUIDANCE, knowledge_base_ids: [project.kbId], model: project.model, supplemental_documents: project.evidenceFiles.map((item) => ({ name: item.name, text: item.preview })) }),
    });
    const data = await readJson(response);
    section.lastQualityResult = { passed: data.quality_passed, generatedChars: data.generated_char_count, targetMinChars: data.target_min_chars, warnings: data.coverage_warnings || [], advisory: data.advisory_warnings || [] };
    section.missingInformation = data.missing_information || [];
    section.generationStatus = data.status;
    if (data.content) {
      section.content = data.content; section.citations = data.citations || []; section.claims = data.claims || []; section.status = "drafting";
    }
    saveState(); renderAll();
    const advisoryNote = (data.advisory_warnings || []).join("；");
    if (data.quality_passed && data.content) setGenerationState(`生成通过章节门禁 · ${data.generated_char_count} 字 · ${data.evidence_document_count} 份文档 · ${data.evidence_sentence_count} 条证据${advisoryNote ? `；建议：${advisoryNote}` : ""}`);
    else if (data.content) setGenerationState(`生成存在硬性警告（必需要素或范围问题），正文已保留供人工核对 · ${data.generated_char_count} 字${advisoryNote ? `；建议：${advisoryNote}` : ""}`, true);
    else setGenerationState(`本节未生成正文${advisoryNote ? ` · ${advisoryNote}` : ""}`, true);
  } catch (error) { setGenerationState(error.message, true); }
  finally { window.clearInterval(progressTimer); button.disabled = false; button.textContent = "检索证据并生成本节"; }
}
function setGenerationState(message, error = false) { $("generationState").textContent = message || ""; $("generationState").className = `generation-state${error ? " error" : ""}`; }

async function parseFile(file) { const form = new FormData(); form.append("file", file); return readJson(await fetch("/api/v1/chat/reference-documents/parse", { method: "POST", body: form })); }
function closeModelDialog() { $("modelDialog").close(); }
async function connectModel(event) { event.preventDefault(); const button = $("connectModelButton"); const message = $("connectionMessage"); button.disabled = true; button.textContent = "正在连接…"; try { const created = await readJson(await fetch("/api/v1/admin/model-connections", { method: "POST", headers: { "Content-Type": "application/json", "X-User-ID": "dev" }, body: JSON.stringify({ name: $("connectionName").value.trim(), base_url: $("connectionBaseUrl").value.trim(), api_key: $("connectionApiKey").value }) })); const source = created.source; await loadModels({ source_id: source.source_id, model_name: source.default_model || source.models[0] }); $("connectionApiKey").value = ""; message.textContent = `已连接 ${source.name}，并选为章节生成模型。`; message.className = "dialog-message ok"; setTimeout(closeModelDialog, 600); } catch (error) { message.textContent = error.message; message.className = "dialog-message error"; } finally { button.disabled = false; button.textContent = "连接并读取模型"; } }
async function loadTemplate(file) {
  if (!file) return;
  showMessage("正在解析结构文件…");
  try {
    const parsed = await parseFile(file);
    const project = activeProject();
    project.template = { name: parsed.name, truncated: parsed.truncated, preview: parsed.text.slice(0, 2000) };
    const outlineSections = OutlineParser.parseOutlineSections(parsed.text);
    let replaced = false;
    if (outlineSections.length >= 2) {
      const hasExisting = project.sections.length > 0;
      if (!hasExisting || window.confirm(`已从结构文件识别出 ${outlineSections.length} 个章节，是否替换当前大纲？`)) {
        project.sections = outlineSections.map((item) => ({ id: uid(), title: item.title, guidance: item.guidance || DEFAULT_GUIDANCE, status: "pending", content: "" }));
        state.activeSectionId = project.sections[0]?.id || null;
        replaced = true;
      }
    }
    saveState(); renderAll();
    if (replaced) showMessage(`已按结构文件生成 ${outlineSections.length} 章大纲；编制要求已按章节主题自动生成，可继续编辑。`);
    else if (outlineSections.length >= 2) showMessage("结构文件已保存为格式模板；大纲未替换。");
    else showMessage(parsed.truncated ? "格式模板已读取，文本较长，当前仅保存结构预览；未识别出编号章节列表。" : "格式模板已加载；未识别出编号章节列表（需形如“1.范围；2.…”），仅保存为格式模板。");
  } catch (error) { showMessage(error.message, true); }
  finally { $("templateFile").value = ""; }
}
async function loadEvidence(files) { if (!files.length) return; showMessage(`正在解析 ${files.length} 份补充资料…`); const project = activeProject(); const failures = []; for (const file of files) { try { const parsed = await parseFile(file); if (!project.evidenceFiles.some((item) => item.name === parsed.name)) project.evidenceFiles.push({ name: parsed.name, truncated: parsed.truncated, preview: parsed.text.slice(0, 4000) }); } catch (error) { failures.push(`${file.name}: ${error.message}`); } } saveState(); renderFiles(); renderReview(); showMessage(failures.length ? failures.join("；") : "补充资料已加载；将作为事实证据使用。", failures.length > 0); $("evidenceFiles").value = ""; }

async function suggestGuidances(items) {
  const project = activeProject();
  const response = await fetch("/api/v1/documents/sections/guidance-suggest", {
    method: "POST", headers: { "Content-Type": "application/json", "X-User-ID": "dev" },
    body: JSON.stringify({ document_title: project.title, objective: project.objective, audience: project.audience || null, template_preview: project.template?.preview || null, sections: items.map((item) => ({ title: item.title, guidance: item.guidance || null })), model: project.model }),
  });
  const data = await readJson(response);
  return data.sections || [];
}
async function generateAllGuidances() {
  const project = activeProject(); if (!project) return;
  if (!project.sections.length) { showMessage("请先生成大纲。", true); return; }
  if (!project.model) { showMessage("请先选择章节生成模型。", true); return; }
  if (!window.confirm(`将用模型为 ${project.sections.length} 个章节生成编制要求并替换当前要求（生成后仍可编辑），继续？`)) return;
  showMessage("正在为各章节生成编制要求…");
  try {
    const suggested = await suggestGuidances(project.sections.map((section) => ({ title: section.title, guidance: section.guidance })));
    const byTitle = new Map(suggested.map((item) => [item.title, item.guidance]));
    let applied = 0;
    for (const section of project.sections) {
      const guidance = byTitle.get(section.title);
      if (guidance) { section.guidance = guidance; applied += 1; }
    }
    saveState(); renderAll();
    showMessage(`已为 ${applied} 个章节生成编制要求，可继续编辑。`);
  } catch (error) { showMessage(error.message, true); }
}
async function optimizeSectionGuidance() {
  const project = activeProject(); const section = activeSection();
  if (!project || !section) return;
  if (!project.model) { showMessage("请先选择章节生成模型。", true); return; }
  showMessage("正在生成本节编制要求…");
  try {
    const suggested = await suggestGuidances([{ title: section.title, guidance: section.guidance }]);
    const item = suggested.find((entry) => entry.title === section.title);
    if (item && item.guidance) { section.guidance = item.guidance; saveState(); renderAll(); showMessage("本节编制要求已生成，可继续编辑。"); }
    else showMessage("模型未生成本节要求，请重试或手动编写。", true);
  } catch (error) { showMessage(error.message, true); }
}

function exportDraft() {
  const project = activeProject(); if (!project) return; const kbName = state.datasets.find((item) => item.id === project.kbId)?.name || project.kbId || "未选择";
  const lines = [`# ${project.title}`, "", `> 编制目标：${project.objective || "未填写"}`, `> 使用对象：${project.audience || "未填写"}`, `> 知识库：${kbName}`, `> 格式模板：${project.template?.name || "未上传"}`, `> 补充证据：${project.evidenceFiles.map((item) => item.name).join("、") || "无"}`, ""];
  project.sections.forEach((section, index) => {
    lines.push(`## ${index + 1}. ${section.title}`, "", section.content || "[待编制]", "");
    const usedIds = new Set((String(section.content || "").match(/\[C\d+\]/g) || []).map((marker) => marker.slice(1, -1)));
    const usedCitations = (section.citations || []).filter((citation) => usedIds.has(citation.citation_id));
    if (usedCitations.length) {
      lines.push("### 本节引用", "");
      usedCitations.forEach((citation) => lines.push(`- [${citation.citation_id}] ${citation.document_name || "未知文档"}${citation.page_number ? `，第 ${citation.page_number} 页` : ""}：${String(citation.quote || "").replace(/\s+/g, " ")}`));
      lines.push("");
    }
    if ((section.missingInformation || []).length) {
      lines.push("### 本节待补信息", "", ...(section.missingInformation.map((item) => `- ${item}`)), "");
    }
  });
  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" }); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `${project.title.replace(/[\\/:*?"<>|]/g, "_") || "文档草稿"}.md`; link.click(); URL.revokeObjectURL(link.href);
}

$("newProject").addEventListener("click", createProject);
$("projectList").addEventListener("click", (event) => { const select = event.target.closest("[data-project]"); const remove = event.target.closest("[data-delete-project]"); if (select) switchProject(select.dataset.project); if (remove) deleteProject(remove.dataset.deleteProject); });
$("outlineList").addEventListener("click", (event) => { const remove = event.target.closest("[data-delete-section]"); if (remove) { deleteSectionById(remove.dataset.deleteSection); return; } const select = event.target.closest("[data-select-section]"); if (!select) return; state.activeSectionId = select.dataset.selectSection; renderAll(); });
$("outlineList").addEventListener("input", (event) => { const input = event.target.closest("[data-outline-title]"); if (!input) return; const project = activeProject(); const section = project?.sections.find((item) => item.id === input.dataset.outlineTitle); if (!section) return; section.title = input.value || "未命名章节"; saveState(); if (state.activeSectionId === section.id) { $("sectionTitle").value = section.title; $("editorBreadcrumb").textContent = section.title; } });
$("outlineList").addEventListener("keydown", (event) => { if (event.key === "Enter" && event.target.matches("[data-outline-title]")) event.target.blur(); });
$("addSection").addEventListener("click", addSection); $("buildOutline").addEventListener("click", buildOutline); $("generateSection").addEventListener("click", generateSection); $("deleteSection").addEventListener("click", deleteSection); $("exportDraft").addEventListener("click", exportDraft); $("generateGuidance").addEventListener("click", generateAllGuidances); $("optimizeGuidance").addEventListener("click", optimizeSectionGuidance);
$("openModelDialog").addEventListener("click", () => { $("connectionMessage").textContent = ""; $("modelDialog").showModal(); }); $("closeModelDialog").addEventListener("click", closeModelDialog); $("cancelModelConnection").addEventListener("click", closeModelDialog); $("modelConnectionForm").addEventListener("submit", connectModel); $("modelDialog").addEventListener("click", (event) => { if (event.target === $("modelDialog")) closeModelDialog(); });
$("documentTitle").addEventListener("input", (event) => persistField("title", event.target.value)); $("documentObjective").addEventListener("input", (event) => persistField("objective", event.target.value)); $("documentAudience").addEventListener("input", (event) => persistField("audience", event.target.value)); $("targetLength").addEventListener("change", (event) => persistField("targetLength", event.target.value)); $("documentType").addEventListener("change", (event) => persistField("documentType", event.target.value)); $("kbSelect").addEventListener("change", (event) => persistField("kbId", event.target.value)); $("modelSelect").addEventListener("change", (event) => persistField("model", state.modelOptions.get(event.target.value) || null));
$("sectionTitle").addEventListener("input", (event) => { const section = activeSection(); if (!section) return; section.title = event.target.value || "未命名章节"; saveState(); renderOutline(); $("editorBreadcrumb").textContent = section.title; });
$("sectionGuidance").addEventListener("input", (event) => { const section = activeSection(); if (!section) return; section.guidance = event.target.value; saveState(); });
$("sectionStatus").addEventListener("change", (event) => { const section = activeSection(); if (!section) return; section.status = event.target.value; saveState(); renderOutline(); renderReview(); });
$("sectionContent").addEventListener("input", (event) => { const section = activeSection(); if (!section) return; section.content = event.target.value; section.claims = []; section.generationStatus = null; section.missingInformation = (section.missingInformation || []).filter((message) => !/^(部分生成内容未通过|生成结果未通过整体证据校验|已省略未通过证据校验的结论)/.test(message)); if (section.status === "pending" && section.content.trim()) { section.status = "drafting"; $("sectionStatus").value = "drafting"; } saveState(); updateSectionStats(); renderOutline(); renderReview(); });
$("chooseTemplate").addEventListener("click", () => $("templateFile").click()); $("templateFile").addEventListener("change", (event) => loadTemplate(event.target.files[0])); $("chooseEvidence").addEventListener("click", () => $("evidenceFiles").click()); $("evidenceFiles").addEventListener("change", (event) => loadEvidence([...event.target.files]));
$("templateList").addEventListener("click", (event) => { if (!event.target.closest("[data-remove-template]")) return; activeProject().template = null; saveState(); renderFiles(); renderReview(); });
$("evidenceList").addEventListener("click", (event) => { const button = event.target.closest("[data-remove-evidence]"); if (!button) return; activeProject().evidenceFiles.splice(Number(button.dataset.removeEvidence), 1); saveState(); renderFiles(); renderReview(); });

loadState(); renderAll(); Promise.allSettled([loadDatasets(), loadModels()]);
