const state = {
  window: 15,
  stats: null,
  services: null,
  ragas: { status: null, taskId: null, pollTimer: null, mode: "question", benchmarks: [], selectedBenchmarkId: null, cases: [], startedAt: null },
  statsTimer: null,
  servicesTimer: null,
  lastStatsFetch: 0,
};

const el = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
})[char]);
const fmtMs = (value) => value == null ? "—" : `${Number(value).toFixed(1)} ms`;
const fmtNum = (value, digits = 2) => value == null ? "—" : Number(value).toFixed(digits);
const fmtPercent = (value) => value == null ? "—" : `${Math.round(Number(value) * 100)}%`;
const fmtTime = (ts) => {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
};

async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function setStatus(text, kind = "ready") {
  el("storeStatus").textContent = text;
  el("storeStatus").className = `service-pill ${kind}`;
}

function refreshHint() {
  const ago = state.lastStatsFetch ? Math.round((Date.now() - state.lastStatsFetch) / 1000) : null;
  el("refreshHint").textContent = ago == null ? "自动刷新中" : `${ago}s 前更新`;
}

/* ---------- services ---------- */

function renderServices() {
  const services = state.services;
  if (!services) return;
  const deps = services.dependencies || [];
  const readyCount = deps.filter((d) => d.status === "ready").length;
  el("servicesMeta").textContent = `${readyCount} / ${deps.length} 个依赖就绪 · ${fmtTime(services.checked_at)}`;
  el("serviceGrid").innerHTML = deps.map((dep) => {
    const label = dep.status === "ready" ? "就绪" : dep.status === "disabled" ? "停用" : "不可用";
    const detail = dep.latency_ms != null ? `${fmtNum(dep.latency_ms, 0)} ms` : (dep.detail || "");
    return `<span class="service-chip ${escapeHtml(dep.status)}">
      <span class="svc-name">${escapeHtml(dep.name)}</span>
      <span class="svc-detail">${escapeHtml(label)}${detail ? ` · ${escapeHtml(detail)}` : ""}</span>
    </span>`;
  }).join("");
}

async function loadServices() {
  try {
    state.services = await api("/api/v1/monitor/services");
    renderServices();
  } catch (_) {
    el("serviceGrid").innerHTML = `<span class="service-chip unavailable"><span class="svc-name">探测失败</span></span>`;
  }
}

/* ---------- metrics ---------- */

function metric(label, value, hint) {
  return `<article class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(hint)}</small></article>`;
}

function renderMetrics(stats) {
  const totals = stats.totals || {};
  const latency = stats.latency_ms || {};
  const modelCalls = stats.model_calls || {};
  const retrievalCalls = stats.retrieval_calls || {};
  const p99 = latency.p99;
  const warn = p99 != null && p99 > 60000;
  el("metricGrid").innerHTML = [
    metric("请求数（窗口内）", totals.requests ?? 0, "成功 + 失败"),
    metric("成功率", fmtPercent(totals.success_rate), `${totals.success ?? 0} 成功 / ${totals.failed ?? 0} 失败`),
    metric("平均耗时", fmtMs(latency.avg), `最大 ${fmtMs(latency.max)}`),
    metric("P50 / P90", `${fmtMs(latency.p50)} / ${fmtMs(latency.p90)}`, "延迟中位数 / 90 分位"),
    metric("P95 / P99", `${fmtMs(latency.p95)} / ${fmtMs(latency.p99)}`, warn ? "⚠ P99 超过 60s，注意慢请求" : "95 分位 / 99 分位"),
    metric("平均模型调用", fmtNum(modelCalls.avg, 1), `共 ${modelCalls.total ?? 0} 次`),
    metric("平均检索调用", fmtNum(retrievalCalls.avg, 1), `共 ${retrievalCalls.total ?? 0} 次`),
    metric("网关已运行", `${Math.floor((stats.uptime_seconds || 0) / 60)} 分钟`, "进程启动至今（内存缓存窗口）"),
  ].join("");
}

/* ---------- bars ---------- */

function barRows(entries, maxValue, valueSuffix = " ms") {
  if (!entries || !Object.keys(entries).length) {
    return `<div class="bar-empty">暂无数据</div>`;
  }
  const max = maxValue || Math.max(...Object.values(entries), 1);
  return Object.entries(entries).map(([label, value]) => {
    const ratio = max > 0 ? Number(value) / max : 0;
    const width = Math.max(2, Math.min(100, Math.round(ratio * 100)));
    const tone = ratio >= 0.9 ? "bad" : ratio >= 0.6 ? "warn" : "";
    const text = Number.isInteger(value) ? String(value) : fmtNum(value, 1);
    return `<div class="bar-row">
      <span class="bar-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
      <div class="bar-track"><div class="bar-fill ${tone}" style="width:${width}%"></div></div>
      <span class="bar-value">${escapeHtml(text)}${escapeHtml(valueSuffix)}</span>
    </div>`;
  }).join("");
}

function renderStageBars(stats) {
  el("stageBars").innerHTML = barRows(stats.stage_timings_ms, null, " ms");
}

function renderErrorBars(stats) {
  const errors = stats.error_distribution || {};
  const total = Object.values(errors).reduce((a, b) => a + b, 0);
  el("errorBars").innerHTML = total
    ? Object.entries(errors).map(([type, count]) => {
        const ratio = count / total;
        return `<div class="bar-row">
          <span class="bar-label" title="${escapeHtml(type)}">${escapeHtml(type)}</span>
          <div class="bar-track"><div class="bar-fill bad" style="width:${Math.max(2, Math.round(ratio * 100))}%"></div></div>
          <span class="bar-value">${count} 次</span>
        </div>`;
      }).join("")
    : `<div class="bar-empty">窗口内没有失败请求 🎉</div>`;
}

function renderStatusBars(stats) {
  const statuses = stats.status_distribution || {};
  const total = Object.values(statuses).reduce((a, b) => a + b, 0);
  el("statusBars").innerHTML = total
    ? Object.entries(statuses).map(([status, count]) => {
        const ratio = count / total;
        return `<div class="bar-row">
          <span class="bar-label" title="${escapeHtml(status)}">${escapeHtml(status)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, Math.round(ratio * 100))}%"></div></div>
          <span class="bar-value">${count} 题</span>
        </div>`;
      }).join("")
    : `<div class="bar-empty">暂无数据</div>`;
}

/* ---------- recent requests ---------- */

function renderRecent(stats) {
  const recent = stats.recent || [];
  el("recentNote").textContent = `最近 ${recent.length} 条`;
  el("recentRows").innerHTML = recent.map((item) => {
    const error = item.ok ? "—" : (item.error_type || "失败");
    return `<tr>
      <td class="lat-cell">${escapeHtml(fmtTime(item.ts))}</td>
      <td><span class="result-dot ${item.ok ? "ok" : ""}"></span>${item.ok ? "通过" : "失败"}</td>
      <td class="q-cell">${escapeHtml(item.question)}</td>
      <td class="${item.ok ? "" : "err-cell"}">${escapeHtml(item.ok ? (item.status || "—") : error)}</td>
      <td class="lat-cell">${escapeHtml(fmtMs(item.latency_ms))}</td>
      <td>${item.model_calls ?? 0}</td>
      <td>${item.retrieval_calls ?? 0}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="7">窗口内暂无请求。</td></tr>`;
}

function renderStats(stats) {
  state.stats = stats;
  el("dashboard").hidden = false;
  el("emptyState").hidden = true;
  renderMetrics(stats);
  renderStageBars(stats);
  renderErrorBars(stats);
  renderStatusBars(stats);
  renderRecent(stats);
  refreshHint();
}

async function loadStats() {
  try {
    const stats = await api(`/api/v1/monitor/stats?minutes=${encodeURIComponent(state.window)}`);
    state.lastStatsFetch = Date.now();
    renderStats(stats);
    setStatus("监控已加载", "ready");
  } catch (error) {
    setStatus("监控数据读取失败", "error");
  }
}

/* ---------- RAGAS calibration ---------- */

const RAGAS_METRICS = [
  ["faithfulness", "忠实度", "回答中的结论是否有检索证据支持"],
  ["answer_relevancy", "答案相关性", "回答是否直接回应了用户问题"],
  ["context_precision", "上下文精确率", "检索内容中有效证据所占比例"],
  ["context_recall", "上下文召回率", "检索证据是否覆盖标准答案要点"],
];

function ragasStatusPill(status) {
  const elStatus = el("ragasStatus");
  if (!status) return;
  if (!status.installed) {
    elStatus.textContent = "RAGAS 未安装";
    elStatus.className = "ragas-status error";
    el("ragasRunBtn").disabled = true;
    el("ragasNote").textContent = "网关环境缺少 ragas，请先安装：pip install \"ragas==0.2.15\" \"langchain-ollama<1\"";
    return;
  }
  if (status.running) {
    elStatus.textContent = "评测进行中";
    elStatus.className = "ragas-status ready";
    el("ragasRunBtn").disabled = true;
    el("ragasCancelBtn").hidden = false;
  } else {
    const backend = status.backend === "api"
      ? `API · ${status.api_model}`
      : `本地 Ollama · ${status.judge_model}`;
    elStatus.textContent = `就绪 · ragas ${status.version || ""} · ${backend}`;
    elStatus.className = "ragas-status ready";
    el("ragasRunBtn").disabled = false;
    el("ragasCancelBtn").hidden = true;
    if (status.backend === "api" && !status.api_key_configured) {
      el("ragasNote").textContent = "已配置 API 模型但缺少 api_key（环境变量 RAGAS_API_KEY），调用会失败。";
    }
  }
  if (status.last_result) {
    renderRagasAggregates(status.last_result.aggregates, status.last_result);
  }
  if (status.last_error) {
    el("ragasNote").textContent = `上次运行出错：${status.last_error}`;
  }
}

function renderRagasAggregates(aggregates, meta) {
  el("ragasMetrics").innerHTML = RAGAS_METRICS.map(([key, label, help]) => {
    const agg = aggregates ? aggregates[key] : null;
    const value = agg && agg.mean != null ? agg.mean.toFixed(3) : "—";
    const hint = agg ? `${agg.count} 条有效` : "未评测";
    const score = agg?.mean == null ? null : Number(agg.mean);
    const tone = score == null ? "na" : score >= .7 ? "good" : score >= .4 ? "warn" : "low";
    const level = score == null ? "无法计算" : score >= .7 ? "表现良好" : score >= .4 ? "需要关注" : "建议排查";
    return `<div class="ragas-metric" data-tone="${tone}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(hint)} · ${level}</small><small class="metric-help">${escapeHtml(help)}</small></div>`;
  }).join("");
  if (meta) {
    const model = meta.model || "";
    const duration = meta.duration_s != null ? ` · 耗时 ${meta.duration_s}s` : "";
    const backend = meta.backend === "api" ? "API" : "本地 Ollama";
    el("ragasResultMeta").textContent = `${backend} · ${model}${duration}`;
  }
}

function ragasScoreCell(value) {
  if (value == null) return `<span class="score-na">—</span>`;
  const num = Number(value);
  const tone = num >= 0.7 ? "score-ok" : num >= 0.4 ? "" : "score-low";
  return `<span class="${tone}">${num.toFixed(3)}</span>`;
}

function setRagasFeedback(message, error = false) {
  el("ragasNote").textContent = message;
  el("ragasNote").className = `ragas-feedback${error ? " error" : ""}`;
}

function hasLowScore(caseItem) {
  return Object.values(caseItem.scores || {}).some((value) => value != null && Number(value) < .4);
}

function renderRagasCases() {
  const query = el("ragasCaseSearch").value.trim().toLowerCase();
  const filter = el("ragasCaseFilter").value;
  const cases = (state.ragas.cases || []).filter((caseItem) => {
    if (filter === "low" && !hasLowScore(caseItem)) return false;
    if (filter === "error" && !caseItem.error) return false;
    return !query || `${caseItem.id} ${caseItem.question}`.toLowerCase().includes(query);
  });
  el("ragasRows").innerHTML = cases.map((caseItem, index) => {
    const scores = caseItem.scores || {};
    const contexts = (caseItem.contexts || []).map((item, i) => `${i + 1}. ${item}`).join("\n\n") || "未返回上下文详情";
    return `<tr data-case-row="${index}"><td>${escapeHtml(caseItem.id)}</td><td class="q-cell">${escapeHtml(caseItem.question)}${caseItem.error ? `<div class="err-cell">${escapeHtml(caseItem.error)}</div>` : ""}</td><td>${ragasScoreCell(scores.faithfulness)}</td><td>${ragasScoreCell(scores.answer_relevancy)}</td><td>${ragasScoreCell(scores.context_precision)}</td><td>${ragasScoreCell(scores.context_recall)}</td><td><button class="case-toggle" type="button" data-case-toggle="${index}" aria-label="展开用例详情">⌄</button></td></tr><tr class="case-detail-row" data-case-detail="${index}" hidden><td colspan="7"><div class="case-detail"><div><strong>系统回答</strong><p>${escapeHtml(caseItem.answer || "未记录回答")}</p></div><div><strong>标准答案</strong><p>${escapeHtml(caseItem.ground_truth || "未提供")}</p></div><div><strong>检索上下文（${caseItem.contexts_count || 0} 条）</strong><p>${escapeHtml(contexts)}</p></div><div><strong>评测状态</strong><p>${escapeHtml(caseItem.error || (hasLowScore(caseItem) ? "存在低于 0.4 的指标，建议检查回答和证据。" : "评测完成，未发现执行错误。"))}</p></div></div></td></tr>`;
  }).join("") || `<tr><td colspan="7">没有符合当前筛选条件的用例。</td></tr>`;
}

function renderRagasTask(task) {
  const progress = task.progress || {};
  el("ragasProgress").hidden = false;
  el("ragasProgressFill").style.width = `${progress.total ? Math.round((progress.current / progress.total) * 100) : 0}%`;
  el("ragasProgressText").textContent = `${progress.current} / ${progress.total}`;
  const percent = progress.total ? progress.current / progress.total : 0;
  el("ragasProgressStage").textContent = task.state === "queued" ? "任务排队中" : percent === 0 ? "生成回答与整理上下文" : percent < 1 ? `正在评测第 ${progress.current + 1} 条用例` : "正在汇总结果";
  [...document.querySelectorAll(".ragas-stage-list span")].forEach((item, index) => item.classList.toggle("active", index <= (task.state === "queued" ? 0 : percent < 1 ? 2 : 3)));
  if (task.state === "running" || task.state === "queued") return;
  if (task.state === "cancelled") {
    setRagasFeedback("评测已取消。取消请求会在当前用例完成后生效。", true);
    el("ragasProgress").hidden = true;
    el("ragasRunBtn").disabled = false;
    el("ragasCancelBtn").hidden = true;
    return;
  }
  if (task.state === "error") {
    setRagasFeedback(`评测失败：${task.error || "未知错误"}`, true);
    el("ragasProgress").hidden = true;
    el("ragasRunBtn").disabled = false;
    el("ragasCancelBtn").hidden = true;
    return;
  }
  // done
  el("ragasProgress").hidden = true;
  el("ragasRunBtn").disabled = false;
  el("ragasCancelBtn").hidden = true;
  const result = task.result || {};
  const aggregates = result.aggregates || {};
  const allBlank = RAGAS_METRICS.every(([key]) => {
    const agg = aggregates[key];
    return !agg || agg.count === 0;
  });
  if (allBlank) {
    setRagasFeedback("所有指标均未产出分数。请检查判题 API、密钥、模型名、额度以及 Embedding 服务。", true);
  } else if (aggregates.context_precision?.count === 0) {
    setRagasFeedback("评测完成。由于没有标准答案，本次只计算忠实度和答案相关性。上下文指标显示为 —。");
  } else {
    setRagasFeedback("评测完成。可通过指标卡和用例明细定位低分项。");
  }
  renderRagasAggregates(result.aggregates, { model: result.model || state.ragas.status?.judge_model, duration_s: result.duration_s });
  state.ragas.cases = result.cases || [];
  el("ragasTableWrap").hidden = state.ragas.cases.length === 0;
  el("ragasCaseTools").hidden = state.ragas.cases.length === 0;
  renderRagasCases();
}

async function pollRagasTask(taskId) {
  try {
    const task = await api(`/api/v1/monitor/ragas/tasks/${encodeURIComponent(taskId)}`);
    renderRagasTask(task);
    if (task.state === "running" || task.state === "queued") {
      if (!state.ragas.pollTimer) {
        state.ragas.pollTimer = setInterval(() => pollRagasTask(taskId), 3000);
      }
    } else {
      if (state.ragas.pollTimer) {
        clearInterval(state.ragas.pollTimer);
        state.ragas.pollTimer = null;
      }
      state.ragas.taskId = null;
      loadRagasStatus();
    }
  } catch (_) {
    if (state.ragas.pollTimer) {
      clearInterval(state.ragas.pollTimer);
      state.ragas.pollTimer = null;
    }
    state.ragas.taskId = null;
    el("ragasNote").textContent = "任务状态读取失败。";
  }
}

async function loadRagasStatus() {
  try {
    state.ragas.status = await api("/api/v1/monitor/ragas/status");
    ragasStatusPill(state.ragas.status);
    if (state.ragas.status.running && !state.ragas.pollTimer && state.ragas.status.running_task_id) {
      state.ragas.taskId = state.ragas.status.running_task_id;
      pollRagasTask(state.ragas.taskId);
    }
  } catch (_) {
    el("ragasStatus").textContent = "RAGAS 状态读取失败";
    el("ragasStatus").className = "ragas-status error";
  }
}

function selectedBenchmarkIds() {
  if (el("ragasBenchmarkAll").checked) return [];
  return [...document.querySelectorAll("[data-benchmark-id]:checked")].map((item) => item.dataset.benchmarkId);
}

function updateRagasRunSummary() {
  const isQuestion = state.ragas.mode === "question";
  const selected = selectedBenchmarkIds();
  el("ragasRunModeSummary").textContent = isQuestion ? "单题评测" : "批量校准";
  el("ragasRunCountSummary").textContent = isQuestion ? "1 条" : `${selected.length || state.ragas.benchmarks.length} 条`;
  const fullMetrics = !isQuestion || Boolean(el("ragasGroundTruth").value.trim());
  el("ragasMetricScope").textContent = fullMetrics ? "4 / 4 项" : "2 / 4 项";
  el("groundTruthHint").textContent = fullMetrics ? "将计算全部四项指标。" : "留空时仅计算忠实度和答案相关性。";
}

function setRagasMode(mode) {
  state.ragas.mode = mode;
  document.querySelectorAll("[data-ragas-mode]").forEach((button) => {
    const active = button.dataset.ragasMode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  el("questionPanel").hidden = mode !== "question";
  el("benchmarkPanel").hidden = mode !== "benchmark";
  el("ragasModeHint").textContent = mode === "question" ? "输入问题，系统将先生成回答与引用证据。" : "选择已有基准用例，对最近一次评测结果执行第三方校准。";
  updateRagasRunSummary();
}

function renderBenchmarkQuestions() {
  const query = el("ragasBenchmarkSearch").value.trim().toLowerCase();
  const filtered = state.ragas.benchmarks.filter((item) => !query || `${item.id} ${item.question} ${item.category || ""}`.toLowerCase().includes(query));
  el("ragasBenchmarkCount").textContent = `${filtered.length} / ${state.ragas.benchmarks.length} 条`;
  el("ragasBenchmarkList").innerHTML = filtered.map((item) => `<label class="benchmark-item"><input type="checkbox" data-benchmark-id="${escapeHtml(item.id)}" ${el("ragasBenchmarkAll").checked ? "disabled" : ""}/><div><strong>${escapeHtml(item.question)}</strong><small>${escapeHtml(item.id)}${item.category ? ` · ${escapeHtml(item.category)}` : ""}</small></div><span>${escapeHtml(item.priority || "")}</span></label>`).join("") || `<div class="benchmark-item"><div><strong>没有匹配的基准题</strong></div></div>`;
  updateRagasRunSummary();
}

async function loadRagasBenchmarks() {
  try {
    const data = await api("/api/v1/monitor/ragas/benchmark-questions");
    state.ragas.benchmarks = data.questions || [];
    renderBenchmarkQuestions();
  } catch (_) {
    el("ragasBenchmarkCount").textContent = "题集读取失败";
    el("ragasBenchmarkList").innerHTML = '<div class="benchmark-item"><div><strong>无法读取基准题集</strong></div></div>';
  }
}

async function runRagas() {
  const mode = state.ragas.mode;
  const question = el("ragasQuestion").value.trim();
  const kbId = el("ragasKbSelect").value;
  if (mode === "question" && !question) {
    setRagasFeedback("请先输入要评测的问题。", true);
    return;
  }
  if (mode === "question" && !kbId) {
    setRagasFeedback("请先选择知识库。", true);
    return;
  }
  const selectedIds = selectedBenchmarkIds();
  if (mode === "benchmark" && !el("ragasBenchmarkAll").checked && !selectedIds.length) {
    setRagasFeedback("请至少选择一条基准题，或启用“使用前 N 条”。", true);
    return;
  }
  const payload = mode === "question" ? {
    mode,
    question,
    ground_truth: el("ragasGroundTruth").value.trim() || null,
    knowledge_base_ids: [kbId],
    answer_model: state.ragas.answerModel || null,
  } : { mode, limit: Math.min(100, Math.max(1, state.ragas.benchmarks.length)), question_ids: selectedIds };
  el("ragasRunBtn").disabled = true;
  el("ragasCancelBtn").hidden = false;
  state.ragas.startedAt = Date.now();
  setRagasFeedback("正在启动评测任务…");
  try {
    const response = await fetch("/api/v1/monitor/ragas/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error?.message || `HTTP ${response.status}`);
    }
    const { task_id: taskId } = await response.json();
    state.ragas.taskId = taskId;
    el("ragasProgress").hidden = false;
    el("ragasProgressFill").style.width = "0%";
    el("ragasProgressText").textContent = "排队中…";
    await pollRagasTask(taskId);
  } catch (error) {
    el("ragasRunBtn").disabled = false;
    el("ragasCancelBtn").hidden = true;
    setRagasFeedback(`启动失败：${error.message}`, true);
  }
}

async function cancelRagas() {
  try {
    await api("/api/v1/monitor/ragas/cancel");
    el("ragasNote").textContent = "已请求取消，将在当前题结束后生效…";
  } catch (_) {
    el("ragasNote").textContent = "取消失败。";
  }
}

async function loadRagasDatasets() {
  try {
    const datasets = await api("/api/v1/admin/datasets");
    const select = el("ragasKbSelect");
    select.innerHTML = "";
    if (!datasets.length) {
      select.innerHTML = '<option value="">（无可用知识库）</option>';
      return;
    }
    datasets.forEach((dataset) => {
      const option = document.createElement("option");
      option.value = dataset.id;
      option.textContent = dataset.name;
      select.appendChild(option);
    });
  } catch (_) {
    el("ragasKbSelect").innerHTML = '<option value="">知识库读取失败</option>';
  }
}

async function loadRagasAnswerModels() {
  try {
    const catalog = await api2("/api/v1/admin/answer-models", "dev");
    state.ragas.answerModel = null;
    const select = el("ragasAnswerModelSelect");
    select.innerHTML = "";
    (catalog.sources || []).forEach((source) => {
      const group = document.createElement("optgroup");
      group.label = `${source.source_type === "local" ? "本地" : "API"} · ${source.name}`;
      (source.models || []).forEach((modelName) => {
        const option = document.createElement("option");
        option.value = `${source.source_id}\u001f${modelName}`;
        option.textContent = modelName;
        option.disabled = source.available === false;
        group.appendChild(option);
      });
      select.appendChild(group);
    });
    const defaultKey = `${catalog.default_source_id || ""}\u001f${catalog.default_model || ""}`;
    if ([...select.options].some((option) => option.value === defaultKey)) {
      select.value = defaultKey;
    }
    if (!select.value && select.options.length) {
      select.value = select.options[0].value;
    }
    updateRagasAnswerModelState();
  } catch (_) {
    el("ragasAnswerModelSelect").innerHTML = '<option value="">回答模型读取失败</option>';
  }
}

function updateRagasAnswerModelState() {
  const value = el("ragasAnswerModelSelect").value;
  if (!value) {
    state.ragas.answerModel = null;
    return;
  }
  const [sourceId, modelName] = value.split("\u001f");
  state.ragas.answerModel = { source_id: sourceId, model_name: modelName };
  el("ragasAnswerModelSummary").textContent = modelName || "使用默认模型";
}

async function api2(path, userId) {
  const response = await fetch(path, { headers: { Accept: "application/json", "X-User-ID": userId } });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

/* ---------- RAGAS model / API switching (mirrors chat model picker) ---------- */

const MODEL_VALUE = (sourceId, model) => `${sourceId}\u001f${model}`;

function parseModelValue(value) {
  const [sourceId, model] = String(value).split("\u001f");
  return { sourceId, model };
}

async function loadRagasProfile(preferredValue = null) {
  let profile;
  try {
    profile = await api("/api/v1/monitor/ragas/profile");
  } catch (_) {
    return;
  }
  state.ragas.profile = profile;
  const select = el("ragasModelSelect");
  const previous = select.value;
  select.innerHTML = "";
  (profile.sources || []).forEach((source) => {
    const group = document.createElement("optgroup");
    group.label = `${source.source_type === "local" ? "本地" : "API"} · ${source.name}`;
    const models = source.models?.length ? source.models : (source.source_type === "local" ? [] : []);
    if (!models.length && source.source_type === "local") {
      const option = document.createElement("option");
      option.value = MODEL_VALUE(source.source_id, profile.active?.model || "");
      option.textContent = profile.active?.model || "（Ollama 不可用）";
      group.appendChild(option);
    }
    models.forEach((model) => {
      const option = document.createElement("option");
      option.value = MODEL_VALUE(source.source_id, model);
      option.textContent = model;
      group.appendChild(option);
    });
    select.appendChild(group);
  });
  const active = profile.active;
  const activeValue = preferredValue || (active ? MODEL_VALUE(active.source_id, active.model) : "");
  if ([...select.options].some((option) => option.value === activeValue)) {
    select.value = activeValue;
  } else if (previous && [...select.options].some((option) => option.value === previous)) {
    select.value = previous;
  }
  const activeSource = (profile.sources || []).find((source) => source.source_id === active?.source_id);
  el("removeRagasApi").hidden = !(activeSource?.removable);
  el("ragasJudgeModelSummary").textContent = active?.model || "未选择";
}

async function applyRagasSelection(payload) {
  const response = await fetch("/api/v1/monitor/ragas/profile", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error?.message || `HTTP ${response.status}`);
  }
  return response.json();
}

function openRagasApiDialog() {
  if (el("ragasSettingsDialog").open) el("ragasSettingsDialog").close();
  el("ragasApiMessage").textContent = "";
  el("ragasApiMessage").className = "dialog-message";
  el("ragasApiDialog").showModal();
}
function closeRagasApiDialog() {
  el("ragasApiDialog").close();
}

async function connectRagasApi(event) {
  event.preventDefault();
  const message = el("ragasApiMessage");
  const baseUrl = el("ragasApiBaseUrl").value.trim();
  if (!baseUrl) {
    message.textContent = "请填写 API Base URL。";
    return;
  }
  const modelsText = el("ragasApiModels").value.trim();
  const payload = {
    name: el("ragasConnName").value.trim(),
    base_url: baseUrl,
    api_key: el("ragasApiKey").value,
    models: modelsText ? modelsText.split(/[,，\s]+/).filter(Boolean) : [],
    embedding_model: el("ragasApiEmbedding").value.trim(),
  };
  try {
    const response = await fetch("/api/v1/monitor/ragas/connections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error?.message || `HTTP ${response.status}`);
    }
    const profile = await response.json();
    await Promise.all([loadRagasProfile(), loadRagasAnswerModels()]);
    if (profile.warning) {
      message.textContent = profile.warning;
      message.className = "dialog-message";
    } else {
      message.textContent = `已连接 ${payload.name || "API"}：模型已加入判题与回答模型下拉。`;
      message.className = "dialog-message ok";
    }
    el("ragasApiKey").value = "";
    el("ragasApiModels").value = "";
    if (!profile.warning) setTimeout(closeRagasApiDialog, 700);
    await loadRagasStatus();
  } catch (error) {
    message.textContent = error.message;
  }
}

async function removeRagasApi() {
  const active = state.ragas.profile?.active;
  if (!active || active.source_id === "local") return;
  try {
    const response = await fetch(
      `/api/v1/monitor/ragas/connections/${encodeURIComponent(active.source_id)}`,
      { method: "DELETE" },
    );
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error?.message || `HTTP ${response.status}`);
    }
    await Promise.all([loadRagasProfile(), loadRagasAnswerModels()]);
    await loadRagasStatus();
    setRagasFeedback("已移除当前 API 连接，判题模型已恢复为本地配置。");
  } catch (error) {
    setRagasFeedback(`移除失败：${error.message}`, true);
  }
}

async function switchRagasModel(event) {
  const { sourceId, model } = parseModelValue(event.target.value);
  if (!model) return;
  try {
    await applyRagasSelection({ source_id: sourceId, model, persist: false });
    await loadRagasStatus();
    el("ragasJudgeModelSummary").textContent = model;
    setRagasFeedback(`已切换判题模型：${model}。该选择仅在当前网关进程内生效。`);
  } catch (error) {
    setRagasFeedback(`切换失败：${error.message}`, true);
    await loadRagasProfile();
  }
}

/* ---------- init ---------- */

el("windowSelect").addEventListener("change", (event) => {
  state.window = Number(event.target.value);
  loadStats();
});

async function initialize() {
  await loadStats();
  if (!state.stats || (state.stats.totals && state.stats.totals.requests === 0)) {
    el("dashboard").hidden = true;
    el("emptyState").hidden = false;
  }
  await Promise.all([
    loadServices(),
    loadRagasStatus(),
    loadRagasProfile(),
    loadRagasDatasets(),
    loadRagasAnswerModels(),
    loadRagasBenchmarks(),
  ]);
  if (!state.ragas.status?.last_result) renderRagasAggregates(null, null);
  updateRagasRunSummary();
  state.statsTimer = setInterval(loadStats, 3000);
  state.servicesTimer = setInterval(() => { loadServices(); loadRagasStatus(); }, 15000);
}

el("ragasRunBtn").addEventListener("click", runRagas);
el("ragasCancelBtn").addEventListener("click", cancelRagas);
el("ragasAnswerModelSelect").addEventListener("change", updateRagasAnswerModelState);
el("ragasModelSelect").addEventListener("change", switchRagasModel);
document.querySelectorAll("[data-ragas-mode]").forEach((button) => button.addEventListener("click", () => setRagasMode(button.dataset.ragasMode)));
el("ragasGroundTruth").addEventListener("input", updateRagasRunSummary);
el("ragasBenchmarkSearch").addEventListener("input", renderBenchmarkQuestions);
el("ragasBenchmarkAll").addEventListener("change", renderBenchmarkQuestions);
el("ragasBenchmarkList").addEventListener("change", updateRagasRunSummary);
el("ragasCaseFilter").addEventListener("change", renderRagasCases);
el("ragasCaseSearch").addEventListener("input", renderRagasCases);
el("ragasRows").addEventListener("click", (event) => { const button = event.target.closest("[data-case-toggle]"); if (!button) return; const detail = document.querySelector(`[data-case-detail="${button.dataset.caseToggle}"]`); detail.hidden = !detail.hidden; button.textContent = detail.hidden ? "⌄" : "⌃"; });
function openRagasSettings(role) { document.querySelectorAll("#ragasSettingsDialog .field").forEach((field) => field.classList.remove("model-focus")); const target = el(role === "judge" ? "ragasModelSelect" : "ragasAnswerModelSelect"); target.closest(".field").classList.add("model-focus"); el("ragasSettingsDialog").showModal(); target.focus(); }
el("openRagasSettings").addEventListener("click", () => openRagasSettings("answer"));
el("openRagasSettingsJudge").addEventListener("click", () => openRagasSettings("judge"));
el("ragasSettingsDialog").addEventListener("click", (event) => { if (event.target === el("ragasSettingsDialog")) el("ragasSettingsDialog").close(); });
el("openRagasApiDialog").addEventListener("click", openRagasApiDialog);
el("removeRagasApi").addEventListener("click", removeRagasApi);
el("closeRagasApiDialog").addEventListener("click", closeRagasApiDialog);
el("cancelRagasApi").addEventListener("click", closeRagasApiDialog);
el("ragasApiForm").addEventListener("submit", connectRagasApi);
el("ragasApiDialog").addEventListener("click", (event) => {
  if (event.target === el("ragasApiDialog")) closeRagasApiDialog();
});

initialize();
