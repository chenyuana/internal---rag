const state = { runs: [], current: null, filter: "all" };

const el = (id) => document.getElementById(id);
const formatNumber = (value, digits = 2) => value == null ? "—" : Number(value).toFixed(digits);
const formatPercent = (value) => value == null ? "—" : `${Math.round(Number(value) * 100)}%`;
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
})[char]);

async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function setStatus(text, kind = "ready") {
  el("storeStatus").textContent = text;
  el("storeStatus").className = `service-pill ${kind}`;
}

function metric(label, value, hint) {
  return `<article class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(hint)}</small></article>`;
}

function renderMetrics(run) {
  const answer = run.summary?.answer_summary || {};
  const judge = run.summary?.judge_summary || {};
  const retrieval = run.summary?.retrieval_summary || {};
  const evaluated = answer.evaluated_count ?? 0;
  const total = answer.case_count ?? run.cases.length;
  el("metricGrid").innerHTML = [
    metric("规则评测", `${evaluated} / ${total}`, "成功得到确定性指标的用例"),
    metric("关键点覆盖", formatPercent(answer.key_points_coverage_avg), "参考答案关键点平均覆盖"),
    metric("检索 MRR", formatNumber(retrieval.mrr), "正确证据首次出现排名"),
    metric("Judge 成功", `${judge.judged_cases ?? 0}`, `失败 ${Object.values(judge.judge_errors || {}).reduce((a, b) => a + b, 0)} 次`),
  ].join("");
}

function renderRegression(regression) {
  const badge = el("regressionBadge");
  const card = el("regressionCard");
  if (!regression) {
    badge.textContent = "未执行回归对比";
    badge.className = "regression-badge neutral";
    card.hidden = true;
    return;
  }
  const failed = Boolean(regression.regression_failed);
  badge.textContent = failed ? "回归门禁未通过" : "回归门禁通过";
  badge.className = `regression-badge ${failed ? "fail" : "pass"}`;
  card.hidden = false;
  el("regressionGrid").innerHTML = [
    ["回归题", regression.regressions?.length ?? 0],
    ["修复题", regression.fixed?.length ?? 0],
    ["新增失败", regression.new_failed?.length ?? 0],
    ["候选版本", regression.candidate_git_commit || "—"],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
}

function caseReasons(item) {
  const reasons = item.answer_rules?.reasons || [];
  if (reasons.length) return reasons.join("；");
  if (item.judge?.judge_error) return `Judge: ${item.judge.judge_error}`;
  return "通过";
}

function judgeAverage(item) {
  const scores = Object.values(item.judge?.scores || {})
    .map((entry) => entry?.score)
    .filter((score) => Number.isInteger(score));
  return scores.length ? formatNumber(scores.reduce((a, b) => a + b, 0) / scores.length, 1) : "—";
}

function renderCases() {
  const cases = state.current?.cases || [];
  const visible = cases.filter((item) => {
    if (state.filter === "failed") return !item.ok;
    if (state.filter === "p0") return item.priority === "P0";
    return true;
  });
  el("caseRows").innerHTML = visible.map((item) => {
    const latency = item.record_summary?.latency_ms;
    return `<tr>
      <td><span class="result-dot ${item.ok ? "ok" : ""}"></span>${item.ok ? "通过" : "失败"}</td>
      <td class="case-name"><strong>${escapeHtml(item.case_id)} · ${escapeHtml(item.priority || "P2")}</strong><span>${escapeHtml(item.question)}</span></td>
      <td>${escapeHtml(item.status || item.record_summary?.error_type || "—")}</td>
      <td class="reason">${escapeHtml(caseReasons(item))}<div class="judge-score">Judge 均分：${escapeHtml(judgeAverage(item))}</div></td>
      <td>${latency == null ? "—" : `${escapeHtml(latency)} ms`}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="5">没有符合筛选条件的用例。</td></tr>`;
}

function renderRun(run) {
  state.current = run;
  el("dashboard").hidden = false;
  el("emptyState").hidden = true;
  el("runTitle").textContent = run.run_id;
  const version = run.version?.answer_model || run.version?.ragflow_version || "版本未记录";
  el("runMeta").textContent = `${run.started_at || "时间未知"} · ${run.knowledge_base || "知识库未知"} · ${version}`;
  renderMetrics(run);
  renderRegression(run.regression);
  renderCases();
}

async function selectRun(runId) {
  setStatus("正在读取运行详情…");
  try {
    renderRun(await api(`/api/v1/evaluation/runs/${encodeURIComponent(runId)}`));
    setStatus("评测数据已加载");
  } catch (error) {
    setStatus("评测详情读取失败", "error");
  }
}

async function initialize() {
  try {
    state.runs = await api("/api/v1/evaluation/runs?limit=50");
    if (!state.runs.length) {
      el("emptyState").hidden = false;
      el("runSelect").innerHTML = "<option>暂无运行</option>";
      setStatus("暂无评测数据", "error");
      return;
    }
    el("runSelect").innerHTML = state.runs.map((run) =>
      `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.started_at || run.run_id)} · ${escapeHtml(run.knowledge_base || "未命名")}</option>`
    ).join("");
    el("runSelect").disabled = false;
    await selectRun(state.runs[0].run_id);
  } catch (error) {
    el("emptyState").hidden = false;
    el("emptyState").querySelector("strong").textContent = "无法读取评测结果";
    el("emptyState").querySelector("span").textContent = "请确认 rag-evaluation 已生成 runs.db。";
    setStatus("评测存储不可用", "error");
  }
}

el("runSelect").addEventListener("change", (event) => selectRun(event.target.value));
document.querySelectorAll(".filter").forEach((button) => button.addEventListener("click", () => {
  state.filter = button.dataset.filter;
  document.querySelectorAll(".filter").forEach((item) => item.classList.toggle("active", item === button));
  renderCases();
}));

initialize();
