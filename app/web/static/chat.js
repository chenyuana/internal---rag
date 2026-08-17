"use strict";

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const STATUS_NAMES = {
  ANSWERABLE: "可回答",
  PARTIALLY_ANSWERABLE: "部分可回答",
  UNANSWERABLE: "无法回答",
  CONFLICTED: "证据冲突",
};

const state = {
  datasets: [],
  selectedKbId: null,
  busy: false,
};

// --- 初始化 ---
async function loadDatasets() {
  const pill = $("servicePill");
  const select = $("kbSelect");
  try {
    const resp = await fetch("/api/v1/admin/datasets", { headers: { "X-User-ID": "dev" } });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    state.datasets = await resp.json();
    if (!Array.isArray(state.datasets) || state.datasets.length === 0) {
      throw new Error("no datasets");
    }
    select.innerHTML = "";
    state.datasets.forEach((ds) => {
      const opt = document.createElement("option");
      opt.value = ds.id;
      opt.textContent = ds.name;
      select.appendChild(opt);
    });
    state.selectedKbId = state.datasets[0].id;
    pill.textContent = `知识库 ${state.datasets.length} 个`;
    pill.className = "service-pill ok";
    enableSend();
  } catch (err) {
    pill.textContent = "服务不可用";
    pill.className = "service-pill bad";
    select.innerHTML = '<option value="">无可用知识库</option>';
    showMessage("无法加载知识库列表，请确认网关与 RAGFlow 服务正常。", false);
  }
}

// --- 交互 ---
function enableSend() {
  const q = $("questionInput").value.trim();
  $("sendButton").disabled = !(q && state.selectedKbId) || state.busy;
}

function showMessage(text, ok) {
  const el = $("formMessage");
  el.textContent = text || "";
  el.className = "form-message" + (ok ? " ok" : "");
}

async function sendQuestion() {
  const question = $("questionInput").value.trim();
  if (!question || !state.selectedKbId || state.busy) return;
  state.busy = true;
  enableSend();
  $("sendButton").classList.add("loading");
  $("sendLabel").textContent = "检索中…";
  showMessage("", false);
  $("emptyState").hidden = true;
  const card = $("answerCard");
  card.hidden = false;
  $("answerBody").textContent = "正在检索并生成回答，复杂问题约需 30~90 秒，请稍候…";
  $("claimsBlock").hidden = true;
  $("coverageBlock").hidden = true;
  $("missingBlock").hidden = true;
  $("citationsBlock").hidden = true;
  $("statusBadge").textContent = "";
  $("statusBadge").className = "status-badge";
  $("answerMeta").textContent = "";

  try {
    const resp = await fetch("/api/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-User-ID": "dev" },
      body: JSON.stringify({
        query: question,
        knowledge_base_ids: [state.selectedKbId],
      }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      const detail = data?.error?.details?.error_type || data?.error?.message || `HTTP ${resp.status}`;
      throw new Error(detail);
    }
    renderAnswer(data);
  } catch (err) {
    $("answerBody").textContent = "请求失败：" + err.message;
    $("statusBadge").textContent = "错误";
    $("statusBadge").className = "status-badge status-UNANSWERABLE";
  } finally {
    state.busy = false;
    enableSend();
    $("sendButton").classList.remove("loading");
    $("sendLabel").textContent = "提问";
  }
}

// --- 渲染 ---
function renderAnswer(data) {
  const status = data.status || "UNANSWERABLE";
  $("statusBadge").textContent = STATUS_NAMES[status] || status;
  $("statusBadge").className = `status-badge status-${status}`;
  $("answerMeta").textContent = data.prompt_version
    ? `v${data.prompt_version.replace("answer-generation-", "")}`
    : "";
  $("answerBody").innerHTML = highlightCitations(escapeHtml(data.answer || ""));

  // claims
  const claims = data.claims || [];
  if (claims.length) {
    $("claimsBlock").hidden = false;
    const list = $("claimsList");
    list.innerHTML = "";
    claims.forEach((cl) => {
      const li = document.createElement("li");
      const cite = (cl.citation_ids || []).map((c) => `[${escapeHtml(c)}]`).join(" ");
      li.innerHTML = `${escapeHtml(cl.claim)}${cite ? ` <span class="marker">${cite}</span>` : ""}`;
      list.appendChild(li);
    });
  }

  // coverage matrix
  const assessment = data.evidence_assessment;
  const covered = (assessment?.covered_requirements || []).map((r) => ({ r, s: "covered" }));
  const missing = (assessment?.missing_requirements || []).map((r) => ({ r, s: "missing" }));
  const cells = [...covered, ...missing];
  if (cells.length) {
    $("coverageBlock").hidden = false;
    const grid = $("coverageGrid");
    grid.innerHTML = "";
    cells.forEach(({ r, s }) => {
      const div = document.createElement("div");
      div.className = "coverage-cell";
      const [subject, ...rest] = r.split("：");
      const aspect = rest.join("：");
      div.innerHTML =
        `<span class="cell-status cell-${s}">${s === "covered" ? "已覆盖" : "缺失"}</span>` +
        `<span class="cell-subject">${escapeHtml(subject)}</span>` +
        (aspect ? `<span class="cell-aspect">${escapeHtml(aspect)}</span>` : "");
      grid.appendChild(div);
    });
  }

  // missing information
  const missingInfo = data.missing_information || [];
  if (missingInfo.length) {
    $("missingBlock").hidden = false;
    const list = $("missingList");
    list.innerHTML = "";
    missingInfo.forEach((m) => {
      const li = document.createElement("li");
      li.textContent = m;
      list.appendChild(li);
    });
  }

  // citations
  const citations = data.citations || [];
  if (citations.length) {
    $("citationsBlock").hidden = false;
    const list = $("citationsList");
    list.innerHTML = "";
    citations.forEach((c) => {
      const li = document.createElement("li");
      const doc = c.document_name || c.document_id || "未知文档";
      const quote = (c.quote || "").trim().slice(0, 220);
      li.innerHTML =
        `<span class="cite-doc">[${escapeHtml(c.citation_id)}] ${escapeHtml(doc)}</span>` +
        (quote ? `<span class="cite-quote">${escapeHtml(quote)}${quote.length >= 220 ? "…" : ""}</span>` : "");
      list.appendChild(li);
    });
  }

  $("requestId").textContent = data.request_id ? `request ${data.request_id.slice(0, 8)}` : "";
}

function highlightCitations(text) {
  return text.replace(/(\[C\d+\])/g, '<span class="marker">$1</span>');
}

// --- 事件绑定 ---
$("kbSelect").addEventListener("change", (e) => {
  state.selectedKbId = e.target.value;
  enableSend();
});
$("questionInput").addEventListener("input", enableSend);
$("questionInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});
$("sendButton").addEventListener("click", sendQuestion);
document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => {
    $("questionInput").value = btn.dataset.q;
    enableSend();
    sendQuestion();
  });
});

loadDatasets();
