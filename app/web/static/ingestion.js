const API = "/api/v1/ingestion";
const state = {
  files: [],
  jobs: [],
  selectedJobId: null,
  busy: false,
  refreshing: false,
  publishingJobId: null,
  selectedPublishJobIds: new Set(),
  batchPublishing: false,
  batchExporting: false,
  batchProgress: { current: 0, total: 0, message: "" },
  jobView: "unreviewed",
  deletingJobId: null,
  previewPages: [],
  firstIssuePage: null,
  jobQuery: "",
  jobSearchTimer: null,
  currentPageNumber: null,
  currentPageDetail: null,
  editingChunkId: null,
  savingChunk: false,
  exportingJobId: null,
  savingLlmStructureSettings: false,
};

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const statusNames = {
  queued: "等待处理",
  running: "正在解析",
  completed: "质量通过",
  warning: "需要抽检",
  needs_review: "需要复核",
  failed: "处理失败",
};

const progressStageNames = {
  queued: "等待处理",
  preparing: "准备文件",
  parsing: "自动解析",
  writing: "写入结果",
  quality_check: "质量检查",
  completed: "处理完成",
  warning: "处理完成",
  needs_review: "等待复核",
  failed: "处理失败",
};

const publicationStageNames = {
  not_requested: "尚未发布",
  planned: "计划已生成",
  uploading_document: "上传原文件",
  publishing_chunks: "写入 Chunk",
  published: "发布完成",
  failed: "发布失败",
};

const reviewStatusNames = {
  unreviewed: "待审核",
  in_review: "审核中",
  approved: "已通过",
  rework: "需返工",
};

function jobReviewStatus(job) {
  return job.review?.status || "unreviewed";
}

function compareReviewOrder(left, right) {
  const createdDifference =
    Date.parse(left.created_at || 0) - Date.parse(right.created_at || 0);
  if (createdDifference) return createdDifference;
  const leftSequence = left.sequence_in_batch ?? Number.MAX_SAFE_INTEGER;
  const rightSequence = right.sequence_in_batch ?? Number.MAX_SAFE_INTEGER;
  if (leftSequence !== rightSequence) return leftSequence - rightSequence;
  const leftName = left.source_relative_path || left.source_name || "";
  const rightName = right.source_relative_path || right.source_name || "";
  return (
    leftName.localeCompare(rightName, "zh-CN", { numeric: true }) ||
    String(left.job_id).localeCompare(String(right.job_id))
  );
}

function reviewOrderedJobs() {
  return [...state.jobs].sort(compareReviewOrder);
}

function matchesJobQuery(job) {
  const query = state.jobQuery.trim().toLocaleLowerCase("zh-CN");
  if (!query) return true;
  const values = [
    job.source_name,
    job.source_relative_path,
    job.job_id,
    job.knowledge_base_id,
    job.status,
    statusNames[job.status],
    job.publication?.status,
    jobReviewStatus(job),
    reviewStatusNames[jobReviewStatus(job)],
    job.review?.reviewer,
    job.review?.note,
  ];
  return values.some((value) =>
    String(value || "").toLocaleLowerCase("zh-CN").includes(query),
  );
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = payload?.error;
    throw new Error(error?.message || `请求失败（HTTP ${response.status}）`);
  }
  return payload;
}

function setMessage(target, message, isError = false) {
  target.textContent = message;
  target.classList.toggle("error", isError);
}

function renderLlmStructureSettings(settings) {
  $("llmStructureBaseUrl").value = settings.base_url || "https://api.deepseek.com/v1";
  $("llmStructureModel").value = settings.model || "deepseek-v4-flash";
  $("llmStructurePageBatchSize").value = settings.page_batch_size || 3;
  $("llmStructureMaxTokens").value = settings.max_tokens || 2048;
  $("llmStructureTimeout").value = settings.timeout_seconds || 180;
  $("llmStructureApiKey").value = "";
  $("llmStructureKeyStatus").textContent = settings.api_key_configured
    ? `API Key 已配置（尾号 ${settings.api_key_last4 || "****"}）`
    : "尚未配置 API Key";
}

async function loadLlmStructureSettings() {
  try {
    const settings = await requestJson(`${API}/settings/llm-structure`, {
      headers: { "X-User-ID": "local-operator" },
    });
    renderLlmStructureSettings(settings);
  } catch (error) {
    setMessage($("llmStructureMessage"), `模型配置加载失败：${error.message}`, true);
  }
}

async function saveLlmStructureSettings() {
  if (state.savingLlmStructureSettings) return;
  const payload = {
    base_url: $("llmStructureBaseUrl").value.trim(),
    model: $("llmStructureModel").value.trim(),
    page_batch_size: Number($("llmStructurePageBatchSize").value),
    max_tokens: Number($("llmStructureMaxTokens").value),
    timeout_seconds: Number($("llmStructureTimeout").value),
  };
  const apiKey = $("llmStructureApiKey").value.trim();
  if (apiKey) payload.api_key = apiKey;
  if (!payload.base_url || !payload.model) {
    setMessage($("llmStructureMessage"), "请填写接口地址和模型名称。", true);
    return;
  }
  state.savingLlmStructureSettings = true;
  $("saveLlmStructureSettings").disabled = true;
  setMessage($("llmStructureMessage"), "正在保存模型配置…");
  try {
    const settings = await requestJson(`${API}/settings/llm-structure`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        "X-User-ID": "local-operator",
      },
      body: JSON.stringify(payload),
    });
    renderLlmStructureSettings(settings);
    setMessage($("llmStructureMessage"), "模型配置已保存；服务重启后需要重新配置 API Key。");
  } catch (error) {
    setMessage($("llmStructureMessage"), error.message, true);
  } finally {
    state.savingLlmStructureSettings = false;
    $("saveLlmStructureSettings").disabled = false;
  }
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function fileKey(file) {
  return `${fileRelativePath(file) || file.name}:${file.size}:${file.lastModified}`;
}

function fileRelativePath(file) {
  return String(file.webkitRelativePath || file.ingestionRelativePath || "")
    .replaceAll("\\", "/")
    .replace(/^\/+/, "");
}

function fileDisplayName(file) {
  return fileRelativePath(file) || file.name;
}

function fileDirectory(file) {
  const relativePath = fileRelativePath(file);
  if (!relativePath.includes("/")) return "";
  return relativePath.slice(0, relativePath.lastIndexOf("/"));
}

function renderSelectedFiles() {
  const container = $("selectedFiles");
  const count = state.files.length;
  container.hidden = count === 0;
  container.innerHTML = state.files
    .map(
      (file, index) => `
        <div class="selected-file">
          <span class="selected-file-index">${String(index + 1).padStart(2, "0")}</span>
          <span class="selected-file-copy">
            <strong>${escapeHtml(file.name)}</strong>
            ${
              fileDirectory(file)
                ? `<small class="selected-file-path">${escapeHtml(fileDirectory(file))}</small>`
                : ""
            }
            <small>${formatBytes(file.size)}</small>
          </span>
          <button type="button" data-remove-file="${index}" aria-label="移除 ${escapeHtml(fileDisplayName(file))}">×</button>
        </div>`,
    )
    .join("");

  container.querySelectorAll("[data-remove-file]").forEach((button) => {
    button.disabled = state.busy;
    button.addEventListener("click", () => {
      state.files.splice(Number(button.dataset.removeFile), 1);
      updateFileSelection();
    });
  });
}

function updateFileSelection() {
  const count = state.files.length;
  const totalBytes = state.files.reduce((total, file) => total + file.size, 0);
  $("dropTitle").textContent = count
    ? `已选择 ${count} 个文档`
    : "拖入一个或多个 PDF/Word 文档，或从下方选择";
  $("dropMeta").textContent = count
    ? `合计 ${formatBytes(totalBytes)} · 将按顺序进入处理队列`
    : "支持批量文件和整个文件夹；接收 PDF 与 DOCX";
  $("uploadButton").textContent = count ? `提交 ${count} 个文件自动处理` : "提交自动处理";
  $("uploadButton").disabled = state.busy || count === 0;
  $("selectFilesButton").disabled = state.busy;
  $("selectFolderButton").disabled = state.busy;
  renderSelectedFiles();
}

function setSelectedFiles(fileList, source = "files") {
  const incoming = Array.from(fileList || []);
  if (!incoming.length) return;
  const accepted = incoming.filter((file) =>
    [".pdf", ".docx"].some((extension) => file.name.toLowerCase().endsWith(extension)),
  );
  if (source === "folder") {
    accepted.sort((left, right) =>
      fileDisplayName(left).localeCompare(fileDisplayName(right), "zh-CN", {
        numeric: true,
      }),
    );
  }
  const rejectedCount = incoming.length - accepted.length;
  const known = new Set(state.files.map(fileKey));
  accepted.forEach((file) => {
    if (!known.has(fileKey(file))) {
      state.files.push(file);
      known.add(fileKey(file));
    }
  });
  updateFileSelection();
  setMessage(
    $("formMessage"),
    rejectedCount
      ? `已忽略 ${rejectedCount} 个非 PDF/DOCX 文件，其余文件可以提交。`
      : source === "folder"
        ? `已从文件夹读取 ${accepted.length} 个文档，共 ${state.files.length} 个文件待提交。`
        : `${state.files.length} 个文件已加入待提交列表。`,
    rejectedCount > 0 && accepted.length === 0,
  );
}

async function uploadFiles() {
  if (!state.files.length || state.busy) return;
  state.busy = true;
  updateFileSelection();
  const pending = [...state.files];
  const failedFiles = [];
  const createdJobs = [];
  const datasetId = $("datasetInput").value.trim();
  const batchId = globalThis.crypto?.randomUUID
    ? globalThis.crypto.randomUUID()
    : `batch-${Date.now()}-${Math.random().toString(16).slice(2)}`;

  for (let index = 0; index < pending.length; index += 1) {
    const file = pending[index];
    setMessage(
      $("formMessage"),
      `正在提交 ${index + 1}/${pending.length}：${fileDisplayName(file)}`,
    );
    const form = new FormData();
    form.append("file", file);
    const relativePath = fileRelativePath(file);
    if (relativePath) form.append("source_relative_path", relativePath);
    if (datasetId) form.append("knowledge_base_id", datasetId);
    form.append("llm_structure_enabled", String($("llmStructureEnabled").checked));
    form.append("batch_id", batchId);
    form.append("sequence_in_batch", String(index));
    try {
      const job = await requestJson(`${API}/jobs`, {
        method: "POST",
        body: form,
        headers: { "X-User-ID": "local-operator" },
      });
      createdJobs.push(job);
      state.selectedJobId = job.job_id;
    } catch (error) {
      failedFiles.push(file);
      file.uploadError = error.message;
    }
  }

  state.files = failedFiles;
  state.busy = false;
  $("fileInput").value = "";
  $("folderInput").value = "";
  updateFileSelection();
  if (failedFiles.length) {
    setMessage(
      $("formMessage"),
      `已创建 ${createdJobs.length} 个任务，${failedFiles.length} 个文件提交失败，可保留后重试。`,
      true,
    );
  } else {
    setMessage(
      $("formMessage"),
      `${createdJobs.length} 个任务已按顺序进入串行处理队列。`,
    );
  }
  await refreshAll();
}

function normalizedProcessingProgress(job) {
  const terminal = ["completed", "warning", "needs_review"].includes(job.status);
  if (terminal && Number(job.progress?.percent || 0) < 100) {
    return {
      stage: job.status,
      percent: 100,
      message: statusNames[job.status],
      indeterminate: false,
    };
  }
  if (job.status === "failed") {
    return {
      stage: "failed",
      percent: Number(job.progress?.percent || 0),
      message: job.error_message || job.progress?.message || "处理失败",
      indeterminate: false,
    };
  }
  return {
    stage: job.progress?.stage || job.status,
    percent: Number(job.progress?.percent || (job.status === "running" ? 15 : 0)),
    message: job.progress?.message || statusNames[job.status] || "等待处理",
    indeterminate: Boolean(job.progress?.indeterminate),
  };
}

function normalizedPublicationProgress(job) {
  const publication = job.publication || {};
  const status = publication.status || "not_requested";
  const total = Number(publication.planned_chunk_count || 0);
  const published = Number(publication.published_chunk_count || 0);
  const skipped = Number(publication.skipped_chunk_count || 0);
  let percent = Number(publication.percent || 0);
  if (status === "published") percent = 100;
  if (status === "publishing" && !percent) {
    const complete = published + skipped;
    percent = total ? Math.min(99, 5 + Math.floor((complete / total) * 94)) : 2;
  }
  let message = publication.message;
  if (!message || (message === "尚未生成发布计划" && status !== "not_requested")) {
    if (status === "published") {
      message = `发布完成：新增 ${published}，跳过 ${skipped}`;
    } else if (status === "publishing") {
      message = total
        ? `正在写入 Chunk：${published + skipped}/${total}`
        : "正在向 RAGFlow 上传原始文件";
    } else if (status === "planned") {
      message = `发布计划已生成，共 ${total} 个 Chunk`;
    } else if (status === "failed") {
      message = publication.error_message || "发布失败";
    }
  }
  return {
    status,
    stage: publication.stage === "not_requested" && status !== "not_requested"
      ? status
      : publication.stage || status,
    percent,
    message: message || publication.error_message || publicationStageNames[status] || "尚未发布",
  };
}

function activeJobProgress(job) {
  const publication = normalizedPublicationProgress(job);
  if (["publishing", "published", "failed"].includes(publication.status)) {
    return {
      ...publication,
      label: publicationStageNames[publication.stage] || publicationStageNames[publication.status],
      publication: true,
      indeterminate: publication.status === "publishing" && publication.percent <= 2,
    };
  }
  const processing = normalizedProcessingProgress(job);
  return {
    ...processing,
    label: progressStageNames[processing.stage] || statusNames[job.status],
    publication: false,
  };
}

function isPublished(job) {
  return job.publication?.status === "published";
}

function hasManualQualityGateOverride(job) {
  return Boolean(
    job.status === "needs_review" &&
      job.review?.status === "approved" &&
      job.review?.quality_gate_override &&
      job.review?.overridden_quality_gates?.length,
  );
}

function isPublishable(job) {
  return (
    (["completed", "warning"].includes(job.status) || hasManualQualityGateOverride(job)) &&
    jobReviewStatus(job) === "approved" &&
    !["publishing", "published"].includes(job.publication?.status)
  );
}

function isDeletable(job) {
  return (
    !["queued", "running"].includes(job.status) &&
    job.publication?.status !== "publishing" &&
    state.deletingJobId !== job.job_id
  );
}

function renderJobItem(job, selectable) {
  const progress = activeJobProgress(job);
  const reviewStatus = jobReviewStatus(job);
  const showProgress =
    ["queued", "running"].includes(job.status) ||
    progress.publication ||
    progress.percent > 0;
  const checked = state.selectedPublishJobIds.has(job.job_id);
  return `
    <div class="job-row ${selectable ? "selectable" : ""}">
      <label class="job-select ${selectable ? "" : "disabled"}"
        title="${selectable ? "加入批量发送" : isPublished(job) ? "已发送" : "解析或质量检查尚未完成"}">
        ${
          selectable
            ? `<input type="checkbox" data-publish-select="${escapeHtml(job.job_id)}" ${checked ? "checked" : ""} />`
            : `<span aria-hidden="true">${isPublished(job) ? "✓" : "·"}</span>`
        }
      </label>
      <button class="job-item ${job.job_id === state.selectedJobId ? "active" : ""}"
        type="button" data-job-id="${escapeHtml(job.job_id)}">
        <span class="job-icon">${job.source_name?.toLowerCase().endsWith(".docx") ? "DOCX" : "PDF"}</span>
        <span class="job-copy">
          <strong>${escapeHtml(job.source_name)}</strong>
          ${
            job.source_relative_path
              ? `<small class="job-source-path">${escapeHtml(job.source_relative_path)}</small>`
              : ""
          }
          <small>${escapeHtml(job.quality?.route || progress.label || "等待自动判断")} ·
            ${formatBytes(job.size_bytes)}</small>
          ${
            isPublished(job)
              ? ""
              : `<span class="review-chip ${escapeHtml(reviewStatus)}">${escapeHtml(
                  reviewStatusNames[reviewStatus] || reviewStatus,
                )}</span>`
          }
          ${
            showProgress
              ? `<span class="mini-progress">
                  <span style="width:${Math.max(2, progress.percent)}%"></span>
                </span>`
              : ""
          }
        </span>
        <span class="job-state">
          <strong>${escapeHtml(progress.label || statusNames[job.status] || job.status)}</strong>
          <small>${progress.percent}%</small>
        </span>
      </button>
      <button class="job-delete" type="button"
        data-delete-job="${escapeHtml(job.job_id)}"
        ${isDeletable(job) ? "" : "disabled"}
        title="${
          isDeletable(job)
            ? isPublished(job)
              ? "仅从资料接入台移除记录，不影响 RAGFlow"
              : "删除该任务及本地解析结果"
            : "处理中或发送中的资料暂不能删除"
        }">
        ${isPublished(job) ? "移除记录" : "删除"}
      </button>
    </div>`;
}

function renderJobGroup(title, jobs, groupName) {
  if (!jobs.length) return "";
  return `
    <section class="job-group ${groupName}">
      <header class="job-group-heading">
        <span>${escapeHtml(title)}</span>
        <strong>${jobs.length}</strong>
      </header>
      <div class="job-group-list">
        ${jobs.map((job) => renderJobItem(job, isPublishable(job))).join("")}
      </div>
    </section>`;
}

function updateBatchControls() {
  const eligible = state.jobs.filter(isPublishable);
  const eligibleIds = new Set(eligible.map((job) => job.job_id));
  state.selectedPublishJobIds.forEach((jobId) => {
    if (!eligibleIds.has(jobId)) state.selectedPublishJobIds.delete(jobId);
  });
  const selectedCount = state.selectedPublishJobIds.size;
  $("batchToolbar").hidden = state.jobView !== "approved";
  const selectAll = $("selectAllApproved");
  const batchBusy = state.batchPublishing || state.batchExporting;
  selectAll.disabled = batchBusy || eligible.length === 0;
  selectAll.checked = eligible.length > 0 && selectedCount === eligible.length;
  selectAll.indeterminate = selectedCount > 0 && selectedCount < eligible.length;
  $("batchPublishButton").disabled = batchBusy || selectedCount === 0;
  $("batchPublishButton").textContent = state.batchPublishing
    ? "正在批量发送"
    : `批量发送${selectedCount ? `（${selectedCount}）` : ""}`;
  $("batchExportButton").disabled = batchBusy || selectedCount === 0;
  $("batchExportButton").textContent = state.batchExporting
    ? "正在批量导出"
    : `批量导出${selectedCount ? `（${selectedCount}）` : ""}`;

  const panel = $("batchPublishProgress");
  panel.hidden = !state.batchPublishing && state.batchProgress.total === 0;
  if (!panel.hidden) {
    const { current, total, message } = state.batchProgress;
    const percent = total ? Math.round((current / total) * 100) : 0;
    $("batchPublishStatus").textContent = message || "准备批量发送";
    $("batchPublishValue").textContent = `${current}/${total}`;
    $("batchPublishFill").style.width = `${percent}%`;
  }
}

function renderJobs() {
  const list = $("jobList");
  const hasQuery = Boolean(state.jobQuery.trim());
  const matchingJobs = reviewOrderedJobs().filter(matchesJobQuery);
  $("clearJobSearch").hidden = !hasQuery;
  $("jobSearchMeta").textContent = hasQuery
    ? `找到 ${matchingJobs.length} 条与“${state.jobQuery.trim()}”匹配的资料`
    : "";
  if (!matchingJobs.length) {
    $("unreviewedJobCount").textContent = "0";
    $("inReviewJobCount").textContent = "0";
    $("approvedJobCount").textContent = "0";
    $("reworkJobCount").textContent = "0";
    $("publishedJobCount").textContent = "0";
    list.innerHTML = hasQuery
      ? '<div class="empty-state">没有找到匹配资料，请尝试文件名、文件夹、状态或任务 ID。</div>'
      : '<div class="empty-state">尚无任务，上传第一份资料开始测试。</div>';
    updateBatchControls();
    return;
  }
  const published = matchingJobs.filter(isPublished);
  const reviewGroups = {
    unreviewed: matchingJobs.filter(
      (job) => !isPublished(job) && jobReviewStatus(job) === "unreviewed",
    ),
    in_review: matchingJobs.filter(
      (job) => !isPublished(job) && jobReviewStatus(job) === "in_review",
    ),
    approved: matchingJobs.filter(
      (job) => !isPublished(job) && jobReviewStatus(job) === "approved",
    ),
    rework: matchingJobs.filter(
      (job) => !isPublished(job) && jobReviewStatus(job) === "rework",
    ),
  };
  $("unreviewedJobCount").textContent = String(reviewGroups.unreviewed.length);
  $("inReviewJobCount").textContent = String(reviewGroups.in_review.length);
  $("approvedJobCount").textContent = String(reviewGroups.approved.length);
  $("reworkJobCount").textContent = String(reviewGroups.rework.length);
  $("publishedJobCount").textContent = String(published.length);
  document.querySelectorAll("[data-job-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.jobView === state.jobView);
  });
  const visibleJobs =
    state.jobView === "published" ? published : reviewGroups[state.jobView] || [];
  const viewTitles = {
    unreviewed: "待审核资料",
    in_review: "审核中资料",
    approved: "已通过资料",
    rework: "需返工资料",
    published: "已发送资料",
  };
  list.innerHTML = visibleJobs.length
    ? renderJobGroup(viewTitles[state.jobView], visibleJobs, state.jobView)
    : `<div class="empty-state">暂无${viewTitles[state.jobView]}。</div>`;
  list.querySelectorAll("[data-job-id]").forEach((button) => {
    button.addEventListener("click", () => selectJob(button.dataset.jobId));
  });
  list.querySelectorAll("[data-publish-select]").forEach((checkbox) => {
    checkbox.disabled = state.batchPublishing;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        state.selectedPublishJobIds.add(checkbox.dataset.publishSelect);
      } else {
        state.selectedPublishJobIds.delete(checkbox.dataset.publishSelect);
      }
      updateBatchControls();
    });
  });
  list.querySelectorAll("[data-delete-job]").forEach((button) => {
    button.addEventListener("click", () => deleteJob(button.dataset.deleteJob));
  });
  updateBatchControls();
}

function renderQuality(job, preview) {
  const quality = job.quality || {};
  const metrics = [
    ["页面", quality.page_count ?? 0],
    ["Chunk", quality.chunk_count ?? 0],
    ["文本覆盖率", `${Math.round((quality.text_coverage || 0) * 100)}%`],
    ["警告", quality.warning_count ?? 0],
    ["失败", quality.failure_count ?? 0],
  ];
  $("qualityGrid").innerHTML = metrics
    .map(
      ([label, value]) =>
        `<div class="quality-metric"><span>${label}</span><strong>${value}</strong></div>`,
    )
    .join("");

  const routes = preview?.page_routes || {};
  $("routeSummary").innerHTML =
    Object.entries(routes)
      .map(
        ([route, count]) =>
          `<div class="route-row"><span>${escapeHtml(route)}</span><strong>${count} 页</strong></div>`,
      )
      .join("") || '<div class="empty-state">处理完成后显示页面路由。</div>';

  const gates = preview?.quality?.quality_gates || [];
  $("qualityGates").innerHTML =
    gates
      .map(
        (gate) => `
          <div class="gate-item">
            <span class="gate-dot ${escapeHtml(gate.status)}"></span>
            <span><strong>${escapeHtml(gate.gate)}</strong><br />${escapeHtml(gate.message)}</span>
          </div>`,
      )
      .join("") || '<div class="empty-state">处理完成后显示质量门禁。</div>';

  renderPageDiagnostics(preview?.pages || [], preview?.quality || {});

  const chunks = preview?.chunks || [];
  $("chunkPreview").innerHTML =
    chunks
      .map(
        (chunk) => `
          <article class="chunk-item">
            <header>
              <strong>${escapeHtml(chunk.title || "未命名 Chunk")} · ${escapeHtml(chunk.content_type || "text")}</strong>
              <span>第 ${escapeHtml(chunk.page_start)}–${escapeHtml(chunk.page_end)} 页</span>
            </header>
            <p>${escapeHtml(chunk.text)}</p>
            ${structuredTableMarkup(chunk.table_html)}
            <details>
              <summary>查看发送到 RAGFlow 的实际内容</summary>
              <pre>${escapeHtml(ragflowContentPreview(preview?.source_name, chunk))}</pre>
            </details>
          </article>`,
      )
      .join("") || '<div class="empty-state">暂无可预览 Chunk。</div>';
}

function pageQualityTones(quality) {
  const tones = new Map();
  const rank = { warning: 1, failed: 2 };
  (quality?.quality_gates || []).forEach((gate) => {
    if (!['warn', 'fail'].includes(gate.status)) return;
    const tone = gate.status === 'fail' ? 'failed' : 'warning';
    const pageNumbers = new Set(
      (Array.isArray(gate.pages) ? gate.pages : []).map(Number).filter(Number.isInteger),
    );
    for (const match of String(gate.message || '').matchAll(/\[([\d,\s]+)\]/g)) {
      match[1]
        .split(',')
        .map((value) => Number(value.trim()))
        .filter(Number.isInteger)
        .forEach((pageNumber) => pageNumbers.add(pageNumber));
    }
    pageNumbers.forEach((pageNumber) => {
      const current = tones.get(pageNumber);
      if (!current || rank[tone] > rank[current]) tones.set(pageNumber, tone);
    });
  });
  return tones;
}

function pageRouteTone(page, qualityTones = new Map()) {
  const route = String(page.route || "");
  if (route === "toc_excluded" || page.indexable === false) return "excluded";
  if (qualityTones.has(Number(page.page_number))) {
    return qualityTones.get(Number(page.page_number));
  }
  if (route === "vector_layout" || route === "remote_layout") return "ready";
  if (route.includes("ocr_required") || route.includes("low_text")) return "failed";
  if (route.includes("review") || route.includes("layout")) return "warning";
  return "ready";
}

function renderPageDiagnostics(pages, quality) {
  const container = $("pageDiagnostics");
  const qualityTones = pageQualityTones(quality);
  state.previewPages = pages;
  state.firstIssuePage =
    pages.find((page) => pageRouteTone(page, qualityTones) === "failed") ||
    pages.find((page) => pageRouteTone(page, qualityTones) === "warning") ||
    null;
  const issueButton = $("firstIssuePageButton");
  issueButton.hidden = !state.firstIssuePage;
  if (state.firstIssuePage) {
    issueButton.textContent = `查看首个问题页：第 ${state.firstIssuePage.page_number} 页`;
  }
  if (!pages.length) {
    container.innerHTML = '<div class="empty-state">处理完成后可逐页检查解析情况。</div>';
    return;
  }
  container.innerHTML = pages
    .map((page) => {
      const articles = (page.article_ids || []).join("、");
      const details = [
        `${Number(page.cleaned_char_count || 0)} 字`,
        articles ? `条号 ${articles}` : "",
        page.table_hints?.length ? `表格 ${page.table_hints.length}` : "",
        page.asset_ids?.length ? `资产 ${page.asset_ids.length}` : "",
        page.page_type ? `类型 ${page.page_type}` : "",
        page.layout_features?.toc_score != null
          ? `目录分 ${Number(page.layout_features.toc_score).toFixed(2)}`
          : "",
        page.manual_edit_count ? `人工修订 ${page.manual_edit_count}` : "",
      ].filter(Boolean);
      return `
        <button
          class="page-diagnostic ${pageRouteTone(page, qualityTones)}"
          type="button"
          data-page-number="${escapeHtml(page.page_number)}"
          title="查看第 ${escapeHtml(page.page_number)} 页解析详情"
        >
          <strong>${escapeHtml(page.page_number)}</strong>
          <span>${escapeHtml(page.route || "unknown")}</span>
          <small>${escapeHtml(details.join(" · ") || "暂无文本")}</small>
        </button>`;
    })
    .join("");
  container.querySelectorAll("[data-page-number]").forEach((button) => {
    button.addEventListener("click", () => {
      openPageDetail(Number(button.dataset.pageNumber));
    });
  });
}

function diagnosticItems(items, renderer, emptyMessage) {
  if (!items.length) return `<div class="empty-state compact">${escapeHtml(emptyMessage)}</div>`;
  return items.map(renderer).join("");
}

function sanitizeStructuredTable(markup) {
  if (!markup) return "";
  const documentFragment = new DOMParser().parseFromString(
    `<div id="table-root">${markup}</div>`,
    "text/html",
  );
  const root = documentFragment.querySelector("#table-root");
  if (!root) return "";
  const allowed = new Set(["TABLE", "THEAD", "TBODY", "TFOOT", "TR", "TH", "TD"]);
  [...root.querySelectorAll("*")].forEach((element) => {
    if (!allowed.has(element.tagName)) {
      element.replaceWith(documentFragment.createTextNode(element.textContent || ""));
      return;
    }
    [...element.attributes].forEach((attribute) => {
      if (!["rowspan", "colspan"].includes(attribute.name.toLowerCase())) {
        element.removeAttribute(attribute.name);
      }
    });
  });
  return root.innerHTML;
}

function structuredTableMarkup(value) {
  const tables = (Array.isArray(value) ? value : [value])
    .map((markup) => sanitizeStructuredTable(markup))
    .filter(Boolean);
  return tables.length
    ? `<div class="structured-table">${tables.join("")}</div>`
    : "";
}

function ragflowContentPreview(sourceName, chunk) {
  const prefix = [`文档：${sourceName || ""}`];
  if (chunk.section_path?.length) {
    prefix.push(`章节：${chunk.section_path.join(" / ")}`);
  }
  if (chunk.article_id_normalized) {
    prefix.push(`条号：${chunk.article_id_normalized}`);
  }
  if (Number(chunk.page_start || 0) > 0) {
    const pageLabel =
      Number(chunk.page_end || 0) === Number(chunk.page_start)
        ? chunk.page_start
        : `${chunk.page_start}-${chunk.page_end}`;
    prefix.push(`页码：${pageLabel}`);
  }
  const content = `${prefix.join("\n")}\n\n${chunk.text || ""}`;
  return content;
}

function selectedJob() {
  return state.jobs.find((item) => item.job_id === state.selectedJobId) || null;
}

function canEditParsedChunk() {
  const job = selectedJob();
  return Boolean(
    job &&
      ["completed", "warning", "needs_review"].includes(job.status) &&
      !["publishing", "published"].includes(job.publication?.status),
  );
}

function renderPageChunks(detail) {
  const editable = canEditParsedChunk();
  $("pageChunks").innerHTML = diagnosticItems(
    detail.chunks || [],
    (chunk) => {
      const revision = Number(chunk.manual_revision?.revision || 0);
      return `
        <article class="page-chunk${revision ? " manually-revised" : ""}">
          <header>
            <span>
              <strong>${escapeHtml(chunk.title || "Chunk")} · ${escapeHtml(chunk.content_type || "text")}</strong>
              ${revision ? `<em class="revision-badge">人工修订 v${revision}</em>` : ""}
            </span>
            <span class="chunk-header-actions">
              <span>第 ${escapeHtml(chunk.page_start)}–${escapeHtml(chunk.page_end)} 页</span>
              ${
                editable
                  ? `<button class="chunk-edit-button" type="button" data-edit-chunk="${escapeHtml(chunk.chunk_id)}">编辑</button>`
                  : ""
              }
            </span>
          </header>
          <p>${escapeHtml(chunk.text || "")}</p>
          ${structuredTableMarkup(chunk.table_html)}
          <small>关键词：${escapeHtml((chunk.keywords || []).join("、") || "无")}</small>
          <details>
            <summary>查看发送到 RAGFlow 的实际内容</summary>
            <pre>${escapeHtml(ragflowContentPreview(detail.source_name, chunk))}</pre>
          </details>
        </article>`;
    },
    "该页没有关联 Chunk。",
  );
  $("pageChunks").querySelectorAll("[data-edit-chunk]").forEach((button) => {
    button.addEventListener("click", () => openChunkEditor(button.dataset.editChunk));
  });
}

function openChunkEditor(chunkId) {
  const chunk = state.currentPageDetail?.chunks?.find(
    (item) => item.chunk_id === chunkId,
  );
  if (!chunk || !canEditParsedChunk()) return;
  state.editingChunkId = chunkId;
  $("chunkEditTitle").textContent = chunk.title || "编辑 Chunk";
  $("chunkEditMeta").textContent =
    `第 ${chunk.page_start}–${chunk.page_end} 页 · ${chunk.content_type || "text"}` +
    (chunk.table_html?.length ? " · 保存文字后将替代当前结构化表格" : "");
  $("chunkEditText").value = chunk.text || "";
  $("chunkEditReason").value = "";
  $("chunkEditCount").textContent = `${$("chunkEditText").value.length} 字`;
  setMessage($("chunkEditMessage"), "");
  const dialog = $("chunkEditDialog");
  if (!dialog.open) dialog.showModal();
  $("chunkEditText").focus();
}

async function saveChunkEdit() {
  if (state.savingChunk || !state.selectedJobId || !state.editingChunkId) return;
  const text = $("chunkEditText").value.trim();
  const reason = $("chunkEditReason").value.trim();
  if (!text) {
    setMessage($("chunkEditMessage"), "Chunk 内容不能为空。", true);
    return;
  }
  state.savingChunk = true;
  $("chunkEditSave").disabled = true;
  $("chunkEditCancel").disabled = true;
  setMessage($("chunkEditMessage"), "正在保存人工修订…");
  try {
    const result = await requestJson(
      `${API}/jobs/${state.selectedJobId}/chunks/${encodeURIComponent(state.editingChunkId)}`,
      {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          "X-User-ID": "local-operator",
        },
        body: JSON.stringify({ text, reason: reason || null }),
      },
    );
    $("chunkEditDialog").close();
    setMessage($("detailMessage"), `人工修订 v${result.revision} 已保存，请重新生成发布计划。`);
    await refreshAll();
    if (state.currentPageNumber) await openPageDetail(state.currentPageNumber);
  } catch (error) {
    setMessage($("chunkEditMessage"), error.message, true);
  } finally {
    state.savingChunk = false;
    $("chunkEditSave").disabled = false;
    $("chunkEditCancel").disabled = false;
  }
}

async function openPageDetail(pageNumber) {
  if (!state.selectedJobId || !pageNumber) return;
  const dialog = $("pageDialog");
  $("pageDialogTitle").textContent = `第 ${pageNumber} 页`;
  $("pageDialogMeta").textContent = "正在加载解析结果…";
  const sourceUrl =
    `${API}/jobs/${state.selectedJobId}/source#page=${pageNumber}&zoom=page-width`;
  const sourceFrame = $("pageSourceFrame");
  if (selectedJob()?.source_name?.toLowerCase().endsWith(".docx")) {
    sourceFrame.removeAttribute("src");
    sourceFrame.srcdoc = `
      <style>
        body { font-family: system-ui, sans-serif; padding: 2rem; color: #334155; line-height: 1.6; }
        a { color: #2563eb; }
      </style>
      <h3>Word 文档原件</h3>
      <p>DOCX 没有稳定的浏览器分页预览；右侧显示的是按 Word 显式分页符划分的逻辑页。</p>
      <a href="${escapeHtml(sourceUrl)}" target="_blank">打开或下载原始 Word 文档</a>`;
  } else {
    sourceFrame.removeAttribute("srcdoc");
    sourceFrame.src = sourceUrl;
  }
  if (!dialog.open) dialog.showModal();
  try {
    const detail = await requestJson(
      `${API}/jobs/${state.selectedJobId}/pages/${pageNumber}`,
    );
    state.currentPageNumber = pageNumber;
    state.currentPageDetail = detail;
    const page = detail.page || {};
    const articles = page.article_ids || [];
    $("pageDialogMeta").textContent =
      `${detail.source_name} · ${page.route || "unknown"} · ` +
      `${page.cleaned_char_count || 0} 个清洗字符`;
    $("pageInspectorSummary").innerHTML = `
      <span><strong>页面路由</strong>${escapeHtml(page.route || "unknown")}</span>
      <span><strong>页面类型</strong>${escapeHtml(page.page_type || "unknown")}</span>
      <span><strong>条号</strong>${escapeHtml(articles.join("、") || "未识别")}</span>
      <span><strong>内容块</strong>${escapeHtml(page.block_count || 0)}</span>
      <span><strong>Chunk</strong>${escapeHtml(page.chunk_count || 0)}</span>
      ${
        page.layout_features?.toc_score != null
          ? `<span><strong>目录分</strong>${escapeHtml(Number(page.layout_features.toc_score).toFixed(2))}</span>`
          : ""
      }
      ${
        page.layout_features?.dominant_font_size
          ? `<span><strong>主字号</strong>${escapeHtml(page.layout_features.dominant_font_size)}pt</span>`
          : ""
      }
      ${
        page.layout_features?.image_count != null
          ? `<span><strong>图片</strong>${escapeHtml(page.layout_features.image_count)}</span>`
          : ""
      }
    `;
    $("pageCleanedText").textContent = page.cleaned_text || "该页没有清洗后文本。";
    $("pageRawText").textContent = page.raw_text || "该页没有可提取的原始文本层。";
    $("pageAssets").innerHTML = diagnosticItems(
      detail.assets || [],
      (asset) => `
        <figure class="page-asset">
          <a href="${escapeHtml(asset.url || "#")}" target="_blank" rel="noreferrer">
            <img
              src="${escapeHtml(asset.url || "")}"
              alt="${escapeHtml(asset.caption || asset.asset_type || "解析资产")}"
              loading="lazy"
            />
          </a>
          <figcaption>
            <strong>${escapeHtml(asset.asset_type || "asset")}</strong>
            ${escapeHtml(asset.caption || "")}
            ${asset.description ? `<p>${escapeHtml(asset.description)}</p>` : ""}
          </figcaption>
        </figure>`,
      "该页没有生成表格或图片资产。",
    );
    $("pageBlocks").innerHTML = diagnosticItems(
      detail.blocks || [],
      (block) => `
        <article>
          <header>
            <strong>${escapeHtml(block.block_type || "block")}</strong>
            <span>${escapeHtml(block.article_id_normalized || "无条号")}</span>
          </header>
          <small>
            ${block.table_id ? `表格 ID：${escapeHtml(block.table_id)} · ` : ""}
            来源页：${escapeHtml(block.source_page_start || block.page_number || pageNumber)}
            –${escapeHtml(block.source_page_end || block.page_number || pageNumber)}
          </small>
          <p>${escapeHtml(block.text || "")}</p>
          ${structuredTableMarkup(block.table_html)}
        </article>`,
      "该页没有内容块。",
    );
    renderPageChunks(detail);
    const traceItems = [
      ...(detail.parser_trace || []).map((item) => ({
        label: `解析器：${item.parser || "unknown"}`,
        value: JSON.stringify(item, null, 2),
      })),
      ...(detail.quality_gates || []).map((item) => ({
        label: `门禁：${item.gate || "unknown"}`,
        value: `${item.status || "unknown"} · ${item.message || ""}`,
      })),
    ];
    $("pageTrace").innerHTML = diagnosticItems(
      traceItems,
      (item) => `
        <article>
          <header><strong>${escapeHtml(item.label)}</strong></header>
          <pre>${escapeHtml(item.value)}</pre>
        </article>`,
      "没有解析器轨迹或门禁信息。",
    );
  } catch (error) {
    $("pageDialogMeta").textContent = error.message;
    $("pageCleanedText").textContent = "页面诊断加载失败。";
  }
}

function renderProgressElements(prefix, progress, labels) {
  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
  $(`${prefix}ProgressLabel`).textContent =
    labels[progress.stage] || labels[progress.status] || progress.stage;
  $(`${prefix}ProgressValue`).textContent = `${percent}%`;
  $(`${prefix}ProgressMessage`).textContent = progress.message;
  const track = $(`${prefix}ProgressTrack`);
  const fill = $(`${prefix}ProgressFill`);
  track.setAttribute("aria-valuenow", String(percent));
  track.classList.toggle("indeterminate", Boolean(progress.indeterminate));
  fill.style.width = `${Math.max(progress.indeterminate ? 18 : 0, percent)}%`;
}

function renderProgress(job) {
  const processing = normalizedProcessingProgress(job);
  renderProgressElements("processing", processing, progressStageNames);

  const publication = normalizedPublicationProgress(job);
  const panel = $("publicationProgress");
  panel.hidden = publication.status === "not_requested";
  if (!panel.hidden) {
    renderProgressElements(
      "publication",
      {
        ...publication,
        indeterminate: publication.status === "publishing" && publication.percent <= 2,
      },
      publicationStageNames,
    );
  }
}

function reviewQueueJobs() {
  return reviewOrderedJobs().filter((job) => !isPublished(job));
}

function adjacentReviewJob(direction) {
  const jobs = reviewQueueJobs();
  const currentIndex = jobs.findIndex((job) => job.job_id === state.selectedJobId);
  if (currentIndex < 0) return null;
  return jobs[currentIndex + direction] || null;
}

function nextPendingReviewJob() {
  const jobs = reviewQueueJobs();
  if (!jobs.length) return null;
  const currentIndex = jobs.findIndex((job) => job.job_id === state.selectedJobId);
  const orderedCandidates = [
    ...jobs.slice(currentIndex + 1),
    ...jobs.slice(0, Math.max(0, currentIndex)),
  ];
  return (
    orderedCandidates.find(
      (job) =>
        ["unreviewed", "in_review"].includes(jobReviewStatus(job)) &&
        !["queued", "running"].includes(job.status),
    ) || null
  );
}

function renderReviewPanel(job) {
  const reviewStatus = jobReviewStatus(job);
  const jobs = reviewQueueJobs();
  const currentIndex = jobs.findIndex((item) => item.job_id === job.job_id);
  const approvedCount = jobs.filter((item) => jobReviewStatus(item) === "approved").length;
  const decidedCount = jobs.filter((item) =>
    ["approved", "rework"].includes(jobReviewStatus(item)),
  ).length;
  const badge = $("reviewStatusBadge");
  badge.textContent = reviewStatusNames[reviewStatus] || reviewStatus;
  badge.className = `review-status-badge ${reviewStatus}`;
  $("reviewProgressSummary").textContent =
    `已完成 ${decidedCount}/${jobs.length} · 已通过 ${approvedCount}`;
  $("reviewQueuePosition").textContent =
    currentIndex >= 0
      ? `审核序列第 ${currentIndex + 1}/${jobs.length} 份 · 按首次上传顺序固定排列`
      : "该资料已离开待审核序列";

  const note = $("reviewNote");
  if (document.activeElement !== note) note.value = job.review?.note || "";

  const publicationIsRunning =
    job.publication?.status === "publishing" || state.publishingJobId === job.job_id;
  const reviewLocked =
    ["queued", "running"].includes(job.status) || publicationIsRunning || isPublished(job);
  const canApprove =
    !reviewLocked && ["completed", "warning"].includes(job.status);
  const canOverrideQualityGate = !reviewLocked && job.status === "needs_review";
  note.disabled = reviewLocked;
  $("previousReviewButton").disabled = !adjacentReviewJob(-1);
  $("nextReviewButton").disabled = !adjacentReviewJob(1);
  $("saveReviewNextButton").disabled = reviewLocked;
  $("reworkReviewNextButton").disabled = reviewLocked;
  $("overrideReviewButton").disabled = !canOverrideQualityGate;
  $("overrideReviewButton").hidden = job.status !== "needs_review";
  $("overrideReviewButton").title = canOverrideQualityGate
    ? "确认已人工核对并放行当前失败质量门禁"
    : "仅待复核且尚未发布的资料可以人工放行";
  $("approveReviewNextButton").disabled = !canApprove;
  $("approveReviewNextButton").title = canApprove
    ? "保存审核通过结论并打开下一份待审核资料"
    : "只有质量通过或需要抽检的资料才能审核通过";
}

async function navigateReview(direction) {
  const target = adjacentReviewJob(direction);
  if (target) await selectJob(target.job_id);
}

async function updateReviewStatus(status, moveNext = true, qualityGateOverride = false) {
  if (!state.selectedJobId) return;
  const job = state.jobs.find((item) => item.job_id === state.selectedJobId);
  if (!job) return;
  const note = $("reviewNote").value.trim();
  if (status === "rework" && !note) {
    setMessage($("detailMessage"), "标记为需返工时，请填写具体问题。", true);
    $("reviewNote").focus();
    return;
  }
  if (qualityGateOverride) {
    const confirmed = window.confirm(
      "请确认：你已检查并在需要时人工修订失败门禁涉及的内容，确认当前内容可安全发布。该操作会记录确认人、时间及被放行的门禁，后续重新编辑或解析后自动失效。是否继续？",
    );
    if (!confirmed) return;
  }
  const nextJob = moveNext ? nextPendingReviewJob() : null;
  [
    "saveReviewNextButton",
    "reworkReviewNextButton",
    "overrideReviewButton",
    "approveReviewNextButton",
  ].forEach((id) => {
    $(id).disabled = true;
  });
  try {
    await requestJson(`${API}/jobs/${state.selectedJobId}/review`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        "X-User-ID": "local-operator",
      },
      body: JSON.stringify({
        status,
        note: note || null,
        quality_gate_override: qualityGateOverride,
      }),
    });
    if (nextJob) state.selectedJobId = nextJob.job_id;
    setMessage(
      $("detailMessage"),
      nextJob
        ? `${reviewStatusNames[status]}已保存，已打开下一份待审核资料。`
        : `${reviewStatusNames[status]}已保存，当前没有下一份待审核资料。`,
    );
    await refreshAll();
  } catch (error) {
    setMessage($("detailMessage"), error.message, true);
    if (job) renderReviewPanel(job);
  }
}

async function selectJob(jobId) {
  state.selectedJobId = jobId;
  renderJobs();
  const job = state.jobs.find((item) => item.job_id === jobId);
  if (!job) return;
  $("detailCard").hidden = false;
  $("detailTitle").textContent = job.source_name;
  $("detailMeta").textContent =
    `${job.job_id.slice(0, 12)} · ${job.knowledge_base_id || "未指定目标知识库"}` +
    (job.llm_structure_enabled ? " · LLM 结构增强" : " · 原生结构解析");
  $("detailStatus").textContent = statusNames[job.status] || job.status;
  $("detailStatus").className = `status-badge ${job.status}`;
  renderProgress(job);
  renderReviewPanel(job);

  const publicationIsRunning =
    job.publication?.status === "publishing" || state.publishingJobId === job.job_id;
  const canRetry = !["queued", "running"].includes(job.status) && !publicationIsRunning;
  const canPlan =
    (["completed", "warning"].includes(job.status) || hasManualQualityGateOverride(job)) &&
    !publicationIsRunning &&
    !isPublished(job);
  const canPublish = canPlan && jobReviewStatus(job) === "approved";
  $("retryButton").disabled = !canRetry;
  $("dryRunButton").disabled = !canPlan;
  $("publishButton").disabled = !canPublish;
  $("exportButton").disabled = Boolean(state.exportingJobId) || !["completed", "warning", "needs_review"].includes(job.status);
  $("publishButton").textContent =
    job.publication?.status === "failed" ? "重新发送到知识库" : "发布到测试知识库";
  $("publishButton").title = canPublish
    ? "需要已配置 RAGFlow API Key"
    : publicationIsRunning
      ? "正在发布，请等待完成"
      : !canPlan
        ? "质量门禁尚未通过"
        : "请先完成人工审核并标记为已通过";

  let preview = null;
  if (job.output_path) {
    try {
      preview = await requestJson(`${API}/jobs/${jobId}/preview?chunk_limit=10`);
    } catch {
      preview = null;
    }
  }
  renderQuality(job, preview);
}

async function refreshServices() {
  try {
    const [parsers, ready] = await Promise.all([
      requestJson(`${API}/parsers/status`),
      requestJson("/ready"),
    ]);
    const ragflow = ready.dependencies?.find((item) => item.name === "ragflow");
    const services = [
      {
        name: "RAGFlow",
        ready: ragflow?.status === "ready",
        enabled: ragflow?.enabled,
      },
      ...parsers.map((item) => ({
        name: item.name,
        ready: item.ready,
        enabled: item.enabled,
      })),
    ];
    $("serviceStrip").innerHTML = services
      .map((service) => {
        const className = !service.enabled ? "muted" : service.ready ? "" : "offline";
        const label = !service.enabled ? "未启用" : service.ready ? "就绪" : "不可用";
        return `<span class="service-pill ${className}">${escapeHtml(service.name)} · ${label}</span>`;
      })
      .join("");
  } catch {
    $("serviceStrip").innerHTML =
      '<span class="service-pill offline">Gateway · 状态获取失败</span>';
  }
}

async function loadAllJobs() {
  const jobs = [];
  const pageSize = 200;
  for (let offset = 0; ; offset += pageSize) {
    const page = await requestJson(`${API}/jobs?limit=${pageSize}&offset=${offset}`);
    jobs.push(...page);
    if (page.length < pageSize) break;
  }
  return jobs.sort(compareReviewOrder);
}

async function refreshAll() {
  try {
    const [jobs, queue] = await Promise.all([
      loadAllJobs(),
      requestJson(`${API}/queue`),
    ]);
    state.jobs = jobs;
    $("activeMetric").textContent = queue.active_job_id ? "1" : "0";
    $("queuedMetric").textContent = queue.queued_job_ids.length;
    renderJobs();
    if (state.selectedJobId) await selectJob(state.selectedJobId);
  } catch (error) {
    setMessage($("formMessage"), error.message, true);
  }
}

async function manualRefresh() {
  if (state.refreshing) return;
  state.refreshing = true;
  const button = $("refreshButton");
  button.disabled = true;
  button.classList.add("refreshing");
  $("refreshLabel").textContent = "刷新中";
  try {
    await Promise.all([refreshAll(), refreshServices()]);
    $("lastRefresh").textContent = `已更新 ${new Date().toLocaleTimeString("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    })}`;
  } finally {
    state.refreshing = false;
    button.disabled = false;
    button.classList.remove("refreshing");
    $("refreshLabel").textContent = "刷新";
  }
}

async function retrySelected() {
  if (!state.selectedJobId) return;
  try {
    await requestJson(`${API}/jobs/${state.selectedJobId}/retry`, { method: "POST" });
    setMessage($("detailMessage"), "任务已重新进入队列。");
    await refreshAll();
  } catch (error) {
    setMessage($("detailMessage"), error.message, true);
  }
}

async function deleteJob(jobId) {
  if (state.deletingJobId) return;
  const job = state.jobs.find((item) => item.job_id === jobId);
  if (!job || !isDeletable(job)) return;
  const published = isPublished(job);
  const confirmed = window.confirm(
    `确定从资料接入台删除“${job.source_name}”吗？\n\n` +
      (published
        ? "这只会清理资料接入台中的发送记录和本地解析结果，不会删除或修改 RAGFlow 中的任何知识库、文档或 Chunk；"
        : "这会删除该任务和本地解析结果；") +
      "如果接入台保存的原文件副本没有被其他任务使用，也会一并清理。" +
      "不会删除您电脑上最初选择上传的文件。",
  );
  if (!confirmed) return;

  state.deletingJobId = jobId;
  renderJobs();
  try {
    await requestJson(`${API}/jobs/${jobId}`, { method: "DELETE" });
    state.selectedPublishJobIds.delete(jobId);
    if (state.selectedJobId === jobId) {
      state.selectedJobId = null;
      $("detailCard").hidden = true;
    }
    setMessage(
      $("formMessage"),
      published
        ? `已从资料接入台移除“${job.source_name}”；RAGFlow 内容未受影响。`
        : `已删除“${job.source_name}”及其本地解析结果。`,
    );
    await refreshAll();
  } catch (error) {
    setMessage($("formMessage"), error.message, true);
  } finally {
    state.deletingJobId = null;
    renderJobs();
  }
}

async function exportSelected() {
  if (!state.selectedJobId || state.exportingJobId) return;
  const job = state.jobs.find((item) => item.job_id === state.selectedJobId);
  if (!job || !["completed", "warning", "needs_review"].includes(job.status)) return;

  state.exportingJobId = job.job_id;
  $("exportButton").disabled = true;
  setMessage($("detailMessage"), "正在生成处理结果导出包…");
  try {
    const response = await fetch(`${API}/jobs/${job.job_id}/export?format=zip`);
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new Error(payload?.detail || payload?.error?.message || `导出失败（HTTP ${response.status}）`);
    }
    const blob = await response.blob();
    const disposition = response.headers.get("content-disposition") || "";
    const match = disposition.match(/filename="?([^";]+)"?/i);
    const filename = match?.[1] || `${job.source_name.replace(/\.[^.]+$/, "")}-cleaned.zip`;
    downloadBlob(blob, filename);
    setMessage($("detailMessage"), `已导出处理结果：${filename}`);
  } catch (error) {
    setMessage($("detailMessage"), error.message, true);
  } finally {
    state.exportingJobId = null;
    $("exportButton").disabled = false;
  }
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function batchExportSelected() {
  if (state.batchPublishing || state.batchExporting || !state.selectedPublishJobIds.size) return;
  const jobIds = state.jobs
    .filter((job) => state.selectedPublishJobIds.has(job.job_id) && isPublishable(job))
    .map((job) => job.job_id);
  if (!jobIds.length) return;

  state.batchExporting = true;
  updateBatchControls();
  setMessage($("formMessage"), `正在打包 ${jobIds.length} 份处理结果…`);
  try {
    const response = await fetch(`${API}/jobs/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_ids: jobIds }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new Error(payload?.detail || payload?.error?.message || `批量导出失败（HTTP ${response.status}）`);
    }
    const blob = await response.blob();
    const disposition = response.headers.get("content-disposition") || "";
    const match = disposition.match(/filename="?([^";]+)"?/i);
    const filename = match?.[1] || "cleaned-export.zip";
    downloadBlob(blob, filename);
    setMessage($("formMessage"), `已导出 ${jobIds.length} 份处理结果：${filename}`);
  } catch (error) {
    setMessage($("formMessage"), error.message, true);
  } finally {
    state.batchExporting = false;
    updateBatchControls();
  }
}
async function publishSelected(dryRun) {
  if (!state.selectedJobId || state.publishingJobId) return;
  const job = state.jobs.find((item) => item.job_id === state.selectedJobId);
  const datasetId = $("datasetInput").value.trim() || job?.knowledge_base_id || "";
  if (!datasetId) {
    setMessage($("detailMessage"), "请先填写目标 RAGFlow Dataset ID。", true);
    return;
  }
  if (!dryRun) {
    const confirmed = window.confirm(
      `即将把 ${job.source_name} 发布到测试知识库 ${datasetId}。是否继续？`,
    );
    if (!confirmed) return;
  }
  state.publishingJobId = dryRun ? null : state.selectedJobId;
  if (!dryRun) {
    job.publication = {
      ...(job.publication || {}),
      status: "publishing",
      stage: "uploading_document",
      percent: 2,
      message: "正在向 RAGFlow 上传原始文件",
    };
    renderProgress(job);
    $("publishButton").disabled = true;
    $("dryRunButton").disabled = true;
  }
  try {
    setMessage(
      $("detailMessage"),
      dryRun ? "正在生成发布计划…" : "正在向 RAGFlow 发布，可在进度条查看状态…",
    );
    const result = await requestJson(`${API}/jobs/${state.selectedJobId}/publish`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, dry_run: dryRun }),
    });
    const publication = result.publication;
    setMessage(
      $("detailMessage"),
      dryRun
        ? `计划已生成：${result.plan.chunk_count} 个 Chunk，哈希 ${result.plan.plan_sha256.slice(0, 12)}…`
        : `发布完成：新增 ${publication.published_chunk_count}，跳过 ${publication.skipped_chunk_count}。`,
    );
  } catch (error) {
    setMessage($("detailMessage"), error.message, true);
  } finally {
    state.publishingJobId = null;
    await refreshAll();
  }
}

async function batchPublishSelected() {
  if (state.batchPublishing || !state.selectedPublishJobIds.size) return;
  const jobs = state.jobs.filter(
    (job) => state.selectedPublishJobIds.has(job.job_id) && isPublishable(job),
  );
  if (!jobs.length) return;
  const defaultDatasetId = $("datasetInput").value.trim();
  const missingTarget = jobs.find(
    (job) => !(defaultDatasetId || job.knowledge_base_id || "").trim(),
  );
  if (missingTarget) {
    setMessage(
      $("formMessage"),
      `“${missingTarget.source_name}”没有目标 Dataset ID，请先在左侧填写。`,
      true,
    );
    return;
  }
  const confirmed = window.confirm(
    `即将按顺序发送 ${jobs.length} 份已审核资料。失败的资料会保留，可再次重试。是否继续？`,
  );
  if (!confirmed) return;

  state.batchPublishing = true;
  state.batchProgress = {
    current: 0,
    total: jobs.length,
    message: `准备发送 ${jobs.length} 份资料`,
  };
  updateBatchControls();
  let succeeded = 0;
  const failed = [];

  for (let index = 0; index < jobs.length; index += 1) {
    const job = jobs[index];
    const datasetId = (defaultDatasetId || job.knowledge_base_id || "").trim();
    state.publishingJobId = job.job_id;
    state.batchProgress = {
      current: index,
      total: jobs.length,
      message: `正在发送：${job.source_name}`,
    };
    updateBatchControls();
    try {
      await requestJson(`${API}/jobs/${job.job_id}/publish`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: datasetId, dry_run: false }),
      });
      succeeded += 1;
      state.selectedPublishJobIds.delete(job.job_id);
    } catch (error) {
      failed.push(`${job.source_name}：${error.message}`);
    }
    state.batchProgress = {
      current: index + 1,
      total: jobs.length,
      message: `已处理 ${index + 1}/${jobs.length}`,
    };
    await refreshAll();
  }

  state.publishingJobId = null;
  state.batchPublishing = false;
  state.batchProgress.message = failed.length
    ? `批量发送完成：成功 ${succeeded}，失败 ${failed.length}`
    : `批量发送完成：成功 ${succeeded}`;
  updateBatchControls();
  setMessage(
    $("formMessage"),
    failed.length
      ? `${state.batchProgress.message}。失败项仍在“已通过”中，可再次发送。`
      : state.batchProgress.message,
    failed.length > 0,
  );
  await refreshAll();
}

function initializeEvents() {
  $("fileInput").addEventListener("change", (event) => setSelectedFiles(event.target.files));
  $("folderInput").addEventListener("change", (event) =>
    setSelectedFiles(event.target.files, "folder"),
  );
  $("selectFilesButton").addEventListener("click", () => $("fileInput").click());
  $("selectFolderButton").addEventListener("click", () => $("folderInput").click());
  $("uploadButton").addEventListener("click", uploadFiles);
  $("saveLlmStructureSettings").addEventListener("click", saveLlmStructureSettings);
  $("refreshButton").addEventListener("click", manualRefresh);
  $("jobSearchInput").addEventListener("input", (event) => {
    state.jobQuery = event.target.value;
    $("clearJobSearch").hidden = !state.jobQuery.trim();
    window.clearTimeout(state.jobSearchTimer);
    state.jobSearchTimer = window.setTimeout(renderJobs, 150);
  });
  $("clearJobSearch").addEventListener("click", () => {
    $("jobSearchInput").value = "";
    state.jobQuery = "";
    $("jobSearchInput").focus();
    renderJobs();
  });
  $("retryButton").addEventListener("click", retrySelected);
  $("dryRunButton").addEventListener("click", () => publishSelected(true));
  $("publishButton").addEventListener("click", () => publishSelected(false));
  $("exportButton").addEventListener("click", exportSelected);
  $("batchPublishButton").addEventListener("click", batchPublishSelected);
  $("batchExportButton").addEventListener("click", batchExportSelected);
  $("previousReviewButton").addEventListener("click", () => navigateReview(-1));
  $("nextReviewButton").addEventListener("click", () => navigateReview(1));
  $("saveReviewNextButton").addEventListener("click", () =>
    updateReviewStatus("in_review"),
  );
  $("reworkReviewNextButton").addEventListener("click", () =>
    updateReviewStatus("rework"),
  );
  $("approveReviewNextButton").addEventListener("click", () =>
    updateReviewStatus("approved"),
  );
  $("overrideReviewButton").addEventListener("click", () =>
    updateReviewStatus("approved", false, true),
  );
  $("pageDialogClose").addEventListener("click", () => $("pageDialog").close());
  $("pageDialog").addEventListener("close", () => {
    state.currentPageNumber = null;
    state.currentPageDetail = null;
  });
  $("chunkEditClose").addEventListener("click", () => $("chunkEditDialog").close());
  $("chunkEditCancel").addEventListener("click", () => $("chunkEditDialog").close());
  $("chunkEditText").addEventListener("input", (event) => {
    $("chunkEditCount").textContent = `${event.target.value.length} 字`;
  });
  $("chunkEditForm").addEventListener("submit", (event) => {
    event.preventDefault();
    saveChunkEdit();
  });
  $("firstIssuePageButton").addEventListener("click", () => {
    if (state.firstIssuePage) openPageDetail(Number(state.firstIssuePage.page_number));
  });
  $("selectAllApproved").addEventListener("change", (event) => {
    state.jobs.filter(isPublishable).forEach((job) => {
      if (event.target.checked) {
        state.selectedPublishJobIds.add(job.job_id);
      } else {
        state.selectedPublishJobIds.delete(job.job_id);
      }
    });
    renderJobs();
  });
  document.querySelectorAll("[data-job-view]").forEach((button) => {
    button.addEventListener("click", () => {
      state.jobView = button.dataset.jobView;
      renderJobs();
    });
  });

  const dropzone = $("dropzone");
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    });
  });
  dropzone.addEventListener("drop", (event) => setSelectedFiles(event.dataTransfer.files));
}

async function boot() {
  initializeEvents();
  updateFileSelection();
  await Promise.all([refreshServices(), refreshAll(), loadLlmStructureSettings()]);
  window.setInterval(refreshAll, 3000);
  window.setInterval(refreshServices, 15000);
}

boot();
