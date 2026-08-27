"use strict";

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
const STORAGE_KEY = "internal-rag.chat-conversations.v1";
const STATUS_NAMES = { ANSWERABLE: "资料充分", PARTIALLY_ANSWERABLE: "部分可回答", UNANSWERABLE: "资料不足", CONFLICTED: "证据冲突" };

const state = {
  datasets: [], selectedKbId: null, modelSources: [], modelOptions: new Map(), selectedModel: null,
  conversations: [], activeId: null, referenceDocument: null, busy: false,
};

function uid() { return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`; }
function activeConversation() { return state.conversations.find((item) => item.id === state.activeId); }
function newConversationObject() { const now = new Date().toISOString(); return { id: uid(), title: "新对话", createdAt: now, updatedAt: now, kbId: state.selectedKbId, auxiliaryDocument: null, messages: [] }; }
function compactAnswer(data) {
  return { ...data, diagnostics: undefined, citations: (data.citations || []).map(({ table_html, table_htmls, ...citation }) => citation) };
}

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data?.error?.message || data?.error?.code || data?.detail || `HTTP ${response.status}`);
  return data;
}

function loadConversations() {
  try { state.conversations = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch { state.conversations = []; }
  if (!Array.isArray(state.conversations) || !state.conversations.length) state.conversations = [newConversationObject()];
  state.conversations = state.conversations.slice(0, 20);
  state.activeId = state.conversations[0].id;
  state.referenceDocument = activeConversation()?.auxiliaryDocument || null;
}

function saveConversations() {
  state.conversations.sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)));
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.conversations.slice(0, 20))); }
  catch {
    state.conversations = state.conversations.slice(0, 8).map((conversation) => ({ ...conversation, messages: conversation.messages.slice(-24) }));
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.conversations)); } catch { showMessage("浏览器存储空间不足，新对话暂时无法保留。", false); }
  }
}

function formatTime(value) {
  const date = new Date(value); const today = new Date();
  return date.toDateString() === today.toDateString() ? date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }) : date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function renderConversationList() {
  const list = $("conversationList"); list.innerHTML = "";
  state.conversations.forEach((conversation) => {
    const item = document.createElement("div"); item.className = `conversation-item${conversation.id === state.activeId ? " active" : ""}`;
    item.innerHTML = `<button class="conversation-select" data-conversation-id="${escapeHtml(conversation.id)}"><span class="conversation-title">${escapeHtml(conversation.title || "新对话")}</span><span class="conversation-time">${formatTime(conversation.updatedAt)}</span></button><button class="delete-conversation" data-delete-id="${escapeHtml(conversation.id)}" title="删除对话">×</button>`;
    list.appendChild(item);
  });
  $("conversationCount").textContent = String(state.conversations.length);
}

function createConversation() {
  if (state.busy) return;
  const conversation = newConversationObject(); state.conversations.unshift(conversation); state.activeId = conversation.id;
  state.referenceDocument = null; updateReferenceChip(); saveConversations(); renderAll(); $("questionInput").focus();
}

function switchConversation(id) {
  if (state.busy || id === state.activeId || !state.conversations.some((item) => item.id === id)) return;
  state.activeId = id; const conversation = activeConversation();
  if (conversation?.kbId && state.datasets.some((item) => item.id === conversation.kbId)) { state.selectedKbId = conversation.kbId; $("kbSelect").value = conversation.kbId; }
  state.referenceDocument = conversation?.auxiliaryDocument || null; updateReferenceChip(); renderAll();
}

function deleteConversation(id) {
  if (state.busy) return;
  state.conversations = state.conversations.filter((item) => item.id !== id);
  if (!state.conversations.length) state.conversations = [newConversationObject()];
  if (!state.conversations.some((item) => item.id === state.activeId)) state.activeId = state.conversations[0].id;
  saveConversations(); renderAll();
}

async function loadDatasets() {
  const datasets = await readJson(await fetch("/api/v1/admin/datasets", { headers: { "X-User-ID": "dev" } }));
  if (!Array.isArray(datasets) || !datasets.length) throw new Error("没有可用知识库");
  state.datasets = datasets; const select = $("kbSelect"); select.innerHTML = "";
  datasets.forEach((dataset) => { const option = document.createElement("option"); option.value = dataset.id; option.textContent = dataset.name; select.appendChild(option); });
  const savedKb = activeConversation()?.kbId; state.selectedKbId = datasets.some((item) => item.id === savedKb) ? savedKb : datasets[0].id; select.value = state.selectedKbId;
}

function modelOptionKey(sourceId, modelName) { return `${sourceId}\u001f${modelName}`; }
async function loadModels(preferred = null) {
  const catalog = await readJson(await fetch("/api/v1/admin/answer-models", { headers: { "X-User-ID": "dev" } }));
  state.modelSources = catalog.sources || []; state.modelOptions.clear(); const select = $("modelSelect"); select.innerHTML = "";
  state.modelSources.forEach((source) => { const group = document.createElement("optgroup"); group.label = `${source.source_type === "local" ? "本地" : "API"} · ${source.name}`; (source.models || []).forEach((modelName) => { const key = modelOptionKey(source.source_id, modelName); state.modelOptions.set(key, { source_id: source.source_id, model_name: modelName }); const option = document.createElement("option"); option.value = key; option.textContent = modelName; option.disabled = source.available === false; group.appendChild(option); }); select.appendChild(group); });
  const preferredKey = preferred ? modelOptionKey(preferred.source_id, preferred.model_name) : modelOptionKey(catalog.default_source_id || "", catalog.default_model || "");
  if (state.modelOptions.has(preferredKey)) select.value = preferredKey;
  if (!select.value && state.modelOptions.size) select.value = state.modelOptions.keys().next().value;
  state.selectedModel = state.modelOptions.get(select.value) || null; updateModelControls();
  if (!state.selectedModel) select.innerHTML = '<option value="">请接入可用模型</option>';
}

async function initialize() {
  loadConversations(); renderAll();
  const results = await Promise.allSettled([loadDatasets(), loadModels()]); const errors = results.filter((item) => item.status === "rejected"); const pill = $("servicePill");
  if (errors.length) { pill.textContent = errors.length === 2 ? "服务不可用" : "部分服务不可用"; pill.className = "service-pill bad"; showMessage(errors.map((item) => item.reason.message).join("；"), false); }
  else { pill.textContent = `${state.datasets.length} 个知识库 · ${state.modelOptions.size} 个模型`; pill.className = "service-pill ok"; }
  const incomingQuestion = new URLSearchParams(window.location.search).get("q");
  if (incomingQuestion) $("questionInput").value = incomingQuestion.slice(0, 2000);
  enableSend(); renderConversationList();
}

function updateModelControls() { const source = state.modelSources.find((item) => item.source_id === state.selectedModel?.source_id); $("removeModelConnection").hidden = !source?.removable; }
function enableSend() { $("sendButton").disabled = !( $("questionInput").value.trim() && state.selectedKbId && state.selectedModel) || state.busy; }
function showMessage(text, ok) { const node = $("formMessage"); node.textContent = text || ""; node.className = `form-message${ok ? " ok" : ""}`; }

function renderAll() {
  renderConversationList(); const conversation = activeConversation(); const messages = conversation?.messages || [];
  $("emptyState").hidden = messages.length > 0; const list = $("messageList"); list.innerHTML = "";
  messages.forEach((message) => list.appendChild(renderMessage(message)));
  if (state.busy) list.appendChild(renderLoading());
  const lastAssistant = [...messages].reverse().find((item) => item.role === "assistant" && item.data);
  renderCitations(lastAssistant?.data?.citations || [], lastAssistant?.id || null);
  requestAnimationFrame(() => { $("conversationStage").scrollTop = $("conversationStage").scrollHeight; });
}

function renderMessage(message) {
  const article = document.createElement("article"); article.className = `message ${message.role}`; article.dataset.messageId = message.id;
  article.innerHTML = `<div class="message-avatar">${message.role === "user" ? "你" : "IR"}</div><div class="message-content"><div class="message-role">${message.role === "user" ? "你" : "智能问答"}</div></div>`;
  const content = article.querySelector(".message-content");
  if (message.role === "user") { const question = document.createElement("p"); question.className = "user-question"; question.textContent = message.content; content.appendChild(question); return article; }
  const data = message.data || {}; const status = data.status || "UNANSWERABLE"; const card = document.createElement("div"); card.className = `answer-card${message.error ? " error-card" : ""}`; card.dataset.answerId = message.id;
  const body = window.ChatMarkdown ? window.ChatMarkdown.renderMarkdown(data.answer || message.content || "") : escapeHtml(data.answer || message.content || "");
  const warnings = [...(data.validation_warnings || []), ...(data.missing_information || []), ...(data.conflicts || [])];
  card.innerHTML = `<div class="answer-head"><span class="answer-status status-${escapeHtml(status)}">${escapeHtml(message.error ? "请求失败" : (STATUS_NAMES[status] || status))}</span><span class="answer-model">${escapeHtml(data.model_name || "")}</span></div><div class="answer-body">${body}</div><div class="trust-line"><span class="trust-dot"></span><span>${message.error ? "请检查服务状态后重试" : `已校验证据 · ${data.citations?.length || 0} 条引用`}</span></div>${warnings.length ? `<details class="answer-details"><summary>查看资料缺失与校验提示（${warnings.length}）</summary><ul>${warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></details>` : ""}<div class="message-actions"><button class="text-button" data-action="sources">查看引用</button><button class="text-button" data-action="copy">复制回答</button></div>`;
  content.appendChild(card); return article;
}

function renderLoading() { const article = document.createElement("article"); article.className = "message assistant"; article.innerHTML = '<div class="message-avatar">IR</div><div class="message-content"><div class="message-role">智能问答</div><div class="answer-card loading-card"><span class="loading-spinner"></span><span>正在检索、生成并校验回答…</span></div></div>'; return article; }

function answerForMessage(id) { return activeConversation()?.messages.find((item) => item.id === id && item.role === "assistant"); }
function selectAnswer(id, citationId = null) {
  const answer = answerForMessage(id); if (!answer?.data) return; renderCitations(answer.data.citations || [], id);
  if (citationId) requestAnimationFrame(() => { const target = document.getElementById(`evidence-${citationId}`); if (target) { target.classList.add("active"); target.scrollIntoView({ behavior: "smooth", block: "nearest" }); } });
}

function renderCitations(citations, answerId) {
  $("evidenceCount").textContent = String(citations.length); $("evidenceIntro").textContent = citations.length ? "点击回答中的引用编号，可定位到对应原文。" : "当前回答没有可展示的引用证据。";
  const list = $("citationsList"); list.innerHTML = "";
  citations.forEach((citation) => { const item = document.createElement("li"); item.className = "citation-card"; item.id = `evidence-${citation.citation_id}`; item.dataset.answerId = answerId || ""; const quote = String(citation.quote || "").replace(/\s+/g, " ").trim().slice(0, 520); const path = [citation.chapter_path, citation.version].filter(Boolean).join(" · "); item.innerHTML = `<div class="cite-top"><span class="cite-id">[${escapeHtml(citation.citation_id)}]</span><span class="cite-page">${citation.page_number ? `第 ${citation.page_number} 页` : ""}</span></div><span class="cite-doc">${escapeHtml(citation.document_name || citation.document_id || "未知文档")}</span>${path ? `<span class="cite-path">${escapeHtml(path)}</span>` : ""}${quote ? `<span class="cite-quote">${escapeHtml(quote)}${quote.length >= 520 ? "…" : ""}</span>` : ""}`; list.appendChild(item); });
}

async function sendQuestion() {
  const question = $("questionInput").value.trim(); const conversation = activeConversation();
  if (!question || !conversation || !state.selectedKbId || !state.selectedModel || state.busy) return;
  state.busy = true; conversation.kbId = state.selectedKbId; conversation.updatedAt = new Date().toISOString(); if (conversation.title === "新对话") conversation.title = question.replace(/\s+/g, " ").slice(0, 26);
  conversation.messages.push({ id: uid(), role: "user", content: question, referenceName: state.referenceDocument?.name || null, createdAt: conversation.updatedAt });
  $("questionInput").value = ""; showMessage("", false); saveConversations(); renderAll(); enableSend();
  try {
    const referenceDocument = state.referenceDocument
      ? { name: state.referenceDocument.name, text: state.referenceDocument.text }
      : null;
    const response = await fetch("/api/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json", "X-User-ID": "dev" }, body: JSON.stringify({ query: question, knowledge_base_ids: [state.selectedKbId], conversation_id: conversation.id, model: state.selectedModel, reference_document: referenceDocument }) });
    const data = compactAnswer(await readJson(response)); conversation.messages.push({ id: uid(), role: "assistant", data, createdAt: new Date().toISOString() });
  } catch (error) {
    conversation.messages.push({ id: uid(), role: "assistant", error: true, data: { status: "UNANSWERABLE", answer: `请求失败：${error.message}`, citations: [] }, createdAt: new Date().toISOString() });
  } finally {
    conversation.updatedAt = new Date().toISOString(); state.busy = false; saveConversations(); renderAll(); enableSend();
  }
}

async function loadReferenceFile(file) {
  if (!file || state.busy) return; const button = $("attachReference"); button.disabled = true; button.textContent = "正在解析…"; showMessage("", false);
  try { const form = new FormData(); form.append("file", file); state.referenceDocument = await readJson(await fetch("/api/v1/chat/reference-documents/parse", { method: "POST", body: form })); const conversation = activeConversation(); if (conversation) { conversation.auxiliaryDocument = state.referenceDocument; conversation.updatedAt = new Date().toISOString(); saveConversations(); } updateReferenceChip(); showMessage(state.referenceDocument.truncated ? "资料较长，已读取前 2 万字并作为辅助证据。" : "辅助资料已加载，将参与回答与引用。", true); }
  catch (error) { state.referenceDocument = null; updateReferenceChip(); showMessage(error.message, false); }
  finally { button.disabled = false; button.innerHTML = "<span>⌕</span> 加载辅助资料"; $("referenceFileInput").value = ""; }
}
function updateReferenceChip() { $("referenceChip").hidden = !state.referenceDocument; $("referenceName").textContent = state.referenceDocument?.name || ""; }

function openModelDialog() { $("connectionMessage").textContent = ""; $("modelDialog").showModal(); }
function closeModelDialog() { $("modelDialog").close(); }
async function connectModel(event) {
  event.preventDefault(); const button = $("connectModelButton"); const message = $("connectionMessage"); button.disabled = true; button.textContent = "正在连接…";
  try { const created = await readJson(await fetch("/api/v1/admin/model-connections", { method: "POST", headers: { "Content-Type": "application/json", "X-User-ID": "dev" }, body: JSON.stringify({ name: $("connectionName").value.trim(), base_url: $("connectionBaseUrl").value.trim(), api_key: $("connectionApiKey").value }) })); const source = created.source; await loadModels({ source_id: source.source_id, model_name: source.default_model || source.models[0] }); $("connectionApiKey").value = ""; message.textContent = `已连接 ${source.name}。`; message.className = "dialog-message ok"; setTimeout(closeModelDialog, 600); enableSend(); }
  catch (error) { message.textContent = error.message; message.className = "dialog-message"; }
  finally { button.disabled = false; button.textContent = "连接并读取模型"; }
}
async function removeCurrentModelConnection() { const sourceId = state.selectedModel?.source_id; const source = state.modelSources.find((item) => item.source_id === sourceId); if (!source?.removable || state.busy) return; try { const response = await fetch(`/api/v1/admin/model-connections/${encodeURIComponent(sourceId)}`, { method: "DELETE", headers: { "X-User-ID": "dev" } }); if (!response.ok) await readJson(response); await loadModels(); enableSend(); } catch (error) { showMessage(error.message, false); } }

$("newConversation").addEventListener("click", createConversation);
$("conversationList").addEventListener("click", (event) => { const select = event.target.closest("[data-conversation-id]"); const remove = event.target.closest("[data-delete-id]"); if (select) switchConversation(select.dataset.conversationId); if (remove) deleteConversation(remove.dataset.deleteId); });
$("kbSelect").addEventListener("change", (event) => { state.selectedKbId = event.target.value; const conversation = activeConversation(); if (conversation) { conversation.kbId = state.selectedKbId; saveConversations(); } enableSend(); });
$("modelSelect").addEventListener("change", (event) => { state.selectedModel = state.modelOptions.get(event.target.value) || null; updateModelControls(); enableSend(); });
$("questionInput").addEventListener("input", enableSend); $("questionInput").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendQuestion(); } });
$("sendButton").addEventListener("click", sendQuestion); $("attachReference").addEventListener("click", () => $("referenceFileInput").click()); $("referenceFileInput").addEventListener("change", (event) => loadReferenceFile(event.target.files[0]));
$("removeReference").addEventListener("click", () => { state.referenceDocument = null; const conversation = activeConversation(); if (conversation) { conversation.auxiliaryDocument = null; saveConversations(); } updateReferenceChip(); showMessage("已移除辅助资料。", true); });
$("messageList").addEventListener("click", async (event) => { const message = event.target.closest(".message"); const marker = event.target.closest("[data-citation-id]"); if (!message) return; if (marker) return selectAnswer(message.dataset.messageId, marker.dataset.citationId); const action = event.target.closest("[data-action]")?.dataset.action; if (action === "sources") selectAnswer(message.dataset.messageId); if (action === "copy") { const answer = answerForMessage(message.dataset.messageId); await navigator.clipboard.writeText(answer?.data?.answer || ""); showMessage("已复制回答。", true); } if (event.target.closest(".answer-card")) selectAnswer(message.dataset.messageId); });
$("openModelDialog").addEventListener("click", openModelDialog); $("removeModelConnection").addEventListener("click", removeCurrentModelConnection); $("closeModelDialog").addEventListener("click", closeModelDialog); $("cancelModelConnection").addEventListener("click", closeModelDialog); $("modelConnectionForm").addEventListener("submit", connectModel); $("modelDialog").addEventListener("click", (event) => { if (event.target === $("modelDialog")) closeModelDialog(); });
document.querySelectorAll(".suggestion").forEach((button) => button.addEventListener("click", () => { $("questionInput").value = button.dataset.q; enableSend(); $("questionInput").focus(); }));

initialize();
