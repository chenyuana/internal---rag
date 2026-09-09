"use strict";
window.CitationPdf = (() => {
  let request = 0, controller = null, objectUrl = null;
  const el = (id) => document.getElementById(id);
  function resolvePageNumber(citation) {
    const explicit = Number(citation.page_number);
    if (Number.isSafeInteger(explicit) && explicit > 0) return explicit;
    // Legacy conversations may retain only the publisher's source header.
    // Never interpret regulation numbers or page mentions in the body as locations.
    const header = String(citation.quote || "").trim().split(/\r?\n\s*\r?\n/, 1)[0].slice(0, 2000);
    if (!/^文档[：:]/.test(header)) return null;
    const match = header.match(/(?:^|\s)页码[：:]\s*([1-9]\d{0,5})(?:\s*[-–—]\s*([1-9]\d{0,5}))?(?=\s|$)/);
    if (!match || (match[2] && Number(match[2]) < Number(match[1]))) return null;
    return Number(match[1]);
  }
  function release() {
    request += 1;
    controller?.abort();
    controller = null;
    window.CitationPdfRenderer.clear();
    el("citationPdfFrame").src = "about:blank";
    el("citationPdfFrame").hidden = true;
    el("citationPdfOpen").hidden = true;
    el("citationPdfOpen").removeAttribute("href");
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  async function open(citation, fallbackDataset) {
    release();
    const current = request;
    const dialog = el("citationPdfDialog");
    el("citationPdfTitle").textContent = `[${citation.citation_id}] ${citation.document_name || "引用原文"}`;
    el("citationPdfQuote").textContent = citation.quote || "";
    el("citationPdfStatus").textContent = "正在加载原始 PDF…";
    if (!dialog.open) dialog.showModal();
    const dataset = citation.dataset_id || fallbackDataset;
    if (citation.source_path === "temporary-upload" || dataset === "temporary-auxiliary") {
      el("citationPdfStatus").textContent = "该引用来自临时辅助资料，仅保留了文本，暂不能打开原始 PDF。";
      return;
    }
    if (!dataset || !citation.document_id) {
      el("citationPdfStatus").textContent = "该历史引用缺少文档定位信息，请重新提问以生成完整引用。";
      return;
    }
    controller = new AbortController();
    try {
      const url = `/api/v1/documents/document/${encodeURIComponent(dataset)}/${encodeURIComponent(citation.document_id)}/file`;
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error?.message || payload.message || payload.detail || "无法加载原始 PDF，请确认文档仍然可用。");
      }
      const blob = await response.blob();
      if (current !== request) return;
      if (!(await blob.slice(0, 1024).text()).includes("%PDF-")) throw new Error("返回的文件不是 PDF，无法定位原文。");
      if (current !== request) return;
      objectUrl = URL.createObjectURL(new Blob([blob], { type: "application/pdf" }));
      const page = resolvePageNumber(citation);
      const hasPage = Number.isSafeInteger(page) && page > 0;
      const target = `${objectUrl}#page=${hasPage ? page : 1}&view=FitH`;
      el("citationPdfOpen").href = target;
      el("citationPdfOpen").hidden = false;
      const rendered = await window.CitationPdfRenderer.open(blob, hasPage ? page : 1);
      if (current !== request || !rendered) return;
      el("citationPdfStatus").textContent = hasPage
        ? `已显示原始 PDF 第 ${rendered.page} / ${rendered.total} 页。可对照引用文字核对；尚不支持段落高亮。`
        : "该引用没有可靠页码，当前显示首页，未执行引用定位；可使用上方页码切换。";
    } catch (error) {
      if (current === request && error.name !== "AbortError") el("citationPdfStatus").textContent = error.message;
    }
  }
  el("closeCitationPdf").addEventListener("click", () => el("citationPdfDialog").close());
  el("citationPdfDialog").addEventListener("close", release);
  return { open, resolvePageNumber };
})();
