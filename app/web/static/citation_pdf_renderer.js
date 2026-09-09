"use strict";
// Explicit page rendering: no dependency on a browser PDF plugin's fragment handling.
window.CitationPdfRenderer = (() => {
  let generation = 0, loading = null, pdf = null, rendering = null, currentPage = 1;
  const el = (id) => document.getElementById(id);
  function clear() {
    generation += 1;
    rendering?.cancel();
    rendering = null;
    if (loading) loading.destroy().catch(() => {});
    loading = null;
    pdf = null;
    el("citationPdfPages").hidden = true;
    el("citationPdfCanvasHost").replaceChildren();
  }
  async function show(pageNumber) {
    if (!pdf || !Number.isSafeInteger(pageNumber) || pageNumber < 1 || pageNumber > pdf.numPages) {
      throw new Error("引用页码不在此 PDF 的有效范围内，无法可靠定位。");
    }
    const run = ++generation;
    rendering?.cancel();
    const page = await pdf.getPage(pageNumber);
    if (run !== generation) return null;
    const natural = page.getViewport({scale: 1});
    const width = Math.max(240, el("citationPdfCanvasHost").clientWidth - 24);
    const viewport = page.getViewport({scale: Math.min(width / natural.width, 2)});
    const ratio = Math.min(window.devicePixelRatio || 1, 2,
      Math.sqrt(16_000_000 / (viewport.width * viewport.height)));
    const canvas = document.createElement("canvas");
    canvas.width = Math.ceil(viewport.width * ratio);
    canvas.height = Math.ceil(viewport.height * ratio);
    canvas.style.width = `${viewport.width}px`;
    canvas.style.height = `${viewport.height}px`;
    canvas.setAttribute("aria-label", `PDF 原文第 ${pageNumber} 页`);
    rendering = page.render({canvasContext: canvas.getContext("2d"), viewport,
      transform: [ratio, 0, 0, ratio, 0, 0]});
    await rendering.promise;
    if (run !== generation) return null;
    currentPage = pageNumber;
    el("citationPdfCanvasHost").replaceChildren(canvas);
    el("citationPdfCanvasHost").scrollTop = 0;
    el("citationPdfPageNumber").value = String(pageNumber);
    el("citationPdfPageNumber").max = String(pdf.numPages);
    el("citationPdfPageTotal").textContent = `/ ${pdf.numPages} 页`;
    el("citationPdfPrevious").disabled = pageNumber === 1;
    el("citationPdfNext").disabled = pageNumber === pdf.numPages;
    el("citationPdfPages").hidden = false;
    return {page: pageNumber, total: pdf.numPages};
  }
  async function open(blob, pageNumber) {
    clear();
    const run = generation;
    const lib = await import("/assets/vendor/pdfjs/build/pdf.mjs");
    const data = new Uint8Array(await blob.arrayBuffer());
    if (run !== generation) return null;
    lib.GlobalWorkerOptions.workerSrc = "/assets/vendor/pdfjs/build/pdf.worker.mjs";
    loading = lib.getDocument({data,
      cMapUrl: "/assets/vendor/pdfjs/cmaps/", cMapPacked: true,
      standardFontDataUrl: "/assets/vendor/pdfjs/standard_fonts/",
      wasmUrl: "/assets/vendor/pdfjs/wasm/",
    });
    const loaded = await loading.promise;
    if (run !== generation) return null;
    pdf = loaded;
    return show(pageNumber);
  }
  async function navigate(page) {
    try {
      const result = await show(page);
      if (result) el("citationPdfStatus").textContent = `当前显示原始 PDF 第 ${result.page} / ${result.total} 页。`;
    } catch (error) {
      if (error.name !== "RenderingCancelledException") el("citationPdfStatus").textContent = error.message;
    }
  }
  el("citationPdfPrevious").addEventListener("click", () => navigate(currentPage - 1));
  el("citationPdfNext").addEventListener("click", () => navigate(currentPage + 1));
  el("citationPdfGo").addEventListener("click", () => navigate(Number(el("citationPdfPageNumber").value)));
  el("citationPdfPageNumber").addEventListener("keydown", (event) => {
    if (event.key === "Enter") navigate(Number(event.target.value));
  });
  return {open, clear};
})();
