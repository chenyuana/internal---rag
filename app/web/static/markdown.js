"use strict";

(function exposeMarkdownRenderer(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.ChatMarkdown = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  function renderInline(value) {
    let html = escapeHtml(value);
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    return html.replace(
      /\[(C\d+)\]/g,
      '<button class="citation-marker" type="button" data-citation-id="$1" aria-label="查看引用 $1">[$1]</button>',
    );
  }

  function splitTableRow(line) {
    let source = String(line ?? "").trim();
    if (!source.includes("|")) return null;
    if (source.startsWith("|")) source = source.slice(1);
    if (source.endsWith("|")) source = source.slice(0, -1);

    const cells = [];
    let cell = "";
    for (let index = 0; index < source.length; index += 1) {
      const character = source[index];
      if (character === "\\" && source[index + 1] === "|") {
        cell += "|";
        index += 1;
      } else if (character === "|") {
        cells.push(cell.trim());
        cell = "";
      } else {
        cell += character;
      }
    }
    cells.push(cell.trim());
    return cells;
  }

  function separatorAlignment(cell) {
    const value = cell.replaceAll(" ", "");
    if (!/^:?-{3,}:?$/.test(value)) return null;
    if (value.startsWith(":") && value.endsWith(":")) return "center";
    if (value.endsWith(":")) return "right";
    if (value.startsWith(":")) return "left";
    return "left";
  }

  function tableAt(lines, start) {
    if (start + 1 >= lines.length) return null;
    const headers = splitTableRow(lines[start]);
    const separators = splitTableRow(lines[start + 1]);
    if (!headers || !separators || headers.length < 2 || separators.length !== headers.length) {
      return null;
    }
    const alignments = separators.map(separatorAlignment);
    if (alignments.some((alignment) => alignment === null)) return null;

    const rows = [];
    let next = start + 2;
    while (next < lines.length && lines[next].trim()) {
      const cells = splitTableRow(lines[next]);
      if (!cells || cells.length < 2) break;
      rows.push(headers.map((_, index) => cells[index] ?? ""));
      next += 1;
    }
    return { headers, alignments, rows, next };
  }

  function alignmentClass(alignment) {
    return alignment === "center"
      ? ' class="align-center"'
      : alignment === "right"
        ? ' class="align-right"'
        : "";
  }

  function renderTable(table) {
    const head = table.headers
      .map(
        (cell, index) =>
          `<th scope="col"${alignmentClass(table.alignments[index])}>${renderInline(cell)}</th>`,
      )
      .join("");
    const body = table.rows
      .map(
        (row) =>
          `<tr>${row
            .map(
              (cell, index) =>
                `<td${alignmentClass(table.alignments[index])}>${renderInline(cell)}</td>`,
            )
            .join("")}</tr>`,
      )
      .join("");
    return (
      '<div class="markdown-table-wrap" role="region" aria-label="回答对比表" tabindex="0">' +
      `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`
    );
  }

  function isListLine(line) {
    return /^\s*[-*+]\s+/.test(line);
  }

  function renderMarkdown(value) {
    const lines = String(value ?? "").replaceAll("\r\n", "\n").replaceAll("\r", "\n").split("\n");
    const blocks = [];
    let index = 0;

    while (index < lines.length) {
      if (!lines[index].trim()) {
        index += 1;
        continue;
      }

      const table = tableAt(lines, index);
      if (table) {
        blocks.push(renderTable(table));
        index = table.next;
        continue;
      }

      if (isListLine(lines[index])) {
        const items = [];
        while (index < lines.length && isListLine(lines[index])) {
          items.push(lines[index].replace(/^\s*[-*+]\s+/, ""));
          index += 1;
        }
        blocks.push(`<ul>${items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</ul>`);
        continue;
      }

      const paragraph = [];
      while (index < lines.length && lines[index].trim()) {
        if (paragraph.length && (tableAt(lines, index) || isListLine(lines[index]))) break;
        paragraph.push(lines[index].trim());
        index += 1;
      }
      blocks.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
    }

    return blocks.join("");
  }

  function decodeTextEntities(value) {
    return String(value ?? "")
      .replace(/&#x([0-9a-f]+);?/gi, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
      .replace(/&#(\d+);?/g, (_, decimal) => String.fromCodePoint(parseInt(decimal, 10)))
      .replaceAll("&nbsp;", " ")
      .replaceAll("&amp;", "&")
      .replaceAll("&lt;", "<")
      .replaceAll("&gt;", ">");
  }

  function tableTitleFromHtml(tableHtml) {
    const match = String(tableHtml ?? "").match(/<caption\b[^>]*>([\s\S]*?)<\/caption>/i);
    return match
      ? decodeTextEntities(match[1].replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim()
      : "";
  }

  function normalizeTableReference(value) {
    return decodeTextEntities(value)
      .toUpperCase()
      .replace(/[\s：:；;，,。．·_\-]/g, "");
  }

  function referencedCitationTables(answer, citation) {
    const tableHtmls = Array.isArray(citation?.table_htmls) && citation.table_htmls.length
      ? citation.table_htmls
      : citation?.table_html
        ? [citation.table_html]
        : [];
    if (!tableHtmls.length || !citation?.citation_id) return [];

    const contexts = [];
    const source = decodeTextEntities(answer);
    const marker = `[${citation.citation_id}]`;
    let index = source.indexOf(marker);
    while (index >= 0) {
      contexts.push(normalizeTableReference(source.slice(Math.max(0, index - 240), index + 60)));
      index = source.indexOf(marker, index + marker.length);
    }
    if (!contexts.length) return [];

    return tableHtmls.filter((tableHtml) => {
      const title = tableTitleFromHtml(tableHtml);
      const number = title.match(
        /表\s*(?:[A-Za-z]\s*[.．]\s*)?\d+(?:\s*[.．]\s*\d+)*/i,
      )?.[0] || "";
      const keys = [number, title]
        .map(normalizeTableReference)
        .filter((key) => key.length >= 2);
      return keys.some((key) => contexts.some((context) => context.includes(key)));
    });
  }

  return {
    decodeTextEntities,
    referencedCitationTables,
    renderMarkdown,
    splitTableRow,
    tableTitleFromHtml,
  };
});
