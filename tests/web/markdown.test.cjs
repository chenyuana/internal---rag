"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  decodeTextEntities,
  referencedCitationTables,
  renderMarkdown,
  splitTableRow,
} = require("../../app/web/static/markdown.js");

test("renders a GitHub-style comparison table with inline formatting", () => {
  const markdown = [
    "| 对比项目 | 地质灾害倾斜航测 | 河湖智能巡检无人机 |",
    "| :--- | :--- | :--- |",
    "| **重叠度要求** | 航向重叠度为65%~85% [C1]。 | 当前资料中未明确说明。 |",
    "| **其他规定** | - | 支持实时传输图像数据 [C4]。 |",
  ].join("\n");

  const html = renderMarkdown(markdown);

  assert.match(html, /<table>/);
  assert.match(html, /<th scope="col">对比项目<\/th>/);
  assert.match(html, /<strong>重叠度要求<\/strong>/);
  assert.match(html, /data-citation-id="C1"/);
  assert.match(html, /<tbody><tr>/);
  assert.doesNotMatch(html, /:---/);
});

test("escapes untrusted HTML before applying supported Markdown", () => {
  const html = renderMarkdown("| 项目 | 内容 |\n| --- | --- |\n| **安全** | <img src=x onerror=alert(1)> [C2] |");

  assert.doesNotMatch(html, /<img/);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.match(html, /<strong>安全<\/strong>/);
});

test("supports escaped pipes inside table cells", () => {
  assert.deepEqual(splitTableRow("| 参数 | A \\| B |"), ["参数", "A | B"]);
});

test("preserves ordinary answer paragraphs, lists, code and citations", () => {
  const html = renderMarkdown("结论 [C1]\n\n- **要求一**\n- 参数 `ABC_1`");

  assert.match(html, /^<p>结论 <button class="citation-marker"/);
  assert.match(html, /<ul><li><strong>要求一<\/strong><\/li><li>参数 <code>ABC_1<\/code><\/li><\/ul>/);
});

test("shows only the table explicitly named beside a citation", () => {
  const citation = {
    citation_id: "C2",
    table_htmls: [
      "<table><caption>表A.1 航摄飞行记录表</caption><tr><td>甲</td></tr></table>",
      "<table><caption>表3 无人机航测成果精度要求</caption><tr><td>乙</td></tr></table>",
    ],
  };

  assert.deepEqual(
    referencedCitationTables("平面和高程限差见表3。[C2]", citation),
    [citation.table_htmls[1]],
  );
  assert.deepEqual(
    referencedCitationTables("成果清单按附录A.5提交。[C2]", citation),
    [],
  );
});

test("matches spaced appendix table numbers and decodes numeric entities", () => {
  const citation = {
    citation_id: "C6",
    table_html: "<table><caption>&#x8868; A.3 平面精度检查记录表</caption></table>",
  };
  assert.equal(decodeTextEntities("适&#x5F53;"), "适当");
  assert.equal(referencedCitationTables("记录样式见表A.3。[C6]", citation).length, 1);
});
