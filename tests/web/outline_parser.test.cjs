"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  parseOutlineFromText,
  parseOutlineSections,
  splitTitleAndGuidance,
  suggestGuidance,
} = require("../../app/web/static/outline_parser.js");

test("parses one-line semicolon separated numbered outline", () => {
  const text =
    "1.范围；2.系统分级与配置基线；3.测试环境；4.功能测试；5.性能/容量测试；" +
    "6.网络与冗余；7.数据留存；8.备份恢复；9.缺陷分级；10.交付文档";
  assert.deepEqual(parseOutlineFromText(text), [
    "范围",
    "系统分级与配置基线",
    "测试环境",
    "功能测试",
    "性能/容量测试",
    "网络与冗余",
    "数据留存",
    "备份恢复",
    "缺陷分级",
    "交付文档",
  ]);
});

test("parses newline separated outline and strips trailing punctuation", () => {
  const text = "1. 适用范围。\n2. 验证对象与构型。\n3. 试验前提。";
  assert.deepEqual(parseOutlineFromText(text), [
    "适用范围",
    "验证对象与构型",
    "试验前提",
  ]);
});

test("supports 、．）number separators", () => {
  assert.deepEqual(parseOutlineFromText("1、范围；2、系统分级"), ["范围", "系统分级"]);
  assert.deepEqual(parseOutlineFromText("1．范围；2．测试环境"), ["范围", "测试环境"]);
  assert.deepEqual(parseOutlineFromText("1）范围；2）测试环境"), ["范围", "测试环境"]);
});

test("ignores nested numbering and unnumbered prose", () => {
  const text =
    "本文件适用于中高风险无人直升机。\n" +
    "1.范围；2.系统分级；2.1 子项一\n" +
    "3.测试环境";
  assert.deepEqual(parseOutlineFromText(text), ["范围", "系统分级", "测试环境"]);
});

test("keeps digit-led real titles like year lists", () => {
  const text = "1.适用范围；2.2024年适航标准清单；3.测试环境";
  assert.deepEqual(parseOutlineFromText(text), [
    "适用范围",
    "2024年适航标准清单",
    "测试环境",
  ]);
});

test("returns empty for non-outline text and invalid input", () => {
  assert.deepEqual(parseOutlineFromText("只有一段说明文字，没有编号章节。"), []);
  assert.deepEqual(parseOutlineFromText(""), []);
  assert.deepEqual(parseOutlineFromText(null), []);
  assert.deepEqual(parseOutlineFromText(undefined), []);
});

test("deduplicates repeated titles keeping first occurrence order", () => {
  assert.deepEqual(parseOutlineFromText("1.范围；2.范围；3.测试环境"), ["范围", "测试环境"]);
});

test("splits dash and paren explanations as initial guidance", () => {
  assert.deepEqual(splitTitleAndGuidance("范围——界定本文档适用范围"), {
    title: "范围",
    guidance: "界定本文档适用范围",
  });
  assert.deepEqual(splitTitleAndGuidance("测试环境（说明测试场地与设备）"), {
    title: "测试环境",
    guidance: "说明测试场地与设备",
  });
  assert.deepEqual(splitTitleAndGuidance("通过判据"), { title: "通过判据" });
});

test("suggests guidance for the user's real ten-section structure", () => {
  const titles = [
    "范围",
    "系统分级与配置基线",
    "测试环境",
    "功能测试",
    "性能/容量测试",
    "网络与冗余",
    "数据留存",
    "备份恢复",
    "缺陷分级",
    "交付文档",
  ];
  const guidances = titles.map(suggestGuidance);
  assert.ok(guidances.every((g) => typeof g === "string" && g.length > 10), JSON.stringify(guidances));
  assert.match(guidances[0], /范围/);
  assert.match(guidances[3], /功能/);
  assert.match(guidances[4], /性能/);
  assert.match(guidances[8], /缺陷/);
  assert.match(guidances[9], /交付/);
});

test("parseOutlineSections attaches suggested guidance per section", () => {
  const text =
    "1.范围；2.系统分级与配置基线；3.测试环境；4.功能测试；5.性能/容量测试；" +
    "6.网络与冗余；7.数据留存；8.备份恢复；9.缺陷分级；10.交付文档";
  const sections = parseOutlineSections(text);
  assert.equal(sections.length, 10);
  assert.equal(sections[0].title, "范围");
  assert.ok(sections[0].guidance && sections[0].guidance.includes("范围"));
  assert.equal(sections[9].title, "交付文档");
  assert.ok(sections[9].guidance && sections[9].guidance.includes("交付"));
});

test("parseOutlineSections prefers in-file explanation over suggestion", () => {
  const sections = parseOutlineSections("1.范围——本文档适用于中高风险无人直升机");
  assert.equal(sections.length, 1);
  assert.deepEqual(sections[0], {
    title: "范围",
    guidance: "本文档适用于中高风险无人直升机",
  });
});

test("suggestGuidance returns null for unrecognized themes", () => {
  assert.equal(suggestGuidance("稳定下降速度"), null);
  assert.equal(suggestGuidance(""), null);
});
