"use strict";

/* 从结构文件文本中解析编号章节列表并给出编制要求建议（纯函数，浏览器/Node 通用）。
 *
 * 支持输入形如：
 *   "1.范围；2.系统分级与配置基线；3.测试环境；…"（一行、分号分隔）
 *   "1.范围\n2.系统分级与配置基线\n3.测试环境"（换行分隔）
 *   "1、范围；2、系统分级与配置基线"（顿号/全角点/右括号编号）
 *   "1.范围——界定本文档适用范围；2.测试环境（说明测试场地与设备）"
 *     （破折号/括号内的说明作为该章初始“编制要求”）
 *
 * 识别规则：
 * - 按 分号(全/半角)、换行 分段；
 * - 每段以 阿拉伯数字 + [.、．)）] 开头视为编号章节，其余文字为章节标题；
 * - 标题去首尾空白与尾部句号；
 * - 丢弃“1.5 小节”这类嵌套编号段（标题以数字开头），避免误判；
 * - 结果去重保序。
 *
 * 编制要求建议（suggestGuidance）：
 * - 标题中的“——说明”“（说明）”优先作为初始编制要求；
 * - 否则按标题关键词匹配主题模板（与预设大纲的编制要求粒度对齐）；
 * - 仍未命中返回 null，由调用方填入默认编制要求。
 */

(function exposeOutlineParser(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.OutlineParser = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const SEGMENT_SPLIT = /[；;\n]+/;
  const NUMBERED_SEGMENT = /^\s*\d+[\.、．)）]\s*(.+?)\s*$/;
  // 嵌套/子项编号（“1.5 小节”“2 子项”）开头为 数字+分隔符或空格，整体丢弃；
  // 数字开头的真实标题（如“2024年清单”）不受影响。
  const NESTED_NUMBER = /^\d+[\.、．)）\s]/;
  const TRAILING_PUNCT = /[。．\s]+$/;
  const DASH_SPLIT = /——|—/;
  const PAREN_TAIL = /^(.+?)[（(]([^（）()]*)[）)]\s*$/;

  // 标题关键词 → 编制要求 的确定性映射（按声明顺序优先匹配）。
  const GUIDANCE_RULES = [
    { pattern: /范围|适用|边界|对象/, text: "界定适用范围、适用对象与边界条件；明确不适用或例外情况；依据知识库中的适用法规、标准与内部文件，保留引用。" },
    { pattern: /术语|定义|缩略/, text: "列出本节使用的主要术语、缩略语及其定义；引用知识库中的标准定义，避免自造术语。" },
    { pattern: /依据|标准|规范|条款|文件/, text: "列出适用的法规、标准、规范与内部文件，说明其适用关系、版本与引用条款，保留引用。" },
    { pattern: /基线|配置|构型|版本/, text: "说明系统分级与配置基线，界定构型、软硬件版本、配置项及其变更控制要求，保留引用。" },
    { pattern: /环境|场地|设备|仪器|工装|条件|资源/, text: "明确测试/运行环境的场地、设备、仪器、工装、软件版本、环境条件（温度、湿度、供电等）及校准要求。" },
    { pattern: /性能|容量|吞吐|并发|响应时间/, text: "说明性能/容量指标、测试方法、数据采集、计算与判定准则；数值与判据保留引用。" },
    { pattern: /功能/, text: "按功能项说明测试目的、输入、步骤、预期结果与判定；覆盖正常、异常与边界场景，数值与判据保留引用。" },
    { pattern: /网络|冗余|链路|切换|拓扑|通信/, text: "说明网络拓扑、通信链路、冗余机制与故障切换的验证要求；覆盖异常与失效场景，保留引用。" },
    { pattern: /数据|留存|存储|记录/, text: "规定数据采集、存储、留存周期、格式与可追溯性要求；引用知识库中的规定。" },
    { pattern: /备份|恢复|RTO|RPO|灾备/, text: "说明备份策略、恢复流程、RTO/RPO 目标与验证方法；覆盖异常与灾备场景，保留引用。" },
    { pattern: /缺陷|问题|不符合|故障/, text: "规定缺陷/问题的分级标准、判定准则、处理流程与关闭条件；引用知识库中的定义。" },
    { pattern: /交付|文档|手册|清单|报告/, text: "规定交付物清单、文档内容与格式要求；引用知识库中的规定。" },
    { pattern: /测试|试验|验证|检查|方法/, text: "说明测试/试验项目的目的、条件、方法、步骤、数据记录与通过判据；每个事实结论保留引用。" },
    { pattern: /安全|风险|应急|停止/, text: "识别风险与停止条件，说明应急处置、隔离措施和责任分工；引用知识库中的规定。" },
    { pattern: /结论|待补|缺口|计划|建议/, text: "汇总覆盖情况、证据缺口、遗留问题与后续计划；如实列出待补信息。" },
  ];

  function parseOutlineFromText(text) {
    if (!text || typeof text !== "string") return [];
    const titles = [];
    for (const segment of text.split(SEGMENT_SPLIT)) {
      const match = segment.match(NUMBERED_SEGMENT);
      if (!match) continue;
      let title = match[1].replace(TRAILING_PUNCT, "").trim();
      if (!title || NESTED_NUMBER.test(title)) continue;
      if (!titles.includes(title)) titles.push(title);
    }
    return titles;
  }

  // 从“标题——说明”“标题（说明）”中拆出初始编制要求；无说明返回 { title }。
  function splitTitleAndGuidance(title) {
    const clean = String(title || "").trim();
    if (!clean) return { title: clean };
    const dash = clean.search(DASH_SPLIT);
    if (dash >= 0) {
      const head = clean.slice(0, dash).trim();
      const tail = clean.slice(dash).replace(/^[——]+/, "").trim();
      if (head && tail) return { title: head, guidance: tail };
    }
    const paren = clean.match(PAREN_TAIL);
    if (paren && paren[1].trim() && paren[2].trim()) {
      return { title: paren[1].trim(), guidance: paren[2].trim() };
    }
    return { title: clean };
  }

  // 按标题关键词给出该主题的编制要求骨架；未命中返回 null。
  function suggestGuidance(title) {
    const clean = String(title || "").trim();
    if (!clean) return null;
    for (const rule of GUIDANCE_RULES) {
      if (rule.pattern.test(clean)) return rule.text;
    }
    return null;
  }

  // 解析结构文件为章节列表，每章附初始编制要求（说明优先，其次关键词映射，可空）。
  function parseOutlineSections(text) {
    if (!text || typeof text !== "string") return [];
    const sections = [];
    for (const segment of text.split(SEGMENT_SPLIT)) {
      const match = segment.match(NUMBERED_SEGMENT);
      if (!match) continue;
      let raw = match[1].replace(TRAILING_PUNCT, "").trim();
      if (!raw || NESTED_NUMBER.test(raw)) continue;
      const { title, guidance } = splitTitleAndGuidance(raw);
      if (!title || sections.some((item) => item.title === title)) continue;
      sections.push({ title, guidance: guidance || suggestGuidance(title) });
    }
    return sections;
  }

  return {
    parseOutlineFromText,
    parseOutlineSections,
    splitTitleAndGuidance,
    suggestGuidance,
  };
});
