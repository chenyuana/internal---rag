const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const code = fs.readFileSync(path.join(__dirname, '../../app/web/static/citation_pdf.js'), 'utf8');
function setup(fetch) {
  const nodes = new Map();
  const revoked = [];
  const document = {getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, {hidden:true, listeners:{},
      addEventListener(e, fn) {this.listeners[e] = fn;},
      removeAttribute(k) {delete this[k];},
      showModal() {this.open = true;}, close() {this.open = false; this.listeners.close();}});
    return nodes.get(id);
  }};
  const rendered = [];
  const context = {window:{CitationPdfRenderer:{clear(){}, async open(blob, page) {
    rendered.push(page); return {page, total:10};
  }}}, document, Blob, AbortController, fetch,
    URL:{createObjectURL: () => 'blob:test', revokeObjectURL: url => revoked.push(url)}};
  vm.runInNewContext(code, context);
  return {open:context.window.CitationPdf.open, el:document.getElementById, revoked, rendered};
}
const pdf = () => ({ok:true, blob:async () => new Blob(['%PDF-1.4\ntest'])});
test('uses citation dataset, jumps to page, and releases on close', async () => {
  let requested;
  const ui = setup(async url => {requested = url; return pdf();});
  await ui.open({citation_id:'C1', dataset_id:'source kb', document_id:'doc', page_number:4}, 'other-kb');
  assert.match(requested, /source%20kb\/doc\/file$/);
  assert.deepEqual(ui.rendered, [4]);
  assert.match(ui.el('citationPdfStatus').textContent, /已显示原始 PDF 第 4/);
  assert.equal(ui.el('citationPdfDialog').open, true);
  ui.el('citationPdfDialog').close();
  assert.equal(ui.el('citationPdfFrame').src, 'about:blank');
  assert.deepEqual(ui.revoked, ['blob:test']);
});
test('late old response cannot overwrite a newer citation', async () => {
  let resolve;
  const ui = setup(url => url.includes('/old/') ? new Promise(r => resolve = r) : Promise.resolve(pdf()));
  const old = ui.open({citation_id:'C1', document_id:'old', page_number:1}, 'kb');
  await ui.open({citation_id:'C2', document_id:'new', page_number:9}, 'kb');
  resolve(pdf()); await old;
  assert.deepEqual(ui.rendered, [9]);
});
test('missing page, temporary evidence and invalid content are explicit', async () => {
  const ui = setup(async () => pdf());
  await ui.open({citation_id:'C1', document_id:'doc'}, 'kb');
  assert.match(ui.el('citationPdfStatus').textContent, /没有可靠页码/);
  await ui.open({citation_id:'C1', source_path:'temporary-upload'}, 'kb');
  assert.equal(ui.el('citationPdfFrame').hidden, true);
  assert.match(ui.el('citationPdfStatus').textContent, /临时辅助资料/);
  const bad = setup(async () => ({ok:true, blob:async () => new Blob(['not pdf'])}));
  await bad.open({citation_id:'C1', document_id:'doc'}, 'kb');
  assert.match(bad.el('citationPdfStatus').textContent, /不是 PDF/);
});
test('failed download is displayed without a blank viewer', async () => {
  const ui = setup(async () => ({ok:false, json:async () => ({error:{message:'PDF unavailable'}})}));
  await ui.open({citation_id:'C1', document_id:'doc'}, 'kb');
  assert.equal(ui.el('citationPdfStatus').textContent, 'PDF unavailable');
  assert.equal(ui.el('citationPdfFrame').hidden, true);
});
test('actual CCAR-25-R4 legacy C1 header locates page 123 without page_number', async () => {
  const ui = setup(async () => pdf());
  const citation = {citation_id:'C1', document_id:'ccar25', page_number:null,
    quote:'文档：CCAR-25-R4 运输类飞机适航标准.pdf\n章节：附录 B 图 3 成线性变化。龙骨处的压力按下式计算: / 第 25.981 条 燃油箱点燃防护\n条号：25.981\n页码：123-124\n\n(c) 本条(b)不适用于采用减轻燃油蒸气点燃影响措施的燃油箱。'};
  await ui.open(citation, 'kb');
  assert.deepEqual(ui.rendered, [123]);
  assert.equal(citation.page_number, null); // do not rewrite browser history
});
test('body page mentions are not used as source locations', async () => {
  const ui = setup(async () => pdf());
  await ui.open({citation_id:'C1',document_id:'doc',quote:'文档：manual.pdf\n\n正文参见 页码：123-124'}, 'kb');
  assert.deepEqual(ui.rendered, [1]);
  assert.match(ui.el('citationPdfStatus').textContent, /没有可靠页码/);
});
