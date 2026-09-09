"""Isolated browser acceptance fixture: real UI renderer + a real four-page PDF.

Run with python -m tests.citation_pdf_browser_fixture; no production state changes.
"""
from pathlib import Path
import re

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

root = Path(__file__).resolve().parents[1]
app = FastAPI()
app.mount("/assets", StaticFiles(directory=root / "app/web/static"), name="assets")


@app.get("/")
def fixture():
    html = (root / "app/web/chat.html").read_text(encoding="utf-8")
    html = re.sub(r'<script src="/assets/chat.js[^\"]*"></script>', '', html)
    return HTMLResponse(html.replace('</body>', '''
      <button id="testCitation" style="position:fixed;top:5px;left:400px;z-index:1000"
        onclick="CitationPdf.open({citation_id:'C1',document_id:'sample',dataset_id:'fixture',page_number:4,document_name:'Docket1186 - test',quote:'TABLE II / TABLE III / TABLE IV'},'fixture')">测试 C1 第四页</button>
      </body>'''))


@app.get("/api/v1/documents/document/fixture/sample/file")
def source():
    return FileResponse(
        "D:/internal-rag/data/ingestion/originals/05/"
        "052b3a89cf60e617127c390fb7619577f45cc1ab83ad2a6c403dd74086ce9b95/source.pdf",
        media_type="application/pdf",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8082)
