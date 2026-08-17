"""交互式文档撰写辅助脚本（试点）。

不修改任何现有代码。直接在终端运行：

    python experiments/draft_section.py

流程：
1. 选择文档类型（审定计划 / 试验计划 / 技术报告）
2. 选择要撰写的章节
3. 输入机型/产品关键词（可选）
4. 系统从 RAGFlow 知识库检索相关素材
5. 调用本地 Ollama 模型生成该节初稿
6. 展示初稿 + 引用来源，供人工审核

依赖：
- RAGFlow 服务运行在 9380（Docker Desktop 启动）
- Ollama 服务运行在 11434，模型 qwen3.5:9b-q4_K_M
- 知识库 dataset_id 已创建
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_DATASET_ID = "8837f5068b2a11f1b7c4d904b160df50"

RAGFLOW_HEALTH = "/api/v1/system/healthz"
RAGFLOW_RETRIEVAL = "api/v1/retrieval"
OLLAMA_CHAT = "/api/chat"

# 文档类型 -> 章节模板骨架
DOC_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "审定计划": [
        {"key": "1", "title": "范围", "prompt_hint": "明确本文件的适用范围和目的。"},
        {
            "key": "2",
            "title": "引用文件",
            "prompt_hint": "列出适用的规章、咨询通告、行业标准和内部文件。",
        },
        {"key": "3", "title": "符号及缩略语", "prompt_hint": "定义文中用到的符号、缩略语和术语。"},
        {
            "key": "4",
            "title": "产品描述",
            "prompt_hint": "描述产品总体情况，包括布局、系统、动力等。",
        },
        {
            "key": "5",
            "title": "预期运行的规章依据",
            "prompt_hint": "说明产品预期遵循的适航规章和运行规章。",
        },
        {
            "key": "6",
            "title": "符合性方法",
            "prompt_hint": "说明如何验证各项要求，引用适用的符合性方法。",
        },
        {"key": "7", "title": "符合性详细说明", "prompt_hint": "逐条说明与规章条款的符合性。"},
        {"key": "8", "title": "试验计划", "prompt_hint": "列出拟进行的试验项目和安排。"},
    ],
    "试验计划": [
        {"key": "1", "title": "范围", "prompt_hint": "明确试验目的、对象和适用范围。"},
        {"key": "2", "title": "引用文件", "prompt_hint": "列出试验依据的规章、标准和方法。"},
        {"key": "3", "title": "试验项目", "prompt_hint": "逐项列出试验内容、条件和判据。"},
        {"key": "4", "title": "试验条件与设备", "prompt_hint": "说明试验环境、设备、场地要求。"},
        {"key": "5", "title": "试验程序", "prompt_hint": "描述每项试验的步骤、数据采集和处理。"},
        {"key": "6", "title": "安全与风险", "prompt_hint": "说明试验安全措施和风险控制。"},
    ],
    "技术报告": [
        {"key": "1", "title": "概述", "prompt_hint": "说明报告目的、背景和范围。"},
        {"key": "2", "title": "依据文件", "prompt_hint": "列出报告依据的技术文件和标准。"},
        {"key": "3", "title": "主要内容", "prompt_hint": "阐述技术分析、计算、试验或评估结果。"},
        {"key": "4", "title": "结论与建议", "prompt_hint": "给出结论和后续工作建议。"},
    ],
}

MODEL = "qwen3.5:9b-q4_K_M"


def load_ragflow_config() -> dict[str, str]:
    """Load RAGFlow base_url and api_key from .env."""
    load_dotenv(ENV_PATH)
    return {
        "base_url": os.environ.get("RAGFLOW_BASE_URL", "http://127.0.0.1:9380"),
        "api_key": os.environ.get("RAGFLOW_API_KEY", ""),
    }


def check_services() -> tuple[dict[str, bool], dict[str, str]]:
    """Probe RAGFlow and Ollama health; return availability map + errors."""
    status: dict[str, bool] = {}
    errors: dict[str, str] = {}
    ragflow = load_ragflow_config()
    headers = {"Authorization": f"Bearer {ragflow['api_key']}"} if ragflow["api_key"] else {}

    try:
        resp = httpx.get(
            ragflow["base_url"].rstrip("/") + RAGFLOW_HEALTH,
            headers=headers,
            timeout=5,
        )
        status["ragflow"] = resp.status_code == 200
        if not status["ragflow"]:
            errors["ragflow"] = f"HTTP {resp.status_code}"
    except Exception as exc:
        status["ragflow"] = False
        errors["ragflow"] = type(exc).__name__

    try:
        resp = httpx.get("http://127.0.0.1:11434/api/version", timeout=5)
        status["ollama"] = resp.status_code == 200
        if not status["ollama"]:
            errors["ollama"] = f"HTTP {resp.status_code}"
    except Exception as exc:
        status["ollama"] = False
        errors["ollama"] = type(exc).__name__

    return status, errors


def retrieve(
    query: str,
    *,
    dataset_id: str,
    base_url: str,
    api_key: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Call RAGFlow retrieval and return normalized chunks."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "question": query,
        "dataset_ids": [dataset_id],
        "document_ids": [],
        "page": 1,
        "page_size": top_k,
        "similarity_threshold": 0.2,
        "vector_similarity_weight": 0.3,
        "top_k": 1024,
        "keyword": False,
        "highlight": False,
    }
    try:
        resp = httpx.post(
            base_url.rstrip("/") + "/" + RAGFLOW_RETRIEVAL,
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]

    if not isinstance(body, dict):
        return []
    code = body.get("code", 0)
    if code not in (0, None):
        return [{"error": f"RAGFlow code={code} message={body.get('message')}"}]
    data = body.get("data", body)
    raw_chunks = data.get("chunks", []) if isinstance(data, dict) else []
    chunks: list[dict[str, Any]] = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            continue
        text = (
            item.get("content")
            or item.get("content_with_weight")
            or item.get("text")
        )
        if not isinstance(text, str) or not text.strip():
            continue
        metadata = item.get("document_metadata") or item.get("metadata") or {}
        document_name = (
            metadata.get("document_name")
            or item.get("document_name")
            or item.get("docnm_kwd")
        )
        chunks.append(
            {
                "chunk_id": item.get("id") or item.get("chunk_id"),
                "document_id": item.get("document_id") or item.get("doc_id"),
                "document_name": document_name,
                "text": text.strip(),
                "score": (
                    item.get("similarity")
                    or item.get("vector_similarity")
                    or 0.0
                ),
            }
        )
    return chunks


def generate(
    prompt: str,
    *,
    model: str = MODEL,
    max_tokens: int = 2000,
) -> str:
    """Call Ollama chat completion."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": max_tokens,
        },
    }
    try:
        resp = httpx.post(
            "http://127.0.0.1:11434" + OLLAMA_CHAT,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        return f"[生成失败: {type(exc).__name__}: {exc}]"
    content = body.get("message", {}).get("content", "")
    return content.strip() or "[模型未返回内容]"


def build_prompt(
    doc_type: str,
    section: dict[str, str],
    subject: str,
    chunks: list[dict[str, Any]],
) -> str:
    """Assemble the generation prompt from template + retrieved context."""
    lines: list[str] = []
    lines.append("你是一名航空领域的技术文档撰写专家。")
    lines.append(f"请撰写《{doc_type}》中的第 {section['key']} 节：{section['title']}。")
    lines.append("")
    if subject:
        lines.append(f"本文件针对的产品/主题：{subject}")
        lines.append("")
    lines.append(f"写作要求：{section['prompt_hint']}")
    lines.append("")
    lines.append("请参考以下从知识库检索到的资料（仅作依据，不要照抄）：")
    lines.append("")
    for i, chunk in enumerate(chunks, start=1):
        doc_name = chunk.get("document_name") or "未知文档"
        text = str(chunk.get("text", ""))[:600]
        lines.append(f"[资料{i}] 来源：{doc_name}")
        lines.append(text)
        lines.append("")
    lines.append("请基于以上资料，撰写该节的完整内容。要求：")
    lines.append("1. 结构清晰，语言规范，符合民航/航空技术文档风格；")
    lines.append("2. 引用资料时标注来源编号，如 [资料1]；")
    lines.append("3. 若资料不足，可基于专业知识补充，但需保持谨慎；")
    lines.append("4. 使用规范的编号和小标题。")
    return "\n".join(lines)


def select_interactive(prompt: str, options: list[str]) -> str:
    """Interactive selection from a list."""
    print(prompt)
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    while True:
        raw = input("  输入编号: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("  无效输入，请重新输入。")


def main() -> int:
    print("=" * 60)
    print("  文档撰写辅助（试点）")
    print("=" * 60)
    print()

    # 1. 检查服务
    status, errors = check_services()
    print("服务状态：")
    print(f"  RAGFlow: {'可用' if status.get('ragflow') else '不可用'} {errors.get('ragflow', '')}")
    print(f"  Ollama : {'可用' if status.get('ollama') else '不可用'} {errors.get('ollama', '')}")
    print()
    if not status.get("ragflow"):
        print("警告：RAGFlow 不可用，检索功能将不可用。请先启动 Docker Desktop 和 RAGFlow。")
    if not status.get("ollama"):
        print("警告：Ollama 不可用，生成功能将不可用。请先启动 Ollama。")
    print()

    # 2. 选择文档类型
    doc_type = select_interactive("选择文档类型：", list(DOC_TEMPLATES.keys()))
    sections = DOC_TEMPLATES[doc_type]
    section_titles = [f"{s['key']}. {s['title']}" for s in sections]
    section_title = select_interactive("选择要撰写的章节：", section_titles)
    section_idx = section_titles.index(section_title)
    section = sections[section_idx]

    # 3. 输入主题
    subject = input("输入产品/主题关键词（直接回车跳过）: ").strip()

    print()
    print(f"文档类型：{doc_type}")
    print(f"章节：第 {section['key']} 节 {section['title']}")
    print(f"主题：{subject or '（未指定）'}")
    print()

    # 4. 检索
    ragflow = load_ragflow_config()
    query = f"{subject} {section['title']} {doc_type}".strip()
    print(f"正在从知识库检索：{query}")
    chunks = retrieve(
        query,
        dataset_id=DEFAULT_DATASET_ID,
        base_url=ragflow["base_url"],
        api_key=ragflow["api_key"],
    )
    if chunks and "error" in chunks[0]:
        print(f"  检索失败：{chunks[0]['error']}")
        chunks = []
    else:
        print(f"  检索到 {len(chunks)} 条相关资料")
        for i, c in enumerate(chunks[:5], start=1):
            doc = c.get("document_name") or "?"
            print(f"    [{i}] {doc}: {str(c.get('text',''))[:50]}...")
    print()

    # 5. 生成
    if status.get("ollama"):
        print(f"正在用 {MODEL} 生成……")
        prompt = build_prompt(doc_type, section, subject, chunks)
        draft = generate(prompt)
        print()
        print("=" * 60)
        print("  生成初稿")
        print("=" * 60)
        print(draft)
        print()
        print("=" * 60)
        print("  引用来源")
        print("=" * 60)
        for i, c in enumerate(chunks, start=1):
            doc = c.get("document_name") or "?"
            print(f"  [资料{i}] {doc} (score={c.get('score', 0):.3f})")
            print(f"          {str(c.get('text',''))[:100]}...")
        print()
    else:
        print("Ollama 不可用，跳过生成。")
        print("检索到的资料：")
        for i, c in enumerate(chunks, start=1):
            doc = c.get("document_name") or "?"
            print(f"  [{i}] {doc}: {str(c.get('text',''))[:80]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
