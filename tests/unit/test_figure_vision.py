from __future__ import annotations

import json

import httpx

from app.core.config import FigureVisionSettings
from app.ingestion.parsers.figure_vision import FigureVisionClient


def test_figure_vlm_uses_local_ollama_without_thinking() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"version": "test"})
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "title": "P1 和 P2 值",
                            "chart_type": "曲线图",
                            "axes": ["横轴：P1/P2", "纵轴：高度"],
                            "series": ["P1", "P2"],
                            "observations": ["图中包含两条曲线"],
                            "uncertainties": ["部分刻度较小"],
                        },
                        ensure_ascii=False,
                    )
                }
            },
        )

    client = FigureVisionClient(
        FigureVisionSettings(
            enabled=True,
            base_url="http://ollama.local",
            model_name="qwen3.5:9b-q4_K_M",
        ),
        transport=httpx.MockTransport(handler),
    )

    ready, detail = client.probe()
    description = client.describe(
        b"png-image",
        page_number=236,
        caption="图 1 P1 和 P2 值",
    )

    assert ready is True
    assert detail is None
    assert requests[0]["think"] is False
    assert requests[0]["stream"] is False
    assert requests[0]["format"] == {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "chart_type": {"type": "string"},
            "axes": {"type": "array", "items": {"type": "string"}},
            "series": {"type": "array", "items": {"type": "string"}},
            "observations": {"type": "array", "items": {"type": "string"}},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "title",
            "chart_type",
            "axes",
            "series",
            "observations",
            "uncertainties",
        ],
        "additionalProperties": False,
    }
    message = requests[0]["messages"]
    assert isinstance(message, list)
    assert message[0]["images"]
    assert "图题：P1 和 P2 值" in description
    assert "无法可靠读取：部分刻度较小" in description


def test_figure_vlm_retries_one_empty_response() -> None:
    attempt = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return httpx.Response(200, json={"message": {"content": ""}})
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "title": "图 A2",
                            "chart_type": "曲线图",
                            "axes": [],
                            "series": [],
                            "observations": ["图表已识别"],
                            "uncertainties": [],
                        },
                        ensure_ascii=False,
                    )
                }
            },
        )

    client = FigureVisionClient(
        FigureVisionSettings(
            enabled=True,
            base_url="http://ollama.local",
            model_name="qwen3.5:9b-q4_K_M",
        ),
        transport=httpx.MockTransport(handler),
    )

    description = client.describe(
        b"png-image",
        page_number=164,
        caption="图 A2",
    )

    assert attempt == 2
    assert "图题：图 A2" in description
