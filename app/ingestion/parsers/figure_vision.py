from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from app.core.config import FigureVisionSettings

FIGURE_SCHEMA: dict[str, Any] = {
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

FIGURE_PROMPT = """你正在为航空法规知识库整理一张图表。
只描述图片中清晰可见的内容，严禁推测、插值或补充图片中没有的信息。
请识别图题、图表类型、坐标轴名称与单位、曲线或区域名称，以及无需估算即可确认的事实。
读不清的文字、数值或关系必须放入 uncertainties，不要猜测。
输出必须严格符合给定 JSON Schema，不要输出代码块或额外解释。"""


class FigureVisionClient:
    name = "figure-vlm"

    def __init__(
        self,
        settings: FigureVisionSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.settings.base_url.rstrip("/") + "/",
            timeout=self.settings.timeout_seconds,
            transport=self._transport,
            trust_env=False,
        )

    def probe(self) -> tuple[bool, str | None]:
        if not self.enabled:
            return False, "disabled"
        try:
            with self._client() as client:
                response = client.get(self.settings.health_path.lstrip("/"))
                response.raise_for_status()
            return True, None
        except httpx.HTTPError as exc:
            return False, type(exc).__name__

    @staticmethod
    def _string_list(payload: dict[str, Any], key: str) -> list[str]:
        value = payload.get(key, [])
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @classmethod
    def _format_description(cls, payload: dict[str, Any]) -> str:
        title = str(payload.get("title", "")).strip()
        chart_type = str(payload.get("chart_type", "")).strip()
        axes = cls._string_list(payload, "axes")
        series = cls._string_list(payload, "series")
        observations = cls._string_list(payload, "observations")
        uncertainties = cls._string_list(payload, "uncertainties")
        lines = ["图表辅助说明（本地 VLM 生成，发布前需复核）"]
        if title:
            lines.append(f"图题：{title}")
        if chart_type:
            lines.append(f"类型：{chart_type}")
        if axes:
            lines.append("坐标轴：" + "；".join(axes))
        if series:
            lines.append("曲线/区域：" + "；".join(series))
        if observations:
            lines.append("可确认信息：" + "；".join(observations))
        if uncertainties:
            lines.append("无法可靠读取：" + "；".join(uncertainties))
        if len(lines) == 1:
            raise ValueError("Figure VLM returned an empty structured description.")
        return "\n".join(lines)

    def describe(
        self,
        image: bytes,
        *,
        page_number: int,
        caption: str,
    ) -> str:
        if not self.enabled:
            return ""
        prompt = (
            f"{FIGURE_PROMPT}\nPDF物理页码：{page_number}。"
            f"\n邻近图题（可能为空或不完整）：{caption or '无'}"
        )
        request = {
            "model": self.settings.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image).decode("ascii")],
                }
            ],
            "stream": False,
            "think": False,
            "format": FIGURE_SCHEMA,
            "options": {
                "temperature": 0,
                "num_predict": self.settings.max_output_tokens,
            },
        }
        last_error: httpx.HTTPError | ValueError | None = None
        for _attempt in range(2):
            try:
                with self._client() as client:
                    response = client.post("api/chat", json=request)
                    response.raise_for_status()
                payload = response.json()
                message = payload.get("message", {}) if isinstance(payload, dict) else {}
                content = message.get("content", "") if isinstance(message, dict) else ""
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Figure VLM returned no message content.")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("Figure VLM returned a non-object JSON response.")
                return self._format_description(parsed)
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError("Figure VLM retry loop exited without a result.")
        raise last_error
