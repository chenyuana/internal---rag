"""Recover a physical page from the ingestion publisher's source header only."""

import re


def published_page_number(text: str) -> int | None:
    header = re.split(r"\r?\n\s*\r?\n", text.strip(), maxsplit=1)[0][:2000]
    if not re.match(r"文档[：:]", header):
        return None
    match = re.search(
        r"(?:^|\s)页码[：:]\s*([1-9]\d{0,5})(?:\s*[-–—]\s*([1-9]\d{0,5}))?(?=\s|$)",
        header,
    )
    if not match:
        return None
    start = int(match[1])
    if match[2] and int(match[2]) < start:
        return None
    return start
