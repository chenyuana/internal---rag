from __future__ import annotations

import base64
import json
import mimetypes
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from app.core.config import RemoteParserSettings

MAX_REMOTE_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_REMOTE_ARCHIVE_MEMBERS = 10_000
MINERU_PAGE_BATCH_SIZE = 10


_LATEX_SYMBOLS = {
    r"\times": "×",
    r"\cdot": "·",
    r"\bullet": "·",
    r"\circ": "°",
    r"\degree": "°",
    r"\pm": "±",
    r"\div": "÷",
    r"\sim": "~",
    r"\approx": "≈",
    r"\le": "≤",
    r"\ge": "≥",
    r"\ne": "≠",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\leftrightarrow": "↔",
    r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐",
    r"\Leftrightarrow": "⇔",
    r"\to": "→",
    r"\uparrow": "↑",
    r"\downarrow": "↓",
    r"\updownarrow": "↕",
    r"\Uparrow": "⇑",
    r"\Downarrow": "⇓",
    r"\infty": "∞",
    r"\ldots": "…",
    r"\Lambda": "Λ",
    r"\Phi": "Φ",
    r"\Gamma": "Γ",
    r"\Delta": "Δ",
    r"\varDelta": "Δ",
    r"\Theta": "Θ",
    r"\Pi": "Π",
    r"\Sigma": "Σ",
    r"\Upsilon": "Υ",
    r"\Omega": "Ω",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\mu": "μ",
    r"\nu": "ν",
    r"\pi": "π",
    r"\rho": "ρ",
    r"\sigma": "σ",
    r"\tau": "τ",
    r"\omega": "ω",
    r"\theta": "θ",
    r"\xi": "ξ",
    r"\psi": "ψ",
    r"\phi": "φ",
    r"\varphi": "φ",
    r"\varepsilon": "ε",
    r"\vartheta": "θ",
    r"\varrho": "ρ",
    r"\varsigma": "ς",
    r"\chi": "χ",
    r"\zeta": "ζ",
    r"\eta": "η",
    r"\kappa": "κ",
    r"\lambda": "λ",
    r"\iota": "ι",
    r"\omicron": "ο",
    r"\upsilon": "υ",
    r"\sum": "∑",
    r"\prod": "∏",
    r"\int": "∫",
    r"\oint": "∮",
    r"\partial": "∂",
    r"\nabla": "∇",
    r"\forall": "∀",
    r"\prime": "'",
    r"\exists": "∃",
    r"\in": "∈",
    r"\notin": "∉",
    r"\subset": "⊂",
    r"\subseteq": "⊆",
    r"\cup": "∪",
    r"\cap": "∩",
    r"\geq": "≥",
    r"\leq": "≤",
    r"\neq": "≠",
    r"\equiv": "≡",
    r"\propto": "∝",
    r"\mp": "∓",
}
_GREEK_OR_SYMBOL_RE = re.compile(
    "|".join(re.escape(key) for key in sorted(_LATEX_SYMBOLS, key=len, reverse=True))
)

_PUNCT_RE = re.compile(r"^[，。；;:：、,.]$")
_STOICHIOMETRIC_RE = re.compile(r"^\d{1,3}$")

# Named math operators whose name is searchable text (not just formatting):
# \lim_{x\to0}, \max_{i}, \log_{10} … Keep the operator and its bound.
_NAMED_OPERATOR_RE = re.compile(
    r"\\(?P<op>lim|max|min|sup|inf|arg|deg|log|ln|exp|sin|cos|tan|cot|"
    r"sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|det|ker|dim|Pr)"
    r"(?![A-Za-z])"
    r"(?:\s*_\s*\{\s*(?P<arg>[^{}]*)\s*\})?"
)


def _latex_if_real(item: dict[str, Any]) -> str | None:
    """Return the raw LaTeX for an equation item, guarding against empty text
    and the string ``"None"`` that ``str(None)`` would otherwise produce."""
    raw = str(item.get("text", "") or "").strip()
    if not raw or raw.casefold() == "none":
        return None
    return raw


def _subscript(content: str) -> str:
    """Collapse a LaTeX subscript group.

    Three cases:
    * punctuation mislabeled as a subscript (``_ { : }``): drop the underscore
      and keep the punctuation bare;
    * stoichiometric numbers (``_ { 4 }`` in ``MgSO_4``): merge directly so
      the chemistry formula reads ``MgSO4`` (the subscript is plain text);
    * letter/word subscripts (``_ { p H }``, ``_ { T }``, ``_ { L E }``): keep
      the underscore so variable subscripts stay distinguishable.
    """
    collapsed = re.sub(r"\s+", "", content)
    # OCR sometimes emits a stray backslash inside a subscript group, e.g.
    # ``_ { \ Y }``; strip the lone backslash so the letter survives.
    collapsed = collapsed.replace("\\", "")
    if _PUNCT_RE.fullmatch(collapsed):
        return collapsed
    if _STOICHIOMETRIC_RE.fullmatch(collapsed):
        return collapsed
    return "_" + collapsed


def _balanced_brace_group(text: str, start: int) -> int | None:
    """Return the index of the closing brace of the group starting at ``start``."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _replace_frac(text: str) -> str:
    """Convert ``\frac { num } { den }`` to ``(num)/(den)`` with balanced braces.

    Parentheses are added around numerator and denominator so retrieval text
    keeps the original grouping: ``\frac{a+b}{c+d}`` must read ``(a+b)/(c+d)``,
    not ``a+b/c+d`` (which would mean ``a + (b/c) + d``).
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith(r"\frac", i):
            j = i + len(r"\frac")
            while j < n and text[j] in " \t":
                j += 1
            num_start = j if j < n and text[j] == "{" else None
            num_end = _balanced_brace_group(text, j) if num_start is not None else None
            if num_end is None:
                out.append(text[i])
                i += 1
                continue
            denom_start = num_end + 1
            while denom_start < n and text[denom_start] in " \t":
                denom_start += 1
            denom_end = (
                _balanced_brace_group(text, denom_start)
                if denom_start < n and text[denom_start] == "{"
                else None
            )
            if denom_end is None:
                out.append(text[i])
                i += 1
                continue
            numerator = text[num_start + 1 : num_end]
            denominator = text[denom_start + 1 : denom_end]
            # Recurse so nested \frac inside num/den are also bracketed.
            numerator = _replace_frac(numerator)
            denominator = _replace_frac(denominator)
            out.append("(" + numerator + ")/(" + denominator + ")")
            i = denom_end + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _replace_sqrt(text: str) -> str:
    """Convert ``\sqrt { ... }`` / ``\sqrt [ n ] { ... }`` to ``sqrt(...)``.

    Preserves the root index for ``\sqrt[n]{x}`` so it reads ``nth_root(x)``
    instead of dropping the index (``[3]x`` is misleading).
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith(r"\sqrt", i):
            j = i + len(r"\sqrt")
            while j < n and text[j] in " \t":
                j += 1
            # Optional `[index]` between `\sqrt` and the braced radicand.
            index: str | None = None
            if j < n and text[j] == "[":
                close = text.find("]", j + 1)
                if close != -1:
                    index = re.sub(r"\s+", "", text[j + 1 : close])
                    j = close + 1
                    while j < n and text[j] in " \t":
                        j += 1
            if j < n and text[j] == "{":
                depth = 0
                k = j
                while k < n:
                    if text[k] == "{":
                        depth += 1
                    elif text[k] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                if k < n:
                    radicand = text[j + 1 : k]
                    if index and index.isdigit() and int(index) != 2:
                        out.append(f"{index}th_root({radicand})")
                    else:
                        out.append("sqrt(" + radicand + ")")
                    i = k + 1
                    continue
        out.append(text[i])
        i += 1
    return "".join(out)


def latex_to_text(latex: str) -> str:
    """Convert a LaTeX formula fragment to readable plain text.

    MinerU emits chemistry formulas such as
    ``$( \\mathrm { N a N O _ { 3 } } ) : 3 . 0 \\ \\mathrm { g } ;$``
    whose visual content is correct but which is unusable as raw text for
    indexing or RAG retrieval. This helper strips the LaTeX syntax, removes
    the inline-math delimiters, and collapses the token spacing so the
    formula reads as ``(NaNO_3):3.0g;`` while preserving subscripts and
    units.
    """

    text = latex.replace("$", "")
    # MinerU wraps display equations with `$$\n...$$`. The embedded newline is
    # a line break, not a LaTeX command; drop it before command cleanup so a
    # trailing letter does not form a bogus command like `\nm`.
    text = text.replace(r"\n", "").replace("\n", "")
    text = re.sub(
        r"\\(?:mathrm|mathbf|mathit|text|mathcal|operatorname|rm)\s*\{([^{}]*)\}",
        lambda match: re.sub(r"\s+", "", match.group(1)),
        text,
    )
    # \boldsymbol { ... } keeps the braced content (possibly bold math).
    text = re.sub(r"\\boldsymbol\s*\{([^{}]*)\}", lambda match: match.group(1), text)
    # Accent/half-open commands (vec/hat/bar/overline/widehat/underbrace…):
    # keep the braced argument, drop the command. e.g. \vec{x} -> x,
    # \overline{x} -> x (the overline is a visual accent, not searchable text).
    text = re.sub(
        r"\\(?:vec|hat|widehat|bar|overline|underline|underbrace|"
        r"widetilde|dot|ddot|bb|mathbb|mathbf|boldsymbol)\s*\{([^{}]*)\}",
        lambda match: match.group(1),
        text,
    )
    # `\small` can wrap a brace group that itself contains `\left...\right`;
    # peel it before pair handling.
    text = re.sub(r"\\small\s*\{([^{}]*)\}", lambda match: match.group(1), text)
    # `\frac` may contain nested braces in numerator/denominator; use a
    # balanced-brace scanner so fractions survive.
    text = _replace_frac(text)
    # `\sqrt` may contain arbitrarily nested braces; use a balanced-brace
    # scanner so sqrt is preserved instead of dropped later.
    text = _replace_sqrt(text)
    text = re.sub(r"\\left\s*([\[\]()])", r"\1", text)
    text = re.sub(r"\\right\s*([\[\]()])", r"\1", text)
    # Named operators keep their name so `\lim_{x\to0}` reads "lim_x→0"
    # instead of dropping the operator and leaving a bare subscript. Runs
    # before generic subscript handling so the bound is not consumed first.
    text = _NAMED_OPERATOR_RE.sub(
        lambda match: match.group("op") + (
            "(" + match.group("arg") + ")"
            if match.group("arg")
            else ""
        ),
        text,
    )
    # Preserve the symbol mapping (∑, ∏, √…) before generic command removal.
    text = _GREEK_OR_SYMBOL_RE.sub(
        lambda match: _LATEX_SYMBOLS[match.group(0)], text
    )
    text = re.sub(r"_\s*\{([^{}]*)\}", lambda m: _subscript(m.group(1)), text)
    text = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: "^" + re.sub(r"\s+", "", m.group(1)), text)
    # `\sqrt` may be written with a space before the brace: `\sqrt { ... }`.
    text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    text = text.replace("~", " ")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\s+([,，。；;:：!?！？、)])", r"\1", text)
    # OCR sometimes emits a duplicated superscript prime (``^'^'``) from a
    # nested bold group; collapse consecutive superscript primes to one.
    text = re.sub(r"\^'(?:\^')+", "'", text)
    # A display equation's right-aligned number (1)/(2) is often OCR'd as a
    # trailing "·1"/"·2" glued to the formula body (e.g. "WERI=Y_T/Y_CK·1").
    # Strip a trailing "·<1-2 digits>" when the formula has no other interpunct
    # (a chemical formula like "MgSO4·7H2O" keeps its middle dot; its text ends
    # in a letter/unit, not a bare "·N").
    if text.count("·") == 1:
        text = re.sub(r"·\d{1,2}$", "", text)
    return text.strip()


def clean_inline_latex(text: str) -> str:
    """Clean ``$...$`` inline-math fragments embedded in plain text items.

    MinerU frequently labels formula fragments as ``type=text`` (they are
    detected as inline math inside a sentence) rather than as dedicated
    ``equation`` blocks. This strips the delimiters and LaTeX syntax of each
    ``$...$`` span while leaving the surrounding prose untouched. Display
    equations wrapped as ``$$\\n...$$`` (with an embedded newline) are handled
    before single-line spans.
    """

    def replace(match: re.Match[str]) -> str:
        return latex_to_text(match.group(1))

    # Display equations: MinerU emits `$$\n...$$` where `\n` is a literal
    # backslash-n in the content list (not a real newline). Match either that
    # literal sequence or an actual newline, spanning until the closing `$$`.
    text = re.sub(r"\$\$\\n([^$]+?)\$\$", replace, text)
    text = re.sub(r"\$\$\n?([^$]+?)\$\$", replace, text)
    return re.sub(r"\$([^$\n]+)\$", replace, text)


@dataclass(slots=True)
class RemoteContentBlock:
    page_number: int
    block_type: str
    text: str
    table_html: str | None = None
    caption: str = ""
    bbox: list[float] | None = None
    image_filename: str | None = None
    image_content: bytes | None = None
    image_mime_type: str | None = None
    # Raw LaTeX for equation blocks. Kept alongside the cleaned text so the
    # readable form drives retrieval while the LaTeX is preserved for
    # faithful rendering / review as chunk metadata.
    latex: str | None = None


@dataclass(slots=True)
class RemoteParseResult:
    parser_name: str
    page_texts: dict[int, str] = field(default_factory=dict)
    page_blocks: dict[int, list[RemoteContentBlock]] = field(default_factory=dict)
    document_text: str = ""
    warnings: list[str] = field(default_factory=list)


class RemoteParserClient:
    name: str

    def __init__(
        self,
        settings: RemoteParserSettings,
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
        if not self.settings.base_url:
            return False, "base_url is empty"
        try:
            with self._client() as client:
                response = client.get(self.settings.health_path.lstrip("/"))
                response.raise_for_status()
            return True, None
        except httpx.HTTPError as exc:
            return False, type(exc).__name__

    def parse(self, path: Path) -> RemoteParseResult:
        raise NotImplementedError


class MinerUClient(RemoteParserClient):
    name = "mineru"

    @staticmethod
    def _block_type(item: dict[str, Any]) -> str:
        item_type = str(item.get("type", "")).casefold()
        return {
            "table": "table",
            "image": "figure",
            "equation": "formula",
            "interline_equation": "formula",
            "inline_equation": "formula",
            "code": "code",
            "list": "list",
        }.get(item_type, "paragraph")

    @staticmethod
    def _item_text(item: dict[str, Any]) -> str:
        item_type = str(item.get("type", "")).casefold()
        if item_type in {"header", "footer", "page_number", "discarded"}:
            return ""
        if item_type == "table":
            parts = [
                clean_inline_latex(str(item.get("table_body", ""))),
                *[str(value) for value in item.get("table_caption", [])],
                *[str(value) for value in item.get("table_footnote", [])],
            ]
            return "\n".join(part for part in parts if part.strip()).strip()
        if item_type == "image":
            parts = [
                *[str(value) for value in item.get("image_caption", [])],
                *[str(value) for value in item.get("image_footnote", [])],
                str(item.get("vlm_description", "")),
            ]
            return "\n".join(part for part in parts if part.strip()).strip()
        if item_type == "code":
            parts = [
                str(item.get("code_body", "")),
                *[str(value) for value in item.get("code_caption", [])],
            ]
            return "\n".join(part for part in parts if part.strip()).strip()
        if item_type == "list":
            return "\n".join(str(value) for value in item.get("list_items", [])).strip()
        raw = str(item.get("text", "")).strip()
        if item_type in {"equation", "interline_equation", "inline_equation"}:
            return latex_to_text(raw)
        return clean_inline_latex(raw)

    @staticmethod
    def _caption(item: dict[str, Any]) -> str:
        values: list[str] = []
        for key in ("table_caption", "image_caption", "code_caption"):
            raw = item.get(key, [])
            if isinstance(raw, list):
                values.extend(str(value).strip() for value in raw if str(value).strip())
            elif isinstance(raw, str) and raw.strip():
                values.append(raw.strip())
        return "\n".join(dict.fromkeys(values))

    @staticmethod
    def _bbox(item: dict[str, Any]) -> list[float] | None:
        raw = item.get("bbox")
        if not isinstance(raw, list) or len(raw) != 4:
            return None
        try:
            return [round(float(value), 2) for value in raw]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _archive_asset(
        archive: zipfile.ZipFile,
        *,
        json_name: str,
        item: dict[str, Any],
    ) -> tuple[str | None, bytes | None, str | None]:
        raw_path = item.get("img_path") or item.get("image_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None, None, None
        normalized = raw_path.replace("\\", "/").lstrip("./")
        parent = PurePosixPath(json_name).parent
        candidates = [
            str(parent / normalized),
            normalized,
        ]
        members = set(archive.namelist())
        resolved = next((name for name in candidates if name in members), None)
        if resolved is None:
            basename = PurePosixPath(normalized).name
            resolved = next(
                (name for name in members if PurePosixPath(name).name == basename),
                None,
            )
        if resolved is None:
            return None, None, None
        mime_type = mimetypes.guess_type(resolved)[0] or "application/octet-stream"
        return PurePosixPath(resolved).name, archive.read(resolved), mime_type

    @classmethod
    def _read_archive(cls, content: bytes) -> RemoteParseResult:
        page_parts: dict[int, list[str]] = {}
        page_blocks: dict[int, list[RemoteContentBlock]] = {}
        markdown = ""
        with zipfile.ZipFile(BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > MAX_REMOTE_ARCHIVE_MEMBERS:
                raise ValueError("MinerU archive contains too many files.")
            total_size = sum(member.file_size for member in members)
            if total_size > MAX_REMOTE_ARCHIVE_BYTES:
                raise ValueError("MinerU archive exceeds the extraction safety limit.")

            json_names = [
                member.filename
                for member in members
                if member.filename.endswith(("_content_list.json", "/content_list.json"))
            ]
            for name in json_names:
                payload = json.loads(archive.read(name))
                if not isinstance(payload, list):
                    continue
                for raw_item in payload:
                    if not isinstance(raw_item, dict):
                        continue
                    page_idx = raw_item.get("page_idx")
                    if not isinstance(page_idx, int) or page_idx < 0:
                        continue
                    text = cls._item_text(raw_item)
                    if text:
                        page_number = page_idx + 1
                        page_parts.setdefault(page_number, []).append(text)
                        item_type = str(raw_item.get("type", "")).casefold()
                        image_filename, image_content, image_mime_type = (
                            cls._archive_asset(
                                archive,
                                json_name=name,
                                item=raw_item,
                            )
                        )
                        page_blocks.setdefault(page_number, []).append(
                            RemoteContentBlock(
                                page_number=page_number,
                                block_type=cls._block_type(raw_item),
                                text=text,
                                table_html=(
                                    clean_inline_latex(
                                        str(raw_item.get("table_body", "")).strip()
                                    )
                                    if item_type == "table"
                                    and "<table" in str(raw_item.get("table_body", "")).casefold()
                                    else None
                                ),
                                caption=cls._caption(raw_item),
                                bbox=cls._bbox(raw_item),
                                image_filename=image_filename,
                                image_content=image_content,
                                image_mime_type=image_mime_type,
                                latex=(
                                    _latex_if_real(raw_item)
                                    if item_type
                                    in {"equation", "interline_equation", "inline_equation"}
                                    else None
                                ),
                            )
                        )
                if page_parts:
                    break

            markdown_names = [
                member.filename for member in members if member.filename.endswith(".md")
            ]
            if markdown_names:
                markdown = archive.read(markdown_names[0]).decode("utf-8", errors="replace")

        return RemoteParseResult(
            parser_name=cls.name,
            page_texts={
                page_number: "\n\n".join(parts)
                for page_number, parts in sorted(page_parts.items())
            },
            page_blocks=page_blocks,
            document_text=markdown.strip(),
            warnings=[] if page_parts else ["MinerU result has no page-indexed content list."],
        )

    def _parse_request(
        self,
        path: Path,
        *,
        start_page: int | None = None,
        end_page: int | None = None,
        parse_method: str | None = None,
    ) -> RemoteParseResult:
        start_page_id = max(0, (start_page or 1) - 1)
        end_page_id = max(start_page_id, (end_page or 100_000) - 1)
        data = {
            "output_dir": "./output",
            "lang_list": "ch",
            "backend": self.settings.backend,
            "parse_method": parse_method or self.settings.parse_method,
            "formula_enable": "true",
            "table_enable": "true",
            "return_md": "true",
            "return_middle_json": "true",
            "return_model_output": "true",
            "return_content_list": "true",
            "return_images": "true",
            "response_format_zip": "true",
            "start_page_id": str(start_page_id),
            "end_page_id": str(end_page_id),
        }
        page_label = (
            f"{start_page}-{end_page}"
            if start_page is not None and end_page is not None
            else "all"
        )
        try:
            with path.open("rb") as source, self._client() as client:
                response = client.post(
                    "file_parse",
                    data=data,
                    files={"files": (path.name, source, "application/pdf")},
                    headers={"Accept": "application/zip"},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"MinerU request failed for pages {page_label}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        content_type = response.headers.get("content-type", "")
        if "zip" not in content_type.casefold() and not response.content.startswith(b"PK"):
            raise RuntimeError(
                f"MinerU returned a non-ZIP response for pages {page_label}: "
                f"content_type={content_type or 'unknown'}, bytes={len(response.content)}"
            )
        try:
            return self._read_archive(response.content)
        except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
            raise RuntimeError(
                f"MinerU ZIP read failed for pages {page_label}: "
                f"{type(exc).__name__}: {exc}; bytes={len(response.content)}"
            ) from exc

    @staticmethod
    def _page_batches(
        page_numbers: list[int],
        *,
        batch_size: int,
    ) -> list[list[int]]:
        batches: list[list[int]] = []
        for page_number in sorted(set(page_numbers)):
            if page_number < 1:
                continue
            if (
                not batches
                or len(batches[-1]) >= batch_size
                or page_number != batches[-1][-1] + 1
            ):
                batches.append([page_number])
            else:
                batches[-1].append(page_number)
        return batches

    @staticmethod
    def _map_batch_pages(
        result: RemoteParseResult,
        *,
        start_page: int,
        end_page: int,
    ) -> dict[int, str]:
        expected = set(range(start_page, end_page + 1))
        returned = set(result.page_texts)
        if returned and returned <= set(range(1, end_page - start_page + 2)):
            return {
                page_number + start_page - 1: text
                for page_number, text in result.page_texts.items()
            }
        return {
            page_number: text
            for page_number, text in result.page_texts.items()
            if page_number in expected
        }

    @staticmethod
    def _map_batch_blocks(
        result: RemoteParseResult,
        *,
        start_page: int,
        end_page: int,
    ) -> dict[int, list[RemoteContentBlock]]:
        expected = set(range(start_page, end_page + 1))
        returned = set(result.page_blocks)
        relative = returned and returned <= set(range(1, end_page - start_page + 2))
        mapped: dict[int, list[RemoteContentBlock]] = {}
        for page_number, blocks in result.page_blocks.items():
            resolved_page = page_number + start_page - 1 if relative else page_number
            if resolved_page not in expected:
                continue
            mapped[resolved_page] = [
                RemoteContentBlock(
                    page_number=resolved_page,
                    block_type=block.block_type,
                    text=block.text,
                    table_html=block.table_html,
                    caption=block.caption,
                    bbox=block.bbox,
                    image_filename=block.image_filename,
                    image_content=block.image_content,
                    image_mime_type=block.image_mime_type,
                )
                for block in blocks
            ]
        return mapped

    def parse_pages(
        self,
        path: Path,
        page_numbers: list[int],
        *,
        batch_size: int | None = None,
        parse_method: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> RemoteParseResult:
        if batch_size is None:
            batch_size = self.settings.page_batch_size
        page_texts: dict[int, str] = {}
        page_blocks: dict[int, list[RemoteContentBlock]] = {}
        document_parts: list[str] = []
        warnings: list[str] = []
        target = sorted(set(page_numbers))
        total = len(target)
        done = 0
        for batch in self._page_batches(target, batch_size=batch_size):
            start_page, end_page = batch[0], batch[-1]
            result = self._parse_request(
                path,
                start_page=start_page,
                end_page=end_page,
                parse_method=parse_method,
            )
            done += len(batch)
            if progress_callback is not None:
                progress_callback(done, total)
            page_texts.update(
                self._map_batch_pages(
                    result,
                    start_page=start_page,
                    end_page=end_page,
                )
            )
            page_blocks.update(
                self._map_batch_blocks(
                    result,
                    start_page=start_page,
                    end_page=end_page,
                )
            )
            if result.document_text:
                document_parts.append(result.document_text)
            warnings.extend(
                f"pages {start_page}-{end_page}: {warning}"
                for warning in result.warnings
            )
        return RemoteParseResult(
            parser_name=self.name,
            page_texts=page_texts,
            page_blocks=page_blocks,
            document_text="\n\n".join(document_parts),
            warnings=warnings,
        )

    def parse(self, path: Path) -> RemoteParseResult:
        return self._parse_request(path)


class DoclingClient(RemoteParserClient):
    name = "docling"

    @staticmethod
    def _documents(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        if isinstance(payload.get("document"), dict):
            return [payload["document"]]
        if isinstance(payload.get("documents"), list):
            return [item for item in payload["documents"] if isinstance(item, dict)]
        if isinstance(payload.get("results"), list):
            documents: list[dict[str, Any]] = []
            for item in payload["results"]:
                if not isinstance(item, dict):
                    continue
                nested = item.get("document") or item.get("result") or item
                if isinstance(nested, dict):
                    documents.append(nested)
            return documents
        return []

    @staticmethod
    def _page_texts_from_markdown(text: str) -> dict[int, str]:
        marker = re.compile(r"(?im)^\s*<!--\s*page\s*:\s*(\d+)\s*-->\s*$")
        matches = list(marker.finditer(text))
        if not matches:
            return {}
        pages: dict[int, str] = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            page_text = text[match.end() : end].strip()
            if page_text:
                pages[int(match.group(1))] = page_text
        return pages

    @classmethod
    def _normalize(cls, payload: Any) -> RemoteParseResult:
        chunk_texts: list[str] = []
        if isinstance(payload, list):
            chunk_items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
            chunk_items = payload["results"]
        else:
            chunk_items = []
        for item in chunk_items:
            if not isinstance(item, dict):
                continue
            value = item.get("text")
            if not value and isinstance(item.get("chunk"), dict):
                value = item["chunk"].get("text")
            if isinstance(value, str) and value.strip():
                chunk_texts.append(value.strip())

        documents = cls._documents(payload)
        document_texts: list[str] = []
        for document in documents:
            for key in ("md_content", "text_content"):
                value = document.get(key)
                if isinstance(value, str) and value.strip():
                    document_texts.append(value.strip())
                    break
            if not document_texts and isinstance(document.get("json_content"), dict):
                value = document["json_content"].get("md_content")
                if isinstance(value, str) and value.strip():
                    document_texts.append(value.strip())

        text = "\n\n".join(document_texts or chunk_texts)
        pages = cls._page_texts_from_markdown(text)
        return RemoteParseResult(
            parser_name=cls.name,
            page_texts=pages,
            document_text=text,
            warnings=[] if pages else ["Docling response does not expose stable page markers."],
        )

    def parse(self, path: Path) -> RemoteParseResult:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        standard_options = {"from_formats": ["pdf"], "to_formats": ["json", "md", "text"]}
        chunked_options = {
            **standard_options,
            "do_chunking": True,
            "chunking_options": {
                "max_tokens": 512,
                "overlap": 50,
                "tokenizer": "sentencepiece",
            },
        }
        attempts = (
            (
                "v1/convert/source",
                {
                    "options": chunked_options,
                    "sources": [
                        {"kind": "file", "filename": path.name, "base64_string": encoded}
                    ],
                },
            ),
            (
                "v1alpha/convert/source",
                {
                    "options": chunked_options,
                    "file_sources": [{"filename": path.name, "base64_string": encoded}],
                },
            ),
            (
                "v1/convert/source",
                {
                    "options": standard_options,
                    "sources": [
                        {"kind": "file", "filename": path.name, "base64_string": encoded}
                    ],
                },
            ),
            (
                "v1alpha/convert/source",
                {
                    "options": standard_options,
                    "file_sources": [{"filename": path.name, "base64_string": encoded}],
                },
            ),
        )
        errors: list[str] = []
        with self._client() as client:
            for endpoint, payload in attempts:
                try:
                    response = client.post(endpoint, json=payload)
                    if response.status_code < 300:
                        return self._normalize(response.json())
                    errors.append(f"{endpoint}: HTTP {response.status_code}")
                except (httpx.HTTPError, ValueError) as exc:
                    errors.append(f"{endpoint}: {type(exc).__name__}")
        raise RuntimeError("Docling conversion failed: " + " | ".join(errors))
