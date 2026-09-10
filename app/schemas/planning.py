from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

PlannerSource: TypeAlias = Literal["deterministic", "model", "legacy_fallback"]
RetrievalQueryKind: TypeAlias = Literal["original", "lexical", "semantic"]


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: RetrievalQueryKind
    text: str = Field(min_length=2, max_length=500)
    generated_by: Literal["user", "planner", "translator"]


class PlannedCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q[1-9][0-9]*$")
    subject: str | None = Field(default=None, max_length=200)
    aspect: str = Field(min_length=2, max_length=200)
    original_query: str = Field(min_length=2, max_length=500)
    retrieval_queries: list[RetrievalQuery] = Field(min_length=1, max_length=2)
    required: bool = True

    @model_validator(mode="after")
    def require_original_query(self) -> PlannedCell:
        originals = [item for item in self.retrieval_queries if item.kind == "original"]
        if len(originals) != 1 or originals[0].text != self.original_query:
            raise ValueError("cell must contain exactly one matching original query")
        return self


class QueryPlanV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["query-plan-v2"] = "query-plan-v2"
    query_type: Literal[
        "fact", "summary", "comparison", "multi_hop", "procedure", "enumeration", "unknown"
    ]
    subjects: list[str] = Field(default_factory=list, max_length=4)
    aspects: list[str] = Field(default_factory=list, max_length=8)
    cells: list[PlannedCell] = Field(min_length=1, max_length=8)
    synthesis_mode: Literal["direct", "matrix", "sequence", "map_reduce"]
    planner_source: PlannerSource
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_cells(self) -> QueryPlanV2:
        cell_ids = [cell.id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("cell ids must be unique")
        if self.query_type == "comparison" and self.synthesis_mode == "matrix":
            expected = len(self.subjects) * len(self.aspects)
            labels = {(cell.subject, cell.aspect) for cell in self.cells}
            expected_labels = {
                (subject, aspect)
                for aspect in self.aspects
                for subject in self.subjects
            }
            if (
                len(self.subjects) < 2
                or not self.aspects
                or len(self.cells) != expected
                or labels != expected_labels
            ):
                raise ValueError("comparison plan must be a complete subject × aspect matrix")
        return self


class TranslationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_translate: bool
    translated_query: str | None = Field(default=None, max_length=500)
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def require_query_when_enabled(self) -> TranslationDecision:
        if self.should_translate and not (self.translated_query or "").strip():
            raise ValueError("translated_query is required when should_translate=true")
        if not self.should_translate and self.translated_query is not None:
            raise ValueError("translated_query must be null when should_translate=false")
        return self


class SubjectResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    document_ids: list[str] = Field(default_factory=list)
    document_names: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    source: Literal["request", "metadata", "retrieval", "title", "model", "none"]
