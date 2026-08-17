from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

QueryType: TypeAlias = Literal[
    "fact",
    "summary",
    "comparison",
    "multi_hop",
    "procedure",
    "unknown",
]

CoverageStatus: TypeAlias = Literal["covered", "missing"]
SynthesisMode: TypeAlias = Literal["direct", "matrix", "sequence", "map_reduce"]
ExpansionMode: TypeAlias = Literal["none", "adjacent", "section", "document"]


class SubQuery(BaseModel):
    id: str = Field(min_length=1, max_length=50)
    query: str = Field(min_length=1, max_length=2000)
    subject: str | None = None
    aspect: str
    required: bool = True


class QueryPlan(BaseModel):
    query_type: QueryType
    subjects: list[str] = Field(default_factory=list)
    aspects: list[str] = Field(default_factory=list)
    subqueries: list[SubQuery] = Field(default_factory=list, max_length=12)
    expansion_mode: ExpansionMode = "none"
    synthesis_mode: SynthesisMode = "direct"

    @property
    def requires_multi_query(self) -> bool:
        return len(self.subqueries) > 1


class CoverageCell(BaseModel):
    subquery_id: str
    subject: str | None = None
    aspect: str
    status: CoverageStatus
    chunk_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


class RetrievalFilters(BaseModel):
    project_name: str | None = None
    device_model: str | None = None
    version: str | None = None
    document_type: str | None = None
    include_historical: bool = False

    @model_validator(mode="after")
    def require_version_for_history(self) -> RetrievalFilters:
        if self.include_historical and not self.version:
            raise ValueError("version is required when include_historical is true")
        return self

    def metadata_values(self) -> dict[str, str]:
        values = {
            "project_name": self.project_name,
            "device_model": self.device_model,
            "version": self.version,
            "document_type": self.document_type,
        }
        if not self.include_historical:
            values["status"] = "effective"
        return {key: value for key, value in values.items() if value is not None}


class RetrievalSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    knowledge_base_ids: list[str] = Field(min_length=1, max_length=50)
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    candidate_top_k: int | None = Field(default=None, ge=1, le=100)


class NormalizedQuery(BaseModel):
    original_query: str
    normalized_query: str
    query_type: QueryType
    exact_tokens: list[str] = Field(default_factory=list)
    article_ids: list[str] = Field(default_factory=list)
    article_aliases: list[str] = Field(default_factory=list)


class MetadataCondition(BaseModel):
    name: str
    comparison_operator: str = "="
    value: str


class MetadataConditions(BaseModel):
    logic: Literal["and", "or"] = "and"
    conditions: list[MetadataCondition]


class ReferenceMetadataRequest(BaseModel):
    include: bool = True
    fields: list[str] = Field(
        default_factory=lambda: [
            "document_name",
            "version",
            "source_path",
            "chapter_path",
            "section_title",
            "page_number",
            "paragraph_index",
            "chunk_index",
            "parent_chunk_id",
            "project_name",
            "device_model",
            "document_type",
            "status",
        ]
    )


class RagflowRetrievalRequest(BaseModel):
    question: str
    dataset_ids: list[str]
    document_ids: list[str] = Field(default_factory=list)
    page: int = 1
    page_size: int = 30
    similarity_threshold: float = 0.25
    vector_similarity_weight: float = 0.3
    top_k: int = 1024
    keyword: bool = False
    highlight: bool = False
    metadata_condition: MetadataConditions | None = None
    reference_metadata: ReferenceMetadataRequest = Field(default_factory=ReferenceMetadataRequest)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude_defaults=False)


class ChunkMetadata(BaseModel):
    document_name: str | None = None
    version: str | None = None
    source_path: str | None = None
    chapter_path: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    paragraph_index: int | None = None
    chunk_index: int | None = None
    parent_chunk_id: str | None = None
    project_name: str | None = None
    device_model: str | None = None
    document_type: str | None = None
    status: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    dataset_id: str
    text: str
    metadata: ChunkMetadata
    hybrid_score: float = 0
    vector_score: float = 0
    keyword_score: float = 0
    rerank_score: float | None = None


class SelectedChunk(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    dataset_id: str
    text: str
    metadata: ChunkMetadata
    hybrid_score: float
    vector_score: float
    keyword_score: float
    rerank_score: float | None = None


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    document_name: str | None = None
    version: str | None = None
    source_path: str | None = None
    chapter_path: str | None = None
    page_number: int | None = None
    paragraph_index: int | None = None
    quote: str


class RetrievalSearchResponse(BaseModel):
    request_id: str
    query: NormalizedQuery
    chunks: list[SelectedChunk]
    citations: list[Citation]
    candidate_count: int
    selected_count: int
    reranker_used: bool
    reranker_fallback: bool
    query_plan: QueryPlan | None = None
    coverage_matrix: list[CoverageCell] = Field(default_factory=list)


class RetrievalCandidateDebug(BaseModel):
    chunk_id: str
    document_id: str
    dataset_id: str
    document_name: str | None = None
    version: str | None = None
    chapter_path: str | None = None
    page_number: int | None = None
    hybrid_score: float
    vector_score: float
    keyword_score: float
    rerank_score: float | None = None
    raw_rank: int
    final_rank: int | None = None
    selected: bool = False
    citation_id: str | None = None
    filter_reason: str | None = None
    subquery_id: str | None = None


class RetrievalDebugResponse(RetrievalSearchResponse):
    ragflow_request: dict[str, Any]
    stage_counts: dict[str, int]
    candidates: list[RetrievalCandidateDebug]


class RerankResult(BaseModel):
    index: int = Field(ge=0)
    score: float
