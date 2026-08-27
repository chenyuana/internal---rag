from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

RequirementType: TypeAlias = Literal[
    "equipment",
    "control_point_layout",
    "flight_operation",
    "data_processing",
    "quality_check",
    "deliverable",
    "acceptance",
    "reporting",
    "safety",
    "management",
    "unknown",
]


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1, max_length=100)
    cell_id: str = Field(min_length=1, max_length=50)
    subject: str | None = None
    aspect: str
    document_id: str
    document_name: str
    section_id: str | None = None
    section_title: str | None = None
    requirement_type: RequirementType = "unknown"
    requirement_text: str = Field(min_length=1)
    citation_id: str
    page_number: int | None = None
    retrieval_score: float
    confidence: float = Field(ge=0, le=1)
