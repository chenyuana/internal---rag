from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

ModelSourceType = Literal["local", "api"]


class AnswerModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=100)
    model_name: str = Field(min_length=1, max_length=200)


class AnswerModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    name: str
    source_type: ModelSourceType
    base_url: str
    models: list[str] = Field(default_factory=list)
    default_model: str | None = None
    removable: bool = False
    available: bool = True
    detail: str | None = None


class AnswerModelCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_source_id: str | None = None
    default_model: str | None = None
    sources: list[AnswerModelSource] = Field(default_factory=list)


class ModelConnectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    base_url: str = Field(min_length=8, max_length=500)
    api_key: SecretStr = Field(default=SecretStr(""), max_length=1000)


class ModelConnectionCreated(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: AnswerModelSource

