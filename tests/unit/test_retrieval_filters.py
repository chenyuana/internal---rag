from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.retrieval import RetrievalFilters


def test_effective_status_is_mandatory_by_default() -> None:
    filters = RetrievalFilters(project_name="MC任务机")

    assert filters.metadata_values() == {
        "project_name": "MC任务机",
        "status": "effective",
    }


def test_history_requires_explicit_version() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilters(include_historical=True)


def test_explicit_history_does_not_force_effective_status() -> None:
    filters = RetrievalFilters(version="V1.0", include_historical=True)

    assert filters.metadata_values() == {"version": "V1.0"}
