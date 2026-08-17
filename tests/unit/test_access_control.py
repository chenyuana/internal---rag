from __future__ import annotations

import pytest

from app.core.config import AccessControlSettings
from app.core.exceptions import AppError
from app.services.access_control import AccessControlService


async def test_static_acl_rejects_partial_scope() -> None:
    service = AccessControlService(
        AccessControlSettings(
            mode="static",
            static_grants={"user-1": ["kb-1"]},
        )
    )

    with pytest.raises(AppError) as error:
        await service.authorize_knowledge_bases(
            user_id="user-1",
            requested_ids=["kb-1", "kb-2"],
        )

    assert error.value.code == "KNOWLEDGE_BASE_ACCESS_DENIED"
