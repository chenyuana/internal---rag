from __future__ import annotations

from app.core.config import AccessControlSettings
from app.core.exceptions import AppError


class AccessControlService:
    """Phase-three ACL boundary; replace with a repository-backed service later."""

    def __init__(self, settings: AccessControlSettings) -> None:
        self._settings = settings

    async def authorize_knowledge_bases(
        self,
        *,
        user_id: str,
        requested_ids: list[str],
    ) -> list[str]:
        if self._settings.mode == "development_passthrough":
            return list(dict.fromkeys(requested_ids))
        if self._settings.mode == "static":
            granted = set(self._settings.static_grants.get(user_id, []))
            allowed = [item for item in dict.fromkeys(requested_ids) if item in granted]
        else:
            allowed = []
        if not allowed:
            raise AppError(
                code="KNOWLEDGE_BASE_ACCESS_DENIED",
                message="The user has no access to the requested knowledge bases.",
                status_code=403,
            )
        if len(allowed) != len(set(requested_ids)):
            raise AppError(
                code="KNOWLEDGE_BASE_ACCESS_DENIED",
                message="At least one requested knowledge base is not permitted.",
                status_code=403,
            )
        return allowed
