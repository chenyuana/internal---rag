from __future__ import annotations

from app.schemas.chat import CandidateClaim, StructuredAnswer
from app.schemas.retrieval import QueryPlan, SelectedChunk


class SubjectGroundingValidator:
    """Reject claims that attribute one comparison subject's evidence to
    another subject.

    For comparison questions the planner builds a subject × aspect matrix
    (subjects like 地质灾害倾斜航测 / 河湖智能巡检无人机) and every selected
    chunk carries the subquery it was retrieved by (``SelectedChunk.subquery_id``),
    hence the subject it belongs to. A claim that names a subject must only cite
    chunks of that subject — unless the claim explicitly names the other subject
    (an honest cross-subject reference) or names no subject at all (a shared
    summary line, left untouched).

    This catches the concrete failure where the model writes "河湖智能巡检
    无人机…重叠度 50%~80%" while citing a chunk that belongs to 地质灾害倾斜航测:
    the number exists in the cited chunk (so value grounding passes) but the
    *subject attribution* is wrong. Conservative by design: no subject name in
    the claim, no known chunk subject, or a missing subquery attribution all
    skip the check.
    """

    def validate(
        self,
        answer: StructuredAnswer,
        *,
        plan: QueryPlan | None,
        chunks: list[SelectedChunk],
    ) -> None:
        if plan is None or plan.query_type != "comparison" or len(plan.subjects) < 2:
            return
        subquery_subjects = {subquery.id: subquery.subject for subquery in plan.subqueries}
        chunk_subjects: dict[str, str] = {}
        for chunk in chunks:
            if chunk.subquery_id is None:
                continue
            subject = subquery_subjects.get(chunk.subquery_id)
            if subject:
                chunk_subjects[chunk.citation_id] = subject
        if not chunk_subjects:
            return
        for claim in answer.claims:
            self._validate_claim(claim, plan.subjects, chunk_subjects)

    def _validate_claim(
        self,
        claim: CandidateClaim,
        subjects: list[str],
        chunk_subjects: dict[str, str],
    ) -> None:
        named_subjects = [subject for subject in subjects if subject in claim.claim]
        if not named_subjects:
            # 未点名任何主体的共享/总结句，无法判定归属，保守放行。
            return
        for citation_id in claim.citation_ids:
            chunk_subject = chunk_subjects.get(citation_id)
            if chunk_subject is None:
                continue
            if chunk_subject in named_subjects:
                continue
            if chunk_subject in claim.claim:
                # 诚实交叉引用：claim 同时点名该 chunk 所属主体。
                continue
            raise ValueError(
                f"claim {claim.claim_id} attributes subject '{chunk_subject}' "
                f"content to {named_subjects} via citation {citation_id} "
                f"without naming '{chunk_subject}'"
            )
