from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.schemas import (
    CodexFollowupAnswerRequest,
    CodexFollowupAnswerResponse,
    CodexFollowupPublicResponse,
)
from app.services.codex_followup_service import (
    build_codex_answer_output,
    claim_codex_followup_for_answer,
    ensure_codex_followup_is_actionable,
    ensure_codex_followup_is_viewable,
    get_codex_followup_by_token,
    refresh_codex_followup_after_lost_claim,
    resume_codex_followup_in_background,
)

router = APIRouter()


@router.get("/{token}", response_model=CodexFollowupPublicResponse)
async def get_codex_followup(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> CodexFollowupPublicResponse:
    followup = await get_codex_followup_by_token(db, token)
    if followup is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Codex follow-up not found"
        )
    ensure_codex_followup_is_viewable(followup)
    return CodexFollowupPublicResponse(
        request_id=followup.id,
        workflow_name=followup.workflow_name,
        codex_label=followup.codex_label,
        summary=followup.summary,
        question=followup.question,
        task_prompt=followup.task_prompt,
        repository_url=followup.repository_url,
        base_branch=followup.base_branch,
        branch_name=followup.branch_name,
        status=followup.status,
        answer_text=followup.answer_text,
        resolved_output=followup.resolved_output or {},
        expires_at=followup.expires_at,
        answered_at=followup.answered_at,
    )


@router.post(
    "/{token}/answer",
    response_model=CodexFollowupAnswerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_codex_followup_answer(
    token: str,
    payload: CodexFollowupAnswerRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> CodexFollowupAnswerResponse:
    followup = await get_codex_followup_by_token(db, token)
    if followup is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Codex follow-up not found"
        )
    ensure_codex_followup_is_actionable(followup)

    answer_text = payload.answer_text.strip()
    now = datetime.now(timezone.utc)
    resolved_output = build_codex_answer_output(followup, answer_text=answer_text)

    claimed = await claim_codex_followup_for_answer(
        db,
        followup,
        answer_text=answer_text,
        resolved_output=resolved_output,
        answered_at=now,
    )
    if not claimed:
        # Lost the race: another answer claimed first, or the request expired
        # between the initial check and the write. Surface the same error a
        # serialized caller would have seen.
        raise await refresh_codex_followup_after_lost_claim(db, followup)

    # The claim is exclusive, so mirroring the claimed state onto the session
    # object and committing can no longer race another answer.
    followup.answer_text = answer_text
    followup.status = "answered"
    followup.answered_at = now
    followup.resume_error = None
    followup.resolved_output = resolved_output
    await db.commit()

    background_tasks.add_task(resume_codex_followup_in_background, followup.id)
    return CodexFollowupAnswerResponse(request_id=followup.id, status=followup.status)
