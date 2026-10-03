"""Regression test: Codex follow-up answers must be claimed atomically.

Concurrent submissions for the same follow-up request used to both pass the
pending check (read-check-write race), both commit, and both schedule a
background resume — so the paused workflow ran twice and one user's answer
silently overwrote the other's while both received a success response.
The claim now happens in a single conditional UPDATE before mutating the ORM object.
"""

from __future__ import annotations

import asyncio
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import delete, select

from app.api.codex_followups import submit_codex_followup_answer
from app.db.models import CodexFollowupRequest, ExecutionHistory, User, Workflow
from app.db.session import async_session_maker, engine
from app.models.schemas import CodexFollowupAnswerRequest
from app.services.codex_followup_service import (
    claim_codex_followup_for_answer,
    refresh_codex_followup_after_lost_claim,
)


def _make_followup(status: str = "pending", expired: bool = False) -> CodexFollowupRequest:
    request_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    expires_at = now - timedelta(hours=1) if expired else now + timedelta(hours=1)
    return CodexFollowupRequest(
        id=request_id,
        workflow_id=uuid.uuid4(),
        execution_history_id=uuid.uuid4(),
        public_token=f"token-{request_id.hex}",
        workflow_name="test-workflow",
        codex_node_id="codex_1",
        codex_label="codexNode",
        summary="Codex needs clarification",
        question="Which branch should be targeted?",
        task_prompt="Fix tests",
        repository_url="https://github.com/example/repo",
        base_branch="main",
        branch_name="codex/run",
        thread_id="thread-123",
        workspace_path="/tmp/workspace",
        original_output={"kind": "codex"},
        resolved_output={},
        execution_snapshot={"paused_node_id": "codex_1", "paused_node_label": "codexNode"},
        status=status,
        answer_text="Previous answer" if status == "answered" else None,
        answered_at=now if status == "answered" else None,
        expires_at=expires_at,
    )


def _result(rowcount: int | None = None, scalar: object = None) -> MagicMock:
    res = MagicMock()
    if rowcount is not None:
        res.rowcount = rowcount
    res.scalar_one_or_none.return_value = scalar
    return res


class SubmitCodexFollowupAnswerAtomicClaimTest(unittest.IsolatedAsyncioTestCase):
    async def test_winner_claims_once_and_schedules_single_resume(self) -> None:
        followup = _make_followup()
        db = AsyncMock()
        db.execute.return_value = _result(rowcount=1)
        background_tasks = MagicMock(spec=BackgroundTasks)

        with (
            patch(
                "app.api.codex_followups.get_codex_followup_by_token",
                new=AsyncMock(return_value=followup),
            ),
            patch("app.api.codex_followups.resume_codex_followup_in_background"),
        ):
            response = await submit_codex_followup_answer(
                followup.public_token,
                CodexFollowupAnswerRequest(answer_text="Use release branch"),
                background_tasks,
                db,
            )

        self.assertEqual(response.status, "answered")
        self.assertEqual(response.request_id, followup.id)
        background_tasks.add_task.assert_called_once()
        db.commit.assert_awaited_once()

    async def test_autoflush_cannot_prematurely_write_status_answered_before_claim(self) -> None:
        """Requirement 1: atomic conditional UPDATE executes before assigning anything to ORM object."""
        followup = _make_followup()
        observed_status_during_claim: list[str] = []

        db = AsyncMock()

        async def fake_execute(statement: object) -> MagicMock:
            # At the moment claim UPDATE executes, followup status must still be 'pending'
            observed_status_during_claim.append(followup.status)
            return _result(rowcount=1)

        db.execute.side_effect = fake_execute
        background_tasks = MagicMock(spec=BackgroundTasks)

        with (
            patch(
                "app.api.codex_followups.get_codex_followup_by_token",
                new=AsyncMock(return_value=followup),
            ),
            patch("app.api.codex_followups.resume_codex_followup_in_background"),
        ):
            await submit_codex_followup_answer(
                followup.public_token,
                CodexFollowupAnswerRequest(answer_text="Proceed with main"),
                background_tasks,
                db,
            )

        self.assertEqual(observed_status_during_claim, ["pending"])
        self.assertEqual(followup.status, "answered")

    async def test_lost_claim_returns_409_and_never_schedules_resume(self) -> None:
        followup = _make_followup()
        db = AsyncMock()
        db.execute.side_effect = [
            _result(rowcount=0),  # atomic claim loses
            _result(scalar=_make_followup(status="answered")),  # re-read shows winner
        ]
        background_tasks = MagicMock(spec=BackgroundTasks)

        with patch(
            "app.api.codex_followups.get_codex_followup_by_token",
            new=AsyncMock(return_value=followup),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await submit_codex_followup_answer(
                    followup.public_token,
                    CodexFollowupAnswerRequest(answer_text="Conflicting answer"),
                    background_tasks,
                    db,
                )

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.detail, "Codex follow-up has already been answered")
        background_tasks.add_task.assert_not_called()
        # A lost claim must not commit or roll back the shared request session.
        db.commit.assert_not_awaited()
        db.rollback.assert_not_called()

    async def test_lost_claim_returns_410_when_request_expired_in_flight(self) -> None:
        followup = _make_followup()
        expired = _make_followup(status="expired", expired=True)
        db = AsyncMock()
        db.execute.side_effect = [
            _result(rowcount=0),  # atomic claim loses on the expiry predicate
            _result(scalar=expired),
        ]
        background_tasks = MagicMock(spec=BackgroundTasks)

        with patch(
            "app.api.codex_followups.get_codex_followup_by_token",
            new=AsyncMock(return_value=followup),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await submit_codex_followup_answer(
                    followup.public_token,
                    CodexFollowupAnswerRequest(answer_text="Too late answer"),
                    background_tasks,
                    db,
                )

        self.assertEqual(ctx.exception.status_code, 410)
        self.assertEqual(ctx.exception.detail, "Codex link has expired")
        background_tasks.add_task.assert_not_called()
        db.commit.assert_not_awaited()


class ClaimCodexFollowupForAnswerTest(unittest.IsolatedAsyncioTestCase):
    async def test_claim_predicates_include_status_and_expiry(self) -> None:
        db = AsyncMock()
        db.execute.return_value = _result(rowcount=1)
        followup = _make_followup()

        claimed = await claim_codex_followup_for_answer(
            db,
            followup,
            answer_text="answer text",
            resolved_output={"status": "answered", "answerText": "answer text"},
        )

        self.assertTrue(claimed)
        statement = db.execute.call_args.args[0]
        predicate_columns = {expression.left.key for expression in statement.whereclause}
        self.assertEqual(predicate_columns, {"id", "status", "expires_at"})

    async def test_claim_reports_loss_when_rowcount_is_zero(self) -> None:
        db = AsyncMock()
        db.execute.return_value = _result(rowcount=0)
        followup = _make_followup()

        claimed = await claim_codex_followup_for_answer(
            db,
            followup,
            answer_text="answer text",
            resolved_output={"status": "answered"},
        )

        self.assertFalse(claimed)


class RefreshCodexFollowupAfterLostClaimTest(unittest.IsolatedAsyncioTestCase):
    async def _refresh(
        self, refreshed: CodexFollowupRequest | None
    ) -> tuple[AsyncMock, HTTPException]:
        db = AsyncMock()
        db.execute.return_value = _result(scalar=refreshed)
        error = await refresh_codex_followup_after_lost_claim(db, _make_followup())
        return db, error

    async def test_reloads_only_the_followup_row_without_rolling_back_the_session(self) -> None:
        db, error = await self._refresh(_make_followup(status="answered"))

        db.rollback.assert_not_called()
        statement = db.execute.call_args.args[0]
        self.assertTrue(statement._execution_options.get("populate_existing"))
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.detail, "Codex follow-up has already been answered")

    async def test_maps_expired_row_to_410(self) -> None:
        _, error = await self._refresh(_make_followup(status="expired", expired=True))
        self.assertEqual(error.status_code, 410)
        self.assertEqual(error.detail, "Codex link has expired")

    async def test_maps_missing_row_to_404(self) -> None:
        _, error = await self._refresh(None)
        self.assertEqual(error.status_code, 404)
        self.assertEqual(error.detail, "Codex follow-up not found")


class RealPostgresCodexFollowupConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    """Real PostgreSQL concurrency tests for atomic Codex follow-up claims.

    Guarantees:
    1. Concurrent submissions for the same follow-up: exactly one wins.
    2. Exactly one background resume task is scheduled.
    3. The winning answer's fields (answer_text, answered_at, resolved_output) are persisted.
    4. The losing competitor receives HTTP 409 with 'Codex follow-up has already been answered'.
    5. Serialized subsequent submissions also receive HTTP 409.
    """

    async def asyncSetUp(self) -> None:
        await engine.dispose()

        self.user_id = uuid.uuid4()
        self.workflow_id = uuid.uuid4()
        self.history_id = uuid.uuid4()
        self.followup_id = uuid.uuid4()
        self.token = f"token-{uuid.uuid4().hex}"

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=2)

        try:
            async with async_session_maker() as session:
                user = User(
                    id=self.user_id,
                    email=f"codex_claim_{self.user_id.hex[:8]}@example.com",
                    hashed_password="hashed_pw",
                    name="Codex Claim Test User",
                )
                session.add(user)
                await session.flush()

                workflow = Workflow(
                    id=self.workflow_id,
                    name="Codex Workflow",
                    owner_id=self.user_id,
                    nodes=[],
                    edges=[],
                )
                session.add(workflow)
                await session.flush()

                history = ExecutionHistory(
                    id=self.history_id,
                    workflow_id=self.workflow_id,
                    inputs={},
                    outputs={},
                    node_results=[],
                    status="pending",
                    execution_time_ms=100.0,
                )
                session.add(history)
                await session.flush()

                followup = CodexFollowupRequest(
                    id=self.followup_id,
                    workflow_id=self.workflow_id,
                    execution_history_id=self.history_id,
                    public_token=self.token,
                    workflow_name="Codex Workflow",
                    codex_node_id="codex_1",
                    codex_label="codexNode",
                    summary="Codex needs your decision",
                    question="Which target branch should be used?",
                    task_prompt="Implement feature",
                    repository_url="https://github.com/example/repo",
                    base_branch="main",
                    branch_name="codex/run",
                    thread_id="thread-abc",
                    workspace_path="/tmp/workspace",
                    original_output={"kind": "codex"},
                    resolved_output={},
                    execution_snapshot={
                        "paused_node_id": "codex_1",
                        "paused_node_label": "codexNode",
                    },
                    status="pending",
                    expires_at=expires_at,
                )
                session.add(followup)
                await session.commit()
        except Exception as exc:
            raise unittest.SkipTest(f"PostgreSQL setup failed: {exc}") from exc

    async def asyncTearDown(self) -> None:
        try:
            async with async_session_maker() as session:
                await session.execute(
                    delete(CodexFollowupRequest).where(
                        CodexFollowupRequest.workflow_id == self.workflow_id
                    )
                )
                await session.execute(
                    delete(ExecutionHistory).where(ExecutionHistory.workflow_id == self.workflow_id)
                )
                await session.execute(delete(Workflow).where(Workflow.id == self.workflow_id))
                await session.execute(delete(User).where(User.id == self.user_id))
                await session.commit()
        except Exception:
            pass

    async def test_concurrent_submissions_single_winner_and_loser_409(self) -> None:
        tasks_1 = MagicMock(spec=BackgroundTasks)
        tasks_2 = MagicMock(spec=BackgroundTasks)

        async def _submit(answer: str, bg_tasks: BackgroundTasks) -> dict:
            async with async_session_maker() as session:
                with patch("app.api.codex_followups.resume_codex_followup_in_background"):
                    try:
                        resp = await submit_codex_followup_answer(
                            self.token,
                            CodexFollowupAnswerRequest(answer_text=answer),
                            bg_tasks,
                            session,
                        )
                        return {"success": True, "answer": answer, "resp": resp}
                    except HTTPException as err:
                        return {"success": False, "answer": answer, "error": err}

        results = await asyncio.gather(
            _submit("Answer Alpha from competitor 1", tasks_1),
            _submit("Answer Beta from competitor 2", tasks_2),
        )

        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]

        # 1. Exactly one concurrent submission wins, exactly one loses
        self.assertEqual(len(successes), 1, f"Expected 1 winner, got {len(successes)}")
        self.assertEqual(len(failures), 1, f"Expected 1 loser, got {len(failures)}")

        winner = successes[0]
        loser = failures[0]

        # 2. Loser received HTTP 409 with exact message
        self.assertEqual(loser["error"].status_code, 409)
        self.assertEqual(loser["error"].detail, "Codex follow-up has already been answered")

        # 3. Background resume scheduled ONLY for the winner
        if winner["answer"] == "Answer Alpha from competitor 1":
            tasks_1.add_task.assert_called_once()
            tasks_2.add_task.assert_not_called()
        else:
            tasks_2.add_task.assert_called_once()
            tasks_1.add_task.assert_not_called()

        # 4. Winner's answer fields are accurately persisted in the database
        async with async_session_maker() as verify_session:
            stmt = select(CodexFollowupRequest).where(CodexFollowupRequest.id == self.followup_id)
            row = (await verify_session.execute(stmt)).scalar_one()

            self.assertEqual(row.status, "answered")
            self.assertEqual(row.answer_text, winner["answer"])
            self.assertIsNotNone(row.answered_at)
            self.assertEqual(row.resolved_output["answerText"], winner["answer"])
            self.assertEqual(row.resolved_output["status"], "answered")
            self.assertEqual(row.resolved_output["requestId"], str(self.followup_id))

        # 5. Subsequent serialized submission receives the identical 409
        tasks_3 = MagicMock(spec=BackgroundTasks)
        async with async_session_maker() as session_3:
            with self.assertRaises(HTTPException) as ctx:
                await submit_codex_followup_answer(
                    self.token,
                    CodexFollowupAnswerRequest(answer_text="Late third answer"),
                    tasks_3,
                    session_3,
                )
            self.assertEqual(ctx.exception.status_code, 409)
            self.assertEqual(ctx.exception.detail, "Codex follow-up has already been answered")
            tasks_3.add_task.assert_not_called()

    async def test_three_concurrent_submissions_single_winner(self) -> None:
        bg_tasks = [MagicMock(spec=BackgroundTasks) for _ in range(3)]

        async def _submit(index: int) -> dict:
            async with async_session_maker() as session:
                with patch("app.api.codex_followups.resume_codex_followup_in_background"):
                    try:
                        resp = await submit_codex_followup_answer(
                            self.token,
                            CodexFollowupAnswerRequest(answer_text=f"Concurrent answer #{index}"),
                            bg_tasks[index],
                            session,
                        )
                        return {"success": True, "index": index, "resp": resp}
                    except HTTPException as err:
                        return {"success": False, "index": index, "error": err}

        results = await asyncio.gather(_submit(0), _submit(1), _submit(2))
        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]

        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 2)

        for failure in failures:
            self.assertEqual(failure["error"].status_code, 409)
            self.assertEqual(failure["error"].detail, "Codex follow-up has already been answered")

        winner_idx = successes[0]["index"]
        bg_tasks[winner_idx].add_task.assert_called_once()
        for idx in range(3):
            if idx != winner_idx:
                bg_tasks[idx].add_task.assert_not_called()

    async def test_concurrent_submissions_when_expired_receive_410(self) -> None:
        # Mark the followup expired in database
        async with async_session_maker() as session:
            stmt = select(CodexFollowupRequest).where(CodexFollowupRequest.id == self.followup_id)
            row = (await session.execute(stmt)).scalar_one()
            row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
            await session.commit()

        bg_tasks_1 = MagicMock(spec=BackgroundTasks)
        bg_tasks_2 = MagicMock(spec=BackgroundTasks)

        async def _submit(answer: str, bg_tasks: BackgroundTasks) -> dict:
            async with async_session_maker() as session:
                try:
                    resp = await submit_codex_followup_answer(
                        self.token,
                        CodexFollowupAnswerRequest(answer_text=answer),
                        bg_tasks,
                        session,
                    )
                    return {"success": True, "resp": resp}
                except HTTPException as err:
                    return {"success": False, "error": err}

        results = await asyncio.gather(
            _submit("Expired attempt 1", bg_tasks_1),
            _submit("Expired attempt 2", bg_tasks_2),
        )

        for r in results:
            self.assertFalse(r["success"])
            self.assertEqual(r["error"].status_code, 410)
            self.assertEqual(r["error"].detail, "Codex link has expired")

        bg_tasks_1.add_task.assert_not_called()
        bg_tasks_2.add_task.assert_not_called()
