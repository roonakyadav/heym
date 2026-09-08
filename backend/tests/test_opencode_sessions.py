"""Conversation scope is preserved from workflows through actual SDK requests."""

import unittest
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from app.models.schemas import CredentialType
from app.services.llm_service import LLMService
from app.services.llm_trace import LLMTraceContext
from app.services.workflow_executor import WorkflowExecutor


class OpenCodeLLMSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_followups_and_tool_roundtrips_keep_conversation_id(self) -> None:
        captured: list[str] = []

        def respond(request: httpx.Request) -> httpx.Response:
            if "/models" in request.url.path:
                self.assertTrue(request.headers["x-opencode-session"])
                return httpx.Response(200, json={"object": "list", "data": []})
            captured.append(request.headers["x-opencode-session"])
            message: dict[str, Any] = {"role": "assistant", "content": "done"}
            finish_reason = "stop"
            if len(captured) == 1:
                message["tool_calls"] = [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ]
                finish_reason = "tool_calls"
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        def transport(**_kwargs: Any) -> httpx.Client:
            client = httpx.Client(transport=httpx.MockTransport(respond))
            self.addCleanup(client.close)
            return client

        def service(session_id: str) -> LLMService:
            return LLMService(
                CredentialType.custom,
                "test-key",
                "https://opencode.ai/zen/go/v1",
                trace_context=LLMTraceContext(
                    user_id=uuid.uuid4(), credential_id=uuid.uuid4(), session_id=session_id
                ),
            )

        with (
            patch("app.services.openai_client.guard_http_url"),
            patch("app.services.openai_client.build_guarded_http_client", side_effect=transport),
            patch("app.services.llm_service.record_llm_trace"),
        ):
            tool_executor = MagicMock(return_value={"result": "inspected"})
            await service("conversation-1").execute_with_tools(
                "test",
                None,
                "inspect this",
                [{"name": "inspect", "parameters": {"type": "object", "properties": {}}}],
                tool_executor,
            )
            tool_executor.assert_called_once()
            await service("conversation-1").execute("test", None, "follow up")
            await service("conversation-2").execute("test", None, "new conversation")

        self.assertEqual(captured, ["conversation-1"] * 3 + ["conversation-2"])

    async def test_service_without_trace_keeps_fallback_across_clients(self) -> None:
        service = LLMService(CredentialType.custom, "test", "https://opencode.ai/zen/go/v1")
        with (
            patch("app.services.openai_client.guard_http_url"),
            patch(
                "app.services.openai_client.build_guarded_http_client",
                side_effect=lambda **_: httpx.Client(),
            ),
        ):
            for _ in range(2):
                client, _provider = service._get_client()
                with client:
                    self.assertEqual(
                        client.default_headers["x-opencode-session"], service.session_id
                    )
                    uuid.UUID(service.session_id)


class WorkflowSessionTests(unittest.TestCase):
    def test_execution_fallback_and_conversation_override(self) -> None:
        execution_id = str(uuid.uuid4())
        executor = WorkflowExecutor([], [], execution_id=execution_id, trace_user_id=uuid.uuid4())
        self.assertEqual(executor.llm_session_id, execution_id)
        context = executor._build_llm_trace_context(str(uuid.uuid4()), None)
        self.assertEqual(context.session_id, execution_id)
        other = WorkflowExecutor([], [])
        uuid.UUID(other.llm_session_id)
        self.assertNotEqual(other.llm_session_id, executor.llm_session_id)
        followup = WorkflowExecutor([], [], execution_id="new-run", llm_session_id=execution_id)
        self.assertEqual(followup.llm_session_id, execution_id)

    def test_pause_and_notification_snapshots_preserve_session(self) -> None:
        executor = WorkflowExecutor([], [], llm_session_id="conversation")
        snapshot = executor.build_resume_snapshot(
            initial_inputs={},
            node_results=[],
            pending_count={},
            completed_nodes=set(),
            paused_node_id="agent",
            paused_node_label="Agent",
        )
        self.assertEqual(snapshot["llm_session_id"], "conversation")
        self.assertEqual(executor.build_notification_snapshot()["llm_session_id"], "conversation")
