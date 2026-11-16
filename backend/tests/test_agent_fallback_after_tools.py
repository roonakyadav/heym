"""Regression tests for Issue #1000 — Resume agent fallback from completed tool history.

Verifies:
1. Same transport, failure after tool execution:
   - Primary runs tool, then fails with HTTP 500.
   - Fallback receives completed tool result.
   - Tool executes exactly once.
   - Final output contains tool calls from both attempts.
   - fallbackUsed is true.
2. Different transport:
   - Primary executes tool, fails.
   - Fallback uses incompatible transport.
   - Fallback starts from original prompt.
   - No invalid cross-transport history loaded.
3. Failure before any tool:
   - Primary fails before executing any tool.
   - Fallback starts from original prompt.
4. Usage and traces:
   - Node usage is sum of both attempts.
   - Primary and fallback traces record per-invocation usage without double counting.
   - Elapsed times are properly partitioned.
5. Responses API reasoning isolation:
   - Primary history contains reasoning item.
   - Fallback uses same Responses transport but different credential:
     Reasoning item stripped from fallback request; tool result preserved.
   - Fallback with same credential keeps reasoning item.
6. No fallback configured:
   - Preserves error and collected tool calls in error output.
7. Tool execution returns an error:
   - Fallback resumes with error result and does not repeat tool action.
8. Inherited initial_tool_calls:
   - Inherited tool calls from prior run/HITL do not falsely trigger ResumableProviderError
     if failure occurs before any new tools run.
"""

from __future__ import annotations

import copy
import uuid
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

import httpx
import openai

from app.db.models import CredentialType
from app.services.llm_service import LLMService, resolve_transport_identity
from app.services.llm_transport import (
    ResumableProviderError,
    are_transports_compatible,
    isolate_responses_history_reasoning,
    prepare_resumable_fallback_attempt,
)
from app.services.workflow_executor import WorkflowExecutor

TOOLS = [
    {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    }
]


def _call(call_id: str, arguments: str = "{}", name: str = "send_email") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion(
    *,
    content: str | None = None,
    calls: list[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=calls or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _internal_server_error() -> openai.InternalServerError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(500, request=request)
    return openai.InternalServerError(
        "HTTP 500: Provider failure",
        response=response,
        body={"error": {"message": "Internal server error"}},
    )


def _make_mock_client(responses: list[Any]) -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    sent_requests: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        sent_requests.append(copy.deepcopy(kwargs))
        if not responses:
            raise RuntimeError("No more mocked responses available")
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    return client, sent_requests


def _responses_item_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _responses_turn(
    items: list[dict[str, Any]], prompt_tokens: int = 100, completion_tokens: int = 20
) -> Any:
    resp = MagicMock()
    resp.output = items
    resp.usage.input_tokens = prompt_tokens
    resp.usage.output_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


class TestTransportCompatibilityHelpers(IsolatedAsyncioTestCase):
    """Unit tests for transport resolution and compatibility helper functions."""

    def test_are_transports_compatible(self) -> None:
        self.assertTrue(are_transports_compatible("chat.completions", "chat.completions"))
        self.assertTrue(are_transports_compatible("responses", "responses"))
        self.assertTrue(are_transports_compatible("messages", "messages"))
        self.assertTrue(are_transports_compatible("generateContent", "generateContent"))
        self.assertFalse(are_transports_compatible("messages", "chat.completions"))
        self.assertFalse(are_transports_compatible("chat.completions", "responses"))
        self.assertFalse(are_transports_compatible("unknown", "unknown"))

    def test_isolate_responses_history_reasoning(self) -> None:
        history = [
            {"role": "user", "content": "hello"},
            {"type": "reasoning", "id": "r1", "summary": ["thinking"]},
            {"type": "function_call", "call_id": "c1", "name": "send_email", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": '{"ok": true}'},
        ]
        # Same credential keeps reasoning
        kept = isolate_responses_history_reasoning(
            history, source_credential_id="cred-1", target_credential_id="cred-1"
        )
        self.assertEqual(len(kept), 4)
        self.assertTrue(any(item.get("type") == "reasoning" for item in kept))

        # Different credential removes reasoning
        stripped = isolate_responses_history_reasoning(
            history, source_credential_id="cred-1", target_credential_id="cred-2"
        )
        self.assertEqual(len(stripped), 3)
        self.assertFalse(any(item.get("type") == "reasoning" for item in stripped))
        self.assertTrue(any(item.get("type") == "function_call" for item in stripped))
        self.assertTrue(any(item.get("type") == "function_call_output" for item in stripped))

    def test_resolve_transport_identity(self) -> None:
        self.assertEqual(
            resolve_transport_identity(
                "gpt-4o", CredentialType.openai, "https://api.openai.com/v1"
            ),
            "chat.completions",
        )
        self.assertEqual(
            resolve_transport_identity(
                "gpt-4o", CredentialType.openai, "https://api.openai.com/v1", use_responses_api=True
            ),
            "responses",
        )
        self.assertEqual(
            resolve_transport_identity(
                "gpt-4o", CredentialType.openai, "https://api.openai.com/v1", has_router=True
            ),
            "chat.completions",
        )
        # OpenCode model starting with claude- uses messages protocol
        self.assertEqual(
            resolve_transport_identity(
                "claude-3.5-sonnet", CredentialType.custom, "https://opencode.ai/zen/go/v1"
            ),
            "messages",
        )

    def test_prepare_resumable_fallback_attempt(self) -> None:
        # None state returns defaults
        res = prepare_resumable_fallback_attempt(
            None,
            primary_credential_id="c1",
            attempt_credential_id="c2",
            model="m2",
            credential_type=CredentialType.openai,
            base_url="https://api.openai.com/v1",
            default_messages=[{"role": "user", "content": "hi"}],
            default_tool_calls=[],
            default_elapsed_ms=0.0,
            default_prompt_tokens=0,
            default_completion_tokens=0,
            default_max_iterations=10,
        )
        self.assertIsNone(res[0])
        self.assertEqual(res[1], [{"role": "user", "content": "hi"}])
        self.assertEqual(res[6], 10)

        # Incompatible transports returns defaults
        rpe = ResumableProviderError(
            RuntimeError("boom"),
            history=[{"role": "user", "content": "hi"}],
            tool_calls=[{"name": "t1"}],
            prompt_tokens=50,
            completion_tokens=10,
            elapsed_ms=100.0,
            remaining_iterations=8,
            transport_name="messages",
        )
        res_incomp = prepare_resumable_fallback_attempt(
            rpe,
            primary_credential_id="c1",
            attempt_credential_id="c2",
            model="gpt-4o",
            credential_type=CredentialType.openai,
            base_url="https://api.openai.com/v1",
            default_messages=[{"role": "user", "content": "orig"}],
            default_tool_calls=[],
            default_elapsed_ms=0.0,
            default_prompt_tokens=0,
            default_completion_tokens=0,
            default_max_iterations=10,
        )
        self.assertIsNone(res_incomp[0])
        self.assertEqual(res_incomp[1], [{"role": "user", "content": "orig"}])


class TestAgentFallbackAfterTools(IsolatedAsyncioTestCase):
    """End-to-end regression test suite for Issue #1000."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch("app.services.llm_service.record_llm_trace", return_value=uuid.uuid4())
        self._record_trace_mock = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_executor(self, node_data: dict[str, Any]) -> WorkflowExecutor:
        ex = WorkflowExecutor.__new__(WorkflowExecutor)
        node_id = "agent-node-1"
        ex.nodes = {node_id: {"type": "agent", "data": node_data}}
        ex.edges = []
        ex.node_results = {}
        ex.agent_progress_queue = None
        ex._sub_agent_call_depth = 0
        ex.check_cancelled = MagicMock()
        ex.hitl_resume_context = {}
        ex.conversation_history = None
        ex.workflow_cache = {}
        ex.trace_user_id = uuid.uuid4()
        ex.actor_user_id = uuid.uuid4()
        ex.workflow_id = None
        ex.llm_session_id = "test-session-id"
        ex.file_scope = None
        ex.cancel_event = None
        ex.get_node_label = MagicMock(return_value="Agent Node")
        ex._resolve_template = MagicMock(side_effect=lambda tmpl, *a, **kw: tmpl)
        ex.resolve_expression = MagicMock(return_value="")
        ex._list_mcp_tools = MagicMock(return_value=[])
        ex._build_hitl_mcp_policy = MagicMock(return_value={})
        ex._build_agent_tool_executor = MagicMock(return_value=None)
        ex._visible_inputs = MagicMock(return_value={})
        return ex

    def test_same_transport_resumes_from_completed_tool_history(self) -> None:
        """Scenario 1: Same transport, failure after tool execution."""
        executed_calls: list[tuple[str, dict[str, Any]]] = []

        def tracking_tool_executor(
            _tool_def: dict[str, Any],
            name: str,
            args: dict[str, Any],
            _timeout: float,
        ) -> dict[str, Any]:
            executed_calls.append((name, args))
            return {"status": "sent", "id": "email-001"}

        primary_client, _primary_sent = _make_mock_client(
            [
                _completion(
                    calls=[
                        _call(
                            "call-1", '{"to": "alice@example.com", "body": "Hi"}', name="send_email"
                        )
                    ],
                    finish_reason="tool_calls",
                    prompt_tokens=100,
                    completion_tokens=20,
                ),
                _internal_server_error(),
            ]
        )

        fallback_client, fallback_sent = _make_mock_client(
            [
                _completion(
                    content="Email has been sent.",
                    calls=[],
                    finish_reason="stop",
                    prompt_tokens=150,
                    completion_tokens=30,
                )
            ]
        )

        def mock_get_client(service: LLMService) -> tuple[Any, str]:
            if service.api_key == "sk-primary":
                return primary_client, "OpenAI"
            return fallback_client, "OpenAI"

        cid_primary = str(uuid.uuid4())
        cid_fallback = str(uuid.uuid4())
        node_data = {
            "label": "Email Agent",
            "model": "primary-model",
            "credentialId": cid_primary,
            "fallbackModel": "fallback-model",
            "fallbackCredentialId": cid_fallback,
            "tools": TOOLS,
            "userMessage": "Send an email to Alice",
        }
        ex = self._make_executor(node_data)

        cred_primary = MagicMock()
        cred_primary.type = CredentialType.openai
        cred_primary.name = "Primary OpenAI"
        cred_primary.encrypted_config = b"enc_primary"

        cred_fallback = MagicMock()
        cred_fallback.type = CredentialType.openai
        cred_fallback.name = "Fallback OpenAI"
        cred_fallback.encrypted_config = b"enc_fallback"

        def mock_get_cred(_db: Any, cid: str) -> Any:
            return cred_primary if cid == cid_primary else cred_fallback

        def mock_decrypt(cfg: bytes) -> dict[str, Any]:
            if cfg == b"enc_primary":
                return {"api_key": "sk-primary", "base_url": "https://api.openai.com/v1"}
            return {"api_key": "sk-fallback", "base_url": "https://api.openai.com/v1"}

        with (
            patch("app.db.session.SessionLocal"),
            patch.object(ex, "_get_accessible_credential", side_effect=mock_get_cred),
            patch("app.services.encryption.decrypt_config", side_effect=mock_decrypt),
            patch.object(LLMService, "_get_client", autospec=True, side_effect=mock_get_client),
            patch(
                "app.services.llm_service._unified_tool_executor",
                side_effect=tracking_tool_executor,
            ),
        ):
            result = ex._execute_agent_node("agent-node-1", {}, node_data)

        # Tool executed exactly once!
        self.assertEqual(len(executed_calls), 1)
        self.assertEqual(executed_calls[0][0], "send_email")

        # Fallback continuation received tool result
        fallback_msgs = fallback_sent[0]["messages"]
        self.assertTrue(any(m.get("role") == "tool" for m in fallback_msgs))
        tool_msg = next(m for m in fallback_msgs if m.get("role") == "tool")
        self.assertEqual(tool_msg.get("tool_call_id"), "call-1")

        # Result verifies fallback used and tool calls retained
        self.assertTrue(result.get("fallbackUsed"))
        self.assertEqual(result.get("model"), "fallback-model")
        self.assertEqual(result.get("text"), "Email has been sent.")
        self.assertEqual(len(result.get("tool_calls", [])), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "send_email")
        self.assertEqual(result["tool_calls"][0]["status"], "success")

    def test_different_transport_restarts_from_prompt(self) -> None:
        """Scenario 2: Incompatible transport restarts from original prompt."""
        executed_calls: list[tuple[str, dict[str, Any]]] = []

        def tracking_tool_executor(
            _tool_def: dict[str, Any],
            name: str,
            args: dict[str, Any],
            _timeout: float,
        ) -> dict[str, Any]:
            executed_calls.append((name, args))
            return {"status": "sent"}

        # Primary uses Messages transport (Anthropic model via OpenCode)
        primary_calls = 0

        def primary_post(url: str, **kwargs: Any) -> Any:
            nonlocal primary_calls
            primary_calls += 1
            if primary_calls == 1:
                resp_mock = MagicMock()
                resp_mock.json.return_value = {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "send_email",
                            "input": {"to": "alice@example.com", "body": "Hi"},
                        }
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                }
                return resp_mock
            raise _internal_server_error()

        primary_client = SimpleNamespace(
            api_key="sk-primary",
            base_url="https://opencode.ai/zen/go/v1",
            post=primary_post,
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )

        fallback_client, fallback_sent = _make_mock_client(
            [
                _completion(
                    content="Fallback restarted from prompt.",
                    calls=[],
                    finish_reason="stop",
                )
            ]
        )

        def mock_get_client(service: LLMService) -> tuple[Any, str]:
            if service.api_key == "sk-primary":
                return primary_client, "Custom"
            return fallback_client, "OpenAI"

        cid_primary = str(uuid.uuid4())
        cid_fallback = str(uuid.uuid4())
        node_data = {
            "model": "claude-3.5-sonnet",
            "credentialId": cid_primary,
            "fallbackModel": "gpt-4o",
            "fallbackCredentialId": cid_fallback,
            "tools": TOOLS,
            "userMessage": "Send an email to Alice",
        }
        ex = self._make_executor(node_data)

        cred_primary = MagicMock(type=CredentialType.custom, encrypted_config=b"enc_primary")
        cred_fallback = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_fallback")

        def mock_get_cred(_db: Any, cid: str) -> Any:
            return cred_primary if cid == cid_primary else cred_fallback

        def mock_decrypt(cfg: bytes) -> dict[str, Any]:
            if cfg == b"enc_primary":
                return {"api_key": "sk-primary", "base_url": "https://opencode.ai/zen/go/v1"}
            return {"api_key": "sk-fallback", "base_url": "https://api.openai.com/v1"}

        with (
            patch("app.db.session.SessionLocal"),
            patch.object(ex, "_get_accessible_credential", side_effect=mock_get_cred),
            patch("app.services.encryption.decrypt_config", side_effect=mock_decrypt),
            patch.object(LLMService, "_get_client", autospec=True, side_effect=mock_get_client),
            patch(
                "app.services.llm_service._unified_tool_executor",
                side_effect=tracking_tool_executor,
            ),
        ):
            result = ex._execute_agent_node("agent-node-1", {}, node_data)

        # Primary used messages transport; fallback is chat.completions.
        # Fallback restarted from the original prompt without messages-format history:
        fallback_msgs = fallback_sent[0]["messages"]
        self.assertFalse(any(m.get("role") == "tool" for m in fallback_msgs))
        self.assertTrue(result.get("fallbackUsed"))
        self.assertEqual(result.get("text"), "Fallback restarted from prompt.")

    def test_failure_before_any_tool_starts_from_prompt(self) -> None:
        """Scenario 3: Primary fails before executing any tool."""
        primary_client, _ = _make_mock_client([_internal_server_error()])
        fallback_client, fallback_sent = _make_mock_client(
            [
                _completion(
                    content="Fallback succeeded without tools.",
                    calls=[],
                    finish_reason="stop",
                )
            ]
        )

        def mock_get_client(service: LLMService) -> tuple[Any, str]:
            if service.api_key == "sk-primary":
                return primary_client, "OpenAI"
            return fallback_client, "OpenAI"

        cid_primary = str(uuid.uuid4())
        cid_fallback = str(uuid.uuid4())
        node_data = {
            "model": "primary-model",
            "credentialId": cid_primary,
            "fallbackModel": "fallback-model",
            "fallbackCredentialId": cid_fallback,
            "tools": TOOLS,
            "userMessage": "Hello",
        }
        ex = self._make_executor(node_data)

        cred_primary = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_primary")
        cred_fallback = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_fallback")

        def mock_get_cred(_db: Any, cid: str) -> Any:
            return cred_primary if cid == cid_primary else cred_fallback

        def mock_decrypt(cfg: bytes) -> dict[str, Any]:
            return {
                "api_key": "sk-primary" if cfg == b"enc_primary" else "sk-fallback",
                "base_url": "https://api.openai.com/v1",
            }

        with (
            patch("app.db.session.SessionLocal"),
            patch.object(ex, "_get_accessible_credential", side_effect=mock_get_cred),
            patch("app.services.encryption.decrypt_config", side_effect=mock_decrypt),
            patch.object(LLMService, "_get_client", autospec=True, side_effect=mock_get_client),
        ):
            result = ex._execute_agent_node("agent-node-1", {}, node_data)

        # Fallback started from original prompt
        fallback_msgs = fallback_sent[0]["messages"]
        self.assertFalse(any(m.get("role") == "tool" for m in fallback_msgs))
        self.assertTrue(result.get("fallbackUsed"))
        self.assertEqual(result.get("text"), "Fallback succeeded without tools.")

    def test_usage_and_trace_accounting_partitions_attempts(self) -> None:
        """Scenario 4: Usage and trace accounting without double counting."""
        primary_client, _ = _make_mock_client(
            [
                _completion(
                    calls=[_call("call-1", '{"to": "alice@example.com", "body": "Hi"}')],
                    finish_reason="tool_calls",
                    prompt_tokens=100,
                    completion_tokens=20,
                ),
                _internal_server_error(),
            ]
        )

        fallback_client, _ = _make_mock_client(
            [
                _completion(
                    content="Done.",
                    calls=[],
                    finish_reason="stop",
                    prompt_tokens=150,
                    completion_tokens=30,
                )
            ]
        )

        def mock_get_client(service: LLMService) -> tuple[Any, str]:
            return (
                primary_client if service.api_key == "sk-primary" else fallback_client
            ), "OpenAI"

        cid_primary = str(uuid.uuid4())
        cid_fallback = str(uuid.uuid4())
        node_data = {
            "model": "primary-model",
            "credentialId": cid_primary,
            "fallbackModel": "fallback-model",
            "fallbackCredentialId": cid_fallback,
            "tools": TOOLS,
            "userMessage": "Send an email",
        }
        ex = self._make_executor(node_data)

        cred_primary = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_primary")
        cred_fallback = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_fallback")

        def mock_get_cred(_db: Any, cid: str) -> Any:
            return cred_primary if cid == cid_primary else cred_fallback

        def mock_decrypt(cfg: bytes) -> dict[str, Any]:
            return {
                "api_key": "sk-primary" if cfg == b"enc_primary" else "sk-fallback",
                "base_url": "https://api.openai.com/v1",
            }

        recorded_traces: list[dict[str, Any]] = []

        def capture_trace(**kwargs: Any) -> None:
            recorded_traces.append(kwargs)

        with (
            patch("app.db.session.SessionLocal"),
            patch.object(ex, "_get_accessible_credential", side_effect=mock_get_cred),
            patch("app.services.encryption.decrypt_config", side_effect=mock_decrypt),
            patch.object(LLMService, "_get_client", autospec=True, side_effect=mock_get_client),
            patch("app.services.llm_service._unified_tool_executor", return_value={"status": "ok"}),
            patch("app.services.llm_service.record_llm_trace", side_effect=capture_trace),
        ):
            result = ex._execute_agent_node("agent-node-1", {}, node_data)

        # Node total usage is sum of both attempts
        # Primary: 100 prompt + 20 completion = 120
        # Fallback: 150 prompt + 30 completion = 180
        # Total: 250 prompt + 50 completion = 300
        self.assertEqual(result["usage"]["prompt_tokens"], 250)
        self.assertEqual(result["usage"]["completion_tokens"], 50)
        self.assertEqual(result["usage"]["total_tokens"], 300)

        # Traces:
        # We expect 2 trace rows: primary error trace, and fallback success trace
        self.assertEqual(len(recorded_traces), 2)
        primary_trace = recorded_traces[0]
        fallback_trace = recorded_traces[1]

        # Primary trace recorded its own invocation tokens
        self.assertIsNotNone(primary_trace["error"])
        self.assertEqual(primary_trace["prompt_tokens"], 100)
        self.assertEqual(primary_trace["completion_tokens"], 20)
        self.assertEqual(primary_trace["total_tokens"], 120)

        # Fallback trace recorded only its invocation tokens
        self.assertIsNone(fallback_trace["error"])
        self.assertEqual(fallback_trace["prompt_tokens"], 150)
        self.assertEqual(fallback_trace["completion_tokens"], 30)
        self.assertEqual(fallback_trace["total_tokens"], 180)

        # Fallback trace elapsed_ms does not duplicate primary trace duration
        self.assertGreaterEqual(result["elapsed_ms"], fallback_trace["elapsed_ms"])

    def test_responses_api_reasoning_isolation_different_credentials(self) -> None:
        """Scenario 5: Responses API isolates encrypted reasoning across credentials."""
        primary_sent: list[dict[str, Any]] = []
        fallback_sent: list[dict[str, Any]] = []

        primary_turn_1 = _responses_turn(
            [
                {
                    "type": "reasoning",
                    "id": "reas-1",
                    "summary": [{"type": "text", "text": "secret thinking"}],
                },
                _responses_item_call(
                    "call-1", "send_email", '{"to": "alice@example.com", "body": "Hi"}'
                ),
            ]
        )

        def mock_primary_create(**kwargs: Any) -> Any:
            primary_sent.append(copy.deepcopy(kwargs))
            if len(primary_sent) == 1:
                return primary_turn_1
            raise _internal_server_error()

        fallback_turn_1 = _responses_turn(
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Email sent."}],
                }
            ]
        )

        def mock_fallback_create(**kwargs: Any) -> Any:
            fallback_sent.append(copy.deepcopy(kwargs))
            return fallback_turn_1

        primary_client = SimpleNamespace(
            base_url="https://api.openai.com/v1",
            responses=SimpleNamespace(create=mock_primary_create),
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )
        fallback_client = SimpleNamespace(
            base_url="https://api.openai.com/v1",
            responses=SimpleNamespace(create=mock_fallback_create),
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )

        def mock_get_client(service: LLMService) -> tuple[Any, str]:
            return (
                primary_client if service.api_key == "sk-primary" else fallback_client
            ), "OpenAI"

        cid_primary = str(uuid.uuid4())
        cid_fallback = str(uuid.uuid4())
        node_data = {
            "model": "gpt-5",
            "credentialId": cid_primary,
            "fallbackModel": "gpt-5",
            "fallbackCredentialId": cid_fallback,
            "tools": TOOLS,
            "responsesApiEnabled": True,
            "userMessage": "Send an email",
        }
        ex = self._make_executor(node_data)

        cred_primary = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_primary")
        cred_fallback = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_fallback")

        def mock_get_cred(_db: Any, cid: str) -> Any:
            return cred_primary if cid == cid_primary else cred_fallback

        def mock_decrypt(cfg: bytes) -> dict[str, Any]:
            return {
                "api_key": "sk-primary" if cfg == b"enc_primary" else "sk-fallback",
                "base_url": "https://api.openai.com/v1",
            }

        with (
            patch("app.db.session.SessionLocal"),
            patch.object(ex, "_get_accessible_credential", side_effect=mock_get_cred),
            patch("app.services.encryption.decrypt_config", side_effect=mock_decrypt),
            patch.object(LLMService, "_get_client", autospec=True, side_effect=mock_get_client),
            patch(
                "app.services.llm_service._unified_tool_executor", return_value={"status": "sent"}
            ),
        ):
            result = ex._execute_agent_node("agent-node-1", {}, node_data)

        self.assertTrue(result.get("fallbackUsed"))
        self.assertEqual(len(fallback_sent), 1)
        fallback_input = fallback_sent[0]["input"]

        # Encrypted reasoning item from cred-primary was stripped!
        self.assertFalse(
            any(isinstance(it, dict) and it.get("type") == "reasoning" for it in fallback_input)
        )
        # But function_call and function_call_output are preserved:
        self.assertTrue(
            any(isinstance(it, dict) and it.get("type") == "function_call" for it in fallback_input)
        )
        self.assertTrue(
            any(
                isinstance(it, dict) and it.get("type") == "function_call_output"
                for it in fallback_input
            )
        )

    def test_responses_api_reasoning_kept_for_same_credential(self) -> None:
        """Responses API keeps reasoning item when primary and fallback share credential."""
        primary_turn_1 = _responses_turn(
            [
                {
                    "type": "reasoning",
                    "id": "reas-1",
                    "summary": [{"type": "text", "text": "secret thinking"}],
                },
                _responses_item_call(
                    "call-1", "send_email", '{"to": "alice@example.com", "body": "Hi"}'
                ),
            ]
        )
        fallback_turn_1 = _responses_turn(
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Email sent."}],
                }
            ]
        )
        all_sent: list[dict[str, Any]] = []

        def mock_responses_create(**kwargs: Any) -> Any:
            all_sent.append(copy.deepcopy(kwargs))
            if len(all_sent) == 1:
                return primary_turn_1
            elif len(all_sent) == 2:
                raise _internal_server_error()
            else:
                return fallback_turn_1

        client = SimpleNamespace(
            base_url="https://api.openai.com/v1",
            responses=SimpleNamespace(create=mock_responses_create),
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )

        cid_same = str(uuid.uuid4())
        node_data = {
            "model": "primary-model",
            "credentialId": cid_same,
            "fallbackModel": "fallback-model",
            "fallbackCredentialId": cid_same,  # Same credential ID
            "tools": TOOLS,
            "responsesApiEnabled": True,
            "userMessage": "Send an email",
        }
        ex = self._make_executor(node_data)
        cred = MagicMock(type=CredentialType.openai, encrypted_config=b"enc")

        with (
            patch("app.db.session.SessionLocal"),
            patch.object(ex, "_get_accessible_credential", return_value=cred),
            patch(
                "app.services.encryption.decrypt_config",
                return_value={"api_key": "sk-same", "base_url": "https://api.openai.com/v1"},
            ),
            patch.object(LLMService, "_get_client", autospec=True, return_value=(client, "OpenAI")),
            patch(
                "app.services.llm_service._unified_tool_executor", return_value={"status": "sent"}
            ),
            patch("app.services.llm_service.record_llm_trace"),
        ):
            result = ex._execute_agent_node("agent-node-1", {}, node_data)

        self.assertTrue(result.get("fallbackUsed"))
        fallback_input = all_sent[2]["input"]
        # Same credential preserves reasoning item!
        self.assertTrue(
            any(isinstance(it, dict) and it.get("type") == "reasoning" for it in fallback_input)
        )

    def test_no_fallback_configured_preserves_error_and_tool_calls(self) -> None:
        """Scenario 6: No fallback configured preserves error and executed tool calls."""
        primary_client, _ = _make_mock_client(
            [
                _completion(
                    calls=[_call("call-1", '{"to": "alice@example.com", "body": "Hi"}')],
                    finish_reason="tool_calls",
                ),
                _internal_server_error(),
            ]
        )

        def mock_get_client(_service: LLMService) -> tuple[Any, str]:
            return primary_client, "OpenAI"

        cid_primary = str(uuid.uuid4())
        node_data = {
            "model": "primary-model",
            "credentialId": cid_primary,
            "tools": TOOLS,
            "userMessage": "Send an email",
        }
        ex = self._make_executor(node_data)
        cred = MagicMock(type=CredentialType.openai, encrypted_config=b"enc")

        with (
            patch("app.db.session.SessionLocal"),
            patch.object(ex, "_get_accessible_credential", return_value=cred),
            patch(
                "app.services.encryption.decrypt_config",
                return_value={"api_key": "sk-primary", "base_url": "https://api.openai.com/v1"},
            ),
            patch.object(LLMService, "_get_client", autospec=True, side_effect=mock_get_client),
            patch(
                "app.services.llm_service._unified_tool_executor", return_value={"status": "sent"}
            ),
        ):
            result = ex._execute_agent_node("agent-node-1", {}, node_data)

        self.assertIn("error", result)
        self.assertIn("500", result["error"])
        # Tool call executed by primary is preserved in error_output!
        self.assertIn("tool_calls", result)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "send_email")

    def test_tool_execution_returns_error_result_resumes_without_repeat(self) -> None:
        """Scenario 7: Tool execution returns error status; fallback resumes without repeat."""
        executed_calls: list[str] = []

        def failing_tool_executor(
            _tool_def: dict[str, Any],
            name: str,
            _args: dict[str, Any],
            _timeout: float,
        ) -> dict[str, Any]:
            executed_calls.append(name)
            return {"error": "Mail server connection refused", "status": "error"}

        primary_client, _ = _make_mock_client(
            [
                _completion(
                    calls=[_call("call-1", '{"to": "alice@example.com", "body": "Hi"}')],
                    finish_reason="tool_calls",
                ),
                _internal_server_error(),
            ]
        )

        fallback_client, fallback_sent = _make_mock_client(
            [
                _completion(
                    content="I noticed the mail server failed so I stopped.",
                    calls=[],
                    finish_reason="stop",
                )
            ]
        )

        def mock_get_client(service: LLMService) -> tuple[Any, str]:
            return (
                primary_client if service.api_key == "sk-primary" else fallback_client
            ), "OpenAI"

        cid_primary = str(uuid.uuid4())
        cid_fallback = str(uuid.uuid4())
        node_data = {
            "model": "primary-model",
            "credentialId": cid_primary,
            "fallbackModel": "fallback-model",
            "fallbackCredentialId": cid_fallback,
            "tools": TOOLS,
            "userMessage": "Send an email",
        }
        ex = self._make_executor(node_data)
        cred_primary = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_primary")
        cred_fallback = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_fallback")

        def mock_get_cred(_db: Any, cid: str) -> Any:
            return cred_primary if cid == cid_primary else cred_fallback

        def mock_decrypt(cfg: bytes) -> dict[str, Any]:
            return {
                "api_key": "sk-primary" if cfg == b"enc_primary" else "sk-fallback",
                "base_url": "https://api.openai.com/v1",
            }

        with (
            patch("app.db.session.SessionLocal"),
            patch.object(ex, "_get_accessible_credential", side_effect=mock_get_cred),
            patch("app.services.encryption.decrypt_config", side_effect=mock_decrypt),
            patch.object(LLMService, "_get_client", autospec=True, side_effect=mock_get_client),
            patch(
                "app.services.llm_service._unified_tool_executor", side_effect=failing_tool_executor
            ),
        ):
            result = ex._execute_agent_node("agent-node-1", {}, node_data)

        # Tool executed once
        self.assertEqual(len(executed_calls), 1)

        # Fallback received the error result in history
        fallback_msgs = fallback_sent[0]["messages"]
        tool_msg = next(m for m in fallback_msgs if m.get("role") == "tool")
        self.assertIn("connection refused", tool_msg["content"])

        self.assertTrue(result.get("fallbackUsed"))
        self.assertEqual(result["tool_calls"][0]["status"], "error")

    def test_inherited_initial_tool_calls_does_not_falsely_count_as_current_invocation(
        self,
    ) -> None:
        """Scenario 8: Inherited initial_tool_calls does not falsely raise ResumableProviderError."""
        primary_client, _ = _make_mock_client([_internal_server_error()])
        fallback_client, fallback_sent = _make_mock_client(
            [
                _completion(
                    content="Fallback restarted.",
                    calls=[],
                    finish_reason="stop",
                )
            ]
        )

        def mock_get_client(service: LLMService) -> tuple[Any, str]:
            return (
                primary_client if service.api_key == "sk-primary" else fallback_client
            ), "OpenAI"

        cid_primary = str(uuid.uuid4())
        cid_fallback = str(uuid.uuid4())
        node_data = {
            "model": "primary-model",
            "credentialId": cid_primary,
            "fallbackModel": "fallback-model",
            "fallbackCredentialId": cid_fallback,
            "tools": TOOLS,
            "userMessage": "Resume work",
        }
        ex = self._make_executor(node_data)
        # Simulate HITL resume state containing an earlier tool call
        ex.hitl_resume_context = {
            "agent_state": {
                "tool_calls": [{"name": "prior_tool", "result": "ok"}],
                "messages": [{"role": "user", "content": "prior"}],
            }
        }

        cred_primary = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_primary")
        cred_fallback = MagicMock(type=CredentialType.openai, encrypted_config=b"enc_fallback")

        def mock_get_cred(_db: Any, cid: str) -> Any:
            return cred_primary if cid == cid_primary else cred_fallback

        def mock_decrypt(cfg: bytes) -> dict[str, Any]:
            return {
                "api_key": "sk-primary" if cfg == b"enc_primary" else "sk-fallback",
                "base_url": "https://api.openai.com/v1",
            }

        with (
            patch("app.db.session.SessionLocal"),
            patch.object(ex, "_get_accessible_credential", side_effect=mock_get_cred),
            patch("app.services.encryption.decrypt_config", side_effect=mock_decrypt),
            patch.object(LLMService, "_get_client", autospec=True, side_effect=mock_get_client),
        ):
            result = ex._execute_agent_node("agent-node-1", {}, node_data)

        self.assertTrue(result.get("fallbackUsed"))
        self.assertEqual(result.get("text"), "Fallback restarted.")
