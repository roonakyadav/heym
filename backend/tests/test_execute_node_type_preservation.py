"""Tests for Execute Workflow node type preservation (Discussion #544).

Verifies that structured data (dicts, lists) and scalars (numbers, booleans, strings)
forwarded into sub-workflows via executeInputMappings and executeInput retain their
native Python types rather than being serialized to JSON strings.
"""

import unittest
import uuid

from app.services.workflow_executor import WorkflowExecutor


def _run_sub_workflow(
    child_nodes: list[dict],
    child_edges: list[dict],
    execute_data: dict,
    initial_body: dict,
) -> dict:
    """Execute parent -> ExecuteNode -> child workflow and return child subOutput result."""
    child_wf_id = str(uuid.uuid4())
    parent_wf_id = str(uuid.uuid4())
    merged_execute_data = dict(execute_data, label="callSub", executeWorkflowId=child_wf_id)
    parent_nodes = [
        {"id": "p1", "type": "textInput", "data": {"label": "parentInput"}},
        {"id": "p2", "type": "execute", "data": merged_execute_data},
        {
            "id": "p3",
            "type": "output",
            "data": {
                "label": "parentOutput",
                "outputSchema": [{"key": "subResult", "value": "$callSub.outputs"}],
            },
        },
    ]
    parent_edges = [
        {"id": "pe1", "source": "p1", "target": "p2"},
        {"id": "pe2", "source": "p2", "target": "p3"},
    ]
    executor = WorkflowExecutor(
        nodes=parent_nodes,
        edges=parent_edges,
        workflow_cache={child_wf_id: {"nodes": child_nodes, "edges": child_edges, "name": "Child"}},
        test_mode=False,
        workflow_id=uuid.UUID(parent_wf_id),
    )
    result = executor.execute(uuid.UUID(parent_wf_id), {"body": initial_body})
    assert result.status == "success"
    outputs = result.outputs.get("parentOutput", {}).get("result", {}).get("subResult", {})
    return outputs.get("subOutput", {}).get("result", {})


class ExecuteNodeTypePreservationTests(unittest.TestCase):
    """End-to-end parent -> ExecuteNode -> child workflow type-preservation tests."""

    def test_execute_input_mappings_preserves_dict_and_nested_access(self) -> None:
        """executeInputMappings with a dict retains dict type and allows child property access."""
        child_nodes = [
            {
                "id": "c1",
                "type": "textInput",
                "data": {"label": "subInput", "inputFields": [{"key": "user"}]},
            },
            {
                "id": "c2",
                "type": "output",
                "data": {
                    "label": "subOutput",
                    "outputSchema": [
                        {"key": "userName", "value": "$subInput.user.name"},
                        {"key": "userRole", "value": "$subInput.user.role"},
                    ],
                },
            },
        ]
        child_edges = [{"id": "ce1", "source": "c1", "target": "c2"}]
        res = _run_sub_workflow(
            child_nodes,
            child_edges,
            execute_data={"executeInputMappings": [{"key": "user", "value": "$parentInput.user"}]},
            initial_body={"user": {"name": "Alice", "role": "admin"}},
        )
        self.assertEqual(res["userName"], "Alice")
        self.assertEqual(res["userRole"], "admin")

    def test_execute_input_mappings_preserves_list_indexing_and_loop(self) -> None:
        """executeInputMappings with a list preserves list type, indexing, and child loop iteration."""
        child_nodes = [
            {
                "id": "c1",
                "type": "textInput",
                "data": {"label": "subInput", "inputFields": [{"key": "items"}]},
            },
            {
                "id": "c_loop",
                "type": "loop",
                "data": {"label": "itemLoop", "arrayExpression": "$subInput.items"},
            },
            {
                "id": "c_out",
                "type": "output",
                "data": {
                    "label": "subOutput",
                    "outputSchema": [
                        {"key": "firstItem", "value": "$subInput.items[0]"},
                        {"key": "loopTotal", "value": "$itemLoop.total"},
                    ],
                },
            },
        ]
        child_edges = [
            {"id": "ce1", "source": "c1", "target": "c_loop"},
            {"id": "ce2", "source": "c_loop", "target": "c_out"},
        ]
        res = _run_sub_workflow(
            child_nodes,
            child_edges,
            execute_data={
                "executeInputMappings": [{"key": "items", "value": "$parentInput.items"}]
            },
            initial_body={"items": ["apple", "banana", "cherry"]},
        )
        self.assertEqual(res["firstItem"], "apple")
        self.assertEqual(res["loopTotal"], 3)

    def test_execute_input_mappings_preserves_scalar_types(self) -> None:
        """executeInputMappings preserves scalar types: int, bool, float, None, and string."""
        child_nodes = [
            {
                "id": "c1",
                "type": "textInput",
                "data": {
                    "label": "subInput",
                    "inputFields": [
                        {"key": "count"},
                        {"key": "active"},
                        {"key": "score"},
                        {"key": "empty"},
                        {"key": "title"},
                    ],
                },
            },
            {
                "id": "c2",
                "type": "output",
                "data": {
                    "label": "subOutput",
                    "outputSchema": [
                        {"key": "count", "value": "$subInput.count"},
                        {"key": "active", "value": "$subInput.active"},
                        {"key": "score", "value": "$subInput.score"},
                        {"key": "empty", "value": "$subInput.empty"},
                        {"key": "title", "value": "$subInput.title"},
                    ],
                },
            },
        ]
        child_edges = [{"id": "ce1", "source": "c1", "target": "c2"}]
        res = _run_sub_workflow(
            child_nodes,
            child_edges,
            execute_data={
                "executeInputMappings": [
                    {"key": "count", "value": "$parentInput.count"},
                    {"key": "active", "value": "$parentInput.active"},
                    {"key": "score", "value": "$parentInput.score"},
                    {"key": "empty", "value": "$parentInput.empty"},
                    {"key": "title", "value": "$parentInput.title"},
                ]
            },
            initial_body={
                "count": 42,
                "active": True,
                "score": 3.14,
                "empty": None,
                "title": "  spaced title  ",
            },
        )
        self.assertIs(type(res["count"]), int)
        self.assertEqual(res["count"], 42)
        self.assertIs(type(res["active"]), bool)
        self.assertTrue(res["active"])
        self.assertIs(type(res["score"]), float)
        self.assertEqual(res["score"], 3.14)
        self.assertIsNone(res["empty"])
        self.assertIs(type(res["title"]), str)
        self.assertEqual(res["title"], "  spaced title  ")

    def test_execute_input_mappings_expression_and_template_routing(self) -> None:
        """executeInputMappings routes $ expressions to evaluator and non-$ strings to template."""
        child_nodes = [
            {
                "id": "c1",
                "type": "textInput",
                "data": {
                    "label": "subInput",
                    "inputFields": [
                        {"key": "sum"},
                        {"key": "cmp"},
                        {"key": "num"},
                        {"key": "strNum"},
                        {"key": "greeting"},
                        {"key": "spacedTitle"},
                        {"key": "spacedCount"},
                    ],
                },
            },
            {
                "id": "c2",
                "type": "output",
                "data": {
                    "label": "subOutput",
                    "outputSchema": [
                        {"key": "sum", "value": "$subInput.sum"},
                        {"key": "cmp", "value": "$subInput.cmp"},
                        {"key": "num", "value": "$subInput.num"},
                        {"key": "strNum", "value": "$subInput.strNum"},
                        {"key": "greeting", "value": "$subInput.greeting"},
                        {"key": "spacedTitle", "value": "$subInput.spacedTitle"},
                        {"key": "spacedCount", "value": "$subInput.spacedCount"},
                    ],
                },
            },
        ]
        child_edges = [{"id": "ce1", "source": "c1", "target": "c2"}]
        res = _run_sub_workflow(
            child_nodes,
            child_edges,
            execute_data={
                "executeInputMappings": [
                    {"key": "sum", "value": "$parentInput.count + $parentInput.other"},
                    {"key": "cmp", "value": "$parentInput.count > $parentInput.other"},
                    {"key": "num", "value": "$42"},
                    {"key": "strNum", "value": "$'42'"},
                    {"key": "greeting", "value": "hello world"},
                    {"key": "spacedTitle", "value": " $parentInput.title "},
                    {"key": "spacedCount", "value": " $parentInput.count "},
                ]
            },
            initial_body={"count": 42, "other": 8, "title": "Report"},
        )
        self.assertIs(type(res["sum"]), int)
        self.assertEqual(res["sum"], 50)
        self.assertIs(type(res["cmp"]), bool)
        self.assertTrue(res["cmp"])
        self.assertIs(type(res["num"]), int)
        self.assertEqual(res["num"], 42)
        self.assertIs(type(res["strNum"]), str)
        self.assertEqual(res["strNum"], "42")
        self.assertIs(type(res["greeting"]), str)
        self.assertEqual(res["greeting"], "hello world")
        self.assertIs(type(res["spacedTitle"]), str)
        self.assertEqual(res["spacedTitle"], " Report ")
        self.assertIs(type(res["spacedCount"]), str)
        self.assertEqual(res["spacedCount"], " 42 ")

    def test_execute_input_template_preserves_dict(self) -> None:
        """executeInput template with a dict expression passes dict directly to child workflow."""
        child_nodes = [
            {"id": "c1", "type": "textInput", "data": {"label": "subInput"}},
            {
                "id": "c2",
                "type": "output",
                "data": {
                    "label": "subOutput",
                    "outputSchema": [
                        {"key": "userName", "value": "$subInput.name"},
                        {"key": "userId", "value": "$subInput.id"},
                    ],
                },
            },
        ]
        child_edges = [{"id": "ce1", "source": "c1", "target": "c2"}]
        res = _run_sub_workflow(
            child_nodes,
            child_edges,
            execute_data={"executeInput": "$parentInput.user"},
            initial_body={"user": {"name": "Alice", "id": 100}},
        )
        self.assertEqual(res["userName"], "Alice")
        self.assertEqual(res["userId"], 100)

    def test_execute_input_template_expression_and_template_routing(self) -> None:
        """executeInput routes $ expressions to evaluator and non-$ strings to template."""
        scalar_child_nodes = [
            {"id": "c1", "type": "textInput", "data": {"label": "subInput"}},
            {
                "id": "c2",
                "type": "output",
                "data": {
                    "label": "subOutput",
                    "outputSchema": [{"key": "val", "value": "$subInput.value"}],
                },
            },
        ]
        scalar_edges = [{"id": "ce1", "source": "c1", "target": "c2"}]

        text_child_nodes = [
            {"id": "c1", "type": "textInput", "data": {"label": "subInput"}},
            {
                "id": "c2",
                "type": "output",
                "data": {
                    "label": "subOutput",
                    "outputSchema": [{"key": "txt", "value": "$subInput.text"}],
                },
            },
        ]
        text_edges = [{"id": "ce1", "source": "c1", "target": "c2"}]

        # Arithmetic expression -> evaluated as int
        res_sum = _run_sub_workflow(
            scalar_child_nodes,
            scalar_edges,
            execute_data={"executeInput": "$parentInput.count + $parentInput.other"},
            initial_body={"count": 42, "other": 8},
        )
        self.assertIs(type(res_sum["val"]), int)
        self.assertEqual(res_sum["val"], 50)

        # Comparison expression -> evaluated as bool
        res_cmp = _run_sub_workflow(
            scalar_child_nodes,
            scalar_edges,
            execute_data={"executeInput": "$parentInput.count > $parentInput.other"},
            initial_body={"count": 42, "other": 8},
        )
        self.assertIs(type(res_cmp["val"]), bool)
        self.assertTrue(res_cmp["val"])

        # Numeric literal expression $42 -> evaluated as int
        res_num = _run_sub_workflow(
            scalar_child_nodes,
            scalar_edges,
            execute_data={"executeInput": "$42"},
            initial_body={},
        )
        self.assertIs(type(res_num["val"]), int)
        self.assertEqual(res_num["val"], 42)

        # Scalar int expression -> evaluated as int
        res_count = _run_sub_workflow(
            scalar_child_nodes,
            scalar_edges,
            execute_data={"executeInput": "$parentInput.count"},
            initial_body={"count": 42},
        )
        self.assertIs(type(res_count["val"]), int)
        self.assertEqual(res_count["val"], 42)

        # String literal expression -> evaluated as str
        res_str = _run_sub_workflow(
            text_child_nodes,
            text_edges,
            execute_data={"executeInput": "$'42'"},
            initial_body={},
        )
        self.assertIs(type(res_str["txt"]), str)
        self.assertEqual(res_str["txt"], "42")

        # Non-$ string template -> evaluated as str template
        res_nondollar = _run_sub_workflow(
            text_child_nodes,
            text_edges,
            execute_data={"executeInput": "hello sub-workflow"},
            initial_body={},
        )
        self.assertIs(type(res_nondollar["txt"]), str)
        self.assertEqual(res_nondollar["txt"], "hello sub-workflow")

        # Leading-whitespace template -> stays on text template path (not stripped into expression)
        res_ws_title = _run_sub_workflow(
            text_child_nodes,
            text_edges,
            execute_data={"executeInput": " $parentInput.title "},
            initial_body={"title": "Report"},
        )
        self.assertIs(type(res_ws_title["txt"]), str)
        self.assertEqual(res_ws_title["txt"], " Report ")

        res_ws_count = _run_sub_workflow(
            text_child_nodes,
            text_edges,
            execute_data={"executeInput": " $parentInput.count "},
            initial_body={"count": 42},
        )
        self.assertIs(type(res_ws_count["txt"]), str)
        self.assertEqual(res_ws_count["txt"], " 42 ")

    def test_whole_object_rendering_into_text(self) -> None:
        """Document expected Python-style dict string representation when a native dict is stringified.

        As noted in Discussion #544 guidance, when a native dict is passed to a child workflow
        and subsequently stringified in Python expression contexts (e.g. str(subInput.user)),
        it renders in Python dict style with single quotes (e.g. {'name': 'Alice'}) rather than JSON style.
        """
        child_nodes = [
            {
                "id": "c1",
                "type": "textInput",
                "data": {"label": "subInput", "inputFields": [{"key": "user"}]},
            },
            {
                "id": "c2",
                "type": "output",
                "data": {
                    "label": "subOutput",
                    "outputSchema": [
                        {"key": "userAsText", "value": "$'User: ' + str(subInput.user)"},
                    ],
                },
            },
        ]
        child_edges = [{"id": "ce1", "source": "c1", "target": "c2"}]
        res = _run_sub_workflow(
            child_nodes,
            child_edges,
            execute_data={"executeInputMappings": [{"key": "user", "value": "$parentInput.user"}]},
            initial_body={"user": {"name": "Alice"}},
        )
        self.assertEqual(res["userAsText"], "User: {'name': 'Alice'}")


if __name__ == "__main__":
    unittest.main()
