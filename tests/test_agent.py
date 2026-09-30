import json
import asyncio
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from jiang.agent import OllamaAgent
from jiang.benchmark import choose_tasks, export, atomic_json, read_priority_ids
from jiang.evaluate import summarize
from jiang.workspace import Workspace, command
from jiang.local_proxy import constrain


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        command(["git", "init", "-q", str(self.root)])
        (self.root / "example.py").write_text("def add(a, b):\n    return a - b\n")
        (self.root / "test_example.py").write_text("assert True\n")
        command(["git", "add", "."], self.root)
        result = command(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                          "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.ws = Workspace(self.root)

    def read_source(self):
        return self.ws.act({"action": "read", "path": "example.py"})

    def test_exact_edit_generates_applicable_diff(self):
        self.read_source()
        self.ws.act({"action": "edit", "path": "example.py", "old": "return a - b", "new": "return a + b"})
        patch = self.ws.patch()
        self.assertIn("+    return a + b", patch)
        patchfile = self.root / "change.patch"
        patchfile.write_text(patch)
        result = command(["git", "apply", "--check", "--reverse", str(patchfile)], self.root)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_edits_leave_source_unchanged(self):
        edit = {"action": "edit", "path": "example.py", "old": "return a - b", "new": "return ("}
        with self.assertRaises(ValueError):
            self.ws.act(edit)
        original = (self.root / "example.py").read_text()
        self.read_source()
        with self.assertRaises(ValueError):
            self.ws.act(edit)
        with self.assertRaises(ValueError):
            self.ws.act(dict(edit, old="not present"))
        self.assertEqual((self.root / "example.py").read_text(), original)

    def test_read_boundaries_and_test_protection(self):
        for name in ("../secret", ".git/config", "/etc/passwd"):
            with self.assertRaises(ValueError):
                self.ws.act({"action": "read", "path": name})
        self.ws.act({"action": "read", "path": "test_example.py"})
        with self.assertRaises(ValueError):
            self.ws.act({"action": "edit", "path": "test_example.py", "old": "True", "new": "False"})

    def test_large_read_is_paginated_instead_of_wasting_model_turn(self):
        (self.root / "example.py").write_text("x = 1\n" * 500)
        result = self.ws.act({"action": "read", "path": "example.py", "start": 1, "end": 500})
        self.assertIn("next page starts at 241", result)
        self.assertNotIn("241: x", result)
        result = self.ws.act({"action": "read", "path": "example.py", "start": 241})
        self.assertIn("241: x", result)

    def test_search_honors_explicit_file_scope(self):
        result = self.ws.act({"action": "search", "query": "return", "path": "example.py"})
        self.assertIn("example.py", result)
        result = self.ws.act({"action": "search", "query": "return", "path": "test_example.py"})
        self.assertIn("No matches", result)

    def test_baseline_guard_and_clear_test_edit_error(self):
        self.ws.test_runner = lambda cmd, patch: {"exit_code": 1, "output": "baseline"}
        self.read_source()
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.ws.act({"action": "edit", "path": "example.py", "old": "a - b", "new": "a + b"})
        self.ws.act({"action": "read", "path": "test_example.py"})
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.ws.act({"action": "edit", "path": "test_example.py", "old": "True", "new": "False"})
        self.ws.act({"action": "test", "command": "baseline"})
        self.ws.act({"action": "edit", "path": "example.py", "old": "a - b", "new": "a + b"})
        self.assertIn("a + b", self.ws.patch())

    def test_full_conversation_receives_failed_test_and_repairs(self):
        actions = iter([
            {"action": "read", "path": "example.py"},
            {"action": "test", "command": "test"},
            {"action": "edit", "path": "example.py", "old": "return a - b", "new": "return a + b"},
            {"action": "test", "command": "test"},
            {"action": "diff"},
            {"action": "finish", "summary": "fixed"},
        ])
        received = []
        agent = OllamaAgent("fixture")
        def chat(messages):
            received.append(json.loads(json.dumps(messages)))
            return {"message": {"content": json.dumps(next(actions))}}
        agent.chat = chat
        self.ws.test_runner = lambda cmd, patch: {"exit_code": 0 if patch else 1, "output": "failure" if not patch else "pass"}
        result = agent.solve("add returns wrong result", self.ws, self.root / "trace.jsonl", max_steps=8)
        self.assertEqual(result["completion"], "finished")
        self.assertEqual([t["exit_code"] for t in result["tests"]], [1, 0])
        self.assertIn("failure", received[2][-1]["content"])
        self.assertEqual(received[2][0]["role"], "system")

    def test_python_reproduction_spelling_is_bounded_test_action(self):
        actions = iter([
            {"action": "read", "path": "example.py"},
            {"action": "python", "command": "python -c 'print(1)'"},
            {"action": "edit", "path": "example.py", "old": "return a - b", "new": "return a + b"},
            {"action": "python", "command": "python -c 'assert 1'"},
            {"action": "diff"},
            {"action": "finish", "summary": "reproduced"},
        ])
        agent = OllamaAgent("fixture")
        agent.chat = lambda messages: {"message": {"content": json.dumps(next(actions))}}
        self.ws.test_runner = lambda cmd, patch: {"command": cmd, "exit_code": 1, "output": cmd}
        result = agent.solve("wrong addition", self.ws, self.root / "trace-python.jsonl", max_steps=6)
        self.assertEqual(result["completion"], "finished")
        self.assertEqual(result["tests"][0]["command"], "python -c 'print(1)'")

    def test_generation_does_not_imply_official_resolution(self):
        output = self.root / "run"
        directory = output / "attempts" / "test-1"
        directory.mkdir(parents=True)
        atomic_json(directory / "result.json", {"instance_id": "test-1", "model": "fixture", "status": "patch_generated"})
        (directory / "candidate.patch").write_text("patch")
        export(output, [{"instance_id": "test-1"}])
        summary = summarize(output, ["test-1", "test-2"])
        self.assertEqual(summary["resolved"], [])
        self.assertEqual(len(summary["missing_reports"]), 2)
        reports = output / "logs/run_evaluation/official/model/test-1"
        reports.mkdir(parents=True)
        atomic_json(reports / "report.json", {"test-1": {"resolved": True}})
        self.assertEqual(summarize(output, ["test-1", "test-2"])["resolved"], ["test-1"])

    @unittest.skipUnless(importlib.util.find_spec("mcp"), "optional Claude backend not installed")
    def test_claude_mcp_transport_uses_bounded_workspace_tools(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        directory = self.root / "logs"
        directory.mkdir()
        config = directory / "config.json"
        config.write_text(json.dumps({"repo": str(self.root), "directory": str(directory),
            # Test tools are enabled, but this transport test never starts Docker.
            "tests": "docker", "test_timeout": 30, "task": {}, "max_steps": 6}))
        async def check():
            spec = StdioServerParameters(command=sys.executable, args=["-m", "jiang.mcp_tools", str(config)])
            async with stdio_client(spec) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    names = [t.name for t in (await client.list_tools()).tools]
                    self.assertEqual(names, ["repository_action"])
                    result = await client.call_tool("repository_action", {"action": "read", "path": "example.py"})
                    self.assertIn("return a - b", str(result))
                    result = await client.call_tool("repository_action", {"action": "read", "path": "../secret"})
                    self.assertIn("Action error", str(result))
                    for _ in range(3):
                        result = await client.call_tool("repository_action", {"action": "search", "query": "add"})
                    self.assertIn("Next, use action='test'", str(result))
        asyncio.run(check())
        state = json.loads((directory / "mcp_state.json").read_text())
        self.assertEqual(state["steps"], 5)


class ConfigurationTests(unittest.TestCase):
    def test_native_requests_use_actual_local_generation_settings(self):
        request = {"model": "some-default", "thinking": {"type": "enabled"},
                   "temperature": 1, "max_tokens": 32000, "output_config": {"effort": "high"}}
        actual = constrain(request, "local-qwen", {"temperature": 0.1, "num_predict": 4096}, False)
        self.assertEqual(actual["model"], "local-qwen")
        self.assertEqual(actual["thinking"], {"type": "disabled"})
        self.assertEqual(actual["max_tokens"], 4096)
        self.assertEqual(actual["temperature"], 0.1)
        self.assertNotIn("output_config", actual)

    def test_reject_cloud_and_remote(self):
        for model, url in (("foo:cloud", "http://localhost:11434"), ("foo", "https://example.com")):
            with self.assertRaises(ValueError):
                OllamaAgent(model, url)

    def test_selection_fills_beyond_distinct_repos(self):
        rows = [{"instance_id": str(i), "repo": str(i % 2)} for i in range(5)]
        self.assertEqual(len(choose_tasks(rows, 5)), 5)
        with self.assertRaises(ValueError):
            choose_tasks(rows, 6)

    def test_selection_prioritizes_easy_reproducible_tasks(self):
        rows = [
            {"instance_id": "hard", "repo": "r/h", "difficulty": "1-4 hours",
             "problem_statement": "large investigation"},
            {"instance_id": "medium", "repo": "r/m", "difficulty": "15 min - 1 hour",
             "problem_statement": "expected behavior"},
            {"instance_id": "easy", "repo": "r/e", "difficulty": "<15 min fix",
             "problem_statement": "```python\nprint(1)\n``` Expected output."},
        ]
        selected = choose_tasks(rows, 3)
        self.assertEqual([row["instance_id"] for row in selected], ["easy", "medium", "hard"])
        balanced = choose_tasks(rows, 3, "balanced")
        self.assertEqual([row["instance_id"] for row in balanced], ["easy", "hard", "medium"])

    def test_preferred_ids_preserve_exact_user_order(self):
        rows = [{"instance_id": iid, "repo": "django/django", "difficulty": "1-4 hours"}
                for iid in ("a", "b", "c")]
        self.assertEqual([r["instance_id"] for r in choose_tasks(rows, 2, preferred_ids=["c", "a"])], ["c", "a"])
        with self.assertRaisesRegex(ValueError, "missing"):
            choose_tasks(rows, 1, preferred_ids=["missing"])
        with tempfile.TemporaryDirectory() as td:
            priority = Path(td) / "priority.txt"
            priority.write_text("a\na\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_priority_ids(priority)


if __name__ == "__main__":
    unittest.main()
