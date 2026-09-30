"""Expose the SAME bounded tools to the downloaded Claude Code agent."""
import contextlib
import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .testing import DockerTests
from .workspace import Workspace


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    directory = Path(config["directory"])
    runner = DockerTests(config["task"], directory / "tests", config["test_timeout"]) if config["tests"] == "docker" else None
    workspace = Workspace(Path(config["repo"]), runner)
    server = FastMCP("jiang")
    count = 0

    @server.tool()
    def repository_action(action: str, query: str = "", path: str = "", start: int = 1,
                          end: int = 0, old: str = "", new: str = "", command: str = "") -> str:
        """Search literal source text; list paths; read paginated lines; edit exact unique text;
        diff changes; test with a shell command in Docker. Actions: search/list/read/edit/diff/test.
        Read files before editing. Run a baseline test before the first edit. Never edit tests.
        """
        nonlocal count
        count += 1
        arguments = dict(action=action, query=query, path=path, start=start, end=end, old=old, new=new, command=command)
        try:
            if count > config["max_steps"]:
                raise ValueError("Action budget exhausted. Finish now and report the observed results.")
            # stdout belongs exclusively to MCP's JSON-RPC transport.
            with contextlib.redirect_stdout(sys.stderr):
                result = workspace.act(arguments)
        except Exception as exc:
            result = f"Action error: {exc}"
        with (directory / "mcp_actions.jsonl").open("a") as trace:
            trace.write(json.dumps({"action": arguments, "observation": result}) + "\n")
        (directory / "mcp_state.json").write_text(json.dumps({"steps": count, "tests": workspace.tests}))
        remaining = max(0, config['max_steps'] - count)
        guidance = f"\nActions remaining: {remaining}"
        if count >= 5 and runner and not workspace.tests:
            guidance += ("\nNext, use action='test' with a short shell command to reproduce "
                         "the issue in Docker. Do not keep searching for an exact existing "
                         "regression test. A python assertion based on the issue is enough. "
                         "Test files are read-only; do not use edit to add a test.")
        if remaining <= 5 and workspace.patch():
            guidance += "\nBudget nearly exhausted: review diff and test the latest patch before finishing."
        return result[:10000] + guidance

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
