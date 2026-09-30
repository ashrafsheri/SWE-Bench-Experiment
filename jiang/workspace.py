"""Bounded repository tools; no model-provided commands run on the host."""
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path


def command(args, cwd=None, timeout=120):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)


class Workspace:
    def __init__(self, root: Path, test_runner=None):
        self.root = root.resolve()
        self.test_runner = test_runner
        self.files = command(["git", "ls-files", "-z"], self.root).stdout.rstrip("\0").split("\0")
        self.seen = set()
        self.tests = []
        self.last_review = self.last_test_patch = None

    def path(self, name):
        if name not in self.files or ".git" in Path(name).parts:
            raise ValueError("Use a tracked file path from list/search")
        path = self.root / name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Symlink/out-of-repository path rejected")
        if path.stat().st_size > 1_000_000:
            raise ValueError("File exceeds 1 MB tool limit")
        return path

    def patch(self):
        result = command(["git", "diff", "--binary", "HEAD", "--"], self.root)
        if result.returncode:
            raise ValueError(result.stderr)
        return result.stdout

    def orientation(self, issue):
        # Ranking uses issue terms and source text, never reference patches/tests.
        words = list(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z_0-9]{3,}", issue)))
        words = [w for w in words if "_" in w or any(c.isupper() for c in w)][:24]
        scored = []
        for name in self.files:
            if not name.endswith(".py"):
                continue
            try:
                source = self.path(name).read_text(errors="replace")
            except (OSError, ValueError):
                continue
            score = sum(min(source.count(w), 3) + 5 * (w.lower() in name.lower()) for w in words)
            if score:
                scored.append((score, name))
        ranked = sorted(scored, key=lambda x: (-x[0], x[1]))[:15]
        top = sorted({f.split('/')[0] for f in self.files})
        return ("TOP LEVEL:\n" + "\n".join(top[:70]) + "\nPOSSIBLY RELEVANT FILES:\n" +
                "\n".join(n for _, n in ranked) + "\nUse search/read to inspect code. No file is pre-read.")

    def act(self, action):
        op = action["action"]
        if op == "list":
            return "\n".join(f for f in self.files if action.get("query", "") in f)[:10000]
        if op == "search":
            query = action["query"]
            if not query or len(query) > 200:
                raise ValueError("Search requires 1–200 characters of literal text")
            scope = action.get("path") or "*.py"
            if scope != "*.py" and scope not in self.files and not any(
                    f.startswith(scope.rstrip("/") + "/") for f in self.files):
                raise ValueError("Search path must name a tracked file or directory")
            result = command(["git", "grep", "-n", "-I", "-F", "-e", query, "--", scope], self.root)
            return result.stdout[:10000] or "No matches. Try a shorter symbol or list file paths."
        if op == "read":
            path = self.path(action["path"])
            start = int(action.get("start", 1))
            end = int(action.get("end") or start + 159)
            if start < 1 or end < start:
                raise ValueError("Use a positive start line and end >= start")
            end = min(end, start + 239)
            lines = path.read_text().splitlines()
            self.seen.add(action["path"])
            rendered = []
            for i in range(start-1, min(end, len(lines))):
                item = f"{i+1}: {lines[i]}"
                if sum(len(s)+1 for s in rendered) + len(item) > 8500:
                    if not rendered:
                        rendered.append(item[:8500] + " [line truncated]")
                    break
                rendered.append(item)
            next_line = start + len(rendered)
            return (f"{action['path']} ({len(lines)} lines, next page starts at {next_line}):\n" +
                    "\n".join(rendered))
        if op == "edit":
            name = action["path"]
            path = self.path(name)
            if name not in self.seen:
                raise ValueError("Read this file before editing it")
            if any(p in ("tests", "test", "testing") for p in Path(name).parts) or path.name.startswith("test_"):
                raise ValueError("Benchmark test files are read-only; use a test command for reproduction")
            if self.test_runner is not None and not self.tests:
                raise ValueError("Run a focused baseline test or issue reproduction before the first edit")
            old, new = action["old"], action["new"]
            source = path.read_text()
            if not old or source.count(old) != 1:
                raise ValueError(f"old must occur exactly once; found {source.count(old) if old else 0}. Read the file again.")
            changed = source.replace(old, new, 1)
            if path.suffix == ".py":
                try:
                    ast.parse(changed)
                except SyntaxError as exc:
                    raise ValueError(f"Edit rejected: Python syntax error: {exc}") from exc
            path.write_text(changed)
            return "Edit applied. Review diff and run a focused test."
        if op == "diff":
            self.last_review = self.patch()
            return self.last_review[:9500] or "No changes yet."
        if op == "test":
            if self.test_runner is None:
                return "Testing unavailable (--tests none). Review code and diff; do not claim tests passed."
            patch = self.patch()
            result = self.test_runner(action["command"], patch)
            self.last_test_patch = patch
            self.tests.append(result)
            return json.dumps(result)
        raise ValueError(f"Unknown action: {op}")
