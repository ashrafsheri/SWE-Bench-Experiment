"""Run local issue repair, preserving every attempt and separating official grading."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .agent import OllamaAgent
from .testing import DockerTests, docker_ready
from .workspace import Workspace, command
from .progress import progress

PUBLIC_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement", "difficulty", "image")


def read_tasks(path):
    text = Path(path).read_text()
    return json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]


def difficulty_rank(value):
    """Map SWE-bench's human effort label to a stable easiest-first rank."""
    label = str(value or "").lower().strip()
    if "<15" in label:
        return 0
    if "15 min" in label:
        return 1
    if "1-4" in label:
        return 2
    if "4+" in label or "4 hour" in label:
        return 3
    return 4


def task_priority(task):
    """Prefer easy, reproducible issues without inspecting reference answers."""
    statement = str(task.get("problem_statement") or "")
    has_reproduction = "```" in statement or "reproduction" in statement.lower()
    has_expected_behavior = any(word in statement.lower() for word in ("expected", "should", "actual"))
    # Difficulty is the main signal. The other signals only break ties: a short
    # issue with an explicit reproduction is generally cheaper for a local agent.
    return (difficulty_rank(task.get("difficulty")), -int(has_reproduction),
            -int(has_expected_behavior), len(statement), task["instance_id"])


def read_priority_ids(path):
    ids = [line.strip() for line in Path(path).read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate instance ID in priority file: {path}")
    return ids


def choose_tasks(rows, count, selection="easy", preferred_ids=None):
    if preferred_ids is not None:
        by_id = {row["instance_id"]: row for row in rows}
        missing = [iid for iid in preferred_ids if iid not in by_id]
        if missing:
            raise ValueError("Priority list contains IDs missing from the dataset: " + ", ".join(missing))
        if count > len(preferred_ids):
            raise ValueError(f"Priority list has {len(preferred_ids)} IDs, but count={count}")
        return [by_id[iid] for iid in preferred_ids[:count]]
    if selection not in ("easy", "balanced"):
        raise ValueError("selection must be 'easy' or 'balanced'")
    selected, seen = [], set()
    ordered = sorted(rows, key=task_priority if selection == "easy" else lambda t: t["instance_id"])
    # Take one task per repository first. This prevents a repository with many
    # easy issues from consuming the whole pilot and keeps setup failures isolated.
    for row in ordered:
        if row["repo"] not in seen:
            selected.append(row); seen.add(row["repo"])
        if len(selected) == count:
            return selected
    ids = {r["instance_id"] for r in selected}
    selected.extend(row for row in ordered if row["instance_id"] not in ids)
    if len(selected) < count:
        raise ValueError(f"Requested {count} tasks, only {len(selected)} available")
    return selected[:count]


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def checked(args, cwd=None, timeout=900):
    result = command(args, cwd, timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-5000:])
    return result.stdout


def prepare(task, work, reuse_from=None):
    if work.exists():
        raise ValueError(f"Checkout already exists: {work}; use --resume to skip completed tasks")
    repo = task["repo"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or not re.fullmatch(r"[0-9a-f]{40}", task["base_commit"]):
        raise ValueError("Invalid repository or base commit")
    source = reuse_from / task["instance_id"] if reuse_from else None
    if source and (source / ".git").exists():
        # Independent object copy: earlier runs can be archived without breaking this one.
        checked(["git", "clone", "--no-hardlinks", "--no-checkout", "--", str(source.resolve()), str(work)], timeout=1800)
    else:
        checked(["git", "clone", "--no-checkout", "--", f"https://github.com/{repo}.git", str(work)], timeout=1800)
    checked(["git", "checkout", "--detach", task["base_commit"]], work)
    if checked(["git", "rev-parse", "HEAD"], work).strip() != task["base_commit"]:
        raise RuntimeError("Wrong base commit")


def export(output, tasks):
    results, predictions = [], []
    for task in tasks:
        path = output / "attempts" / task["instance_id"] / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        results.append(result)
        patch = path.with_name("candidate.patch").read_text() if path.with_name("candidate.patch").exists() else ""
        predictions.append({"instance_id": task["instance_id"], "model_name_or_path": result["model"],
                            "model_patch": patch})
    for name, rows in (("results.jsonl", results), ("predictions.jsonl", predictions)):
        temp = output / (name + ".tmp")
        temp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        temp.replace(output / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--engine", choices=("jiang", "claude"), default="jiang")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, default=Path("runs/v2"))
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--reuse-repos", type=Path, help="Old run's worktrees directory (read only)")
    parser.add_argument("--context", type=int, help="Default: 16384 for Jiang, 32768 for Claude")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--tests", choices=("docker", "none"), default="docker")
    parser.add_argument("--test-timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, help="Run only the first N selected tasks as a pilot")
    parser.add_argument("--selection", choices=("easy", "balanced"), default="easy",
                        help="Task order: easiest metadata-ranked tasks (default) or instance ID order")
    parser.add_argument("--preferred-ids-file", type=Path,
                        help="Exact instance-ID order; overrides --selection and repository diversity")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cleanup-after-each", action="store_true",
                        help="Grade each issue, save all artifacts, then remove its checkout and Docker image")
    parser.add_argument("--evaluation-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.context is None:
        args.context = 32768 if args.engine == "claude" else 16384
    if args.count < 1 or args.max_steps < 1 or args.context < 8192 or (args.limit is not None and args.limit < 1):
        parser.error("count/steps/limit must be positive; context must be at least 8192")
    if args.evaluation_timeout < 1 or (args.cleanup_after_each and args.tests != "docker"):
        parser.error("evaluation-timeout must be positive; cleanup-after-each requires Docker tests")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".running"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        parser.error(f"Run already locked: {lock}. After confirming no process is running, remove this lock.")
    try:
        if args.engine == "claude":
            from .claude_agent import ClaudeAgent
            agent_class = ClaudeAgent
        else:
            agent_class = OllamaAgent
        agent = agent_class(args.model, args.endpoint, context=args.context,
                            temperature=args.temperature, seed=args.seed, think=args.think)
        model = agent.preflight()
        if args.tests == "docker":
            docker_ready()
        priority_ids = read_priority_ids(args.preferred_ids_file) if args.preferred_ids_file else None
        config = {k: getattr(args, k) for k in ("engine", "model", "endpoint", "context", "max_steps", "temperature", "seed", "think", "tests", "test_timeout", "count", "selection")}
        config["preferred_ids_file"] = str(args.preferred_ids_file.resolve()) if args.preferred_ids_file else None
        manifest = output / "manifest.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if not args.resume:
                raise ValueError("Output already contains a run; use --resume or a new output directory")
            if saved["config"] != config or saved["model"]["tag"]["digest"] != model["tag"]["digest"]:
                raise ValueError("Resume configuration or model digest differs; use a new output directory")
            tasks = read_tasks(output / "tasks.jsonl")
        else:
            if args.tasks_file:
                rows = read_tasks(args.tasks_file)
                tasks = choose_tasks(rows, args.count, args.selection, priority_ids)
                source = str(args.tasks_file.resolve())
            else:
                from datasets import load_dataset
                dataset = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
                tasks = choose_tasks(list(dataset), args.count, args.selection, priority_ids)
                source = f"SWE-bench/SWE-bench_Verified:test fingerprint={dataset._fingerprint}"
            tasks = [{k: t[k] for k in PUBLIC_FIELDS if k in t} for t in tasks]
            if len({t["instance_id"] for t in tasks}) != len(tasks):
                raise ValueError("Duplicate task IDs")
            task_text = "".join(json.dumps(t) + "\n" for t in tasks)
            (output / "tasks.jsonl").write_text(task_text)
            atomic_json(manifest, {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                "model": model, "dataset_source": source, "task_sha256": hashlib.sha256(task_text.encode()).hexdigest(),
                "platform": platform.platform(), "python": platform.python_version(),
                "versions": {p: importlib.metadata.version(p) for p in
                             (("requests", "datasets", "mcp") if args.engine == "claude" else ("requests", "datasets"))},
                "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.glob("*.py")}})
        if args.cleanup_after_each:
            from .cleanup import finish_attempt
            saved = json.loads(manifest.read_text())
            saved["storage_policy"] = {"cleanup_after_each": True,
                                       "evaluation_timeout": args.evaluation_timeout}
            atomic_json(manifest, saved)
        for index, task in enumerate(tasks[:args.limit] if args.limit else tasks, 1):
            iid = task["instance_id"]
            if not re.fullmatch(r"[\w.-]+", iid):
                raise ValueError("Unsafe instance ID")
            directory = output / "attempts" / iid
            saved_result = directory / "result.json"
            if saved_result.exists() and json.loads(saved_result.read_text()).get("status") != "interrupted":
                progress(f"[{index}/{len(tasks)}] {iid}: saved result, skipping")
                if args.cleanup_after_each:
                    finish_attempt(output, task, tasks, args.evaluation_timeout)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            work = directory / "repo"
            if work.exists():
                # Keep interrupted edits/logs. A restart gets a new checkout.
                archived = directory / f"interrupted-{time.time_ns()}"
                archived.mkdir()
                for child in list(directory.iterdir()):
                    if child != archived and not child.name.startswith("interrupted-"):
                        child.rename(archived / child.name)
            progress(f"[{index}/{len(tasks)}] {iid}: preparing checkout")
            started = time.monotonic()
            record = {"instance_id": iid, "model": args.model, "engine": args.engine, "status": "infrastructure_error",
                      "official_resolved": None, "started_at": datetime.now(timezone.utc).isoformat()}
            workspace = None
            try:
                prepare(task, work, args.reuse_repos)
                runner = DockerTests(task, directory / "tests", args.test_timeout) if args.tests == "docker" else None
                workspace = Workspace(work, runner)
                record.update(agent.solve(task["problem_statement"], workspace, directory / "trajectory.jsonl", args.max_steps))
                patch = workspace.patch()
                (directory / "candidate.patch").write_text(patch)
                # Reverse-check against modified checkout to confirm git-generated diff integrity.
                validation = command(["git", "apply", "--reverse", "--check", str(directory / "candidate.patch")], work) if patch else None
                record["status"] = ("patch_generated" if validation.returncode == 0 else "patch_invalid") if validation else "no_changes"
                if record["completion"] == "model_error":
                    record["status"] = "model_error"
                record["patch_bytes"] = len(patch.encode())
                if validation and validation.returncode:
                    record["validation_error"] = validation.stderr
            except (Exception, KeyboardInterrupt) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, KeyboardInterrupt):
                    record["status"] = "interrupted"
                    raise
            finally:
                if workspace is not None:
                    (directory / "candidate.patch").write_text(workspace.patch())
                    record["tests"] = workspace.tests
                record["seconds"] = round(time.monotonic()-started, 2)
                atomic_json(directory / "result.json", record)
                export(output, tasks)
            progress(f"[{index}/{len(tasks)}] {iid}: {record['status']} ({record['seconds']}s)")
            if args.cleanup_after_each:
                finish_attempt(output, task, tasks, args.evaluation_timeout)
        if args.cleanup_after_each:
            subprocess.run([sys.executable, "-m", "jiang.analyze", str(output)], check=True)
            progress(f"Saved patches, logs, cleanup records and grading reports in {output}")
        else:
            progress(f"Saved {output / 'predictions.jsonl'}. Official resolution requires separate evaluation.")
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
