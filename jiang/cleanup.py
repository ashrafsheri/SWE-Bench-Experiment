"""Grade saved attempts, then discard only their reproducible bulky assets."""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .benchmark import atomic_json, export, read_tasks
from .evaluate import summarize
from .progress import progress
from .workspace import command


def remove_assets(output, task):
    """Never prune Docker globally or follow a checkout symlink."""
    output = Path(output).resolve()
    iid = task["instance_id"]
    image = task.get("image", "")
    if not re.fullmatch(r"[\w.-]+", iid) or iid in (".", ".."):
        raise ValueError("Unsafe instance ID for cleanup")
    if not re.fullmatch(r"swebench/[a-z0-9][a-z0-9_.-]*(?::[a-zA-Z0-9_.-]+)?", image):
        raise ValueError("Cleanup requires an exact official swebench image tag")
    directory = output / "attempts" / iid
    work = directory / "repo"
    if (directory.is_symlink() or work.is_symlink() or
            directory.resolve().parent != (output / "attempts").resolve() or
            (output / "attempts").is_symlink()):
        raise ValueError("Refusing cleanup outside the run's attempt directory")
    if not (directory / "result.json").is_file() or not (directory / "candidate.patch").is_file():
        raise ValueError("Save result.json and candidate.patch before cleanup")
    before = shutil.disk_usage(output).free
    listed = command(["docker", "image", "ls", "--quiet", "--no-trunc", image])
    if listed.returncode:
        raise RuntimeError(listed.stderr)
    image_output = "Image already absent."
    if listed.stdout.strip():
        # No --force: do not remove images used by other containers.
        removed = command(["docker", "image", "rm", image], timeout=180)
        image_output = removed.stdout + removed.stderr
        if removed.returncode:
            raise RuntimeError(image_output)
    if work.exists():
        shutil.rmtree(work)
    return {"status": "complete", "removed_checkout": str(work), "image": image,
            "image_output": image_output, "disk_free_before_bytes": before,
            "disk_free_after_bytes": shutil.disk_usage(output).free}


def finish_attempt(output, task, tasks, timeout=1800):
    """Called only after generation has stopped and all artifacts are exported."""
    directory = output / "attempts" / task["instance_id"]
    cleanup_path = directory / "cleanup.json"
    record = json.loads((directory / "result.json").read_text())
    if record.get("status") == "interrupted":
        return
    # Preparation errors may have no checkout or patch; preserve an explicit empty patch.
    patch = directory / "candidate.patch"
    if not patch.exists():
        patch.write_text("")
    export(output, tasks)
    previous = record.get("evaluation", {})
    if (previous.get("status") in ("complete", "no_patch") and cleanup_path.exists()
            and json.loads(cleanup_path.read_text()).get("status") == "complete"):
        return
    if previous.get("status") not in ("complete", "no_patch"):
        started = time.monotonic()
        if not patch.read_text().strip():
            evaluation = {"status": "no_patch", "reason": "No candidate to grade"}
        else:
            progress(f"  grading {task['instance_id']} before cleanup")
            log = directory / "evaluation.log"
            try:
                with log.open("a") as stream:
                    result = subprocess.run([sys.executable, "-u", "-m", "jiang.evaluate",
                        "--run", str(output), "--instance-id", task["instance_id"],
                        "--timeout", str(timeout)], stdout=stream, stderr=subprocess.STDOUT)
                evaluation = {"status": "complete" if result.returncode == 0 else "error",
                              "exit_code": result.returncode, "log": str(log)}
            except OSError as exc:
                evaluation = {"status": "error", "error": str(exc), "log": str(log)}
        evaluation["seconds"] = round(time.monotonic() - started, 2)
        record["evaluation"] = evaluation
        grading = output / "evaluation"
        grading.mkdir(exist_ok=True)
        ids = [r["instance_id"] for r in read_tasks(output / "predictions.jsonl")]
        summary = summarize(grading, ids)
        record["official_resolved"] = (True if task["instance_id"] in summary["resolved"] else
                                       False if task["instance_id"] in summary["not_resolved"] else None)
        atomic_json(directory / "result.json", record)
        export(output, tasks)
    # Retry failed cleanup on resume; failed grading is also retried without re-running the model.
    progress(f"  cleaning checkout and test image for {task['instance_id']}")
    try:
        cleanup = remove_assets(output, task)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        atomic_json(cleanup_path, {"status": "error", "error": str(exc)})
        raise RuntimeError(f"Cleanup failed for {task['instance_id']}; stopping to avoid disk growth: {exc}") from exc
    atomic_json(cleanup_path, cleanup)
