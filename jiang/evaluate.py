"""Invoke the installed official harness using the saved task IDs and patches."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .benchmark import read_tasks, atomic_json
from .testing import docker_ready


def summarize(output, ids):
    resolved, failed, missing = [], [], []
    reports = {}
    for path in (output / "logs" / "run_evaluation").rglob("report.json"):
        for iid, report in json.loads(path.read_text()).items():
            if iid in ids:
                reports[iid] = report
    for iid in ids:
        report = reports.get(iid)
        if report is None:
            missing.append(iid)
        elif report.get("resolved") is True:
            resolved.append(iid)
        else:
            failed.append(iid)
    summary = {"attempted": len(ids), "resolved": resolved, "not_resolved": failed,
               "missing_reports": missing, "resolved_fraction": len(resolved)/len(ids) if ids else 0}
    atomic_json(output / "official_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--dataset-file", type=Path, help="Full original task snapshot with evaluation fields")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--instance-id", help="Grade only this saved prediction; keep a cumulative summary")
    args = parser.parse_args()
    if args.instance_id and (not re.fullmatch(r"[\w.-]+", args.instance_id) or args.instance_id in (".", "..")):
        parser.error("Unsafe instance ID")
    root = args.run.resolve()
    predictions = root / "predictions.jsonl"
    rows = read_tasks(predictions)
    all_ids = [r["instance_id"] for r in rows]
    if args.instance_id and args.instance_id not in all_ids:
        parser.error("Requested instance has no saved prediction")
    ids = [args.instance_id] if args.instance_id else all_ids
    if not ids:
        parser.error("No completed predictions")
    grading = root / "evaluation"
    grading.mkdir(exist_ok=True)
    if args.instance_id:
        # Preserve each issue's grading inputs without re-grading old images.
        inputs = grading / "inputs" / args.instance_id
        inputs.mkdir(parents=True, exist_ok=True)
        predictions = inputs / "predictions.jsonl"
        predictions.write_text("".join(json.dumps(r) + "\n" for r in rows if r["instance_id"] in ids))
    else:
        inputs = grading
    if not args.summary_only:
        docker_ready()
        if args.dataset_file:
            tasks = read_tasks(args.dataset_file)
        else:
            from datasets import load_dataset
            tasks = list(load_dataset("SWE-bench/SWE-bench_Verified", split="test"))
        tasks = [t for t in tasks if t["instance_id"] in ids]
        snapshots = {t["instance_id"]: t for t in read_tasks(root / "tasks.jsonl")}
        if {t["instance_id"] for t in tasks} != set(ids):
            parser.error("Evaluation dataset is missing a prediction ID")
        for task in tasks:
            expected = snapshots[task["instance_id"]]
            if task["base_commit"] != expected["base_commit"] or task["problem_statement"] != expected["problem_statement"]:
                parser.error("Evaluation dataset differs from inference snapshot")
            if "eval_script" not in task or "image" not in task:
                parser.error("Use the FULL dataset snapshot, not the sanitized inference tasks.jsonl")
        dataset = inputs / "dataset.json"
        atomic_json(dataset, tasks)
        result = subprocess.run([sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", str(dataset), "--predictions_path", str(predictions),
            "--run_id", "official", "--max_workers", "1", "--timeout", str(args.timeout),
            "--report_dir", str(inputs)], cwd=grading)
        summary = summarize(grading, all_ids)
        print(json.dumps(summary, indent=2))
        if result.returncode or set(ids).intersection(summary["missing_reports"]):
            raise SystemExit(result.returncode or 1)
    else:
        print(json.dumps(summarize(grading, all_ids), indent=2))


if __name__ == "__main__":
    main()
