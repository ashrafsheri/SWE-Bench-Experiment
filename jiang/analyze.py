"""Summarize saved runs without confusing candidate patches with resolutions."""
import argparse
import json
from collections import Counter
from pathlib import Path

from .benchmark import read_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    records = read_tasks(root / "results.jsonl")
    official_path = root / "evaluation" / "official_summary.json"
    official = json.loads(official_path.read_text()) if official_path.exists() else {}
    lines = ["# Run analysis", "", f"Run: `{root.name}`", "",
             f"Recorded attempts: {len(records)}", "",
             f"Generation statuses: {dict(Counter(r['status'] for r in records))}", "",
             "Official grading: " + (f"{len(official['resolved'])}/{official['attempted']} resolved; "
                f"{len(official['missing_reports'])} missing reports." if official else "not run / no summary available."), "",
             "| Issue | Generation status | Agent steps | Local tests executed | Seconds | Official result |",
             "|---|---|---:|---:|---:|---|"]
    total_tokens = total_seconds = token_records = 0
    for record in records:
        iid = record["instance_id"]
        result = ("resolved" if iid in official.get("resolved", []) else
                  "not resolved" if iid in official.get("not_resolved", []) else "ungraded")
        seconds = record.get("seconds", 0)
        total_seconds += seconds
        lines.append(f"| {iid} | {record['status']} | {record.get('steps', '—')} | "
                     f"{len(record.get('tests', []))} | {seconds:.1f} | {result} |")
        trace = root / "attempts" / iid / "trajectory.jsonl"
        if trace.exists():
            for event in read_tasks(trace):
                if "eval_count" in event.get("response", {}):
                    total_tokens += event["response"]["eval_count"]
                    token_records += 1
                if event.get("type") == "result" and "output_tokens" in event.get("usage", {}):
                    total_tokens += event.get("usage", {}).get("output_tokens", 0)
                    token_records += 1
    lines.extend(["", f"Sum of task durations: {total_seconds/60:.1f} minutes.", "",
                  "Model output tokens reported in completed usage records: " +
                  (str(total_tokens) if token_records else "unavailable") + ". "
                  "Interrupted requests may have unreported usage.", "",
                  "A passing agent-selected test is not the official resolution metric. "
                  "Environment errors and missing grading reports are not successes.", ""])
    report = root / "analysis.md"
    report.write_text("\n".join(lines))
    print(report.read_text())


if __name__ == "__main__":
    main()
