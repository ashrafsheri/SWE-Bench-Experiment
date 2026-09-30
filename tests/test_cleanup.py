import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jiang.benchmark import atomic_json, export
from jiang.cleanup import finish_attempt, remove_assets
from jiang.evaluate import main as evaluate_main


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = {"instance_id": "test-1", "image": "swebench/test-1:latest"}
        self.directory = self.root / "attempts/test-1"
        (self.directory / "repo").mkdir(parents=True)
        (self.directory / "repo/source.py").write_text("large disposable checkout")
        atomic_json(self.directory / "result.json", {
            "instance_id": "test-1", "model": "fixture", "status": "patch_generated"})
        (self.directory / "candidate.patch").write_text("saved patch")
        (self.directory / "trajectory.jsonl").write_text("saved trace")
        (self.root / "unrelated").mkdir()
        (self.root / "unrelated/keep").write_text("keep")

    def report(self, iid, resolved=True):
        directory = self.root / "evaluation/logs/run_evaluation/official/fixture" / iid
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "report.json", {iid: {"resolved": resolved}})

    def docker(self, args, **kwargs):
        self.assertIn(args[:3], (["docker", "image", "ls"], ["docker", "image", "rm"]))
        self.assertEqual(args[-1], self.task["image"])
        self.assertNotIn("--force", args)
        return subprocess.CompletedProcess(args, 0, "sha256:fixture\n", "")

    def test_grade_before_deletion_and_resume_without_repeating(self):
        def grade(args, **kwargs):
            self.assertTrue((self.directory / "repo/source.py").exists())
            self.assertEqual(args[args.index("--instance-id") + 1], "test-1")
            self.assertTrue((self.root / "predictions.jsonl").exists())
            self.report("test-1")
            return subprocess.CompletedProcess(args, 0)
        with patch("jiang.cleanup.subprocess.run", side_effect=grade) as grading, \
                patch("jiang.cleanup.command", side_effect=self.docker) as docker:
            finish_attempt(self.root, self.task, [self.task])
            finish_attempt(self.root, self.task, [self.task])
            self.assertEqual(grading.call_count, 1)
            self.assertEqual(docker.call_count, 2)
        self.assertFalse((self.directory / "repo").exists())
        self.assertEqual((self.directory / "candidate.patch").read_text(), "saved patch")
        self.assertEqual((self.directory / "trajectory.jsonl").read_text(), "saved trace")
        self.assertTrue((self.directory / "evaluation.log").exists())
        self.assertTrue((self.root / "unrelated/keep").exists())
        self.assertTrue(json.loads((self.directory / "result.json").read_text())["official_resolved"])

    def test_failed_grading_keeps_diagnostics_and_can_retry(self):
        with patch("jiang.cleanup.subprocess.run", return_value=subprocess.CompletedProcess([], 1)), \
                patch("jiang.cleanup.command", side_effect=self.docker):
            finish_attempt(self.root, self.task, [self.task])
        result = json.loads((self.directory / "result.json").read_text())
        self.assertEqual(result["evaluation"]["status"], "error")
        self.assertIsNone(result["official_resolved"])
        self.assertFalse((self.directory / "repo").exists())
        with patch("jiang.cleanup.subprocess.run", return_value=subprocess.CompletedProcess([], 1)) as grading, \
                patch("jiang.cleanup.command", side_effect=self.docker):
            finish_attempt(self.root, self.task, [self.task])
            grading.assert_called_once()

    def test_in_use_image_stops_cleanup_and_preserves_checkout(self):
        def docker(args, **kwargs):
            return subprocess.CompletedProcess(args, 1 if args[2] == "rm" else 0,
                                               "sha256:fixture", "image in use")
        (self.directory / "candidate.patch").write_text("")
        with patch("jiang.cleanup.command", side_effect=docker):
            with self.assertRaisesRegex(RuntimeError, "stopping to avoid disk growth"):
                finish_attempt(self.root, self.task, [self.task])
        self.assertTrue((self.directory / "repo/source.py").exists())
        self.assertEqual(json.loads((self.directory / "cleanup.json").read_text())["status"], "error")

    def test_symlinks_and_missing_artifacts_block_deletion(self):
        (self.directory / "candidate.patch").unlink()
        with patch("jiang.cleanup.command") as docker:
            with self.assertRaisesRegex(ValueError, "Save result"):
                remove_assets(self.root, self.task)
            (self.directory / "candidate.patch").write_text("patch")
            (self.directory / "repo").rename(self.directory / "original")
            (self.directory / "repo").symlink_to(self.root / "unrelated", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Refusing cleanup"):
                remove_assets(self.root, self.task)
            docker.assert_not_called()
        self.assertTrue((self.root / "unrelated/keep").exists())

    def test_interrupted_attempt_is_retained(self):
        atomic_json(self.directory / "result.json", {"status": "interrupted"})
        with patch("jiang.cleanup.command") as docker, patch("jiang.cleanup.subprocess.run") as grading:
            finish_attempt(self.root, self.task, [self.task])
            docker.assert_not_called()
            grading.assert_not_called()
        self.assertTrue((self.directory / "repo").exists())

    def test_evaluator_selects_one_issue_and_preserves_cumulative_results(self):
        tasks = [dict(self.task, base_commit="a" * 40, problem_statement="first"),
                 dict(self.task, instance_id="test-2", base_commit="b" * 40, problem_statement="second")]
        second = self.root / "attempts/test-2"
        second.mkdir()
        atomic_json(second / "result.json", {"model": "fixture", "instance_id": "test-2"})
        (second / "candidate.patch").write_text("second patch")
        export(self.root, tasks)
        (self.root / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
        dataset = self.root / "full.json"
        atomic_json(dataset, [dict(t, eval_script="hidden grading only") for t in tasks])
        self.report("test-1")
        def grade(args, **kwargs):
            predictions = Path(args[args.index("--predictions_path") + 1])
            self.assertEqual([json.loads(s)["instance_id"] for s in predictions.read_text().splitlines()], ["test-2"])
            grading_data = Path(args[args.index("--dataset_name") + 1])
            self.assertEqual([t["instance_id"] for t in json.loads(grading_data.read_text())], ["test-2"])
            self.report("test-2", False)
            return subprocess.CompletedProcess(args, 0)
        with patch("sys.argv", ["evaluate", "--run", str(self.root), "--dataset-file", str(dataset),
                                 "--instance-id", "test-2"]), patch("jiang.evaluate.docker_ready"), \
                patch("jiang.evaluate.subprocess.run", side_effect=grade):
            evaluate_main()
        summary = json.loads((self.root / "evaluation/official_summary.json").read_text())
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["resolved"], ["test-1"])
        self.assertEqual(summary["not_resolved"], ["test-2"])


if __name__ == "__main__":
    unittest.main()
