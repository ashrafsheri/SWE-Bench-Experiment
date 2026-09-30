"""Execute agent tests in disposable containers, with no host mounts or network."""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

from .workspace import command
from .progress import progress


def docker_ready():
    result = command(["docker", "info", "--format", "{{.OSType}}"], timeout=30)
    if result.returncode:
        raise RuntimeError("Docker is not running. Start Docker Desktop or use --tests none for generation only.")


class DockerTests:
    def __init__(self, task, logs: Path, timeout=180, memory="3g"):
        self.task, self.logs, self.timeout, self.memory = task, logs, timeout, memory
        self.logs.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def __call__(self, test_command, patch):
        self.counter += 1
        name = "jiang-test-" + uuid.uuid4().hex[:12]
        image, base = self.task.get("image", ""), self.task["base_commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", base) or not image.startswith("swebench/"):
            raise ValueError("Expected an official swebench image and a base commit SHA")
        output = []
        record = {"command": test_command, "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                  "exit_code": None, "status": "environment_error"}
        started = False
        try:
            if command(["docker", "image", "inspect", image]).returncode:
                progress(f"  downloading test image {image}")
                pull = command(["docker", "pull", "--platform", "linux/amd64", image], timeout=1800)
                if pull.returncode:
                    raise RuntimeError(pull.stderr[-3000:])
            create = command(["docker", "run", "-d", "--name", name, "--platform", "linux/amd64",
                "--network", "none", "--memory", self.memory, "--cpus", "2", "--pids-limit", "256",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--entrypoint", "bash",
                image, "-c", "while true; do sleep 30; done"])
            if create.returncode:
                raise RuntimeError(create.stderr)
            started = True
            with tempfile.TemporaryDirectory(prefix="jiang-test-") as temporary:
                directory = Path(temporary)
                (directory / "candidate.patch").write_text(patch)
                # A fresh container for EVERY call ensures tests cannot affect later tests.
                script = ("set -e\nsource /opt/miniconda3/bin/activate\nconda activate testbed\n"
                          "cd /testbed\ngit config --global --add safe.directory /testbed\n"
                          f"git reset --hard {base}\ngit clean -fd\n")
                if patch:
                    script += "git apply /tmp/candidate.patch\n"
                script += "export PYTHONPATH=/testbed\n" + test_command + "\n"
                (directory / "run.sh").write_text(script)
                for file in ("candidate.patch", "run.sh"):
                    cp = command(["docker", "cp", str(directory / file), f"{name}:/tmp/{file}"])
                    if cp.returncode:
                        raise RuntimeError(cp.stderr)
                result = command(["docker", "exec", name, "bash", "/tmp/run.sh"], timeout=self.timeout)
                output.append(result.stdout + result.stderr)
                record.update(exit_code=result.returncode, status="passed" if result.returncode == 0 else "failed")
        except subprocess.TimeoutExpired:
            record["status"] = "timeout"
            output.append(f"Test/container operation timed out (test limit {self.timeout}s).")
        except (RuntimeError, OSError) as exc:
            output.append(str(exc))
        finally:
            if started:
                command(["docker", "rm", "-f", name], timeout=30)
        log = self.logs / f"test-{self.counter:03d}.txt"
        log.write_text("\n".join(output))
        record.update(output="\n".join(output)[-7500:], log=str(log))
        return record
