# Jiang — local SWE-bench repair agent

Start with [ARCHITECTURE.md](ARCHITECTURE.md) for a plain-language explanation,
the review findings, model advice and the limitations of the experiment.
[REPORT.md](REPORT.md) is the short experimental report.

The v2 agent reads real source code, searches symbols, makes exact text edits,
runs focused tests in Docker, and feeds results back to the local model.
Git generates the final patch. A generated patch is **not** a resolved issue.

## Install and check

From the repository, with your virtual environment active:

```bash
source .venv/bin/activate
python -m pip install -e '.[evaluation,claude-agent]'
ollama list
open -a Docker
docker info
python -m unittest discover -s tests -v
```

Ollama must be running. Use `ollama serve` in another terminal only if needed.
Qwen 2.5 Coder 7B and Qwen 3.5 9B have been downloaded locally during validation.

## Assignment workflow: Claude Code + local Qwen

Use this backend when the assignment requires the downloaded agent to perform
the repairs. Claude Code controls the loop; a small MCP server supplies the
same bounded source/edit/test tools as the custom agent.

```bash
python -m jiang.benchmark \
  --engine claude --model qwen3.5:9b \
  --tasks-file runs/qwen2.5-coder-7b/tasks.jsonl \
  --reuse-repos runs/qwen2.5-coder-7b/worktrees \
  --count 10 --limit 1 --context 32768 --max-steps 24 \
  --output runs/claude-qwen35
```

Remove `--limit 1` and add `--resume` to finish all ten selected issues. The
backend creates a local Ollama alias with the configured context and sampling
parameters; it reuses the downloaded weights. It disables built-in tools and
exposes only the project-scoped MCP tool. Test commands execute in Docker.
The alias and CLI version are recorded in the manifest.
An in-process loopback adapter explicitly sets Ollama's thinking, temperature
and output budget. `--think` requests thinking; otherwise the adapter requests
that it be disabled. **Ollama 0.20.4 still emitted thinking blocks for Qwen 3.5
through its Anthropic endpoint in our probe**, so this is not a guarantee that
thinking is off. Each forwarded request's settings are logged in
`inference_requests.jsonl`; actual responses are in the trajectory.

Task selection is easiest-first by default. The runner ranks SWE-bench's
`difficulty` label first (`<15 min fix`, then `15 min - 1 hour`, then longer
labels). Ties prefer an issue with a short reproduction and clear expected
behavior. It still takes at most one task per repository before filling any
remaining slots. Use `--selection balanced` to restore deterministic instance-ID
ordering. The selected order is saved in `tasks.jsonl` and the manifest.

The supplied high-success-rate order is stored in
`configs/swebench_9b_priority.txt`. To use exactly that order, omit the old
`--tasks-file` snapshot so the runner loads the full Verified dataset, then pass
`--preferred-ids-file`; this option overrides the normal difficulty selector.

For a five-task easy pilot, use `--count 5 --limit 5`; with the current saved
snapshot this starts with the four `<15 min fix` issues and then one
`15 min - 1 hour` issue.

Run the requested ten-task priority list:

```bash
python -m jiang.benchmark \
  --engine claude --model qwen3.5:9b \
  --preferred-ids-file configs/swebench_9b_priority.txt \
  --count 10 --context 32768 --max-steps 24 \
  --output runs/priority-10-claude-qwen35
```

For a five-task pilot, use `--count 5 --limit 5` and a new output directory.

### Run ten issues with cleanup after each

With Docker and Ollama running, use:

```bash
bash scripts/run_low_disk.sh
```

The launcher uses the same local Claude/Qwen configuration and priority list,
with `--cleanup-after-each`. Each issue is repaired and officially graded before
its checkout and exact Docker image tag are removed. The next issue then starts.
Logs, patches, manifests, predictions, test output and grading reports remain in
`runs/final-10-claude-qwen35-low-disk/`. An optional first argument changes that
output directory. Rerun the same command to resume.

`attempts/<id>/evaluation.log` preserves grader output;
`attempts/<id>/cleanup.json` records cleanup and observed free disk space.
`result.json` also records evaluation status and elapsed grading time.
The cumulative official summary is updated after each evaluation, and
`analysis.md` is generated when the run finishes. No separate full evaluation
command is needed; it would potentially download removed images again.

Failed evaluations retain their diagnostics and missing reports remain visible;
their bulky assets are still removed. Resuming retries failed grading from the
saved patch without rerunning the repair model. Empty patches are recorded as
`evaluation.status: no_patch` and remain ungraded. If image cleanup fails, the
runner stops before starting another issue to avoid accumulating disk usage.
An interrupted attempt retains its checkout for inspection. No global Docker
prune is performed, and unrelated images and previous run directories remain.
Space for one checkout and one test image is still required; shared Docker
layers can remain while referenced by other images.

Grade and summarize:

```bash
python -m jiang.evaluate --run runs/claude-qwen35 \
  --dataset-file runs/qwen2.5-coder-7b/tasks.jsonl
python -m jiang.analyze runs/claude-qwen35
```

This uses a compact replacement system prompt and a 32K context for this
16 GB Mac. It is a constrained configuration, not the full default Claude Code
experience. Inspect the pilot before committing to a long run.
The custom JSON backend below retains its smaller 16K default. Claude's
automatic compaction is disabled; an exhausted context ends an attempt rather
than repeatedly trying to summarize with an undersized context reserve.

## Pilot before another full run

Reuse your saved ten tasks and existing clones. The old run is kept intact.
The new run uses a different output directory.

```bash
python -m jiang.benchmark \
  --tasks-file runs/qwen2.5-coder-7b/tasks.jsonl \
  --reuse-repos runs/qwen2.5-coder-7b/worktrees \
  --count 10 --limit 1 \
  --model qwen2.5-coder:7b \
  --context 16384 --max-steps 30 \
  --output runs/v2-qwen
```

The first image download may take a while. Each turn prints progress.
Docker tests use the dataset's Linux x86-64 image, emulated on Apple Silicon.
Model-generated shell commands run inside disposable containers, without host
mounts or network. Only image downloads use the network. The official harness
is a separate process with its own container settings.

After the pilot, finish the SAME ten-task run:

```bash
python -m jiang.benchmark \
  --tasks-file runs/qwen2.5-coder-7b/tasks.jsonl \
  --reuse-repos runs/qwen2.5-coder-7b/worktrees \
  --count 10 \
  --model qwen2.5-coder:7b \
  --context 16384 --max-steps 30 \
  --output runs/v2-qwen --resume
```

`--resume` skips saved results. Interrupted tasks restart in a fresh checkout,
with partial artifacts archived. It does not resume an in-flight conversation.
Use a new output directory to change the model or budget, or retry a completed
failure. A stale `.running` lock after a hard crash must be removed only after
confirming that the process has stopped.

If Docker cannot run, use `--tests none` in a separate run directory.
That produces **untested** candidates and loses the test-feedback advantage.

## Grade with the official SWE-bench harness

```bash
python -m jiang.evaluate \
  --run runs/v2-qwen \
  --dataset-file runs/qwen2.5-coder-7b/tasks.jsonl
```

This uses the full original dataset snapshot for grading, including held-out
tests, but the repair agent sees only the issue and base repository. Grading
never feeds hidden test patches or reference fixes back into the agent.

The evaluator runs sequentially to reduce memory pressure and writes
`runs/v2-qwen/evaluation/official_summary.json`. Missing reports indicate
evaluation failures or incomplete evaluation, not successful repairs.

Create a readable per-issue analysis from the saved artifacts:

```bash
python -m jiang.analyze runs/v2-qwen
```

## Compare models

The Mac inspected for this project has 16 GB RAM and had about 23 GB of free
disk space. Start one model at a time. Downloads plus ten Docker images can
exceed that free disk space.

A candidate upgrade:

```bash
ollama pull qwen3.5:9b
```

A Gemma comparison candidate:

```bash
ollama pull gemma4:12b
```

These are optional, substantial downloads. Repeat the same pilot command with
the chosen `--model` and a new `--output`, holding the task IDs, context,
temperature and step limit fixed. Use official resolved counts and time to
compare. Neither recommendation guarantees 4–5 fixes; neither has been measured
on this machine by the v2 experiment unless documented in the report.

## Claude Code compatibility and assignment requirement

Claude Code is already installed on this Mac. Check local inference:

```bash
python -m jiang.claude_local --model qwen2.5-coder:7b --smoke
```

For a tool-capable local model, launch in a disposable project checkout:

```bash
python -m jiang.claude_local --model qwen3.5:9b
```

The wrapper fixes the model endpoint to local Ollama, suppresses user/project
configuration and MCP servers, and exposes only reading/editing/search tools.
The interactive wrapper does not provide the Docker test tool.

`--engine claude` runs the downloaded Claude Code agent. The default
`--engine jiang` runs the custom JSON-action loop. Their results are labeled
separately in manifests. A connection smoke test alone is not a ten-task run;
use the Claude backend for parts (b) and (c) if that is the required agent.

## Artifacts

- `manifest.json`: model digest, configuration, platform, package versions and code hashes.
- `tasks.jsonl`: sanitized task snapshot, without reference fixes or hidden test patches.
- `attempts/<id>/trajectory.jsonl`: model responses, tools, observations, timing and token counts.
- Claude backend: `mcp_actions.jsonl` contains tool calls/results and
  `claude.stderr.log` records CLI diagnostics; trajectory is Claude's native event stream.
- `inference_requests.jsonl` records the actual model and generation controls
  forwarded by the Claude backend's local adapter.
- `attempts/<id>/tests/`: complete focused test logs.
- `attempts/<id>/candidate.patch`: actual changes produced by Git.
- `attempts/<id>/result.json`: generation outcome, duration and local test records.
- `predictions.jsonl`: patches in official SWE-bench format.
- `evaluation/official_summary.json`: verified resolutions, failures and missing reports.

`runs/` is git-ignored because repository clones and logs are large.
Include the manifests, predictions, trajectories and evaluation reports in your
submission archive separately; exclude cloned repositories and Docker images.
