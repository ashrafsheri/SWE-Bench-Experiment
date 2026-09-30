# Local LLM repository-repair experiment

## Architecture and design

Jiang runs a downloaded model through Ollama on the local computer. No cloud
model generates benchmark repairs. The machine inspected is an Apple Silicon
Mac with 16 GB RAM; the installed model is `qwen2.5-coder:7b` (approximately
4.7 GB on disk). The agent prepares a separate Git checkout at each issue's
base commit. Git generates the final repair patch.

The initial agent supplied a file list and project metadata and asked for a
unified diff. Its revised design gives the model search, source-reading,
exact-text editing, diff-review and Docker-test actions. Each action's result
is returned to the model. Exact matching and Python syntax validation reject
bad edits early. This reduces guessed source code and diff-format errors.
The custom v2 configuration uses a 16,384-token context and a bounded number of
actions. The custom backend trims old conversation pairs with a character
budget; complete traces remain on disk. A second backend delegates the loop to
the installed Claude Code CLI, exposing repository/test tools via MCP.

Agent tests run in fresh, network-disabled containers using the dataset's
prebuilt project images. Linux x86-64 images use emulation on this Mac.
The official SWE-bench harness grades saved predictions separately. Reference
fixes, hidden test patches and evaluation scripts are withheld from the repair
agent. Generated patches, local test outcomes and official resolutions are
distinct metrics.

## Protocol and observed results

The original ten saved Verified instances are retained for comparisons. Each
task has one attempt with a fixed action budget in a v2 run; additional runs
must be disclosed separately. Model identity, parameters, code hashes, prompts,
responses, observations, patches and timings are recorded.

| Measurement | Observed result |
|---|---|
| Original run after path correction | 10 attempts, 0 patches accepted by local Git check |
| Invalid diff hunk structure | 5 tasks |
| Source context mismatch | 2 tasks |
| No usable patch | 2 tasks |
| Missing diff header | 1 task |
| Original local tests | Not executed |
| Original official resolutions | Not measured |
| Revised implementation automated tests | 13 passed |
| Claude Code local connection | `LOCAL_OK` received from Ollama/Qwen 7B |
| Revised one-task pilot | See pilot validation below |

An earlier run also failed because the runner passed relative patch paths
from the wrong working directory. That is an infrastructure failure, not model
performance. Original post-correction records are preserved under
`runs/qwen2.5-coder-7b/`. Local patch rejection alone is not an official
SWE-bench result, so no official resolution percentage is inferred from it.

## Analysis

The original result cannot fairly isolate model quality. The model usually
could not see the relevant implementation, had no interactive file tools,
lost prior answers on retries, and received no actual project-test feedback.
The v2 changes address these general deficiencies without manually repairing
benchmark source code. Passing the implementation's tests establishes tool
behavior, not benchmark success.

The first v2 pilot located the relevant source and generated an edit, but an
execution-service restart closed stdout and interrupted the run. Its status is
`infrastructure_error`; artifacts are retained in `runs/v2-qwen-pilot/`.
A disconnect-tolerant progress logger was added. The fresh 7B custom-loop pilot
in `runs/v2-qwen-pilot-2/` exhausted 24 actions without a patch in 542.68 seconds.
It repeatedly read unrelated source. Paginated reads and a baseline-test guard
were added afterward. Qwen 3.5 9B was downloaded for a Claude-backend pilot;
this changes both agent and model and is not an isolated model comparison.
Claude integration pilots exposed output-reservation and compaction issues.
The 16K pilot eventually reached five tool actions and one reproduction command,
but stopped at its context limit without a patch (488.22 seconds). The next
configuration uses 32K context, 4K maximum output, compact prompts and small edits.
A subsequent 32K development run was stopped after detecting that thinking
remained enabled despite the CLI preference. A loopback adapter now explicitly
forwards the requested generation controls; request metadata is saved. A direct
probe confirmed that Ollama 0.20.4 still emitted thinking blocks despite the
explicit disabled setting. Requested controls must not be confused with actual
server behavior. These
setup pilots are disclosed separately from a completed benchmark experiment.
The 32K Claude/Qwen 9B pilot in `runs/v3-claude-qwen35/` repeated a search for
a nonexistent test name, produced no patch and ran no tests. It was stopped
after 25 tool requests and 979.87 seconds because the action budget was exceeded.
This revealed that CLI turns and tool requests are not the same counter; the
runner now stops further requests once the tool budget is exhausted.
The native JSON comparison in `runs/v3-qwen35-json/` reached the relevant
implementation and attempted a reproduction, but repeatedly returned a long
unsupported `python` action. It was stopped at 585.26 seconds with no patch;
the result is recorded as interrupted. The custom loop now normalizes that
spelling into its bounded Docker `test` action, and the regression test covers
the behavior.
The native tool responses now remind the model to run a baseline after five
actions without testing. A unit/transport check covers this feedback; its
effect on benchmark success still requires measurement.

The 4–5 resolution target remains an experimental objective. Compare a newer
Qwen model and a suitable Gemma variant on the same tasks only after validating
the loop. No Gemma/Qwen superiority claim follows from this experiment.
Memory, free disk space, emulation time, limited context and environmental
failures must be included in interpreting future results.

## Assignment scope and reproducibility

Claude Code is installed and a local inference connection was verified through
the included wrapper. Use `--engine claude` for assignment runs that require
the downloaded agent to do the repair work. Earlier custom-loop runs remain
labeled as such. A connection test alone is not completion of ten repairs.

[README.md](README.md) gives exact commands;
[ARCHITECTURE.md](ARCHITECTURE.md) explains the design in plain language.
Grading is performed with the installed `swebench==5.0.2` harness.
Sources: [SWE-bench evaluation](https://www.swebench.com/SWE-bench/guides/evaluation/),
[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs),
[local Claude Code integration](https://docs.ollama.com/integrations/claude-code).
