#!/usr/bin/env bash
# Run from any directory. An optional argument selects a different output folder.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="${1:-runs/final-10-claude-qwen35-low-disk}"
mkdir -p "$RUN_DIR"

caffeinate -i .venv/bin/python -u -m jiang.benchmark \
  --engine claude \
  --model qwen3.5:9b \
  --preferred-ids-file configs/swebench_9b_priority.txt \
  --count 10 \
  --context 32768 \
  --max-steps 24 \
  --tests docker \
  --cleanup-after-each \
  --output "$RUN_DIR" \
  --resume \
  2>&1 | tee -a "$RUN_DIR/benchmark.log"
