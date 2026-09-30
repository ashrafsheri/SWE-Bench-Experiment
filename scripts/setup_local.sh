#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e '.[evaluation,claude-agent]'
echo "Local dependencies installed. Start Ollama with: ollama serve"
echo "Then download an open model, for example: ollama pull qwen2.5-coder:7b"
