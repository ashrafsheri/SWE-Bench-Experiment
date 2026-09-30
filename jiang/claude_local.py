"""Optional Claude Code connection check, with local Ollama as the only model."""
import argparse
import json
import os
import shutil
import subprocess

from .agent import OllamaAgent


def local_environment(model, endpoint="http://127.0.0.1:11434"):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("ANTHROPIC_", "CLAUDE_", "CLAUDE_CODE_")):
            env.pop(key)
    env.update(ANTHROPIC_BASE_URL=endpoint,
               ANTHROPIC_API_KEY="ollama", ANTHROPIC_AUTH_TOKEN="ollama",
               ANTHROPIC_DEFAULT_OPUS_MODEL=model,
               ANTHROPIC_DEFAULT_SONNET_MODEL=model,
               ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
               DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1")
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--smoke", action="store_true", help="Test local inference without tools")
    args = parser.parse_args()
    agent = OllamaAgent(args.model)
    metadata = agent.preflight()
    if not shutil.which("claude"):
        parser.error("Claude Code is not installed; see https://code.claude.com/docs/en/overview")
    if not args.smoke and "tools" not in metadata["capabilities"]:
        parser.error("This model does not advertise native tools. Use --smoke or install a tool-capable local model.")
    env = local_environment(args.model)
    cli = ["claude", "--bare", "--setting-sources", "", "--strict-mcp-config",
           "--mcp-config", json.dumps({"mcpServers": {}}), "--model", args.model]
    if args.smoke:
        cli += ["--tools", "", "--system-prompt", "Reply briefly.", "-p", "Reply with LOCAL_OK."]
    else:
        cli += ["--tools", "Read,Edit,Glob,Grep"]
    print(f"Claude Code -> {agent.endpoint} -> {args.model}; local model digest {metadata['tag']['digest']}", flush=True)
    raise SystemExit(subprocess.call(cli, env=env))


if __name__ == "__main__":
    main()
