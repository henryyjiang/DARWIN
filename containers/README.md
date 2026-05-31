# Containers (ARCHITECTURE.md §8.5)

Docker images and network/filesystem policy for the sandboxed Linux containers that run all
training/agent/eval work (the controller stays cross-platform on the host, §10.2). Three
images, to be added as their phases land:

- **`darwin-agent`** — mutation-time tools: Python, Git, the agent harness (Claude SDK or
  OpenHands), MCP client, smoke-test deps. (Phase 2 / §4.5–4.6)
- **`darwin-finetune`** — CUDA + training stack (vLLM, PEFT/LoRA, the base model). (Phase 3 / §5)
- **`darwin-eval`** — benchmark harnesses only, **zero egress**, base weights baked in; only
  the per-offspring adapter + the private eval slice are mounted in. (Phase 3 / §6.2)

Network policy is default-deny egress with a whitelist (arXiv/Semantic Scholar, Hugging Face
Hub, PyPI, the Anthropic API, the internal MCP server); the eval image has zero egress (§8.3).

_Scaffolding only — no Dockerfiles yet. The v1 root `Dockerfile` (nanoGPT) was removed in the
Phase 0 teardown._
