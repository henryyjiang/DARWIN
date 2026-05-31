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

## Files
- `darwin-agent.Dockerfile` — mutation-window image (`--build-arg HARNESS=agent|local`).
- `darwin-finetune.Dockerfile` — CUDA + PEFT/LoRA/QLoRA training stack; runs the genome's
  finetune entrypoint.
- `darwin-eval.Dockerfile` — benchmark harnesses, base weights baked in, **zero egress**
  (`--build-arg BASE_SNAPSHOT=<path>`).
- `setup_whitelist_network.sh` — creates the `darwin-egress` user-defined network the
  whitelist containers attach to, and documents the §8.3 allow-list (enforce egress filtering
  via a forward proxy / iptables; the bridge driver alone does not filter by hostname).

## How the controller launches these
The Python side is `darwin/sandbox/` — `ContainerSpec` + `build_docker_run_args` (pure, tested)
and role constructors (`agent_container` / `finetune_container` / `eval_container`) that set the
correct mounts + network per §8.5, plus `DockerContainerRunner` to shell them out. The spec
builder refuses to mount the Docker socket and never emits `--privileged` (§8.2).

_Dockerfiles are reference builds; image tags + the base-model snapshot are wired when the live
GPU plane lands. The v1 root `Dockerfile` (nanoGPT) was removed in the Phase 0 teardown._
