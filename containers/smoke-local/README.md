# Local-mutator smoke test (GPU-free)

End-to-end exercise of the **local-model mutation path** (`backend="local"`, ARCHITECTURE.md §4.6):
a real OpenHands V1-SDK agent session driving a small model over an OpenAI-compatible endpoint,
calling the `darwin-mcp` tools, editing a throwaway git genome, running `smoke.run`, and writing an
iteration memory file. It validates the live SDK glue (`_run_openhands`) that the unit tests can't
reach — **no GPU and no vLLM**, by serving the model with [Ollama](https://ollama.com) instead.

This is a debugging/validation harness, not part of the test suite or the production image set.

## What it checks

The pieces that are real here but stubbed/deferred in the unit tests:

- the `openhands-sdk` `LLM`/`Agent`/`Conversation` wiring and `build_llm_kwargs` routing
  (`openai/<model>` + `base_url`) actually connect to an OpenAI-compatible server;
- `darwin-mcp` attaches via the SDK's native `mcp_config` (`to_openhands_mcp_config`) and the model
  can call `memory.*` / `smoke.run` / `finalize`;
- the OpenHands terminal + file-editor tools can edit the mounted genome;
- a green `smoke.run` auto-commits and `run_mutation_window` finalizes an always-green genome.

## Prerequisites

Docker Desktop (or any Docker Engine). No GPU required — a small CPU-served model is enough to
validate the *plumbing* (not mutation quality). First run pulls the `ollama/ollama` image and the
model (~4.7GB for `qwen2.5-coder:7b`); pick a tool-calling-capable model.

## Run

From this directory (`containers/smoke-local`):

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5-coder:7b   # one-time, ~4.7GB
docker compose run --rm --build darwin
```

The `--build` on the last line rebuilds the `darwin` image against the current code/Dockerfile.
The image installs its deps with **uv**, not pip: `openhands-sdk`'s core dep `lmnr` pins
opentelemetry versions that pip's resolver can't reconcile, but uv (the SDK's own build tool) can.

The `darwin` service runs [`scripts/smoke_local.py`](../../scripts/smoke_local.py), which creates a
throwaway genome, points the serve config at Ollama via `DARWIN_SERVE_*`, and runs a short mutation
window with the **"small"** test directive. On success you'll see the result JSON with
`produced_green=True` and the agent's iteration memory in the run output.

Tear down with `docker compose down -v` (the `-v` also drops the pulled-model volume).

## Knobs (env on the `darwin` service)

| env | default | meaning |
|-----|---------|---------|
| `SMOKE_MODEL` | `qwen2.5-coder:7b` | Ollama model tag (must support tool calling; pull it first) |
| `OLLAMA_BASE_URL` | `http://ollama:11434/v1` | OpenAI-compatible endpoint |
| `DARWIN_WINDOW_H` | `0.17` | wall-clock window (hours); raise for a slow CPU model |
| `DARWIN_DIRECTIVE_STYLE` | `small` | `small` test directive vs. the `full` mission |

## Notes & caveats

- A CPU-served 7B model is **slow** (tens of seconds per step). The smoke goal is "the loop runs
  and produces a green edit," not benchmark gains — bump `DARWIN_WINDOW_H` if it times out.
- Tool-calling reliability varies by model. `qwen2.5-coder` and `llama3.1` are reasonable small
  choices; very small models may fail to call tools at all.
- This swaps Ollama in for vLLM purely as a GPU-free OpenAI-compatible server; the harness code
  path (`build_llm_kwargs` → litellm `openai/` provider) is identical to the real vLLM run.
- The production GPU path uses the `darwin-agent` image (`.[local]`, bundles vLLM); see
  `containers/darwin-agent.Dockerfile`.
