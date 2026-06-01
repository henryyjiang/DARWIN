# DARWIN

**An evolutionary system where coding models autonomously improve each other's training code.**

DARWIN runs a Darwin–Gödel-style evolutionary loop: a population of capable coding models take
turns acting as *mutation agents* that rewrite one another's LoRA-finetuning recipes, the results
are benchmarked, and a genetic algorithm keeps the fittest into the next generation — with **no
human writing the improvements**.

> **Central research question:** can an AI system independently improve its own code and
> architecture — discovering, testing, and keeping ideas that move it toward state-of-the-art?

This repository is **v2**, a near-complete rewrite of the [published v1 demo](#background--v1-paper).
The ground-truth design lives in [`ARCHITECTURE.md`](ARCHITECTURE.md); current build status is
tracked there in §10.3.

---

## How it works

Each **generation** runs a five-stage loop, orchestrated by a cross-platform Python controller:

1. **Select** — a GA keeps the top survivors and spawns offspring by duplicating a survivor and
   pairing it with another survivor as its *mutator*.
2. **Mutate** — each offspring gets a **2–4 hr autonomous agent window** inside its own Docker
   container. The mutator agent reads memory, forms a thesis, edits the genome (data mix, LoRA
   hyperparameters, loss, training loop, adapters), smoke-tests it, and Git-checkpoints every green
   state — so a model that runs out of time reverts to its last working version.
3. **Finetune** — each offspring's recipe is LoRA-finetuned on GPU. Survivors are not re-finetuned.
4. **Benchmark** — only the new offspring are evaluated (coding/math/reasoning), with anti-gaming
   heuristics applied to the fitness signal.
5. **Global-memory pass** — a Claude pass reads all per-model memory files + stats and rewrites the
   shared global memory (what's working, objectives, to-do, cost ledger).

Everything is reversible (Git checkpoints), provenance-tracked (any idea taken from a paper is
attributed), and cost is a first-class constraint surfaced in memory so the system self-regulates
spend.

## Two interchangeable mutation backends

Anywhere DARWIN calls "an AI agent," two implementations sit behind one interface (§4):

- **Claude backend** (`backend="claude"`) — a headless [Claude Agent SDK](https://docs.claude.com)
  session driving Claude Code's built-in tools. Highest capability; used for bootstrapping and to
  prove the loop produces gains. (§4.5)
- **Local backend** (`backend="local"`) — the **population model itself** as the mutator. Its
  LoRA-merged weights are served by **vLLM** behind an OpenAI-compatible endpoint, and the
  **[OpenHands](https://github.com/OpenHands/software-agent-sdk) V1 agent SDK** turns that raw chat
  model into a tool-using coder. This is the purest form of the research claim — self-improvement
  with no API cost. (§4.6)

Both backends attach the **same `darwin-mcp` server** (memory, smoke-test, finalize, cost, paper/
dataset retrieval) and run the **same mutation directive**, so results are comparable across
backends and "the model's identity drives the mutation" holds meaningfully (§9).

## Safety & provenance

Every training/agent/eval task runs in a sandboxed Linux container (the controller stays
cross-platform on the host). Network egress is default-deny with a small whitelist (arXiv, Hugging
Face Hub, PyPI, the Anthropic API); the eval image has zero egress and never sees secrets. Agents
reach the web *only* through the whitelisted MCP `paper.*`/`data.*` tools, and any idea drawn from a
paper must be recorded in the genome and memory. See [`containers/`](containers/) and §8.

---

## Repository layout

| Path | What's there |
|------|--------------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The full design spec — **ground truth**. |
| `darwin/controller/` | Master controller: generation state machine, GA, container lifecycle. |
| `darwin/mutation_agent/` | The §4 mutation window: backends (Claude / local-OpenHands), deadline, Git checkpointer, smoke test, directive. |
| `darwin/mcp/` | The `darwin-mcp` tool server shared by both backends (§9). |
| `darwin/memory/`, `darwin/global_memory/` | Per-model markdown memory store + the Claude-managed global memory pass. |
| `darwin/finetune/`, `darwin/bench/` | LoRA finetune + benchmark pipelines (GPU/live parts deferred). |
| `darwin/sandbox/` | Container spec builders + Docker run wiring (§8.5). |
| `containers/` | Dockerfiles + network policy; `smoke-local/` is the GPU-free local-mutator test. |

## Setup

The controller is cross-platform Python managed with [`uv`](https://docs.astral.sh/uv/):

```bash
# Run the test suite (the pure, infra-free surface — currently 362 tests)
uv run --python 3.14 --extra dev python -m pytest -q
```

The Claude-backed components (the global-memory pass §7.4, the Claude mutation backend §4.5) read
the Anthropic API key from the environment:

```bash
export ANTHROPIC_API_KEY='...'   # add to your shell profile
```

The Linux/GPU-only deps for the local backend (`vllm`, `openhands-sdk`, `openhands-tools`) are
environment-gated to Linux + Python ≥3.12 and lazy-imported, so the dev host installs and tests
without them; a Linux GPU host installs them with `--extra local`.

## Try the local mutator without a GPU

[`containers/smoke-local/`](containers/smoke-local/) runs a **real** `backend="local"` mutation
window end-to-end against a small model served by [Ollama](https://ollama.com) over an
OpenAI-compatible endpoint — no GPU, no vLLM (the harness only needs a `base_url`). It exercises the
full path: OpenHands → `darwin-mcp` tools → genome edits → `smoke.run` → memory write.

```bash
cd containers/smoke-local
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.1:8b
$env:SMOKE_MODEL="llama3.1:8b"; docker compose run --rm --build darwin   # PowerShell
```

See that directory's README for knobs and caveats. (A CPU-served small model is below the §4.6
capability floor — this validates the *plumbing*, not mutation quality; the production path uses a
32B-class model on vLLM.)

## Running the full loop

The end-to-end generational loop depends on live GPU/Docker/API infrastructure. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) §10 for the phased build plan and §10.3 for exactly what is
implemented today versus deferred behind injectable seams.

---

## Background — v1 paper

The published **v1** system trained nanoGPT **from scratch** and used the OpenAI API to mutate the
training code, with persistent JSON memory and a human-in-the-loop interface. In 5 iterations it
achieved a 1.26% improvement in model FLOPS utilization and 2.07% improvement in perplexity over
baseline — a promising foundation for scaling evolutionary GPT training.

**Paper:** [arXiv:2602.05848](http://arxiv.org/abs/2602.05848)

v2 reframes the problem from "learn a model from scratch" to "**discover which architecture/
training/data ideas improve a strong base model**" — LoRA-finetuning a capable coding model in the
evolutionary loop, dropping OpenAI in favor of Anthropic (Claude) + a local model, and adding tools,
online paper retrieval, Git-checkpointed mutation windows, structured memory, and per-model
container isolation.
