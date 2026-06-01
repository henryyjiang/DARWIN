# DARWIN v2 — Implementation Status, Run & Test Guide

> Companion to `ARCHITECTURE.md` (the spec). This doc describes **what is actually built**, the
> **steps to finish** the live system, and **how to run and test** it. Section refs (§x) point
> into `ARCHITECTURE.md`. As of the current `v2-foundation` branch: **323 tests passing**,
> all phase cores (0–7) + the live-infra cores + the run wiring + the **container execution path**
> implemented; only the genuinely-live seams remain (real GPUs / Docker / live APIs + heavy ML deps).
>
> **Run it:** `python -m darwin --config run.example.yaml --generations 5` — bootstraps the
> population on disk, assembles the controller, and drives the loop (uses the Claude mutation
> backend + Claude global-memory pass by default, so it needs an API key for a real run).

---

## 1. The one-paragraph picture

DARWIN evolves a population of 10 coding models. Each generation: a GA keeps the top 5 and spawns
5 offspring (clone a survivor's *genome* = its finetuning/architecture code; assign another
survivor as the **mutator agent**); each offspring's mutator runs a 2–4 h autonomous coding
window (Claude Agent SDK or the local model via vLLM+OpenHands) editing the genome behind the
`darwin-mcp` tool surface, Git-checkpointing every green state; the green genome is **finetuned**
on a runtime-sized Lambda GPU (LoRA/QLoRA, and now **parameter-scaling** via depth expansion / MoE
upcycling); the finetuned offspring is **benchmarked** in a zero-egress eval container; fitness
(benchmarks − cost − anti-gaming penalties) drives the next cull; a **Claude global-memory pass**
synthesizes what's working across the population. Everything is reversible, sandboxed, and
cost-accounted.

The **engine is fully built and unit-tested without any GPU/Docker/API**; the remaining work is
standing up the live substrate and filling a handful of injected "live seams."

---

## 2. Repository layout

```
darwin/                      # the importable package (all logic)
  config.py                  # DarwinConfig + all §10.1 sub-configs
  controller/                # generation state machine, GA, resumable state, ops seams
  mutation_agent/            # §4 window: smoke, checkpoint, deadline, directive, backends, entrypoint
  finetune/                  # §5 jobs, backends, runner, Lambda client, run-sizing, entrypoint
  bench/                     # §6 benchmark jobs, rotation, fitness, eval entrypoint, swe_bench/
  antigaming/                # §6.4 contamination / genome-review / generalization-gap producers
  cost/                      # §5.4 cost ledger + budget guard
  memory/                    # §7.2 per-model schema + store
  global_memory/             # §7.4 digest + synthesizer + pass
  mcp/                       # §9.3 darwin-mcp server + toolsets
  sources/                   # §8.3 whitelisted arXiv/HF retrieval (backs paper.*/data.*)
  sandbox/                   # §8 ContainerSpec + docker-run builder + role ctors + runner
  observability/             # §9.5 run dashboard + §8.4 attribution audit
memory/global/               # seeded global-memory store (objectives/todo/whats_working/cost)
containers/                  # Dockerfiles (agent/finetune/eval) + egress-network setup
models/ , runs/              # population dirs + per-generation resumable state
tests/                       # 45 test files, 323 tests
ARCHITECTURE.md              # the spec (§10.3 = living status)
IMPLEMENTATION.md            # this file
```

---

## 3. What's implemented (by subsystem)

Legend: ✅ built + unit-tested · 🔌 injected **live seam** (interface + orchestration tested, the
live driver needs infra) · 📦 reference artifact (real, not yet built/run).

### Run entrypoint & workspace — ✅ (the "run `main`" path)
- `run.py` + `__main__.py` + `[project.scripts] darwin` — load a YAML run config
  (`run.example.yaml`), **bootstrap** the gen-0 population on disk or **resume** the latest
  generation, assemble the controller (`build_controller` seam), run `Controller.run(generations)`.
- `controller/workspace.py` — `bootstrap_population` (5 survivor seeds with cached scores + 5
  offspring slots), `reset_slot` (the GA **drop** step — wipe a recycled slot; survivors persist),
  `materialize_model` (**move offspring results back** into `models/` from a container workdir).
  The controller resets offspring slots once per generation at SPAWN (resume-safe).

### Orchestration & GA — ✅
- `controller/controller.py` — the §2.3 state machine (SELECT→SPAWN→[MUTATE→FINETUNE→BENCHMARK]→
  AGGREGATE→cull→GLOBAL_MEMORY→CHECKPOINT), persists `runs/gen_<n>/state.json` after every step,
  **resumes at the first incomplete offspring stage**.
- `controller/ga.py` — rank/cull, (S,M) pairing (S with replacement, M≠S), diversity pick.
- `controller/population.py`, `state.py` — `Model`/`Population`, resumable `GenerationState`.
- `controller/ops.py` — `LocalGenerationOps` wiring the real cores on the local FS.
- `controller/container_ops.py` — `ContainerGenerationOps`: runs the window/finetune/eval **inside
  the §8.5 images** (composes `LocalGenerationOps` for spawn/finetune/benchmark with the container
  backends; overrides `mutate` to launch `darwin-agent`). The live container/GPU execution path.
- `controller/diversity.py` — §3.4 genome code-distance (n-gram Jaccard; embedding distance later).

### Mutation agent (§4) — ✅ cores, 🔌 live sessions
- `smoke.py` / `checkpoint.py` / `deadline.py` / `directive.py` / `runner.py` — the §4.2 lifecycle:
  smoke test → green Git checkpoint (`last-green` tag) → always-green finalization; soft/hard/kill
  deadlines; the ORIENT→HYPOTHESIZE→IMPLEMENT→VALIDATE→REFLECT directive.
- `claude_backend.py` — Claude Agent SDK option-building + deadline injection (✅); **live SDK
  session** behind a lazy import (🔌).
- `local_backend.py` — `LocalMutationBackend` + `build_harness_config` (✅); **live OpenHands
  session** = injected `harness_runner` (🔌, default raises).
- `vllm_serving.py` — serve-command builder + `VLLMServer` launch/poll/terminate (✅, seams for
  popen/readiness); real `vllm serve` needs a GPU (🔌).
- `openai_tool_shim.py` — MCP→OpenAI tool-call adapter fallback (✅).

### Finetune (§5) — ✅ cores, 🔌 GPU execution
- `job.py` — `FinetuneJob`/`Outcome`/`Result` + failure taxonomy; carries an optional `RunSize`.
- `backend.py` — `SubprocessFinetuneBackend` (✅, CPU-runnable) + `LambdaFinetuneBackend`
  (✅ provision→run→**always-terminate** orchestration; 🔌 SSH `job_runner`).
- `runner.py` — §5.3 policy (one OOM safe-mode retry, infra-vs-recipe split, per-job cost cap) +
  §5.4 ledger recording (✅).
- `lambda_api.py` — Lambda Cloud REST client (launch/poll/terminate) behind an injectable HTTP fn (✅).
- `sizing.py` — **runtime GPU allocation** for param-scaling: VRAM model + GPU catalog +
  `plan_instance` (✅). Sizes the instance to the (expanded) model + token budget (≤250B).
- `entrypoint.py` — 📦 reference QLoRA/LoRA recipe (pure kwarg builders ✅; training body lazy).

### Benchmark (§6) — ✅ cores, 🔌 eval execution
- `job.py` — `SubprocessBenchmarkBackend` (✅) + `EvalContainerBenchmarkBackend` (🔌).
- `rotation.py` (held-out slice rotation), `fitness.py` (§6.3 reduction) — ✅.
- `entrypoint.py` — 📦 reference eval (env→config + suite dispatch/aggregation ✅; harness load lazy).
- `swe_bench/` — salvaged v1 coding harness (feeds the coding slice).

### Anti-gaming (§6.4) — ✅
- `antigaming/` — contamination n-gram scan, genome-diff hack review (rule-based + Claude reviewer),
  generalization-gap check, composed by `run_antigaming_scan`. Wired into fitness via the
  controller's injectable `AntiGamingScanner` (`LocalAntiGamingScanner`); the live eval-data
  providers (`eval_items_provider`/`ood_probe`) are 🔌.

### Cost (§5.4) — ✅
- `cost/ledger.py` (append-only JSONL) + `cost/budget.py` (`BudgetGuard`, hard cap → `deferred`).

### Memory (§7) — ✅
- `memory/` per-model schema + store (controller-only post-benchmark patch); `global_memory/`
  digest + `ClaudeSynthesizer` + `run_global_memory_pass` (the only sanctioned global writer).
- `memory/global/` is **seeded** with the param-scaling objectives/todo/cost priorities.

### MCP server (§9.3) — ✅
- `mcp/` — `memory_*`, `smoke_run`, `finalize`, `cost_*`, and now `paper_*`/`data_*` tools.
- `sources/` — whitelisted arXiv (`PaperSource`, citation strings) + HF Hub (`DataSource`, card +
  license) behind a default-deny egress whitelist + injectable transport.

### Containers & safety (§8) — ✅ Python (now an execution path), 📦 images
- `sandbox/` — `ContainerSpec` → `docker run` builder (refuses Docker socket, no `--privileged`,
  network policy, resource caps) + role constructors (`agent`/`finetune`/`eval`, the eval role now
  takes a writable scores mount) + `DockerContainerRunner` + the `ContainerRunner` protocol.
- The §8.5 images are now wired as the actual execution path via `ContainerGenerationOps` +
  `ContainerFinetuneBackend` (`finetune/backend.py`) + `EvalContainerBenchmarkBackend`
  (`bench/job.py`) + the in-container `mutation_agent/entrypoint.py` (`mode: container`).
- `containers/` — 📦 three Dockerfiles (the `darwin-agent` CMD now runs the mutation entrypoint) +
  `setup_whitelist_network.sh`.

### Observability (§9.5) — ✅
- `observability/dashboard.py` (run-status reader + markdown + CLI) + `attribution.py` (§8.4 audit).

---

## 4. The remaining live seams (what's left to implement)

Each is a single injected interface; everything around it is already tested.

| Seam | Where | What to implement |
|---|---|---|
| Lambda SSH job runner | `finetune.backend.LambdaJobRunner` | SSH to the instance, sync the green genome, `docker run darwin-finetune`, fetch the adapter back to `adapter_out`. |
| OpenHands session | `mutation_agent.local_backend` `harness_runner` | Drive OpenHands against the vLLM endpoint with `darwin-mcp` + the directive; inject the soft-deadline nudge; stop on hard deadline. |
| Claude live session | `claude_backend` (lazy `claude-agent-sdk`) | Validate the real multi-hour streamed session inside the `darwin-agent` container. |
| vLLM live serve | `VLLMServer` defaults | Real `vllm serve` of a LoRA-merged (and/or expanded) model on a GPU. |
| Eval data providers | `LocalAntiGamingScanner` `eval_items_provider`/`ood_probe` | Supply host-only held-out items + an OOD probe run to turn on the live contamination + generalization-gap checks (genome-diff review already runs). |
| Real training/eval | `finetune/entrypoint.py`, `bench/entrypoint.py` | The bodies are written; they run only inside the CUDA images with torch/peft/trl + real datasets/harnesses. |

The **container generation ops are now built** (`controller/container_ops.py`): set `mode: container`
in the run config and the window runs in `darwin-agent`, finetune in `darwin-finetune`, eval in
`darwin-eval` via `darwin/sandbox/`. The three design points are resolved — an in-container mutation
entrypoint (`mutation_agent/entrypoint.py`), a writable scores mount on the eval container
(`sandbox/roles.eval_container(scores_out_host=...)`), and host↔container path mapping (the
offspring's `genome` dir bind-mounts rw so edits land in place; a scratch mount carries the result
JSON + the seeded/ingested per-model memory). It is unit-tested end-to-end with a fake
`ContainerRunner`; the only thing still needed is a real Docker host + the built images + GPUs.

---

## 5. Steps to finish the implementation

Ordered so each step is independently verifiable.

1. **Build the images.** `docker build` the three `containers/*.Dockerfile`; bake a base-model
   snapshot into `darwin-eval` (`--build-arg BASE_SNAPSHOT=...`). Smoke each: agent has git +
   harness, finetune imports torch/peft, eval runs offline.
2. **Stand up egress control.** Run `containers/setup_whitelist_network.sh`; put a forward proxy
   (or iptables) behind the `darwin-egress` network enforcing the §8.3 allow-list; confirm the
   eval container has zero egress (`--network none`).
3. **One real finetune.** Provide a Lambda API key + SSH key; implement the `LambdaJobRunner`;
   finetune a small model end-to-end on one GPU using `finetune/entrypoint.py`; confirm the
   adapter materializes and GPU-hours land in the cost ledger.
4. **One real benchmark.** Implement the `bench/entrypoint.py` harness adapters (HumanEval+/GSM8K
   via `lm-eval`; coding via `swe_bench/`); run `base+adapter` in `darwin-eval`; get a score vector.
5. **One real Claude mutation window** in `darwin-agent` (SDK + API key); verify thesis.md, a green
   final commit, and a written memory file.
6. **One real local (OpenHands) window.** Implement the `harness_runner`; `VLLMServer.start()` a
   served model; drive OpenHands with `darwin-mcp`; verify parity with the Claude path.
7. **Parameter-scaling experiment.** Have a genome do depth expansion / MoE upcycling; confirm
   `sizing.plan_instance` allocates multi-GPU and the run trains.
8. **Turn on anti-gaming live checks.** Inject `eval_items_provider`/`ood_probe`.
9. **Multi-generation run** on a small/cheap base to confirm the loop produces real fitness gains;
   add in-loop infra-failure re-provision; then scale to the 32B target.

---

## 6. How to run

The dev/orchestration host is **Windows + PowerShell**; training/agent/eval run in **Linux Docker
containers on Lambda GPUs**. The controller is cross-platform.

```powershell
# tests
uv run --python 3.14 --extra dev python -m pytest -q

# the darwin-mcp server (stdio) — the agent tool surface
uv run python -m darwin.mcp.server --root .

# the run-status dashboard (works mid-run)
uv run python -m darwin.observability --runs runs --cost runs\cost.jsonl

# the full loop
uv run darwin --config run.example.yaml --generations 5

# build a container image (from repo root)
docker build -f containers/darwin-agent.Dockerfile --build-arg HARNESS=agent -t darwin-agent .
docker build -f containers/darwin-finetune.Dockerfile -t darwin-finetune .
docker build -f containers/darwin-eval.Dockerfile --build-arg BASE_SNAPSHOT=./.base-snapshot -t darwin-eval .
```

Driving a generation in code (the seam wiring the real cores):

```python
from darwin.config import DarwinConfig
from darwin.controller import Controller, GenerationStateStore, LocalGenerationOps, Population
from darwin.cost import CostLedger
from darwin.memory import MemoryStore

cfg = DarwinConfig()                       # tune ga / mutation.backend / cost caps / antigaming
store = MemoryStore("memory"); ledger = CostLedger("runs/cost.jsonl")
ops = LocalGenerationOps(config=cfg, store=store, ledger=ledger, workspace="models",
                         mutation_backend_factory=..., finetune_backend=..., benchmark_backend=...,
                         smoke_command=[...])
ctrl = Controller(config=cfg, store=store, ledger=ledger,
                  state_store=GenerationStateStore("runs"), ops=ops, synthesizer=...,
                  antigaming=..., budget=...)
final_population = ctrl.run(generations=5, population=Population(...))
```

Key config switches (`DarwinConfig`, defaults from §10.1): `mutation.backend` (`claude`/`local`/
`mixed`), `ga.diversity_pick`, `cost.gen_budget_usd` + `per_job_*`, `antigaming.genome_reviewer`
(`claude`/`rule`/`none`), `finetune.target_params_b` / `max_train_tokens` / `dynamic_gpu_allocation`,
`benchmark.suite` / `eval_rotation`.

---

## 7. How to test

```powershell
uv run --python 3.14 --extra dev python -m pytest -q     # 323 passing
```

> The heavy `local`-extra deps (`vllm`, `openhands-sdk`, `openhands-tools`) are environment-marked
> to `sys_platform == 'linux'` + Python `>=3.12`, so uv's universal resolution stays satisfiable
> and the Windows/3.14 dev host installs + tests without them — they're lazy-imported and never
> touched by the suite. A Linux 3.12+ GPU host gets them with `--extra local`.

**What the unit tests cover (no infra needed):** config; GA + selection + diversity; the full
controller loop with fakes incl. crash/resume + budget `deferred`; the mutation lifecycle (smoke,
checkpoint green/zero-green, deadlines, runner) with a fake backend; finetune classification +
OOM retry + cost; benchmark + rotation + fitness; anti-gaming producers + scan + controller wiring;
cost ledger + budget guard; memory schema/store + global-memory pass (offline fake client); the
MCP toolsets incl. `paper.*`/`data.*` (offline canned responses) + egress-whitelist enforcement;
the sandbox docker-run builder + role policies; the Lambda client + backend orchestration (fake
HTTP + fake job runner); runtime GPU sizing; the vLLM launcher (fake popen/readiness); the
reference-entrypoint pure cores; the observability dashboard + attribution audit.

**What still needs live validation** (the §4/§5 table above — needs GPU/Docker/live API + heavy
deps): real finetune + adapter; real benchmark scores in the zero-egress container; a real Claude
multi-hour window; a real vLLM serve + OpenHands session; the Lambda SSH job runner + always-
terminate against the real API; a real depth-expansion / MoE-upcycling run sized onto multi-GPU;
the live anti-gaming eval-data providers; and an end-to-end multi-generation run showing real
fitness gains.

---

## 8. Invariants enforced in code (don't regress these)

Finetuning only ever runs on a green commit (§4.4); the held-out eval set never enters mutation/
finetune containers (§6.2/§6.4); no population agent writes global memory — only the global-memory
pass (§7.3); agents acquire data via precompiled datasets, never scrapers, and all agent web access
is the whitelisted `paper.*`/`data.*` (§8.3); autonomous `bypassPermissions` agents only run inside
a sandboxed container that never mounts the Docker socket and is never `--privileged` (§8); the
budget cap stops new launches without killing in-flight jobs (§5.4); a `finetune_failed` recipe
gets floor fitness while infra failure is never charged to the recipe (§5.3/§6.3).
