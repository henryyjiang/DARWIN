# DARWIN v2 — Full-Run Test Plan (the "demo profile")

> Companion to `ARCHITECTURE.md` (spec) and `IMPLEMENTATION.md` (status). This doc specifies a
> **budget-free end-to-end test** of the whole framework: run the real generational loop, in real
> Docker containers, for several generations, autonomously — while substituting out the only two
> things that cost real money/time (GPU finetune, multi-hour Claude session). Build this on
> `v2-foundation` in a fresh session; it is self-contained.

---

## 1. Goal & what it proves

Run `python -m darwin --config run.test.yaml` and have the controller drive **3+ generations to
completion on its own**, so we exercise — for real — everything except the GPU and the long agent
session:

- the §2.3 state machine: SELECT → SPAWN → MUTATE → FINETUNE → BENCHMARK → AGGREGATE → cull →
  GLOBAL_MEMORY → CHECKPOINT, and **crash/resume** (kill mid-run, restart, no recompute);
- the GA: keep-5 / drop-5, (S,M) pairing, stable population names, **real selection** (fitness
  varies → different models survive each generation);
- the **container path**: `ContainerGenerationOps` actually launching Docker containers with the
  §8 mounts + network policy (agent = whitelist, eval = `--network none`), the memory seed/ingest,
  and the result/score/adapter handoffs through bind mounts;
- the cost ledger + `BudgetGuard` (mock GPU-hours accrue; the cap can be tripped on purpose);
- per-model memory written + the global-memory pass updating global memory each generation;
- the observability dashboard showing fitness progression across generations.

**Substitution map:**

| Real component | Test substitute | Still real? |
|---|---|---|
| Lambda GPU finetune (`torch`/`peft`, $) | **mock finetune entrypoint** — writes a tiny deterministic "adapter" (a genome fingerprint), reports a small synthetic GPU-hours | runs in a real container |
| Zero-egress GPU eval (`lm-eval`, GPU) | **mock eval entrypoint** — reads the adapter fingerprint → **genome-dependent scores + drift** | runs in a real `--network none` container |
| Multi-hour Claude SDK window | **configurable**: default **mock mutator** (deterministic genome edit, offline); opt-in **real Claude, ~5 min cap** | mock = offline; claude = real SDK |
| "reason for an hour" global pass | default **mock synthesizer**; opt-in **real Claude, ~5 min cap** | mock = offline; claude = real API |
| 32B base + real weights | none needed (mock finetune/eval don't load a model) | n/a |

Decisions already locked (from review): mutator = **both, configurable**; Docker = **real,
lightweight images**; scores = **genome-dependent + drift**.

---

## 2. What you (the operator) must provide

A checklist to do once, before the run. Most of it is only needed for the *opt-in real-Claude*
paths; the default mock run needs only Docker.

1. **Docker Desktop (Windows, Linux-container mode).** Confirm it works:
   `docker run --rm hello-world`. This is the only hard requirement for the default mock run.
2. **Build the test image** (one slim image serves all three roles):
   `docker build -f containers/darwin-agent.Dockerfile --build-arg HARNESS=agent -t darwin-agent .`
   (The mock finetune/eval reuse this image with a different command — no CUDA image to build.)
3. **Create the egress network** the agent container attaches to:
   `docker network create darwin-egress`
   (For the mock mutator no egress actually happens; the network just has to exist. For the
   real-Claude path it must allow `api.anthropic.com` — see §6.)
4. **(Only for the real-Claude mutator and/or real global pass)** an Anthropic API key:
   set `ANTHROPIC_API_KEY` in your PowerShell session (`$env:ANTHROPIC_API_KEY = "sk-..."`).
   - The controller (host) reads it for the **global-memory pass**.
   - `run.py` forwards it into the agent container (`agent_env`) for the **mutator session**.
   - Cost: a 5-min-capped session per offspring × generation. Keep generations small (≤3) and
     consider enabling real Claude for **one** offspring only (see §5 `claude_sample`).
5. Disk: a few GB for the one slim image; the run writes `models/`, `runs/`, `memory/`.

You do **not** need: a GPU, a Lambda key, the CUDA images, a base-model snapshot, or the heavy
`local`/CUDA Python extras.

---

## 3. Components to implement (next session)

Each is small and unit-testable offline (no Docker) the same way the rest of the repo is; the
Docker run is the final manual smoke. Ordered so each step is independently verifiable.

### 3.1 Mock finetune entrypoint — `darwin/finetune/mock_entrypoint.py`
- `main(env)` reads the same `DARWIN_*` env as `finetune/entrypoint.py` (`DARWIN_ADAPTER_OUT`,
  `DARWIN_GENOME_DIR`/cwd, lora knobs, `DARWIN_SAFE_MODE`).
- Computes a **genome fingerprint**: `sha256` over the sorted genome source files, plus a count of
  an "improvement marker" token (see 3.3) and the resolved hyperparams.
- Writes the adapter as small JSON to `DARWIN_ADAPTER_OUT`:
  `{"genome_sha": ..., "markers": N, "lora_rank": R, "method": ...}`.
- Optional brief `time.sleep(DARWIN_MOCK_SLEEP_S)` (default ~1s) so wall-clock GPU-hours are
  non-zero and the ledger/budget logic has something to record.
- Pure helpers (`fingerprint_genome`, `build_adapter_payload`) unit-tested; `main` writes the file.
- Honors `DARWIN_MOCK_FAIL` (`"oom"`/`"nonzero"`) so we can also test the §5.3 failure taxonomy +
  floor fitness through the real container path.

### 3.2 Mock eval entrypoint — `darwin/bench/mock_entrypoint.py`
- `main(env)` reads `DARWIN_ADAPTER_PATH`, `DARWIN_SUITE`, `DARWIN_EVAL_SLICE`,
  `DARWIN_SCORES_OUT` (same contract as `bench/entrypoint.py`).
- Reads the adapter JSON (the eval container sees **only base+adapter**, never the genome — so the
  genome signal must travel inside the adapter, which mirrors the real data flow, §6.2).
- Score per benchmark = `clamp01(base(genome_sha) + IMPROVE_STEP*markers + drift(slice_id, seed))`:
  - `base(genome_sha)` deterministic in ~[0.40, 0.60];
  - each mutation adds a marker → small score gain → **mutated offspring tend to beat the parent →
    fitness trends up across generations** (a watchable demo);
  - `drift(slice_id)` a small per-slice delta so eval-rotation re-benchmarking (§6.2) is exercised.
- `score_vector` pure + unit-tested; `main` writes JSON.

### 3.3 Mock mutation backend — `darwin/mutation_agent/mock_backend.py`
- `MockMutationBackend.run(ctx, deadline)`: make a **deterministic green edit** to the genome —
  append a marker line (e.g. `# darwin-improve <iteration>`) to the recipe (or bump a hyperparam),
  call `ctx.checkpoint(...)` (green → commit), and `ctx.write_memory(...)` with a synthesized
  thesis/changes. This produces genome drift → score drift → real selection, fully offline.
- Register it in the `make_mutation_backend_factory` router and in
  `mutation_agent/entrypoint.py::_default_backend_factory` under `DARWIN_BACKEND=mock`.
- Add `"mock"` to `config.Backend` (`config.py:13`) and to `Controller._mutation_backend`
  (`controller/controller.py`) so `backend: mock` propagates to `state.backend` → `DARWIN_BACKEND`.

### 3.4 Mock global-memory synthesizer — `darwin/global_memory/` (extend)
- `MockSynthesizer.synthesize(digest, current)` returns a deterministic `GlobalMemory` derived from
  the digest (e.g. carry objectives, append a one-line "gen N: best=<model> fitness=<x>" to
  `whats_working`). Offline, instant.
- For the opt-in real path, wrap `ClaudeSynthesizer` with a **wall-clock cap** (a `timeout_s`
  ~300s; on timeout fall back to `current` unchanged so the loop never stalls).

### 3.5 Test profile wiring — `darwin/run.py` (extend)
- Add a `profile: test` (or `mode: test`) that, in `build_controller`, defaults to:
  - **images/commands**: finetune + eval use `image="darwin-agent"` (the slim image) with the mock
    entrypoint commands; the agent uses `darwin-agent` with the in-container mutation entrypoint
    (image CMD) and `DARWIN_BACKEND` from `config.mutation.backend` (`mock` default).
  - **synthesizer**: `MockSynthesizer` unless `ANTHROPIC_API_KEY` is set **and** a real flag is on.
  - **short windows**: `mutation.mutation_window_h ≈ 0.083` (5 min), `soft_deadline_min ≈ 1`.
- Make the container backends' **image + command configurable** from the run config (today
  `_build_container_ops` hardcodes `darwin-finetune` / `darwin-eval` + default CMD). Add an
  `images:`/`commands:` block to the YAML and thread it through.
- Add an optional `agent_network` knob (`whitelist`/`open`) so the real-Claude path can reach the
  API without standing up a proxy (dev-only; see §6).
- Add `claude_sample: int` (default 0) — number of offspring per generation that use the *real*
  Claude mutator while the rest use mock, to cap API spend while still validating the SDK path.

### 3.6 Windows host-path bind-mount normalization — `darwin/sandbox/spec.py` (fix)
- **Required for real Docker on Windows.** `Mount.to_arg()` emits `f"{host}:{container}:ro"`; a
  Windows host path (`C:\Users\...\genome`) contains a `\` and a drive-letter `:` that breaks
  `docker -v` parsing. Normalize host paths to a Docker-acceptable form (`C:\Users\x` →
  `C:/Users/x`, or `//c/Users/x`) when building args. Add unit tests for the Windows→Docker mapping.
- Also confirm `git` inside the container works on the bind-mounted genome (uid 1000 / virtiofs
  ownership). If git complains about "dubious ownership", the agent image already runs
  `git config --system`; may need `safe.directory` for the mount — note and handle.

### 3.7 `run.test.yaml` + a build/run script
- A `run.test.yaml` (see §4) plus a small PowerShell helper (`scripts/test_run.ps1`) that builds
  the image, ensures the network, optionally sets up resume, and runs the loop + dashboard.

### 3.8 Tests (offline, no Docker)
- mock entrypoints (fingerprint + score vector + failure injection); mock mutator window (green
  edit + memory) via the existing fake-backend harness; mock synthesizer; capped real-synth
  fallback; the profile assembly through `build_controller` with a fake `ContainerRunner` running a
  **multi-generation** loop and asserting fitness drift + selection + resume.
- Target: the full multi-gen loop test green offline, then the manual Docker smoke confirms the
  real-container parity.

---

## 4. `run.test.yaml` (target shape)

```yaml
generations: 3
mode: container            # real Docker; the test profile sets mock images/commands + short windows
profile: test              # default: mock mutator, mock synth, mock finetune/eval, 5-min windows

paths:
  workspace: models
  runs: runs
  memory: memory
  base_genome: base_genome
  cost_ledger: runs/cost.jsonl
  eval_slices: eval_slices  # mock eval ignores contents; the dir just has to exist per slice id

images:                    # test profile: one slim image for all three roles
  agent: darwin-agent
  finetune: darwin-agent
  eval: darwin-agent
commands:                  # mock entrypoints (the agent uses its image CMD = mutation entrypoint)
  finetune: ["python", "-m", "darwin.finetune.mock_entrypoint"]
  benchmark: ["python", "-m", "darwin.bench.mock_entrypoint"]

smoke_command: ["python", "smoke_test.py"]

seed_scores:               # gen-0 baseline so fitness normalizes
  s0: {humaneval+: 0.50, gsm8k: 0.42}
  s1: {humaneval+: 0.49, gsm8k: 0.41}
  s2: {humaneval+: 0.51, gsm8k: 0.43}
  s3: {humaneval+: 0.48, gsm8k: 0.40}
  s4: {humaneval+: 0.50, gsm8k: 0.44}

config:
  ga: {population_size: 10, num_survivors: 5, diversity_pick: false}
  mutation: {backend: mock, mutation_window_h: 0.083, soft_deadline_min: 1}  # backend: claude to test SDK
  cost: {gen_budget_usd: 0, gpu_rate_usd_per_h: 1.79}    # gen_budget_usd>0 to test the cap
  benchmark: {suite: ["humaneval+", "gsm8k"], eval_rotation: true, num_eval_slices: 3}
  antigaming: {enabled: false}     # turn on once the loop is confirmed
```

A minimal `base_genome/` (a trivial `recipe.py` + `smoke_test.py` that exits 0) is enough — the
mock finetune/eval don't train or load a model.

---

## 5. How to run & what success looks like

```powershell
# 0. one-time setup
docker run --rm hello-world
docker build -f containers/darwin-agent.Dockerfile --build-arg HARNESS=agent -t darwin-agent .
docker network create darwin-egress

# 1. default mock run (no API key, no cost)
uv run darwin --config run.test.yaml --generations 3

# 2. watch progress (separate terminal, works mid-run)
uv run python -m darwin.observability --runs runs --cost runs\cost.jsonl

# 3. resume test: Ctrl-C mid-run, then re-run the same command — it must continue, not recompute
uv run darwin --config run.test.yaml --generations 3

# 4. opt-in: validate the real Claude mutator + global pass (≤5 min each), 1 offspring/gen
$env:ANTHROPIC_API_KEY = "sk-..."
#   set mutation.backend: claude (or claude_sample: 1) and agent_network: open in run.test.yaml
uv run darwin --config run.test.yaml --generations 1
```

**Acceptance checks**
- The run prints per-generation progress and completes 3 generations without intervention.
- `docker ps -a` shows ephemeral `darwin-agent-*` / `darwin-finetune-*` / `darwin-eval-*` containers
  having run (and the eval one with no network).
- `runs/gen_<n>/state.json` reach `completed: true`; killing mid-run and re-running resumes at the
  first incomplete stage (completed offspring are not recomputed).
- The dashboard shows **fitness changing across generations** and the survivor set shifting (real
  selection), not a flat line.
- `models/` holds 5 survivors + 5 offspring slots; names stable across generations.
- `memory/models/<name>/memory/iter_*.md` written per offspring; `memory/global/*.md` updated each
  generation.
- `runs/cost.jsonl` accrues mock `finetune` GPU-hour entries; setting `gen_budget_usd` small marks
  later offspring `deferred` (unscored, not floored) — the §5.4 cap.
- (Opt-in) with `ANTHROPIC_API_KEY` + `backend: claude`, an agent container runs a real ≤5-min SDK
  session that produces a green commit + a written memory file; the global pass writes real text.

---

## 6. Networking notes (mock vs. real Claude)

- **Mock mutator (default):** no container egress happens. `darwin-egress` only needs to *exist*
  (the agent role attaches to it); eval runs `--network none`.
- **Real Claude mutator:** the agent container needs to reach `api.anthropic.com`. Docker's bridge
  network doesn't filter by host, so for a *quick dev test* set `agent_network: open` (host
  network) — acceptable only because this is a throwaway test box; do **not** ship that for real
  autonomous runs (§8.3 wants the whitelist proxy). The proper path is the forward proxy from
  `containers/setup_whitelist_network.sh`.
- The **global-memory pass runs on the host** (the controller), so it uses your host network +
  `ANTHROPIC_API_KEY` directly — no container involved.

---

## 7. Cost expectations

- Default mock run: **$0**, a few minutes wall-clock (dominated by container start overhead ×
  10 offspring × 3 gens; tune `DARWIN_MOCK_SLEEP_S` to keep it snappy).
- Opt-in real Claude: ~one small ≤5-min Sonnet/Opus session per *sampled* offspring + one capped
  global pass per generation. Use `claude_sample: 1` and `generations: 1` first to bound spend.

---

## 8. Out of scope (still genuinely live, untouched by this test)
Real GPU training/eval numbers, the Lambda SSH `job_runner`, the OpenHands/vLLM local mutator, the
32B scale-up + sharding, and the live anti-gaming eval-data providers. Those remain the "needs real
infra" seams in `IMPLEMENTATION.md §4` — this test deliberately brackets them out.
