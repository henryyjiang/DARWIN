# DARWIN v2 — Architecture Specification

> Reference architecture for the second generation of DARWIN. This document is the
> ground truth for implementation. It describes the *new* design; the v1 nanoGPT
> backbone is largely rewritten. Implementation phases are in §10.

---

## 1. Overview & Goals

### 1.1 Central research question
Can an AI system **independently improve its own code and architecture** — discovering,
testing, and keeping ideas that move it toward state-of-the-art — with no human writing
the improvements? DARWIN tests this via a Darwin-Gödel-style evolutionary loop where
coding models mutate each other's training code and the fittest survive.

### 1.2 What changed from v1
| | v1 (demo) | v2 (this doc) |
|---|---|---|
| Target | nanoGPT trained **from scratch** | Capable coding model (Qwen2.5-Coder-32B class), **LoRA finetuned** |
| Mutation | Fixed-chunk streaming + selective rewrite via GPT-4o-mini | Autonomous **2–4 hr agent window** (Claude Agent SDK or local model harness) |
| Agent ability | Single-shot completion, no tools | Runs scripts, reads papers online, smoke-tests, Git-checkpoints, reverts |
| Memory | Per-model JSONL diff summaries | Per-model markdown (1 file/iteration) + Claude-managed **global memory** |
| Selection signal | MFU + perplexity | Coding/math/reasoning benchmarks + anti-gaming heuristics |
| Backend | OpenAI API | **Anthropic (Claude) API + local model**, OpenAI dropped |
| Isolation | Single Docker image | Per-model Docker container, network + filesystem restrictions |

### 1.3 Why finetuning, not from-scratch
Training multiple 30B+ models from scratch for several generations would cost months and
enormous compute. Finetuning (LoRA by default) reframes the problem from "learn a model"
to "**discover which architecture/training/data ideas improve a strong base model**,"
which is both cheaper and a more direct test of the research question. Models may argue in
their own reasoning for a full finetune when they have a thesis for it; **cost is a
first-class constraint surfaced in global memory**, so the system self-regulates spend.

### 1.4 Design principles
- **Agents do the work, Claude orchestrates meta-reasoning.** The 10 population agents
  perform mutation and self-improvement. A separate Claude pass does global reasoning and
  maintains global memory. Population models never write global memory directly.
- **Everything reversible.** Git checkpoints every working version inside the mutation
  window so a model that runs out of time reverts to its last green state.
- **Provenance and safety are non-negotiable.** Sandboxed containers, attribution of any
  idea taken from a paper, no illegal scraping, no uncontrolled network egress.
- **Two interchangeable AI backends everywhere.** Any place that calls "an AI agent"
  supports both a Claude-API path and a local-model path behind one interface.

---

## 2. System Topology

### 2.1 The generational loop
```
                         ┌─────────────────────────────────────────────┐
                         │            MASTER CONTROLLER                  │
                         │  (orchestrates one generation end-to-end)     │
                         └───────────────┬───────────────────────────────┘
                                         │
   ┌─────────────────────────────────────┼──────────────────────────────────┐
   │                                     │                                    │
   ▼                                     ▼                                    ▼
[1. SELECT]                       [2. MUTATE]                          [3. FINETUNE]
GA keeps top 5 survivors;         For each of 5 offspring:             LoRA finetune each
spawns 5 offspring by             duplicate a survivor, assign         offspring on Lambda
duplicating a random              another survivor as mutator          Labs (sharded,
survivor + pairing a              agent. 2–4 hr autonomous             parallel, own GPU).
random survivor as its            window in its own Docker             Survivors NOT
mutator.                          container. Writes per-model          re-finetuned.
                                  memory file at the end.
   │                                     │                                    │
   └─────────────────────────────────────┼──────────────────────────────────┘
                                         ▼
                                  [4. BENCHMARK]
                          Only the 5 new offspring are benchmarked
                          (coding/math/reasoning). Survivor scores
                          are cached. Anti-gaming heuristics applied.
                                         │
                                         ▼
                                  [5. GLOBAL MEMORY PASS]
                          Claude API reads all 10 per-model memory files
                          + performance stats → rewrites global memory
                          (what's working, objectives, to-do, cost ledger).
                                         │
                                         ▼
                                  next generation → [1. SELECT]
```

### 2.2 Components
- **Master Controller** — Python orchestrator. Owns the generation state machine, the GA,
  container lifecycle, Lambda provisioning, and the global-memory pass trigger. Replaces
  v1 `main_controller.py`.
- **Mutation Agent** — the autonomous coder. Two implementations behind one interface
  (Claude Agent SDK; local-model harness). Runs *inside* a per-model container. (§4)
- **Finetuning Pipeline** — LoRA/QLoRA training jobs on Lambda Labs GPUs. (§5)
- **Benchmark Runner** — executes the eval suite against a finetuned offspring, produces a
  fitness vector, applies anti-gaming checks. (§6)
- **Memory Subsystem** — per-model markdown store + Claude-managed global memory, exposed
  to agents via an **MCP server** so retrieval/writes are structured. (§7)
- **Container/Safety Layer** — Docker images, network policy, attribution enforcement. (§8)
- **MCP Server** — the structured tool surface (memory, smoke test, paper search,
  cost ledger, finalize) shared by both agent backends. (§9.3)

### 2.3 Generation state machine (controller)
```
PROVISION → SELECT → SPAWN_OFFSPRING → [per offspring] MUTATE → FINETUNE → BENCHMARK
          → AGGREGATE_FITNESS → GA_CULL → GLOBAL_MEMORY_PASS → CHECKPOINT_GENERATION → loop
```
Each transition is idempotent and resumable: the controller persists generation state to
disk (`runs/gen_<n>/state.json`) so a crash mid-generation resumes without re-running
completed offspring.

---

## 3. Population & Genetic Algorithm

### 3.1 Population
- **10 models alive** at any time. Each is a directory holding: the base-model reference,
  the LoRA adapter(s), the *training/architecture code* (the thing that actually mutates),
  the finetune config, and the model's memory folder.
- A model = `(genome, weights)` where the **genome is the code+config** the agent edits and
  the **weights are the LoRA adapter** produced by finetuning that genome.

### 3.2 Selection & reproduction (one generation)
1. **Benchmark** the 5 offspring from the previous step. Survivors keep cached scores *unless
   the eval slice rotated this generation*, in which case survivors are cheaply re-benchmarked
   (not re-finetuned) on the current slice so all 10 fitnesses share one slice (§6.2).
2. **GA cull**: rank all 10 by fitness (§6.3); keep top 5 survivors, delete bottom 5.
3. **Reproduce**: create 5 offspring. For each offspring:
   - pick a **random survivor S** → `clone(S)` becomes the offspring's starting genome;
   - pick a **different random survivor M** → M is the **mutator agent** that edits the clone.
   - This is the crossover analogue: M brings its own "knowledge" (its memory, its code
     ideas) to bear on S's genome. The offspring inherits S's code but is reshaped by M's
     reasoning.
4. New population = 5 survivors + 5 offspring → next generation.

**Pairing rules (S and M assignment).** Across the 5 offspring, **S is drawn with
replacement** (a strong survivor may be cloned more than once — desirable, it gets more
exploration). **M is also drawn with replacement** but constrained `M ≠ S` for that
offspring. The same M may therefore mutate multiple offspring in a generation (fine — the
windows run in parallel containers, and M's weights are read-only during mutation). If a
generation has < 2 survivors (degenerate early/edge case), M falls back to the Claude
backend regardless of the configured backend, since no distinct local mutator exists.

> **Note on the mutator's identity.** When the mutation backend is the *local model*, M's
> agent literally *is* M's finetuned weights driving the harness — so a model's accumulated
> finetuning genuinely changes how it mutates others. When the backend is Claude API, M's
> "identity" is injected via M's memory + genome in the prompt context.

### 3.3 Fitness
Fitness is a scalar reduced from a benchmark vector (§6.3), penalized by cost and by
anti-gaming flags. Selection pressure parameters (elitism count, number culled, any
diversity bonus) live in the controller config so they can be tuned without code changes.

### 3.4 Diversity safeguard (optional, recommended)
Pure greedy top-5 can collapse the population into near-duplicates. Recommended: reserve
1 of the 5 survivor slots for a **diversity pick** (the highest-fitness model whose genome
is most *different* — by code-embedding distance — from the already-selected elites). Flag
in config; off by default until baseline behavior is understood.

---

## 4. The Mutation Agent

This is the heart of v2 and the part requiring the most AI engineering. One offspring's
mutation = **one autonomous agent session, 2–4 hours, inside one Docker container.**

### 4.1 What the agent is asked to do
The mutator M is dropped into the offspring's container with:
- the offspring genome (cloned from survivor S),
- read access to its own (M's) memory and to global memory via MCP,
- a tool surface: shell, file edit, Git, web (whitelisted), smoke-test runner, MCP tools.

Its objective prompt (the **mutation directive**) instructs it to: study what has worked
(global memory) and its own past attempts (its memory), form a thesis for how to improve
the offspring's finetuning/architecture code, implement it, **smoke-test that the code is
functional**, Git-commit each working version, and write a memory file documenting what it
tried and why before time expires.

### 4.2 The agent loop (backend-agnostic contract)
Every backend must implement this lifecycle so the controller treats them identically:
```
start(container, genome, directive, budget) → session
  loop until FINALIZE or time budget exhausted:
      agent reasons → calls a tool → observes result
      on each green smoke test: git commit (checkpoint)
  on FINALIZE signal OR T-minus-15min warning:
      ensure last commit is green (else `git reset --hard` to last green)
      write per-model memory file (§7.2)
  return final genome (HEAD of the offspring's git branch)
```

### 4.3 Time management & the "finalize" signal
The 2–4 hr window is enforced by the controller, not trusted to the model. Three layers:
- **Soft deadline (T-minus-15min):** controller injects a system message — *"15 minutes
  remain. Stop new experiments. Ensure the current code passes smoke tests, commit it, and
  write your memory file now."*
- **Hard deadline (T-0):** controller sends `FINALIZE`. Agent must commit + write memory.
- **Kill (T+grace, grace = 5 min default):** if the agent hasn't returned, controller
  force-stops the container and recovers the **last green commit** as the final genome.
  Memory file, if missing, is synthesized from the Git log + tool transcript by a short
  Claude call.

A model can also self-declare `FINALIZE` early (via an MCP `finalize` tool) if it believes
it's converged — saving cost. This is the "tell the model it's time to finalize" mechanism
working in both directions.

**Zero-green-commit case.** If the window ends with *no* green commit (the agent never
produced functional code — likely with the weaker local backend), the offspring's final
genome **falls back to the unchanged clone of survivor S** (which is green by construction,
since S was a finetuned survivor). The offspring is flagged `mutation_failed: true` in its
memory file and receives a small fitness penalty (`λ_failed`), so a failed mutation is
mildly selected against but not catastrophic. It still proceeds to finetune/benchmark as a
near-copy of S — this naturally re-tests S's recipe and keeps population size stable.

### 4.4 Git checkpointing, "green", & revert
Each offspring container is a Git repo on a branch `offspring/<id>`. **"Green" is defined
concretely** (no ambiguity for the implementer):
- A commit is **green** iff the smoke test (§4.4.1) exited 0 on the tree at that commit.
- Green commits are recorded two ways: commit-message prefix `darwin-green:` **and** a
  moving Git tag `last-green` that the smoke-test runner force-moves to each new green
  commit. "Revert to last green" = `git reset --hard last-green`. If the `last-green` tag
  is absent, the zero-green-commit fallback above applies.

The smoke-test runner is wired so that **a passing smoke test auto-commits** (`darwin-green:
<agent summary>`) and advances `last-green`. On any failure or at deadline the agent (or
controller, on kill) resets to `last-green`. The **final genome is always a green commit** —
never a half-edited broken state. This guarantees finetuning never runs on broken code.

#### 4.4.1 What the smoke test actually verifies
"Green" must mean *the recipe will train*, not merely *the code imports*. The smoke test is a
**fast, tiny end-to-end finetune dry-run** (target < ~2 min, runs in the `darwin-agent`
container on CPU or a single small GPU slice). It must:
1. **Import + config validation** — the genome's training entrypoint imports cleanly and its
   config parses against a schema (valid LoRA rank, lr, batch, data paths resolvable).
2. **Data pipeline check** — the data scripts produce ≥1 well-formed batch of the expected
   shape/dtype (no empty/NaN batch).
3. **One real train step** — run a single forward+backward+optimizer step on a tiny base
   stub (or 1-layer proxy) using the genome's loss/objective; assert loss is finite and a
   gradient flowed (param delta ≠ 0). This catches semantically-broken loss/objective edits
   that import fine but can't train.
4. **Adapter materializes** — a LoRA adapter object is produced and serializable.

The smoke test is **deterministic** (fixed seed, fixed tiny data) so green-ness is not
flaky. The smoke-test harness itself is part of the controller-owned scaffolding, **mounted
read-only** into the container so the agent cannot weaken it to force false greens (an
attack surface noted in §6.4 / §8.2).

### 4.5 Backend A — Claude Agent SDK (headless, scripted, multi-hour)
The "stream Claude Code automatically for 2 hours via a Python script + predetermined
prompt" path. Use the **Claude Agent SDK** (the programmatic form of Claude Code), not the
interactive CLI.

```python
# Conceptual — mutation_agent/claude_backend.py
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def run_claude_mutation(container, genome_path, directive, deadline):
    options = ClaudeAgentOptions(
        cwd=genome_path,                       # offspring repo inside the container
        permission_mode="bypassPermissions",   # autonomous; safe only because container is sandboxed
        allowed_tools=["Bash", "Edit", "Write", "Read", "Glob", "Grep", "WebSearch", "WebFetch"],
        mcp_servers={"darwin": {...}},          # memory, finalize, cost, paper-search (§9.3)
        system_prompt=DIRECTIVE_SYSTEM_PROMPT,  # role, constraints, attribution rules
        max_turns=None,                         # bounded by wall-clock, not turns
    )
    async with ClaudeSDKClient(options) as client:
        await client.query(directive)          # the predetermined mutation prompt
        async for msg in client.receive_response():
            log(msg)                            # full transcript persisted for the memory fallback
            if past(deadline.soft):  await client.query(SOFT_DEADLINE_NUDGE)
            if past(deadline.hard):  await client.query(FINALIZE); break
```
Key engineering points:
- **`bypassPermissions`** is what makes it autonomous. It is *only* acceptable because the
  container is the security boundary (§8) — never run this backend un-sandboxed.
- The session is driven by **wall-clock**, not turn count; the controller injects deadline
  nudges as additional `query()` messages mid-stream.
- The **full message stream is persisted** to `runs/gen_<n>/<offspring>/transcript.jsonl`
  for the memory-synthesis fallback and for auditing what the agent actually did.
- Headless equivalence: this is the same engine as `claude -p "<prompt>"
  --output-format stream-json --dangerously-skip-permissions`, but the SDK gives
  structured messages and the ability to inject mid-session nudges, which the bare CLI
  print-mode cannot.

### 4.6 Backend B — Local-model harness (the population model mutates)
When the mutator is the population model itself (the analogue of "Ollama running Claude
Code"), we need an **agentic harness** that turns a raw chat-completion model into a
tool-using coder. We do **not** reimplement an agent loop from scratch; we drive an
existing open agent harness against an OpenAI-compatible local endpoint.

```
[ M's LoRA-merged weights ]  →  served by vLLM  →  OpenAI-compatible /v1/chat/completions
                                                          ▲
                                                          │  same MCP tools, same directive
                                          [ agentic harness: OpenHands / Aider / Cline-core ]
```
Engineering points:
- **Serving:** merge M's LoRA adapter into the base, serve with **vLLM** (`--api-key`,
  tool-call parsing enabled). vLLM exposes an OpenAI-compatible endpoint, so any harness
  that speaks the OpenAI tool-calling schema works unmodified.
- **Harness choice:** **OpenHands** (formerly OpenDevin) is the recommended default — it is
  built for exactly this (autonomous, containerized, multi-hour software tasks, runs in its
  own Docker sandbox, supports arbitrary OpenAI-compatible backends). Aider is a lighter
  alternative for pure code-edit loops; Cline-core if we want VS Code-protocol tooling.
- **Tool parity:** the harness is configured to expose the **same** shell/edit/git/web/MCP
  tools as the Claude backend, so the mutation directive is identical across backends and
  results are comparable. This parity is what lets §3.2's "M's identity drives mutation"
  hold meaningfully.
- **Capability floor:** a 32B coder is materially weaker than Claude at long-horizon agentic
  work. Mitigations: (a) more explicit, checklist-style directives for the local backend;
  (b) shorter sub-goals with frequent smoke-test gates; (c) the controller may allow the
  local backend a larger turn budget within the same wall-clock window.

### 4.7 Backend selection policy
Per-offspring, the controller chooses the backend by config:
- `claude` — always use Claude Agent SDK (highest capability, API cost).
- `local` — always use the population model (no API cost, lower capability, pure
  self-improvement — the strongest form of the research claim).
- `mixed` — e.g., local for mutation, Claude only for the global-memory pass (§7.3).
The research-purest configuration is `local` everywhere; `claude` is for bootstrapping and
for sanity-checking that the loop produces gains at all before trusting the local backend.

> **Honesty note — `local` is not fully Claude-free.** Even under `local`, two meta-roles
> still use Claude by design: the **global-memory pass** (§7.4) and the **anti-gaming genome
> review** (§6.4). These are *orchestration/judging* roles, not mutation — the research
> claim is specifically that *the mutation and self-improvement* is done by the population
> model, not that zero Claude calls occur anywhere. A `strict-local` flag (uses a rule-based
> anti-gaming check and a local-model global-memory pass) is available for a fully
> Claude-free run at reduced quality; it is not the default.

### 4.8 Making agents *actually* useful (AI-engineering guidance)
The flagged risk: "how do we ensure agents genuinely process memory, write useful scripts,
reason about improvements, and read papers — not just thrash?" Concrete mechanisms:
- **Structured directive, not a vibe.** The mutation directive is a template with explicit
  phases: `ORIENT (read global + own memory) → HYPOTHESIZE (write a 1-paragraph thesis to
  thesis.md) → IMPLEMENT → VALIDATE (smoke test) → REFLECT (memory file)`. The agent is
  told to literally produce `thesis.md` before editing code; this forces grounded reasoning
  and gives the global-memory pass something to read.
- **Memory is a tool call, not a wall of text.** Memory is served via MCP with retrieval
  (`memory.search(query)`, `memory.get_global()`, `memory.recent(k)`) so the agent pulls
  what's relevant instead of being flooded. Writes go through `memory.write_iteration(...)`
  with a required schema (§7.2) — you can't write junk that doesn't fit the schema.
- **Paper reading is a constrained tool.** `paper.search(query)` / `paper.fetch(arxiv_id)`
  via the MCP server (whitelisted to arXiv/Semantic Scholar) returns text + the citation
  string. The directive requires that any idea drawn from a paper be recorded with its
  citation in both `thesis.md` and the memory file (attribution enforcement, §8.4).
- **Smoke tests as the reasoning anchor.** The agent is rewarded (by the checkpoint commit)
  only when code is green, which keeps "useful script writing" tethered to "code that runs."
- **Reflection requirement.** The memory file must answer: *what did I hypothesize, what did
  I change, what did the smoke tests show, what should the next mutator try?* This is the
  signal the global-memory pass synthesizes across the population.

---

## 5. Finetuning Pipeline

### 5.1 Method
- **LoRA / QLoRA by default.** Adapter-only training keeps per-offspring cost low and makes
  "the weights" a small artifact tied to a genome. Full finetune is permitted when a model's
  thesis argues for it *and* the cost ledger allows it.
- **Base model:** Qwen2.5-Coder-32B class (final choice = cheapest-to-finetune strong coder
  at build time). The base is fixed across the population; offspring differ in genome
  (training code/config/data) and resulting adapter.

### 5.2 What gets finetuned
The genome the mutator edits *is* the finetuning recipe: data mix/curation scripts, LoRA
hyperparameters, training-loop code, objective/loss modifications, and any architectural
adapter changes the model devises. Finetuning executes that recipe to produce the adapter.

### 5.3 Lambda Labs orchestration
- **Provisioning:** controller requests N GPU instances (one per offspring being finetuned)
  via Lambda's API; offspring finetune **in parallel, each on its own GPU**, sharded as
  needed for 32B + LoRA.
- **Job contract:** each finetune job receives the green genome (from §4.4), runs in the
  finetuning Docker image (§8), writes the adapter + training logs to the run directory,
  and reports cost (GPU-hours × rate) to the cost ledger.
- **Resumability:** finetune jobs checkpoint; a preempted/failed instance resumes or
  re-provisions without restarting the whole generation.
- **Finetune failure (distinct from infra preemption).** A green smoke test proves the recipe
  is *runnable on a proxy*, not that the full 32B run succeeds — it can still OOM, diverge
  (NaN/inf loss), or fail to converge at scale. The controller distinguishes:
  - *Infra failure* (preemption, transient GPU error) → resume/re-provision (above).
  - *Recipe failure* (OOM that won't fit even after a single auto-retry at a safer config,
    NaN/inf loss, or job exits non-zero) → the offspring is flagged `finetune_failed: true`
    and assigned **floor fitness** (ranked below all valid offspring, ahead of nothing), so a
    recipe that can't actually train at scale is strongly selected against without crashing
    the generation. It is **not** retried indefinitely (cost guard). Its memory file records
    the failure mode for the next mutator and the global-memory pass to learn from.
  This closes the "green-but-won't-train-at-scale" class the proxy smoke test can't catch.
- **Sharding:** 32B full-precision won't fit a single mid-tier GPU; use the base-model's
  recommended tensor/pipeline sharding (or QLoRA 4-bit to fit a single larger GPU and avoid
  multi-GPU complexity early — recommended for the first working version).

### 5.4 Cost as a constraint
Every finetune and every API-backed agent session writes to a **cost ledger** (§7.4). The
ledger's running total and per-generation budget are surfaced in global memory so the
global-memory pass can steer the population away from expensive strategies when needed.

**Two enforcement levels (distinct mechanisms):**
- **Soft (fitness):** `λ_cost` penalizes expensive offspring in fitness (§6.3) — *steers*
  evolution toward cheaper strategies over generations.
- **Hard (controller budget caps):** the controller enforces caps so a single generation
  can't blow the budget. On hitting a cap mid-generation it **lets in-flight finetunes
  finish** (killing them wastes the spend already incurred) but **does not launch new ones**;
  offspring that never got to finetune are carried as `deferred` and re-attempted next
  generation only if budget frees. A per-job wall-clock/cost cap also kills a runaway
  individual finetune (→ treated as `finetune_failed`, §5.3). Caps live in config (§10.1).

---

## 6. Benchmarking & Anti-Gaming

### 6.1 Benchmark suite
Targeted at the system's actual job (improving code & model performance): **coding** (e.g.
SWE-bench subsets already in repo, HumanEval+, LiveCodeBench), **math** (e.g. GSM8K, MATH
subset), **advanced reasoning** (e.g. a held-out reasoning set). Exact set is config; the
suite returns a per-benchmark score vector.

### 6.2 What gets benchmarked, when
- **Newly-finetuned offspring are always benchmarked.** Survivors are benchmarked only when
  the eval slice rotates (see "rotation" below); otherwise their scores are cached and
  reused. (This supersedes the simplified "only offspring are benchmarked" shown in the §2.1
  diagram and §3.2 step 1 — those describe the no-rotation case.)
- Benchmarks run in a **clean, network-isolated eval container** (§8) to prevent the model
  from phoning home or fetching test answers.

**Benchmark input & handoff (finetune → eval).** Benchmarking is **controller-driven and
post-finetune** — it operates on a *finetuned offspring*, never on a bare genome. The
handoff: the finetune job (§5.3) outputs the LoRA adapter; the controller **applies the
adapter on top of the base at load time in the eval container** (the default — keeps the
adapter as the small portable artifact and avoids materializing a merged 32B per offspring;
a pre-merge is used only for serving backends like vLLM that need it, §4.6). The
**`darwin-eval` image bakes in the fixed base-model weights** (so the zero-egress eval
container needs no network to obtain them); only the small adapter is mounted in per
offspring. The eval container loads `base + adapter`, runs the suite, returns the score
vector. (There is no agent-callable benchmark tool — see the §9.3 correction; agents only
run the local *smoke test*, not the scored benchmark, so they cannot probe the held-out
eval.)

**Held-out eval set storage & rotation.** The full eval set (all slices) lives **only on the
controller host**, never baked into any image and never in the agent/finetune containers.
Each generation the controller picks the private held-out slice (rotation = a seeded
permutation keyed by generation number so it's reproducible), and **mounts only that slice
read-only into the zero-egress eval container** at benchmark time. "Zero egress" + "mounted
from host" are consistent: the data arrives via a local bind mount, not the network. The
controller records which slice was used per generation for auditability.

**Eval-set rotation vs. cached survivor scores.** Because the private held-out slice
*rotates* per generation (§6.4), a survivor's cached score was computed on a *different*
slice than the current offspring. To keep the fitness baseline consistent within a
generation, the rule is: **the held-out slice is fixed for a generation and all 10 models
(5 offspring + 5 survivors) are scored on that same slice.** Survivors are *not* re-finetuned
but their *cheap* benchmark *is* re-run on the current slice when the slice rotates (eval is
far cheaper than finetuning). If rotation is disabled in config, survivor scores are reused
as-is. This resolves the cache-vs-rotation inconsistency the normalization depends on.

### 6.3 Fitness reduction
`fitness = w·normalized_benchmark_vector − λ_cost·cost − λ_penalty·antigaming_flags
− λ_failed·mutation_failed`.
Weights are config; **defaults** in §10.1. Normalization is against the survivor baseline
(scored on the current generation's slice, per §6.2) so improvements are measured relative
to the current population.

### 6.4 Anti-gaming heuristics
The flagged risk: a model optimizing *for the benchmark* rather than for genuine capability
(reward hacking). Defenses, layered:
- **Held-out / rotating eval sets.** Keep a private slice of each benchmark the agents never
  see; fitness uses the held-out slice. Rotate which slice is private across generations.
- **Train/eval separation enforced by the container.** The eval set is mounted only into the
  isolated eval container at benchmark time — never present in the mutation or finetune
  containers, so it cannot leak into training data.
- **Contamination scan.** Before finetuning, scan the genome's data scripts/datasets for
  n-gram overlap with eval items; flag/penalize matches (catches a model that tries to pull
  benchmark data into its training mix).
- **Genome inspection for benchmark-specific hacks.** A lightweight Claude (or rule-based)
  review of the genome diff flags code that special-cases benchmark formats, hardcodes
  answers, or detects-and-branches on eval harness internals.
- **Plausibility / generalization gap check.** If an offspring's held-out score wildly
  exceeds a quick out-of-distribution probe, flag as suspected overfit/gaming → penalty.
- **Penalties feed fitness**, so gaming is selected *against* rather than merely logged.

---

## 7. Memory System

Two tiers, different owners, different lifetimes.

> **Canonical paths (resolves the layout used everywhere in this doc):**
> per-model memory lives **under each model's own directory** so it travels with the model
> through clone/cull: `models/<model>/memory/iter_<n>.md`. Global memory is the single
> shared store at the repo top level: `memory/global/`. The Phase-0 top-level `memory/`
> dir (§10) refers to this global store only.

### 7.1 Per-model memory — overview
- One **markdown file per iteration per model**: `models/<model>/memory/iter_<n>.md`.
- **Owner:** the mutation agent that produced that iteration writes it.
- **Visibility:** the model's *full* memory history is available to its mutator via MCP
  retrieval (`memory.recent(k)`, `memory.search(...)`), so it can reference any past attempt.
- These files are the model's evolving "lab notebook."

### 7.2 Per-model memory file schema
```markdown
---
model: model7
iteration: 12
generation: 5
parent_survivor: model3        # the genome this offspring was cloned from
mutator: model7                # who edited it (self if local backend)
backend: local | claude
base_fitness: 0.612            # parent's fitness at clone time
final_fitness: <filled post-benchmark by controller>
mutation_failed: false         # controller-set; true → fell back to clone of S (§4.3)
finetune_failed: false         # controller-set; true → full finetune failed at scale (§5.3)
cost_usd: 4.18
datasets_used: ["<hf_dataset_id>@<revision>"]   # provenance (§8.3)
papers_cited: ["arXiv:2401.xxxxx"]
---

## Thesis
<the 1-paragraph hypothesis from thesis.md — what improvement was bet on and why>

## Changes implemented
<concrete genome edits: data mix, LoRA rank, loss change, arch tweak…>

## Smoke-test / validation results
<what passed, what broke, what was reverted>

## Outcome & reflection
<did it work; what the next mutator should try or avoid>
```
The schema is **enforced by the MCP `memory.write_iteration` tool** — the agent cannot write
a malformed/empty memory. The agent writes everything *except* the post-benchmark fields
(`final_fitness`, and `mutation_failed` if set by the controller). Those are patched in by
the **controller**, which owns the memory store on the host (the agent's container has the
memory dir read-only and writes only via MCP, §8.2) — so the controller patches the file
directly host-side after benchmarking. No agent round-trip is involved.

### 7.3 Global memory — overview
- A single curated store: `memory/global/` — `objectives.md`, `whats_working.md`,
  `todo.md`, `cost_ledger.md` (or one `GLOBAL_MEMORY.md` with those sections).
- **Owner: the global-memory pass only** (Claude by default; a local model under the
  non-default `strict-local` flag, §4.7). The key invariant is that **no population/mutation
  model writes global memory** — it is written solely by the dedicated, controller-invoked
  meta-pass, never by the 10 agents during their mutation windows. Population models **read**
  it (via `memory.get_global()`) but **never write** it. This keeps the global signal
  coherent and prevents 10 agents from fighting over it. (The invariant is about *role*, not
  *which model* — "not the population agents," which is why `strict-local`'s local-model
  pass does not violate it.)

### 7.4 The global-memory pass (Claude)
After benchmarking each generation, the controller invokes a **Claude API call** (not the
agent loop — a focused reasoning call, optionally an extended-thinking "reason for ~an hour"
budget) that:
1. reads all 10 per-model `iter_<n>.md` files from this generation,
2. reads the performance/fitness table and the cost ledger,
3. rewrites global memory:
   - `whats_working.md` — patterns correlated with fitness gains across the population,
   - `objectives.md` — current high-level direction (toward SotA targets),
   - `todo.md` — open problems / unexplored ideas for next-gen mutators to pick up,
   - `cost_ledger.md` — running spend, per-generation budget, cost guidance.
4. The new global memory is what next generation's mutators ORIENT on.

This is the system's long-horizon "what works / what to do next" brain, deliberately
centralized in the strongest model so cross-population learning compounds.

### 7.5 Why this split
Per-model memory = depth and continuity for an individual lineage. Global memory =
population-level synthesis and direction. Keeping global memory write-access to Claude only
avoids drift, contradiction, and gaming of the shared signal, while still letting every
agent benefit from it read-only.

---

## 8. Containerization & Safety

Running agents in `bypassPermissions` is only safe because the **container is the security
boundary.** Each population model gets its own container for both the mutation window and
its finetune job.

### 8.1 Isolation model
- **One container per model**, mounting only that model's directory. A mutator editing an
  offspring cannot read or write sibling models' directories or the controller's host.
- **Eval runs in a separate, network-isolated container** with the (private) eval set
  mounted read-only and nothing else.
- Resource caps (CPU/mem/GPU, disk quota) per container so a runaway agent can't starve the
  host or rack up cost.

### 8.2 Filesystem policy
- Writable: the offspring's own repo + a scratch dir.
- Read-only: base-model cache, MCP-served memory (writes go through MCP, not the FS).
- No host mounts beyond the model directory; no Docker socket inside containers (prevents
  container escape / spawning privileged siblings).

### 8.3 Network policy
- **Default deny egress.** Allow only a **whitelist**:
  - **Papers:** arXiv, Semantic Scholar.
  - **Datasets & models:** Hugging Face Hub (`huggingface.co`, `*.hf.co`, CDN), and other
    trusted dataset hosts (e.g. the registries backing the configured benchmark/data sets).
  - **Packages:** the package index needed for installs (PyPI).
  - **API:** the Anthropic API endpoint (Claude backend) and the internal MCP server.
- Web access for the agent is mediated **only** by the MCP `paper.*` and `data.*` tools
  against the whitelist — there is **no general-purpose web/fetch tool and no raw,
  unrestricted scraping.** This blocks illegal scraping and limits infection vectors from
  arbitrary external content. (If a future need for narrow whitelisted page reads arises, add
  an explicit `web.fetch(url)` restricted to the whitelist — deliberately omitted for now.)
- The eval container has **zero egress.**

**Data philosophy — use precompiled datasets, do not self-scrape.** Models acquire training
data by pulling **existing, precompiled, license-clear datasets** from trusted hosts (Hugging
Face Hub first) rather than scraping the open web. This removes the legal/safety surface of
ad-hoc scraping, makes data mixes reproducible, and keeps provenance clean (the dataset card
+ license travels with the data). The MCP `data.search(query)` / `data.fetch(dataset_id,
revision)` tools wrap the HF Hub API, return the dataset card + license string, and record
both in the genome and memory file (same attribution discipline as papers, §8.4). A model
that wants new data composes a *mix of existing datasets*; it does not write scrapers.

### 8.4 Attribution & plagiarism control
- Any idea taken from a paper **must** be recorded with its citation (`papers_cited` in the
  memory schema + an inline note in the genome where the idea is implemented). The
  `paper.fetch` tool returns the canonical citation string to make this frictionless.
- A pre-finetune review step flags genome code that appears copied verbatim from an external
  source without attribution.

### 8.5 Images
- **`darwin-agent` image:** mutation-time tools — Python, Git, the harness (Claude SDK or
  OpenHands), MCP client, smoke-test deps.
- **`darwin-finetune` image:** CUDA + training stack (vLLM, PEFT/LoRA, the base model).
- **`darwin-eval` image:** benchmark harnesses only, no egress.

---

## 9. Orchestration & Infrastructure

### 9.1 Master Controller responsibilities
- Drive the generation state machine (§2.3), persist resumable state.
- Run the GA (§3), manage population directory layout.
- Spin up/tear down per-model containers; provision Lambda GPUs; enforce wall-clock budgets
  and deadline nudges (§4.3).
- Trigger benchmarking and the global-memory pass.
- Own the cost ledger and budget enforcement.

### 9.2 Host vs. cloud split
- The **controller** can run anywhere (your Windows box is fine — it orchestrates, it
  doesn't train). It talks to Lambda Labs over the API for GPU work and manages containers
  on whichever host(s) run them.
- **Mutation containers** run where GPUs for local-model serving live (Lambda) when the
  backend is `local`; can run on cheaper CPU/GPU hosts when the backend is `claude`
  (no local serving needed — only API + tools).
- **Finetune jobs** always run on Lambda GPUs.

### 9.3 The MCP server (`darwin-mcp`)
A single MCP server exposes the structured tool surface to **both** agent backends, so the
mutation directive and tool semantics are identical regardless of who's driving.
Tools:
- `memory.get_global()`, `memory.search(query)`, `memory.recent(k)` — read memory.
- `memory.write_iteration(schema...)` — schema-validated per-model memory write.
- `paper.search(query)`, `paper.fetch(id)` — whitelisted paper retrieval + citation string.
- `data.search(query)`, `data.fetch(dataset_id, revision)` — whitelisted dataset retrieval
  from Hugging Face Hub / trusted hosts; returns the dataset card + license string and
  records provenance (§8.3). No scraping tool is provided by design.
- `smoke.run()` — run the read-only smoke test (§4.4.1); returns pass/fail + log. (This is
  the agent's *only* eval-like tool. There is **no** agent-callable scored-benchmark tool —
  scored benchmarking is controller-only and post-finetune, §6.2 — so agents cannot probe
  the held-out eval set. This corrects the earlier `bench.submit` design.)
- `cost.report(amount, reason)` / `cost.get_budget()` — cost ledger I/O.
- `finalize()` — agent self-declares convergence to end its window early.

Build with the MCP Python SDK. **Backend attachment:**
- *Claude backend* attaches it natively via `ClaudeAgentOptions.mcp_servers`.
- *Local/OpenHands backend* — OpenHands supports MCP servers natively (config-level), so the
  same `darwin-mcp` is attached directly; no shim needed there. The **"OpenAI-tool shim"** is
  only the fallback for a harness that speaks *only* OpenAI function-calling: a thin adapter
  that enumerates the MCP tool schemas, exposes them as OpenAI `tools=[...]` function specs,
  and on a function-call response forwards the call to the MCP server and returns the result
  as the tool message. ~100 lines; specified as a fallback, not the primary path.

**Deadline injection across backends.** The controller enforces wall-clock identically but
the injection channel differs: for **Claude** it injects nudges/`FINALIZE` as additional
`query()` messages mid-stream (§4.5). For **OpenHands** (a separate process), the controller
injects a soft-deadline message via OpenHands' message/event API, and enforces the hard
deadline + kill by stopping the harness process and the container, then recovering
`last-green` (§4.4). The `finalize()` MCP tool works for both (agent-initiated). The
**kill-and-recover-last-green path is backend-agnostic** and is the actual guarantee; the
mid-stream nudges are best-effort politeness on top.

### 9.4 Why MCP (vs. ad-hoc tools)
MCP gives **one tool contract** consumed identically by Claude and by the local harness,
makes memory/cost/paper access *structured and auditable* (every call logged), and enforces
schemas/whitelists at the tool boundary rather than trusting prompt instructions. It is the
seam that makes the two backends interchangeable (§4.7).

### 9.5 Observability
- Per-offspring **transcript** (`transcript.jsonl`) — every agent message/tool call.
- Per-generation **fitness table**, **cost ledger**, **Git log** of each genome.
- A simple run dashboard (later) reading `runs/gen_<n>/` for live status.

---

## 10. Implementation Phases

Staged so each phase is independently testable and the loop is runnable end-to-end as early
as possible (with the cheapest backend) before adding sophistication.

### 10.1 Default config values (starting points, all tunable)
These are sane initial values so nothing has to be invented at implementation time; tune
after the loop runs.
| Parameter | Default | Notes |
|---|---|---|
| `population_size` | 10 | 5 survivors + 5 offspring |
| `num_survivors` / `num_culled` | 5 / 5 | |
| `mutation_window` | 3 h | within the 2–4 h range |
| `soft_deadline` | T−15 min | wrap-up nudge |
| `kill_grace` | 5 min | after hard deadline before force-stop |
| `backend` | `claude` (bootstrap) → `local` | start Claude to prove gains, then switch |
| `w` (benchmark weights) | uniform across benchmarks, sum=1 | re-weight later |
| `λ_cost` | 0.05 per $ (normalized) | tune so cost ≈ 5–10% of fitness swing |
| `λ_penalty` (anti-gaming) | 0.5 per flag | strong — gaming should be clearly negative |
| `λ_failed` (mutation failed) | 0.1 | mild penalty, not catastrophic (§4.3) |
| `gen_budget_usd` | set per run | hard cap/generation; on hit, finish in-flight, launch no new (§5.4) |
| `per_job_cap_usd` / `per_job_max_h` | set per run | runaway finetune kill → `finetune_failed` (§5.3) |
| `finetune_failed` fitness | floor (below all valid) | recipe that can't train at scale (§5.3) |
| `diversity_pick` | off | enable after baseline understood (§3.4) |
| `eval_rotation` | on | held-out slice rotates per generation (§6.2/§6.4) |
| `finetune_method` | QLoRA 4-bit, single GPU | avoid sharding until Phase 4 works (§5.3) |
| `lora_rank` / `alpha` | 16 / 32 | base genome default the agent may change |

### Phase 0 — Scaffolding & teardown of v1
- Strip OpenAI; remove the fixed-chunk mutation code (`improve_code.py` chunking).
- New repo layout: `controller/`, `mutation_agent/`, `finetune/`, `bench/`, `memory/`,
  `mcp/`, `containers/`, `runs/`.
- Define config schema (population size, backend, weights, budgets, benchmark set).

### Phase 1 — Memory subsystem + MCP server (no training yet)
- Implement `darwin-mcp` with `memory.*`, `cost.*`, `paper.*`, `finalize`.
- Per-model markdown schema + validation; global memory files.
- Unit-test memory read/write and the global-memory Claude pass against fixture data.

### Phase 2 — Mutation agent, Claude backend, single offspring
- Claude Agent SDK headless session with deadline nudges + FINALIZE (§4.5).
- Git auto-checkpoint on green smoke test; revert-to-green on kill.
- Run one mutation window end-to-end in a `darwin-agent` container on a trivial genome.
- Verify: thesis.md produced, memory file written, final genome is a green commit.

### Phase 3 — Finetune + benchmark for one offspring
- LoRA finetune on Lambda (start with **QLoRA single-GPU** to avoid sharding early).
- Benchmark runner in the isolated eval container; produce fitness vector.
- Cost ledger populated from real GPU-hours + API usage.

### Phase 4 — Full GA loop, 10-model population
- Controller state machine, SPAWN/clone/mutator pairing, GA cull, resumable state.
- Global-memory pass wired in after benchmarking.
- Run several generations on a small/cheap base model to validate the loop produces gains.

### Phase 5 — Local-model backend
- vLLM serving of LoRA-merged population models; OpenHands harness against it with the same
  MCP tools and directive (§4.6).
- Backend-selection policy (`claude` / `local` / `mixed`).

### Phase 6 — Anti-gaming, diversity, scale-up
- Held-out/rotating evals, contamination scan, genome hack inspection (§6.4).
- Diversity safeguard (§3.4).
- Move to the target 32B coder; tensor/pipeline sharding if not using QLoRA; parallel
  per-offspring GPUs.

### Phase 7 — Hardening & observability
- Crash/resume coverage, cost guardrails, run dashboard, audit of attribution enforcement.

### 10.2 Fresh-session handoff (read this first when starting implementation)
Context a new session needs that isn't obvious from the repo alone:

- **This is DARWIN v2, a near-total rewrite.** The existing `main/` code (v1) is a nanoGPT
  from-scratch demo and is being *replaced*, not extended. Treat `main/improve_code.py`,
  `main/main_controller.py`, `make_new_model.py`, `requests_and_memory.py`, and the
  `models/model{1..4}/` nanoGPT trees as **reference-only / to-be-removed** in Phase 0. The
  `main/swe_bench/` harness is the one piece worth salvaging for the benchmark runner (§6).
- **OpenAI is fully dropped.** All AI calls are either **Anthropic Claude** (orchestration:
  mutation when `backend=claude`, the global-memory pass, anti-gaming review) or the
  **local population model** via vLLM + OpenHands (mutation when `backend=local`). Do not
  reintroduce `openai`-package usage except as the OpenAI-*compatible* client pointed at the
  local vLLM endpoint.
- **Environment reality.** Dev/orchestration host is **Windows + PowerShell**; the
  controller runs there but **all training/agent/eval work runs in Linux Docker containers on
  Lambda Labs GPUs.** Write container-targeting code as Linux; keep the controller
  cross-platform. Don't assume bash on the host.
- **Two backends behind one interface is the central abstraction.** Implement the §4.2 agent
  lifecycle contract (`start/loop/finalize → returns green genome`) as an interface with two
  impls (`claude_backend`, `local_backend`). Everything else (controller, GA, finetune,
  bench, memory) should be backend-agnostic.
- **The MCP server is the seam.** Build `darwin-mcp` (§9.3) early (Phase 1) — both backends
  and the safety/attribution guarantees depend on it. Tools: `memory.*`, `paper.*`,
  `data.*`, `smoke.run`, `cost.*`, `finalize`. No agent-callable scored-benchmark tool.
- **Build order is deliberately cheapest-first:** memory+MCP (Phase 1) → Claude mutation on a
  trivial genome (Phase 2) → one real finetune+bench on QLoRA single-GPU (Phase 3) → full GA
  loop on a *small/cheap* base model to prove gains (Phase 4) → only then local backend
  (Phase 5) and the 32B target (Phase 6). **Do not start with the 32B model or local
  backend** — prove the loop produces fitness gains cheaply first.
- **Key invariants to never violate:** (1) finetuning only ever runs on a green commit
  (§4.4); (2) the held-out eval set never enters mutation/finetune containers (§6.2/§6.4);
  (3) no population/mutation agent writes global memory — only the dedicated global-memory
  meta-pass does (§7.3); (4) agents acquire data via
  precompiled datasets, never scrapers (§8.3); (5) `bypassPermissions`/autonomous agents
  only ever run inside a sandboxed container (§8).
- **Start by reading this whole doc**, then produce the implementation plan from §10's phases
  + §10.1 defaults. The doc has been reader-tested; §4.4.1 (smoke test), §6.2 (benchmark
  handoff), and §9.3 (MCP/backends) are the most implementation-dense sections.

### 10.3 Implementation status (living section)

> Updated as phases land. Records what *exists in the repo* vs. what the spec above describes.
> **As of 2026-05-31 (Phases 0–7 + live-infra cores + runnable `main` wired; only live seams remain).**

**Tooling.** The project standardizes on **`uv`** (the Windows host has no bare `python`; uv
manages CPython 3.14.4). Layout/deps in `pyproject.toml`. Run the suite with:
`uv run --python 3.14 --extra dev python -m pytest -q` — **308 tests passing**. (The heavy
Linux/GPU-only `local`-extra deps — `vllm`, `openhands-ai` — are environment-marked to
`sys_platform == 'linux'` + Python `>=3.12,<3.14` since `openhands-ai` doesn't support 3.14; this
keeps uv's universal resolution satisfiable and the Windows/3.14 dev host installs + tests without
them. They're lazy-imported and never touched by the suite.)

**Code layout (actual).** All Python lives under a single importable `darwin/` package
(`darwin.config`, `darwin.controller`, `darwin.mutation_agent`, `darwin.finetune`,
`darwin.bench`, `darwin.memory`, `darwin.cost`, `darwin.mcp`, `darwin.global_memory`); non-code
dirs are at the repo root (`memory/global/`, `models/`, `runs/`, `containers/`). This maps the
spec's flat dir names onto `darwin.*` submodules + top-level data/asset dirs to keep everything
importable and cross-platform.

**Phase 0 — complete.**
- ✅ Config schema with all §10.1 defaults as dataclasses (`darwin/config.py`).
- ✅ Full repo layout scaffolded: `darwin/{controller,mutation_agent,finetune,bench}` (docstring
  stubs pointing at their spec sections) + top-level `containers/` (image plan, §8.5) +
  `memory/global/`, `models/`, `runs/`.
- ✅ v1 teardown done: `main/` removed (nanoGPT trees, OpenAI usage, `improve_code.py` chunking,
  `main_controller.py`), along with the v1 root `Dockerfile` and `requirements.txt` (deps now
  live in `pyproject.toml`). The one salvaged piece (§10.2) — the SWE-bench harness — was moved
  to `darwin/bench/swe_bench/` (DARWIN's `harness.py`/`report.py`/`utils.py`/`subsets/`/
  `ref_agent_results/`; the re-clonable upstream gitlink was dropped). README operational
  sections updated to v2 (OpenAI removed; the published-paper abstract + arXiv link are kept,
  with a note that they describe v1).

**Phase 1 — memory side complete; tool surface partial.**
- ✅ Per-model memory schema (§7.2) with strict validation + markdown↔object
  (`darwin/memory/schema.py`); `MemoryStore` with `write_iteration` / `recent` / `search` /
  controller-only `patch_iteration`, and global memory read/write (§7.3)
  (`darwin/memory/store.py`).
- ✅ `darwin-mcp` **memory tool group** over the store — `memory_get_global` / `memory_recent`
  / `memory_search` / `memory_write_iteration` (FastMCP; dots aren't allowed in tool ids, so
  the `memory.*` group uses `memory_*`). Logic layer (`darwin/mcp/tools.py`) is
  transport-free and unit-tested; FastMCP wiring + stdio entrypoint in `darwin/mcp/server.py`
  (`python -m darwin.mcp.server --root .`). The agent-facing write tool deliberately omits the
  controller-owned post-benchmark fields.
- ✅ **Global-memory pass (§7.4)** — `darwin/global_memory/`: pure `GenerationDigest`
  gathering, a `Synthesizer` interface (so `strict-local` / tests can swap the writer), and
  `ClaudeSynthesizer` (Anthropic API: `claude-opus-4-8`, adaptive thinking + `effort`,
  structured outputs for the four sections, prompt caching on the system prompt, streaming),
  plus `run_global_memory_pass` (gather → synthesize → write) as the sole sanctioned global
  writer.
- ✅ **MCP `cost.*` tools** (`cost_report` / `cost_get_budget`) landed with Phase 3, bound to
  an offspring's generation/model when a mutation context + cost ledger are attached.
- ✅ **MCP `paper.*` and `data.*` tools** (§9.3/§8.3/§8.4) now built on a new `darwin/sources/`
  retrieval subsystem: a default-deny egress **whitelist** (`whitelist.py` — arXiv / Semantic
  Scholar / HF Hub only, all other hosts raise `EgressBlocked`) + an injectable, whitelist-gated
  `Transport` (stdlib `urllib` default, fakeable); `papers.py` (`PaperSource` over arXiv, returns
  the canonical **citation string** for §8.4 attribution) and `datasets.py` (`DataSource` over the
  HF Hub, returns the dataset **card + license** + an `id@revision` pin for §8.3 provenance — no
  scraping tool by design). Exposed as `paper_search`/`paper_fetch`/`data_search`/`data_fetch`
  (`PaperToolset`/`DataToolset` in `darwin/mcp/tools.py`, registered in the server behind
  `enable_retrieval=True`; egress is enforced at call time so registration is network-free). Parse
  cores are pure and unit-tested offline against canned arXiv-Atom / HF-JSON responses. (`smoke.run`
  + `finalize` landed with Phase 2.) The mutation directive already instructs agents to use these
  and record attribution; the §8.4 audit verifies they did.

**Phase 2 — backend-agnostic core complete; live containerized run deferred.**
- ✅ The §4.2 lifecycle as a backend-agnostic core in `darwin/mutation_agent/`:
  `smoke.SmokeTest` (generic §4.4.1 runner; exit 0 == green), `checkpoint.GitCheckpointer`
  (offspring branch, `darwin-green:` commits + moving `last-green` tag, revert, **zero-green
  fallback to the clone of S**, §4.3/§4.4), `deadline.DeadlineManager` (soft/hard/kill phases,
  §4.3), the structured `directive` (ORIENT→HYPOTHESIZE→IMPLEMENT→VALIDATE→REFLECT, §4.8), the
  `MutationContext`/`MutationBackend`/`MutationResult` contract, and `run_mutation_window` —
  which guarantees an **always-green final genome** regardless of how the session ended.
- ✅ Agent-facing MCP tools `smoke.run` + `finalize` (§9.3), registered on `darwin-mcp` when a
  mutation context is attached, bound to the offspring's checkpointer.
- ✅ `claude_backend.ClaudeMutationBackend` (§4.5): `bypassPermissions`, wall-clock-driven
  deadline-nudge/FINALIZE injection, transcript persistence. Option-building + injection policy
  are pure/tested; per §8.3 the web tools are excluded (web is MCP `paper.*`/`data.*` only).
- ✅ Verified end-to-end with a scripted fake backend on a real Git repo + trivial genome
  (green path → final == last-green; zero-green path → fallback to clone + `mutation_failed`).
- ⏳ **Deferred (needs infra):** the *live* multi-hour run — Docker `darwin-agent` container +
  `claude-agent-sdk` (optional `agent` extra) + the API. The §4.4.1 finetune-specific smoke
  harness (real train step) lands with Phase 3; today's runner is the generic command-based one.

**Phase 3 — backend-agnostic cores complete; live GPU/eval-container infra deferred.**
- ✅ **Cost ledger** (`darwin/cost/`): append-only JSONL `CostLedger` (record finetune
  GPU-hours×rate, API/agent/benchmark spend; per-generation + per-kind totals; markdown render
  for `cost_ledger.md`, §7.4) + `BudgetGuard` for the hard per-generation `gen_budget_usd` cap
  (§5.4: refuse to launch new jobs when exhausted; never kill in-flight). `gpu_rate_usd_per_h`
  added to `CostConfig`.
- ✅ **MCP `cost.*` tools** (above) — `cost_report` / `cost_get_budget` wired into the server.
- ✅ **Finetune pipeline core** (`darwin/finetune/`): the `FinetuneJob`/`FinetuneOutcome`/
  `FinetuneResult` contract, a CPU-runnable `SubprocessFinetuneBackend` (runs the green
  genome's finetune entrypoint, classifies OOM/NaN/non-zero/no-adapter from exit+log), a
  scaffolded `LambdaFinetuneBackend`, and `run_finetune_job` applying the §5.3 policy (one OOM
  safe-mode retry; infra-vs-recipe split; per-job cost-cap kill) and the §5.4 cost contract
  (each attempt's GPU-hours×rate recorded to the ledger).
- ✅ **Benchmark runner + fitness + rotation** (`darwin/bench/`): `BenchmarkJob`/
  `BenchmarkResult`/`BenchmarkBackend` + a `SubprocessBenchmarkBackend` (reads back a JSON
  score vector) and a scaffolded `EvalContainerBenchmarkBackend`; the seeded held-out-slice
  rotation keyed by generation (`rotation.py`, §6.2/§6.4); the §6.3 fitness reduction
  (`fitness.py`: normalize vs. survivor baseline, floor on `finetune_failed`, cost/anti-gaming/
  mutation-failed penalties).
- ⏳ **Deferred (needs infra):** the *live* Lambda GPU provisioning + `darwin-finetune` image;
  the *live* zero-egress `darwin-eval` container + real benchmark harness adapters (the
  salvaged `darwin/bench/swe_bench/` feeds the coding slice); the §4.4.1 finetune-specific
  smoke test's *real* train step (the generic command runner + the genome's own entrypoint
  cover the contract today). Anti-gaming heuristics (§6.4) are Phase 6.

**Phase 4 — full GA loop / controller composition complete; live base-model run deferred.**
- ✅ **Population & GA** (`darwin/controller/population.py`, `ga.py`): `Model`/`Population`
  (§3.1, JSON-round-tripping for state); `rank_models`/`select_survivors` (GA cull — floor/None
  fitness sorts last) and `pair_offspring` (§3.2: S with replacement, M≠S with replacement,
  <2-survivor → claude fallback `mutator=None`). Optional §3.4 diversity pick behind a flag +
  distance callable (off by default). Seeded RNG, pure.
- ✅ **Resumable generation state** (`state.py`): `GenerationState`/`OffspringState` with
  per-offspring phase flags + `PHASE_ORDER`, `GenerationStateStore` reading/writing
  `runs/gen_<n>/state.json`; `latest_generation` for run resume (§2.3).
- ✅ **Controller state machine** (`controller.py`): `Controller.run_generation` walks
  SELECT→SPAWN→[per offspring MUTATE→FINETUNE→BENCHMARK]→AGGREGATE_FITNESS→form-next-population
  →GLOBAL_MEMORY_PASS→CHECKPOINT (§2.3), persisting after every step and **resuming at the
  first incomplete offspring stage**. Fitness is reduced vs. the survivor baseline (§6.3); the
  controller patches the §7.2 post-benchmark memory fields and triggers the §7.4 pass. The
  per-offspring execution is an injectable `GenerationOps` seam.
- ✅ **`LocalGenerationOps`** (`ops.py`): the concrete seam wiring the real cores — clones S's
  genome into the offspring slot, runs the §4.2 mutation window (backend chosen per offspring
  via an injected factory), runs the §5 finetune job + §6 eval, on the local filesystem with
  subprocess/injected backends. Verified end-to-end (clone → green mutation window → subprocess
  finetune → subprocess benchmark → fitness → memory patch → global-memory pass) without
  Docker/GPU/Claude.
- ⏳ **Deferred (needs infra):** the *live* multi-generation run on a small/cheap base model to
  validate the loop produces real fitness gains (needs the Phase 3 live finetune/eval infra);
  full infra-failure re-provision/retry in the loop is treated as no-score for now (Phase 7
  hardening).

**Phase 5 — local-model backend cores complete; live vLLM/OpenHands run deferred.**
- ✅ **OpenAI-tool shim** (`darwin/mutation_agent/openai_tool_shim.py`, §9.3): translates MCP
  tool schemas → OpenAI `tools=[...]` function specs, parses a function-call, dispatches to an
  injected `invoke` callable, and wraps the result as a `role:"tool"` message (unknown-tool /
  invoker errors surface in the message rather than killing the loop). Transport-free, pure.
- ✅ **vLLM serving** (`vllm_serving.py`, §4.6): `VLLMServeConfig` + `build_serve_command` (base
  model, `--api-key`, `--enable-auto-tool-choice`/`--tool-call-parser`, dynamic LoRA via
  `--enable-lora --lora-modules` or a pre-merged model, `base_url`); `VLLMServer` launcher
  scaffolded (GPU + `vllm` needed → deferred).
- ✅ **`LocalMutationBackend`** (`local_backend.py`, §4.6): implements the §4.2
  `MutationBackend.run` contract driving the population model as mutator via the **same**
  directive + `darwin-mcp` tools as the Claude backend (parity, §9.4), with a larger turn budget
  (§4.6 capability floor). `build_harness_config` is pure/tested; the live OpenHands session is
  behind an injectable `harness_runner` (deferred default raises with the live-path note).
  Verified end-to-end through `run_mutation_window` with a fake harness (green path).
- ✅ **Backend factory** (`make_mutation_backend_factory`): routes `local` →
  `LocalMutationBackend` / else → `ClaudeMutationBackend`, ready to drop into
  `LocalGenerationOps`'s `mutation_backend_factory` seam (the controller already routes
  `backend="local"`, §4.7). Optional `local` extra (vllm + openhands) added to `pyproject.toml`.
- ⏳ **Deferred (needs infra):** the *live* vLLM serve of a LoRA-merged model on a GPU + the
  OpenHands harness session against it; validating multi-hour stability of a 32B model on the
  harness (Appendix A open question).

**Phase 6 — anti-gaming + diversity cores complete; live eval-data/GPU producers deferred.**
- ✅ **Anti-gaming heuristics** (`darwin/antigaming/`, §6.4) as fitness-penalty *producers*:
  `report.py` (`AntiGamingFlag`/`AntiGamingReport`; `.count` is the integer fed to fitness);
  `contamination.py` (word n-gram overlap of the genome's data/source vs. held-out eval items,
  capped); `genome_review.py` (a `GenomeReviewer` interface with a no-API `RuleBasedGenomeReviewer`
  — regex flags for benchmark-name special-casing / eval-harness detection / hardcoded answer
  tables / per-item id branching — and a `ClaudeGenomeReviewer` mirroring `ClaudeSynthesizer`:
  lazy SDK, structured outputs, prompt caching; only *added* diff lines are reviewed so reverting
  never trips a flag); `plausibility.py` (generalization-gap: a held-out score that overshoots an
  OOD probe by `max_gap` flags, severity scaling with the gap); `scan.py`
  (`run_antigaming_scan` composing the three — each check no-ops when its inputs are absent).
  All pure cores (n-gram math, rule patterns, gap math, prompt build/parse) are unit-tested
  without the network.
- ✅ **Wired into fitness** (§6.3): `AntiGamingConfig` (enable, n-gram width, thresholds,
  reviewer choice `claude`/`rule`/`none`) added to `DarwinConfig`; `OffspringState` carries
  `antigaming_done`/`antigaming_flags` (resumable); the controller gained an injectable
  `AntiGamingScanner` seam (default None => disabled, flags stay 0) run per offspring after
  benchmark (gated to scored offspring), and `reduce_fitness` now receives `antigaming_flags`.
  `LocalAntiGamingScanner` (`controller/antigaming_ops.py`) wires the real producers: the
  mutator's `root..HEAD` genome diff to the reviewer, the genome source as contamination
  `data_texts`, and **injected** `eval_items_provider` (host-only eval items, §6.2) + `ood_probe`
  (both absent today => only the no-infra genome-diff review runs). Verified end-to-end: a
  benchmark-gaming offspring is penalized through the full controller loop; no scanner => no flags.
- ✅ **Diversity safeguard** (§3.4): the controller now actually passes a `diversity_fn` into
  `select_survivors` when `ga.diversity_pick` is on (it previously passed none, so the safeguard
  was dead). The default `genome_code_distance` (`controller/diversity.py`) is a dependency-free
  Jaccard distance over genome-source token n-grams (real code-embedding distance deferred,
  Appendix A); the seam takes any `(Model, Model) -> float` so an embedding model drops in later.
  Still **off by default** until baseline behavior is understood.
- ✅ **Scale-up config surface** (§5.3/§5.1): `FinetuneConfig` gained `base_model`
  (`Qwen/Qwen2.5-Coder-32B`), `sharding` (`none`/`tensor`/`pipeline`), and `num_gpus` — knobs the
  *live* Lambda finetune backend will consume; defaults stay QLoRA-4-bit single-GPU until the
  loop is validated.
- ⏳ **Deferred (needs infra/data):** the *live* contamination scan + generalization-gap check
  (both need the host-only held-out eval items / an OOD probe run in the zero-egress eval
  container, §6.2 — same deferral as the rest of the eval infra); validating the Claude genome
  reviewer against real diffs; and the actual move to the 32B coder with tensor/pipeline sharding
  on parallel per-offspring Lambda GPUs.

**Phase 7 — hardening & observability cores complete; live re-provision deferred.**
- ✅ **Hard budget guardrails** (§5.4): the `BudgetGuard` is wired into the controller as an
  injectable `budget` (default None => uncapped). Before launching each offspring's finetune the
  controller checks the generation's spend against `gen_budget_usd`; once exhausted it launches no
  new finetunes and marks the remaining offspring `deferred` (a new `finetune_status`) — **unscored
  (fitness None), not floor**, so a budget-skipped offspring isn't punished like a recipe failure
  and is carried into the next population for re-attempt if budget frees. The sequential loop never
  kills an already-launched (in-flight) job; an in-flight overshoot of the cap is allowed by design.
- ✅ **Run-status dashboard** (`darwin/observability/`, §9.5): a pure reader over the persisted
  `runs/gen_<n>/state.json` + cost ledger producing per-generation / whole-run summaries (fitness
  table, spend by kind, phase, deferred/failed/flagged counts) with markdown renderers; works on a
  completed *or* in-progress generation (doubles as a live monitor). CLI:
  `python -m darwin.observability --runs runs [--cost cost.jsonl]`.
- ✅ **Attribution-enforcement audit** (`observability/attribution.py`, §8.4): cross-checks a
  finished iteration's recorded provenance against the genome source — arXiv ids in the genome
  missing from `papers_cited` (`uncited_paper`, error), `papers_cited` entries with no inline note
  in the genome (`missing_inline_paper`, error), and `datasets_used` not referenced in the genome
  (`unrecorded_dataset`, warning). Pure text analysis + an `audit_iteration` convenience that reads
  the memory file + genome dir.
- ✅ **Crash/resume coverage** (§2.3): explicit tests crash mid-offspring (a finetune raises) and
  during the global-memory pass, then resume with a fresh controller and assert completed offspring
  are not recomputed and the generation finalizes — exercising the per-step `state.json`
  persistence as the actual recovery guarantee.
- ⏳ **Deferred (needs infra):** the *live* infra-failure **re-provision/retry** in the loop (an
  `infra_failed` finetune is still treated as no-score, not yet re-provisioned on Lambda — the
  retry policy exists in `run_finetune_job`, but actual re-provisioning is live GPU infra); and a
  richer live dashboard surface (the markdown reader is the current "simple run dashboard" §9.5).

**Post-phase work — live-infra *cores* landed (provisioning, sandbox, retrieval, entrypoints).**
With all phase cores done, the remaining "needs infra" items were implemented down to their last
irreducibly-live seam (each behind an injected interface, so the orchestration around it is tested):
- ✅ **MCP `paper.*`/`data.*` + `darwin/sources/`** (§8.3/§8.4/§9.3) — see the Phase 1 block above;
  the whitelisted arXiv + HF Hub retrieval that closes the agent's only web-access path.
- ✅ **Container/safety layer** (`darwin/sandbox/`, §8): `ContainerSpec` + `build_docker_run_args`
  (refuses the Docker socket, no `--privileged`, network policy none/whitelist/open, resource caps)
  + role constructors for the three §8.5 images + `DockerContainerRunner`; the three **Dockerfiles**
  and the egress-network setup script under `containers/`.
- ✅ **Lambda provisioning** (`darwin/finetune/lambda_api.py` + a real `LambdaFinetuneBackend`, §5.3):
  the Lambda Cloud REST client (launch/poll/terminate) behind an injectable HTTP fn, and the
  provision→run→**always-terminate** orchestration; the only live seam is the SSH `job_runner`
  (inject one). API failures → `failure_mode="infra"`.
- ✅ **vLLM launcher** (`VLLMServer`, §4.6): `Popen` + readiness-poll + terminate, injectable
  `popen`/`readiness_check`/`sleep`/`clock`; defaults shell out to real `vllm` + localhost HTTP.
- ✅ **Reference entrypoints** — `darwin/finetune/entrypoint.py` (QLoRA/LoRA recipe; pure
  config/LoRA/BnB/TrainingArguments builders + safe-mode lever tested, heavy training lazy) and
  `darwin/bench/entrypoint.py` (env→config, suite dispatch/aggregation, scores handoff tested,
  model+harness load lazy) — the default genomes the images run.
- ⏳ **Still genuinely live (needs GPU/Docker/live API + heavy deps):** the OpenHands
  `harness_runner` session for `local` mutation; the `claude-agent-sdk` session for `claude`
  mutation; the Lambda SSH `job_runner` (sync genome → run image → fetch adapter); the actual GPU
  training / benchmark-harness execution inside the images; and the in-loop infra re-provision retry.

**Runnable loop wired (`main` + bootstrap + GA disk reconcile + dynamic GPU sizing + richer
global pass).**
- ✅ **Run entrypoint** `python -m darwin --config run.yaml` (`darwin/run.py`, `[project.scripts]`):
  loads a YAML run config, **bootstraps** the gen-0 population on disk (5 survivor seeds with cached
  benchmark scores + 5 offspring slots, `controller/workspace.py`) or **resumes** the latest
  generation, assembles the controller via the `build_controller` seam, and runs the loop.
- ✅ **GA disk reconcile (§3.2):** offspring slots are reset once per generation at SPAWN
  (`workspace.reset_slot`, resume-safe) so a culled model's slot is wiped before the next clone —
  the 5 survivors persist on disk, the dropped 5 are cleared (the cull lands at the next gen's
  SELECT). `workspace.materialize_model` copies offspring results back into `models/` (for the
  container path). Fixes the cross-generation stale-slot reuse.
- ✅ **Runtime GPU allocation** for parameter scaling (`finetune/sizing.py`): instance + GPU count
  sized from the run's (post-expansion) params + token budget (≤250B); `LambdaFinetuneBackend`
  uses it when a job carries a `RunSize`. Global memory seeded with the depth-expansion / MoE-
  upcycling / invent-better priorities + cost/train-time framing.
- ✅ **Global-memory pass enriched (§7.4):** the Claude synthesizer now reasons about overarching
  guiding principles and over cited papers + what's-working to produce prioritized concrete
  concepts-to-test, upholding the scaling direction and weighing cost/train-time.
- ✅ **§4.3 memory-synthesis fallback** (`mutation_agent/memory_synthesis.py`): if a mutator never
  wrote its memory file, the controller synthesizes the agent-written fields from the genome's git
  log + a transcript excerpt via an injectable `MemorySynthesizer` (`ClaudeMemorySynthesizer`
  default; None keeps the skip behavior), so the global pass always has a notebook to read.
- ✅ **§6.2 survivor re-benchmark on rotation**: `Model.scored_slice` tracks the slice a model's
  cached scores were computed on; when the held-out slice rotates, survivors are cheaply re-scored
  on the current slice (not re-finetuned) so all 10 share one slice. No-op when rotation is off.
- ✅ **Container execution path** (`controller/container_ops.py`, `mode: container`): the
  window/finetune/eval now actually execute *inside* the `darwin-agent`/`darwin-finetune`/
  `darwin-eval` images. `ContainerGenerationOps` composes `LocalGenerationOps` (with the new
  `ContainerFinetuneBackend` + `EvalContainerBenchmarkBackend`, both behind an injectable
  `ContainerRunner`) for spawn/finetune/benchmark and overrides `mutate` to launch the agent
  container — whose process is the new in-container `mutation_agent/entrypoint.py`. The three open
  design points are resolved: the in-container mutation entrypoint; a **writable scores mount** on
  the eval container (`eval_container(scores_out_host=...)`, zero-egress preserved — scores leave by
  a local bind mount, not the network); and **host↔container path mapping** (the offspring `genome`
  bind-mounts rw so edits/checkpoints land in place — no move-back step; a scratch mount carries the
  result JSON, and the per-model memory is seeded in / ingested back so the agent's ORIENT reads its
  lineage + global memory and the controller's post-benchmark patch + global pass see the new file).
  Unit-tested end-to-end through the controller with a fake `ContainerRunner`; what's left is the
  live substrate (a Docker host + built images + GPUs).
- ⏳ **Remaining glue:** the live anti-gaming **eval-data providers** for container mode
  (`eval_slice_dir` host-only slice dir is wired through to the eval mount; the
  `eval_items_provider`/`ood_probe` for the contamination + generalization-gap checks are still
  injected seams), and the live `agent_env` secrets passthrough validated against a real
  `darwin-agent` session.

**Invariants already enforced in code:** only the controller patches post-benchmark fields
(§7.2); there is no agent-facing global-memory write path (§7.3); the global-memory pass is
the only sanctioned writer of global memory; finetuning runs only on a green commit (the
mutation window always finalizes to one, §4.4); a `finetune_failed` recipe gets floor fitness
and infra failure is never charged to the recipe (§5.3/§6.3); the budget cap stops *new*
launches without killing in-flight jobs (§5.4); the GA keeps population size + names stable
across generations (5 survivors carried with cached scores + 5 offspring filling culled slots,
§3.2).

**Resume point (next session) — every *core* is built and the live-infra paths are wired down to
their last live seam; what remains needs real GPUs/Docker/live APIs + the heavy ML deps.** The
full loop, both mutation backends, cost/finetune/bench, §6.4 anti-gaming + §3.4 diversity, the
Phase 7 hardening/observability cores, and the live-infra cores (sandbox + Dockerfiles, Lambda
client + backend, vLLM launcher, retrieval tools, reference entrypoints) are all in the repo and
unit-tested without Docker/GPU/Claude. To take it live, implement the injected live seams + stand
up the substrate:
- **Inject the live seams:** the Lambda `job_runner` (SSH → sync genome → run `darwin-finetune`
  image → fetch adapter); the OpenHands `harness_runner` (drive the local model over the vLLM
  endpoint with the `darwin-mcp` tools); confirm the `claude-agent-sdk` session path; the
  `LocalAntiGamingScanner` `eval_items_provider`/`ood_probe` (host-only eval data + OOD probe).
- **Stand up the substrate:** build the three images; create the `darwin-egress` whitelist network
  (`containers/setup_whitelist_network.sh`); bake the base-model snapshot into `darwin-eval`; wire
  the in-loop infra-failure re-provision/retry; the 32B scale-up via
  `FinetuneConfig.base_model`/`sharding`/`num_gpus`; validate a multi-hour run end-to-end.
- **Config to flip once baselines are understood:** enable `ga.diversity_pick` (optionally swap
  `genome_code_distance` for an embedding distance, Appendix A); set `cost.gen_budget_usd` /
  `per_job_*` caps; choose `antigaming.genome_reviewer` (`claude` default vs. `rule` for
  strict-local). Work continues on branch `v2-foundation`. Run tests with
  `uv run --python 3.14 --extra dev python -m pytest -q`.

---

## Appendix A — Open questions / decisions to revisit
- **Local harness final choice** (OpenHands vs. Aider vs. Cline-core) — validate OpenHands
  multi-hour stability against a 32B model before committing.
- **Sharding vs. QLoRA** for the 32B target once past Phase 4.
- **Extended-thinking budget** for the global-memory pass (the "reason for ~an hour" call).
- **Diversity metric** (code-embedding model) for §3.4.
- **Exact benchmark set + held-out split sizes** for §6.
- **"Token-count expansion of a fixed-size model via finetuning"** idea raised in scoping —
  parked as a candidate *objective* the system could pursue, not core infrastructure;
  revisit as a seed entry in `todo.md`.
