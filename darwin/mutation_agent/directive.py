"""The mutation directive (ARCHITECTURE.md §4.1, §4.8).

A structured template — not a vibe — with explicit phases the agent must follow:
ORIENT → HYPOTHESIZE (write thesis.md) → IMPLEMENT → VALIDATE (smoke) → REFLECT (memory file).
Producing thesis.md before editing forces grounded reasoning and gives the global-memory pass
something to read. Paper/data provenance must be attributed (§8.4). Also builds the soft-
deadline nudge and the FINALIZE message the controller injects mid-session (§4.3).
"""

from __future__ import annotations

THESIS_FILENAME = "thesis.md"

DIRECTIVE_SYSTEM_PROMPT = """\
You are a DARWIN mutation agent. You are dropped into one offspring model's genome — the
training/architecture/data CODE and config that produces a LoRA finetune of a fixed base
model. Your job is to improve that genome so the finetuned model scores higher on coding,
math, and reasoning benchmarks, working autonomously inside a sandboxed container.

Hard rules:
- Work ONLY inside the genome repo you are given. Everything is reversible via Git.
- "Green" means the smoke test passes (the recipe actually trains). Call the `smoke.run` tool
  to check; a passing smoke test auto-commits a checkpoint. NEVER leave the genome in a broken
  state — if an edit fails the smoke test, fix it or revert.
- Acquire data only from precompiled, license-clear datasets via the `data.*` tools — never
  write scrapers. Any idea taken from a paper (via `paper.*`) MUST be recorded with its
  citation in thesis.md and your memory file (attribution is non-negotiable).
- You cannot see the held-out benchmark; do not try to. Optimize for genuine capability, not
  for the eval.

Follow the five phases in order: ORIENT, HYPOTHESIZE, IMPLEMENT, VALIDATE, REFLECT."""


# A pared-down system prompt + task for validating the live agent path on a TRIVIAL test genome
# (TEST_RUN_PLAN, real-Claude opt-in): one small, safe, self-contained green change — NOT the full
# param-scaling mission. Keeps a real ~5-minute SDK session cheap and focused.
SMALL_DIRECTIVE_SYSTEM_PROMPT = """\
You are a DARWIN test mutation agent in a sandboxed container. Make ONE small, safe, self-contained
improvement to the Python code in this repo — for example a tiny helper function, a constant, a
clarifying docstring, or a minor readability refactor. Keep it minimal; do NOT redesign anything or
add dependencies, datasets, or network calls.

Hard rules:
- Work ONLY inside this repo; everything is reversible via Git.
- The repo MUST stay green: after your edit, call the `smoke.run` tool (a pass auto-commits a green
  checkpoint). If smoke fails, fix or revert. Never leave the genome broken.
- You have a short (~5 minute) wall-clock budget. Make one change, verify it green, then record a
  one-line note with `memory.write_iteration` and call `finalize`."""


def build_directive(
    *,
    offspring_id: str,
    model: str,
    parent_survivor: str,
    mutator: str,
    generation: int,
    style: str = "full",
) -> str:
    """The per-window task message (the predetermined mutation prompt).

    `style="small"` returns a compact task that asks for one small green change (for validating the
    live agent path on the trivial test genome); the default `"full"` is the real §4.1 mission.
    """
    if style == "small":
        return f"""\
# Test mutation window — offspring {offspring_id} (model {model}), generation {generation}

Make ONE small, safe improvement to the code in this repo, following these steps:

1. ORIENT: read the existing files; optionally read prior notes via `memory.recent` (model
   {model}) and the shared `memory.get_global`.
2. IMPLEMENT: make a single minimal, self-contained edit (a small helper, constant, docstring, or
   tidy refactor). Do not redesign, add dependencies, or fetch anything.
3. VALIDATE: call `smoke.run` to confirm the repo still works (a green run auto-checkpoints). Fix
   or revert if it fails.
4. REFLECT: call `memory.write_iteration` with your one-line change summary, then `finalize`."""
    return f"""\
# Mutation window — offspring {offspring_id} (model {model}), generation {generation}

You are mutator {mutator}. This offspring's genome was cloned from survivor {parent_survivor}.
Improve it within your time budget, following these phases:

## 1. ORIENT
Read global memory (`memory.get_global`) for what's working and the current objectives/to-do.
Read this lineage's past attempts (`memory.recent`, `memory.search`). Inspect the genome.

## 2. HYPOTHESIZE
Write a one-paragraph thesis to `{THESIS_FILENAME}`: the single improvement you are betting on
and why, grounded in memory and (if used) cited papers. Do this BEFORE editing code.

## 3. IMPLEMENT
Make the change to the genome (data mix, LoRA hyperparameters, loss/objective, training loop,
or an architectural adapter). Keep changes focused on your thesis.

## 4. VALIDATE
Call `smoke.run` to verify the recipe still trains. Each green run checkpoints automatically.
Iterate IMPLEMENT↔VALIDATE; keep the genome green. If you converge early, call `finalize`.

## 5. REFLECT
Before time runs out, record this iteration with `memory.write_iteration`: your thesis, the
concrete changes, what the smoke test showed, and what the next mutator should try or avoid."""


def soft_deadline_nudge(minutes_remaining: int) -> str:
    """Injected at the soft deadline (§4.3)."""
    return (
        f"{minutes_remaining} minutes remain. Stop new experiments now. Ensure the current "
        "code passes `smoke.run` (so it commits a green checkpoint), then write your memory "
        "file with `memory.write_iteration`."
    )


FINALIZE_MESSAGE = (
    "FINALIZE. Time is up. Make a final `smoke.run` so the working code is a green checkpoint, "
    "then immediately write your memory file with `memory.write_iteration` and stop."
)
