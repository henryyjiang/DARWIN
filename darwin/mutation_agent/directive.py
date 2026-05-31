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


def build_directive(
    *,
    offspring_id: str,
    model: str,
    parent_survivor: str,
    mutator: str,
    generation: int,
) -> str:
    """The per-window task message (the predetermined mutation prompt)."""
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
