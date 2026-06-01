"""Mock mutation backend — the offline, deterministic mutator (TEST_RUN_PLAN §3.3).

A third `MutationBackend` alongside `ClaudeMutationBackend` (§4.5) and `LocalMutationBackend`
(§4.6). The live backends drive an autonomous agent session (Claude SDK / a vLLM-served model) for
hours; this one makes a single **deterministic green edit** with no network and no model, so the
whole generational loop — selection, finetune, eval, fitness, the cull, global memory, resume —
can run end-to-end offline (and inside the real `darwin-agent` container) for the budget-free test.

What it does each window (the §4.2 contract):
1. append an improvement marker line (`# darwin-improve <iteration>`) to the genome — a green,
   no-op-to-behavior edit that the mock finetune counts (§3.1) and the mock eval turns into a score
   gain (§3.2), so a surviving lineage's fitness trends up across generations;
2. `ctx.checkpoint(...)` — run the smoke test and, on green, commit (advancing `last-green`);
3. `ctx.write_memory(...)` — write this iteration's per-model memory file (so the global-memory
   pass and the §4.3 fallback both see a real notebook entry).

The always-green finalization + kill/recover guarantee are handled by `run_mutation_window` /
the checkpointer exactly as for the live backends — this backend only supplies the edit.
"""

from __future__ import annotations

from pathlib import Path

from darwin.finetune.mock_entrypoint import IMPROVEMENT_MARKER
from darwin.mutation_agent.backend import MutationContext
from darwin.mutation_agent.deadline import DeadlineManager

# Preference order for the file the marker is appended to: the recipe the agent would really edit,
# else a dedicated improvements file we create. A trailing comment line is always smoke-safe.
_PREFERRED_TARGETS = ("recipe.py",)
_FALLBACK_TARGET = "improvements.py"


def _marker_target(genome_dir: Path) -> Path:
    for name in _PREFERRED_TARGETS:
        p = genome_dir / name
        if p.exists():
            return p
    return genome_dir / _FALLBACK_TARGET


def apply_marker_edit(genome_dir: Path, iteration: int) -> Path:
    """Append the improvement marker for `iteration` to the genome; return the edited file (pure).

    A comment line is appended so the edit changes the genome fingerprint (driving the score) while
    keeping the recipe importable — the smoke test stays green.
    """
    target = _marker_target(genome_dir)
    line = f"# {IMPROVEMENT_MARKER} {iteration}\n"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(line)
    return target


class MockMutationBackend:
    """Deterministic, offline mutation backend for the test profile (§3.3)."""

    def run(self, ctx: MutationContext, deadline: DeadlineManager) -> None:
        target = apply_marker_edit(ctx.genome_dir, ctx.iteration)

        thesis = (
            f"Deterministic mock improvement #{ctx.iteration} on lineage {ctx.parent_survivor}: "
            f"accumulate one improvement marker to drive the synthetic fitness signal."
        )
        changes = f"Appended `# {IMPROVEMENT_MARKER} {ctx.iteration}` to {target.name}."

        green = ctx.checkpoint(f"mock improvement {ctx.iteration}")
        ctx.write_memory(
            thesis=thesis,
            changes=changes,
            smoke_results="green" if green else "red",
            outcome="committed green checkpoint" if green else "smoke failed; no checkpoint",
            cost_usd=0.0,
        )
