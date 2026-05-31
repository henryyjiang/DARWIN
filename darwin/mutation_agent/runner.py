"""Backend-agnostic mutation-window orchestrator (ARCHITECTURE.md §4.2-§4.4).

Sets up the offspring's Git branch, runs the backend's autonomous session, then enforces the
core guarantee regardless of how the session ended (clean FINALIZE, soft-deadline wrap-up, or
a force-kill): **the final genome is always a green commit.** If the window produced no green
commit, the zero-green fallback (§4.3) resets to the unchanged clone of S and flags
`mutation_failed`. Memory-file presence is reported so the controller can synthesize a missing
one from the transcript (§4.3) if needed.
"""

from __future__ import annotations

from darwin.mutation_agent.backend import (
    MutationBackend,
    MutationContext,
    MutationResult,
)
from darwin.mutation_agent.deadline import DeadlineManager


def run_mutation_window(
    ctx: MutationContext,
    backend: MutationBackend,
    deadline: DeadlineManager,
) -> MutationResult:
    """Run one offspring's mutation window end-to-end and return its result."""
    ctx.checkpointer.init_offspring(ctx.offspring_id, ctx.parent_survivor)

    # The backend drives the autonomous session. It is responsible for respecting the
    # deadline phases and for checkpointing on green; the orchestrator backstops the
    # always-green guarantee below even if the backend was killed mid-thought.
    backend.run(ctx, deadline)

    final_commit, fell_back = ctx.checkpointer.finalize_genome()
    memory_written = ctx.store.iter_path(ctx.model, ctx.iteration).exists()

    return MutationResult(
        offspring_id=ctx.offspring_id,
        final_commit=final_commit,
        produced_green=not fell_back,
        mutation_failed=fell_back,
        memory_written=memory_written,
    )
