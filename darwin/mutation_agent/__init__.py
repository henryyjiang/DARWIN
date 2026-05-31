"""Mutation Agent (ARCHITECTURE.md §4).

One offspring's mutation = one autonomous agent session inside one Docker container, driven by
a wall-clock budget and checkpointed in Git so the final genome is always a green commit.

Backend-agnostic core (this package):
- `smoke.SmokeTest`        — the §4.4.1 smoke-test runner ("green" = the recipe trains).
- `checkpoint.GitCheckpointer` — offspring branch, green commits, last-green tag, revert (§4.4).
- `deadline.DeadlineManager`   — soft / hard / kill wall-clock phases (§4.3).
- `directive`              — the structured mutation directive + deadline nudges (§4.1/§4.8).
- `backend.MutationContext`/`MutationBackend`/`MutationResult` — the §4.2 lifecycle contract.
- `runner.run_mutation_window` — orchestrates a window; guarantees an always-green final genome.

Backends behind the one interface:
- `claude_backend.ClaudeMutationBackend` — Claude Agent SDK headless session (§4.5).
- local-model harness (§4.6) — Phase 5, not yet implemented.
"""

from darwin.mutation_agent.smoke import SmokeTest, SmokeResult
from darwin.mutation_agent.checkpoint import GitCheckpointer
from darwin.mutation_agent.deadline import DeadlineManager, Phase
from darwin.mutation_agent.backend import (
    MutationBackend,
    MutationContext,
    MutationResult,
)
from darwin.mutation_agent.runner import run_mutation_window

__all__ = [
    "SmokeTest",
    "SmokeResult",
    "GitCheckpointer",
    "DeadlineManager",
    "Phase",
    "MutationBackend",
    "MutationContext",
    "MutationResult",
    "run_mutation_window",
]
