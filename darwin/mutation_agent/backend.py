"""The mutation-agent contract (ARCHITECTURE.md §4.2).

Every backend implements the same lifecycle so the controller treats them identically:

    start(container, genome, directive, budget) → session
      loop until FINALIZE or time budget exhausted:
          agent reasons → calls a tool → observes result
          on each green smoke test: git commit (checkpoint)
      on FINALIZE / soft-deadline: ensure last commit is green, write the memory file
      return final genome (HEAD of the offspring's branch)

Here that contract is `MutationBackend.run(ctx, deadline)`. The backend drives the agent; the
shared `MutationContext` exposes the safe capabilities it needs (checkpointing, memory write),
and the orchestrator (`runner.py`) handles offspring setup and the always-green finalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from darwin.memory import IterationMemory, MemoryStore
from darwin.mutation_agent.checkpoint import GitCheckpointer
from darwin.mutation_agent.deadline import DeadlineManager
from darwin.mutation_agent.smoke import SmokeResult, SmokeTest


@dataclass
class MutationContext:
    """Everything a backend needs to mutate one offspring, plus safe capabilities."""

    offspring_id: str
    genome_dir: Path
    model: str  # the offspring's model name (memory is keyed on this)
    parent_survivor: str
    mutator: str
    generation: int
    iteration: int
    backend_name: str  # "claude" | "local"
    base_fitness: float
    directive: str
    checkpointer: GitCheckpointer
    smoke: SmokeTest
    store: MemoryStore

    def run_smoke(self) -> SmokeResult:
        return self.smoke.run(self.genome_dir)

    def checkpoint(self, summary: str) -> bool:
        """Run the smoke test; on green, commit a checkpoint (§4.4). Returns green-ness.

        This is what the agent's `smoke.run` tool maps onto: a passing smoke test
        auto-commits and advances `last-green`.
        """
        result = self.run_smoke()
        if result.passed:
            self.checkpointer.commit_green(summary)
            return True
        return False

    def write_memory(
        self,
        *,
        thesis: str,
        changes: str,
        smoke_results: str,
        outcome: str,
        cost_usd: float,
        datasets_used: list[str] | None = None,
        papers_cited: list[str] | None = None,
    ) -> Path:
        """Write this iteration's per-model memory file (§7.2). Controller-owned fields
        (final_fitness, mutation_failed, finetune_failed) are patched in later."""
        mem = IterationMemory(
            model=self.model,
            iteration=self.iteration,
            generation=self.generation,
            parent_survivor=self.parent_survivor,
            mutator=self.mutator,
            backend=self.backend_name,
            base_fitness=self.base_fitness,
            cost_usd=cost_usd,
            thesis=thesis,
            changes=changes,
            smoke_results=smoke_results,
            outcome=outcome,
            datasets_used=list(datasets_used or []),
            papers_cited=list(papers_cited or []),
        )
        return self.store.write_iteration(mem)


@dataclass
class MutationResult:
    """Outcome of a mutation window."""

    offspring_id: str
    final_commit: str
    produced_green: bool  # a smoke-verified green commit was made this window
    mutation_failed: bool  # fell back to the unchanged clone of S (§4.3)
    memory_written: bool


class MutationBackend(Protocol):
    """One autonomous mutation session. Implementations: claude_backend, local_backend."""

    def run(self, ctx: MutationContext, deadline: DeadlineManager) -> None: ...
