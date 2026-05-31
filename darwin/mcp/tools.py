"""Memory tool logic for the DARWIN MCP server (ARCHITECTURE.md §9.3, memory group).

This is the backend-agnostic logic layer: it wraps a `MemoryStore` and returns JSON-friendly
values, with no dependency on the MCP transport so it can be unit-tested directly. The
FastMCP server (`server.py`) is a thin layer that registers these methods as tools.

Tools implemented here:
- `memory.get_global()`        -> read the shared global memory store
- `memory.recent(model, k)`    -> the k most recent iterations (newest first)
- `memory.search(model, query)`-> keyword search over a model's own history
- `memory.write_iteration(...)`-> schema-validated per-model memory write

Invariants enforced at this boundary (§7.2 / §7.3):
- agents write per-model memory **only** through the schema-validated `write_iteration`;
- the controller-owned post-benchmark fields (`final_fitness`, `mutation_failed`,
  `finetune_failed`) are **not** accepted from agents here — only the controller patches them;
- there is **no** global-memory write tool — population/mutation agents only read it.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from darwin.memory import IterationMemory, MemoryStore, MemoryValidationError


class MemoryToolset:
    """Memory tools bound to a single MemoryStore."""

    def __init__(self, store: MemoryStore):
        self.store = store

    # ------------------------------------------------------------------ reads
    def get_global(self) -> dict[str, str]:
        """Return the global memory store as a section->content mapping (`memory.get_global`)."""
        return asdict(self.store.get_global())

    def recent(self, model: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the k most recent iterations for a model, newest first (`memory.recent`)."""
        return [m.to_dict() for m in self.store.recent(model, k)]

    def search(self, model: str, query: str) -> list[dict[str, Any]]:
        """Keyword search over a model's own memory history (`memory.search`)."""
        return [m.to_dict() for m in self.store.search(model, query)]

    # ------------------------------------------------------------------ write
    def write_iteration(
        self,
        *,
        model: str,
        iteration: int,
        generation: int,
        parent_survivor: str,
        mutator: str,
        backend: str,
        base_fitness: float,
        cost_usd: float,
        thesis: str,
        changes: str,
        smoke_results: str,
        outcome: str,
        datasets_used: list[str] | None = None,
        papers_cited: list[str] | None = None,
    ) -> dict[str, Any]:
        """Schema-validated per-model memory write (`memory.write_iteration`, §7.2).

        Only agent-owned fields are accepted; controller-owned post-benchmark fields are
        deliberately absent from the signature. On a schema violation this returns a
        structured error (rather than raising) so the agent can read the problem and retry.
        """
        mem = IterationMemory(
            model=model,
            iteration=iteration,
            generation=generation,
            parent_survivor=parent_survivor,
            mutator=mutator,
            backend=backend,
            base_fitness=base_fitness,
            cost_usd=cost_usd,
            thesis=thesis,
            changes=changes,
            smoke_results=smoke_results,
            outcome=outcome,
            datasets_used=list(datasets_used or []),
            papers_cited=list(papers_cited or []),
        )
        try:
            path = self.store.write_iteration(mem)
        except MemoryValidationError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "model": model,
            "iteration": iteration,
            "path": str(path),
        }
